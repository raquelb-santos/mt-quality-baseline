"""Benchmark orchestration: resolve glossary matches, run post-mt, score MT and APE against the reference."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from .dataset import Dataset
from .postmt import Usage, extract_post_edited, reported_has_glossary, segment_error
from .score import Aggregate, TranslationScore, aggregate, score_translation

logger = logging.getLogger(__name__)


def _unique(items: Sequence[str]) -> list[str]:
    """Order-preserving dedup — index alignment downstream depends on stable ordering."""
    return list(dict.fromkeys(items))


@dataclass
class SegmentResult:
    source_segment_id: str
    source_content: str
    target_content: str
    #: The human answer key both scores are taken against, never a scored column of its own.
    reference_content: str
    post_edited_text: str
    changed_by_ape: bool
    has_glossary_reported: bool | None
    has_glossary_resolved: bool
    glossary_terms: list[dict[str, str]]
    mt: TranslationScore
    ape: TranslationScore


@dataclass
class Delta:
    adherence_rate: float | None
    #: Substitutions only. An omission APE could not fix was never a terminology failure, so
    #: counting it here would credit or blame the post-edit for an empty segment.
    violations_resolved: int
    terms_fixed_by_ape: int
    terms_regressed_by_ape: int


@dataclass
class BenchmarkResult:
    dataset: str
    started_at: str
    finished_at: str
    parameters: dict[str, Any]
    glossary_ids: list[str]
    steps: list[str]
    config: dict[str, Any]
    totals: dict[str, int]
    mt_baseline: Aggregate
    post_edited: Aggregate
    delta: Delta
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[SegmentResult] = field(default_factory=list)


class Benchmark:
    def __init__(self, *, postmt: Any, stanza: Any, glossary: Any, config: Any) -> None:
        self.postmt = postmt
        self.stanza = stanza
        self.glossary = glossary
        self.config = config
        self.last_usage = Usage()
        self.last_failures: list[str] = []

    def resolve_glossary(
        self, segments: Sequence[dict[str, Any]], parameters: dict[str, Any], glossary_ids: Sequence[str]
    ) -> list[list[dict[str, str]]]:
        source_language = parameters.get("clean_source_language_code")
        target_language = parameters.get("clean_target_language_code")

        unique_sources = _unique([s["source_content"] for s in segments])

        logger.info("[GLOSSARY] lemmatizing %d unique sources (%s)", len(unique_sources), source_language)

        # Degrade rather than abort: no glossary at all reads as a clean 0-instance scorecard.
        lemmatized = self.stanza.lemmatize_batch_safe(unique_sources, source_language)
        if lemmatized is None:
            logger.warning(
                "[GLOSSARY] falling back to un-lemmatized sources - retrieval will differ from production"
            )
            lemmatized = unique_sources

        if not lemmatized:
            logger.warning("[GLOSSARY] no source texts to match - no glossary can be resolved")
            return [[] for _ in segments]

        fetch_kwargs = dict(
            glossary_ids=list(glossary_ids),
            source_language=source_language,
            target_language=target_language,
            texts=lemmatized,
            provider=parameters.get("cat_tool_provider"),
        )
        if getattr(self.glossary, "wants_raw_texts", False):
            fetch_kwargs["raw_texts"] = unique_sources

        matches = self.glossary.fetch_matches(**fetch_kwargs)

        logger.info(
            "[GLOSSARY] %d distinct term mappings across %d unique sources",
            len(matches.mappings), len(unique_sources),
        )

        index_of = {text: i for i, text in enumerate(unique_sources)}
        positions = [index_of.get(s["source_content"]) for s in segments]
        return [[] if i is None else matches.per_text_mappings[i] for i in positions]

    def run_pipeline(self, dataset: Dataset) -> list[dict[str, Any]]:
        size = self.config.benchmark.batch_size
        segments = dataset.segments
        batches = [segments[i : i + size] for i in range(0, len(segments), size)]
        logger.info("[PIPELINE] %d segments in %d batch(es)", len(segments), len(batches))

        processed: list[dict[str, Any]] = []
        self.last_usage = Usage()
        self.last_failures = []

        for number, batch in enumerate(batches, start=1):

            def on_progress(body: dict[str, Any], _n: int = number) -> None:
                percent = (body.get("progress") or {}).get("percent", 0)
                logger.info("[PIPELINE] batch %d/%d - %s%%", _n, len(batches), percent)

            # The human reference is the answer key: stripped so it can never reach post-mt.
            payload = [
                {k: v for k, v in segment.items() if k != "reference_content"}
                for segment in batch
            ]

            result = self.postmt.run(
                parameters=dataset.parameters,
                segments=payload,
                steps=dataset.steps,
                on_progress=on_progress,
            )

            if result.error:
                logger.warning("[PIPELINE] batch %d reported: %s", number, result.error)

            if len(result.segments) != len(batch):
                logger.warning(
                    "[PIPELINE] batch %d returned %d segments for %d inputs - realigning by index",
                    number, len(result.segments), len(batch),
                )

            self.last_usage = self.last_usage + result.usage

            for i, original in enumerate(batch):
                returned = result.segments[i] if i < len(result.segments) else None
                processed.append(returned if returned is not None else original)

        self.last_failures = [error for s in processed if (error := segment_error(s))]
        if self.last_failures:
            logger.error(
                "[PIPELINE] %d/%d segments failed inside post-mt - first: %s",
                len(self.last_failures), len(processed), self.last_failures[0],
            )

        return processed

    def build_target_lemmas(
        self, texts: Sequence[str], terms: Sequence[str], target_language: str
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        if not self.config.benchmark.lemma_matching:
            return None, None

        unique_texts = _unique([t for t in texts if t])
        unique_terms = _unique([t for t in terms if t])

        combined = unique_texts + unique_terms
        if not combined:
            return None, None

        lemmas = self.stanza.lemmatize_batch_safe(combined, target_language)
        if lemmas is None:
            logger.warning(
                "[SCORE] proceeding with surface-form matching only - inflected forms will count as violations"
            )
            return None, None

        text_lemmas = dict(zip(unique_texts, lemmas[: len(unique_texts)]))
        term_lemmas = dict(zip(unique_terms, lemmas[len(unique_texts) :]))
        return text_lemmas, term_lemmas

    def _stub_pipeline(self, dataset: Dataset) -> list[dict[str, Any]]:
        return [
            {**segment, "ape_results": {"text": segment.get("target_content", "")}}
            for segment in dataset.segments
        ]

    def run(self, dataset: Dataset, *, skip_pipeline: bool = False) -> BenchmarkResult:
        started_at = datetime.now(timezone.utc).isoformat()
        target_language = dataset.parameters.get("clean_target_language_code")
        source_language = dataset.parameters.get("clean_source_language_code")

        per_segment_mappings = self.resolve_glossary(
            dataset.segments, dataset.parameters, dataset.glossary_ids
        )

        glossary_bearing = sum(1 for m in per_segment_mappings if m)
        logger.info(
            "[BENCH] %d/%d segments carry at least one glossary term",
            glossary_bearing, len(dataset.segments),
        )
        if glossary_bearing == 0:
            logger.warning(
                "[BENCH] no glossary matches at all - check glossary_ids and language codes "
                "before trusting a 0-instance result"
            )

        processed = self._stub_pipeline(dataset) if skip_pipeline else self.run_pipeline(dataset)

        baseline_texts = [s.get("target_content") or "" for s in processed]
        post_edited_texts = [extract_post_edited(s) for s in processed]
        reference_texts = [s.get("reference_content") or "" for s in dataset.segments]
        all_terms = [m["target_content"] for mappings in per_segment_mappings for m in mappings]

        text_lemmas, term_lemmas = self.build_target_lemmas(
            baseline_texts + post_edited_texts + reference_texts, all_terms, target_language
        )

        results: list[SegmentResult] = []
        for i, segment in enumerate(processed):
            mappings = per_segment_mappings[i]
            baseline = baseline_texts[i]
            post_edited = post_edited_texts[i]

            reference_text = reference_texts[i]

            common = dict(
                mappings=mappings,
                language_code=target_language,
                term_lemmas=term_lemmas,
                reference_text=reference_text,
                reference_lemmas=(text_lemmas or {}).get(reference_text),
            )

            mt = score_translation(
                text=baseline, text_lemmas=(text_lemmas or {}).get(baseline), **common
            )
            ape = score_translation(
                text=post_edited, text_lemmas=(text_lemmas or {}).get(post_edited), **common
            )

            results.append(
                SegmentResult(
                    reference_content=reference_text,
                    source_segment_id=str(
                        segment.get("source_segment_id")
                        or dataset.segments[i].get("source_segment_id")
                        or i
                    ),
                    source_content=dataset.segments[i].get("source_content", ""),
                    target_content=baseline,
                    post_edited_text=post_edited,
                    changed_by_ape=baseline != post_edited,
                    has_glossary_reported=reported_has_glossary(segment),
                    has_glossary_resolved=bool(mappings),
                    glossary_terms=mappings,
                    mt=mt,
                    ape=ape,
                )
            )

        if not skip_pipeline:
            if self.last_failures:
                logger.error(
                    "[BENCH] %d/%d segments carry a post-mt error, so their 'post-edited' text is "
                    "just the raw MT echoed back. The post-edited column and every delta derived "
                    "from it are NOT a measurement of APE. First error: %s",
                    len(self.last_failures), len(results), self.last_failures[0],
                )

            resolved = [r for r in results if r.has_glossary_resolved]
            blind = [r for r in resolved if r.has_glossary_reported is False]
            if blind:
                logger.warning(
                    "[BENCH] post-mt reported no glossary on %d/%d segments where this benchmark "
                    "resolved terms - the pipeline was very likely never shown them. Check "
                    "cat_project_id, cat_tool_provider and ecosystem_id; the post-edited column "
                    "is not meaningful until these agree.",
                    len(blind), len(resolved),
                )

        mt_aggregate = aggregate([r.mt for r in results])
        ape_aggregate = aggregate([r.ape for r in results])

        terms_fixed = terms_regressed = 0

        for result in results:
            mt_violated = {v.source_content for v in result.mt.violations}
            ape_violated = {v.source_content for v in result.ape.violations}
            terms_fixed += len(mt_violated - ape_violated)
            terms_regressed += len(ape_violated - mt_violated)

        mt_rate = mt_aggregate.adherence_rate
        ape_rate = ape_aggregate.adherence_rate
        delta_rate = ape_rate - mt_rate if mt_rate is not None and ape_rate is not None else None

        return BenchmarkResult(
            dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc).isoformat(),
            parameters={
                "source_language": source_language,
                "target_language": target_language,
                "domain": dataset.parameters.get("domain"),
                "cat_tool_provider": dataset.parameters.get("cat_tool_provider"),
                "cat_project_id": dataset.parameters.get("cat_project_id"),
            },
            glossary_ids=list(dataset.glossary_ids),
            steps=["(dry-run)"] if skip_pipeline else list(dataset.steps),
            config={
                "lemma_matching": self.config.benchmark.lemma_matching,
                "batch_size": self.config.benchmark.batch_size,
                "glossary_source": getattr(self.glossary, "source_label", "unknown"),
                "glossary_ids_source": dataset.glossary_ids_source,
            },
            usage=self.last_usage,
            failed_segments=0 if skip_pipeline else len(self.last_failures),
            failure_reason=(self.last_failures[0] if (self.last_failures and not skip_pipeline) else None),
            totals={
                "segments": len(dataset.segments),
                "segments_with_glossary": glossary_bearing,
                "segments_changed_by_ape": sum(1 for r in results if r.changed_by_ape),
            },
            mt_baseline=mt_aggregate,
            post_edited=ape_aggregate,
            delta=Delta(
                adherence_rate=delta_rate,
                violations_resolved=(
                    mt_aggregate.misses.substituted - ape_aggregate.misses.substituted
                ),
                terms_fixed_by_ape=terms_fixed,
                terms_regressed_by_ape=terms_regressed,
            ),
            segments=results,
        )
