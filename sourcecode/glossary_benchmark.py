"""Benchmark orchestration: resolve glossary matches, run post-mt, score MT, APE and REF."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence

from pathlib import Path

from .text_processing import Dataset, load
from .pipeline import run_pipeline, stub_pipeline
from .postmt import (
    Usage, extract_post_edited, preflight_parameters, raise_for_preflight, reported_has_glossary,
)
from .glossary_score import (
    Aggregate, TranslationScore, ViolationReport, aggregate, find_violations, score_translation,
)

logger = logging.getLogger(__name__)


def load_dataset(path: Path, *, glossary: Any, node: str, dry_run: bool) -> Dataset:
    """One terminology dataset, with the two guards that stop a meaningless run before it costs."""
    data = load(path, component="glossary")

    # An id not in this cluster does not fail: it matches nothing and scores a clean-looking 0.
    if not glossary.count_terms(data.glossary_ids, data.parameters.get("cat_tool_provider")):
        raise RuntimeError(
            f"None of the glossary ids ({', '.join(data.glossary_ids)}) exist in the "
            f"term-bases index at {node}. The run would score 0 expected instances and read "
            f"like a clean result. Check the ids are CAT term-base uids rather than another "
            f"system's and that this is the cluster post-mt queries."
        )

    # For some parameter sets post-mt silently retrieves no glossary, at full LLM cost.
    if not dry_run:
        raise_for_preflight(
            preflight_parameters(data.parameters),
            "Preflight failed: post-mt would run these segments but retrieve no glossary, "
            "so the APE column would measure nothing - at full LLM cost. "
            "Fix the parameters above.",
        )

    return data


@dataclass
class SegmentResult:
    """One segment's versions, named the same way everywhere: SRC, MT, APE, REF."""

    source_segment_id: str
    src_text: str
    mt_text: str
    ape_text: str
    ref_text: str
    changed_by_ape: bool
    has_glossary_reported: bool | None
    has_glossary_resolved: bool
    glossary_terms: list[dict[str, str]]
    mt: TranslationScore
    ape: TranslationScore
    ref: TranslationScore


@dataclass
class Delta:
    adherence_rate: float | None
    terms_fixed_by_ape: int
    terms_regressed_by_ape: int


@dataclass
class BenchmarkResult:
    dataset: str
    started_at: str
    finished_at: str
    parameters: dict[str, Any]
    glossary_ids: list[str]
    config: dict[str, Any]
    totals: dict[str, int]
    mt: Aggregate
    ape: Aggregate
    ref: Aggregate
    delta: Delta
    mt_violations: ViolationReport = field(default_factory=ViolationReport)
    ape_violations: ViolationReport = field(default_factory=ViolationReport)
    ref_violations: ViolationReport = field(default_factory=ViolationReport)
    # Miss / inconsistency / over-application, and the share of segments carrying at least one.
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

    def resolve_glossary(
        self, segments: Sequence[dict[str, Any]], parameters: dict[str, Any], glossary_ids: Sequence[str]
    ) -> list[list[dict[str, str]]]:
        source_language = parameters.get("clean_source_language_code")
        # Order-preserving dedup — index alignment below depends on stable ordering.
        unique_sources = list(dict.fromkeys(s["source_content"] for s in segments))

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

        matches = self.glossary.fetch_matches(
            glossary_ids=list(glossary_ids),
            source_language=source_language,
            target_language=parameters.get("clean_target_language_code"),
            texts=lemmatized,
            provider=parameters.get("cat_tool_provider"),
        )

        logger.info(
            "[GLOSSARY] %d distinct term mappings across %d unique sources",
            len(matches.mappings), len(unique_sources),
        )

        index_of = {text: i for i, text in enumerate(unique_sources)}
        return [
            matches.per_text_mappings[index_of[s["source_content"]]]
            if s["source_content"] in index_of else []
            for s in segments
        ]

    def build_target_lemmas(
        self, texts: Sequence[str], terms: Sequence[str], target_language: str
    ) -> tuple[dict[str, str] | None, dict[str, str] | None]:
        if not self.config.benchmark.lemma_matching:
            return None, None

        unique_texts = list(dict.fromkeys(t for t in texts if t))
        unique_terms = list(dict.fromkeys(t for t in terms if t))
        combined = unique_texts + unique_terms
        if not combined:
            return None, None

        lemmas = self.stanza.lemmatize_batch_safe(combined, target_language)
        if lemmas is None:
            logger.warning(
                "[SCORE] proceeding with surface-form matching only - inflected forms will count as violations"
            )
            return None, None

        return (
            dict(zip(unique_texts, lemmas[: len(unique_texts)])),
            dict(zip(unique_terms, lemmas[len(unique_texts) :])),
        )

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

        outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
            self.postmt, dataset, batch_size=self.config.benchmark.batch_size
        )
        processed = outcome.segments
        failures = [] if skip_pipeline else outcome.failures

        mt_texts = [s.get("target_content") or "" for s in processed]
        ape_texts = [extract_post_edited(s) for s in processed]
        ref_texts = [s.get("reference_content") or "" for s in dataset.segments]

        text_lemmas, term_lemmas = self.build_target_lemmas(
            mt_texts + ape_texts + ref_texts,
            [m["target_content"] for mappings in per_segment_mappings for m in mappings],
            target_language,
        )
        text_lemmas = text_lemmas or {}

        results: list[SegmentResult] = []
        for i, segment in enumerate(processed):
            original = dataset.segments[i]
            mappings = per_segment_mappings[i]
            mt_text = mt_texts[i]
            ape_text = ape_texts[i]
            ref_text = ref_texts[i]

            common = dict(
                mappings=mappings,
                language_code=target_language,
                term_lemmas=term_lemmas,
                ref_text=ref_text,
                ref_lemmas=text_lemmas.get(ref_text),
            )

            results.append(SegmentResult(
                source_segment_id=str(
                    segment.get("source_segment_id") or original.get("source_segment_id") or i
                ),
                src_text=original.get("source_content", ""),
                mt_text=mt_text,
                ape_text=ape_text,
                ref_text=ref_text,
                changed_by_ape=mt_text != ape_text,
                has_glossary_reported=reported_has_glossary(segment),
                has_glossary_resolved=bool(mappings),
                glossary_terms=mappings,
                mt=score_translation(text=mt_text, text_lemmas=text_lemmas.get(mt_text), **common),
                ape=score_translation(text=ape_text, text_lemmas=text_lemmas.get(ape_text), **common),
                ref=score_translation(
                    text=ref_text, text_lemmas=text_lemmas.get(ref_text), **common
                ),
            ))

        # Resolved here but never shown to post-mt, so the APE column measures something else.
        resolved = [r for r in results if r.has_glossary_resolved]
        blind = [] if skip_pipeline else [r for r in resolved if r.has_glossary_reported is False]

        if failures:
            logger.error(
                "[BENCH] %d/%d segments carry a post-mt error, so their APE text is just MT "
                "echoed back. The APE column and every delta derived "
                "from it are NOT a measurement of APE. First error: %s",
                len(failures), len(results), failures[0],
            )

        if blind:
            logger.warning(
                "[BENCH] post-mt reported no glossary on %d/%d segments where this benchmark "
                "resolved terms - the pipeline was very likely never shown them. Check "
                "cat_project_id, cat_tool_provider and ecosystem_id; the APE column "
                "is not meaningful until these agree.",
                len(blind), len(resolved),
            )

        mt_aggregate = aggregate([r.mt for r in results])
        ape_aggregate = aggregate([r.ape for r in results])
        ref_aggregate = aggregate([r.ref for r in results])

        # Corpus-level, so it runs once over every segment rather than inside the loop above.
        violation_kwargs = dict(
            ref_texts=ref_texts,
            per_segment_mappings=per_segment_mappings,
            corpus_mappings=[m for mappings in per_segment_mappings for m in mappings],
            language_code=target_language,
            text_lemmas=text_lemmas,
            term_lemmas=term_lemmas,
        )

        terms_fixed = terms_regressed = 0
        for result in results:
            mt_violated = {v.source_content for v in result.mt.violations}
            ape_violated = {v.source_content for v in result.ape.violations}
            terms_fixed += len(mt_violated - ape_violated)
            terms_regressed += len(ape_violated - mt_violated)

        mt_rate, ape_rate = mt_aggregate.adherence_rate, ape_aggregate.adherence_rate

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
            config={"lemma_matching": self.config.benchmark.lemma_matching},
            usage=outcome.usage,
            failed_segments=len(failures),
            failure_reason=failures[0] if failures else None,
            totals={
                "segments": len(dataset.segments),
                "segments_with_glossary": glossary_bearing,
                "segments_glossary_never_shown": len(blind),
                "segments_changed_by_ape": sum(1 for r in results if r.changed_by_ape),
            },
            mt=mt_aggregate,
            ape=ape_aggregate,
            ref=ref_aggregate,
            mt_violations=find_violations(texts=mt_texts, **violation_kwargs),
            ape_violations=find_violations(texts=ape_texts, **violation_kwargs),
            ref_violations=find_violations(texts=ref_texts, **violation_kwargs),
            delta=Delta(
                adherence_rate=(
                    None if mt_rate is None or ape_rate is None else ape_rate - mt_rate
                ),
                terms_fixed_by_ape=terms_fixed,
                terms_regressed_by_ape=terms_regressed,
            ),
            segments=results,
        )
