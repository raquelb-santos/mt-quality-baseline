"""Benchmark orchestration: resolve glossary matches, run post-mt, score MT, APE and REF."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .text_processing import Dataset, load
from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, preflight_parameters, raise_for_preflight, reported_has_glossary, segment_id
from .report import delta, report_parameters
from .glossary_score import Aggregate, ReferenceCheck, Score, ViolationReport, aggregate, check_reference, find_violations, score_glossary

logger = logging.getLogger(__name__)


def load_dataset(path: Path, *, glossary: Any, node: str, dry_run: bool) -> Dataset:
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
    source_segment_id: str
    src_text: str
    mt_text: str
    ape_text: str
    ref_text: str
    changed_by_ape: bool
    has_glossary_reported: bool | None
    has_glossary_resolved: bool
    glossary_terms: list[dict[str, str]]
    mt: Score
    ape: Score
    ref: Score


@dataclass
class Delta:
    ape_adherence_rate: float | None
    terms_fixed_by_ape: int
    terms_broken_by_ape: int


@dataclass
class Result:
    dataset: str
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
    ref_check: ReferenceCheck = field(default_factory=ReferenceCheck)
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[SegmentResult] = field(default_factory=list)


def _failing(score: Score) -> set[str]:
    return {v.source_content for v in score.violations}


def _repairs(results: list[SegmentResult], before: str, after: str) -> tuple[int, int]:
    """Fixed and broken counted separately: swapping one failure for the other is not a repair."""
    fixed = broken = 0
    for result in results:
        was = _failing(getattr(result, before))
        now = _failing(getattr(result, after))
        fixed += len(was - now)
        broken += len(now - was)
    return fixed, broken


def _resolve_glossary(
    segments: Sequence[dict[str, Any]],
    parameters: dict[str, Any],
    glossary_ids: Sequence[str],
    *,
    stanza: Any,
    glossary: Any,
) -> list[list[dict[str, str]]]:
    source_language = parameters.get("clean_source_language_code")
    # Order-preserving dedup — index alignment below depends on stable ordering.
    unique_sources = list(dict.fromkeys(s["source_content"] for s in segments))

    logger.info("[GLOSSARY] lemmatizing %d unique sources (%s)", len(unique_sources), source_language)

    # Degrade rather than abort: no glossary at all reads as a clean 0-instance scorecard.
    lemmatized = stanza.lemmatize_batch_safe(unique_sources, source_language)
    if lemmatized is None:
        logger.warning(
            "[GLOSSARY] falling back to un-lemmatized sources - retrieval will differ from production"
        )
        lemmatized = unique_sources

    matches = glossary.fetch_matches(
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

    by_source = dict(zip(unique_sources, matches.per_text_mappings))
    return [by_source[s["source_content"]] for s in segments]

def _build_target_lemmas(
    texts: Sequence[str], terms: Sequence[str], target_language: str, *, stanza: Any, config: Any
) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    if not config.benchmark.lemma_matching:
        return None, None

    unique_texts = list(dict.fromkeys(t for t in texts if t))
    unique_terms = list(dict.fromkeys(t for t in terms if t))
    if not unique_texts and not unique_terms:
        return None, None

    lemmas = stanza.lemmatize_batch_safe(unique_texts + unique_terms, target_language)
    if lemmas is None:
        logger.warning(
            "[SCORE] proceeding with surface-form matching only - inflected forms will count as violations"
        )
        return None, None

    return (
        dict(zip(unique_texts, lemmas[: len(unique_texts)])),
        dict(zip(unique_terms, lemmas[len(unique_texts) :])),
    )


def run_benchmark(
    dataset: Dataset, *, postmt: Any, stanza: Any, glossary: Any, config: Any,
    skip_pipeline: bool = False,
) -> Result:
    target_language = dataset.parameters.get("clean_target_language_code")

    per_segment_mappings = _resolve_glossary(
        dataset.segments, dataset.parameters, dataset.glossary_ids,
        stanza=stanza, glossary=glossary,
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
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    processed = outcome.segments
    failures = outcome.failures

    mt_texts = [s.get("target_content") or "" for s in dataset.segments]
    ape_texts = [extract_post_edited(s) for s in processed]
    ref_texts = [s.get("reference_content") or "" for s in dataset.segments]

    text_lemmas, term_lemmas = _build_target_lemmas(
        mt_texts + ape_texts + ref_texts,
        [m["target_content"] for mappings in per_segment_mappings for m in mappings],
        target_language,
        stanza=stanza,
        config=config,
    )
    text_lemmas = text_lemmas or {}

    results: list[SegmentResult] = []
    for i, segment in enumerate(processed):
        original = dataset.segments[i]
        mappings = per_segment_mappings[i]
        mt_text, ape_text, ref_text = mt_texts[i], ape_texts[i], ref_texts[i]

        common = dict(
            mappings=mappings,
            language_code=target_language,
            term_lemmas=term_lemmas,
            ref_text=ref_text,
            ref_lemmas=text_lemmas.get(ref_text),
        )

        results.append(SegmentResult(
            source_segment_id=segment_id(segment, original, i),
            src_text=original.get("source_content", ""),
            mt_text=mt_text,
            ape_text=ape_text,
            ref_text=ref_text,
            changed_by_ape=mt_text != ape_text,
            has_glossary_reported=reported_has_glossary(segment),
            has_glossary_resolved=bool(mappings),
            glossary_terms=mappings,
            mt=score_glossary(text=mt_text, text_lemmas=text_lemmas.get(mt_text), **common),
            ape=score_glossary(text=ape_text, text_lemmas=text_lemmas.get(ape_text), **common),
            ref=score_glossary(text=ref_text, text_lemmas=text_lemmas.get(ref_text), **common),
        ))

    resolved = [r for r in results if r.has_glossary_resolved]
    # Resolved here but never shown to post-mt, so the APE column measures something else.
    blind = [] if skip_pipeline else [r for r in resolved if r.has_glossary_reported is False]

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

    # Corpus-level, so it runs once over every segment rather than inside the loop above.
    mt_violations, ape_violations = find_violations(
        versions=[mt_texts, ape_texts],
        ref_texts=ref_texts,
        per_segment_mappings=per_segment_mappings,
        corpus_mappings=[m for mappings in per_segment_mappings for m in mappings],
        language_code=target_language,
        text_lemmas=text_lemmas,
        term_lemmas=term_lemmas,
    )

    return Result(
        dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
        parameters=report_parameters(dataset),
        glossary_ids=list(dataset.glossary_ids),
        config={"lemma_matching": config.benchmark.lemma_matching},
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
        ref=aggregate([r.ref for r in results]),
        mt_violations=mt_violations,
        ape_violations=ape_violations,
        ref_check=check_reference(
            ref_texts=ref_texts,
            per_segment_mappings=per_segment_mappings,
            language_code=target_language,
            text_lemmas=text_lemmas,
            term_lemmas=term_lemmas,
        ),
        delta=Delta(
            delta(mt_aggregate.adherence_rate, ape_aggregate.adherence_rate),
            *_repairs(results, "mt", "ape"),
        ),
        segments=results,
    )
