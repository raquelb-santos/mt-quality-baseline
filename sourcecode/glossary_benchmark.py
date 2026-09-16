"""Benchmark orchestration: resolve glossary matches, run post-mt, score MT, APE and REF."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .text_processing import Dataset, Task, load
from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, preflight_parameters, preflight_tasks, raise_for_preflight, reported_has_glossary, segment_id
from .report import delta, report_parameters, stratum_of
from .glossary_score import Aggregate, ReferenceCheck, Score, ViolationReport, aggregate, check_reference, find_violations, score_glossary

logger = logging.getLogger(__name__)


def load_dataset(path: Path, *, term_bases: Any, dry_run: bool, languages: Sequence[tuple[str, str]] = ()) -> Dataset:
    data = load(path, component="glossary", languages=languages)

    # Nothing in this file is in the pairs asked for, so there is no term base to expect either.
    if not data.tasks:
        return data

    # For some parameter sets post-mt silently retrieves no glossary, at full LLM cost.
    if not dry_run:
        raise_for_preflight(
            preflight_tasks(data.tasks, preflight_parameters),
            "Preflight failed: post-mt would run these segments but retrieve no glossary, "
            "so the APE column would measure nothing - at full LLM cost. "
            "Fix the parameters above.",
        )

    # Term bases that are not there do not fail: they match nothing and score a clean-looking 0.
    resolved = sum(
        1 for task in data.tasks
        if term_bases.ids_for(task.parameters.get("cat_project_id"),
                              task.parameters.get("cat_tool_provider"))
    )
    if not resolved:
        # Dropped rather than scored, because a 0 from no terms reads like a clean result.
        logger.warning(
            "[BENCH] none of the %d CAT projects in %s has a term base attached - not scored. "
            "Check cat_project_id and cat_tool_provider, and that these credentials can see "
            "those projects.", len(data.tasks), data.name,
        )
        data.tasks = []

    return data


@dataclass
class SegmentResult:
    source_segment_id: str
    stratum: tuple[str, str]
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
    # Corpus-level violations landing on this segment, which pool into its stratum's row.
    mt_corpus_violations: int = 0
    ape_corpus_violations: int = 0


@dataclass
class Delta:
    ape_adherence_rate: float | None
    terms_fixed_by_ape: int
    terms_broken_by_ape: int


@dataclass
class Result:
    dataset: str
    parameters: dict[str, Any]
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


def _resolve_task(
    task: Task, *, stanza: Any, glossary: Any, term_bases: Any
) -> list[list[dict[str, str]]]:
    """Each task retrieves from the term bases its own CAT project has attached, for its own
    language pair. A project with none retrieves nothing, and never reaches the lemmatizer."""
    source_language = task.parameters.get("clean_source_language_code")
    provider = task.parameters.get("cat_tool_provider")

    glossary_ids = term_bases.ids_for(task.parameters.get("cat_project_id"), provider)
    if not glossary_ids:
        return [[] for _ in task.segments]

    # Order-preserving dedup — index alignment below depends on stable ordering.
    unique_sources = list(dict.fromkeys(s["source_content"] for s in task.segments))

    logger.debug("[GLOSSARY] lemmatizing %d unique sources (%s)", len(unique_sources), source_language)

    # Degrade rather than abort: no glossary at all reads as a clean 0-instance scorecard.
    lemmatized = stanza.lemmatize_batch_safe(unique_sources, source_language)
    if lemmatized is None:
        logger.warning(
            "[GLOSSARY] falling back to un-lemmatized sources - retrieval will differ from production"
        )
        lemmatized = unique_sources

    matches = glossary.fetch_matches(
        glossary_ids=glossary_ids,
        source_language=source_language,
        target_language=task.parameters.get("clean_target_language_code"),
        texts=lemmatized,
        provider=provider,
    )

    by_source = dict(zip(unique_sources, matches.per_text_mappings))
    return [by_source[s["source_content"]] for s in task.segments]


def _resolve_glossary(
    dataset: Dataset, *, stanza: Any, glossary: Any, term_bases: Any
) -> list[list[dict[str, str]]]:
    per_segment = [
        mappings
        for task in dataset.tasks
        for mappings in _resolve_task(task, stanza=stanza, glossary=glossary, term_bases=term_bases)
    ]

    distinct = {(m["source_content"], m["target_content"]) for mappings in per_segment for m in mappings}
    logger.info(
        "[GLOSSARY] %d distinct term mappings across %d segments in %d task(s)",
        len(distinct), len(per_segment), len(dataset.tasks),
    )
    return per_segment


def _build_lemmas(
    texts_by_language: Mapping[str, Sequence[str]], *, stanza: Any, config: Any
) -> dict[str, dict[str, str]]:
    """One lemma map per language, since one file can cover several."""
    if not config.benchmark.lemma_matching:
        return {}

    by_language: dict[str, dict[str, str]] = {}
    for language, texts in texts_by_language.items():
        unique = [text for text in dict.fromkeys(texts) if text]
        if not unique:
            continue

        lemmas = stanza.lemmatize_batch_safe(unique, language)
        if lemmas is None:
            logger.warning(
                "[SCORE] proceeding with surface-form matching only for %s - inflected forms will "
                "count as violations", language,
            )
            continue

        by_language[language] = dict(zip(unique, lemmas))

    return by_language


def run_benchmark(
    dataset: Dataset, *, postmt: Any, stanza: Any, glossary: Any, term_bases: Any, config: Any,
    skip_pipeline: bool = False,
) -> Result:
    target_languages = dataset.per_segment("clean_target_language_code")
    source_languages = dataset.per_segment("clean_source_language_code")
    strata = [stratum_of(p) for p in dataset.parameters_per_segment()]

    per_segment_mappings = _resolve_glossary(
        dataset, stanza=stanza, glossary=glossary, term_bases=term_bases
    )

    glossary_bearing = sum(1 for m in per_segment_mappings if m)
    logger.info(
        "[BENCH] %d/%d segments carry at least one glossary term",
        glossary_bearing, len(dataset.segments),
    )
    if glossary_bearing == 0:
        logger.warning(
            "[BENCH] no glossary matches at all - check the language codes and that these CAT "
            "projects have term bases attached, before trusting a 0-instance result"
        )

    outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    processed = outcome.segments
    failures = outcome.failures

    originals = dataset.segments
    mt_texts = [s.get("target_content") or "" for s in originals]
    ape_texts = [extract_post_edited(s) for s in processed]
    ref_texts = [s.get("reference_content") or "" for s in originals]
    source_texts = [s.get("source_content") or "" for s in originals]

    # Terms and texts are lemmatized in the language of the task they belong to.
    texts_by_language: dict[str, list[str]] = {}
    for i, language in enumerate(target_languages):
        texts_by_language.setdefault(language, []).extend(
            [mt_texts[i], ape_texts[i], ref_texts[i], *(m["target_content"] for m in per_segment_mappings[i])]
        )

    # Sources too, so over-application can tell a term the source carries from one it does not.
    corpus_sources = list(dict.fromkeys(m["source_content"] for ms in per_segment_mappings for m in ms))
    for i, language in enumerate(source_languages):
        texts_by_language.setdefault(language, []).append(source_texts[i])
    for language in dict.fromkeys(source_languages):
        texts_by_language[language].extend(corpus_sources)

    lemmas_by_language = _build_lemmas(texts_by_language, stanza=stanza, config=config)

    results: list[SegmentResult] = []
    for i, segment in enumerate(processed):
        original = originals[i]
        mappings = per_segment_mappings[i]
        mt_text, ape_text, ref_text = mt_texts[i], ape_texts[i], ref_texts[i]
        lemmas = lemmas_by_language.get(target_languages[i], {})

        common = dict(
            mappings=mappings,
            language_code=target_languages[i],
            term_lemmas=lemmas,
            ref_text=ref_text,
            ref_lemmas=lemmas.get(ref_text),
        )

        results.append(SegmentResult(
            source_segment_id=segment_id(segment, original, i),
            stratum=strata[i],
            src_text=original.get("source_content", ""),
            mt_text=mt_text,
            ape_text=ape_text,
            ref_text=ref_text,
            changed_by_ape=mt_text != ape_text,
            has_glossary_reported=reported_has_glossary(segment),
            has_glossary_resolved=bool(mappings),
            glossary_terms=mappings,
            mt=score_glossary(text=mt_text, text_lemmas=lemmas.get(mt_text), **common),
            ape=score_glossary(text=ape_text, text_lemmas=lemmas.get(ape_text), **common),
            ref=score_glossary(text=ref_text, text_lemmas=lemmas.get(ref_text), **common),
        ))

    resolved = [r for r in results if r.has_glossary_resolved]
    # Resolved here but never shown to post-mt, so the APE column measures something else.
    blind = [] if skip_pipeline else [r for r in resolved if r.has_glossary_reported is False]

    if blind:
        logger.warning(
            "[BENCH] post-mt reported no glossary on %d/%d segments where terms were resolved - "
            "the APE column means nothing until cat_project_id and cat_tool_provider agree.",
            len(blind), len(resolved),
        )

    mt_aggregate = aggregate([r.mt for r in results])
    ape_aggregate = aggregate([r.ape for r in results])

    # Corpus-level, so it runs once over every segment rather than inside the loop above.
    mt_violations, ape_violations = find_violations(
        versions=[mt_texts, ape_texts],
        ref_texts=ref_texts,
        source_texts=source_texts,
        per_segment_mappings=per_segment_mappings,
        corpus_mappings=[m for mappings in per_segment_mappings for m in mappings],
        language_codes=target_languages,
        source_language_codes=source_languages,
        lemmas_by_language=lemmas_by_language,
    )

    for report, column in ((mt_violations, "mt_corpus_violations"),
                           (ape_violations, "ape_corpus_violations")):
        for item in report.items:
            result = results[item.segment_index]
            setattr(result, column, getattr(result, column) + 1)

    return Result(
        dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
        parameters=report_parameters(dataset),
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
            language_codes=target_languages,
            lemmas_by_language=lemmas_by_language,
        ),
        delta=Delta(
            delta(mt_aggregate.adherence_rate, ape_aggregate.adherence_rate),
            *_repairs(results, "mt", "ape"),
        ),
        segments=results,
    )
