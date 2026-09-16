"""DNT orchestration: revert, then score MT, APE, REV and REF on one item list."""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from .text_processing import Dataset
from .dnt_score import Aggregate, ReversionAggregate, ReversionScore, Score, aggregate, aggregate_reversion, score_dnt, score_reversion
from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, segment_id
from .report import delta, report_parameters, stratum_of

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    source_segment_id: str
    stratum: tuple[str, str]
    src_text: str
    mt_text: str
    ape_text: str
    rev_text: str
    ref_text: str
    changed_by_ape: bool
    changed_by_rev: bool
    items: list[str]
    unread: bool
    mt: Score
    ape: Score
    rev: Score
    ref: Score


@dataclass
class Delta:
    ape_preservation_rate: float | None
    rev_preservation_rate: float | None
    items_fixed_by_ape: int
    items_broken_by_ape: int
    items_fixed_by_rev: int
    items_broken_by_rev: int


@dataclass
class Result:
    dataset: str
    parameters: dict[str, Any]
    totals: dict[str, int]
    mt: Aggregate
    ape: Aggregate
    rev: Aggregate
    ref: Aggregate
    delta: Delta
    # Detection runs in an LLM call, so two runs with one fingerprint shared a denominator.
    fingerprint: str = ""
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[SegmentResult] = field(default_factory=list)


def fingerprint_of(item_lists: list[list[str]]) -> str:
    items = sorted({item for items in item_lists for item in items})
    if not items:
        return "none"
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:8]


def _failing(score: Score) -> set[str]:
    return {item.text for item in score.item_scores if item.leaked or item.over_kept}


def _repairs(results: list[SegmentResult], before: str, after: str) -> tuple[int, int]:
    """Fixed and broken counted separately: swapping one failure for the other is not a repair."""
    fixed = broken = 0
    for result in results:
        was = _failing(getattr(result, before))
        now = _failing(getattr(result, after))
        fixed += len(was - now)
        broken += len(now - was)
    return fixed, broken


def run_benchmark(dataset: Dataset, *, postmt: Any, dnt: Any, config: Any, skip_pipeline: bool = False) -> Result:
    source_languages = dataset.per_segment("clean_source_language_code")
    target_languages = dataset.per_segment("clean_target_language_code")
    strata = [stratum_of(p) for p in dataset.parameters_per_segment()]

    outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    processed = outcome.segments

    originals = dataset.segments
    src_texts = [s.get("source_content") or "" for s in originals]
    ref_texts = [s.get("reference_content") or "" for s in originals]
    mt_texts = [s.get("target_content") or "" for s in originals]
    ape_texts = [extract_post_edited(s) for s in processed]

    # One pass per language pair: the service detects for the pair it is told.
    by_pair: dict[tuple[str, str], list[int]] = {}
    for index, pair in enumerate(zip(source_languages, target_languages)):
        by_pair.setdefault(pair, []).append(index)

    # Reversion runs on the last version there is; --dry-run makes that MT.
    reversions: list[Any] = [None] * len(originals)
    for (source_language, target_language), indexes in by_pair.items():
        returned = dnt.revert(
            [
                {"id": str(index), "source": src_texts[index], "target": ape_texts[index]}
                for index in indexes
            ],
            batch_size=config.dnt.batch_size,
            source_language=source_language,
            target_language=target_language,
        )
        for index, reversion in zip(indexes, returned):
            reversions[index] = reversion

    unread = sum(1 for reversion in reversions if reversion is None)
    if unread:
        logger.error(
            "[DNT] %d/%d segments came back from no revert batch. They are excluded from the "
            "denominator rather than scored as having nothing to preserve - a rate over the "
            "rest is still meaningful, one that counted them as perfect would not be.",
            unread, len(reversions),
        )

    per_segment_items = [[] if r is None else list(r.items) for r in reversions]
    carrying = sum(1 for items in per_segment_items if items)
    logger.info("[DNT] %d/%d segments carry at least one DNT item", carrying, len(dataset.segments))
    if carrying == 0 and unread < len(reversions):
        logger.warning(
            "[DNT] no DNT items at all - check DNT_BASE_URL and the language pair before "
            "trusting a 0-instance result"
        )

    results: list[SegmentResult] = []
    for index, segment in enumerate(processed):
        reversion = reversions[index]
        items = per_segment_items[index]
        src_text, ref_text = src_texts[index], ref_texts[index]
        mt_text, ape_text = mt_texts[index], ape_texts[index]
        rev_text = ape_text if reversion is None else reversion.rev_text

        common = dict(
            items=items,
            src_text=src_text,
            ref_text=ref_text,
            source_language_code=source_languages[index],
            target_language_code=target_languages[index],
        )

        results.append(SegmentResult(
            source_segment_id=segment_id(segment, originals[index], index),
            stratum=strata[index],
            src_text=src_text,
            mt_text=mt_text,
            ape_text=ape_text,
            rev_text=rev_text,
            ref_text=ref_text,
            changed_by_ape=mt_text != ape_text,
            changed_by_rev=ape_text != rev_text,
            items=items,
            unread=reversion is None,
            mt=score_dnt(text=mt_text, **common),
            ape=score_dnt(text=ape_text, **common),
            rev=score_dnt(text=rev_text, **common),
            ref=score_dnt(text=ref_text, **common),
        ))

    failures = outcome.failures

    scored = [r for r in results if not r.unread]
    mt = aggregate([r.mt for r in scored], segments_unread=unread)
    ape = aggregate([r.ape for r in scored], segments_unread=unread)
    rev = aggregate([r.rev for r in scored], segments_unread=unread)
    ref = aggregate([r.ref for r in scored], segments_unread=unread)

    return Result(
        dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
        parameters=report_parameters(dataset),
        fingerprint=fingerprint_of(per_segment_items),
        usage=outcome.usage,
        failed_segments=len(failures),
        failure_reason=failures[0] if failures else None,
        totals={
            "segments": len(dataset.segments),
            "segments_read": len(scored),
            "segments_with_items": carrying,
            "segments_changed_by_ape": sum(1 for r in results if r.changed_by_ape),
            "segments_changed_by_rev": sum(1 for r in results if r.changed_by_rev),
        },
        mt=mt,
        ape=ape,
        rev=rev,
        ref=ref,
        delta=Delta(
            delta(mt.preservation_rate, ape.preservation_rate),
            delta(ape.preservation_rate, rev.preservation_rate),
            *_repairs(scored, "mt", "ape"),
            *_repairs(scored, "ape", "rev"),
        ),
        segments=results,
    )


@dataclass
class ReversionSegmentResult:
    source_segment_id: str
    stratum: tuple[str, str]
    src_text: str
    mt_text: str
    ape_text: str
    rev_text: str
    rev_ape_text: str
    changed_by_ape: bool
    changed_by_rev: bool
    changed_by_rev_ape: bool
    unread: bool
    expected_terms: list[str]
    detected_items: list[str]
    score: ReversionScore


@dataclass
class ReversionResult:
    dataset: str
    parameters: dict[str, Any]
    totals: dict[str, int]
    scored: ReversionAggregate
    fingerprint: str = ""
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[ReversionSegmentResult] = field(default_factory=list)


def run_reversion(dataset: Dataset, *, postmt: Any, dnt: Any, config: Any,
                  skip_pipeline: bool = False) -> ReversionResult:
    """Post-edit, then revert both MT and APE so the same gold terms are checked on each."""
    originals = dataset.segments
    strata = [stratum_of(p) for p in dataset.parameters_per_segment()]
    target_languages = dataset.per_segment("clean_target_language_code")

    src_texts = [s.get("source_content") or "" for s in originals]
    mt_texts = [s.get("target_content") or "" for s in originals]

    outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    ape_texts = [extract_post_edited(s) for s in outcome.segments]

    # One pass per language pair, which is also how the gold set is grouped into tasks.
    reversions: list[Any] = [None] * len(originals)
    ape_reversions: list[Any] = [None] * len(originals)
    index = 0
    for task in dataset.tasks:
        span = range(index, index + len(task.segments))
        languages = dict(
            source_language=task.parameters.get("clean_source_language_code"),
            target_language=task.parameters.get("clean_target_language_code"),
        )
        for texts, into in ((mt_texts, reversions), (ape_texts, ape_reversions)):
            returned = dnt.revert(
                [{"id": str(i), "source": src_texts[i], "target": texts[i]} for i in span],
                batch_size=config.dnt.batch_size,
                **languages,
            )
            for i, reversion in zip(span, returned):
                into[i] = reversion
        index += len(task.segments)

    # A pair is unread when either arm came back from no batch: one arm alone cannot be compared.
    unread = sum(1 for mt, ape in zip(reversions, ape_reversions) if mt is None or ape is None)
    if unread:
        logger.error(
            "[DNT] %d/%d pairs came back from no revert batch. They are excluded from the "
            "denominator rather than scored as having lost every term.",
            unread, len(reversions),
        )

    results: list[ReversionSegmentResult] = []
    for i, segment in enumerate(originals):
        reversion, ape_reversion = reversions[i], ape_reversions[i]
        rev_text = mt_texts[i] if reversion is None else reversion.rev_text
        rev_ape_text = ape_texts[i] if ape_reversion is None else ape_reversion.rev_text
        results.append(ReversionSegmentResult(
            source_segment_id=segment.get("source_segment_id", str(i)),
            stratum=strata[i],
            src_text=src_texts[i],
            mt_text=mt_texts[i],
            ape_text=ape_texts[i],
            rev_text=rev_text,
            rev_ape_text=rev_ape_text,
            changed_by_ape=mt_texts[i] != ape_texts[i],
            changed_by_rev=mt_texts[i] != rev_text,
            changed_by_rev_ape=ape_texts[i] != rev_ape_text,
            unread=reversion is None or ape_reversion is None,
            expected_terms=list(segment.get("expected_terms") or []),
            detected_items=[] if reversion is None else list(reversion.items),
            score=score_reversion(
                terms=segment.get("expected_terms") or [],
                mt_text=mt_texts[i],
                ape_text=ape_texts[i],
                rev_text=rev_text,
                rev_ape_text=rev_ape_text,
                target_language_code=target_languages[i],
            ),
        ))

    scored = [r for r in results if not r.unread]
    carrying = sum(1 for r in results if r.expected_terms)
    failures = outcome.failures

    return ReversionResult(
        dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
        parameters=report_parameters(dataset),
        fingerprint=fingerprint_of([r.detected_items for r in results]),
        usage=outcome.usage,
        failed_segments=len(failures),
        failure_reason=failures[0] if failures else None,
        totals={
            "segments": len(originals),
            "segments_read": len(scored),
            "segments_with_terms": carrying,
            "segments_changed_by_ape": sum(1 for r in results if r.changed_by_ape),
            "segments_changed_by_rev": sum(1 for r in results if r.changed_by_rev),
            "segments_changed_by_rev_ape": sum(1 for r in results if r.changed_by_rev_ape),
        },
        scored=aggregate_reversion([r.score for r in scored], segments_unread=unread),
        segments=results,
    )
