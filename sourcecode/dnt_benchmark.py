"""DNT orchestration: revert, then score MT, APE, REV and REF on one item list."""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Any

from .text_processing import Dataset
from .dnt_score import Aggregate, Score, aggregate, score_dnt
from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, segment_id
from .report import delta, report_parameters

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    source_segment_id: str
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
    source_language = dataset.parameters.get("clean_source_language_code")
    target_language = dataset.parameters.get("clean_target_language_code")

    outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    processed = outcome.segments

    src_texts = [s.get("source_content") or "" for s in dataset.segments]
    ref_texts = [s.get("reference_content") or "" for s in dataset.segments]
    mt_texts = [s.get("target_content") or "" for s in dataset.segments]
    ape_texts = [extract_post_edited(s) for s in processed]

    # Reversion runs on the last version there is; --dry-run makes that MT.
    reversions = dnt.revert(
        [
            {"id": str(index), "source": source, "target": target}
            for index, (source, target) in enumerate(zip(src_texts, ape_texts))
        ],
        batch_size=config.dnt.batch_size,
        source_language=source_language,
        target_language=target_language,
    )

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
            source_language_code=source_language,
            target_language_code=target_language,
        )

        results.append(SegmentResult(
            source_segment_id=segment_id(segment, dataset.segments[index], index),
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
