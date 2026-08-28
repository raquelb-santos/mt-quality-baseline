"""DNT orchestration: revert, then score MT, APE, REV and REF on one item list."""

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

from .text_processing import Dataset, load
from .dnt_score import DntAggregate, DntScore, aggregate, score_dnt
from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, preflight_submission, raise_for_preflight

logger = logging.getLogger(__name__)


def load_dataset(path: Path, *, dry_run: bool) -> Dataset:
    """Only the submission preflight applies: glossary retrieval has no bearing on DNT items."""
    data = load(path, component="dnt")

    if not dry_run:
        raise_for_preflight(
            preflight_submission(data.parameters),
            "Preflight failed: post-mt would reject every segment, so there would be no "
            "APE column to score. Fix the parameters above.",
        )

    return data


@dataclass
class DntSegmentResult:
    """One segment's five versions, named the same way everywhere: SRC, MT, APE, REV, REF."""

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
    mt: DntScore
    ape: DntScore
    rev: DntScore
    ref: DntScore


@dataclass
class DntDelta:
    ape_preservation_rate: float | None
    rev_preservation_rate: float | None
    items_fixed_by_ape: int
    items_broken_by_ape: int
    items_fixed_by_rev: int
    items_broken_by_rev: int


@dataclass
class DntResult:
    dataset: str
    started_at: str
    finished_at: str
    parameters: dict[str, Any]
    totals: dict[str, int]
    mt: DntAggregate
    ape: DntAggregate
    rev: DntAggregate
    ref: DntAggregate
    delta: DntDelta
    # Detection runs in an LLM call, so two runs with one fingerprint shared a denominator.
    fingerprint: str = ""
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[DntSegmentResult] = field(default_factory=list)


def fingerprint_of(item_lists: list[list[str]]) -> str:
    items = sorted({item for items in item_lists for item in items})
    if not items:
        return "none"
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:8]


def _delta(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def _failing(score: DntScore) -> set[str]:
    return {item.text for item in score.item_scores if item.leaked or item.over_kept}


def _repairs(
    results: list[DntSegmentResult], before: str, after: str, label: str
) -> dict[str, int]:
    """Fixed and broken counted separately: swapping one failure for the other is not a repair."""
    fixed = broken = 0
    for result in results:
        was = _failing(getattr(result, before))
        now = _failing(getattr(result, after))
        fixed += len(was - now)
        broken += len(now - was)
    return {f"items_fixed_by_{label}": fixed, f"items_broken_by_{label}": broken}


class DntBenchmark:
    def __init__(self, *, postmt: Any, dnt: Any, config: Any) -> None:
        self.postmt = postmt
        self.dnt = dnt
        self.config = config

    def run(self, dataset: Dataset, *, skip_pipeline: bool = False) -> DntResult:
        started_at = datetime.now(timezone.utc).isoformat()
        source_language = dataset.parameters.get("clean_source_language_code")
        target_language = dataset.parameters.get("clean_target_language_code")

        outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
            self.postmt, dataset, batch_size=self.config.benchmark.batch_size
        )
        processed = outcome.segments
        failures = [] if skip_pipeline else outcome.failures

        src_texts = [s.get("source_content") or "" for s in dataset.segments]
        ref_texts = [s.get("reference_content") or "" for s in dataset.segments]
        mt_texts = [s.get("target_content") or "" for s in processed]
        ape_texts = [extract_post_edited(s) for s in processed]

        # Reversion runs on the last version there is; --dry-run makes that MT.
        reversions = self.dnt.revert(
            [
                {"id": str(index), "source": source, "target": target}
                for index, (source, target) in enumerate(zip(src_texts, ape_texts))
            ],
            batch_size=self.config.dnt.batch_size,
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
        logger.info(
            "[DNT] %d/%d segments carry at least one DNT item", carrying, len(dataset.segments)
        )
        if carrying == 0 and unread < len(reversions):
            logger.warning(
                "[DNT] no DNT items at all - check DNT_BASE_URL and the language pair before "
                "trusting a 0-instance result"
            )

        results: list[DntSegmentResult] = []
        for index, segment in enumerate(processed):
            reversion = reversions[index]
            items = per_segment_items[index]
            ape_text = ape_texts[index]
            rev_text = ape_text if reversion is None else reversion.rev_text

            common = dict(
                items=items,
                src_text=src_texts[index],
                ref_text=ref_texts[index],
                source_language_code=source_language,
                target_language_code=target_language,
            )

            results.append(DntSegmentResult(
                source_segment_id=str(
                    segment.get("source_segment_id")
                    or dataset.segments[index].get("source_segment_id")
                    or index
                ),
                src_text=src_texts[index],
                mt_text=mt_texts[index],
                ape_text=ape_text,
                rev_text=rev_text,
                ref_text=ref_texts[index],
                changed_by_ape=mt_texts[index] != ape_text,
                changed_by_rev=ape_text != rev_text,
                items=items,
                unread=reversion is None,
                mt=score_dnt(text=mt_texts[index], **common),
                ape=score_dnt(text=ape_text, **common),
                rev=score_dnt(text=rev_text, **common),
                ref=score_dnt(text=ref_texts[index], **common),
            ))

        if failures:
            logger.error(
                "[DNT] %d/%d segments carry a post-mt error, so their APE text is just MT "
                "echoed back. The APE column and every delta derived from it "
                "are NOT a measurement of APE. First error: %s",
                len(failures), len(results), failures[0],
            )

        scored = [r for r in results if not r.unread]
        mt_aggregate = aggregate([r.mt for r in scored], segments_unread=unread)
        ape_aggregate = aggregate([r.ape for r in scored], segments_unread=unread)
        rev_aggregate = aggregate([r.rev for r in scored], segments_unread=unread)
        ref_aggregate = aggregate([r.ref for r in scored], segments_unread=unread)

        return DntResult(
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
            mt=mt_aggregate,
            ape=ape_aggregate,
            rev=rev_aggregate,
            ref=ref_aggregate,
            delta=DntDelta(
                ape_preservation_rate=_delta(
                    mt_aggregate.preservation_rate, ape_aggregate.preservation_rate
                ),
                rev_preservation_rate=_delta(
                    ape_aggregate.preservation_rate, rev_aggregate.preservation_rate
                ),
                **_repairs(scored, "mt", "ape", "ape"),
                **_repairs(scored, "ape", "rev", "rev"),
            ),
            segments=results,
        )
