"""Tag integrity orchestration: read the tags off SRC, then score MT, APE and REF against them."""

import logging
from dataclasses import dataclass, field
from typing import Any

from .pipeline import run_pipeline, stub_pipeline
from .postmt import Usage, extract_post_edited, segment_id
from .report import delta, report_parameters, stratum_of
from .tags import extract_tags
from .tags_score import Aggregate, Score, aggregate, score_tags
from .text_processing import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SegmentResult:
    source_segment_id: str
    stratum: tuple[str, str]
    src_text: str
    mt_text: str
    ape_text: str
    ref_text: str
    changed_by_ape: bool
    tags: list[str]
    mt: Score
    ape: Score
    ref: Score


@dataclass
class Delta:
    ape_integrity_rate: float | None
    tags_fixed_by_ape: int
    tags_broken_by_ape: int


@dataclass
class Result:
    dataset: str
    parameters: dict[str, Any]
    totals: dict[str, int]
    mt: Aggregate
    ape: Aggregate
    ref: Aggregate
    delta: Delta
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    segments: list[SegmentResult] = field(default_factory=list)


def _failing(score: Score) -> set[str]:
    return {tag.text for tag in score.tag_scores if tag.dropped or tag.added}


def _repairs(results: list[SegmentResult], before: str, after: str) -> tuple[int, int]:
    """Fixed and broken counted separately: swapping one failure for the other is not a repair."""
    fixed = broken = 0
    for result in results:
        was = _failing(getattr(result, before))
        now = _failing(getattr(result, after))
        fixed += len(was - now)
        broken += len(now - was)
    return fixed, broken


def run_benchmark(
    dataset: Dataset, *, postmt: Any, config: Any, skip_pipeline: bool = False
) -> Result:
    outcome = stub_pipeline(dataset) if skip_pipeline else run_pipeline(
        postmt, dataset, batch_size=config.benchmark.batch_size
    )
    processed = outcome.segments

    originals = dataset.segments
    strata = [stratum_of(p) for p in dataset.parameters_per_segment()]
    src_texts = [s.get("source_content") or "" for s in originals]
    ref_texts = [s.get("reference_content") or "" for s in originals]
    mt_texts = [s.get("target_content") or "" for s in originals]
    ape_texts = [extract_post_edited(s) for s in processed]

    per_segment_tags = [[tag.text for tag in extract_tags(text)] for text in src_texts]
    carrying = sum(1 for tags in per_segment_tags if tags)
    logger.info("[TAGS] %d/%d segments carry at least one tag", carrying, len(dataset.segments))
    if carrying == 0:
        logger.warning(
            "[TAGS] no tags at all - the segments reached the benchmark stripped of their "
            "markup, so a 100% result would only mean there was nothing to preserve"
        )

    results: list[SegmentResult] = []
    for index, segment in enumerate(processed):
        src_text = src_texts[index]

        results.append(SegmentResult(
            source_segment_id=segment_id(segment, originals[index], index),
            stratum=strata[index],
            src_text=src_text,
            mt_text=mt_texts[index],
            ape_text=ape_texts[index],
            ref_text=ref_texts[index],
            changed_by_ape=mt_texts[index] != ape_texts[index],
            tags=per_segment_tags[index],
            mt=score_tags(src_text=src_text, text=mt_texts[index]),
            ape=score_tags(src_text=src_text, text=ape_texts[index]),
            ref=score_tags(src_text=src_text, text=ref_texts[index]),
        ))

    failures = outcome.failures

    mt = aggregate([r.mt for r in results])
    ape = aggregate([r.ape for r in results])
    ref = aggregate([r.ref for r in results])

    return Result(
        dataset=f"{dataset.name} (dry-run)" if skip_pipeline else dataset.name,
        parameters=report_parameters(dataset),
        usage=outcome.usage,
        failed_segments=len(failures),
        failure_reason=failures[0] if failures else None,
        totals={
            "segments": len(dataset.segments),
            "segments_with_tags": carrying,
            "segments_changed_by_ape": sum(1 for r in results if r.changed_by_ape),
        },
        mt=mt,
        ape=ape,
        ref=ref,
        delta=Delta(delta(mt.integrity_rate, ape.integrity_rate), *_repairs(results, "mt", "ape")),
        segments=results,
    )
