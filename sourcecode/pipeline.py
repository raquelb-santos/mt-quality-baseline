"""Driving post-mt over a dataset's segments, shared by every quality component."""

import logging
from dataclasses import dataclass, field
from typing import Any

from .text_processing import Dataset
from .postmt import Usage, segment_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOutcome:
    segments: list[dict[str, Any]]
    usage: Usage = field(default_factory=Usage)
    # One entry per failed segment; their text is the raw MT echoed back, so it looks untouched.
    failures: list[str] = field(default_factory=list)


def run_pipeline(postmt: Any, dataset: Dataset, *, batch_size: int) -> PipelineOutcome:
    segments = dataset.segments
    batches = [segments[i : i + batch_size] for i in range(0, len(segments), batch_size)]
    logger.info("[PIPELINE] %d segments in %d batch(es)", len(segments), len(batches))

    processed: list[dict[str, Any]] = []
    usage = Usage()

    for number, batch in enumerate(batches, start=1):

        def on_progress(body: dict[str, Any], _n: int = number) -> None:
            percent = (body.get("progress") or {}).get("percent", 0)
            logger.info("[PIPELINE] batch %d/%d - %s%%", _n, len(batches), percent)

        result = postmt.run(
            parameters=dataset.parameters,
            # The human reference is the answer key: stripped so it can never reach post-mt.
            segments=[
                {k: v for k, v in segment.items() if k != "reference_content"}
                for segment in batch
            ],
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

        usage = usage + result.usage

        for i, original in enumerate(batch):
            returned = result.segments[i] if i < len(result.segments) else None
            processed.append(returned if returned is not None else original)

    failures = [error for s in processed if (error := segment_error(s))]
    if failures:
        logger.error(
            "[PIPELINE] %d/%d segments failed inside post-mt - first: %s",
            len(failures), len(processed), failures[0],
        )

    return PipelineOutcome(segments=processed, usage=usage, failures=failures)


def stub_pipeline(dataset: Dataset) -> PipelineOutcome:
    """The dry-run stand-in: the post-edited column mirrors the MT baseline, so every delta is zero."""
    return PipelineOutcome(
        segments=[
            {**segment, "ape_results": {"text": segment.get("target_content", "")}}
            for segment in dataset.segments
        ]
    )
