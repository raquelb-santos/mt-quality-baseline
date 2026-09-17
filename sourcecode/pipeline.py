"""Driving post-mt over a dataset's segments, shared by every component."""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .text_processing import Dataset, load
from .postmt import Usage, preflight_submission, preflight_tasks, raise_for_preflight, segment_error

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOutcome:
    segments: list[dict[str, Any]]
    usage: Usage = field(default_factory=Usage)
    # One entry per failed segment; their text is the raw MT echoed back, so it looks untouched.
    failures: list[str] = field(default_factory=list)


def load_dataset(path: Path, *, component: str, dry_run: bool, languages: Sequence[tuple[str, str]] = ()) -> Dataset:
    """Only the submission preflight applies: these components read what a segment carries."""
    data = load(path, component=component, languages=languages)
    if not dry_run:
        raise_for_preflight(
            preflight_tasks(data.tasks, preflight_submission),
            "Preflight failed: post-mt would reject every segment, so there would be no "
            "APE column to score. Fix the parameters above.",
        )
    return data


def run_pipeline(postmt: Any, dataset: Dataset, *, batch_size: int) -> PipelineOutcome:
    # A batch never spans two tasks: post-mt takes one parameter set per submission.
    batches = [
        (task, task.segments[i : i + batch_size])
        for task in dataset.tasks
        for i in range(0, len(task.segments), batch_size)
    ]
    logger.info(
        "[PIPELINE] %d segments in %d batch(es) across %d task(s)",
        len(dataset.segments), len(batches), len(dataset.tasks),
    )

    processed: list[dict[str, Any]] = []
    usage = Usage()

    for number, (task, batch) in enumerate(batches, start=1):
        result = postmt.run(
            parameters=task.parameters,
            # REF is the answer key, dropped so it can never reach post-mt.
            segments=[{k: v for k, v in segment.items() if k != "reference_content"} for segment in batch],
            steps=dataset.steps,
            on_progress=lambda body, n=number: logger.info(
                "[PIPELINE] batch %d/%d - %s%%", n, len(batches), (body.get("progress") or {}).get("percent", 0)
            ),
        )

        if result.error:
            logger.warning("[PIPELINE] batch %d reported: %s", number, result.error)

        if len(result.segments) != len(batch):
            logger.warning(
                "[PIPELINE] batch %d returned %d segments for %d inputs - realigning by index",
                number, len(result.segments), len(batch),
            )

        usage += result.usage
        for i, original in enumerate(batch):
            returned = result.segments[i] if i < len(result.segments) else None
            processed.append(returned if returned is not None else original)

    failures = [error for s in processed if (error := segment_error(s))]
    if failures:
        logger.error(
            "[PIPELINE] %d/%d segments failed inside post-mt: %s",
            len(failures), len(processed), failures[0],
        )

    return PipelineOutcome(segments=processed, usage=usage, failures=failures)


def stub_pipeline(dataset: Dataset) -> PipelineOutcome:
    """The dry-run stand-in: the APE column mirrors MT, so every delta is zero."""
    return PipelineOutcome(
        segments=[
            {**segment, "ape_results": {"text": segment["target_content"]}}
            for segment in dataset.segments
        ]
    )
