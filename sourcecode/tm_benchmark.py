"""TM orchestration: the gold set says which entries were on offer and carries the raw MT, post-mt
edits that MT into APE, and REF and APE are each scored against those entries."""

import logging
from dataclasses import dataclass, field
from typing import Any

from . import tm_match, tm_score
from .config import Config
from .pipeline import run_pipeline
from .postmt import Usage, extract_post_edited, preflight_submission, raise_for_preflight
from .text_processing import Dataset
from .tm_gold import GoldRow, GoldSet, Submission, load_gold_set
from .tm_index import Entry, FetchReport, TmIndexClient

logger = logging.getLogger(__name__)


@dataclass
class TmResult:
    """One gold set, scored."""

    dataset: str
    parameters: dict[str, Any]
    index: str
    floors: dict[str, float]
    totals: dict[str, int]
    aggregate: tm_score.TmAggregate
    calibration: tm_score.Calibration = field(default_factory=tm_score.Calibration)
    usage: Usage = field(default_factory=Usage)
    failed_segments: int = 0
    failure_reason: str | None = None
    # False where no candidate carried an embedding score: the semantic band was not looked for.
    semantic_measured: bool = False
    scored_versions: tuple[str, ...] = tm_score.VERSIONS
    segments: list[tm_score.SegmentScore] = field(default_factory=list)


def load_dataset(path: Any, *, dry_run: bool) -> GoldSet:
    """The gold set, and - unless this is a dry run - a check that post-mt would accept it."""
    gold = load_gold_set(path)

    if not dry_run:
        for submission in gold.submissions():
            raise_for_preflight(
                preflight_submission(submission.parameters),
                f'Preflight failed for {submission.name}: post-mt would reject every segment, so '
                f'there would be no output to score. Fix the parameters above.',
            )

    return gold


def as_dataset(gold: GoldSet, submission: Submission) -> Dataset:
    """One submission in the shape the shared post-mt driver takes."""
    return Dataset(
        name=f'{gold.name} · {submission.name}',
        parameters=submission.parameters,
        glossary_ids=[],
        segments=[
            {
                'source_segment_id': row.query_id,
                'source_content': row.source,
                'target_content': row.raw_mt,
                'reference_content': row.reference,
            }
            for row in submission.rows
        ],
        component='tm',
        steps=submission.steps,
    )


def _pairs(gold: GoldSet) -> dict[tuple[str, str], list[str]]:
    """Entry ids to fetch, grouped by the language pair they have to come back in."""
    grouped: dict[tuple[str, str], list[str]] = {}
    for row in gold.rows:
        wanted = grouped.setdefault(row.pair, [])
        for entry_id in (*row.candidate_ids, *row.hard_negatives):
            if entry_id not in wanted:
                wanted.append(entry_id)
    return grouped


class TmBenchmark:
    def __init__(
        self,
        *,
        postmt: Any,
        index: TmIndexClient,
        config: Config,
        embedder: tm_match.Embedder | None = None,
    ) -> None:
        self.postmt = postmt
        self.index = index
        self.config = config
        # Without one the semantic band cannot be reached and the match test stops at characters.
        self.embedder = embedder

    def semantic_scores(self, row: GoldRow, entries: dict[str, Entry]) -> dict[str, float] | None:
        """Source-to-source similarity per listed entry, which is what bands a candidate semantic."""
        if self.embedder is None:
            return None

        listed = [entry_id for entry_id in row.candidate_ids if entry_id in entries]
        if not listed:
            return {}

        query, *rest = self.embedder([row.source, *(entries[i].source for i in listed)])
        return {
            entry_id: tm_match.cosine(query, vector) for entry_id, vector in zip(listed, rest)
        }

    def fetch(self, gold: GoldSet) -> tuple[dict[tuple[str, str], dict[str, Entry]], FetchReport]:
        """Every listed entry and hard negative, one pass per pair. Results stay keyed by pair so
        an id accepted for one pair cannot reach the rows of a pair that rejected its language."""
        by_pair: dict[tuple[str, str], dict[str, Entry]] = {}
        report = FetchReport()

        for (source_language, target_language), ids in _pairs(gold).items():
            found, pair_report = self.index.fetch_entries_by_id(
                ids, source_language=source_language, target_language=target_language,
            )
            by_pair[(source_language, target_language)] = found
            report.add(pair_report)

        return by_pair, report

    def post_edit(self, gold: GoldSet) -> tuple[dict[str, str], Usage, list[str]]:
        """APE per query id. Submissions are grouped by pair and project, as post-mt requires."""
        edited: dict[str, str] = {}
        usage = Usage()
        failures: list[str] = []

        for submission in gold.submissions():
            outcome = run_pipeline(
                self.postmt, as_dataset(gold, submission),
                batch_size=self.config.benchmark.batch_size,
            )
            usage = usage + outcome.usage
            failures.extend(outcome.failures)
            for row, segment in zip(submission.rows, outcome.segments):
                edited[row.query_id] = extract_post_edited(segment)

        return edited, usage, failures

    def run(self, gold: GoldSet, *, skip_pipeline: bool = False) -> TmResult:
        settings = self.config.tm

        by_pair, fetch = self.fetch(gold)

        edited: dict[str, str] = {}
        usage, failures = Usage(), []
        if not skip_pipeline:
            edited, usage, failures = self.post_edit(gold)

        scored = []
        for row in gold.rows:
            entries = by_pair.get(row.pair, {})
            texts = {tm_score.REF: row.reference}
            if not skip_pipeline:
                texts[tm_score.APE] = edited.get(row.query_id, '')
            scored.append(tm_score.score_segment(
                row, entries, texts,
                fuzzy_floor=settings.fuzzy_floor,
                semantic_floor=settings.semantic_floor,
                reference_floor=settings.reference_floor,
                reference_semantic_floor=settings.reference_semantic_floor,
                partial_floor=settings.partial_floor,
                length_guard=settings.length_guard,
                min_relevance=settings.min_relevance,
                embedder=self.embedder,
                semantic_scores=self.semantic_scores(row, entries),
                hard_negatives=settings.hard_negatives,
            ))

        aggregate = tm_score.aggregate(scored)
        calibration = tm_score.Calibration()
        for segment in scored:
            for evidence in segment.calibration:
                calibration.record(evidence)

        logger.info(
            '[TM] %d/%d segments eligible, %d banded, %d unbanded',
            aggregate.declared, aggregate.sampled, aggregate.banded, aggregate.unbanded,
        )
        if aggregate.below_relevance:
            logger.warning(
                '[TM] %d/%d segments listed only entries graded under TM_MIN_RELEVANCE (%d), so '
                'they are excluded from every rate. That is the threshold, not the index.',
                aggregate.below_relevance, aggregate.fetched, settings.min_relevance,
            )
        if calibration.tested:
            logger.info(
                '[TM] hard negatives: %d/%d false matches at a reference floor of %.2f',
                calibration.matched, calibration.tested, settings.reference_floor,
            )

        blank = sum(
            1 for segment in scored
            if tm_score.APE in segment.versions and not segment.versions[tm_score.APE].evidence.score
            and not edited.get(segment.query_id, '').strip()
        )
        if blank and not skip_pipeline:
            logger.error(
                '[TM] %d/%d segments came back with no APE text. `raw_mt` went out as the MT '
                'column, so check that `steps` names APE and that post-mt returned it.',
                blank, len(scored),
            )

        first = gold.rows[0]
        return TmResult(
            dataset=f'{gold.name} (dry-run)' if skip_pipeline else gold.name,
            parameters={
                'source_language': first.source_language,
                'target_language': first.target_language,
                'domain': first.domain,
            },
            index=self.index.index,
            floors={
                'fuzzy': settings.fuzzy_floor,
                'semantic': settings.semantic_floor,
                'reference': settings.reference_floor,
                'partial': settings.partial_floor,
            },
            totals={'segments': aggregate.sampled, 'entries_fetched': fetch.found},
            aggregate=aggregate,
            calibration=calibration,
            usage=usage,
            failed_segments=len(failures),
            failure_reason=failures[0] if failures else None,
            semantic_measured=any(
                candidate.semantic is not None
                for segment in scored for candidate in segment.candidates
            ),
            scored_versions=(tm_score.REF,) if skip_pipeline else tm_score.VERSIONS,
            segments=scored,
        )
