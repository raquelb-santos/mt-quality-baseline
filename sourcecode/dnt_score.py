"""The DNT preservation metric: scored only where the source and reference both carry the item."""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .text_processing import count_surface
from .report import rate


CASE_DRIFT = "case_drift"
TRANSLATED = "translated"


@dataclass
class LeakBreakdown:
    """Exhaustive over the leaked instances, so the buckets sum to `leaked` and pool by addition."""

    case_drift: int = 0   # came through, but cased differently — the deterministic repair
    translated: int = 0   # the item is not in the version in any casing

    @property
    def total(self) -> int:
        return self.case_drift + self.translated

    def add(self, other: LeakBreakdown) -> None:
        self.case_drift += other.case_drift
        self.translated += other.translated


@dataclass
class ItemBreakdown:
    """Each item bucketed against the reference's count; exclusive and exhaustive."""

    never_kept: int = 0     # none kept while the reference kept it
    kept_partly: int = 0    # fewer than REF — kept in one place, translated in another
    matched_ref: int = 0    # as many as REF, which kept it
    over_kept: int = 0      # more than REF — kept more often than the human kept it

    @property
    def distinct_items(self) -> int:
        return self.never_kept + self.kept_partly + self.matched_ref + self.over_kept

    def add(self, other: ItemBreakdown) -> None:
        self.never_kept += other.never_kept
        self.kept_partly += other.kept_partly
        self.matched_ref += other.matched_ref
        self.over_kept += other.over_kept


@dataclass(frozen=True)
class ItemScore:
    """The REF/version count pair every rate is built from, recorded even when fully preserved."""

    text: str
    expected: int   # occurrences kept verbatim in REF
    preserved: int  # occurrences kept verbatim in this version, bounded by REF's
    kept: int       # occurrences kept verbatim in this version, unbounded
    in_src: int     # occurrences in SRC; the detector's own claim
    leak_kind: str = TRANSLATED

    @property
    def leaked(self) -> int:
        return self.expected - self.preserved

    @property
    def over_kept(self) -> int:
        return max(0, self.kept - self.expected)


@dataclass
class DntScore:
    expected: int = 0
    preserved: int = 0
    # Counted beside the rate, never inside it: the cap already discarded these from `preserved`.
    over_kept: int = 0
    leaks: LeakBreakdown = field(default_factory=LeakBreakdown)
    items: ItemBreakdown = field(default_factory=ItemBreakdown)
    # Named by the detector but absent from SRC; in SRC but not kept by REF.
    not_in_src: int = 0
    not_in_ref: int = 0
    item_scores: list[ItemScore] = field(default_factory=list)

    @property
    def leaked(self) -> int:
        return self.expected - self.preserved


def score_dnt(
    *,
    items: Iterable[str],
    text: str,
    src_text: str,
    ref_text: str,
    source_language_code: str | None,
    target_language_code: str | None,
) -> DntScore:
    """Two language codes: SRC is counted with the source, REF and the version with the target."""
    result = DntScore()

    # An item named twice for one segment is still one item.
    for item in dict.fromkeys(i for i in items if i):
        expected = count_surface(ref_text, item, target_language_code, casefold=False)
        kept = count_surface(text, item, target_language_code, casefold=False)
        in_src = count_surface(src_text, item, source_language_code, casefold=False)

        # Tested in this order: a string the source lacks cannot be put to the reference at all.
        if in_src == 0:
            result.not_in_src += 1
            continue
        if expected == 0:
            result.not_in_ref += 1
            continue

        preserved = min(kept, expected)

        result.expected += expected
        result.preserved += preserved
        result.over_kept += max(0, kept - expected)

        leaked = expected - preserved
        leak_kind = TRANSLATED
        if leaked:
            # The casefolding twin of the same count: what it finds beyond it is the drift.
            loose = count_surface(text, item, target_language_code)
            drift = max(0, min(loose, expected) - preserved)
            leak_kind = CASE_DRIFT if drift == leaked else TRANSLATED
            result.leaks.case_drift += drift
            result.leaks.translated += leaked - drift

        # Classified on the counts, not presence: a hit cannot tell "as often" from "more often".
        if kept == 0:
            result.items.never_kept += 1
        elif kept < expected:
            result.items.kept_partly += 1
        elif kept == expected:
            result.items.matched_ref += 1
        else:
            result.items.over_kept += 1

        result.item_scores.append(ItemScore(
            text=item, expected=expected, preserved=preserved, kept=kept,
            in_src=in_src, leak_kind=leak_kind,
        ))

    return result


@dataclass
class DntAggregate:
    """Preservation is the only share, and it is a share of what the reference kept."""

    expected: int = 0
    preserved: int = 0
    leaked: int = 0
    over_kept: int = 0
    leaks: LeakBreakdown = field(default_factory=LeakBreakdown)
    preservation_rate: float | None = None
    items: ItemBreakdown = field(default_factory=ItemBreakdown)
    not_in_src: int = 0
    not_in_ref: int = 0
    distinct_items: int = 0
    segments_with_items: int = 0
    # Segments with no error in either direction, reported alongside the instance rate.
    segments_fully_preserved: int = 0
    segment_preservation_rate: float | None = None
    # Segments the service never reported on, so they are outside every count above.
    segments_unread: int = 0

    @property
    def errors(self) -> int:
        """Both directions, never netted: they are different failures with different fixes."""
        return self.leaked + self.over_kept


def _with_rates(total: DntAggregate) -> DntAggregate:
    total.leaked = total.expected - total.preserved
    total.preservation_rate = rate(total.preserved, total.expected)
    total.segment_preservation_rate = rate(
        total.segments_fully_preserved, total.segments_with_items
    )
    return total


def aggregate(scores: Sequence[DntScore], *, segments_unread: int = 0) -> DntAggregate:
    total = DntAggregate(segments_unread=segments_unread)

    for score in scores:
        total.not_in_src += score.not_in_src
        total.not_in_ref += score.not_in_ref
        total.distinct_items += len(score.item_scores)

        if not score.item_scores:
            continue

        total.expected += score.expected
        total.preserved += score.preserved
        total.over_kept += score.over_kept
        total.leaks.add(score.leaks)
        total.items.add(score.items)

        # Every scored item is kept at least once in REF, so a segment with any has a denominator.
        total.segments_with_items += 1
        if score.preserved == score.expected and score.over_kept == 0:
            total.segments_fully_preserved += 1

    return _with_rates(total)


def pool(aggregates: Sequence[DntAggregate]) -> DntAggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    total = DntAggregate()

    for agg in aggregates:
        total.expected += agg.expected
        total.preserved += agg.preserved
        total.over_kept += agg.over_kept
        total.leaks.add(agg.leaks)
        total.items.add(agg.items)
        total.not_in_src += agg.not_in_src
        total.not_in_ref += agg.not_in_ref
        total.distinct_items += agg.distinct_items
        total.segments_with_items += agg.segments_with_items
        total.segments_fully_preserved += agg.segments_fully_preserved
        total.segments_unread += agg.segments_unread

    return _with_rates(total)
