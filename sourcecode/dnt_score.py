"""The DNT preservation metric: scored only where the source and reference both carry the item."""

from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .text_processing import bounded_pattern, is_unspaced_language, normalize_text
from .report import rate


CASE_DRIFT = "case_drift"
TRANSLATED = "translated"


def count_item(text: object, item: object, language_code: str | None, *, casefold: bool = False) -> int:
    """Case-SENSITIVE, unlike `match`: different casing did not come through. No lemma fallback."""
    haystack = normalize_text(text, casefold=casefold)
    needle = normalize_text(item, casefold=casefold)
    if not haystack or not needle:
        return 0
    if is_unspaced_language(language_code):
        return haystack.count(needle)
    return len(bounded_pattern(needle).findall(haystack))


@dataclass
class LeakBreakdown:
    """Exhaustive over the leaked instances, so the buckets sum to `leaked` and pool by addition."""

    case_drift: int = 0   # came through, but cased differently — the deterministic repair
    translated: int = 0   # the item is not in the version in any casing

    @property
    def total(self) -> int:
        return self.case_drift + self.translated

    def record(self, kind: str, count: int) -> None:
        if kind == CASE_DRIFT:
            self.case_drift += count
        else:
            self.translated += count

    def add(self, other: LeakBreakdown) -> None:
        self.case_drift += other.case_drift
        self.translated += other.translated


@dataclass
class ItemBreakdown:
    """Each item bucketed against the reference's count; exclusive and exhaustive."""

    never_kept: int = 0         # none kept while the reference kept it
    kept_partly: int = 0        # fewer than REF — kept in one place, translated in another
    matched_ref: int = 0  # as many as REF, which kept it
    over_kept: int = 0          # more than REF — kept more often than the human kept it

    @property
    def distinct_items(self) -> int:
        return self.never_kept + self.kept_partly + self.matched_ref + self.over_kept

    def add(self, other: ItemBreakdown) -> None:
        self.never_kept += other.never_kept
        self.kept_partly += other.kept_partly
        self.matched_ref += other.matched_ref
        self.over_kept += other.over_kept


@dataclass
class DntTally:
    expected: int = 0
    preserved: int = 0
    over_kept: int = 0
    leaks: LeakBreakdown = field(default_factory=LeakBreakdown)

    @property
    def leaked(self) -> int:
        return self.expected - self.preserved


@dataclass(frozen=True)
class ItemScore:
    """The REF/version count pair every rate is built from, recorded even when fully preserved."""

    text: str
    expected: int      # occurrences kept verbatim in REF
    preserved: int     # occurrences kept verbatim in this version, bounded by REF's
    kept: int          # occurrences kept verbatim in this version, unbounded
    in_src: int     # occurrences in SRC; the detector's own claim
    # Meaningless without a shortfall.
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
    # Items the detector named that the source does not carry verbatim — a detector error.
    not_in_src: int = 0
    # In the source but not kept by the reference: no expectation to measure against, so flagged.
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
    """Two language codes, not one: SRC is counted with the source, REF and the version with the target."""
    result = DntScore()

    # An item named twice for one segment is still one item.
    for item in dict.fromkeys(i for i in items if i):
        expected = count_item(ref_text, item, target_language_code)
        kept = count_item(text, item, target_language_code)
        in_src = count_item(src_text, item, source_language_code)

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
            loose = count_item(text, item, target_language_code, casefold=True)
            drift = max(0, min(loose, expected) - preserved)
            leak_kind = CASE_DRIFT if drift == leaked else TRANSLATED
            result.leaks.record(CASE_DRIFT, drift)
            result.leaks.record(TRANSLATED, leaked - drift)

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
class DntTallyReport:
    """Preservation is the only share, and it is a share of what the reference kept."""

    expected: int
    preserved: int
    leaked: int
    over_kept: int
    leaks: LeakBreakdown
    preservation_rate: float | None

    @property
    def errors(self) -> int:
        """Both directions, never netted: they are different failures with different fixes."""
        return self.leaked + self.over_kept


@dataclass
class DntAggregate(DntTallyReport):
    items: ItemBreakdown
    not_in_src: int
    not_in_ref: int
    distinct_items: int
    segments_with_items: int
    # Segments with no error in either direction, reported alongside the instance rate.
    segments_fully_preserved: int
    segment_preservation_rate: float | None
    # Segments the service never reported on, so they are outside every count above.
    segments_unread: int = 0


def _build(
    tally: DntTally,
    items: ItemBreakdown,
    *,
    not_in_src: int,
    not_in_ref: int,
    distinct_items: int,
    segments_with_items: int,
    segments_fully_preserved: int,
    segments_unread: int,
) -> DntAggregate:
    return DntAggregate(
        expected=tally.expected,
        preserved=tally.preserved,
        leaked=tally.leaked,
        over_kept=tally.over_kept,
        leaks=tally.leaks,
        preservation_rate=rate(tally.preserved, tally.expected),
        items=items,
        not_in_src=not_in_src,
        not_in_ref=not_in_ref,
        distinct_items=distinct_items,
        segments_with_items=segments_with_items,
        segments_fully_preserved=segments_fully_preserved,
        segment_preservation_rate=rate(segments_fully_preserved, segments_with_items),
        segments_unread=segments_unread,
    )


def aggregate(scores: Sequence[DntScore], *, segments_unread: int = 0) -> DntAggregate:
    total = DntTally()
    items = ItemBreakdown()
    not_in_src = not_in_ref = distinct_items = 0
    segments_with_items = segments_fully_preserved = 0

    for score in scores:
        if score is None:
            continue

        not_in_src += score.not_in_src
        not_in_ref += score.not_in_ref
        distinct_items += len(score.item_scores)

        if not score.item_scores:
            continue

        items.add(score.items)
        total.expected += score.expected
        total.preserved += score.preserved
        total.over_kept += score.over_kept
        total.leaks.add(score.leaks)

        # Every scored item is kept at least once in REF, so a segment with any has a denominator.
        segments_with_items += 1
        if score.preserved == score.expected and score.over_kept == 0:
            segments_fully_preserved += 1

    return _build(
        total, items,
        not_in_src=not_in_src, not_in_ref=not_in_ref,
        distinct_items=distinct_items, segments_with_items=segments_with_items,
        segments_fully_preserved=segments_fully_preserved, segments_unread=segments_unread,
    )


def pool(aggregates: Sequence[DntAggregate]) -> DntAggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    total = DntTally()
    items = ItemBreakdown()
    not_in_src = not_in_ref = distinct_items = 0
    segments_with_items = segments_fully_preserved = segments_unread = 0

    for item in aggregates:
        if item is None:
            continue
        total.expected += item.expected
        total.preserved += item.preserved
        total.over_kept += item.over_kept
        total.leaks.add(item.leaks)
        items.add(item.items)
        not_in_src += item.not_in_src
        not_in_ref += item.not_in_ref
        distinct_items += item.distinct_items
        segments_with_items += item.segments_with_items
        segments_fully_preserved += item.segments_fully_preserved
        segments_unread += item.segments_unread

    return _build(
        total, items,
        not_in_src=not_in_src, not_in_ref=not_in_ref,
        distinct_items=distinct_items, segments_with_items=segments_with_items,
        segments_fully_preserved=segments_fully_preserved, segments_unread=segments_unread,
    )
