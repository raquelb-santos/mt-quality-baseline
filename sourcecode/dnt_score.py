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
    found: int      # occurrences kept verbatim in this version, unbounded
    in_src: int     # occurrences in SRC; the detector's own claim
    leak_kind: str = TRANSLATED

    @property
    def leaked(self) -> int:
        return self.expected - self.preserved

    @property
    def over_kept(self) -> int:
        return max(0, self.found - self.expected)


@dataclass
class Score:
    expected: int = 0
    preserved: int = 0
    # Counted beside the rate, never inside it: the cap already discarded these from `preserved`.
    over_kept: int = 0
    leaks: LeakBreakdown = field(default_factory=LeakBreakdown)
    items: ItemBreakdown = field(default_factory=ItemBreakdown)
    # What the detector found in SRC, and how much of it this version still carries.
    in_src: int = 0
    kept_from_src: int = 0
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
) -> Score:
    """Two language codes: SRC is counted with the source, REF and the version with the target."""
    result = Score()

    # An item named twice for one segment is still one item.
    for item in dict.fromkeys(i for i in items if i):
        expected = count_surface(ref_text, item, target_language_code, casefold=False)
        found = count_surface(text, item, target_language_code, casefold=False)
        in_src = count_surface(src_text, item, source_language_code, casefold=False)

        # Tested in this order: a string the source lacks cannot be put to the reference at all.
        if in_src == 0:
            result.not_in_src += 1
            continue

        result.in_src += in_src
        result.kept_from_src += min(found, in_src)

        if expected == 0:
            result.not_in_ref += 1
            continue

        preserved = min(found, expected)

        result.expected += expected
        result.preserved += preserved
        result.over_kept += found - preserved

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
        if found == 0:
            result.items.never_kept += 1
        elif found < expected:
            result.items.kept_partly += 1
        elif found == expected:
            result.items.matched_ref += 1
        else:
            result.items.over_kept += 1

        result.item_scores.append(ItemScore(
            text=item, expected=expected, preserved=preserved, found=found,
            in_src=in_src, leak_kind=leak_kind,
        ))

    return result


@dataclass
class Aggregate:
    """Preservation is the only share, and it is a share of what the reference kept."""

    expected: int = 0
    preserved: int = 0
    over_kept: int = 0
    leaks: LeakBreakdown = field(default_factory=LeakBreakdown)
    preservation_rate: float | None = None
    items: ItemBreakdown = field(default_factory=ItemBreakdown)
    in_src: int = 0
    kept_from_src: int = 0
    src_retention_rate: float | None = None
    not_in_src: int = 0
    not_in_ref: int = 0
    segments_scored: int = 0
    # Segments with no error in either direction, reported alongside the instance rate.
    segments_clean: int = 0
    segment_preservation_rate: float | None = None
    # Segments the service never reported on, so they are outside every count above.
    segments_unread: int = 0

    @property
    def leaked(self) -> int:
        return self.expected - self.preserved

    @property
    def errors(self) -> int:
        """Both directions, never netted: they are different failures with different fixes."""
        return self.leaked + self.over_kept


def _with_rates(total: Aggregate) -> Aggregate:
    total.preservation_rate = rate(total.preserved, total.expected)
    total.src_retention_rate = rate(total.kept_from_src, total.in_src)
    total.segment_preservation_rate = rate(total.segments_clean, total.segments_scored)
    return total


def aggregate(scores: Sequence[Score], *, segments_unread: int = 0) -> Aggregate:
    total = Aggregate(segments_unread=segments_unread)

    for score in scores:
        total.not_in_src += score.not_in_src
        total.not_in_ref += score.not_in_ref
        total.in_src += score.in_src
        total.kept_from_src += score.kept_from_src

        if not score.item_scores:
            continue

        total.expected += score.expected
        total.preserved += score.preserved
        total.over_kept += score.over_kept
        total.leaks.add(score.leaks)
        total.items.add(score.items)

        # Every scored item is kept at least once in REF, so a segment with any has a denominator.
        total.segments_scored += 1
        if score.preserved == score.expected and score.over_kept == 0:
            total.segments_clean += 1

    return _with_rates(total)


def pool(aggregates: Sequence[Aggregate]) -> Aggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    total = Aggregate()

    for agg in aggregates:
        total.expected += agg.expected
        total.preserved += agg.preserved
        total.over_kept += agg.over_kept
        total.leaks.add(agg.leaks)
        total.items.add(agg.items)
        total.not_in_src += agg.not_in_src
        total.not_in_ref += agg.not_in_ref
        total.in_src += agg.in_src
        total.kept_from_src += agg.kept_from_src
        total.segments_scored += agg.segments_scored
        total.segments_clean += agg.segments_clean
        total.segments_unread += agg.segments_unread

    return _with_rates(total)


@dataclass(frozen=True)
class TermOutcome:
    """One gold term, and whether each of the four versions carries it verbatim.

    Reversion is measured on two arms, because it is run over the raw MT and over the
    post-edited text so the two can be compared on the same terms.
    """

    text: str
    in_mt: bool
    in_ape: bool
    in_rev_mt: bool
    in_rev_ape: bool

    @property
    def repaired(self) -> bool:
        return self.in_rev_mt and not self.in_mt

    @property
    def broken(self) -> bool:
        return self.in_mt and not self.in_rev_mt

    @property
    def repaired_from_ape(self) -> bool:
        return self.in_rev_ape and not self.in_ape

    @property
    def broken_from_ape(self) -> bool:
        return self.in_ape and not self.in_rev_ape


@dataclass
class ReversionScore:
    terms: list[TermOutcome] = field(default_factory=list)

    @property
    def expected(self) -> int:
        return len(self.terms)

    @property
    def in_mt(self) -> int:
        return sum(1 for term in self.terms if term.in_mt)

    @property
    def in_ape(self) -> int:
        return sum(1 for term in self.terms if term.in_ape)

    @property
    def in_rev(self) -> int:
        return sum(1 for term in self.terms if term.in_rev_mt)

    @property
    def in_rev_ape(self) -> int:
        return sum(1 for term in self.terms if term.in_rev_ape)

    @property
    def repaired(self) -> int:
        return sum(1 for term in self.terms if term.repaired)

    @property
    def broken(self) -> int:
        return sum(1 for term in self.terms if term.broken)

    @property
    def repaired_from_ape(self) -> int:
        return sum(1 for term in self.terms if term.repaired_from_ape)

    @property
    def broken_from_ape(self) -> int:
        return sum(1 for term in self.terms if term.broken_from_ape)

    @property
    def clean(self) -> bool:
        """Every gold term carried once reversion had run over the MT."""
        return bool(self.terms) and self.in_rev == self.expected

    @property
    def clean_from_ape(self) -> bool:
        return bool(self.terms) and self.in_rev_ape == self.expected


def score_reversion(
    *,
    terms: Iterable[str],
    mt_text: str,
    ape_text: str,
    rev_text: str,
    rev_ape_text: str,
    target_language_code: str | None,
) -> ReversionScore:
    """A gold term counts as carried where it survives verbatim, casing included."""
    def carries(text: str, term: str) -> bool:
        return bool(count_surface(text, term, target_language_code, casefold=False))

    outcomes = [
        TermOutcome(
            term,
            carries(mt_text, term),
            carries(ape_text, term),
            carries(rev_text, term),
            carries(rev_ape_text, term),
        )
        for term in dict.fromkeys(term for term in terms if term and term.strip())
    ]
    return ReversionScore(outcomes)


@dataclass
class ReversionAggregate:
    expected: int = 0
    in_mt: int = 0
    in_ape: int = 0
    in_rev: int = 0
    in_rev_ape: int = 0
    repaired: int = 0
    broken: int = 0
    repaired_from_ape: int = 0
    broken_from_ape: int = 0
    segments_scored: int = 0
    segments_clean: int = 0
    segments_clean_from_ape: int = 0
    # Segments no revert batch came back for; excluded rather than scored as having lost everything.
    segments_unread: int = 0

    @property
    def mt_rate(self) -> float | None:
        return rate(self.in_mt, self.expected)

    @property
    def ape_rate(self) -> float | None:
        return rate(self.in_ape, self.expected)

    @property
    def rev_rate(self) -> float | None:
        return rate(self.in_rev, self.expected)

    @property
    def rev_ape_rate(self) -> float | None:
        return rate(self.in_rev_ape, self.expected)

    @property
    def segment_rate(self) -> float | None:
        return rate(self.segments_clean, self.segments_scored)

    @property
    def segment_rate_from_ape(self) -> float | None:
        return rate(self.segments_clean_from_ape, self.segments_scored)


def _combine_reversion(
    counted: Sequence[Any], *, segments_scored: int, segments_clean: int,
    segments_clean_from_ape: int, segments_unread: int
) -> ReversionAggregate:
    total = ReversionAggregate(
        segments_scored=segments_scored,
        segments_clean=segments_clean,
        segments_clean_from_ape=segments_clean_from_ape,
        segments_unread=segments_unread,
    )
    for item in counted:
        total.expected += item.expected
        total.in_mt += item.in_mt
        total.in_ape += item.in_ape
        total.in_rev += item.in_rev
        total.in_rev_ape += item.in_rev_ape
        total.repaired += item.repaired
        total.broken += item.broken
        total.repaired_from_ape += item.repaired_from_ape
        total.broken_from_ape += item.broken_from_ape
    return total


def aggregate_reversion(
    scores: Sequence[ReversionScore], *, segments_unread: int = 0
) -> ReversionAggregate:
    """A segment with no gold term sets no expectation, so it carries no denominator."""
    scored = [score for score in scores if score.expected]
    return _combine_reversion(
        scored,
        segments_scored=len(scored),
        segments_clean=sum(1 for score in scored if score.clean),
        segments_clean_from_ape=sum(1 for score in scored if score.clean_from_ape),
        segments_unread=segments_unread,
    )
