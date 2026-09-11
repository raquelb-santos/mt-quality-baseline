"""The tag integrity metric: the source is the answer key, so every version is scored against SRC."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .report import rate
from .tags import Tag, extract_tags, unpaired


@dataclass
class ErrorBreakdown:
    """Exhaustive over the tag instances that went wrong, so the buckets pool by addition."""

    dropped: int = 0       # the source carried it and the version has fewer
    duplicated: int = 0    # the version has more of it than the source did
    hallucinated: int = 0  # a token the source never carried at all

    @property
    def total(self) -> int:
        return self.dropped + self.duplicated + self.hallucinated

    def add(self, other: "ErrorBreakdown") -> None:
        self.dropped += other.dropped
        self.duplicated += other.duplicated
        self.hallucinated += other.hallucinated


@dataclass
class TagBreakdown:
    """Each source tag bucketed against the source's count; exclusive and exhaustive."""

    never_carried: int = 0    # none in the version while the source had it
    carried_partly: int = 0   # fewer than SRC
    matched_source: int = 0   # as many as SRC
    over_carried: int = 0     # more than SRC

    @property
    def distinct_tags(self) -> int:
        return self.never_carried + self.carried_partly + self.matched_source + self.over_carried

    def add(self, other: "TagBreakdown") -> None:
        self.never_carried += other.never_carried
        self.carried_partly += other.carried_partly
        self.matched_source += other.matched_source
        self.over_carried += other.over_carried


@dataclass(frozen=True)
class TagScore:
    """The SRC/version count pair every rate is built from, recorded even when fully preserved."""

    text: str
    kind: str
    expected: int  # occurrences in SRC
    present: int   # occurrences in this version, bounded by SRC's
    found: int     # occurrences in this version, unbounded

    @property
    def dropped(self) -> int:
        return self.expected - self.present

    @property
    def added(self) -> int:
        return max(0, self.found - self.expected)


@dataclass
class Score:
    expected: int = 0
    present: int = 0
    errors: ErrorBreakdown = field(default_factory=ErrorBreakdown)
    tags: TagBreakdown = field(default_factory=TagBreakdown)
    # The surviving tags stand in a different order than the source put them in.
    mis_ordered: bool = False
    # Ids this version broke; ids the source already had broken, which no version is blamed for.
    unpaired: int = 0
    source_unpaired: int = 0
    tag_scores: list[TagScore] = field(default_factory=list)

    @property
    def dropped(self) -> int:
        return self.expected - self.present

    @property
    def defects(self) -> int:
        """Every way the segment is wrong, so that zero means it re-imports as the source did."""
        return self.errors.total + self.unpaired + int(self.mis_ordered)


def _surviving(tags: Sequence[Tag], budget: Counter[str]) -> list[str]:
    """One side's token order trimmed to what both sides carry, so order is compared on survivors."""
    remaining = Counter(budget)
    kept = []
    for tag in tags:
        if remaining[tag.text] > 0:
            remaining[tag.text] -= 1
            kept.append(tag.text)
    return kept


def score_tags(*, src_text: str, text: str) -> Score:
    """SRC is the contract: the version owes the same tokens, as often, in the same order."""
    result = Score()

    src_tags = extract_tags(src_text)
    out_tags = extract_tags(text)
    src_counts = Counter(tag.text for tag in src_tags)
    out_counts = Counter(tag.text for tag in out_tags)
    # SRC names the kind where both carry the token, so a hallucination cannot relabel it.
    kinds = {tag.text: tag.kind for tag in (*out_tags, *src_tags)}

    # Merged, not `|`: Counter union takes the max count, this keeps SRC order then OUT-only.
    for token in {**src_counts, **out_counts}:
        expected, found = src_counts[token], out_counts[token]
        present = min(found, expected)

        result.expected += expected
        result.present += present

        if expected == 0:
            result.errors.hallucinated += found
        else:
            result.errors.dropped += expected - present
            result.errors.duplicated += max(0, found - expected)

            # Classified on the counts, not presence: a hit cannot tell "as often" from "more".
            if found == 0:
                result.tags.never_carried += 1
            elif found < expected:
                result.tags.carried_partly += 1
            elif found == expected:
                result.tags.matched_source += 1
            else:
                result.tags.over_carried += 1

        result.tag_scores.append(TagScore(
            text=token, kind=kinds[token], expected=expected, present=present, found=found,
        ))

    result.mis_ordered = _surviving(src_tags, out_counts) != _surviving(out_tags, src_counts)

    src_broken = unpaired(src_tags)
    result.source_unpaired = len(src_broken)
    result.unpaired = len(unpaired(out_tags) - src_broken)

    return result


@dataclass
class Aggregate:
    """Integrity is the only share, and it is a share of what the source carried."""

    expected: int = 0
    present: int = 0
    errors: ErrorBreakdown = field(default_factory=ErrorBreakdown)
    tags: TagBreakdown = field(default_factory=TagBreakdown)
    integrity_rate: float | None = None
    segments_scored: int = 0
    # Segments with no defect of any kind, reported alongside the instance rate.
    segments_clean: int = 0
    segment_integrity_rate: float | None = None
    segments_mis_ordered: int = 0
    segments_unpaired: int = 0
    segments_source_unpaired: int = 0


def _with_rates(total: Aggregate) -> Aggregate:
    total.integrity_rate = rate(total.present, total.expected)
    total.segment_integrity_rate = rate(total.segments_clean, total.segments_scored)
    return total


def aggregate(scores: Sequence[Score]) -> Aggregate:
    total = Aggregate()

    for score in scores:
        # A segment counts once either side carries a tag, so an invented tag cannot hide in a
        # source that carried none.
        if not score.tag_scores:
            continue

        total.expected += score.expected
        total.present += score.present
        total.errors.add(score.errors)
        total.tags.add(score.tags)

        total.segments_scored += 1
        total.segments_mis_ordered += int(score.mis_ordered)
        total.segments_unpaired += int(bool(score.unpaired))
        total.segments_source_unpaired += int(bool(score.source_unpaired))
        if score.defects == 0:
            total.segments_clean += 1

    return _with_rates(total)


def pool(aggregates: Sequence[Aggregate]) -> Aggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    total = Aggregate()

    for agg in aggregates:
        total.expected += agg.expected
        total.present += agg.present
        total.errors.add(agg.errors)
        total.tags.add(agg.tags)
        total.segments_scored += agg.segments_scored
        total.segments_clean += agg.segments_clean
        total.segments_mis_ordered += agg.segments_mis_ordered
        total.segments_unpaired += agg.segments_unpaired
        total.segments_source_unpaired += agg.segments_source_unpaired

    return _with_rates(total)
