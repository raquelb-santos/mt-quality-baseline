"""Scoring one TM segment and pooling the counts. The band comes from the strongest entry the gold
set listed; the verdict comes from whichever entry the version used, which is not always the same."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import tm_match
from .report import rate
from .tm_gold import GoldRow
from .tm_index import Entry

REF, APE = 'REF', 'APE'
VERSIONS = (REF, APE)


@dataclass(frozen=True)
class Candidate:
    """One listed entry, scored against SRC."""

    entry: Entry
    grade: int | None
    char: float
    semantic: float | None = None
    band: str | None = None
    # Banded on the window of SRC it covers
    partial: bool = False
    covered: float | None = None

    @property
    def entry_id(self) -> str:
        return self.entry.entry_id

    @property
    def score(self) -> float:
        """The score the band was decided on."""
        return self.char if self.covered is None else self.covered


@dataclass(frozen=True)
class Use:
    """One entry a version used, and what it did with it."""

    candidate: Candidate
    evidence: tm_match.MatchEvidence
    length_flagged: bool = False

    @property
    def band(self) -> str:
        return self.candidate.band or tm_match.UNBANDED

    @property
    def verdict(self) -> str:
        return self.evidence.verdict

    @property
    def compliant(self) -> bool:
        return tm_match.is_compliant(self.band, self.verdict)

    @property
    def violation(self) -> str | None:
        return tm_match.violation(self.band, self.verdict)


@dataclass(frozen=True)
class VersionScore:
    """What one version did with the entries on offer."""

    evidence: tm_match.MatchEvidence
    # Every entry this version used, strongest first: a segment can be served by more than one.
    uses: tuple[Use, ...] = ()
    violation: str | None = None
    matched_lesser: bool = False
    length_flagged: bool = False

    @property
    def verdict(self) -> str:
        return self.evidence.verdict

    @property
    def conclusive(self) -> bool:
        return self.evidence.conclusive

    @property
    def used_entry(self) -> bool:
        return bool(self.uses)

    @property
    def stitched(self) -> bool:
        return len(self.uses) > 1

    @property
    def whole_match(self) -> bool:
        """An entry's claim covered the whole text, so there was no room for a second one."""
        return bool(self.uses) and self.uses[0].evidence.span is None


@dataclass(frozen=True)
class SegmentScore:
    """One gold-set row, scored. Ineligible and unusable rows are kept: they carry the exclusions."""

    query_id: str
    domain: str = ''
    declared: bool = False
    # The index delivered at least one listed entry.
    fetched: bool = False
    usable: bool = False
    band: str | None = None
    best: Candidate | None = None
    candidates: tuple[Candidate, ...] = ()
    versions: dict[str, VersionScore] = field(default_factory=dict)
    # Version text against each hard negative's target - every match here is a false positive.
    calibration: tuple[tm_match.MatchEvidence, ...] = ()
    # REF matched none of the listed entries: a flag on the gold set, never a filter.
    ref_matched_none: bool = False

    @property
    def eligible(self) -> bool:
        """Listed and fetched: the row the per-band rates are computed over."""
        return self.declared and self.usable


def length_flagged(left: object, right: object, *, guard: float) -> bool:
    """A short segment can clear the floor on wording it never took from the entry."""
    first, second = len(str(left or '')), len(str(right or ''))
    longest = max(first, second)
    return bool(longest) and abs(first - second) / longest > guard


def score_candidates(
    row: GoldRow,
    entries: dict[str, Entry],
    *,
    fuzzy_floor: float,
    semantic_floor: float,
    partial_floor: float,
    min_relevance: int = 0,
    semantic_scores: dict[str, float] | None = None,
) -> list[Candidate]:
    """Every listed entry we could fetch, scored and banded. Order follows the gold set's grades."""
    scored: list[Candidate] = []

    for entry_id in row.candidate_ids:
        entry = entries.get(entry_id)
        grade = row.grade(entry_id)
        if entry is None or (grade is not None and grade < min_relevance):
            continue

        char = tm_match.char_score(row.source, entry.source)
        semantic = (semantic_scores or {}).get(entry_id)
        floors = {'fuzzy_floor': fuzzy_floor, 'semantic_floor': semantic_floor}
        band = tm_match.classify(char=char, semantic=semantic, **floors)

        # An entry answering a fragment of SRC reaches no floor against the whole of it, so it is
        # banded on the part it covers, where a 100% match is exact and applied like any other.
        covered = None
        if band is None:
            window = tm_match.contained_score(row.source, entry.source)
            if window >= partial_floor:
                covered = window
                band = tm_match.classify(char=window, semantic=None, **floors)

        scored.append(Candidate(
            entry=entry,
            grade=grade,
            char=char,
            semantic=semantic,
            band=band,
            partial=covered is not None,
            covered=covered,
        ))

    return scored


def _best(candidates: Sequence[Candidate]) -> Candidate | None:
    """Best band first, then best score - the opportunity the TM gave."""
    banded = [candidate for candidate in candidates if candidate.band in tm_match.BANDS]
    if not banded:
        return None
    return max(banded, key=lambda c: (-tm_match.BANDS.index(c.band), c.score))


@dataclass(frozen=True)
class _Claim:
    """One entry's claim on a span of the version's text, all of it for a whole-text match and only
    itself for a window, settled strongest first so the two tests need no special case."""

    candidate: Candidate
    evidence: tm_match.MatchEvidence
    span: tuple[int, int]
    weight: float


def _claims(
    text: str,
    candidates: Sequence[Candidate],
    *,
    partial_floor: float,
    **floors: object,
) -> tuple[list[_Claim], tm_match.MatchEvidence]:
    """Every claim the listed entries make, with the strongest non-match evidence saying whether a
    `not_used` was conclusive. Each candidate is tried whole and windowed, and a claim weighs the
    characters it covers, capped at the entry's length so a short one cannot outweigh a real one."""
    length = tm_match.normalized_length(text)
    claims: list[_Claim] = []
    missed = tm_match.NO_MATCH

    for candidate in candidates:
        target = candidate.entry.target

        whole = tm_match.match_evidence(text, target, **floors)  # type: ignore[arg-type]
        if whole.matched:
            covered = min(tm_match.normalized_length(target), length)
            claims.append(_Claim(candidate, whole, (0, length), whole.score * covered))
        elif whole.score > missed.score:
            missed = whole

        window = tm_match.contained_evidence(text, target, floor=partial_floor)
        if window.matched and window.span is not None:
            start, end = window.span
            claims.append(_Claim(candidate, window, window.span, window.score * (end - start)))

    return claims, missed


def _settle(claims: Sequence[_Claim], *, text: str, length_guard: float) -> list[Use]:
    """The entries the version was built from, claimed strongest first. A claim overlapping one
    already settled is dropped, so two entries are never given the same words and an entry that
    matched both ways is credited once. Ties keep the gold set's own order: the sort is stable and
    the claims arrive in it."""
    uses: list[Use] = []
    taken: list[tuple[int, int]] = []

    for claim in sorted(claims, key=lambda item: -item.weight):
        start, end = claim.span
        if any(start < over and under < end for under, over in taken):
            continue
        taken.append(claim.span)
        # A window is the length of the entry's target by construction, so it can never trip the
        # length guard; only a whole-text use can.
        flagged = claim.evidence.span is None and length_flagged(
            text, claim.candidate.entry.target, guard=length_guard)
        uses.append(Use(claim.candidate, claim.evidence, flagged))

    return uses


def _uses(
    text: str,
    candidates: Sequence[Candidate],
    *,
    partial_floor: float,
    length_guard: float,
    **floors: object,
) -> tuple[list[Use], tm_match.MatchEvidence]:
    """Every entry this version used, strongest first, and the evidence the segment is judged on.

    A whole-text claim covers all of the text, so settling it leaves no room for a second entry and
    the segment has exactly one use. A claim that covers only a window leaves the rest of the text
    free, and another entry may answer it - not as an exception to the whole-text pass, but because
    that is what its span says."""
    claims, missed = _claims(text, candidates, partial_floor=partial_floor, **floors)
    uses = _settle(claims, text=text, length_guard=length_guard)
    return uses, uses[0].evidence if uses else missed


def score_version(
    text: str,
    candidates: Sequence[Candidate],
    best: Candidate | None,
    *,
    reference_floor: float,
    semantic_floor: float,
    partial_floor: float,
    length_guard: float,
    embedder: tm_match.Embedder | None = None,
) -> VersionScore:
    uses, evidence = _uses(
        text, candidates,
        partial_floor=partial_floor, length_guard=length_guard,
        reference_floor=reference_floor, semantic_floor=semantic_floor, embedder=embedder,
    )

    used = uses[0].candidate if uses else None

    # Named across every entry it used, not just the strongest one.
    violation = next((use.violation for use in uses if use.violation), None) if uses else 'not_used'

    return VersionScore(
        evidence=evidence,
        uses=tuple(uses),
        violation=violation,
        matched_lesser=used is not None and best is not None and used.entry_id != best.entry_id,
        length_flagged=any(use.length_flagged for use in uses),
    )


def score_segment(
    row: GoldRow,
    entries: dict[str, Entry],
    texts: dict[str, str],
    *,
    fuzzy_floor: float,
    semantic_floor: float,
    reference_floor: float,
    reference_semantic_floor: float,
    partial_floor: float,
    length_guard: float,
    min_relevance: int = 0,
    embedder: tm_match.Embedder | None = None,
    semantic_scores: dict[str, float] | None = None,
    hard_negatives: bool = True,
) -> SegmentScore:
    """One row, start to finish. `texts` carries the versions to score, keyed REF and APE."""
    candidates = score_candidates(
        row, entries,
        fuzzy_floor=fuzzy_floor, semantic_floor=semantic_floor, partial_floor=partial_floor,
        min_relevance=min_relevance, semantic_scores=semantic_scores,
    )
    # What the index could deliver, before `min_relevance` had a say.
    fetched = any(entry_id in entries for entry_id in row.candidate_ids)
    usable = bool(candidates)
    best = _best(candidates)

    floors = {
        'reference_floor': reference_floor,
        'semantic_floor': reference_semantic_floor,
        'partial_floor': partial_floor,
        'length_guard': length_guard,
        'embedder': embedder,
    }
    versions = {
        version: score_version(texts.get(version, ''), candidates, best, **floors)
        for version in VERSIONS if version in texts
    }

    calibration: list[tm_match.MatchEvidence] = []
    if hard_negatives:
        for entry_id in row.hard_negatives:
            negative = entries.get(entry_id)
            if negative is None:
                continue
            # Tested like the uses, containment included, or the rate calibrates nothing.
            for version, version_score in versions.items():
                text = texts.get(version, '')
                evidence = tm_match.match_evidence(
                    text, negative.target,
                    reference_floor=reference_floor,
                    semantic_floor=reference_semantic_floor,
                    embedder=embedder,
                )
                if not evidence.matched and not version_score.whole_match:
                    contained = tm_match.contained_evidence(
                        text, negative.target, floor=partial_floor,
                    )
                    evidence = contained if contained.matched else evidence
                calibration.append(evidence)

    ref = versions.get(REF)
    return SegmentScore(
        query_id=row.query_id,
        domain=row.domain,
        declared=row.declared,
        fetched=fetched,
        usable=usable,
        band=(best.band if best is not None else tm_match.UNBANDED) if usable else None,
        best=best,
        candidates=tuple(candidates),
        versions=versions,
        calibration=tuple(calibration),
        ref_matched_none=bool(usable and ref is not None and not ref.used_entry),
    )


@dataclass
class TierTally:
    tested: int = 0
    matched: int = 0

    @property
    def false_match_rate(self) -> float | None:
        return rate(self.matched, self.tested)


@dataclass
class Calibration:
    """False positives over the hard negatives, whole and split by the tier that produced them."""

    tested: int = 0
    matched: int = 0
    by_tier: dict[str, TierTally] = field(default_factory=dict)

    @property
    def false_match_rate(self) -> float | None:
        return rate(self.matched, self.tested)

    def record(self, evidence: tm_match.MatchEvidence) -> None:
        self.tested += 1
        self.matched += int(evidence.matched)
        tier = self.by_tier.setdefault(evidence.tier, TierTally())
        tier.tested += 1
        tier.matched += int(evidence.matched)


@dataclass
class BandTally:
    """One band's counts in two units. Segment counts are filed under the band the TM offered, so
    an ignored exact match stays on the exact row, and entry counts under the band of the entry a
    version used, so a rate over them is that band's pass rate. Rates are always recomputed."""

    # The opportunity the TM gave, and the denominator of every segment count below.
    eligible: int = 0
    used: Counter = field(default_factory=Counter)
    # Entries used: a segment served by several contributes several.
    entries: Counter = field(default_factory=Counter)
    # Segments where the version used more than one entry.
    stitched: Counter = field(default_factory=Counter)
    compliant: Counter = field(default_factory=Counter)
    # Offered here, but the match test could not be finished - see MatchEvidence.conclusive.
    inconclusive: Counter = field(default_factory=Counter)
    verdicts: dict[str, Counter] = field(default_factory=lambda: {v: Counter() for v in VERSIONS})
    violations: dict[str, Counter] = field(default_factory=lambda: {v: Counter() for v in VERSIONS})
    matched_lesser: Counter = field(default_factory=Counter)
    grades: Counter = field(default_factory=Counter)
    # REF against APE, on the segments where REF used an entry.
    both: int = 0
    ref_only: int = 0
    ape_only: int = 0

    def reference_rate(self, version: str) -> float | None:
        """The TM offered this band - how often was it used at all."""
        return rate(self.used[version], self.eligible)

    def compliance_rate(self, version: str) -> float | None:
        """Per entry used, not per segment and not over the offers: declining a match is a
        translator's choice and costs the reference rate instead, and a segment stitched from
        several entries is right about the ones it got right."""
        return rate(self.compliant[version], self.entries[version])

    @property
    def carry_over_rate(self) -> float | None:
        return rate(self.both, self.both + self.ref_only)

    def add(self, other: 'BandTally') -> None:
        self.eligible += other.eligible
        self.both += other.both
        self.ref_only += other.ref_only
        self.ape_only += other.ape_only
        for name in ('used', 'entries', 'stitched', 'compliant', 'inconclusive',
                     'matched_lesser', 'grades'):
            getattr(self, name).update(getattr(other, name))
        for name in ('verdicts', 'violations'):
            for version, counts in getattr(other, name).items():
                getattr(self, name)[version].update(counts)


@dataclass
class TmAggregate:
    """The exclusion counts and the per-band counts, poolable across datasets in a stratum."""

    sampled: int = 0
    declared: int = 0
    fetched: int = 0
    usable: int = 0
    ref_matched_none: int = 0
    bands: dict[str, BandTally] = field(
        default_factory=lambda: {band: BandTally() for band in tm_match.REPORTED_BANDS}
    )
    # Eligible and used segments by the grade of their best entry, since a grade is not an order.
    grade_eligible: Counter = field(default_factory=Counter)
    grade_used: dict[str, Counter] = field(
        default_factory=lambda: {version: Counter() for version in VERSIONS}
    )

    @property
    def not_fetched(self) -> int:
        """Listed, but nothing came back from the index - a data fault, excluded from the rates."""
        return self.declared - self.fetched

    @property
    def below_relevance(self) -> int:
        """Fetched, but every entry graded under `min_relevance` - a threshold, not a fault."""
        return self.fetched - self.usable

    @property
    def banded(self) -> int:
        return sum(self.bands[band].eligible for band in tm_match.BANDS)

    @property
    def unbanded(self) -> int:
        return self.bands[tm_match.UNBANDED].eligible

    @property
    def eligibility_rate(self) -> float | None:
        """The gold set's judgement: it listed an entry, so the TM had something to offer."""
        return rate(self.declared, self.sampled)

    def reference_rate(self, version: str) -> float | None:
        used = sum(tally.used[version] for tally in self.bands.values())
        return rate(used, self.usable)

    def grade_rates(self, version: str) -> list[tuple[int, float | None, int]]:
        """Reference rate per relevance grade, most relevant first."""
        return [
            (grade, rate(self.grade_used[version][grade], eligible), eligible)
            for grade, eligible in sorted(self.grade_eligible.items(), reverse=True)
        ]

    def stitched(self, version: str) -> int:
        """Segments the version served out of more than one entry."""
        return sum(tally.stitched[version] for tally in self.bands.values())

    def inconclusive(self, version: str) -> int:
        """`not_used` verdicts resting on characters alone - counted in the rates, named beside
        them, never silently dropped."""
        return sum(tally.inconclusive[version] for tally in self.bands.values())

    @property
    def carry_over_rate(self) -> float | None:
        both = sum(tally.both for tally in self.bands.values())
        ref_only = sum(tally.ref_only for tally in self.bands.values())
        return rate(both, both + ref_only)

    def add(self, other: 'TmAggregate') -> None:
        self.sampled += other.sampled
        self.declared += other.declared
        self.fetched += other.fetched
        self.usable += other.usable
        self.ref_matched_none += other.ref_matched_none
        self.grade_eligible.update(other.grade_eligible)
        for version, counts in other.grade_used.items():
            self.grade_used[version].update(counts)
        for band, tally in other.bands.items():
            self.bands[band].add(tally)


def aggregate(scores: Sequence[SegmentScore]) -> TmAggregate:
    result = TmAggregate(sampled=len(scores))

    for score in scores:
        result.declared += int(score.declared)
        result.fetched += int(score.declared and score.fetched)
        result.usable += int(score.eligible)
        result.ref_matched_none += int(score.ref_matched_none)

        if not score.eligible or score.band is None:
            continue

        # The opportunity band: what the TM had to offer, whatever the versions then did with it.
        tally = result.bands[score.band]
        tally.eligible += 1
        # Taken from the candidates, not `best`, so a segment banding nothing still has a grade.
        grade = max((c.grade for c in score.candidates if c.grade is not None), default=None)
        if grade is not None:
            tally.grades[grade] += 1
            result.grade_eligible[grade] += 1

        for version, version_score in score.versions.items():
            if version_score.used_entry:
                tally.used[version] += 1
                if grade is not None:
                    result.grade_used[version][grade] += 1
            if version_score.stitched:
                tally.stitched[version] += 1
            if not version_score.conclusive:
                tally.inconclusive[version] += 1
            if version_score.matched_lesser:
                tally.matched_lesser[version] += 1

            # Each use under the band of the entry it took, which is that rule's own denominator.
            for use in version_score.uses:
                use_tally = result.bands[use.band]
                use_tally.entries[version] += 1
                use_tally.verdicts[version][use.verdict] += 1
                if use.compliant:
                    use_tally.compliant[version] += 1
                if use.violation:
                    use_tally.violations[version][use.violation] += 1

        ref, ape = score.versions.get(REF), score.versions.get(APE)
        if ref is not None and ape is not None:
            if ref.used_entry and ape.used_entry:
                tally.both += 1
            elif ref.used_entry:
                tally.ref_only += 1
            elif ape.used_entry:
                tally.ape_only += 1

    return result


def pool(aggregates: Iterable[TmAggregate]) -> TmAggregate:
    """Rates recomputed from pooled counts; averaging would weight 3 segments like 300."""
    total = TmAggregate()
    for item in aggregates:
        total.add(item)
    return total
