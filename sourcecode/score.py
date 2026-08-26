from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .match import count_occurrences, normalize_text, term_present


def build_glossary_map(mappings: Iterable[Mapping[str, str]] | None) -> dict[str, set[str]]:
    glossary_map: dict[str, set[str]] = {}
    for mapping in mappings or []:
        source = mapping.get("source_content")
        target = mapping.get("target_content")
        if not source or not target:
            continue
        glossary_map.setdefault(source, set()).add(target)
    return glossary_map


#: The two kinds a missed instance can be, named once so the value written into a CSV row and the
#: field it was counted into cannot drift apart.
SUBSTITUTED = "substituted"
OMITTED = "omitted"


@dataclass
class MissBreakdown:
    """Why an expected instance was not delivered, split by what the translation did instead.

    The buckets are mutually exclusive and exhaustive over the missed instances, so they sum to
    `Tally.violations` and pool by addition rather than by recomputing anything. Splitting them is
    what stops the violation count being a restatement of the shortfall: `substituted` is a
    terminology failure — the concept was rendered, but not with the glossary's target term —
    whereas `omitted` is the absence of any translation to judge, which is not a terminology
    failure at all and so is counted beside the violations rather than among them.
    """

    substituted: int = 0   # rendered, but not with any target term the glossary gives
    omitted: int = 0       # no text produced for the segment, so nothing to render it in

    @property
    def total(self) -> int:
        return self.substituted + self.omitted

    def record(self, kind: str, count: int) -> None:
        if kind == OMITTED:
            self.omitted += count
        else:
            self.substituted += count

    def add(self, other: MissBreakdown) -> None:
        """Pooling is plain addition — unlike a rate, nothing needs recomputing from totals."""
        self.substituted += other.substituted
        self.omitted += other.omitted


@dataclass
class Tally:
    expected: int = 0
    adherent: int = 0
    misses: MissBreakdown = field(default_factory=MissBreakdown)

    @property
    def violations(self) -> int:
        return self.expected - self.adherent


@dataclass
class TermBreakdown:
    """Where each distinct glossary term falls, comparing its target count to the reference's.

    The four buckets are mutually exclusive and exhaustive, so they sum to the number of distinct
    terms scored for the text. Counting terms rather than occurrences makes this the lenient
    reading of the same evidence the adherence rate is built from: `present` is everything except
    `never_used`, and the gap between `used_everywhere` and `used_partly` is inconsistency inside
    segments. `over_used` is recorded but never scored — a term the human never used at all lands
    there, as does a source pronoun rendered as a repeated noun or a compound whose constituents
    also matched.
    """

    never_used: int = 0        # no target term anywhere in the text
    used_partly: int = 0       # rendered, but fewer times than the reference renders the term
    used_everywhere: int = 0   # rendered exactly as often as the reference renders the term
    over_used: int = 0         # rendered more often than the reference renders the term

    @property
    def distinct_terms(self) -> int:
        return self.never_used + self.used_partly + self.used_everywhere + self.over_used

    @property
    def present(self) -> int:
        return self.distinct_terms - self.never_used

    def add(self, other: TermBreakdown) -> None:
        """Pooling is plain addition — unlike a rate, nothing needs recomputing from totals."""
        self.never_used += other.never_used
        self.used_partly += other.used_partly
        self.used_everywhere += other.used_everywhere
        self.over_used += other.over_used


@dataclass(frozen=True)
class Violation:
    source_content: str
    expected_targets: list[str]
    strictness: str
    #: Fewer missed than expected means the term was rendered somewhere but not everywhere.
    missed_occurrences: int = 1
    expected_occurrences: int = 1
    #: What the translation did instead: `SUBSTITUTED` or `OMITTED`.
    miss_kind: str = SUBSTITUTED


@dataclass(frozen=True)
class TermScore:
    """One glossary term inside one segment: the `R`/`T` pair the adherence rate is built from.

    Recorded for every scored term, not only the violating ones — a fully adherent term produces
    no `Violation`, so without this its denominator would be lost and no per-term rate could be
    computed. `rendered` keeps the raw target count that `adherent` discards when the cap applies.
    """

    source_content: str
    expected_targets: list[str]
    strictness: str
    expected: int   # R — occurrences in the human reference
    adherent: int   # occurrences found in this translation, bounded by R
    rendered: int   # T — occurrences found in this translation, unbounded
    #: What this translation did instead, where it fell short. Meaningless without a shortfall,
    #: so read it only when `missed` is non-zero.
    miss_kind: str = SUBSTITUTED

    @property
    def missed(self) -> int:
        return self.expected - self.adherent

    @property
    def violations(self) -> int:
        """Substitutions only, as everywhere else: an omission is not a terminology failure."""
        return self.missed if self.miss_kind != OMITTED else 0

    @property
    def omissions(self) -> int:
        return self.missed if self.miss_kind == OMITTED else 0


@dataclass(frozen=True)
class Hit:
    source_content: str
    matched_target: str
    via: str
    strictness: str


@dataclass
class TranslationScore:
    expected: int = 0
    adherent: int = 0
    strict: Tally = field(default_factory=Tally)
    permissive: Tally = field(default_factory=Tally)
    terms: TermBreakdown = field(default_factory=TermBreakdown)
    #: Why the missed instances were missed; sums to `expected - adherent`.
    misses: MissBreakdown = field(default_factory=MissBreakdown)
    #: Per-term counts, in match order — the grain the per-term and per-segment rates roll up from.
    term_scores: list[TermScore] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)


def score_translation(
    *,
    mappings: Iterable[Mapping[str, str]] | None,
    text: str,
    language_code: str | None,
    reference_text: str,
    text_lemmas: str | None = None,
    reference_lemmas: str | None = None,
    term_lemmas: Mapping[str, str] | None = None,
) -> TranslationScore:
    """Score one translation of a segment against the human reference for that same segment.

    Both counts are the same measurement taken twice — how often the target term for that source
    term appears — once over the reference, which sets what the translation owed, and once over
    `text`, which is what it delivered. `reference_text` carries no default: it is what the
    metric is defined against, so a caller cannot omit it and still get a number back.
    """
    glossary_map = build_glossary_map(mappings)
    result = TranslationScore()

    # `text` does not change across the loop, so neither does what a shortfall in it means. With
    # no text at all there is no wording to hold against the glossary and every miss is an
    # omission; otherwise the segment rendered the concept in wording the glossary did not
    # target term, which is the failure the violation count exists to name.
    miss_kind = OMITTED if not normalize_text(text) else SUBSTITUTED

    for source, targets in glossary_map.items():

        expected_n = _count_renderings(
            reference_text, targets, language_code, reference_lemmas, term_lemmas
        )
        rendered = _count_renderings(text, targets, language_code, text_lemmas, term_lemmas)

        # Neither the human nor this translation used the term: retrieval proposed it, the
        # reference declined it, and there is no evidence here to score either way.
        if expected_n == 0 and rendered == 0:
            continue

        is_strict = len(targets) == 1
        slice_ = result.strict if is_strict else result.permissive
        strictness = "strict" if is_strict else "permissive"

        # Capped at the reference count so a translation cannot outscore its denominator.
        adherent_n = min(rendered, expected_n)

        result.expected += expected_n
        slice_.expected += expected_n
        result.adherent += adherent_n
        slice_.adherent += adherent_n

        missed = expected_n - adherent_n
        if missed:
            result.misses.record(miss_kind, missed)
            slice_.misses.record(miss_kind, missed)

        # Classified on the counts, not on `hit`: a hit cannot tell a term rendered exactly as
        # often as the reference renders it from one rendered more often, and the cap above has
        # already discarded that difference from `adherent_n`.
        if rendered == 0:
            result.terms.never_used += 1
        elif rendered < expected_n:
            result.terms.used_partly += 1
        elif rendered == expected_n:
            result.terms.used_everywhere += 1
        else:
            result.terms.over_used += 1

        result.term_scores.append(
            TermScore(
                source_content=source,
                expected_targets=sorted(targets),
                strictness=strictness,
                expected=expected_n,
                adherent=adherent_n,
                rendered=rendered,
                miss_kind=miss_kind,
            )
        )

        hit = _first_hit(source, targets, text, language_code, text_lemmas, term_lemmas,
                         strictness)
        if hit is not None:
            result.hits.append(hit)

        if adherent_n < expected_n:
            result.violations.append(
                Violation(
                    source_content=source,
                    expected_targets=sorted(targets),
                    strictness=strictness,
                    missed_occurrences=missed,
                    expected_occurrences=expected_n,
                    miss_kind=miss_kind,
                )
            )

    return result


def _count_renderings(
    text: str,
    targets: Iterable[str],
    language_code: str | None,
    text_lemmas: str | None,
    term_lemmas: Mapping[str, str] | None,
) -> int:
    """How often any target term for one source term appears in `text`.

    The reference and the translation are both counted with this, so `R` and `T` differ only in
    which text they were taken over, never in how.
    """
    return sum(
        count_occurrences(
            text=text, term=target, language_code=language_code,
            text_lemmas=text_lemmas, term_lemmas=(term_lemmas or {}).get(target),
        )
        for target in sorted(targets)
    )


def _first_hit(
    source: str,
    targets: Iterable[str],
    text: str,
    language_code: str | None,
    text_lemmas: str | None,
    term_lemmas: Mapping[str, str] | None,
    strictness: str,
) -> Hit | None:
    for target in sorted(targets):
        match = term_present(
            text=text,
            term=target,
            language_code=language_code,
            text_lemmas=text_lemmas,
            term_lemmas=(term_lemmas or {}).get(target),
        )
        if match.found:
            return Hit(source, target, match.via or "surface", strictness)
    return None


def rate(numerator: int, denominator: int) -> float | None:
    """None, never 0, with no denominator: a misconfigured run is not total failure."""
    return None if denominator == 0 else numerator / denominator


@dataclass
class TallyReport:
    """One rate and the counts behind it.

    Adherence is the only thing reported as a share, and it is a share of `expected` — what the
    human reference demanded. The misses stay counts: a rate over every missed instance would be
    `1 - adherence_rate` by construction and would say nothing new, while one over substitutions
    alone would move for two different reasons at once. A count says how many terms there are to go
    and fix, which is what the number is for, and it stays legible on a small denominator.
    """

    expected: int
    adherent: int
    #: Every missed instance, whatever the reason — `misses.total` by another name.
    violations: int
    #: Those same misses split by what the version did instead. Counts, never rates.
    misses: MissBreakdown
    adherence_rate: float | None


def with_rates(tally: Tally) -> TallyReport:
    return TallyReport(
        expected=tally.expected,
        adherent=tally.adherent,
        violations=tally.violations,
        misses=tally.misses,
        adherence_rate=rate(tally.adherent, tally.expected),
    )


@dataclass
class Aggregate(TallyReport):
    strict: TallyReport
    permissive: TallyReport
    #: Counts, not a rate: the lenient reading of the same terms the adherence rate scores.
    terms: TermBreakdown
    segments_with_glossary: int
    #: Segments with zero violations; reported alongside, never instead of, the instance rate.
    segments_fully_adherent: int
    segment_adherence_rate: float | None


def aggregate(scores: Sequence[TranslationScore]) -> Aggregate:
    total, strict, permissive = Tally(), Tally(), Tally()
    terms = TermBreakdown()
    segments_with_glossary = 0
    segments_fully_adherent = 0

    for score in scores:
        if score is None or not score.term_scores:
            continue
        # Pooled even when nothing was expected: a term the human avoided and this version used
        # is over-use, and skipping the segment outright would lose the only record of it.
        terms.add(score.terms)
        if score.expected == 0:
            continue
        segments_with_glossary += 1
        if score.adherent == score.expected:
            segments_fully_adherent += 1
        total.expected += score.expected
        total.adherent += score.adherent
        total.misses.add(score.misses)
        strict.expected += score.strict.expected
        strict.adherent += score.strict.adherent
        strict.misses.add(score.strict.misses)
        permissive.expected += score.permissive.expected
        permissive.adherent += score.permissive.adherent
        permissive.misses.add(score.permissive.misses)

    return Aggregate(
        expected=total.expected,
        adherent=total.adherent,
        violations=total.violations,
        misses=total.misses,
        adherence_rate=rate(total.adherent, total.expected),
        strict=with_rates(strict),
        permissive=with_rates(permissive),
        terms=terms,
        segments_with_glossary=segments_with_glossary,
        segments_fully_adherent=segments_fully_adherent,
        segment_adherence_rate=rate(segments_fully_adherent, segments_with_glossary),
    )


def pool(aggregates: Sequence[Aggregate]) -> Aggregate:
    """Combine already-aggregated results into one, for reporting a stratum.

    Counts are summed and every rate is recomputed from the pooled totals. Averaging the rates
    instead would weight a 3-instance dataset the same as a 300-instance one, so a tiny outlier
    could move a stratum's headline number more than the bulk of its evidence.
    """
    total, strict, permissive = Tally(), Tally(), Tally()
    terms = TermBreakdown()
    segments_with_glossary = 0
    segments_fully_adherent = 0

    for item in aggregates:
        if item is None:
            continue
        total.expected += item.expected
        total.adherent += item.adherent
        total.misses.add(item.misses)
        strict.expected += item.strict.expected
        strict.adherent += item.strict.adherent
        strict.misses.add(item.strict.misses)
        permissive.expected += item.permissive.expected
        permissive.adherent += item.permissive.adherent
        permissive.misses.add(item.permissive.misses)
        terms.add(item.terms)
        segments_with_glossary += item.segments_with_glossary
        segments_fully_adherent += item.segments_fully_adherent

    return Aggregate(
        expected=total.expected,
        adherent=total.adherent,
        violations=total.violations,
        misses=total.misses,
        adherence_rate=rate(total.adherent, total.expected),
        strict=with_rates(strict),
        permissive=with_rates(permissive),
        terms=terms,
        segments_with_glossary=segments_with_glossary,
        segments_fully_adherent=segments_fully_adherent,
        segment_adherence_rate=rate(segments_fully_adherent, segments_with_glossary),
    )
