from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from .text_processing import count_occurrences, normalize_text
from .report import rate


def build_glossary_map(mappings: Iterable[Mapping[str, str]] | None) -> dict[str, set[str]]:
    glossary_map: dict[str, set[str]] = {}
    for mapping in mappings or []:
        source = mapping.get("source_content")
        target = mapping.get("target_content")
        if not source or not target:
            continue
        glossary_map.setdefault(source, set()).add(target)
    return glossary_map


@dataclass
class Tally:
    expected: int = 0
    adherent: int = 0

    @property
    def violations(self) -> int:
        return self.expected - self.adherent

    def add(self, other: Tally | TallyReport | TranslationScore) -> None:
        self.expected += other.expected
        self.adherent += other.adherent


@dataclass
class TermBreakdown:
    """Each term bucketed against the reference's count; exclusive and exhaustive."""

    never_used: int = 0
    used_partly: int = 0
    used_everywhere: int = 0
    over_used: int = 0

    @property
    def distinct_terms(self) -> int:
        return self.never_used + self.used_partly + self.used_everywhere + self.over_used

    def add(self, other: TermBreakdown) -> None:
        self.never_used += other.never_used
        self.used_partly += other.used_partly
        self.used_everywhere += other.used_everywhere
        self.over_used += other.over_used


@dataclass(frozen=True)
class Violation:
    source_content: str
    expected_targets: list[str]
    strictness: str
    missed_occurrences: int = 1
    expected_occurrences: int = 1


@dataclass(frozen=True)
class TermScore:
    """Recorded even when fully adherent, or a clean term would lose its denominator."""

    source_content: str
    expected_targets: list[str]
    strictness: str
    expected: int   # occurrences in REF
    adherent: int   # occurrences found in this translation, bounded by REF's
    rendered: int   # occurrences found in this translation, unbounded

    @property
    def violations(self) -> int:
        return self.expected - self.adherent


@dataclass
class TranslationScore:
    expected: int = 0
    adherent: int = 0
    strict: Tally = field(default_factory=Tally)
    permissive: Tally = field(default_factory=Tally)
    terms: TermBreakdown = field(default_factory=TermBreakdown)
    term_scores: list[TermScore] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)


def score_translation(
    *,
    mappings: Iterable[Mapping[str, str]] | None,
    text: str,
    language_code: str | None,
    ref_text: str,
    text_lemmas: str | None = None,
    ref_lemmas: str | None = None,
    term_lemmas: Mapping[str, str] | None = None,
) -> TranslationScore:
    """Score one translation of a segment against the human reference for that same segment."""
    result = TranslationScore()

    for source, targets in build_glossary_map(mappings).items():
        expected_targets = sorted(targets)
        expected_n = _count_renderings(
            ref_text, expected_targets, language_code, ref_lemmas, term_lemmas
        )
        rendered = _count_renderings(
            text, expected_targets, language_code, text_lemmas, term_lemmas
        )
        # Neither the human nor this translation used the term: no denominator, nothing to score.
        if expected_n == 0 and rendered == 0:
            continue

        strictness = "strict" if len(expected_targets) == 1 else "permissive"
        # Capped at the reference count so a translation cannot outscore its denominator.
        adherent_n = min(rendered, expected_n)
        missed = expected_n - adherent_n

        for tally in (result, result.strict if strictness == "strict" else result.permissive):
            tally.expected += expected_n
            tally.adherent += adherent_n

        # Bucketed on the counts: presence alone cannot tell "as often as the human" from "more".
        if rendered == 0:
            result.terms.never_used += 1
        elif rendered < expected_n:
            result.terms.used_partly += 1
        elif rendered == expected_n:
            result.terms.used_everywhere += 1
        else:
            result.terms.over_used += 1

        result.term_scores.append(TermScore(
            source_content=source, expected_targets=expected_targets, strictness=strictness,
            expected=expected_n, adherent=adherent_n, rendered=rendered,
        ))
        if missed:
            result.violations.append(Violation(
                source_content=source, expected_targets=expected_targets, strictness=strictness,
                missed_occurrences=missed, expected_occurrences=expected_n,
            ))

    return result


def _count_renderings(
    text: str,
    targets: Sequence[str],
    language_code: str | None,
    text_lemmas: str | None,
    term_lemmas: Mapping[str, str] | None,
) -> int:
    """How often any target term for one source term appears in `text`."""
    return sum(
        count_occurrences(
            text=text, term=target, language_code=language_code,
            text_lemmas=text_lemmas, term_lemmas=(term_lemmas or {}).get(target),
        )
        for target in targets
    )


@dataclass
class TallyReport:
    expected: int
    adherent: int
    violations: int
    adherence_rate: float | None


def with_rates(tally: Tally) -> TallyReport:
    return TallyReport(
        expected=tally.expected,
        adherent=tally.adherent,
        violations=tally.violations,
        adherence_rate=rate(tally.adherent, tally.expected),
    )


@dataclass
class Aggregate(TallyReport):
    strict: TallyReport
    permissive: TallyReport
    terms: TermBreakdown
    segments_with_glossary: int
    segments_fully_adherent: int
    segment_adherence_rate: float | None


def _combine(
    items: Sequence[TranslationScore] | Sequence[Aggregate],
    scored: Sequence[TranslationScore] | Sequence[Aggregate],
    segments_with_glossary: int,
    segments_fully_adherent: int,
) -> Aggregate:
    """Term counts pool over `items`, the rate-bearing tallies over `scored`."""
    total, strict, permissive = Tally(), Tally(), Tally()
    terms = TermBreakdown()
    for item in items:
        terms.add(item.terms)
    for item in scored:
        total.add(item)
        strict.add(item.strict)
        permissive.add(item.permissive)

    return Aggregate(
        expected=total.expected,
        adherent=total.adherent,
        violations=total.violations,
        adherence_rate=rate(total.adherent, total.expected),
        strict=with_rates(strict),
        permissive=with_rates(permissive),
        terms=terms,
        segments_with_glossary=segments_with_glossary,
        segments_fully_adherent=segments_fully_adherent,
        segment_adherence_rate=rate(segments_fully_adherent, segments_with_glossary),
    )


def aggregate(scores: Sequence[TranslationScore]) -> Aggregate:
    matched = [s for s in scores if s is not None and s.term_scores]
    scored = [s for s in matched if s.expected]
    return _combine(
        matched,
        scored,
        segments_with_glossary=len(scored),
        segments_fully_adherent=sum(1 for s in scored if s.adherent == s.expected),
    )


def pool(aggregates: Sequence[Aggregate]) -> Aggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    items = [a for a in aggregates if a is not None]
    return _combine(
        items,
        items,
        segments_with_glossary=sum(a.segments_with_glossary for a in items),
        segments_fully_adherent=sum(a.segments_fully_adherent for a in items),
    )


# ── corpus-level violations ──────────────────────────────────────────────────────────────────
# Adherence asks what this version reproduced; this asks what it got wrong.

MISS = "miss"                              # no sanctioned target form in the output
INCONSISTENCY = "inconsistency"            # an approved target other than the one the reference used
OVER_APPLICATION = "over-application"      # a target term used where its source term was not


@dataclass(frozen=True)
class TermViolation:
    segment_index: int
    source_content: str
    kind: str
    # The rendering that triggered it - the variant used, or "" for a miss.
    detail: str = ""


@dataclass
class ViolationReport:
    """The rate is over every segment: one with no term retrieved can still over-apply one."""

    miss: int = 0
    inconsistency: int = 0
    over_application: int = 0
    total: int = 0
    segments: int = 0
    segments_with_violation: int = 0
    violation_rate: float | None = None
    items: list[TermViolation] = field(default_factory=list)


def _rendered_variants(
    text: str,
    targets: Sequence[str],
    normalized_text: str,
    normalized_lemmas: str,
    language_code: str | None,
    text_lemmas: str | None,
    term_lemmas: Mapping[str, str] | None,
) -> list[str]:
    """Collapsed so one rendering never reads as two: case and spacing variants, and substrings."""
    found: dict[str, str] = {}
    for target in targets:
        lemma = (term_lemmas or {}).get(target)
        # Necessary for either mode and far cheaper: every term is tested against every segment.
        if normalize_text(target) not in normalized_text and not (
            lemma and normalized_lemmas and normalize_text(lemma) in normalized_lemmas
        ):
            continue
        if count_occurrences(
            text=text, term=target, language_code=language_code,
            text_lemmas=text_lemmas, term_lemmas=lemma,
        ):
            found.setdefault(normalize_text(target), target)

    return [
        target for form, target in found.items()
        if not any(other != form and form in other for other in found)
    ]


def find_violations(
    *,
    texts: Sequence[str],
    ref_texts: Sequence[str],
    per_segment_mappings: Sequence[Iterable[Mapping[str, str]]],
    corpus_mappings: Iterable[Mapping[str, str]],
    language_code: str | None,
    text_lemmas: Mapping[str, str] | None = None,
    term_lemmas: Mapping[str, str] | None = None,
) -> ViolationReport:
    """Only where the reference rendered it: a declined proposal would score retrieval."""
    if not (len(texts) == len(ref_texts) == len(per_segment_mappings)):
        raise ValueError(
            f"{len(texts)} texts, {len(ref_texts)} references and "
            f"{len(per_segment_mappings)} segments of mappings must be the same length"
        )

    corpus_map = build_glossary_map(corpus_mappings)
    lookup = text_lemmas or {}
    report = ViolationReport(segments=len(texts))

    def variants_in(text: str) -> dict[str, list[str]]:
        lemmas = lookup.get(text)
        normalized_text = normalize_text(text)
        normalized_lemmas = normalize_text(lemmas)
        return {
            source: _rendered_variants(
                text, sorted(targets), normalized_text, normalized_lemmas, language_code,
                text_lemmas=lemmas, term_lemmas=term_lemmas,
            )
            for source, targets in corpus_map.items()
        }

    for index, text in enumerate(texts):
        mappings = list(per_segment_mappings[index])
        segment_map = build_glossary_map(mappings)
        # From the raw mappings, so a term retrieved with a blank target still counts as retrieved.
        retrieved = {m.get("source_content") for m in mappings if m.get("source_content")}

        found = variants_in(text)
        in_reference = variants_in(ref_texts[index])

        # Any wording the reference used, plus any a term retrieved here sanctions.
        licensed = {
            normalize_text(variant)
            for source, variants in found.items()
            if source in segment_map
            for variant in variants
        } | {
            normalize_text(variant)
            for variants in in_reference.values()
            for variant in variants
        }

        for source in segment_map:
            # The reference declined the term here, so there is nothing to hold this version to.
            if not in_reference[source]:
                continue
            used = found[source]
            if not used:
                report.items.append(TermViolation(index, source, MISS))
                continue
            # The reference settles the wording: a majority vote would let the version grade itself.
            intended = {normalize_text(variant) for variant in in_reference[source]}
            for variant in used:
                if normalize_text(variant) not in intended:
                    report.items.append(TermViolation(index, source, INCONSISTENCY, variant))

        for source, used in found.items():
            if source in retrieved:
                continue
            for variant in used:
                if normalize_text(variant) not in licensed:
                    report.items.append(TermViolation(index, source, OVER_APPLICATION, variant))

    report.items.sort(key=lambda item: (item.segment_index, str(item.source_content), item.kind))
    report.miss = sum(1 for item in report.items if item.kind == MISS)
    report.inconsistency = sum(1 for item in report.items if item.kind == INCONSISTENCY)
    report.over_application = sum(1 for item in report.items if item.kind == OVER_APPLICATION)
    report.total = len(report.items)
    report.segments_with_violation = len({item.segment_index for item in report.items})
    report.violation_rate = rate(report.segments_with_violation, report.segments)
    return report


def pool_violations(reports: Sequence[ViolationReport]) -> ViolationReport:
    """`items` is left empty: a segment index only means something inside its own dataset."""
    pooled = ViolationReport()
    for item in reports:
        if item is None:
            continue
        pooled.miss += item.miss
        pooled.inconsistency += item.inconsistency
        pooled.over_application += item.over_application
        pooled.total += item.total
        pooled.segments += item.segments
        pooled.segments_with_violation += item.segments_with_violation
    pooled.violation_rate = rate(pooled.segments_with_violation, pooled.segments)
    return pooled
