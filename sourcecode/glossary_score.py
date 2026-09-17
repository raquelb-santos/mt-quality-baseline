"""The terminology adherence metric: every version is scored against the human reference."""

from collections import Counter
from dataclasses import dataclass, field
from functools import cache
from typing import Any, Iterable, Mapping, Sequence

from .text_processing import count_occurrences, find_occurrences, normalize_text
from .report import rate

MISS = "miss"                              # no sanctioned target form in the output
INCONSISTENCY = "inconsistency"            # an approved target other than the one the reference used
OVER_APPLICATION = "over-application"      # a target term used where its source term was not


def bucket_of(found: int, expected: int) -> str:
    """The one ladder both grains classify on - the per-segment counters and the pooled row."""
    if found == 0:
        return "never_used" if expected else ""
    if found < expected:
        return "used_partly"
    return "matched_ref" if found == expected else "over_used"


def build_glossary_map(mappings: Iterable[Mapping[str, str]] | None) -> dict[str, set[str]]:
    glossary_map: dict[str, set[str]] = {}
    for mapping in mappings or []:
        source, target = mapping.get("source_content"), mapping.get("target_content")
        if source and target:
            glossary_map.setdefault(source, set()).add(target)
    return glossary_map


@dataclass
class Tally:
    expected: int = 0
    adherent: int = 0
    exact: int = 0

    @property
    def violations(self) -> int:
        return self.expected - self.adherent

    @property
    def adherence_rate(self) -> float | None:
        return rate(self.adherent, self.expected)

    @property
    def exact_rate(self) -> float | None:
        return rate(self.exact, self.expected)

    def add(self, other: Any) -> None:
        self.expected += other.expected
        self.adherent += other.adherent
        self.exact += other.exact


@dataclass
class TermBreakdown:
    """Each term bucketed against the reference's count; exclusive and exhaustive."""

    never_used: int = 0
    used_partly: int = 0
    matched_ref: int = 0
    over_used: int = 0

    @property
    def distinct_terms(self) -> int:
        return self.never_used + self.used_partly + self.matched_ref + self.over_used

    def add(self, other: TermBreakdown) -> None:
        self.never_used += other.never_used
        self.used_partly += other.used_partly
        self.matched_ref += other.matched_ref
        self.over_used += other.over_used


@dataclass(frozen=True)
class Violation:
    source_content: str
    kind: str
    expected_targets: list[str] = field(default_factory=list)
    strictness: str = ""
    missed_occurrences: int = 1
    expected_occurrences: int = 1
    # Corpus-level only: where it was found, and the rendering that triggered it.
    segment_index: int = -1
    detail: str = ""


@dataclass(frozen=True)
class TermScore:
    """Recorded even when fully adherent, or a clean term would lose its denominator."""

    source_content: str
    expected_targets: list[str]
    strictness: str
    expected: int   # occurrences in REF
    adherent: int   # occurrences found in this translation, bounded by REF's
    rendered: int   # occurrences found in this translation, unbounded
    exact: int      # adherent occurrences worded as REF words them

    @property
    def violations(self) -> int:
        return self.expected - self.adherent


@dataclass
class Score:
    expected: int = 0
    adherent: int = 0
    exact: int = 0
    strict: Tally = field(default_factory=Tally)
    permissive: Tally = field(default_factory=Tally)
    terms: TermBreakdown = field(default_factory=TermBreakdown)
    term_scores: list[TermScore] = field(default_factory=list)
    violations: list[Violation] = field(default_factory=list)


def _owned(
    text: str,
    targets: Iterable[str],
    language_code: str | None,
    text_lemmas: str | None,
    term_lemmas: Mapping[str, str] | None,
) -> dict[str, list[str | None]]:
    """Each target's occurrences as worded, outside a longer target covering it, so one wording counts once."""
    lemma_of = term_lemmas or {}
    found = {}
    for target in dict.fromkeys(targets):
        occurrences = find_occurrences(
            text=text, term=target, language_code=language_code,
            text_lemmas=text_lemmas, term_lemmas=lemma_of.get(target),
        )
        if occurrences:
            found[target] = occurrences

    # Longest first, so each target subtracts only what the longer ones kept for themselves.
    owned: dict[str, list[str | None]] = {}
    for target in sorted(found, key=lambda t: len(normalize_text(t)), reverse=True):
        form = normalize_text(target)
        covered = sum(
            len(kept) * count_occurrences(
                text=longer, term=target, language_code=language_code,
                text_lemmas=lemma_of.get(longer), term_lemmas=lemma_of.get(target),
            )
            for longer, kept in owned.items() if normalize_text(longer) != form
        )
        owned[target] = found[target][: max(len(found[target]) - covered, 0)]
    return owned


def score_glossary(
    *,
    mappings: Iterable[Mapping[str, str]] | None,
    text: str,
    language_code: str | None,
    ref_text: str,
    text_lemmas: str | None = None,
    ref_lemmas: str | None = None,
    term_lemmas: Mapping[str, str] | None = None,
) -> Score:
    """Score one translation of a segment against REF for that same segment."""
    result = Score()
    glossary_map = build_glossary_map(mappings)
    targets_here = [target for targets in glossary_map.values() for target in targets]
    in_ref = _owned(ref_text, targets_here, language_code, ref_lemmas, term_lemmas)
    in_text = _owned(text, targets_here, language_code, text_lemmas, term_lemmas)

    for source, targets in glossary_map.items():
        expected_targets = sorted(targets)
        ref_words = [w for target in expected_targets for w in in_ref.get(target, [])]
        text_words = [w for target in expected_targets for w in in_text.get(target, [])]
        expected_n, rendered = len(ref_words), len(text_words)
        # Neither the human nor this translation used the term: no denominator, nothing to score.
        if expected_n == 0 and rendered == 0:
            continue

        strictness = "strict" if len(expected_targets) == 1 else "permissive"
        # Capped at the reference count so a translation cannot outscore its denominator.
        adherent_n = min(rendered, expected_n)
        missed = expected_n - adherent_n
        # The same words as REF; the rest of the adherent ones matched on lemma only.
        in_ref_words = Counter(w for w in ref_words if w)
        exact_n = min(
            sum(min(n, in_ref_words[w]) for w, n in Counter(w for w in text_words if w).items()),
            adherent_n,
        )

        for tally in (result, result.strict if strictness == "strict" else result.permissive):
            tally.expected += expected_n
            tally.adherent += adherent_n
            tally.exact += exact_n

        # Bucketed on the counts: presence alone cannot tell "as often as the human" from "more".
        bucket = bucket_of(rendered, expected_n)
        setattr(result.terms, bucket, getattr(result.terms, bucket) + 1)

        result.term_scores.append(TermScore(
            source_content=source, expected_targets=expected_targets, strictness=strictness,
            expected=expected_n, adherent=adherent_n, rendered=rendered, exact=exact_n,
        ))
        if missed:
            result.violations.append(Violation(
                source_content=source, kind=MISS,
                expected_targets=expected_targets, strictness=strictness,
                missed_occurrences=missed, expected_occurrences=expected_n,
            ))

    return result


@dataclass
class Aggregate:
    expected: int
    adherent: int
    violations: int
    adherence_rate: float | None
    strict: Tally
    permissive: Tally
    terms: TermBreakdown
    segments_scored: int
    segments_clean: int
    segment_adherence_rate: float | None
    exact: int = 0
    exact_rate: float | None = None


def _combine(
    items: Sequence[Score] | Sequence[Aggregate],
    scored: Sequence[Score] | Sequence[Aggregate],
    segments_scored: int,
    segments_clean: int,
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
        strict=strict,
        permissive=permissive,
        terms=terms,
        segments_scored=segments_scored,
        segments_clean=segments_clean,
        segment_adherence_rate=rate(segments_clean, segments_scored),
        exact=total.exact,
        exact_rate=total.exact_rate,
    )


def aggregate(scores: Sequence[Score]) -> Aggregate:
    scored = [s for s in scores if s.expected]
    return _combine(
        scores,
        scored,
        segments_scored=len(scored),
        segments_clean=sum(1 for s in scored if s.adherent == s.expected),
    )


def pool(aggregates: Sequence[Aggregate]) -> Aggregate:
    """Rates recomputed from the pooled totals; averaging would weight 3 instances like 300."""
    items = list(aggregates)
    return _combine(
        items,
        items,
        segments_scored=sum(a.segments_scored for a in items),
        segments_clean=sum(a.segments_clean for a in items),
    )


@dataclass
class ViolationReport:
    miss: int = 0
    inconsistency: int = 0
    over_application: int = 0
    total: int = 0
    # Every segment, not just the glossary-bearing ones: one with no term retrieved can over-apply.
    segments: int = 0
    segments_with_violation: int = 0
    violation_rate: float | None = None
    items: list[Violation] = field(default_factory=list)


def find_violations(
    *,
    versions: Sequence[Sequence[str]],
    ref_texts: Sequence[str],
    source_texts: Sequence[str],
    per_segment_mappings: Sequence[Iterable[Mapping[str, str]]],
    corpus_mappings: Iterable[Mapping[str, str]],
    language_codes: Sequence[str | None],
    source_language_codes: Sequence[str | None],
    lemmas_by_language: Mapping[str, Mapping[str, str]] | None = None,
) -> list[ViolationReport]:
    """One report per version, with what REF rendered derived once for all of them."""
    lemmas_by_language = lemmas_by_language or {}
    for texts in versions:
        lengths = {len(texts), len(ref_texts), len(source_texts), len(per_segment_mappings),
                   len(language_codes), len(source_language_codes)}
        if len(lengths) > 1:
            raise ValueError(
                f"{len(texts)} texts, {len(ref_texts)} references, {len(source_texts)} sources, "
                f"{len(per_segment_mappings)} segments of mappings, {len(language_codes)} "
                f"language codes and {len(source_language_codes)} source language codes "
                "must be the same length"
            )

    corpus_map = build_glossary_map(corpus_mappings)

    def variants_in(text: str, index: int, sources: Iterable[str]) -> dict[str, list[str]]:
        """Each source's wordings in `text`, with only these sources' targets competing for overlaps."""
        known = lemmas_by_language.get(language_codes[index]) or {}
        lemmas = known.get(text)
        normalized_text, normalized_lemmas = normalize_text(text), normalize_text(lemmas)
        sources = list(sources)
        # Necessary for either mode and far cheaper: every term is tested against every segment.
        candidates = [
            target for source in sources for target in corpus_map[source]
            if normalize_text(target) in normalized_text
            or (known.get(target) and normalized_lemmas
                and normalize_text(known[target]) in normalized_lemmas)
        ]
        owned = _owned(text, candidates, language_codes[index], lemmas, known)

        variants = {}
        for source in sources:
            # Case and spacing variants of one wording are one rendering.
            forms: dict[str, str] = {}
            for target in sorted(corpus_map[source]):
                if owned.get(target):
                    forms.setdefault(normalize_text(target), target)
            variants[source] = list(forms.values())
        return variants

    @cache
    def in_source(index: int, source: str) -> bool:
        known = lemmas_by_language.get(source_language_codes[index]) or {}
        return bool(count_occurrences(
            text=source_texts[index], term=source, language_code=source_language_codes[index],
            text_lemmas=known.get(source_texts[index]), term_lemmas=known.get(source),
        ))

    # Version-independent, so none of this is redone per version.
    segment_maps = [build_glossary_map(mappings) for mappings in per_segment_mappings]
    # From the raw mappings, so a term retrieved with a blank target still counts as retrieved.
    retrieved_per_segment = [
        {m.get("source_content") for m in mappings if m.get("source_content")}
        for mappings in per_segment_mappings
    ]
    in_reference = [
        variants_in(ref_text, index, segment_maps[index]) for index, ref_text in enumerate(ref_texts)
    ]
    ref_licensed = [
        {normalize_text(variant)
         for variants in variants_in(ref_text, index, corpus_map).values() for variant in variants}
        for index, ref_text in enumerate(ref_texts)
    ]

    reports = []
    for texts in versions:
        report = ViolationReport(segments=len(texts))

        for index, text in enumerate(texts):
            segment_map = segment_maps[index]
            reference = in_reference[index]
            # Retrieved terms compete only with each other, as they do in score_glossary.
            found = variants_in(text, index, segment_map)

            # Any wording the reference used, plus any a term retrieved here sanctions.
            licensed = ref_licensed[index] | {
                normalize_text(variant) for variants in found.values() for variant in variants
            }

            for source in segment_map:
                # The reference declined the term here, so there is nothing to hold it to.
                if not reference[source]:
                    continue
                used = found[source]
                if not used:
                    report.items.append(Violation(source, MISS, segment_index=index))
                    continue
                # The reference settles the wording, or the version would grade itself.
                intended = {normalize_text(variant) for variant in reference[source]}
                for variant in used:
                    if normalize_text(variant) not in intended:
                        report.items.append(
                            Violation(source, INCONSISTENCY, segment_index=index, detail=variant))

            for source, used in variants_in(text, index, corpus_map).items():
                # A source term the segment carries is a retrieval gap, not the translation's doing.
                if not used or source in retrieved_per_segment[index] or in_source(index, source):
                    continue
                for variant in used:
                    if normalize_text(variant) not in licensed:
                        report.items.append(
                            Violation(source, OVER_APPLICATION, segment_index=index, detail=variant))

        report.items.sort(key=lambda item: (item.segment_index, str(item.source_content), item.kind))
        kinds = Counter(item.kind for item in report.items)
        report.miss, report.inconsistency, report.over_application = (
            kinds[MISS], kinds[INCONSISTENCY], kinds[OVER_APPLICATION])
        report.total = len(report.items)
        report.segments_with_violation = len({item.segment_index for item in report.items})
        report.violation_rate = rate(report.segments_with_violation, report.segments)
        reports.append(report)

    return reports


@dataclass
class ReferenceCheck:
    """Whether the human translation rendered the glossary terms the source retrieved."""

    terms_checked: int = 0
    terms_rendered: int = 0
    segments_to_review: int = 0
    items: list[Violation] = field(default_factory=list)

    @property
    def to_review(self) -> int:
        return self.terms_checked - self.terms_rendered


def check_reference(
    *,
    ref_texts: Sequence[str],
    per_segment_mappings: Sequence[Iterable[Mapping[str, str]]],
    language_codes: Sequence[str | None],
    lemmas_by_language: Mapping[str, Mapping[str, str]] | None = None,
) -> ReferenceCheck:
    """Indicative only: percolation also retrieves terms whose sense does not apply here."""
    check = ReferenceCheck()
    for index, text in enumerate(ref_texts):
        known = (lemmas_by_language or {}).get(language_codes[index]) or {}
        glossary_map = build_glossary_map(per_segment_mappings[index])
        owned = _owned(
            text, [target for targets in glossary_map.values() for target in targets],
            language_codes[index], known.get(text), known,
        )
        for source, targets in glossary_map.items():
            expected_targets = sorted(targets)
            check.terms_checked += 1
            if any(owned.get(target) for target in expected_targets):
                check.terms_rendered += 1
            else:
                check.items.append(Violation(
                    source, MISS, expected_targets=expected_targets, segment_index=index,
                ))

    check.items.sort(key=lambda item: (item.segment_index, str(item.source_content)))
    check.segments_to_review = len({item.segment_index for item in check.items})
    return check


def pool_violations(reports: Sequence[ViolationReport]) -> ViolationReport:
    """`items` is left empty: a segment index only means something inside its own dataset."""
    pooled = ViolationReport()
    for report in reports:
        pooled.miss += report.miss
        pooled.inconsistency += report.inconsistency
        pooled.over_application += report.over_application
        pooled.total += report.total
        pooled.segments += report.segments
        pooled.segments_with_violation += report.segments_with_violation
    pooled.violation_rate = rate(pooled.segments_with_violation, pooled.segments)
    return pooled
