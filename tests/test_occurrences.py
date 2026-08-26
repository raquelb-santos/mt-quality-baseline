"""Occurrence-weighted counting: catching inconsistent terminology inside one segment."""

import pytest

from sourcecode.match import count_lemma, count_occurrences, count_surface
from sourcecode.score import aggregate, pool, rate, score_translation

MAPPINGS = [{"source_content": "engine", "target_content": "moteur"}]

#: The human reference is the denominator: it states the term three times, so a translation owes
#: three renderings of it — the source is never counted.
REFERENCE = "Le moteur entraîne le support moteur et le capot moteur."


def test_count_surface_is_boundary_aware():
    assert count_surface("the engine and the engine", "engine", "en-gb") == 2
    # must not count inside a longer word
    assert count_surface("engineering engines", "engine", "en-gb") == 0


def test_count_surface_is_non_overlapping_for_unspaced_languages():
    assert count_surface("エンジンとエンジン", "エンジン", "ja-jp") == 2


def test_count_lemma_is_non_overlapping():
    assert count_lemma(["a", "a", "a"], ["a", "a"]) == 1  # not 2
    assert count_lemma(["moteur", "x", "moteur"], ["moteur"]) == 2


def test_count_occurrences_does_not_double_count_surface_and_lemma():
    # An uninflected match is found on the surface; the lemma pass must not add to it.
    assert count_occurrences(
        text="le moteur", term="moteur", language_code="fr-fr",
        text_lemmas="le moteur", term_lemmas="moteur",
    ) == 1


def test_a_term_the_reference_never_uses_is_not_scored():
    """Retrieval proposes, the reference disposes. A term the human declined has no denominator,
    so it can be neither adhered to nor violated, and it is dropped rather than scored as clean."""
    score = score_translation(
        mappings=MAPPINGS, text="Le bloc et le groupe.", language_code="fr-fr",
        reference_text="Le bloc entraîne le support.",
    )
    assert (score.expected, score.adherent) == (0, 0)
    assert score.term_scores == []
    assert score.violations == []


def test_a_term_the_reference_repeats_but_the_mt_renders_once_is_a_partial_violation():
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (3, 1)

    violation = score.violations[0]
    assert violation.missed_occurrences == 2
    assert violation.expected_occurrences == 3
    # Rendered somewhere but not everywhere: inconsistent use, not outright absence.
    assert violation.missed_occurrences < violation.expected_occurrences


def test_fully_consistent_use_scores_clean():
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (3, 3)
    assert score.violations == []


def test_absent_term_misses_every_occurrence():
    score = score_translation(
        mappings=MAPPINGS, text="Rien ici.", language_code="fr-fr", reference_text=REFERENCE,
    )
    # Not partial: nothing was rendered, so every occurrence the human made is missed.
    assert score.violations[0].missed_occurrences == 3
    assert score.violations[0].expected_occurrences == 3


def test_an_inflected_reference_still_sets_the_denominator():
    """The reference is counted with the same lemma fallback as the translation, so a term the
    human inflected is not silently dropped out of the measurement."""
    score = score_translation(
        mappings=[{"source_content": "cable", "target_content": "câble"}],
        text="le câble", language_code="fr-fr",
        reference_text="Branchez les câbles.",        # "câble" never appears verbatim
        reference_lemmas="brancher le câble",
        term_lemmas={"câble": "câble"},
    )
    assert score.expected == 1
    assert score.adherent == 1


def test_renderings_cannot_exceed_the_reference_denominator():
    score = score_translation(
        mappings=MAPPINGS, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        reference_text="Le moteur.",
    )
    assert score.expected == 1
    assert score.adherent == 1  # capped, never 5


def test_permissive_variants_sum_toward_the_denominator():
    mappings = [
        {"source_content": "battery", "target_content": "batterie"},
        {"source_content": "battery", "target_content": "accumulateur"},
    ]
    score = score_translation(
        mappings=mappings, text="la batterie et l'accumulateur", language_code="fr-fr",
        reference_text="la batterie et la batterie",
    )
    # The human used two of the target terms, the MT two others — different variants both count.
    assert (score.expected, score.adherent) == (2, 2)


# ------------------------------------------- the term breakdown, reported beside the one rate
#
# REFERENCE states its term three times, so `expected` is 3 and each bucket is reachable from it.


def test_a_term_with_no_rendering_at_all_is_never_used():
    score = score_translation(
        mappings=MAPPINGS, text="Rien ici.", language_code="fr-fr", reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (3, 0)
    assert score.terms.never_used == 1
    assert score.terms.present == 0


def test_a_term_rendered_fewer_times_than_the_reference_is_used_partly():
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (3, 1)
    assert score.terms.used_partly == 1
    # Present, but not everywhere: the distinction a single-use metric cannot draw.
    assert score.terms.present == 1


def test_a_term_rendered_as_often_as_the_reference_is_used_everywhere():
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", reference_text=REFERENCE,
    )
    assert score.terms.used_everywhere == 1


def test_over_use_is_recorded_but_never_moves_the_rate():
    score = score_translation(
        mappings=MAPPINGS, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    assert score.terms.over_used == 1
    # The cap still holds: five renderings against the human's three score 3/3, never 5/3.
    assert (score.expected, score.adherent) == (3, 3)
    assert score.violations == []


def test_a_term_the_human_avoided_is_over_use_rather_than_adherence():
    """The review queue the old "APE used, human did not" counter carried: the glossary entry
    usually does not fit that context, so it is recorded and never scored either way."""
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur est là.", language_code="fr-fr",
        reference_text="Le bloc est là.",
    )
    assert (score.expected, score.adherent) == (0, 0)
    assert score.terms.over_used == 1
    assert score.violations == []
    assert score.term_scores[0].rendered == 1


def test_the_four_buckets_are_exhaustive():
    mappings = [
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "brake", "target_content": "frein"},
    ]
    score = score_translation(
        mappings=mappings, text="Le moteur et le frein.", language_code="fr-fr",
        reference_text="Le moteur, le moteur et le frein.",
    )
    assert (score.expected, score.adherent) == (3, 2)      # engine ×2 + brake ×1
    assert score.terms.distinct_terms == 2                 # engine used partly, brake everywhere
    assert (score.terms.used_partly, score.terms.used_everywhere) == (1, 1)


def test_presence_is_recoverable_from_the_violation_detail():
    """The property that made a separate presence rate redundant: a term is present unless it has
    a violation missing every one of its expected occurrences."""
    for text in ("Rien ici.",
                 "Le moteur, le bloc et le groupe.",
                 "Le moteur, le support moteur et le capot moteur.",
                 "moteur moteur moteur moteur moteur"):
        score = score_translation(
            mappings=MAPPINGS, text=text, language_code="fr-fr", reference_text=REFERENCE,
        )
        never = sum(1 for v in score.violations
                    if v.missed_occurrences == v.expected_occurrences)
        assert score.terms.present == score.terms.distinct_terms - never


def test_aggregate_adds_the_buckets_across_segments():
    consistent = score_translation(
        mappings=MAPPINGS, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", reference_text=REFERENCE,
    )
    inconsistent = score_translation(
        mappings=MAPPINGS, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    totals = aggregate([consistent, inconsistent])

    assert (totals.expected, totals.adherent) == (6, 4)
    assert totals.adherence_rate == pytest.approx(4 / 6)
    assert totals.terms.distinct_terms == 2
    assert (totals.terms.used_everywhere, totals.terms.used_partly) == (1, 1)


def test_aggregate_keeps_over_use_from_a_segment_with_nothing_expected():
    """The segment carries no denominator, so it cannot enter the rate — but the term the human
    avoided is the only thing it has to report, and dropping it would hide the review item."""
    avoided = score_translation(
        mappings=MAPPINGS, text="Le moteur est là.", language_code="fr-fr",
        reference_text="Le bloc est là.",
    )
    totals = aggregate([avoided])

    assert totals.expected == 0
    assert totals.adherence_rate is None
    assert totals.segments_with_glossary == 0
    assert totals.terms.over_used == 1


def test_pooling_adds_the_buckets_rather_than_averaging_anything():
    """Counts pool by addition, so the breakdown is split-invariant for free."""
    small = aggregate([score_translation(
        mappings=MAPPINGS, text="Rien ici.", language_code="fr-fr", reference_text="Le moteur.",
    )])
    large = aggregate([score_translation(
        mappings=MAPPINGS, text=f"Le moteur {i}.", language_code="fr-fr",
        reference_text="Le moteur.",
    ) for i in range(9)])

    pooled = pool([small, large])
    assert pooled.terms.distinct_terms == 10
    assert (pooled.terms.never_used, pooled.terms.used_everywhere) == (1, 9)
    assert pooled.terms.present == 9


# ------------------------------------- the same pair rolled up: term, segment, stratum
#
# Each level recomputes its rate from its own pooled counts. Averaging the level below would let a
# term matched once outweigh one matched forty times.


def test_a_term_rate_pools_its_counts_across_segments():
    """Two segments, same term: 1/3 in one and 3/3 in the other pools to 4/6, not (33%+100%)/2."""
    poor = score_translation(
        mappings=MAPPINGS, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    good = score_translation(
        mappings=MAPPINGS, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", reference_text=REFERENCE,
    )
    expected = sum(t.expected for s in (poor, good) for t in s.term_scores)
    adherent = sum(t.adherent for s in (poor, good) for t in s.term_scores)
    assert (adherent, expected) == (4, 6)
    assert rate(adherent, expected) == pytest.approx(4 / 6)   # not the 66.67% an average gives


def test_term_scores_are_recorded_for_adherent_terms_too():
    """A fully adherent term raises no violation, so only this record carries its denominator."""
    score = score_translation(
        mappings=MAPPINGS, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", reference_text=REFERENCE,
    )
    assert score.violations == []
    assert len(score.term_scores) == 1
    term = score.term_scores[0]
    assert (term.expected, term.adherent, term.violations) == (3, 3, 0)


def test_a_term_score_keeps_the_raw_target_count_the_cap_discards():
    score = score_translation(
        mappings=MAPPINGS, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    term = score.term_scores[0]
    assert term.rendered == 5      # T, unbounded
    assert term.adherent == 3      # bounded by R, which is what the rate uses


def test_term_scores_sum_to_the_segment_totals():
    """The per-term grain must reconcile with the segment rate built on top of it."""
    mappings = [
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "brake", "target_content": "frein"},
    ]
    score = score_translation(
        mappings=mappings, text="Le moteur et le frein.", language_code="fr-fr",
        reference_text="Le moteur, le moteur et le frein.",
    )
    assert sum(t.expected for t in score.term_scores) == score.expected
    assert sum(t.adherent for t in score.term_scores) == score.adherent
    assert len(score.term_scores) == score.terms.distinct_terms
