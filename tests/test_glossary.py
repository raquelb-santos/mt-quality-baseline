"""The terminology adherence component: the metric, its rendering, and the orchestration around it."""

from dataclasses import dataclass, field
import pytest

from sourcecode.text_processing import Dataset, normalize_language
from sourcecode.report import by_stratum, pct, stratum_of
from sourcecode.glossary import GlossaryMatches
from sourcecode.glossary_benchmark import Benchmark
from sourcecode.glossary_report import (
    glossary_scorecard,
    render_strata,
    render_term_adherence,
    render_term_adherence_console,
    stratum_rows,
    term_rows,
)
from sourcecode.glossary_score import (
    INCONSISTENCY,
    MISS,
    OVER_APPLICATION,
    ViolationReport,
    find_violations,
    pool_violations,
    aggregate,
    build_glossary_map,
    pool,
    rate,
    score_translation,
)
from sourcecode.glossary_score import Aggregate, TallyReport, TermBreakdown, pool
from sourcecode.postmt import RunResult


# ================================================================================================
# the metric
# ================================================================================================

MAPPINGS = [
    {"source_content": "brake", "target_content": "frein"},
    {"source_content": "brake", "target_content": "freinage"},   # permissive: two target terms
    {"source_content": "engine", "target_content": "moteur"},    # strict: one target term
]

# What the human wrote, and therefore what every version below is measured against.
REFERENCE = "le frein du moteur"

# One term stated three times in the reference, so every occurrence bucket is reachable.
ENGINE = [{"source_content": "engine", "target_content": "moteur"}]
THRICE = "Le moteur entraîne le support moteur et le capot moteur."


# --- instances: strict, permissive, and what the mapping is built from -------------------------------

def test_glossary_map_groups_targets_by_source():
    # Mirrors how post-mt phrases the prompt, so we score the contract the model was shown.
    mapping = build_glossary_map(
        [
            {"source_content": "brake", "target_content": "frein"},
            {"source_content": "brake", "target_content": "freinage"},
            {"source_content": "engine", "target_content": "moteur"},
            {"source_content": "", "target_content": "ignored"},
            {"source_content": "x", "target_content": ""},
        ]
    )
    assert set(mapping) == {"brake", "engine"}
    assert mapping["brake"] == {"frein", "freinage"}
    assert mapping["engine"] == {"moteur"}


def test_any_permissive_variant_is_adherent():
    score = score_translation(
        mappings=MAPPINGS, text="le freinage du moteur", language_code="fr-fr",
        ref_text="le freinage du moteur",
    )
    assert (score.expected, score.adherent) == (2, 2)
    assert score.violations == []
    assert score.permissive.adherent == 1
    assert score.strict.adherent == 1


def test_missing_strict_term_is_recorded():
    score = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        ref_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 1)
    assert len(score.violations) == 1

    violation = score.violations[0]
    assert violation.source_content == "engine"
    assert violation.expected_targets == ["moteur"]
    assert violation.strictness == "strict"


def test_expected_instances_count_per_term_not_per_segment():
    # Two terms in one reference must contribute 2 instances, or the metric tracks segment length.
    score = score_translation(
        mappings=MAPPINGS, text="rien ici", language_code="fr-fr", ref_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 0)
    assert {v.strictness for v in score.violations} == {"strict", "permissive"}


def test_segments_without_glossary_terms_contribute_nothing():
    score = score_translation(
        mappings=[], text="whatever", language_code="fr-fr", ref_text="peu importe",
    )
    assert score.expected == 0

    agg = aggregate([score])
    assert agg.expected == 0
    # None, never 0 — no evidence is not zero adherence, and a 0 would drag rollups down.
    assert agg.adherence_rate is None
    assert agg.segments_with_glossary == 0


def test_lemma_matching_is_used_when_lemmas_are_supplied():
    mappings = [{"source_content": "engine", "target_content": "moteur electrique"}]

    without = score_translation(
        mappings=mappings, text="les moteurs electriques sont bons", language_code="fr-fr",
        ref_text="le moteur electrique est bon",
    )
    assert without.expected == 1
    assert without.adherent == 0

    with_lemmas = score_translation(
        mappings=mappings,
        text="les moteurs electriques sont bons",
        language_code="fr-fr",
        text_lemmas="le moteur electrique etre bon",
        ref_text="le moteur electrique est bon",
        term_lemmas={"moteur electrique": "moteur electrique"},
    )
    assert with_lemmas.adherent == 1


# --- the reference is the denominator ------------------------------------------------------------------

def test_term_reference_never_uses_is_not_scored():
    """Retrieval proposes, the reference disposes: no denominator, so it is dropped."""
    score = score_translation(
        mappings=ENGINE, text="Le bloc et le groupe.", language_code="fr-fr",
        ref_text="Le bloc entraîne le support.",
    )
    assert (score.expected, score.adherent) == (0, 0)
    assert score.term_scores == []
    assert score.violations == []


def test_partial_rendering_is_violation():
    score = score_translation(
        mappings=ENGINE, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        ref_text=THRICE,
    )
    assert (score.expected, score.adherent) == (3, 1)

    violation = score.violations[0]
    assert violation.missed_occurrences == 2
    assert violation.expected_occurrences == 3
    # Rendered somewhere but not everywhere: inconsistent use, not outright absence.
    assert violation.missed_occurrences < violation.expected_occurrences


def test_fully_consistent_use_scores_clean():
    score = score_translation(
        mappings=ENGINE, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", ref_text=THRICE,
    )
    assert (score.expected, score.adherent) == (3, 3)
    assert score.violations == []


def test_absent_term_misses_every_occurrence():
    score = score_translation(
        mappings=ENGINE, text="Rien ici.", language_code="fr-fr", ref_text=THRICE,
    )
    # Not partial: nothing was rendered, so every occurrence the human made is missed.
    assert score.violations[0].missed_occurrences == 3
    assert score.violations[0].expected_occurrences == 3


def test_inflected_reference_still_sets_denominator():
    """The reference uses the same lemma fallback, so an inflected term is not silently dropped."""
    score = score_translation(
        mappings=[{"source_content": "cable", "target_content": "câble"}],
        text="le câble", language_code="fr-fr",
        ref_text="Branchez les câbles.",        # "câble" never appears verbatim
        ref_lemmas="brancher le câble",
        term_lemmas={"câble": "câble"},
    )
    assert score.expected == 1
    assert score.adherent == 1


def test_renderings_cannot_exceed_reference_denominator():
    score = score_translation(
        mappings=ENGINE, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        ref_text="Le moteur.",
    )
    assert score.expected == 1
    assert score.adherent == 1  # capped, never 5


def test_permissive_variants_sum_toward_denominator():
    mappings = [
        {"source_content": "battery", "target_content": "batterie"},
        {"source_content": "battery", "target_content": "accumulateur"},
    ]
    score = score_translation(
        mappings=mappings, text="la batterie et l'accumulateur", language_code="fr-fr",
        ref_text="la batterie et la batterie",
    )
    # The human used two of the target terms, the MT two others — different variants both count.
    assert (score.expected, score.adherent) == (2, 2)


# --- the term breakdown, reported beside the one rate ------------------------------------------------

def test_term_with_no_rendering_at_all_is_never_used():
    score = score_translation(
        mappings=ENGINE, text="Rien ici.", language_code="fr-fr", ref_text=THRICE,
    )
    assert (score.expected, score.adherent) == (3, 0)
    assert score.terms.never_used == 1


def test_fewer_renderings_is_used_partly():
    score = score_translation(
        mappings=ENGINE, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        ref_text=THRICE,
    )
    assert (score.expected, score.adherent) == (3, 1)
    # Present, but not everywhere: the distinction a single-use metric cannot draw.
    assert (score.terms.used_partly, score.terms.never_used) == (1, 0)


def test_matching_reference_is_used_everywhere():
    score = score_translation(
        mappings=ENGINE, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", ref_text=THRICE,
    )
    assert score.terms.used_everywhere == 1


def test_over_use_is_recorded_but_never_moves_rate():
    score = score_translation(
        mappings=ENGINE, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        ref_text=THRICE,
    )
    assert score.terms.over_used == 1
    # The cap still holds: five renderings against the human's three score 3/3, never 5/3.
    assert (score.expected, score.adherent) == (3, 3)
    assert score.violations == []


def test_avoided_term_is_over_use():
    """The entry usually does not fit that context: recorded, never scored either way."""
    score = score_translation(
        mappings=ENGINE, text="Le moteur est là.", language_code="fr-fr",
        ref_text="Le bloc est là.",
    )
    assert (score.expected, score.adherent) == (0, 0)
    assert score.terms.over_used == 1
    assert score.violations == []
    assert score.term_scores[0].rendered == 1


def test_four_buckets_are_exhaustive():
    mappings = [
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "brake", "target_content": "frein"},
    ]
    score = score_translation(
        mappings=mappings, text="Le moteur et le frein.", language_code="fr-fr",
        ref_text="Le moteur, le moteur et le frein.",
    )
    assert (score.expected, score.adherent) == (3, 2)      # engine ×2 + brake ×1
    assert score.terms.distinct_terms == 2                 # engine used partly, brake everywhere
    assert (score.terms.used_partly, score.terms.used_everywhere) == (1, 1)


def test_presence_is_recoverable_from_violation_detail():
    """A term is present unless a violation misses every one of its expected occurrences."""
    for text in ("Rien ici.",
                 "Le moteur, le bloc et le groupe.",
                 "Le moteur, le support moteur et le capot moteur.",
                 "moteur moteur moteur moteur moteur"):
        score = score_translation(
            mappings=ENGINE, text=text, language_code="fr-fr", ref_text=THRICE,
        )
        never = sum(1 for v in score.violations
                    if v.missed_occurrences == v.expected_occurrences)
        assert score.terms.never_used == never


# --- a missed instance is a violation, whatever the version did -----------------------------------------

def test_miss_in_rendered_segment_is_violation():
    """The ordinary failure: the segment was translated, just not with the target term."""
    score = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        ref_text=REFERENCE,
    )
    assert score.violations[0].missed_occurrences == 1

    agg = aggregate([score])
    assert agg.violations == 1
    assert agg.adherence_rate == 0.5


def test_miss_with_no_translation_at_all_is_violation():
    """An empty version is not excused: every instance the reference rendered is still owed."""
    score = score_translation(
        mappings=MAPPINGS, text="", language_code="fr-fr", ref_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 0)

    agg = aggregate([score])
    assert agg.adherence_rate == 0.0
    assert agg.violations == 2


def test_whitespace_only_is_violation():
    score = score_translation(
        mappings=MAPPINGS, text="      ", language_code="fr-fr", ref_text=REFERENCE,
    )
    assert aggregate([score]).violations == 2


def test_violations_account_for_whole_shortfall():
    missing = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        ref_text=REFERENCE,
    )
    empty = score_translation(
        mappings=MAPPINGS, text="", language_code="fr-fr", ref_text=REFERENCE,
    )
    clean = score_translation(
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        ref_text=REFERENCE,
    )

    agg = aggregate([missing, empty, clean])

    assert agg.violations == agg.expected - agg.adherent == 3
    # Adherence is the only rate, and it is a share of what the reference demanded.
    assert agg.adherence_rate == agg.adherent / agg.expected

    # And it pools per slice as well as overall.
    assert agg.strict.violations + agg.permissive.violations == 3


# --- aggregating across segments -------------------------------------------------------------------------

def test_aggregate_computes_both_rates():
    clean = score_translation(                                                                # 2/2
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        ref_text=REFERENCE,
    )
    partial = score_translation(                                                              # 1/2
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        ref_text=REFERENCE,
    )

    agg = aggregate([clean, partial])

    assert (agg.expected, agg.adherent, agg.violations) == (4, 3, 1)
    assert agg.adherence_rate == 0.75

    # Segment level: only one of the two segments is fully clean.
    assert agg.segments_with_glossary == 2
    assert agg.segments_fully_adherent == 1
    assert agg.segment_adherence_rate == 0.5

    # The strict slice is the one failing here.
    assert agg.strict.adherence_rate == 0.5
    assert agg.permissive.adherence_rate == 1.0


def test_aggregate_ignores_empty_scores():
    scored = score_translation(
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        ref_text=REFERENCE,
    )
    empty = score_translation(
        mappings=[], text="nothing", language_code="fr-fr", ref_text="rien",
    )

    agg = aggregate([scored, empty, empty])

    # Three segments in, but only one had terms to judge.
    assert agg.segments_with_glossary == 1
    assert agg.segment_adherence_rate == 1.0


def test_aggregate_adds_buckets_across_segments():
    consistent = score_translation(
        mappings=ENGINE, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", ref_text=THRICE,
    )
    inconsistent = score_translation(
        mappings=ENGINE, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        ref_text=THRICE,
    )
    totals = aggregate([consistent, inconsistent])

    assert (totals.expected, totals.adherent) == (6, 4)
    assert totals.adherence_rate == pytest.approx(4 / 6)
    assert totals.terms.distinct_terms == 2
    assert (totals.terms.used_everywhere, totals.terms.used_partly) == (1, 1)


def test_aggregate_keeps_unexpected_over_use():
    """No denominator, so it cannot enter the rate - but dropping it would hide the review item."""
    avoided = score_translation(
        mappings=ENGINE, text="Le moteur est là.", language_code="fr-fr",
        ref_text="Le bloc est là.",
    )
    totals = aggregate([avoided])

    assert totals.expected == 0
    assert totals.adherence_rate is None
    assert totals.segments_with_glossary == 0
    assert totals.terms.over_used == 1


# --- the same evidence rolled up: term, segment, dataset ---------------------------------------------------
# Each level recomputes its rate from its own pooled counts rather than averaging the level below.

def test_term_rate_pools_counts_across_segments():
    """Two segments, same term: 1/3 in one and 3/3 in the other pools to 4/6, not (33%+100%)/2."""
    poor = score_translation(
        mappings=ENGINE, text="Le moteur, le bloc et le groupe.", language_code="fr-fr",
        ref_text=THRICE,
    )
    good = score_translation(
        mappings=ENGINE, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", ref_text=THRICE,
    )
    expected = sum(t.expected for s in (poor, good) for t in s.term_scores)
    adherent = sum(t.adherent for s in (poor, good) for t in s.term_scores)
    assert (adherent, expected) == (4, 6)
    assert rate(adherent, expected) == pytest.approx(4 / 6)   # not the 66.67% an average gives


def test_term_scores_are_recorded_for_adherent_terms_too():
    """A fully adherent term raises no violation, so only this record carries its denominator."""
    score = score_translation(
        mappings=ENGINE, text="Le moteur, le support moteur et le capot moteur.",
        language_code="fr-fr", ref_text=THRICE,
    )
    assert score.violations == []
    assert len(score.term_scores) == 1
    term = score.term_scores[0]
    assert (term.expected, term.adherent, term.violations) == (3, 3, 0)


def test_term_score_keeps_raw_count():
    score = score_translation(
        mappings=ENGINE, text="moteur moteur moteur moteur moteur", language_code="fr-fr",
        ref_text=THRICE,
    )
    term = score.term_scores[0]
    assert term.rendered == 5      # unbounded
    assert term.adherent == 3      # bounded by REF's, which is what the rate uses


def test_term_scores_sum_to_segment_totals():
    """The per-term grain must reconcile with the segment rate built on top of it."""
    mappings = [
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "brake", "target_content": "frein"},
    ]
    score = score_translation(
        mappings=mappings, text="Le moteur et le frein.", language_code="fr-fr",
        ref_text="Le moteur, le moteur et le frein.",
    )
    assert sum(t.expected for t in score.term_scores) == score.expected
    assert sum(t.adherent for t in score.term_scores) == score.adherent
    assert len(score.term_scores) == score.terms.distinct_terms


def test_pooling_adds_buckets():
    """Counts pool by addition, so the breakdown is split-invariant for free."""
    small = aggregate([score_translation(
        mappings=ENGINE, text="Rien ici.", language_code="fr-fr", ref_text="Le moteur.",
    )])
    large = aggregate([score_translation(
        mappings=ENGINE, text=f"Le moteur {i}.", language_code="fr-fr",
        ref_text="Le moteur.",
    ) for i in range(9)])

    pooled = pool([small, large])
    assert pooled.terms.distinct_terms == 10
    assert (pooled.terms.never_used, pooled.terms.used_everywhere) == (1, 9)



# ================================================================================================
# corpus-level violations
# ================================================================================================

def pairs(*mapping):
    return [{"source_content": s, "target_content": t} for s, t in mapping]


BRAKE_PAIRS = pairs(("brake pad", "frein"))
ENGINE_PAIRS = pairs(("engine", "moteur"))


def run(texts, references, per_segment, corpus=None, **kwargs):
    return find_violations(
        texts=texts, ref_texts=references, per_segment_mappings=per_segment,
        corpus_mappings=corpus if corpus is not None else [m for g in per_segment for m in g],
        language_code="fr-fr", **kwargs,
    )


def kinds(report):
    return sorted((item.kind, item.source_content, item.detail) for item in report.items)


# --- misses -------------------------------------------------------------------------------------

def test_term_reference_used_and_output_dropped_is_miss():
    report = run(["La plaquette est usee."], ["Le frein est use."], [BRAKE_PAIRS])
    assert kinds(report) == [(MISS, "brake pad", "")]


def test_term_reference_declined_is_not_miss():
    """Retrieval fires on senses a translator rightly avoids; only the reference decides."""
    report = run(["Le dispositif."], ["Le dispositif."], [BRAKE_PAIRS])
    assert report.items == []


def test_empty_output_misses_every_term_reference_used():
    report = run([""], ["Le frein est use."], [BRAKE_PAIRS])
    assert kinds(report) == [(MISS, "brake pad", "")]


def test_lemma_match_is_rendering_not_miss():
    """The lemma fallback applies here as it does to adherence, or an inflected term reads as a miss."""
    report = run(
        ["Les freins sont uses."], ["Le frein est use."], [BRAKE_PAIRS],
        text_lemmas={"Les freins sont uses.": "le frein etre use", "Le frein est use.": "le frein etre use"},
        term_lemmas={"frein": "frein"},
    )
    assert report.items == []


# --- inconsistency ------------------------------------------------------------------------------

def test_two_renderings_of_one_term_is_inconsistency():
    corpus = ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(["Le moteur.", "Le moteur.", "Le bloc moteur."], ["Le moteur."] * 3,
                 [ENGINE_PAIRS] * 3, corpus=corpus)
    assert kinds(report) == [(INCONSISTENCY, "engine", "bloc moteur")]


def test_one_rendering_throughout_is_consistent():
    report = run(["Le moteur."] * 3, ["Le moteur."] * 3, [ENGINE_PAIRS] * 3)
    assert report.items == []


def test_reference_wording_wins_over_majority():
    """Two segments the wrong way, one right: counting would elect the wrong wording as intended."""
    corpus = ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(["Le bloc moteur.", "Le bloc moteur.", "Le moteur."], ["Le moteur."] * 3,
                 [ENGINE_PAIRS] * 3, corpus=corpus)
    assert [i.detail for i in report.items if i.kind == INCONSISTENCY] == ["bloc moteur"] * 2


def test_split_evenly_still_follows_reference():
    """One each way: the reference settles it, so there is no tie to break."""
    corpus = ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(["Le moteur.", "Le bloc moteur."], ["Le moteur."] * 2, [ENGINE_PAIRS] * 2,
                 corpus=corpus)
    assert [i.detail for i in report.items if i.kind == INCONSISTENCY] == ["bloc moteur"]


def test_reference_may_render_one_term_two_ways():
    """Different segments, different contexts: the human's own variation is not a violation."""
    corpus = ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(["Le moteur.", "Le bloc moteur."], ["Le moteur.", "Le bloc moteur."],
                 [ENGINE_PAIRS] * 2, corpus=corpus)
    assert report.items == []


def test_overlapping_targets_are_one_rendering():
    """`moteur` inside `bloc moteur` is one wording, not two competing ones."""
    corpus = ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(["Le bloc moteur."] * 2, ["Le bloc moteur."] * 2, [ENGINE_PAIRS] * 2, corpus=corpus)
    assert [i for i in report.items if i.kind == INCONSISTENCY] == []


def test_case_and_spacing_variants_are_one():
    corpus = ENGINE_PAIRS + pairs(("engine", "Moteur"))
    report = run(["Le moteur."] * 2, ["Le moteur."] * 2, [ENGINE_PAIRS] * 2, corpus=corpus)
    assert [i for i in report.items if i.kind == INCONSISTENCY] == []


# --- over-application ---------------------------------------------------------------------------

def test_target_without_its_source_is_over_application():
    """Nothing retrieved here and the human chose otherwise, so the wording is unlicensed."""
    report = run(["Le frein est ici."], ["Le dispositif est ici."], [[]], corpus=BRAKE_PAIRS)
    assert kinds(report) == [(OVER_APPLICATION, "brake pad", "frein")]


def test_wording_reference_also_chose_is_never_over_application():
    report = run(["Le frein est ici."], ["Le frein est ici."], [[]], corpus=BRAKE_PAIRS)
    assert report.items == []


def test_shared_target_is_not_over_application():
    """Two source terms sharing one target is ordinary; the wording is licensed either way."""
    corpus = BRAKE_PAIRS + pairs(("brake", "frein"))
    report = run(["Le frein est use."], ["Le frein est use."], [BRAKE_PAIRS], corpus=corpus)
    assert [i for i in report.items if i.kind == OVER_APPLICATION] == []


def test_blank_target_still_licenses_rendering():
    """A term retrieved with a blank target counts as retrieved, so its rendering is not charged."""
    retrieved = [{"source_content": "brake pad", "target_content": ""}]
    report = run(["Le frein est ici."], ["Le dispositif est ici."], [retrieved], corpus=BRAKE_PAIRS)
    assert [i for i in report.items if i.kind == OVER_APPLICATION] == []


# --- totals and the rate ------------------------------------------------------------------------

def test_three_kinds_sum_to_total():
    corpus = BRAKE_PAIRS + ENGINE_PAIRS + pairs(("engine", "bloc moteur"))
    report = run(
        ["La plaquette.", "Le moteur.", "Le bloc moteur."],
        ["Le frein.", "Le moteur.", "Le moteur."],
        [BRAKE_PAIRS, ENGINE_PAIRS, ENGINE_PAIRS], corpus=corpus,
    )
    assert report.miss + report.inconsistency + report.over_application == report.total
    assert report.total == len(report.items)


def test_violation_rate_is_over_all_segments():
    """Not just the glossary-bearing ones: a segment with no term retrieved can still over-apply."""
    report = run(["La plaquette.", "Le frein."], ["Le frein.", "Le frein."], [BRAKE_PAIRS, []])
    assert report.segments == 2
    assert report.violation_rate == 0.5


def test_one_segment_counts_once():
    corpus = BRAKE_PAIRS + ENGINE_PAIRS
    report = run(["La plaquette et le bloc."], ["Le frein et le moteur."], [corpus], corpus=corpus)
    assert report.miss == 2
    assert report.segments_with_violation == 1
    assert report.violation_rate == 1.0


def test_no_segments_gives_no_rate_rather_than_zero():
    report = run([], [], [])
    assert report.violation_rate is None


def test_inputs_must_be_same_length():
    with pytest.raises(ValueError, match="same length"):
        run(["a", "b"], ["a"], [[]])


# --- pooling ------------------------------------------------------------------------------------

def test_pooling_recomputes_rate_from_summed_counts():
    one = run(["La plaquette."], ["Le frein."], [BRAKE_PAIRS])
    clean = run(["Le frein.", "Le frein."], ["Le frein."] * 2, [BRAKE_PAIRS] * 2)
    pooled = pool_violations([one, clean])
    assert pooled.miss == 1
    assert pooled.segments == 3
    assert pooled.violation_rate == 1 / 3


def test_pooling_leaves_items_empty():
    """A segment index only means something inside the dataset it came from."""
    one = run(["La plaquette."], ["Le frein."], [BRAKE_PAIRS])
    assert pool_violations([one]).items == []

# ================================================================================================
# rendering
# ================================================================================================

REPORT_GLOSSARY = {
    "The brake pad is worn.": [{"source_content": "brake pad", "target_content": "frein"}],
    "Connect the cable.": [{"source_content": "cable", "target_content": "câble de recharge"}],
    "Connect the cable, it is supplied.":
        [{"source_content": "cable", "target_content": "câble de recharge"}],
}

REPORT_SEGMENTS = [
    {"source_segment_id": "s1", "source_content": "The brake pad is worn.",
     "target_content": "La plaquette est usée.", "reference_content": "Le frein est usé."},
    {"source_segment_id": "s2", "source_content": "Connect the cable.",
     "target_content": "Branchez le fil.", "reference_content": "Branchez le câble de recharge."},
]


def _run(name, segments):
    dataset = Dataset(
        name=name,
        component="glossary",
        parameters=normalize_language(
            {
                "source_language": "English (United Kingdom)",
                "target_language": "French (France)",
                "domain": "Automotive",
            }
        ),
        glossary_ids=["tb1"],
        segments=[dict(s) for s in segments],
    )
    benchmark = Benchmark(
        postmt=FakePostMt(glossary=REPORT_GLOSSARY),
        stanza=FakeStanza(),
        glossary=FakeGlossary(REPORT_GLOSSARY),
        config=FakeConfig(FakeBenchmarkConfig(lemma_matching=True)),
    )
    return benchmark.run(dataset)


@pytest.fixture
def result():
    return _run("accents", REPORT_SEGMENTS)


# --- the summary and the three tables under it ---------------------------------------------------------

def test_scorecard_names_stratum(result):
    """What was measured and where; the dataset name and steps sit on the tables below."""
    line = glossary_scorecard(result).as_console().splitlines()[0]

    assert line == "glossary  -  en-gb → fr-fr  ·  Automotive"
    assert result.dataset not in line and "steps" not in line


def test_glossary_blind_run_says_so(result):
    """Blindness the probe let through, segment by segment: without this the scorecard reads as a
    pipeline given the terms and ignoring them."""
    # The healthy run says nothing of the kind, so the warning cannot be background noise.
    assert "post-mt was shown no glossary" not in glossary_scorecard(result).as_markdown()

    result.totals["segments_glossary_never_shown"] = 2
    for rendered in (glossary_scorecard(result).as_markdown(), glossary_scorecard(result).as_console()):
        assert "post-mt was shown no glossary on 2/2" in rendered
        assert "cat_project_id" in rendered


def test_summary_shows_na_not_zero(result):
    result.mt.adherence_rate = None
    result.ape.adherence_rate = None
    assert "n/a" in glossary_scorecard(result).as_markdown()


def test_summary_reports_two_columns(result):
    rendered = glossary_scorecard(result).as_markdown()

    for column in ("MT ", "APE "):
        assert column in rendered


def test_reference_metrics_stay_out_of_reports(result):
    """REF is fully adherent by construction, so scoring it says nothing worth a column."""
    scorecard = glossary_scorecard(result)

    assert result.ref.adherence_rate == 1.0
    for destination in (scorecard.as_console(), scorecard.as_markdown()):
        assert "REF 100.00%" not in destination
        assert "(REF" not in destination


def test_term_rows_report_rate_per_glossary_term(result):
    rows = {row["source_term"]: row for row in term_rows(result)}
    assert set(rows) == {"brake pad", "cable"}

    # Neither fixture translation carries its glossary target, so both terms score 0/1.
    for term in rows.values():
        assert term["segments"] == 1
        assert term["ref_rendered"] == 1
        assert term["ape_adherent"] == 0
        assert term["ape_violations"] == 1
        assert term["ape_adherence_rate"] == 0.0

    # The accented target must survive into the per-term rollup too.
    assert "câble de recharge" in rows["cable"]["expected_targets"]


def test_term_rows_are_worst_first(result):
    rows = term_rows(result)
    assert [row["source_term"] for row in rows] == ["brake pad", "cable"]


def test_per_term_table_shows_counts_and_violations(result):
    table = render_term_adherence(result)
    header = next(line for line in table.splitlines() if "Source term" in line)
    assert [column.strip() for column in header.strip("|").split("|")] == [
        "Source term", "Targets", "MT", "APE", "REF", "Violations", "Adherence", "Bucket",
        "Kind",
    ]
    # No legend under it: the columns are named in the header and nowhere else.
    assert "renderings in the human reference" not in table
    assert table.rstrip().splitlines()[-1].startswith("| cable")


def test_console_matches_file(result):
    """The terminal narrows the table; it must not restate it. Both come from `term_rows`."""
    console = render_term_adherence_console(result)

    assert "term" in console.splitlines()[2] and "REF" in console.splitlines()[2]
    for row in term_rows(result):
        line = next(l for l in console.splitlines() if l.strip().startswith(row["source_term"]))
        assert line.split() == [
            *row["source_term"].split(),
            str(row["mt_rendered"]),
            str(row["ape_rendered"]),
            str(row["ref_rendered"]),
            str(row["ape_violations"]),
            pct(row["ape_adherence_rate"]),
        ]


def test_console_table_is_worst_first_like_file(result):
    console = render_term_adherence_console(result)
    printed = [l.strip().rsplit("  ", 1)[0] for l in console.splitlines()[3:] if l.strip()]

    assert [p.split()[0] for p in printed] == [r["source_term"].split()[0] for r in term_rows(result)]


def test_run_that_matched_no_terms_says_so_on_console(result):
    """Zeroes with no table underneath read as a clean run rather than one that measured nothing."""
    result.segments = []

    assert "No glossary terms matched." in render_term_adherence_console(result)


def test_term_rows_expose_raw_count(result):
    """The version count is reported unbounded, so over-rendering survives the cap that folds it."""
    for row in term_rows(result):
        assert row["ape_rendered"] == 0                    # neither fixture target was used
        assert row["ape_adherent"] == min(row["ape_rendered"], row["ref_rendered"])


@pytest.mark.parametrize("source, target, reference, bucket", [
    ("Connect the cable.", "Branchez le fil.",
     "Branchez le câble de recharge.", "never"),
    ("Connect the cable, it is supplied.", "Branchez le câble de recharge, il est fourni.",
     "Branchez le câble de recharge, le câble de recharge est fourni.", "partly"),
    ("Connect the cable.", "Branchez le câble de recharge.",
     "Branchez le câble de recharge.", "matched"),
    ("Connect the cable, it is supplied.",
     "Branchez le câble de recharge, le câble de recharge est fourni.",
     "Branchez le câble de recharge, il est fourni.", "over-used"),
])
def test_per_term_table_names_each_bucket(source, target, reference, bucket):
    """The scorecard counts the four buckets; the table says which term is in which."""
    result = _run(bucket, [{"source_segment_id": "s1", "source_content": source,
                            "target_content": target, "reference_content": reference}])

    row, = term_rows(result)
    assert row["ape_bucket"] == bucket
    assert f"| {bucket} |" in render_term_adherence(result)


def test_over_use_is_flagged_for_review():
    """Over-use is adherent by the cap, so its bucket is the only place it can surface."""
    result = _run("over-used", [
        {
            "source_segment_id": "s1",
            "source_content": "Connect the cable, it is supplied.",
            "target_content": "Branchez le câble de recharge, le câble de recharge est fourni.",
            "reference_content": "Branchez le câble de recharge, il est fourni.",
        }
    ])

    row, = term_rows(result)
    assert row["ref_rendered"] == 1
    assert row["ape_rendered"] == 2
    assert row["ape_violations"] == 0
    assert row["ape_adherence_rate"] == 1.0
    assert row["ape_bucket"] == "over-used"
    assert "flagged for review, never counted as violations" in glossary_scorecard(result).as_markdown()


def test_per_term_adherence_is_only_rate(result):
    """Adherence is a share of that term's own REF count; its misses stay counts."""
    for row in term_rows(result):
        assert "ape_violation_rate" not in row
        if row["ref_rendered"]:
            assert row["ape_adherence_rate"] == row["ape_adherent"] / row["ref_rendered"]


def test_per_term_violations_reconcile(result):
    """The per-term column sums to the dataset's violation count, as the table exists to allow."""
    rows = term_rows(result)
    assert sum(row["ape_violations"] for row in rows) == result.ape.violations
    for row in rows:
        assert row["ape_violations"] == row["ref_rendered"] - row["ape_adherent"]


def test_three_levels_reconcile(result):
    """Per-term counts must sum to the dataset total the stratum row pools from."""
    rows = term_rows(result)
    assert sum(row["ref_rendered"] for row in rows) == result.mt.expected
    assert sum(row["mt_adherent"] for row in rows) == result.mt.adherent


# --- strata: one figure per language pair × domain -------------------------------------------------------
# These drive the pooling directly, because what matters is how counts combine.

def _aggregate(expected, adherent, *, segments=1, fully_adherent=1, terms=None):
    """An Aggregate carrying only the counts pooling cares about; pass a TermBreakdown to vary."""
    breakdown = terms or TermBreakdown(
        used_everywhere=adherent, never_used=expected - adherent
    )
    missed = expected - adherent
    return Aggregate(
        expected=expected,
        adherent=adherent,
        violations=missed,
        adherence_rate=None if expected == 0 else adherent / expected,
        strict=TallyReport(expected, adherent, missed, None),
        permissive=TallyReport(0, 0, 0, None),
        terms=breakdown,
        segments_with_glossary=segments,
        segments_fully_adherent=fully_adherent,
        segment_adherence_rate=None,
    )


class _Result:
    """The few BenchmarkResult fields the stratum code reads."""

    def __init__(self, name, source, target, domain, mt, ape=None, segments=1, violations=None):
        self.dataset = name
        self.parameters = {"source_language": source, "target_language": target, "domain": domain}
        self.mt = mt
        self.ape = ape if ape is not None else mt
        self.totals = {"segments": segments}
        self.started_at = "2026-01-01T00:00:00+00:00"
        self.mt_violations = violations or ViolationReport(segments=segments)
        self.ape_violations = violations or ViolationReport(segments=segments)


def test_pooling_sums_counts_rather_than_averaging_rates():
    """Averaging the rates would give 75%; the evidence says 50.5%."""
    pooled = pool([_aggregate(2, 2), _aggregate(200, 100)])

    assert pooled.expected == 202
    assert pooled.adherent == 102
    assert pooled.adherence_rate == pytest.approx(102 / 202)
    assert pooled.adherence_rate != pytest.approx((1.0 + 0.5) / 2)


def test_pooling_is_split_invariant():
    """Splitting a dataset and pooling must reproduce the unsplit number."""
    whole = _aggregate(8, 5)
    split = pool([_aggregate(6, 4), _aggregate(2, 1)])

    assert (split.expected, split.adherent) == (whole.expected, whole.adherent)
    assert split.adherence_rate == whole.adherence_rate


def test_pooling_keeps_violations_and_segment_counts():
    pooled = pool([_aggregate(4, 3, segments=2, fully_adherent=1),
                   _aggregate(6, 6, segments=3, fully_adherent=3)])

    assert pooled.violations == 1
    assert pooled.segments_with_glossary == 5
    assert pooled.segments_fully_adherent == 4


def test_empty_stratum_reports_no_rate_rather_than_zero():
    """`None`, never 0: no evidence is not total failure."""
    pooled = pool([_aggregate(0, 0, segments=0, fully_adherent=0)])

    assert pooled.expected == 0
    assert pooled.adherence_rate is None


def test_stratum_is_pair_and_domain():
    result = _Result("d", "en-gb", "fr-fr", "Automotive", _aggregate(1, 1))

    assert stratum_of(result) == ("en-gb->fr-fr", "Automotive")


def test_same_pair_different_domain_are_different_strata():
    a = _Result("a", "en-gb", "fr-fr", "Automotive", _aggregate(4, 2))
    b = _Result("b", "en-gb", "fr-fr", "Forestry", _aggregate(4, 4))

    assert len(by_stratum([a, b])) == 2


def test_one_stratum_pools_into_one_row():
    a = _Result("a", "en-gb", "fr-fr", "Automotive", _aggregate(6, 4), segments=2)
    b = _Result("b", "en-gb", "fr-fr", "Automotive", _aggregate(2, 1), segments=1)

    rows = stratum_rows([a, b])

    assert len(rows) == 1
    assert rows[0]["datasets"] == 2
    assert rows[0]["segments"] == 3
    assert rows[0]["expected_instances"] == 8
    assert rows[0]["mt_adherence_rate"] == pytest.approx(5 / 8)


def test_missing_domain_still_forms_stratum():
    result = _Result("d", "en-gb", "fr-fr", None, _aggregate(2, 1))

    assert stratum_of(result) == ("en-gb->fr-fr", "(no domain)")
    assert len(stratum_rows([result])) == 1


def test_render_shows_instance_count_next_to_rate():
    """A 100% stratum resting on 2 instances and one resting on 400 must not look identical."""
    rendered = render_strata([
        _Result("d", "en-gb", "fr-fr", "Automotive", _aggregate(2, 2)),
        _Result("e", "en-gb", "de-de", "Forestry", _aggregate(400, 400)),
    ])

    assert "en-gb->fr-fr" in rendered
    assert "Automotive" in rendered
    assert "100.00%" in rendered
    assert "2 inst" in rendered
    assert "400 inst" in rendered


def test_single_dataset_still_renders_stratum():
    """A stratum figure must not depend on how many files happened to be in the folder."""
    rendered = render_strata([_Result("d", "en-gb", "fr-fr", "Automotive", _aggregate(2, 2))])
    assert "en-gb->fr-fr · Automotive · 2 inst" in rendered
    # No ALL row: pooling one stratum into itself would just repeat the line above.
    assert "ALL ·" not in rendered


# ================================================================================================
# orchestration
# ================================================================================================

class FakeStanza:
    """Identity lemmatizer: keeps the tests off the network without changing what is matched."""

    def lemmatize_batch_safe(self, texts, language):
        return list(texts)


class FakeGlossary:
    """The term-bases index, serving a fixed source-text -> mappings table."""

    def __init__(self, table):
        self.table = table

    def fetch_matches(self, *, glossary_ids, source_language, target_language, texts, provider=None):
        per_text = [self.table.get(text, []) for text in texts]
        return GlossaryMatches(
            mappings=[m for group in per_text for m in group], per_text_mappings=per_text
        )


class FakePostMt:
    """Returns the MT unchanged except where `fixes` names a source text; `submitted` records it."""

    def __init__(self, fixes=None, glossary=None):
        self.fixes = fixes or {}
        self.glossary = glossary or {}
        self.batches_seen = []
        self.submitted = []

    def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
        self.batches_seen.append([segment.get("source_segment_id") for segment in segments])
        self.submitted.extend(segments)
        return RunResult(
            task_id="stub",
            error=None,
            segments=[
                {
                    **segment,
                    "has_glossary": bool(self.glossary.get(segment["source_content"])),
                    "ape_results": {
                        "text": self.fixes.get(segment["source_content"], segment["target_content"])
                    },
                }
                for segment in segments
            ],
        )


@dataclass
class FakeBenchmarkConfig:
    batch_size: int = 10
    lemma_matching: bool = False


@dataclass
class FakeConfig:
    benchmark: FakeBenchmarkConfig = field(default_factory=FakeBenchmarkConfig)

# "brake pad" -> frein (strict); "engine" -> moteur (strict); "battery" -> two variants (permissive).
GLOSSARY = {
    "The brake pad is worn.": [{"source_content": "brake pad", "target_content": "frein"}],
    "The engine is electric.": [{"source_content": "engine", "target_content": "moteur"}],
    "Charge the battery.": [
        {"source_content": "battery", "target_content": "batterie"},
        {"source_content": "battery", "target_content": "accumulateur"},
    ],
    "No glossary terms here.": [],
}

# APE fixes the engine segment and leaves everything else as the MT wrote it.
FIXES = {"The engine is electric.": "Le moteur est électrique."}

SEGMENTS = [
    {"source_segment_id": "s1", "source_content": "The brake pad is worn.",
     "target_content": "La plaquette est usée.", "reference_content": "Le frein est usé."},
    {"source_segment_id": "s2", "source_content": "The engine is electric.",
     "target_content": "Le bloc est électrique.", "reference_content": "Le moteur est électrique."},
    {"source_segment_id": "s3", "source_content": "No glossary terms here.",
     "target_content": "Aucun terme ici.", "reference_content": "Aucun terme ici."},
]

# The MT gets both terms wrong and the human got both right, so the reference sets the denominator.
REFERENCE_SEGMENTS = [
    {"source_segment_id": "s1", "source_content": "The engine is electric.",
     "target_content": "Le bloc est électrique.",          # MT: wrong term
     "reference_content": "Le moteur est électrique."},    # human: correct
    {"source_segment_id": "s2", "source_content": "Charge the battery.",
     "target_content": "Chargez la pile.",                 # MT: wrong term
     "reference_content": "Chargez la batterie."},         # human: correct
]


def _dataset(name, segments):
    return Dataset(
        name=name,
        component="glossary",
        parameters=normalize_language(
            {
                "source_language": "English (United Kingdom)",
                "target_language": "French (France)",
                "domain": "Automotive",
                "cat_tool_provider": "MemSource",
            }
        ),
        glossary_ids=["tb1"],
        segments=[dict(s) for s in segments],
    )


def _benchmark(postmt=None, *, batch_size=2):
    return Benchmark(
        postmt=postmt if postmt is not None else FakePostMt(fixes=FIXES, glossary=GLOSSARY),
        stanza=FakeStanza(),
        glossary=FakeGlossary(GLOSSARY),
        config=FakeConfig(FakeBenchmarkConfig(batch_size=batch_size)),
    )


@pytest.fixture
def dataset():
    return _dataset("wiring", SEGMENTS)


@pytest.fixture
def benchmark():
    return _benchmark()


# --- end to end --------------------------------------------------------------------------------------

def test_end_to_end_scores_mt_baseline_against_post_edited(benchmark, dataset):
    result = benchmark.run(dataset)

    assert result.totals["segments"] == 3
    assert result.totals["segments_with_glossary"] == 2

    # MT baseline: both terms missing ("plaquette" not "frein", "bloc" not "moteur").
    assert result.mt.expected == 2
    assert result.mt.adherent == 0
    assert result.mt.violations == 2   # a count, not a rate

    # Post-edited: APE fixed one of the two.
    assert result.ape.adherent == 1
    assert result.ape.adherence_rate == 0.5

    assert result.delta.adherence_rate == 0.5
    assert result.delta.terms_fixed_by_ape == 1


def test_segment_without_glossary_does_not_dilute_rate(benchmark, dataset):
    result = benchmark.run(dataset)
    clean = next(s for s in result.segments if s.source_segment_id == "s3")
    assert clean.mt.expected == 0
    assert clean.has_glossary_resolved is False


def test_blind_run_is_refused_before_pipeline(dataset):
    """The probe costs one segment; continuing costs the whole dataset and measures nothing."""
    postmt = FakePostMt(fixes=FIXES, glossary={})

    with pytest.raises(RuntimeError, match="cat_project_id"):
        _benchmark(postmt).run(dataset)

    # Only the probe was ever submitted, so the dataset itself was never billed.
    assert postmt.batches_seen == [["s1"]]


def test_dry_run_never_probes_postmt(dataset):
    postmt = FakePostMt(fixes=FIXES, glossary={})
    _benchmark(postmt).run(dataset, skip_pipeline=True)

    assert postmt.batches_seen == []


def test_no_resolved_terms_skips_probe(dataset):
    """Nothing resolved means nothing to be blind to, and the probe would answer False regardless."""
    postmt = FakePostMt(fixes=FIXES, glossary=GLOSSARY)
    dataset.segments = [dict(SEGMENTS[2])]

    _benchmark(postmt).run(dataset)

    assert postmt.batches_seen == [["s3"]]


def test_batching_preserves_order_and_index_alignment(dataset):
    postmt = FakePostMt(fixes=FIXES, glossary=GLOSSARY)
    result = _benchmark(postmt).run(dataset)

    # batch_size 2 over 3 segments => two batches, behind the one-segment retrieval probe;
    # misalignment would misattribute terms.
    assert postmt.batches_seen == [["s1"], ["s1", "s2"], ["s3"]]
    assert [s.source_segment_id for s in result.segments] == ["s1", "s2", "s3"]

    first = result.segments[0]
    assert first.glossary_terms == [{"source_content": "brake pad", "target_content": "frein"}]
    assert [v.source_content for v in first.ape.violations] == ["brake pad"]

    second = result.segments[1]
    assert second.changed_by_ape is True
    assert second.ape.adherent == 1
    assert second.ape.violations == []


def test_repairs_and_regressions_are_counted_separately(dataset):
    # One term repaired, one broken: net delta 0, which a single number would present as "no change".
    dataset.segments = [
        {"source_segment_id": "s1", "source_content": "The brake pad is worn.",
         "target_content": "Le frein est usé.", "reference_content": "Le frein est usé."},
        {"source_segment_id": "s2", "source_content": "The engine is electric.",
         "target_content": "Le bloc est électrique.",
         "reference_content": "Le moteur est électrique."},
    ]

    postmt = FakePostMt(
        glossary=GLOSSARY,
        fixes={
            "The engine is electric.": "Le moteur est électrique.",   # repaired
            "The brake pad is worn.": "La garniture est usée.",        # regressed away from "frein"
        },
    )
    result = _benchmark(postmt).run(dataset)

    assert result.mt.adherent == 1
    assert result.ape.adherent == 1
    assert result.delta.adherence_rate == 0          # net change: nothing

    # ...but one term was repaired and a different one was broken.
    assert result.delta.terms_fixed_by_ape == 1
    assert result.delta.terms_regressed_by_ape == 1


def test_postmts_has_glossary_is_retained(benchmark, dataset):
    result = benchmark.run(dataset)
    for segment in result.segments:
        assert segment.has_glossary_reported == segment.has_glossary_resolved


def test_short_pipeline_response_realigns(dataset):
    class TruncatingPostMt(FakePostMt):
        def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
            full = super().run(parameters=parameters, segments=segments, steps=steps)
            return RunResult(task_id=full.task_id, segments=full.segments[:1], error=None)

    result = _benchmark(TruncatingPostMt(fixes=FIXES, glossary=GLOSSARY)).run(dataset)

    # Dropped segments fall back to their inputs rather than shifting every later index.
    assert [s.source_segment_id for s in result.segments] == ["s1", "s2", "s3"]
    assert result.totals["segments"] == 3


# --- the human reference is the denominator ----------------------------------------------------------

@pytest.fixture
def reference_dataset():
    return _dataset("against-the-reference", REFERENCE_SEGMENTS)


def test_reference_is_never_sent_to_postmt(reference_dataset):
    """The corrected translation is the answer key — it must not reach the pipeline."""
    postmt = FakePostMt(fixes=FIXES, glossary=GLOSSARY)
    _benchmark(postmt).run(reference_dataset)

    assert postmt.submitted, "nothing was submitted"
    for submitted in postmt.submitted:
        assert "reference_content" not in submitted
    # ...while the fields production does receive are intact.
    assert {s["source_content"] for s in postmt.submitted} == {
        "The engine is electric.", "Charge the battery."
    }
    assert all(s.get("target_content") for s in postmt.submitted)


def test_reference_sets_what_each_version_owed(reference_dataset):
    result = _benchmark().run(reference_dataset)

    # The human used one target term per segment, so two instances are owed.
    assert result.mt.expected == 2
    assert result.ape.expected == 2

    # MT reproduced neither; APE reproduced one.
    assert result.mt.adherent == 0
    assert result.ape.adherent == 1
    assert result.ape.adherence_rate == 0.5
    assert result.delta.adherence_rate == pytest.approx(0.5)


def test_avoided_term_is_not_held_against_ape():
    """The human declined the entry, so applying it is over-use, reported and never scored."""
    dataset = _dataset("human-avoided", [
        {"source_segment_id": "s1", "source_content": "The engine is electric.",
         "target_content": "Le bloc est électrique.",
         "reference_content": "Le groupe est électrique."},
    ])
    result = _benchmark().run(dataset)

    assert result.mt.expected == 0
    # No denominator anywhere, so there is no rate to report rather than a 0% or a 100%.
    assert result.mt.adherence_rate is None
    assert result.ape.adherence_rate is None
    assert result.ape.terms.over_used == 1
    assert result.ape.violations == 0


# --- skip_pipeline: dry run and full run share one code path -------------------------------------------

def test_skip_pipeline_never_contacts_postmt(dataset):
    class ExplodingPostMt:
        def run(self, **kwargs):
            raise AssertionError("post-mt must not be called when skip_pipeline=True")

    result = _benchmark(ExplodingPostMt()).run(dataset, skip_pipeline=True)

    assert result.dataset.endswith("(dry-run)")


def test_skip_pipeline_scores_same_way_as_full_run(dataset):
    """A dry run and a full run must not disagree about a number."""
    # APE returns the MT unchanged, which is what a dry run assumes.
    full = _benchmark(FakePostMt(fixes={}, glossary=GLOSSARY)).run(dataset)
    dry = _benchmark().run(dataset, skip_pipeline=True)

    assert dry.mt.expected == full.mt.expected
    assert dry.mt.adherent == full.mt.adherent
    assert dry.mt.adherence_rate == full.mt.adherence_rate
    # Post-edited mirrors the baseline when nothing was post-edited.
    assert dry.ape.adherent == dry.mt.adherent
    assert dry.totals["segments_changed_by_ape"] == 0


def test_skip_pipeline_still_measures_against_reference(dataset):
    """The denominator comes from the dataset, not from post-mt, so a dry run has the full one."""
    dry = _benchmark().run(dataset, skip_pipeline=True)

    assert dry.mt.expected == 2
    assert [s.ref_text for s in dry.segments] == [s["reference_content"] for s in SEGMENTS]


# --- a misconfigured or failed run must not read as a clean measurement ---------------------------------

def test_warns_when_postmt_saw_no_glossary(dataset, caplog):
    """A misconfigured run reads as "APE does not help terminology" unless it is called out."""
    class BlindAfterProbePostMt(FakePostMt):
        """Retrieval fires for the probe and then stops - the case the probe cannot catch."""

        def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
            result = super().run(parameters=parameters, segments=segments, steps=steps)
            if len(self.batches_seen) > 1:
                for segment in result.segments:
                    segment["has_glossary"] = False
            return result

    _benchmark(BlindAfterProbePostMt(fixes=FIXES, glossary=GLOSSARY)).run(dataset)

    assert "post-mt reported no glossary" in caplog.text
    assert "ecosystem_id" in caplog.text


def test_no_warning_when_two_agree(dataset, caplog):
    _benchmark().run(dataset)
    assert "post-mt reported no glossary" not in caplog.text


class FailingPostMt(FakePostMt):
    """A missing required parameter: status done, task error None, the error on each segment."""

    def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
        enriched = [
            {
                **segment,
                "has_glossary": False,
                "ape_results": {
                    "text": "",
                    "error": "Missing required parameters fields: tempo_task_id",
                },
            }
            for segment in segments
        ]
        return RunResult(task_id="stub", segments=enriched, error=None)


def test_per_segment_failures_are_surfaced(dataset, caplog):
    """The empty APE text falls back to raw MT, so every metric says "APE changed nothing"."""
    result = _benchmark(FailingPostMt()).run(dataset)

    # On the numbers alone this is indistinguishable from a clean run...
    assert result.totals["segments_changed_by_ape"] == 0
    assert result.delta.adherence_rate == 0.0

    # ...so only the failure count tells them apart.
    assert result.failed_segments == 3
    assert "tempo_task_id" in result.failure_reason
    assert "tempo_task_id" in caplog.text


def test_healthy_run_records_no_failures(benchmark, dataset):
    assert benchmark.run(dataset).failed_segments == 0
    assert benchmark.run(dataset).failure_reason is None


def test_dry_run_never_reports_post_mt_failures(benchmark, dataset):
    """A dry run does not call post-mt at all, so it cannot inherit a stale failure count."""
    assert benchmark.run(dataset, skip_pipeline=True).failed_segments == 0
