"""The metric definition: instance counting, strict vs permissive, and aggregation."""

from sourcecode.score import OMITTED, SUBSTITUTED, aggregate, build_glossary_map, score_translation

MAPPINGS = [
    {"source_content": "brake", "target_content": "frein"},
    {"source_content": "brake", "target_content": "freinage"},   # permissive: two target terms
    {"source_content": "engine", "target_content": "moteur"},    # strict: one target term
]

#: What the human wrote, and therefore what every version below is measured against.
REFERENCE = "le frein du moteur"


def test_glossary_map_groups_targets_by_source():
    # Deliberately mirrors how post-mt phrases the prompt, so we score the contract the model
    # was actually shown.
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


def test_permissive_term_is_adherent_when_any_variant_appears():
    score = score_translation(
        mappings=MAPPINGS, text="le freinage du moteur", language_code="fr-fr",
        reference_text="le freinage du moteur",
    )
    assert (score.expected, score.adherent) == (2, 2)
    assert score.violations == []
    assert score.permissive.adherent == 1
    assert score.strict.adherent == 1


def test_missing_strict_term_is_recorded_with_expected_renderings():
    score = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 1)
    assert len(score.violations) == 1

    violation = score.violations[0]
    assert violation.source_content == "engine"
    assert violation.expected_targets == ["moteur"]
    assert violation.strictness == "strict"


def test_expected_instances_count_per_term_not_per_segment():
    # Two terms in one reference must contribute 2 instances — otherwise the metric tracks
    # segment length rather than terminology discipline.
    score = score_translation(
        mappings=MAPPINGS, text="rien ici", language_code="fr-fr", reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 0)
    assert {v.strictness for v in score.violations} == {"strict", "permissive"}


def test_hits_record_which_variant_matched():
    score = score_translation(
        mappings=MAPPINGS, text="le freinage du moteur", language_code="fr-fr",
        reference_text="le freinage du moteur",
    )
    matched = {hit.source_content: hit.matched_target for hit in score.hits}
    assert matched == {"brake": "freinage", "engine": "moteur"}


def test_segments_without_glossary_terms_contribute_nothing():
    score = score_translation(
        mappings=[], text="whatever", language_code="fr-fr", reference_text="peu importe",
    )
    assert score.expected == 0

    agg = aggregate([score])
    assert agg.expected == 0
    # None, never 0 — no evidence is not zero adherence, and a 0 would drag rollups down.
    assert agg.adherence_rate is None
    assert agg.segments_with_glossary == 0


def test_aggregate_computes_instance_and_segment_level_rates():
    clean = score_translation(                                                                # 2/2
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    partial = score_translation(                                                              # 1/2
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        reference_text=REFERENCE,
    )

    agg = aggregate([clean, partial])

    assert (agg.expected, agg.adherent, agg.violations) == (4, 3, 1)
    assert agg.adherence_rate == 0.75
    assert agg.misses.substituted == 1

    # Segment level: only one of the two segments is fully clean.
    assert agg.segments_with_glossary == 2
    assert agg.segments_fully_adherent == 1
    assert agg.segment_adherence_rate == 0.5

    # The strict slice is the one failing here.
    assert agg.strict.adherence_rate == 0.5
    assert agg.permissive.adherence_rate == 1.0


def test_aggregate_ignores_empty_scores_in_segment_denominator():
    scored = score_translation(
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    empty = score_translation(
        mappings=[], text="nothing", language_code="fr-fr", reference_text="rien",
    )

    agg = aggregate([scored, empty, empty])

    # Three segments in, but only one had terms to judge.
    assert agg.segments_with_glossary == 1
    assert agg.segment_adherence_rate == 1.0


def test_lemma_matching_is_used_when_lemmas_are_supplied():
    mappings = [{"source_content": "engine", "target_content": "moteur electrique"}]

    without = score_translation(
        mappings=mappings, text="les moteurs electriques sont bons", language_code="fr-fr",
        reference_text="le moteur electrique est bon",
    )
    assert without.expected == 1
    assert without.adherent == 0

    with_lemmas = score_translation(
        mappings=mappings,
        text="les moteurs electriques sont bons",
        language_code="fr-fr",
        text_lemmas="le moteur electrique etre bon",
        reference_text="le moteur electrique est bon",
        term_lemmas={"moteur electrique": "moteur electrique"},
    )
    assert with_lemmas.adherent == 1
    assert with_lemmas.hits[0].via == "lemma"


# ------------------------------------------------------- why an instance was missed


def test_a_miss_in_a_rendered_segment_is_a_substitution():
    """The ordinary failure: the segment was translated, just not with the target term."""
    score = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    assert (score.misses.substituted, score.misses.omitted) == (1, 0)
    assert score.violations[0].miss_kind == SUBSTITUTED

    agg = aggregate([score])
    assert (agg.misses.substituted, agg.misses.omitted) == (1, 0)
    assert agg.adherence_rate == 0.5


def test_a_miss_with_no_translation_at_all_is_an_omission():
    """An empty segment has no wording to hold against the glossary, so it is not a terminology
    failure. Adherence still reads 0 — the reference demanded two renderings and got none — but
    the violation count stays empty, because there is no wrong wording to go and fix."""
    score = score_translation(
        mappings=MAPPINGS, text="", language_code="fr-fr", reference_text=REFERENCE,
    )
    assert (score.expected, score.adherent) == (2, 0)
    assert (score.misses.substituted, score.misses.omitted) == (0, 2)
    assert {v.miss_kind for v in score.violations} == {OMITTED}

    agg = aggregate([score])
    assert agg.adherence_rate == 0.0
    assert (agg.misses.substituted, agg.misses.omitted) == (0, 2)


def test_whitespace_only_output_counts_as_omitted_not_substituted():
    score = score_translation(
        mappings=MAPPINGS, text="      ", language_code="fr-fr", reference_text=REFERENCE,
    )
    assert score.misses.omitted == 2


def test_the_miss_counts_partition_the_shortfall():
    substituted = score_translation(
        mappings=MAPPINGS, text="le frein du bloc", language_code="fr-fr",
        reference_text=REFERENCE,
    )
    omitted = score_translation(
        mappings=MAPPINGS, text="", language_code="fr-fr", reference_text=REFERENCE,
    )
    clean = score_translation(
        mappings=MAPPINGS, text="le frein du moteur", language_code="fr-fr",
        reference_text=REFERENCE,
    )

    agg = aggregate([substituted, omitted, clean])

    # The split is exhaustive over the misses, so it reconciles with the headline counts.
    assert agg.misses.total == agg.violations == agg.expected - agg.adherent
    assert (agg.misses.substituted, agg.misses.omitted) == (1, 2)
    # Adherence is the only rate, and it is a share of what the reference demanded.
    assert agg.adherence_rate == agg.adherent / agg.expected

    # And it pools per slice as well as overall.
    assert agg.strict.misses.substituted + agg.permissive.misses.substituted == 1
    assert agg.strict.misses.omitted + agg.permissive.misses.omitted == 2
