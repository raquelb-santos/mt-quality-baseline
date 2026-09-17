"""The tag integrity component: the extractor, the metric, its rendering, and the orchestration."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sourcecode import report, run, tags_report
from sourcecode.postmt import RunResult
from sourcecode.tags import ENTITY, PAIRED_CLOSE, PAIRED_OPEN, STANDALONE, XML, extract_tags, unpaired
from sourcecode.tags_benchmark import run_benchmark
from sourcecode.tags_score import aggregate, pool, score_tags
from sourcecode.text_processing import Dataset, Task, normalize_language


CONFIG = SimpleNamespace(benchmark=SimpleNamespace(batch_size=50))


def score(text, *, source):
    return score_tags(src_text=source, text=text)


def kinds(text):
    return [(tag.text, tag.kind) for tag in extract_tags(text)]


class FakePostMt:
    """post-mt returning the MT unchanged except where `fixes` names a source text to replace."""

    def __init__(self, fixes=None):
        self.fixes = fixes or {}

    def run(self, *, segments, **_):
        return RunResult(
            task_id="stub",
            error=None,
            segments=[
                {
                    **segment,
                    "ape_results": {
                        "text": self.fixes.get(segment["source_content"], segment["target_content"])
                    },
                }
                for segment in segments
            ],
        )


SEGMENTS = [
    # Every tag comes through.
    {"source_segment_id": "s1", "source_content": "{b>Tooling<b}",
     "target_content": "{b>Outillage<b}", "reference_content": "{b>Outillage<b}"},
    # The closer is gone from the MT, and the human kept it.
    {"source_segment_id": "s2", "source_content": "Work{b> only <b}by qualified staff.",
     "target_content": "Intervention{b> uniquement par personnel qualifié.",
     "reference_content": "Intervention{b> uniquement <b}par personnel qualifié."},
    # `{1}` is renumbered to `{2}`, which is one drop and one hallucination.
    {"source_segment_id": "s3", "source_content": "{1}MOULDS WITH 1 CIRCUIT",
     "target_content": "{2}MOULES AVEC 1 CIRCUIT",
     "reference_content": "{1}MOULES AVEC 1 CIRCUIT"},
    # Nothing is lost, but the two pairs come back the other way round.
    {"source_segment_id": "s4", "source_content": "{1>A<1} and {2>B<2}",
     "target_content": "{2>B<2} et {1>A<1}", "reference_content": "{1>A<1} et {2>B<2}"},
]


def _dataset(name="tags-set", domain="Test", segments=SEGMENTS):
    parameters = normalize_language(
        {"source_language": "en-gb", "target_language": "fr-fr", "domain": domain}
    )
    return Dataset(
        name=name,
        component="tags",
        parameters=parameters,
        tasks=[Task(parameters, list(segments))],
    )


def _result(name="tags-set", domain="Test", segments=SEGMENTS):
    return run_benchmark(
        _dataset(name, domain, segments), postmt=None, config=CONFIG, skip_pipeline=True
    )


# extraction - the shapes the CAT tools actually emit, in the form that reaches the pipeline


def test_phrase_pair_is_read_as_two_tags():
    """`{b>`/`<b}` is the dominant notation; missing it leaves the majority of tags unmeasured."""
    assert kinds("{b>Outillage<b}") == [("{b>", PAIRED_OPEN), ("<b}", PAIRED_CLOSE)]


def test_closer_does_not_swallow_the_translation():
    """A decoded `<b}` read as an XML opener consumes running text up to the next `>`."""
    text = "Intervention{b> uniquement <b}réalisable par personnel qualifié -> voir la notice."
    assert [tag.text for tag in extract_tags(text)] == ["{b>", "<b}"]


def test_standalone_tag_is_found():
    assert kinds("{1}MOULES AVEC 1 CIRCUIT") == [("{1}", STANDALONE)]


def test_suffixed_ids_are_ids():
    """`^` falls outside `\\w`, so an id class built on `\\w` silently drops `{b^>`."""
    assert [tag.tag_id for tag in extract_tags("{b^>A<b^} et {b_>B<b_}")] == ["b^", "b^", "b_", "b_"]


def test_xml_placeholder_and_html_passthrough_are_tags():
    assert kinds('<ph x="1"/>We are delighted') == [('<ph x="1"/>', XML)]
    assert kinds('<span style="font-weight: 400;">a</span>') == [
        ('<span style="font-weight: 400;">', XML), ("</span>", XML)
    ]


@pytest.mark.parametrize("text, expected", [
    ("Hello %s", ["%s"]),
    ("Hello %1$s and %2$s", ["%1$s", "%2$s"]),
    ("Total %.2f", ["%.2f"]),
])
def test_printf_placeholders_are_tags(text, expected):
    assert [tag.text for tag in extract_tags(text)] == expected


def test_bare_percent_is_not_a_placeholder():
    """`100% of them` would otherwise register `% o` and charge a drop for ordinary prose."""
    assert extract_tags("100% of them, 50% off") == []


def test_double_braces_are_taken_whole():
    """Split into `{` + `{var}` + `}`, a preserved variable would read as two broken tags."""
    assert [tag.text for tag in extract_tags("{{brand}} ships")] == ["{{brand}}"]


def test_entities_are_placeholders():
    assert kinds("&brand; and &#160;") == [("&brand;", ENTITY), ("&#160;", ENTITY)]


def test_tag_identity_is_the_id_not_the_position():
    assert [tag.tag_id for tag in extract_tags("{1>A<1}")] == ["1", "1"]


def test_attribute_whitespace_is_collapsed_before_comparison():
    """Reflowed attributes are the same tag; only the id the CAT tool re-imports by matters."""
    assert extract_tags('<span  style="a" >')[0].text == '<span style="a" >'


# pairing - well-formedness, which is what re-import actually needs


def test_matched_pair_is_well_formed():
    assert unpaired(extract_tags("{1>A<1}")) == set()


def test_opener_without_closer_is_unpaired():
    assert unpaired(extract_tags("{1>A")) == {"phrase:1"}


def test_closer_before_its_opener_is_unpaired():
    assert unpaired(extract_tags("<1}A{1>")) == {"phrase:1"}


def test_renumbered_pair_breaks_both_ids():
    assert unpaired(extract_tags("{1>A<2}")) == {"phrase:1", "phrase:2"}


def test_html_pairs_are_checked_too():
    assert unpaired(extract_tags("<span>A")) == {"xml:span"}


def test_void_and_self_closing_elements_owe_no_closer():
    """`<br>` and `<ph x="1"/>` never take one, so counting them as openers invents defects."""
    assert unpaired(extract_tags('a <br> b <ph x="1"/> c')) == set()


# the metric - the source is the answer key


def test_intact_tags_score_full_integrity():
    result = score("{b>Outillage<b}", source="{b>Tooling<b}")
    assert (result.expected, result.present, result.dropped) == (2, 2, 0)
    assert result.defects == 0


def test_dropped_tag_is_counted_against_the_source():
    result = score("Intervention{b> uniquement", source="Work{b> only <b}here")
    assert (result.expected, result.present, result.errors.dropped) == (2, 1, 1)


def test_duplicated_tag_is_an_error_not_a_bonus():
    """The cap keeps a repeated tag out of `present`, so it can never raise the rate."""
    result = score("{1>A<1} {1>B<1}", source="{1>A<1}")
    assert (result.present, result.expected) == (2, 2)
    assert result.errors.duplicated == 2


def test_renumbering_is_a_drop_and_a_hallucination():
    """The id is the whole payload, so `{1}` answered with `{2}` loses one and invents another."""
    result = score("{2}MOULES", source="{1}MOULDS")
    assert (result.errors.dropped, result.errors.hallucinated) == (1, 1)


def test_hallucinated_tag_in_a_tag_free_source_is_still_caught():
    """No denominator, but a tag the source never had still breaks re-import."""
    result = score("{1}Bonjour", source="Hello")
    assert (result.expected, result.errors.hallucinated) == (0, 1)
    assert result.defects == 1


def test_reordering_is_flagged_though_nothing_is_lost():
    result = score("{2>B<2} et {1>A<1}", source="{1>A<1} and {2>B<2}")
    assert result.expected == result.present
    assert result.mis_ordered is True


def test_order_is_compared_on_the_tags_that_survived():
    """A dropped tag must not read as a reordering as well; the two are separate failures."""
    result = score("{1>A<1}", source="{1>A<1} and {2>B<2}")
    assert result.mis_ordered is False
    assert result.errors.dropped == 2


def test_tags_are_case_sensitive():
    """`{B>` is a different id than `{b>`, so casefolding would hide a renumbering."""
    result = score("{B>A<B}", source="{b>A<b}")
    assert (result.errors.dropped, result.errors.hallucinated) == (2, 2)


def test_source_arriving_broken_is_not_charged_to_the_version():
    result = score("{1>A", source="{1>A")
    assert (result.unpaired, result.source_unpaired) == (0, 1)


def test_version_breaking_a_sound_pair_is_charged():
    result = score("{1>A", source="{1>A<1}")
    assert result.unpaired == 1


def test_each_source_tag_lands_in_exactly_one_bucket():
    result = score("{1>A<1} {1>B<1} {3>", source="{1>A<1} {2>B<2} {3>C")
    buckets = result.tags
    assert buckets.distinct_tags == sum(1 for tag in result.tag_scores if tag.expected)


def test_error_buckets_are_exhaustive():
    result = score("{2}A {1>B", source="{1}A {1>B<1} {1>C<1}")
    breakdown = result.errors
    assert breakdown.total == breakdown.dropped + breakdown.duplicated + breakdown.hallucinated


# aggregation and pooling


def test_aggregation_sums_every_direction():
    scores = [score("{1}", source="{1}"), score("{2}", source="{1}")]
    total = aggregate(scores)
    assert (total.expected, total.present) == (2, 1)
    assert (total.errors.dropped, total.errors.hallucinated) == (1, 1)


def test_tag_free_segments_are_left_out_of_the_denominator():
    """A segment with nothing to preserve is not evidence that preservation works."""
    total = aggregate([score("Bonjour", source="Hello"), score("{1}", source="{1}")])
    assert total.segments_scored == 1


def test_a_segment_is_clean_only_with_no_defect_at_all():
    total = aggregate([score("{2>B<2} et {1>A<1}", source="{1>A<1} and {2>B<2}")])
    assert (total.segments_scored, total.segments_clean) == (1, 0)
    assert total.segments_mis_ordered == 1


def test_pooling_sums_counts_rather_than_averaging_rates():
    small = aggregate([score("{2}", source="{1}")])
    large = aggregate([score("{1}" * 9, source="{1}" * 9)])
    assert pool([small, large]).integrity_rate == pytest.approx(9 / 10)


def test_pooling_is_split_invariant():
    scores = [score("{1}", source="{1}"), score("{2}", source="{1}"), score("{1}", source="{1}")]
    whole = aggregate(scores)
    split = pool([aggregate(scores[:1]), aggregate(scores[1:])])
    assert split.integrity_rate == whole.integrity_rate
    assert split.errors.total == whole.errors.total


def test_rate_has_no_denominator_rather_than_being_zero():
    assert aggregate([score("Bonjour", source="Hello")]).integrity_rate is None


# orchestration


def test_every_column_is_scored_against_the_same_source_tags():
    result = _result()
    for segment in result.segments:
        assert segment.mt.expected == segment.ape.expected == segment.ref.expected


def test_reference_is_a_control_not_the_answer_key():
    """The human kept every tag here, so REF is the ceiling the pipeline is read against."""
    result = _result()
    assert result.ref.integrity_rate == 1.0
    assert result.mt.integrity_rate < 1.0


def test_source_tags_are_read_off_the_segments_themselves():
    result = _result()
    assert result.totals["segments_with_tags"] == len(SEGMENTS)
    assert result.segments[0].tags == ["{b>", "<b}"]


def test_ape_repair_shows_up_as_a_delta():
    postmt = FakePostMt(fixes={
        "Work{b> only <b}by qualified staff.":
            "Intervention{b> uniquement <b}par personnel qualifié.",
    })
    result = run_benchmark(_dataset(), postmt=postmt, config=CONFIG, skip_pipeline=False)
    assert result.delta.ape_integrity_rate > 0
    assert result.delta.tags_fixed_by_ape == 1


def test_repairs_and_regressions_are_never_netted():
    postmt = FakePostMt(fixes={"{b>Tooling<b}": "Outillage"})
    result = run_benchmark(_dataset(), postmt=postmt, config=CONFIG, skip_pipeline=False)
    assert result.delta.tags_broken_by_ape == 2
    assert result.delta.tags_fixed_by_ape == 0


def test_dry_run_never_calls_post_mt():
    class Explode:
        def run(self, **_):
            raise AssertionError("post-mt was called under --dry-run")

    result = run_benchmark(_dataset(), postmt=Explode(), config=CONFIG, skip_pipeline=True)
    assert result.mt.integrity_rate == result.ape.integrity_rate


def test_segment_failure_is_surfaced():
    class Failing(FakePostMt):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            result.segments[0]["ape_results"] = {"error": "APE step failed"}
            return result

    result = run_benchmark(_dataset(), postmt=Failing(), config=CONFIG, skip_pipeline=False)
    assert result.failed_segments == 1
    assert result.failure_reason == "APE step failed"


# rendering


def test_scorecard_names_the_stratum():
    card = tags_report.scorecard(_result(domain="Industrial"))
    assert card.subheading == "en-gb → fr-fr  ·  Industrial"


def test_console_headlines_what_the_file_details():
    card = tags_report.scorecard(_result())
    console, markdown = card.as_console(), card.as_markdown()
    assert "Integrity delta" not in console
    assert "Integrity delta" in markdown


def test_three_columns_are_reported():
    facts = "\n".join(tags_report.scorecard(_result()).facts)
    assert "MT" in facts and "APE" in facts and "REF" in facts


def test_error_kinds_are_reported_separately():
    facts = "\n".join(tags_report.scorecard(_result()).facts)
    assert "Dropped" in facts and "Duplicated" in facts and "Hallucinated" in facts


def test_tag_free_corpus_is_warned_about():
    """100% over nothing reads like success, so the empty corpus has to say so."""
    segments = [{"source_segment_id": "s1", "source_content": "Hello",
                 "target_content": "Bonjour", "reference_content": "Bonjour"}]
    card = tags_report.scorecard(_result(segments=segments))
    assert any("No segment carries a tag" in warning for warning in card.warnings)


def test_tag_table_says_so_when_no_version_carries_a_tag():
    """Silence would leave a scorecard of zeroes reading like a clean result."""
    segments = [{"source_segment_id": "s1", "source_content": "plain text",
                 "target_content": "texte simple", "reference_content": "texte simple"}]
    result = _result(segments=segments)

    assert "No tags were found" in tags_report.render_tags(result)
    assert "No tags were found" in tags_report.render_tags_console(result)


def test_hallucinated_tag_still_gets_a_row_with_no_source_tags():
    """The source carried none, so only the per-tag table can show what APE invented."""
    segments = [{"source_segment_id": "s1", "source_content": "plain text",
                 "target_content": "texte {1} simple", "reference_content": "texte simple"}]

    assert "{1}" in tags_report.render_tags(_result(segments=segments))


def test_family_table_splits_the_corpus_by_notation():
    rendered = tags_report.render_families(_result())
    assert "paired_open" in rendered and "standalone" in rendered


def test_family_rates_are_pooled_from_counts():
    rows = {row["family"]: row for row in tags_report.family_rows(_result())}
    assert rows["standalone"]["mt_integrity_rate"] == 0.0
    assert rows["paired_open"]["mt_integrity_rate"] == 1.0


def test_every_tag_gets_an_untruncated_row():
    rows = tags_report.tag_cells(_result())
    assert len(rows) == len({row[0] for row in rows})
    assert "{b>" in {row[0] for row in rows}


def test_worklist_is_worst_first():
    rates = [
        row["ape_integrity_rate"] for row in tags_report.tag_rows(_result())
        if row["ape_integrity_rate"] is not None
    ]
    assert rates == sorted(rates)


def test_defect_worklist_traces_numbers_back_to_segments():
    rendered = tags_report.render_defects(_result())
    assert "s2" in rendered and "dropped" in rendered
    assert "s3" in rendered and "hallucinated" in rendered
    assert "s4" in rendered and "mis-ordered" in rendered


def test_clean_segments_stay_off_the_worklist():
    ids = [row["segment_id"] for row in tags_report.defect_rows(_result())]
    assert "s1" not in ids


def test_pipe_in_a_tag_cannot_start_a_column():
    segments = [{"source_segment_id": "s1", "source_content": '<a t="a|b"/>x',
                 "target_content": "x", "reference_content": "x"}]
    assert "a\\|b" in tags_report.render_tags(_result(segments=segments))


def test_single_dataset_still_gets_a_stratum_row():
    rendered = tags_report.render_strata([_result()])
    assert "en-gb->fr-fr" in rendered


def test_one_pair_splits_into_the_domains_under_it():
    rows = tags_report.stratum_rows([_result(domain="Legal"), _result(domain="Medical")])

    assert [row["label"] for row in rows] == ["en-gb->fr-fr", "↳ Legal", "↳ Medical"]
    assert rows[0]["expected_instances"] == sum(row["expected_instances"] for row in rows[1:])


def test_stratum_row_is_the_pooled_scorecard():
    result = _result()
    row = tags_report.stratum_rows([result])[0]
    assert row["mt_integrity_rate"] == result.mt.integrity_rate
    assert row["expected_instances"] == result.mt.expected


def test_comparison_is_only_drawn_for_several_datasets():
    assert tags_report.render_comparison([_result()]) == ""
    assert "Across datasets" in tags_report.render_comparison([_result(), _result("other")])


def test_per_family_counts_sum_to_the_scorecard():
    result = _result()
    assert sum(row["expected"] for row in tags_report.family_rows(result)) == result.mt.expected


def test_tags_are_a_component_the_report_files():
    now = datetime(2026, 8, 26, 14, 22, 7, tzinfo=timezone.utc)
    rendered = report.render_report(
        {"tags": [_result()]}, run.COMPONENT_SECTIONS, dry_run=True, now=now
    )
    assert "# Tag and placeholder integrity" in rendered
    assert "Integrity by tag family" in rendered


def test_tags_share_a_report_with_the_other_components():
    now = datetime(2026, 8, 26, 14, 22, 7, tzinfo=timezone.utc)
    path = report.report_path(["glossary", "tags"], dry_run=False, now=now)
    assert path.name == "glossary+tags_20260826-142207.md"
