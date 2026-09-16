"""The DNT preservation component: the metric, its rendering, and the orchestration around it."""

import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from sourcecode import dnt_benchmark, dnt_report, report, run
from sourcecode.dnt import Reversion
from sourcecode.dnt_benchmark import fingerprint_of, run_benchmark
from sourcecode.dnt_score import (
    CASE_DRIFT,
    TRANSLATED,
    aggregate,
    pool,
    score_dnt,
)
from sourcecode.postmt import RunResult
from sourcecode.text_processing import Dataset, Task, count_surface, normalize_language


EN, FR = "en-gb", "fr-fr"

_NOW = datetime(2026, 8, 26, 14, 22, 7, tzinfo=timezone.utc)


def score(items, *, text, source="", reference="", source_lang=EN, target_lang=FR):
    return score_dnt(
        items=items, text=text, src_text=source, ref_text=reference,
        source_language_code=source_lang, target_language_code=target_lang,
    )


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


class FailingPostMt(FakePostMt):
    """One segment fails inside a successful task: status done, the error on the segment."""

    def run(self, **kwargs):
        result = super().run(**kwargs)
        for segment in result.segments:
            if segment["source_content"] == "AcoladPro is here.":
                segment["ape_results"] = {"error": "APE step failed"}
        return result


class FakeDnt:
    """Reports the same items for every segment; `reversions` answers with a fixed list instead."""

    def __init__(self, items=("AcoladPro", "Cleaner"), reversions=None):
        self.items = list(items)
        self.reversions = reversions
        self.seen = []

    def revert(self, pairs, *, batch_size, source_language=None, target_language=None):
        self.seen = [dict(pair) for pair in pairs]
        if self.reversions is not None:
            return list(self.reversions)
        return [
            Reversion(rev_text=pair["target"].replace("Pro Acolad", "AcoladPro"), items=self.items)
            for pair in pairs
        ]


CONFIG = SimpleNamespace(
    benchmark=SimpleNamespace(batch_size=10), dnt=SimpleNamespace(batch_size=25)
)

SEGMENTS = [
    # Leaks `AcoladPro`; reversion restores it.
    {"source_segment_id": "s1", "source_content": "AcoladPro is here.",
     "target_content": "Le Pro Acolad est ici.", "reference_content": "AcoladPro est ici."},
    # The reference keeps `Cleaner` once and every version keeps both: an over-keep.
    {"source_segment_id": "s2", "source_content": "The Cleaner cleans the Cleaner.",
     "target_content": "Le Cleaner nettoie le Cleaner.",
     "reference_content": "Le Cleaner nettoie le nettoyeur."},
    # Non-ASCII, to keep the encoding honest.
    {"source_segment_id": "s3", "source_content": "Café Pro is sold here.",
     "target_content": "Le Café Pro est vendu ici.",
     "reference_content": "Le Café Pro est vendu ici."},
    # `Widget` is in the source but the human translated it, so it is flagged and scored nowhere.
    {"source_segment_id": "s4", "source_content": "The Widget ships today.",
     "target_content": "Le Widget est expédié aujourd'hui.",
     "reference_content": "Le bidule est expédié aujourd'hui."},
]

ITEMS = ["AcoladPro", "Cleaner", "Café Pro", "Widget"]


def _dataset(name="dnt-set", domain="Test", segments=SEGMENTS):
    parameters = normalize_language(
        {"source_language": EN, "target_language": FR, "domain": domain}
    )
    return Dataset(
        name=name,
        component="dnt",
        parameters=parameters,
        tasks=[Task(parameters, list(segments))],
    )


def _result(name="dnt-set", domain="Test"):
    return run_benchmark(
        _dataset(name, domain), postmt=None, dnt=FakeDnt(items=ITEMS), config=CONFIG,
        skip_pipeline=True,
    )


def _run(dnt, *, skip_pipeline=True, postmt=None):
    return run_benchmark(
        _dataset(segments=SEGMENTS[:2]), postmt=postmt, dnt=dnt, config=CONFIG,
        skip_pipeline=skip_pipeline,
    )


def test_counting_is_case_sensitive():
    """A DNT item that came through in different casing did not come through."""
    assert count_surface("Le iPhone est ici.", "iPhone", FR, casefold=False) == 1
    assert count_surface("Le iphone est ici.", "iPhone", FR, casefold=False) == 0


def test_normalized_but_not_casefolded():
    """A line wrap or a decomposed accent is an encoding difference, not a translation."""
    assert count_surface("Le  Café\nPro", "Café Pro", FR, casefold=False) == 1
    assert count_surface("Le café Pro", "café Pro", FR, casefold=False) == 1


def test_translated_item_is_leak():
    result = score(["AcoladPro"], source="AcoladPro is here.",
                   reference="AcoladPro est ici.", text="Le Pro Acolad est ici.")

    assert (result.expected, result.preserved, result.leaked) == (1, 0, 1)
    assert result.over_kept == 0
    assert result.leaks.translated == 1


def test_over_keeping_is_error():
    """The direction terminology loses: a DNT flag is not a proposal."""
    result = score(["Alpha"], source="Alpha and Alpha", reference="Alpha et alpha",
                   text="Alpha et Alpha")

    assert result.expected == 1               # the reference kept it once
    assert result.over_kept == 1              # this version kept it twice
    assert result.items.over_kept == 1
    assert result.preserved == 1              # capped at REF's, so the extra never flatters the rate
    assert result.leaked == 0


def test_two_directions_are_never_netted():
    """A pipeline that swaps one failure for the other has not improved."""
    result = score(["Alpha", "Beta"], source="Alpha Beta Beta",
                   reference="Alpha gamma Beta", text="Beta Beta")

    assert result.leaked == 1                 # Alpha: the reference kept it, this version did not
    assert result.over_kept == 1              # Beta: kept twice where the reference kept it once


def test_item_reference_translated_is_excluded():
    """The human translated the string, so it sets no expectation: a flag, not an error."""
    result = score(["Cleaner"], source="The Cleaner works.",
                   reference="Le nettoyeur fonctionne.", text="Le Cleaner fonctionne.")

    assert result.not_in_ref == 1
    assert result.expected == 0
    assert result.over_kept == 0
    assert result.items.distinct_items == 0
    assert result.item_scores == []


def test_item_not_in_source_is_excluded():
    """Tested before the reference gate, so an item failing both is counted once."""
    result = score(["Ghost"], source="Nothing like it", reference="Ghost est ici", text="Ghost")

    assert result.not_in_src == 1
    assert result.not_in_ref == 0
    assert result.expected == 0
    assert result.item_scores == []


def test_three_outcomes_partition():
    """Scored, not in source, not in reference - a reader adding them up gets back the item list."""
    result = score(
        ["Alpha", "Ghost", "Cleaner"],
        source="Alpha and the Cleaner", reference="Alpha et le nettoyeur", text="Alpha et Cleaner",
    )

    assert len(result.item_scores) == 1       # Alpha
    assert result.not_in_src == 1             # Ghost
    assert result.not_in_ref == 1             # Cleaner


def test_casing_only_drift_is_own_bucket():
    """The service repairs these deterministically, so they are a different worklist."""
    result = score(["iPhone"], source="iPhone here",
                   reference="Le iPhone est ici.", text="Le iphone est ici.")

    assert result.leaks.case_drift == 1
    assert result.leaks.translated == 0
    assert result.item_scores[0].leak_kind == CASE_DRIFT


def test_empty_version_leaks_like_any_other():
    """An empty version is not excused: the reference kept the item, so it is owed."""
    result = score(["AcoladPro"], source="AcoladPro here",
                   reference="AcoladPro est ici.", text="")

    assert result.leaks.translated == 1
    assert result.leaks.case_drift == 0
    assert result.item_scores[0].leak_kind == TRANSLATED


def test_mixed_segment_splits_drift_from_translation():
    """One kept exactly, one gone entirely: the buckets must not both claim the same instance."""
    result = score(["Sony"], source="Sony and Sony",
                   reference="Sony et Sony ici.", text="Sony et le fabricant ici.")

    assert result.expected == 2
    assert result.preserved == 1
    assert result.leaks.translated == 1
    assert result.leaks.case_drift == 0


def test_leak_buckets_are_exhaustive():
    result = score(["Sony", "iPhone"], source="Sony iPhone",
                   reference="Sony et iPhone ici.", text="le fabricant et iphone ici.")

    assert result.leaks.total == result.leaked


@pytest.mark.parametrize("text, bucket", [
    ("rien ici", "never_kept"),
    ("Sony ici", "kept_partly"),
    ("Sony et Sony ici", "matched_ref"),
    ("Sony Sony Sony ici", "over_kept"),
])
def test_each_item_lands_in_exactly_one_bucket(text, bucket):
    result = score(["Sony"], source="Sony and Sony",
                   reference="Sony et Sony ici", text=text)

    assert getattr(result.items, bucket) == 1
    assert result.items.distinct_items == 1


def test_item_buckets_sum_to_items_scored():
    result = score(["Sony", "iPhone", "Cleaner"], source="Sony iPhone Cleaner",
                   reference="Sony et iPhone ici", text="Sony et Cleaner ici")

    assert result.items.distinct_items == len(result.item_scores)


def test_source_and_target_codes_are_distinct():
    """One code for both breaks every en<->ja pair, since word boundaries differ."""
    result = score_dnt(
        items=["Sony"],
        src_text="Sony makes it",          # spaced: needs word boundaries
        ref_text="これはSonyの製品です",   # unspaced: needs substring counting
        text="これはSonyの製品です",
        source_language_code="en-gb",
        target_language_code="ja-jp",
    )

    assert result.item_scores[0].in_src == 1
    assert result.expected == 1
    assert result.preserved == 1


def test_aggregation_carries_both_directions():
    leak = score(["A"], source="A", reference="A ici", text="rien ici")
    over = score(["B"], source="B B", reference="B ici", text="B et B")

    result = aggregate([leak, over])

    assert result.leaked == 1
    assert result.over_kept == 1
    assert result.errors == 2


def test_exclusions_are_counted_not_scored():
    """An excluded item still cost an LLM call, so the counts carry it and the rates ignore it."""
    excluded = score(["B"], source="B", reference="rien ici", text="B ici")

    result = aggregate([excluded])

    assert result.not_in_ref == 1
    assert result.expected == 0
    assert result.errors == 0
    assert result.segments_scored == 0
    assert result.preservation_rate is None


def test_fully_preserved_needs_both_directions():
    over = score(["B"], source="B B", reference="B ici", text="B et B")

    result = aggregate([over])

    assert result.segments_scored == 1
    assert result.segments_clean == 0


def test_pooling_sums_counts_rather_than_averaging_rates():
    """A 1-instance dataset must not weigh the same as a 100-instance one."""
    small = aggregate([score(["A"], source="A", reference="A ici", text="A ici")])
    big = aggregate([score(["B"], source="B", reference="B " * 4, text="rien")])

    pooled = pool([small, big])

    assert pooled.expected == small.expected + big.expected
    assert pooled.preserved == small.preserved + big.preserved
    assert pooled.preservation_rate == pooled.preserved / pooled.expected


def test_pooling_is_split_invariant():
    """However the same evidence is divided into datasets, the pooled figure is the same."""
    scores = [
        score(["A"], source="A", reference="A ici", text="A ici"),
        score(["B"], source="B", reference="B ici", text="rien ici"),
        score(["C"], source="C", reference="rien ici", text="C ici"),
    ]

    whole = aggregate(scores)
    split = pool([aggregate(scores[:1]), aggregate(scores[1:])])

    assert (split.expected, split.preserved) == (whole.expected, whole.preserved)
    assert split.over_kept == whole.over_kept
    assert split.preservation_rate == whole.preservation_rate


def test_unread_segments_are_carried():
    result = aggregate([], segments_unread=3)

    assert result.segments_unread == 3
    assert result.preservation_rate is None


def test_pooling_carries_unread_count_forward():
    a = aggregate([], segments_unread=2)
    b = aggregate([], segments_unread=1)

    assert pool([a, b]).segments_unread == 3


def test_scorecard_names_stratum():
    """The component and the stratum, and nothing the tables below already carry."""
    result = _result()
    line = dnt_report.scorecard(result).as_console().splitlines()[0]

    assert line == "dnt  -  en-gb → fr-fr  ·  Test"
    assert result.dataset not in line and "steps" not in line


def test_console_headlines_file_details():
    """Every line in the file breaks down one the console already showed; neither contradicts."""
    result = _result()
    console = dnt_report.scorecard(result).as_console()
    rendered = dnt_report.scorecard(result).as_markdown()

    for headline in ("Preservation", "Leaked", "Over-kept", "Segments clean"):
        assert headline in console and headline in rendered

    for finer in ("Leak kinds", "Item outcomes", "APE repaired", "Preservation delta"):
        assert finer not in console
        assert finer in rendered


def test_scorecard_explains_nothing():
    """The label names the number; what it means is in the README, written once."""
    rendered = dnt_report.scorecard(_result()).as_markdown()

    assert "never netted" not in rendered
    assert "leaked + over-kept" not in rendered


def test_defect_list_names_the_item_the_segment_and_the_fault():
    """The detection table is an items x segments grid; only a filtered list is a worklist."""
    rows = {row["item"]: row for row in dnt_report.defect_rows(_result())}

    assert rows["AcoladPro"]["segment_id"] == "s1"
    assert rows["AcoladPro"]["mt_found"] == 0 and rows["AcoladPro"]["ref_kept"] == 1
    assert "MT translated" in rows["AcoladPro"]["faults"]
    assert "REV" not in rows["AcoladPro"]["faults"]      # reversion repaired it
    assert "MT over-kept" in rows["Cleaner"]["faults"]


def test_defect_list_leaves_out_items_every_version_kept():
    """A row per preserved item would bury the failures, which is what the grid already does."""
    segments = [{"source_segment_id": "s1", "source_content": "AcoladPro is here.",
                 "target_content": "AcoladPro est ici.", "reference_content": "AcoladPro est ici."}]
    result = run_benchmark(
        _dataset(segments=segments), postmt=None, dnt=FakeDnt(items=["AcoladPro"]),
        config=CONFIG, skip_pipeline=True,
    )

    assert dnt_report.defect_rows(result) == []
    assert dnt_report.render_defects(result) == ""


def test_defect_list_separates_case_drift_from_translation():
    """The two failures need different fixes, so the worklist must not call them both a leak."""
    segments = [{"source_segment_id": "s1", "source_content": "AcoladPro is here.",
                 "target_content": "acoladpro est ici.", "reference_content": "AcoladPro est ici."}]
    result = run_benchmark(
        _dataset(segments=segments), postmt=None, dnt=FakeDnt(items=["AcoladPro"]),
        config=CONFIG, skip_pipeline=True,
    )

    assert "MT case drift" in dnt_report.defect_rows(result)[0]["faults"]


def test_src_retention_counts_items_the_reference_translated():
    """REF-against-REF is 100% by construction; SRC is the only denominator that can fall."""
    segments = [
        # Kept by everyone.
        {"source_segment_id": "s1", "source_content": "AcoladPro is here.",
         "target_content": "AcoladPro est ici.", "reference_content": "AcoladPro est ici."},
        # The human translated it away; the MT kept it.
        {"source_segment_id": "s2", "source_content": "AcoladPro is sold.",
         "target_content": "AcoladPro est vendu.", "reference_content": "Le produit est vendu."},
    ]
    result = run_benchmark(
        _dataset(segments=segments), postmt=None, dnt=FakeDnt(items=["AcoladPro"]),
        config=CONFIG, skip_pipeline=True,
    )

    assert result.mt.in_src == 2
    assert result.ref.src_retention_rate == 0.5   # the human kept one of the two
    assert result.mt.src_retention_rate == 1.0    # the MT kept both
    # REF against itself still says nothing, which is why the SRC ratio was added.
    assert result.ref.preservation_rate == 1.0


def test_scorecard_says_when_fewer_segments_were_scored_than_counted():
    """`with items` and the rate's denominator are different numbers; the gap has to be visible."""
    segments = [
        # Both SRC and REF carry it, so REF sets an expectation here.
        {"source_segment_id": "s1", "source_content": "AcoladPro is here.",
         "target_content": "AcoladPro est ici.", "reference_content": "AcoladPro est ici."},
        # REF translated it away, so this segment is named but never scored.
        {"source_segment_id": "s2", "source_content": "AcoladPro is here.",
         "target_content": "AcoladPro est ici.", "reference_content": "Le produit est ici."},
    ]
    result = run_benchmark(
        _dataset(segments=segments), postmt=None, dnt=FakeDnt(items=["AcoladPro"]),
        config=CONFIG, skip_pipeline=True,
    )

    assert result.totals["segments_with_items"] == 2
    assert result.mt.segments_scored == 1
    assert "Scored against REF 1 of 2 segments" in dnt_report.scorecard(result).as_console()


def test_scorecard_stays_quiet_when_every_counted_segment_was_scored():
    segments = [
        {"source_segment_id": "s1", "source_content": "AcoladPro is here.",
         "target_content": "AcoladPro est ici.", "reference_content": "AcoladPro est ici."},
        {"source_segment_id": "s2", "source_content": "AcoladPro is sold.",
         "target_content": "AcoladPro est vendu.", "reference_content": "AcoladPro est vendu."},
    ]
    result = run_benchmark(
        _dataset(segments=segments), postmt=None, dnt=FakeDnt(items=["AcoladPro"]),
        config=CONFIG, skip_pipeline=True,
    )

    assert result.totals["segments_with_items"] == result.mt.segments_scored
    assert "Scored against REF" not in dnt_report.scorecard(result).as_console()


def test_summary_reports_three_columns():
    rendered = dnt_report.scorecard(_result()).as_markdown()

    for column in ("MT ", "APE ", "REV "):
        assert column in rendered


def test_reference_metrics_stay_out_of_reports():
    """REF preserves everything by construction, so scoring it says nothing worth a column."""
    result = _result()
    scorecard = dnt_report.scorecard(result)

    assert result.ref.preservation_rate == 1.0
    for destination in (scorecard.as_console(), scorecard.as_markdown()):
        assert "REF 100.00%" not in destination
        assert "(REF" not in destination


def test_summary_carries_detection_fingerprint():
    """Detection is not reproducible, so a reader must see whether the denominator moved."""
    result = _result()

    assert result.fingerprint in dnt_report.scorecard(result).as_markdown()


def test_counts_travel_beside_rates():
    rendered = dnt_report.scorecard(_result()).as_markdown()

    assert "REF instances" in rendered
    assert "not in REF" in rendered


def test_non_ascii_items_survive_rendering():
    rendered = (dnt_report.scorecard(_result()).as_markdown()
                + dnt_report.render_items(_result()))

    assert "Café Pro" in rendered


def test_every_item_gets_untruncated_row():
    """Not a top-N: a table that stops somewhere tells a reader they have seen the whole worklist."""
    rendered = dnt_report.render_items(_result())

    for item in ("AcoladPro", "Cleaner", "Café Pro"):
        assert item in rendered


def test_out_of_scope_item_is_flagged():
    """`Widget` moves no number but still cost an LLM call, so the run has to say so somewhere."""
    result = _result()

    assert "Widget" not in dnt_report.render_items(result)

    detection = dnt_report.render_detection(result)
    assert "Widget" in detection
    assert "not in REF" in detection


def test_detection_table_is_keyed_by_segment_and_item():
    """The evidence is per segment; pooling it leaves a ratio matching neither."""
    rows = {(row["segment_id"], row["item"]): row for row in dnt_report.detection_rows(_result())}

    # The ratio reads the reverted output, which restored the item the MT had translated.
    assert (rows[("s1", "AcoladPro")]["preserved"], rows[("s1", "AcoladPro")]["expected"]) == (1, 1)
    assert (rows[("s1", "AcoladPro")]["in_mt"], rows[("s1", "AcoladPro")]["in_rev"]) == (0, 1)
    # Kept twice where the reference kept it once: capped, so the ratio cannot exceed itself.
    assert (rows[("s2", "Cleaner")]["preserved"], rows[("s2", "Cleaner")]["expected"]) == (1, 1)

    assert rows[("s4", "Widget")]["flag"] == "not in REF"
    assert rows[("s1", "Cleaner")]["flag"] == "not in SRC"
    assert {segment_id for segment_id, _ in rows} == {"s1", "s2", "s3", "s4"}


def test_detection_ratios_sum_to_scorecard():
    """Two readings of one measurement: a table with its own total would leave two figures."""
    result = _result()
    rows = dnt_report.detection_rows(result)

    assert sum(row["expected"] for row in rows) == result.rev.expected
    assert sum(row["preserved"] for row in rows) == result.rev.preserved


def test_out_of_scope_item_shows_dash():
    """`0/0` would read as a failure; the reference simply asked nothing of it."""
    console = dnt_report.render_detection_console(_result())
    widget = next(
        line for line in console.splitlines()
        if line.split()[:2] == ["s4", "Widget"]
    )

    assert "-  not in REF" in widget


def test_over_keeping_surfaces_past_cap():
    """`Cleaner` reads 100% because the cap discards the extra; the count is where it surfaces."""
    result = _result()
    rows = {row["item"]: row for row in dnt_report.item_rows(result)}

    assert rows["Cleaner"]["rev_over_kept"] == 1
    assert rows["Cleaner"]["rev_preservation_rate"] == 1.0
    assert "Over-kept" in dnt_report.scorecard(result).as_markdown()


def test_no_denominator_renders_as_na():
    """A run whose items were all out of scope must read as `n/a`, not as a clean zero."""
    result = _result()
    result.segments = [s for s in result.segments if s.source_segment_id == "s4"]
    result.mt = aggregate([s.mt for s in result.segments])

    assert result.mt.preservation_rate is None
    assert "n/a" in dnt_report.scorecard(result).as_markdown()


def test_per_item_table_counts_every_version():
    """MT alone hides what reversion repaired, and a rate alone hides what was over-kept."""
    rendered = dnt_report.render_items(_result())
    header = next(line for line in rendered.splitlines() if line.startswith("| DNT item"))

    assert [column.strip() for column in header.strip("|").split("|")] == [
        "DNT item", "MT", "APE", "REV", "REF", "Preservation",
    ]
    # `AcoladPro` was translated in the MT and reversion restored it: both counts on the one row.
    acolad = next(line for line in rendered.splitlines() if line.startswith("| AcoladPro"))
    assert acolad.strip("|").split("|")[1:] == [" 0 ", " 0 ", " 1 ", " 1 ", " 100.00% "]


def test_worklist_is_worst_first():
    """Worst as delivered, then worst as the MT had it, so a rescued term still sorts high."""
    rows = dnt_report.item_rows(_result())
    rates = [(row["rev_preservation_rate"], row["mt_preservation_rate"]) for row in rows]

    assert rates == sorted(rates)
    assert rows[0]["item"] == "AcoladPro"        # delivered clean, but the MT translated it


def test_no_items_says_so():
    result = _result()
    result.segments = []

    assert "No DNT items" in dnt_report.render_items(result)


def test_single_dataset_still_gets_stratum_row():
    """A run that measured one dataset must still say what its language pair preserved."""
    rendered = dnt_report.render_strata([_result()])

    assert "Preservation by language pair" in rendered
    assert "en-gb->fr-fr" in rendered
    assert "ALL" not in rendered                 # nothing to pool across


def test_one_pair_splits_into_the_domains_under_it():
    """The pair leads, and each domain it was measured in is listed beneath it."""
    results = [_result("a", "Automotive"), _result("b", "Legal")]

    rows = dnt_report.stratum_rows(results)

    assert [row["label"] for row in rows] == ["en-gb->fr-fr", "↳ Automotive", "↳ Legal"]
    assert rows[0]["expected_instances"] == sum(row["expected_instances"] for row in rows[1:])
    # One pair, so its row is already the total; an ALL row would only repeat it.
    assert "ALL" not in dnt_report.render_strata(results)


def test_stratum_rows_are_printed_as_well_as_written():
    """The console and the file carry the same rows, off one builder."""
    results = [_result("a"), _result("b")]
    console = dnt_report.render_strata_console(results)

    for label, rates in dnt_report.stratum_rate_rows(results):
        assert label in console
        for rate in rates:
            assert rate in console


def test_stratum_row_carries_both_directions():
    rows = dnt_report.stratum_rows([_result("a"), _result("b")])

    assert rows[0]["mt_leaked"] == 2          # one leak per dataset
    assert rows[0]["mt_over_kept"] == 2


def test_per_item_counts_sum_to_scorecard():
    result = _result()
    rows = dnt_report.item_rows(result)

    assert sum(r["ref_kept"] for r in rows) == result.mt.expected
    assert sum(r["mt_over_kept"] for r in rows) == result.mt.over_kept


def test_per_segment_counts_sum_to_scorecard():
    result = _result()

    assert sum(s.mt.preserved for s in result.segments) == result.mt.preserved
    assert sum(s.mt.leaked for s in result.segments) == result.mt.leaked
    assert sum(s.mt.over_kept for s in result.segments) == result.mt.over_kept


def test_stratum_row_is_pooled_scorecard():
    results = [_result("a"), _result("b")]
    row = dnt_report.stratum_rows(results)[0]
    pooled = pool([r.mt for r in results])

    assert row["expected_instances"] == pooled.expected
    assert row["mt_preservation_rate"] == pooled.preservation_rate


def test_source_count_sums_over_segments():
    """The SRC count sits beside REF's, so both count the same way; a maximum under-reports."""
    result = _result()
    # Each item appears in exactly one source, so a per-column sum would treble it.
    rows = {row["item"]: row for row in dnt_report.item_rows(result)}

    assert rows["AcoladPro"]["in_src"] == 1
    assert rows["Cleaner"]["in_src"] == 2      # twice in the one source that carries it
    assert rows["Café Pro"]["in_src"] == 1


def test_both_components_can_share_one_report():
    """A run measuring several components writes them into one file, so they are read together."""
    rendered = report.render_report(
        {"dnt": [_result()]}, run.COMPONENT_SECTIONS, dry_run=True, now=_NOW
    )

    assert rendered.startswith("# Quality baseline — dnt")
    assert "# DNT preservation" in rendered


def test_component_with_nothing_is_skipped():
    rendered = report.render_report({"dnt": []}, run.COMPONENT_SECTIONS, dry_run=True, now=_NOW)

    assert "# DNT preservation" not in rendered


def test_report_name_carries_run():
    """Without the timestamp every run overwrites the last, leaving nothing to compare against."""
    assert report.report_path(
        ["glossary", "dnt"], dry_run=False, now=_NOW
    ).name == "glossary+dnt_20260826-142207.md"
    assert report.report_path(
        ["dnt"], dry_run=True, now=_NOW
    ).name == "dnt_dry-run_20260826-142207.md"


def test_reversion_is_given_post_edited_text():
    """Delivery order: MT, then post-editing, then reversion on top of what would be shipped."""
    dnt = FakeDnt()
    postmt = FakePostMt(fixes={"AcoladPro is here.": "Le Pro Acolad EST ici."})

    _run(dnt, skip_pipeline=False, postmt=postmt)

    assert dnt.seen[0]["source"] == "AcoladPro is here."
    assert dnt.seen[0]["target"] == "Le Pro Acolad EST ici."   # the APE text, not MT


def test_under_dry_run_reversion_is_given_raw_mt():
    """The stub makes the APE column the raw MT, so the same call reverts it."""
    dnt = FakeDnt()

    _run(dnt, skip_pipeline=True)

    assert dnt.seen[0]["target"] == "Le Pro Acolad est ici."


def test_segments_are_sent_with_their_index_as_id():
    dnt = FakeDnt()

    _run(dnt)

    assert [pair["id"] for pair in dnt.seen] == ["0", "1"]


def test_every_column_is_scored_against_same_item_list():
    """Re-detecting per column would move the ground under each and make the arrows meaningless."""
    result = _run(FakeDnt())

    assert result.mt.expected == result.ape.expected
    assert result.ape.expected == result.rev.expected


def test_reversion_repairs_show_up_as_delta():
    """Two scored instances: `AcoladPro`, which reversion restores, and the one `Cleaner` kept."""
    result = _run(FakeDnt())

    assert result.mt.preservation_rate == 0.5
    assert result.rev.preservation_rate == 1.0
    assert result.delta.rev_preservation_rate == 0.5
    assert result.delta.items_fixed_by_rev == 1


def test_over_keeping_is_counted_in_every_column():
    """`Cleaner` is kept twice by every version where the reference kept it once."""
    result = _run(FakeDnt())

    assert result.mt.over_kept == 1
    assert result.rev.over_kept == 1


def test_unread_segment_is_excluded():
    """No evidence is neither a perfect score nor a failure, so it leaves the denominator."""
    dnt = FakeDnt(reversions=[None, Reversion("Le nettoyeur fonctionne.", ["Cleaner"])])

    result = _run(dnt)

    assert result.mt.segments_unread == 1
    assert result.totals["segments"] == 2
    assert result.totals["segments_read"] == 1
    assert len(result.segments) == 2                    # still reported
    assert result.segments[0].unread is True
    assert result.segments[0].rev_text == result.segments[0].ape_text


def test_segment_failure_is_surfaced():
    """A failed post-edit echoes the raw MT back, which reads as untouched unless counted."""
    result = _run(FakeDnt(), skip_pipeline=False, postmt=FailingPostMt(fixes={}))

    assert result.failed_segments == 1
    assert result.failure_reason == "APE step failed"
    assert len(result.segments) == 2


def test_fingerprint_ignores_order():
    """Detection runs in an LLM call, so the report says whether two runs shared a denominator."""
    assert fingerprint_of([["A", "B"]]) == fingerprint_of([["B"], ["A"]])


def test_different_item_set_gets_different_fingerprint():
    assert fingerprint_of([["A", "B"]]) != fingerprint_of([["A", "C"]])


def test_no_items_at_all_is_named_rather_than_hashed():
    assert fingerprint_of([[], []]) == "none"


def test_run_records_fingerprint_of_what_scored():
    result = _run(FakeDnt())

    assert result.fingerprint == fingerprint_of([["AcoladPro", "Cleaner"]] * 2)


def test_dry_run_never_calls_post_mt():
    class ExplodingPostMt:
        def run(self, **kwargs):
            raise AssertionError("post-mt must not be called under --dry-run")

    result = _run(FakeDnt(), skip_pipeline=True, postmt=ExplodingPostMt())

    assert result.dataset.endswith("(dry-run)")


def test_dry_run_mirrors_mt_baseline():
    result = _run(FakeDnt())

    assert result.mt.preserved == result.ape.preserved
    assert result.delta.ape_preservation_rate == 0.0


# the reversion gold set: the terms to restore are given, so nothing is read off a reference

from sourcecode.dnt_score import aggregate_reversion, score_reversion   # noqa: E402
from sourcecode.postmt import preflight_submission   # noqa: E402
from sourcecode.text_processing import load as load_dataset_file   # noqa: E402


def _gold(tmp_path, pairs, name="gold.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"version": "1.0", "segments": pairs}), encoding="utf-8")
    return path


def _pair(identifier="eval-1", target="Die RITFIT Bank.", terms=("RITFIT",), target_language="de_DE"):
    return {
        "id": identifier, "source_language": "en_US", "target_language": target_language,
        "source": "The RITFIT bench.", "target": target, "expected_terms": list(terms),
    }


def test_a_gold_set_needs_no_reference(tmp_path):
    """The gold names the terms outright, so there is nothing to compare against a human."""
    data = load_dataset_file(_gold(tmp_path, [_pair()]), component="dnt")

    assert data.is_gold_set
    assert data.segments[0]["expected_terms"] == ["RITFIT"]
    assert "reference_content" not in data.segments[0]


def test_each_target_language_is_its_own_task(tmp_path):
    path = _gold(tmp_path, [_pair(target_language="de_DE"), _pair(target_language="ja_JP")])
    data = load_dataset_file(path, component="dnt")

    assert [t.parameters["clean_target_language_code"] for t in data.tasks] == ["de-de", "ja-jp"]


def test_a_pair_with_no_target_leaves_the_set(tmp_path):
    """An empty machine translation has nothing to revert, and would score as every term lost."""
    path = _gold(tmp_path, [_pair(), _pair("eval-2", target="   ")])
    data = load_dataset_file(path, component="dnt")

    assert [s["source_segment_id"] for s in data.segments] == ["eval-1"]


def test_a_detection_gold_set_says_what_it_is(tmp_path):
    path = tmp_path / "detection.json"
    path.write_text(json.dumps({"segments": [
        {"id": "eval-1", "text": "The RITFIT bench.", "source_language": "en_US",
         "expected_terms": ["RITFIT"]},
    ]}), encoding="utf-8")

    with pytest.raises(ValueError, match="detection gold set"):
        load_dataset_file(path, component="dnt")


def _score(terms, mt_text, rev_text, *, ape_text=None, rev_ape_text=None, language="de-de"):
    """The APE arm mirrors the MT arm unless a test is about the two coming apart."""
    return score_reversion(
        terms=terms,
        mt_text=mt_text,
        ape_text=mt_text if ape_text is None else ape_text,
        rev_text=rev_text,
        rev_ape_text=rev_text if rev_ape_text is None else rev_ape_text,
        target_language_code=language,
    )


def test_a_term_restored_by_reversion_counts_as_repaired():
    score = _score(["RITFIT", "LA Galaxy"], "RITFIT und LA-Galaxie", "RITFIT und LA Galaxy")

    assert (score.expected, score.in_mt, score.in_rev) == (2, 1, 2)
    assert (score.repaired, score.broken) == (1, 0)
    assert score.clean


def test_a_term_reversion_loses_counts_as_broken():
    score = _score(["RITFIT"], "RITFIT hier", "Ritfit hier")

    assert (score.in_mt, score.in_rev, score.broken) == (1, 0, 1)
    assert not score.clean


def test_the_two_arms_are_scored_against_their_own_baseline():
    """Post-editing can lose a term the MT carried, and reverting the APE has to restore it."""
    score = _score(
        ["RITFIT"], "RITFIT hier", "RITFIT hier",
        ape_text="Ritfit hier", rev_ape_text="RITFIT hier",
    )

    assert (score.in_mt, score.in_ape) == (1, 0)
    assert (score.in_rev, score.in_rev_ape) == (1, 1)
    # Nothing to repair on the MT arm; the APE arm restored what post-editing dropped.
    assert (score.repaired, score.repaired_from_ape) == (0, 1)
    assert score.clean and score.clean_from_ape


def test_an_arm_can_fail_where_the_other_carries():
    score = _score(["RITFIT"], "RITFIT hier", "RITFIT hier", rev_ape_text="Ritfit hier")

    assert (score.in_rev, score.in_rev_ape) == (1, 0)
    assert (score.broken, score.broken_from_ape) == (0, 1)
    assert score.clean and not score.clean_from_ape


def test_casing_is_part_of_carrying_the_term():
    """A DNT item that came through in different casing did not come through."""
    score = _score(["RITFIT"], "Ritfit", "Ritfit")

    assert (score.in_mt, score.in_rev) == (0, 0)


def test_an_unspaced_language_matches_without_word_boundaries():
    score = _score(["RITFIT"], "RITFITのベンチ", "RITFITのベンチ", language="ja-jp")

    assert (score.in_mt, score.in_rev) == (1, 1)


def test_a_pair_with_no_gold_term_carries_no_denominator():
    """It sets no expectation, so counting it as perfect would inflate every rate."""
    empty = _score([], "x", "x")
    one = _score(["RITFIT"], "RITFIT", "RITFIT")

    pooled = aggregate_reversion([empty, one])

    assert (pooled.expected, pooled.segments_scored) == (1, 1)
    assert pooled.rev_rate == 1.0
    assert pooled.rev_ape_rate == 1.0


def test_the_same_term_twice_on_a_pair_is_one_expectation():
    score = _score(["RITFIT", "RITFIT"], "RITFIT", "RITFIT")

    assert score.expected == 1


def test_a_gold_set_is_submitted_under_a_task_id_post_mt_accepts(tmp_path):
    """post-mt rejects a submission with no `tempo_task_id`, so the gold set is given one."""
    data = load_dataset_file(_gold(tmp_path, [_pair(), _pair("eval-2", target_language="fr_FR")]),
                             component="dnt")

    assert [t.parameters["tempo_task_id"] for t in data.tasks] == [
        "mt-dnt-bench-gold-en-us_de-de", "mt-dnt-bench-gold-en-us_fr-fr",
    ]
    assert preflight_submission(data.tasks[0].parameters) == []


def test_the_gold_set_goes_through_post_mt(tmp_path):
    """The pairs are post-edited like any other dataset, so there is an APE arm to compare."""
    data = load_dataset_file(_gold(tmp_path, [_pair()]), component="dnt")
    postmt = FakePostMt(fixes={"The RITFIT bench.": "Die RITFIT Sitzbank."})

    result = dnt_benchmark.run_reversion(
        data, postmt=postmt, dnt=FakeDnt(reversions=[None]), config=CONFIG
    )

    [segment] = result.segments
    assert segment.ape_text == "Die RITFIT Sitzbank."
    assert segment.changed_by_ape


def test_reversion_runs_on_the_mt_and_on_the_post_edited_text(tmp_path):
    """Both arms are asked for the same pair, so the two rates are over the same terms."""
    data = load_dataset_file(_gold(tmp_path, [_pair()]), component="dnt")
    postmt = FakePostMt(fixes={"The RITFIT bench.": "Die Ritfit Sitzbank."})

    seen = []

    class _Dnt:
        def revert(self, pairs, *, batch_size, source_language, target_language):
            seen.extend(pair["target"] for pair in pairs)
            return [Reversion(rev_text=p["target"].replace("Ritfit", "RITFIT"), items=[])
                    for p in pairs]

    result = dnt_benchmark.run_reversion(data, postmt=postmt, dnt=_Dnt(), config=CONFIG)

    assert seen == ["Die RITFIT Bank.", "Die Ritfit Sitzbank."]
    # The MT carried the term and APE dropped it; reverting the APE put it back.
    assert (result.scored.in_mt, result.scored.in_ape) == (1, 0)
    assert (result.scored.in_rev, result.scored.in_rev_ape) == (1, 1)
    assert result.scored.repaired_from_ape == 1


def test_reversion_runs_one_pass_per_language_pair(tmp_path):
    path = _gold(tmp_path, [_pair(target_language="de_DE"), _pair("eval-2", target_language="fr_FR")])
    data = load_dataset_file(path, component="dnt")

    seen = []

    class _Dnt:
        def revert(self, pairs, *, batch_size, source_language, target_language):
            seen.append((source_language, target_language, len(pairs)))
            return [Reversion(rev_text=p["target"], items=[]) for p in pairs]

    dnt_benchmark.run_reversion(data, postmt=FakePostMt(), dnt=_Dnt(), config=CONFIG)

    # Two arms per pair: the MT first, then the post-edited text.
    assert seen == [("en-us", "de-de", 1), ("en-us", "de-de", 1),
                    ("en-us", "fr-fr", 1), ("en-us", "fr-fr", 1)]


def test_a_pair_no_batch_came_back_for_leaves_the_denominator(tmp_path):
    data = load_dataset_file(_gold(tmp_path, [_pair(), _pair("eval-2")]), component="dnt")

    class _HalfBroken:
        def revert(self, pairs, *, batch_size, source_language, target_language):
            return [Reversion(rev_text=pairs[0]["target"], items=[]), None]

    result = dnt_benchmark.run_reversion(
        data, postmt=FakePostMt(), dnt=_HalfBroken(), config=CONFIG
    )

    assert result.scored.segments_unread == 1
    assert result.scored.segments_scored == 1
    assert result.totals["segments_read"] == 1
