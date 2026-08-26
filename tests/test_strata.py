"""Stratum aggregation: pooling per-dataset results into one figure per language pair × domain,
and resolving `BENCH_DATASET` into the datasets to score."""

import pytest

from sourcecode.cli import resolve_datasets
from sourcecode.report import by_stratum, render_strata, stratum_of, stratum_rows
from sourcecode.score import Aggregate, MissBreakdown, TallyReport, TermBreakdown, pool


def _aggregate(expected, adherent, *, segments=1, fully_adherent=1, terms=None, omitted=0):
    """An Aggregate carrying just the counts the pooling cares about.

    `terms` defaults to one term per occurrence, either used everywhere or never used, which is
    what a corpus with no repeated source term looks like; pass a TermBreakdown explicitly to
    exercise partial or over-use. Misses default to substitutions, the ordinary case; `omitted`
    moves some of them across so pooling of the split gets exercised too.
    """
    breakdown = terms or TermBreakdown(
        used_everywhere=adherent, never_used=expected - adherent
    )
    missed = expected - adherent
    misses = MissBreakdown(substituted=missed - omitted, omitted=omitted)
    return Aggregate(
        expected=expected,
        adherent=adherent,
        violations=missed,
        misses=misses,
        adherence_rate=None if expected == 0 else adherent / expected,
        strict=TallyReport(expected, adherent, missed, misses, None),
        permissive=TallyReport(0, 0, 0, MissBreakdown(), None),
        terms=breakdown,
        segments_with_glossary=segments,
        segments_fully_adherent=fully_adherent,
        segment_adherence_rate=None,
    )


class _Result:
    """The few BenchmarkResult fields the stratum code reads."""

    def __init__(self, name, source, target, domain, mt, ape=None, segments=1):
        self.dataset = name
        self.parameters = {"source_language": source, "target_language": target, "domain": domain}
        self.mt_baseline = mt
        self.post_edited = ape if ape is not None else mt
        self.totals = {"segments": segments}
        self.started_at = "2026-01-01T00:00:00+00:00"


# --------------------------------------------------------------------------- pooling


def test_pooling_sums_counts_rather_than_averaging_rates():
    """The whole point of pooling: a 2-instance dataset must not carry the same weight as a
    200-instance one. Averaging the rates here would give 75%; the evidence says 50.5%."""
    pooled = pool([_aggregate(2, 2), _aggregate(200, 100)])

    assert pooled.expected == 202
    assert pooled.adherent == 102
    assert pooled.adherence_rate == pytest.approx(102 / 202)
    assert pooled.adherence_rate != pytest.approx((1.0 + 0.5) / 2)


def test_pooling_is_split_invariant():
    """Splitting one dataset in two and pooling must reproduce the unsplit number exactly,
    otherwise how the data happens to be filed would change the score."""
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


def test_an_empty_stratum_reports_no_rate_rather_than_zero():
    """`None`, never 0: no evidence is not total failure."""
    pooled = pool([_aggregate(0, 0, segments=0, fully_adherent=0)])

    assert pooled.expected == 0
    assert pooled.adherence_rate is None


# --------------------------------------------------------------------------- grouping


def test_a_stratum_is_a_pair_and_a_domain():
    result = _Result("d", "en-gb", "fr-fr", "Automotive", _aggregate(1, 1))

    assert stratum_of(result) == ("en-gb->fr-fr", "Automotive")


def test_same_pair_different_domain_are_different_strata():
    a = _Result("a", "en-gb", "fr-fr", "Automotive", _aggregate(4, 2))
    b = _Result("b", "en-gb", "fr-fr", "Forestry", _aggregate(4, 4))

    assert len(by_stratum([a, b])) == 2


def test_datasets_in_one_stratum_are_pooled_into_a_single_row():
    a = _Result("a", "en-gb", "fr-fr", "Automotive", _aggregate(6, 4), segments=2)
    b = _Result("b", "en-gb", "fr-fr", "Automotive", _aggregate(2, 1), segments=1)

    rows = stratum_rows([a, b])

    assert len(rows) == 1
    assert rows[0]["datasets"] == 2
    assert rows[0]["segments"] == 3
    assert rows[0]["expected_instances"] == 8
    assert rows[0]["mt_adherence_rate"] == pytest.approx(5 / 8)


def test_a_missing_domain_still_forms_a_stratum():
    result = _Result("d", "en-gb", "fr-fr", None, _aggregate(2, 1))

    assert stratum_of(result) == ("en-gb->fr-fr", "(no domain)")
    assert len(stratum_rows([result])) == 1


def test_render_shows_the_instance_count_next_to_the_rate():
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


def test_a_single_dataset_renders_no_strata_table():
    """Its one row would restate the scorecard: same pair, domain, instances and both rates."""
    assert render_strata([_Result("d", "en-gb", "fr-fr", "Automotive", _aggregate(2, 2))]) == ""


# --------------------------------------------------------------------------- resolving input


def test_a_single_file_is_used_as_given(tmp_path):
    path = tmp_path / "d.json"
    path.write_text("{}", encoding="utf-8")

    assert resolve_datasets(str(path)) == [path]


def test_a_folder_expands_to_every_dataset_inside_it_sorted(tmp_path):
    for name in ("b.json", "a.csv", "notes.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")

    resolved = resolve_datasets(str(tmp_path))

    assert [p.name for p in resolved] == ["a.csv", "b.json"]      # .txt ignored


def test_a_folder_is_not_searched_recursively(tmp_path):
    """A nested folder is usually a different experiment; absorbing it would silently change
    what the run measures."""
    (tmp_path / "top.json").write_text("x", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.json").write_text("x", encoding="utf-8")

    assert [p.name for p in resolve_datasets(str(tmp_path))] == ["top.json"]


def test_the_configured_value_is_what_gets_scored(tmp_path):
    path = tmp_path / "from-env.json"
    path.write_text("{}", encoding="utf-8")

    assert resolve_datasets(str(path)) == [path]


def test_nothing_configured_says_how_to_set_it(tmp_path):
    with pytest.raises(ValueError, match="BENCH_DATASET"):
        resolve_datasets("")


def test_an_empty_folder_is_an_error_rather_than_a_silent_no_op(tmp_path):
    """Scoring nothing must not exit 0 as though everything passed."""
    with pytest.raises(ValueError, match="No dataset files"):
        resolve_datasets(str(tmp_path))


def test_a_configured_path_that_does_not_exist_names_it(tmp_path):
    missing = tmp_path / "nope.json"

    with pytest.raises(ValueError, match="nope.json"):
        resolve_datasets(str(missing))


def test_the_folder_scan_accepts_every_format_the_loader_does(tmp_path):
    """A folder of .xlf files once reported 'no dataset files' even though load() reads them."""
    from sourcecode import dataset as dataset_module
    from sourcecode.cli import DATASET_SUFFIXES

    for suffix in DATASET_SUFFIXES:
        (tmp_path / f"d{suffix}").write_text("x", encoding="utf-8")

    resolved = resolve_datasets(str(tmp_path))

    assert len(resolved) == len(DATASET_SUFFIXES)
    # and none of them is a format load() would reject outright
    for path in resolved:
        assert path.suffix.lower() in {".json", ".csv", ".mxliff", ".xliff", ".xlf"}
