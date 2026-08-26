"""Report rendering; the non-ASCII fixtures guard the encoding path, which an ASCII-only suite
would let regress."""

import pytest

from sourcecode.benchmark import Benchmark
from sourcecode.dataset import Dataset
from sourcecode.normalize import normalize_language
from sourcecode.report import (
    render_summary, render_term_adherence, segment_rows, term_rows, violation_rows,
)
from sourcecode.glossary import GlossaryMatches

GLOSSARY = {
    "The brake pad is worn.": [{"source_content": "brake pad", "target_content": "frein"}],
    "Connect the cable.": [{"source_content": "cable", "target_content": "câble de recharge"}],
    "Connect the cable, it is supplied.":
        [{"source_content": "cable", "target_content": "câble de recharge"}],
}

SEGMENTS = [
    {"source_segment_id": "s1", "source_content": "The brake pad is worn.", "target_content": "La plaquette est usée.",
     "reference_content": "Le frein est usé."},
    {"source_segment_id": "s2", "source_content": "Connect the cable.", "target_content": "Branchez le fil.",
     "reference_content": "Branchez le câble de recharge."},
]


class FakeStanza:
    def lemmatize_batch(self, texts, language):
        return list(texts)

    def lemmatize_batch_safe(self, texts, language):
        return list(texts)


class FakeGlossary:
    def fetch_matches(self, *, glossary_ids, source_language, target_language, texts, provider=None):
        per_text = [GLOSSARY.get(text, []) for text in texts]
        return GlossaryMatches(mappings=[m for g in per_text for m in g], per_text_mappings=per_text)


class FakePostMt:
    def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
        from tests.test_pipeline import FakeRunResult  # reuse the shared stub shape

        return FakeRunResult(
            task_id="stub",
            segments=[
                {**s, "has_glossary": bool(GLOSSARY.get(s["source_content"])),
                 "ape_results": {"text": s["target_content"]}}
                for s in segments
            ],
            error=None,
        )


class _BenchCfg:
    batch_size = 10
    lemma_matching = True


class _Cfg:
    benchmark = _BenchCfg()


@pytest.fixture
def result():
    dataset = Dataset(
        name="accents",
        parameters=normalize_language(
            {
                "source_language": "English (United Kingdom)",
                "target_language": "French (France)",
                "domain": "Automotive",
            }
        ),
        glossary_ids=["tb1"],
        segments=[dict(s) for s in SEGMENTS],
    )
    benchmark = Benchmark(postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg())
    return benchmark.run(dataset)


def test_summary_shows_na_rather_than_zero_when_nothing_was_expected(result):
    result.mt_baseline.adherence_rate = None
    result.post_edited.adherence_rate = None
    assert "n/a" in render_summary(result)


def test_violation_rows_distinguish_regressions_from_pre_existing(result):
    rows = violation_rows(result)
    # APE changed nothing here, so every violation was already wrong in MT.
    assert all(row["already_wrong_in_mt"] == "yes" for row in rows)
    assert all(row["introduced_by_ape"] == "no" for row in rows)


def test_term_rows_report_a_rate_per_glossary_term(result):
    rows = {row["source_term"]: row for row in term_rows(result)}
    assert set(rows) == {"brake pad", "cable"}

    # Neither fixture translation carries its glossary target, so both terms score 0/1.
    for term in rows.values():
        assert term["segments"] == 1
        assert term["expected_instances"] == 1
        assert term["ape_adherent"] == 0
        assert term["ape_violations"] == 1
        assert term["ape_adherence_rate"] == 0.0

    # The accented target must survive into the per-term rollup too.
    assert "câble de recharge" in rows["cable"]["expected_targets"]


def test_term_rows_are_worst_first(result):
    rows = term_rows(result)
    assert [row["source_term"] for row in rows] == ["brake pad", "cable"]


def test_segment_rows_carry_the_segment_rate(result):
    rows = segment_rows(result)
    for row in rows:
        assert row["mt_adherence_rate"] == pytest.approx(
            row["mt_adherent"] / row["expected_terms"]
        )
        assert row["mt_violations"] == row["expected_terms"] - row["mt_adherent"]


def test_the_three_levels_reconcile(result):
    """Per-term counts must sum to the dataset total the stratum row pools from."""
    rows = term_rows(result)
    assert sum(row["expected_instances"] for row in rows) == result.mt_baseline.expected
    assert sum(row["mt_adherent"] for row in rows) == result.mt_baseline.adherent

    segments = segment_rows(result)
    assert sum(row["expected_terms"] for row in segments) == result.mt_baseline.expected
    assert sum(row["mt_adherent"] for row in segments) == result.mt_baseline.adherent


def test_term_rows_expose_the_raw_target_count_beside_the_capped_one(result):
    """`T` is reported unbounded, so an over-rendering stays visible after the cap folds it away."""
    for row in term_rows(result):
        assert row["ape_rendered"] == 0                    # neither fixture target was used
        assert row["ape_adherent"] == min(row["ape_rendered"], row["expected_instances"])


def test_the_per_term_table_shows_r_t_and_violations(result):
    table = render_term_adherence(result)
    header = next(line for line in table.splitlines() if "Source term" in line)
    assert header.split() == [
        "Source", "term", "Expected", "R", "T", "Adherent", "Viol.", "Adherence", "Kind", "Flag",
    ]
    # No legend under it: the columns are named in the header and nowhere else.
    assert "renderings in the human reference" not in table
    assert table.rstrip().splitlines()[-1].strip().startswith("cable")


def test_a_term_used_more_often_than_the_reference_is_flagged_for_review():
    """The human wrote the term once and a pronoun the second time; the pipeline repeated the term.
    That is over-use — adherent by the cap, so the row reads 100% and the only place it can surface
    is the flag column, which is what sends a reviewer to look at it."""
    dataset = Dataset(
        name="over-used",
        parameters=normalize_language({"source_language": "en-gb", "target_language": "fr-fr"}),
        glossary_ids=["tb1"],
        segments=[
            {
                "source_segment_id": "s1",
                "source_content": "Connect the cable, it is supplied.",
                "target_content": "Branchez le câble de recharge, le câble de recharge est fourni.",
                "reference_content": "Branchez le câble de recharge, il est fourni.",
            }
        ],
    )
    result = Benchmark(
        postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg()
    ).run(dataset)

    row, = term_rows(result)
    assert row["expected_instances"] == 1
    assert row["ape_rendered"] == 2
    assert row["ape_over_used"] is True
    assert row["ape_violations"] == 0
    assert row["ape_adherence_rate"] == 1.0
    assert "review" in render_term_adherence(result)
    assert "flagged for review, never counted as violations" in render_summary(result)


def test_per_term_adherence_is_the_only_rate(result):
    """Adherence is a share of that term's own R; its misses stay counts."""
    for row in term_rows(result):
        assert "ape_violation_rate" not in row
        if row["expected_instances"]:
            assert row["ape_adherence_rate"] == row["ape_adherent"] / row["expected_instances"]


def test_per_term_violations_are_substitutions_and_reconcile_with_the_totals(result):
    """Both fixture segments carry text, so every miss is a substitution and the per-term column
    sums to the dataset's violation count — the reconciliation the table exists to allow."""
    rows = term_rows(result)
    assert all(row["ape_omissions"] == 0 for row in rows)
    assert sum(row["ape_violations"] for row in rows) == result.post_edited.misses.substituted
    for row in rows:
        assert row["ape_violations"] + row["ape_omissions"] == (
            row["expected_instances"] - row["ape_adherent"]
        )
