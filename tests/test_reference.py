"""The human reference as the metric's denominator: what the MT and the post-edit are measured
against, and why a dataset cannot be scored without one."""

import pytest

from sourcecode.benchmark import Benchmark
from sourcecode.dataset import Dataset, DatasetError, parse_csv, validate
from sourcecode.glossary import GlossaryMatches
from sourcecode.normalize import normalize_language

GLOSSARY = {
    "The engine is electric.": [{"source_content": "engine", "target_content": "moteur"}],
    "Charge the battery.": [
        {"source_content": "battery", "target_content": "batterie"},
        {"source_content": "battery", "target_content": "accumulateur"},
    ],
}

SEGMENTS = [
    {
        "source_segment_id": "s1",
        "source_content": "The engine is electric.",
        "target_content": "Le bloc est électrique.",          # MT: wrong term
        "reference_content": "Le moteur est électrique.",     # human: correct
    },
    {
        "source_segment_id": "s2",
        "source_content": "Charge the battery.",
        "target_content": "Chargez la pile.",                 # MT: wrong term
        "reference_content": "Chargez la batterie.",          # human: correct
    },
]


class FakeStanza:
    def lemmatize_batch(self, texts, language):
        return list(texts)

    def lemmatize_batch_safe(self, texts, language):
        return list(texts)


class FakeGlossary:
    def fetch_matches(self, *, glossary_ids, source_language, target_language, texts, provider=None):
        per_text = [GLOSSARY.get(t, []) for t in texts]
        return GlossaryMatches(mappings=[m for g in per_text for m in g], per_text_mappings=per_text)


class RecordingPostMt:
    """Captures what was submitted, and post-edits only the first segment."""

    FIXES = {"The engine is electric.": "Le moteur est électrique."}

    def __init__(self):
        self.submitted = []

    def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
        from sourcecode.postmt import RunResult

        self.submitted.extend(segments)
        return RunResult(
            task_id="stub",
            error=None,
            segments=[
                {**s, "has_glossary": True,
                 "ape_results": {"text": self.FIXES.get(s["source_content"], s["target_content"])}}
                for s in segments
            ],
        )


class _BenchCfg:
    batch_size = 10
    lemma_matching = False


class _Cfg:
    benchmark = _BenchCfg()


@pytest.fixture
def dataset():
    return Dataset(
        name="against-the-reference",
        parameters=normalize_language(
            {"source_language": "English (United Kingdom)", "target_language": "French (France)"}
        ),
        glossary_ids=["tb1"],
        segments=[dict(s) for s in SEGMENTS],
    )


def test_reference_is_never_sent_to_postmt(dataset):
    """The corrected translation is the answer key — it must not reach the pipeline."""
    postmt = RecordingPostMt()
    Benchmark(postmt=postmt, stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg()).run(dataset)

    assert postmt.submitted, "nothing was submitted"
    for submitted in postmt.submitted:
        assert "reference_content" not in submitted
    # ...while the fields production does receive are intact.
    assert {s["source_content"] for s in postmt.submitted} == {
        "The engine is electric.", "Charge the battery."
    }
    assert all(s.get("target_content") for s in postmt.submitted)


def test_the_reference_sets_what_each_version_owed(dataset):
    result = Benchmark(
        postmt=RecordingPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg()
    ).run(dataset)

    # The human used one target term per segment, so two instances are owed.
    assert result.mt_baseline.expected == 2
    assert result.post_edited.expected == 2

    # MT reproduced neither; APE reproduced one.
    assert result.mt_baseline.adherent == 0
    assert result.post_edited.adherent == 1
    assert result.post_edited.adherence_rate == 0.5
    assert result.delta.adherence_rate == pytest.approx(0.5)


def test_a_term_the_human_avoided_is_not_held_against_the_pipeline():
    """The human deliberately did not use the glossary entry here, so it expects nothing. The
    pipeline applying it anyway is over-use — reported for review, never scored either way."""
    dataset = Dataset(
        name="human-avoided",
        parameters=normalize_language({"source_language": "en-gb", "target_language": "fr-fr"}),
        glossary_ids=["tb1"],
        segments=[
            {
                "source_segment_id": "s1",
                "source_content": "The engine is electric.",
                "target_content": "Le bloc est électrique.",
                "reference_content": "Le groupe est électrique.",
            }
        ],
    )
    result = Benchmark(
        postmt=RecordingPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg()
    ).run(dataset)

    assert result.mt_baseline.expected == 0
    # No denominator anywhere, so there is no rate to report rather than a 0% or a 100%.
    assert result.mt_baseline.adherence_rate is None
    assert result.post_edited.adherence_rate is None
    assert result.post_edited.terms.over_used == 1
    assert result.post_edited.violations == 0


def test_a_segment_without_a_reference_is_rejected(dataset):
    """There is no scoring path that does not need it, so the dataset is refused up front rather
    than producing a column the metric cannot fill."""
    dataset.segments[0].pop("reference_content")

    with pytest.raises(DatasetError, match="reference_content"):
        validate(dataset)


def test_csv_reference_column_aliases():
    for column in ("corrected", "reference", "post_edited", "human", "reference_content"):
        rows = parse_csv(f"source,mt,{column}\nThe engine.,Le bloc.,Le moteur.\n")
        assert rows[0]["reference_content"] == "Le moteur.", column
        assert rows[0]["target_content"] == "Le bloc."
