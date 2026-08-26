"""Wiring tests for the orchestrator: batching, index alignment, MT-vs-APE comparison and
aggregation, all against fakes."""

from dataclasses import dataclass, field

import pytest

from sourcecode.benchmark import Benchmark
from sourcecode.postmt import Usage
from sourcecode.dataset import Dataset
from sourcecode.glossary import GlossaryMatches
from sourcecode.normalize import normalize_language

# Glossary: "brake pad" -> frein (strict); "engine" -> moteur (strict).
GLOSSARY = {
    "The brake pad is worn.": [{"source_content": "brake pad", "target_content": "frein"}],
    "The engine is electric.": [{"source_content": "engine", "target_content": "moteur"}],
    "No glossary terms here.": [],
}

SEGMENTS = [
    {"source_segment_id": "s1", "source_content": "The brake pad is worn.", "target_content": "La plaquette est usée.",
     "reference_content": "Le frein est usé."},
    {"source_segment_id": "s2", "source_content": "The engine is electric.", "target_content": "Le bloc est électrique.",
     "reference_content": "Le moteur est électrique."},
    {"source_segment_id": "s3", "source_content": "No glossary terms here.", "target_content": "Aucun terme ici.",
     "reference_content": "Aucun terme ici."},
]


class FakeStanza:
    def lemmatize_batch(self, texts, language):
        return list(texts)

    def lemmatize_batch_safe(self, texts, language):
        return list(texts)


class FakeGlossary:
    def ping(self):
        return True

    def fetch_matches(self, *, glossary_ids, source_language, target_language, texts, provider=None):
        per_text = [GLOSSARY.get(text, []) for text in texts]
        return GlossaryMatches(mappings=[m for group in per_text for m in group], per_text_mappings=per_text)


@dataclass
class FakeRunResult:
    task_id: str
    segments: list
    error: str | None
    usage: Usage = field(default_factory=Usage)


class FakePostMt:
    """APE fixes segment 2 ("bloc" -> "moteur") but leaves segment 1 violating."""

    FIXES = {"The engine is electric.": "Le moteur est électrique."}

    def __init__(self, fixes=None):
        self.fixes = self.FIXES if fixes is None else fixes
        self.batches_seen = []

    def health(self):
        return True

    def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
        self.batches_seen.append([s["source_segment_id"] for s in segments])
        enriched = [
            {
                **segment,
                "has_glossary": bool(GLOSSARY.get(segment["source_content"])),
                "ape_results": {
                    "text": self.fixes.get(segment["source_content"], segment["target_content"])
                },
            }
            for segment in segments
        ]
        return FakeRunResult(task_id="stub", segments=enriched, error=None)


@dataclass
class _BenchCfg:
    batch_size: int = 2
    lemma_matching: bool = False


@dataclass
class _Cfg:
    benchmark: _BenchCfg


@pytest.fixture
def dataset():
    return Dataset(
        name="wiring",
        parameters=normalize_language(
            {
                "source_language": "English (United Kingdom)",
                "target_language": "French (France)",
                "domain": "Automotive",
                "cat_tool_provider": "MemSource",
            }
        ),
        glossary_ids=["tb1"],
        segments=[dict(s) for s in SEGMENTS],
    )


@pytest.fixture
def benchmark():
    return Benchmark(
        postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    )


def test_end_to_end_scores_mt_baseline_against_post_edited(benchmark, dataset):
    result = benchmark.run(dataset)

    assert result.totals["segments"] == 3
    assert result.totals["segments_with_glossary"] == 2

    # MT baseline: both terms missing ("plaquette" not "frein", "bloc" not "moteur").
    assert result.mt_baseline.expected == 2
    assert result.mt_baseline.adherent == 0
    assert result.mt_baseline.misses.substituted == 2   # a count, not a rate

    # Post-edited: APE fixed one of the two.
    assert result.post_edited.adherent == 1
    assert result.post_edited.adherence_rate == 0.5

    assert result.delta.adherence_rate == 0.5
    assert result.delta.violations_resolved == 1


def test_segment_without_glossary_does_not_dilute_the_rate(benchmark, dataset):
    result = benchmark.run(dataset)
    clean = next(s for s in result.segments if s.source_segment_id == "s3")
    assert clean.mt.expected == 0
    assert clean.has_glossary_resolved is False


def test_batching_preserves_order_and_index_alignment(dataset):
    postmt = FakePostMt()
    benchmark = Benchmark(postmt=postmt, stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg()))
    result = benchmark.run(dataset)

    # batch_size 2 over 3 segments => two batches; misalignment would misattribute terms.
    assert postmt.batches_seen == [["s1", "s2"], ["s3"]]
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
        {"source_segment_id": "s1", "source_content": "The brake pad is worn.", "target_content": "Le frein est usé.",
         "reference_content": "Le frein est usé."},
        {"source_segment_id": "s2", "source_content": "The engine is electric.", "target_content": "Le bloc est électrique.",
         "reference_content": "Le moteur est électrique."},
    ]

    postmt = FakePostMt(
        fixes={
            "The engine is electric.": "Le moteur est électrique.",   # repaired
            "The brake pad is worn.": "La garniture est usée.",        # regressed away from "frein"
        }
    )
    benchmark = Benchmark(postmt=postmt, stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg()))
    result = benchmark.run(dataset)

    assert result.mt_baseline.adherent == 1
    assert result.post_edited.adherent == 1
    assert result.delta.adherence_rate == 0          # net change: nothing

    # ...but one term was repaired and a different one was broken.
    assert result.delta.terms_fixed_by_ape == 1
    assert result.delta.terms_regressed_by_ape == 1


def test_postmt_has_glossary_is_retained_alongside_our_resolution(benchmark, dataset):
    result = benchmark.run(dataset)
    for segment in result.segments:
        assert segment.has_glossary_reported == segment.has_glossary_resolved


def test_short_pipeline_response_realigns_by_index_without_crashing(dataset):
    class TruncatingPostMt(FakePostMt):
        def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
            full = super().run(parameters=parameters, segments=segments, steps=steps)
            return FakeRunResult(task_id=full.task_id, segments=full.segments[:1], error=None)

    benchmark = Benchmark(
        postmt=TruncatingPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    )
    result = benchmark.run(dataset)

    # Dropped segments fall back to their inputs rather than shifting every later index.
    assert [s.source_segment_id for s in result.segments] == ["s1", "s2", "s3"]
    assert result.totals["segments"] == 3


# ── skip_pipeline: dry run and full run share one code path ──────────────────

def test_skip_pipeline_never_contacts_postmt(dataset):
    class ExplodingPostMt:
        def run(self, **kwargs):
            raise AssertionError("post-mt must not be called when skip_pipeline=True")

    benchmark = Benchmark(
        postmt=ExplodingPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    )
    result = benchmark.run(dataset, skip_pipeline=True)

    assert result.steps == ["(dry-run)"]
    assert result.dataset.endswith("(dry-run)")


def test_skip_pipeline_scores_the_same_way_as_a_full_run(dataset):
    """A dry run and a full run must not disagree about a number."""
    stub_postmt = FakePostMt(fixes={})   # APE returns the MT unchanged
    full = Benchmark(
        postmt=stub_postmt, stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    ).run(dataset)

    dry = Benchmark(
        postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    ).run(dataset, skip_pipeline=True)

    assert dry.mt_baseline.expected == full.mt_baseline.expected
    assert dry.mt_baseline.adherent == full.mt_baseline.adherent
    assert dry.mt_baseline.adherence_rate == full.mt_baseline.adherence_rate
    # Post-edited mirrors the baseline when nothing was post-edited.
    assert dry.post_edited.adherent == dry.mt_baseline.adherent
    assert dry.totals["segments_changed_by_ape"] == 0


def test_skip_pipeline_still_measures_against_the_reference(dataset):
    """The denominator comes from the dataset, not from post-mt, so a dry run has the full one."""
    dry = Benchmark(
        postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    ).run(dataset, skip_pipeline=True)

    assert dry.mt_baseline.expected == 2
    assert [s.reference_content for s in dry.segments] == [s["reference_content"] for s in SEGMENTS]


def test_warns_when_postmt_saw_no_glossary_but_we_resolved_terms(dataset, caplog):
    """post-mt skips glossary retrieval silently on bad parameters, so a misconfigured run looks
    like "APE does not help terminology" unless it is called out."""
    class BlindPostMt(FakePostMt):
        def run(self, *, parameters, segments, steps=("AQE", "APE"), on_progress=None):
            result = super().run(parameters=parameters, segments=segments, steps=steps)
            for segment in result.segments:
                segment["has_glossary"] = False      # pipeline never retrieved anything
            return result

    Benchmark(
        postmt=BlindPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    ).run(dataset)

    assert "post-mt reported no glossary" in caplog.text
    assert "ecosystem_id" in caplog.text


def test_no_warning_when_the_two_agree(dataset, caplog):
    Benchmark(
        postmt=FakePostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    ).run(dataset)
    assert "post-mt reported no glossary" not in caplog.text


# ── a run that failed inside post-mt must not read as a clean measurement ─────

class FailingPostMt(FakePostMt):
    """post-mt on a missing required parameter: status "done", task error None, but every segment
    carrying an error and empty APE text."""

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
        return FakeRunResult(task_id="stub", segments=enriched, error=None)


def test_per_segment_failures_are_surfaced_not_scored_as_no_change(dataset, caplog):
    """The empty APE text falls back to raw MT, so every metric says "APE changed nothing"."""
    benchmark = Benchmark(
        postmt=FailingPostMt(), stanza=FakeStanza(), glossary=FakeGlossary(), config=_Cfg(_BenchCfg())
    )
    result = benchmark.run(dataset)

    # On the numbers alone this is indistinguishable from a clean run...
    assert result.totals["segments_changed_by_ape"] == 0
    assert result.delta.adherence_rate == 0.0

    # ...so only the failure count tells them apart.
    assert result.failed_segments == 3
    assert "tempo_task_id" in result.failure_reason
    assert "tempo_task_id" in caplog.text


def test_a_healthy_run_records_no_failures(benchmark, dataset):
    assert benchmark.run(dataset).failed_segments == 0
    assert benchmark.run(dataset).failure_reason is None


def test_dry_run_never_reports_post_mt_failures(benchmark, dataset):
    """A dry run does not call post-mt at all, so it cannot inherit a stale failure count."""
    assert benchmark.run(dataset, skip_pipeline=True).failed_segments == 0
