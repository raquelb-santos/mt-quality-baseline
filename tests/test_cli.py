"""CLI-level tests. Importing this module is itself a smoke test: cli.py has no other importer,
so a syntax error in it once survived a fully green suite.

Every run is configured through `.env` — `BENCH_DATASET` for what to score — so these set that
rather than passing a path as an argument."""

import json

import pytest

from sourcecode import cli


class _StubPostMt:
    """Healthy client that records whether anything was ever submitted."""

    def __init__(self):
        self.submitted = False
        self.authenticated = True
        self.active_task_id = None

    def health(self):
        return True

    def run(self, **kwargs):
        self.submitted = True
        raise AssertionError("preflight should have prevented this submission")

    def cancel_active(self):
        return False

    def close(self):
        pass


@pytest.fixture
def stub_postmt(monkeypatch):
    stub = _StubPostMt()
    monkeypatch.setattr(cli, "PostMtClient", lambda *a, **k: stub)
    return stub


class _StubGlossary:
    """Reachable term-bases index that records the ids it was asked for."""

    source_label = "term-bases-index"

    def __init__(self, *args, term_count=1, **kwargs):
        self.asked_for = None
        self._term_count = term_count

    def ping(self):
        return True

    def count_terms(self, glossary_ids, provider=None):
        return self._term_count

    def fetch_matches(self, *, glossary_ids, texts, **kwargs):
        from sourcecode.glossary import GlossaryMatches

        self.asked_for = list(glossary_ids)
        return GlossaryMatches(mappings=[], per_text_mappings=[[] for _ in texts])

    def close(self):
        pass


class _StubStanza:
    """Identity lemmatizer: keeps the tests off the network without changing what is matched."""

    def lemmatize_batch_safe(self, texts, language):
        return list(texts)

    def close(self):
        pass


@pytest.fixture(autouse=True)
def stub_stanza(monkeypatch):
    """Every CLI run builds a Stanza client, so stub it for all tests in this module."""
    monkeypatch.setattr(cli, "StanzaClient", lambda *a, **k: _StubStanza())


@pytest.fixture
def stub_glossary(monkeypatch):
    """The term-bases index is the only glossary source, so every run needs a reachable one."""
    glossary = _StubGlossary()
    monkeypatch.setenv("SEARCH_ENGINE_URL", "http://search.test")
    monkeypatch.setattr(cli, "GlossaryClient", lambda *a, **k: glossary)
    return glossary


def _write_dataset(folder, glossary_ids=("tb1",), name="d.json", **overrides):
    """A pinned dataset. Pass ``key=None`` to omit a parameter and trip the preflight."""
    parameters = {
        "cat_project_id": "P1",
        "cat_tool_provider": "MemSource",
        "ecosystem_id": "E1",
        "tempo_task_id": "T1",
        "source_language": "en-gb",
        "target_language": "fr-fr",
    }
    parameters.update(overrides)
    parameters = {k: v for k, v in parameters.items() if v is not None}

    path = folder / name
    path.write_text(
        json.dumps({
            "name": path.stem,
            "parameters": parameters,
            "glossary_ids": list(glossary_ids),
            "segments": [{"source_segment_id": "1", "source_content": "The engine.",
                          "target_content": "Le bloc.", "reference_content": "Le moteur."}],
        }),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def configure(monkeypatch, tmp_path):
    """Write a dataset and point BENCH_DATASET at it, as a real run would."""
    def _configure(**overrides):
        path = _write_dataset(tmp_path, **overrides)
        monkeypatch.setenv("BENCH_DATASET", str(path))
        return path
    return _configure


def test_preflight_blocks_submission_when_glossary_would_be_skipped(
    configure, stub_postmt, stub_glossary, capsys
):
    """A real run with no tempo_task_id/cat_tool_provider would cost money and measure nothing."""
    configure(tempo_task_id=None, cat_tool_provider=None)

    code = cli.main([])

    assert code == 1
    assert stub_postmt.submitted is False        # nothing was ever sent
    assert "Preflight failed" in capsys.readouterr().err


def test_dry_run_skips_preflight_and_never_touches_postmt(configure, stub_postmt, stub_glossary):
    configure(tempo_task_id=None, cat_tool_provider=None)

    assert cli.main(["--dry-run"]) == 0
    assert stub_postmt.submitted is False


def test_missing_search_engine_url_is_a_config_error_not_a_network_one(
    configure, stub_postmt, monkeypatch, capsys
):
    """The term-bases index is the only source; an unset URL must say so rather than
    reporting the empty string as unreachable."""
    configure()
    monkeypatch.setenv("SEARCH_ENGINE_URL", "")

    code = cli.main(["--dry-run"])

    assert code == 2
    assert "SEARCH_ENGINE_URL" in capsys.readouterr().err


def test_pinned_ids_are_the_ones_queried(configure, stub_postmt, stub_glossary):
    """Ids come from the dataset, so a run is a fixed experiment rather than a live lookup."""
    configure(glossary_ids=["tb1", "tb2"])

    assert cli.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == ["tb1", "tb2"]


def test_glossary_ids_absent_from_the_index_stop_the_run(
    configure, stub_postmt, monkeypatch, capsys
):
    """An id from another system matches nothing and would score a clean-looking zero."""
    configure(glossary_ids=["041cf63c-3f16-4d79-a386-35cf7688faf0"])
    monkeypatch.setattr(cli, "GlossaryClient", lambda *a, **k: _StubGlossary(term_count=0))
    monkeypatch.setenv("SEARCH_ENGINE_URL", "http://search.test")

    code = cli.main(["--dry-run"])

    assert code == 1
    error = capsys.readouterr().err
    assert "None of the glossary ids" in error
    assert "cluster post-mt queries" in error


def test_a_configured_run_needs_no_arguments_at_all(
    configure, stub_postmt, stub_glossary, capsys
):
    """The dataset comes from .env, so the command line carries only behaviour flags."""
    configure()

    assert cli.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == ["tb1"]
    assert "TERMINOLOGY ADHERENCE" in capsys.readouterr().out


def test_a_run_writes_nothing_to_disk(configure, stub_postmt, stub_glossary, tmp_path):
    """The report is printed and not kept, so a run leaves the filesystem as it found it."""
    dataset = configure()
    before = set(tmp_path.rglob("*"))

    assert cli.main(["--dry-run"]) == 0
    assert set(tmp_path.rglob("*")) == before == {dataset}


def test_no_dataset_configured_is_an_error_naming_the_variable(
    stub_postmt, stub_glossary, monkeypatch, tmp_path, caplog
):
    """Scoring nothing must not look like a clean run, and the error has to say how to fix it."""
    monkeypatch.setenv("BENCH_DATASET", "")

    assert cli.main(["--dry-run"]) == 1
    assert "BENCH_DATASET" in caplog.text


def test_a_folder_scores_every_dataset_in_it_and_pools_them(
    monkeypatch, tmp_path, stub_postmt, stub_glossary, capsys
):
    """BENCH_DATASET may name a folder; each dataset inside is scored and pooled by stratum."""
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="a.json")
    _write_dataset(folder, name="b.json")
    monkeypatch.setenv("BENCH_DATASET", str(folder))

    assert cli.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "BY STRATUM" in out
    assert "en-gb->fr-fr" in out
