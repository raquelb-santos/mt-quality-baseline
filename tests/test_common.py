"""The parts both components share: matching, dataset loading, the HTTP clients, and the run."""

import json

import httpx
import pytest

from sourcecode import run
from sourcecode import dnt
from sourcecode.text_processing import (
    count_lemma,
    count_occurrences,
    count_surface,
    is_unspaced_language,
    load,
    normalize_language,
    normalize_text,
    parse_csv,
    parse_mxliff,
)
from sourcecode.dnt import DntClient, Reversion
from sourcecode.dnt import Reversion
from sourcecode.dnt_score import count_item
from sourcecode.glossary import GlossaryClient
from sourcecode.postmt import (
    extract_post_edited,
    preflight_parameters,
    reported_has_glossary,
    segment_error,
)


# ================================================================================================
# matching and counting
# ================================================================================================

# --- does the term appear at all -------------------------------------------------------------------

def test_word_boundary_prevents_substring_false_positives():
    assert count_surface("a category of things", "cat", "en-us") == 0
    assert count_surface("the cat sat", "cat", "en-us") == 1


def test_accents_compare_under_nfc():
    decomposed = "le café est ouvert"   # e + combining acute
    composed = "café"                     # é
    assert count_surface(decomposed, composed, "fr-fr") == 1


def test_accented_word_boundary_is_respected():
    # "café" must not match inside "cafés" on surface form alone.
    assert count_surface("trois cafés ouverts", "café", "fr-fr") == 0


def test_regex_metacharacters_in_terms_are_literal():
    assert count_surface("the C++ compiler", "C++", "en-us") == 1
    assert count_surface("the C-- compiler", "C++", "en-us") == 0


def test_terms_with_trailing_punctuation_still_match():
    # This is why \b is unusable: it needs a \w/\W transition, which "C++ " does not provide.
    assert count_surface("use C++ here", "c++", "en-us") == 1


def test_unspaced_languages_fall_back_to_containment():
    assert is_unspaced_language("ja-jp") is True
    assert count_surface("自動車のエンジン", "エンジン", "ja-jp") == 1


def test_casefold_handles_non_ascii_case():
    assert count_surface("DIE STRASSE", "strasse", "de-de") == 1


def test_normalize_text_collapses_whitespace():
    assert normalize_text("  a   b \n c ") == "a b c"
    assert normalize_text(None) == ""


def test_lemma_match_requires_contiguous_run():
    assert count_lemma(["le", "moteur", "electrique", "etre"], ["moteur", "electrique"]) == 1
    assert count_lemma(["le", "moteur", "etre"], ["moteur", "electrique"]) == 0
    # non-contiguous must not match
    assert count_lemma(["moteur", "de", "electrique"], ["moteur", "electrique"]) == 0


def test_lemma_match_accepts_strings_or_sequences():
    assert count_lemma("le moteur electrique", "moteur electrique") == 1


def test_lemma_fallback_is_skipped_for_unspaced_languages():
    # Token alignment is meaningless without word separation; must not report a lemma match.
    assert count_occurrences(
        text="全然違う", term="エンジン", language_code="ja-jp",
        text_lemmas="全然 違う", term_lemmas="エンジン",
    ) == 0


# --- how many times it appears ----------------------------------------------------------------------
# Double-counting one rendering would read as consistency that is not there.

def test_count_surface_is_boundary_aware():
    assert count_surface("the engine and the engine", "engine", "en-gb") == 2
    # must not count inside a longer word
    assert count_surface("engineering engines", "engine", "en-gb") == 0


def test_unspaced_counting_is_non_overlapping():
    assert count_surface("エンジンとエンジン", "エンジン", "ja-jp") == 2


def test_count_lemma_is_non_overlapping():
    assert count_lemma(["a", "a", "a"], ["a", "a"]) == 1  # not 2
    assert count_lemma(["moteur", "x", "moteur"], ["moteur"]) == 2


def test_count_occurrences_falls_back_to_lemmas():
    assert count_occurrences(
        text="les moteurs electriques", term="moteur electrique", language_code="fr-fr",
        text_lemmas="le moteur electrique", term_lemmas="moteur electrique",
    ) == 1


def test_surface_and_lemma_never_double_count():
    # An uninflected match is found on the surface; the lemma pass must not add to it.
    assert count_occurrences(
        text="le moteur", term="moteur", language_code="fr-fr",
        text_lemmas="le moteur", term_lemmas="moteur",
    ) == 1


# --- languages --------------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "source,target,expected_source,expected_target",
    [
        ("English (United Kingdom)", "French (France)", "en-gb", "fr-fr"),
        ("en-us", "de-de", "en-us", "de-de"),
        ("Chinese (Simplified, China)", "Japanese (Japan)", "zh-cn", "ja-jp"),
    ],
)
def test_language_names_and_codes_both_resolve(source, target, expected_source, expected_target):
    out = normalize_language({"source_language": source, "target_language": target})
    assert out["clean_source_language_code"] == expected_source
    assert out["clean_target_language_code"] == expected_target


def test_unknown_languages_pass_through_rather_than_raising():
    out = normalize_language({"source_language": "xx-yy", "target_language": "Klingon"})
    assert out["clean_source_language_code"] == "xx-yy"
    assert out["clean_target_language_code"] == "klingon"


# --- verbatim counting, for DNT -----------------------------------------------------------------------
# `count_item` is the odd one out: everything above casefolds and it does not.

def test_count_item_is_case_sensitive_unlike_neighbours():
    assert count_item("Le iPhone est ici.", "iPhone", "fr-fr") == 1
    assert count_item("Le iphone est ici.", "iPhone", "fr-fr") == 0
    assert count_surface("Le iphone est ici.", "iPhone", "fr-fr") == 1


def test_count_item_has_no_lemma_fallback():
    """There is no parameter to pass one: a lemma-matched DNT item is by definition a leak."""
    import inspect

    assert "lemma" not in str(inspect.signature(count_item))


def test_count_item_inherits_underscore_boundary():
    """`_` is excluded from the boundary class, so it does not separate words."""
    assert count_item("foo_BAR here", "BAR", "en-gb") == 1


# ================================================================================================
# dataset loading
# ================================================================================================

PARAMETERS = {
    "cat_project_id": "P1",
    "cat_tool_provider": "MemSource",
    "source_language": "English (United Kingdom)",
    "target_language": "French (France)",
}


def _write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_load_json_normalizes_languages(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "name": "demo",
                "parameters": PARAMETERS,
                "glossary_ids": ["tb1"],
                "segments": [{"source_segment_id": "1", "source_content": "a", "target_content": "b",
                             "reference_content": "c"}],
            }
        ),
    )
    dataset = load(path, component="glossary")
    assert dataset.name == "demo"
    assert dataset.parameters["clean_source_language_code"] == "en-gb"
    assert dataset.parameters["clean_target_language_code"] == "fr-fr"
    assert dataset.steps == ["AQE", "APE"]


def test_sidecar_takes_precedence_over_file(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "parameters": PARAMETERS,
                "glossary_ids": ["from-file"],
                "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}],
            }
        ),
    )
    _write(tmp_path, "d.params.json", json.dumps({"glossary_ids": ["from-sidecar"], "steps": ["AQE"]}))
    dataset = load(path, component="glossary")
    assert dataset.glossary_ids == ["from-sidecar"]
    assert dataset.steps == ["AQE"]


def test_missing_glossary_ids_is_rejected(tmp_path):
    """A dataset without ids cannot be scored, so it must fail rather than report a clean zero."""
    path = _write(
        tmp_path, "d.json",
        json.dumps({"parameters": PARAMETERS, "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}]}),
    )
    with pytest.raises(ValueError, match="glossary_ids"):
        load(path, component="glossary")


def test_blank_segment_fields_are_rejected(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "parameters": PARAMETERS,
                "glossary_ids": ["tb1"],
                "segments": [{"source_content": "a", "target_content": "   ",
                              "reference_content": "c"}],
            }
        ),
    )
    with pytest.raises(ValueError, match="target_content"):
        load(path, component="glossary")


def test_missing_languages_are_rejected(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "parameters": {"cat_project_id": "P1"},
                "glossary_ids": ["tb1"],
                "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}],
            }
        ),
    )
    with pytest.raises(ValueError, match="source_language"):
        load(path, component="glossary")


def test_parse_csv_accepts_short_and_long_column_names():
    rows = parse_csv("source,target\nhello,bonjour\n")
    assert rows[0]["source_content"] == "hello"
    assert rows[0]["target_content"] == "bonjour"

    rows = parse_csv("source_segment_id,source_content,target_content\n7,hello,bonjour\n")
    assert rows[0]["source_segment_id"] == "7"


def test_parse_csv_handles_quoted_fields_with_commas():
    rows = parse_csv('source_content,target_content\n"a, b","c, d"\n')
    assert rows[0]["source_content"] == "a, b"
    assert rows[0]["target_content"] == "c, d"


def test_load_csv_takes_parameters_from_sidecar(tmp_path):
    path = _write(tmp_path, "d.csv", "source,target,reference\nhello,bonjour,salut\n")
    _write(tmp_path, "d.params.json", json.dumps({"parameters": PARAMETERS, "glossary_ids": ["tb1"]}))
    dataset = load(path, component="glossary")
    assert len(dataset.segments) == 1
    assert dataset.parameters["clean_target_language_code"] == "fr-fr"


MXLIFF = """<?xml version="1.0" encoding="utf-8"?>
<xliff version="1.2" xmlns="urn:oasis:names:tc:xliff:document:1.2">
  <file source-language="en-gb" target-language="fr-fr">
    <body>
      <trans-unit id="1"><source>The brake pad.</source><target>La plaquette de frein.</target>
        <alt-trans origin="machine-trans"><target>La plaquette.</target></alt-trans></trans-unit>
      <trans-unit id="2"><source>Empty target.</source><target></target>
        <alt-trans origin="machine-trans"><target>Cible vide.</target></alt-trans></trans-unit>
      <trans-unit id="3"><source>Accented</source><target>Café</target>
        <alt-trans origin="machine-trans"><target>Cafe</target></alt-trans></trans-unit>
      <trans-unit id="4"><source>Confirmed by hand.</source><target>Saisi à la main.</target></trans-unit>
    </body>
  </file>
</xliff>
"""


def test_parse_mxliff_reads_proposal():
    segments = parse_mxliff(MXLIFF)
    # Unit 2 has an empty target and unit 4, typed by hand, carries no machine proposal.
    assert segments[0]["target_content"] == "La plaquette."
    assert segments[0]["reference_content"] == "La plaquette de frein."
    assert segments[1]["target_content"] == "Cafe"
    assert segments[1]["reference_content"] == "Café"


def test_load_mxliff_end_to_end(tmp_path):
    path = _write(tmp_path, "job.mxliff", MXLIFF)
    _write(tmp_path, "job.params.json", json.dumps({"parameters": PARAMETERS, "glossary_ids": ["tb1"]}))
    dataset = load(path, component="glossary")
    assert dataset.name == "job"
    assert len(dataset.segments) == 2


# ================================================================================================
# the HTTP boundary
# ================================================================================================

# ══════════════════════════════════════════════════════════════════════════════
# Term bases - pinned queries: drift means scoring against a glossary production never sent.
# ══════════════════════════════════════════════════════════════════════════════

PERCOLATE = {
    "responses": [
        {"hits": {"hits": [
            {"_source": {"term_text": "brake pad", "concept_id": "c1"}},
            {"_source": {"term_text": "engine", "concept_id": "c2"}},
        ]}},
        {"hits": {"hits": [{"_source": {"term_text": "engine", "concept_id": "c2"}}]}},
    ]
}

CONCEPTS = {
    "hits": {"hits": [
        {"_source": {"concept_id": "c1", "term_text": "plaquette de frein"}},
        {"_source": {"concept_id": "c2", "term_text": "moteur"}},
        {"_source": {"concept_id": "c2", "term_text": "bloc moteur"}},
    ]}
}

TEXTS = ["The brake pad and the engine.", "The engine only."]


def _glossary_client(percolate=PERCOLATE, concepts=CONCEPTS, capture=None):
    def handler(request):
        path = request.url.path
        if path.endswith("/_msearch"):
            if capture is not None:
                capture["msearch_path"] = path
                capture["ndjson"] = [
                    json.loads(line)
                    for line in request.content.decode("utf-8").splitlines()
                    if line.strip()
                ]
                capture["content_type"] = request.headers.get("Content-Type")
            return httpx.Response(200, json=percolate)
        if capture is not None:
            capture["search_path"] = path
            capture["search_body"] = json.loads(request.content)
        return httpx.Response(200, json=concepts)

    client = GlossaryClient("http://search.test")
    client._client = httpx.Client(
        base_url="http://search.test", transport=httpx.MockTransport(handler)
    )
    return client


def _fetch(client, **overrides):
    kwargs = dict(
        glossary_ids=["tb1"], source_language="en-gb", target_language="fr-fr", texts=TEXTS
    )
    kwargs.update(overrides)
    return client.fetch_matches(**kwargs)


# ── request shape, pinned to what production sends ───────────────────────────

def test_percolate_request_matches_post_mt():
    capture = {}
    _fetch(_glossary_client(capture=capture))

    assert capture["content_type"] == "application/x-ndjson"
    # One index header + one body per text, in order.
    assert len(capture["ndjson"]) == 2 * len(TEXTS)

    header, body = capture["ndjson"][0], capture["ndjson"][1]
    assert header == {"index": "term-bases"}
    assert body["size"] == 50
    assert body["sort"] == ["_score"]
    assert body["query"]["bool"]["must"] == [
        {"percolate": {"field": "query", "document": {"content": TEXTS[0]}}}
    ]
    assert {"terms": {"glossary_id": ["tb1"]}} in body["query"]["bool"]["filter"]
    # Permissive language matching: full code and base code, as post-mt does.
    languages = next(
        f["terms"]["language"] for f in body["query"]["bool"]["filter"] if "language" in f["terms"]
    )
    assert languages == ["en-gb", "en"]

    assert capture["ndjson"][3]["query"]["bool"]["must"][0]["percolate"]["document"]["content"] == TEXTS[1]


def test_concept_lookup_matches_post_mt():
    capture = {}
    _fetch(_glossary_client(capture=capture))

    body = capture["search_body"]
    assert capture["search_path"] == "/term-bases/_search"
    assert body["size"] == 1000
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"concept_id": ["c1", "c2"]}} in filters
    assert {"terms": {"language": ["fr-fr", "fr"]}} in filters


def test_xtm_provider_uses_xtm_index():
    capture = {}
    _fetch(_glossary_client(capture=capture), provider="XTM")
    assert capture["ndjson"][0] == {"index": "xtm-term-bases"}
    assert capture["search_path"] == "/xtm-term-bases/_search"


def test_non_xtm_providers_use_default_index():
    capture = {}
    _fetch(_glossary_client(capture=capture), provider="MemSource")
    assert capture["ndjson"][0] == {"index": "term-bases"}


def test_glossary_ids_are_trimmed_and_blanks_dropped():
    capture = {}
    _fetch(_glossary_client(capture=capture), glossary_ids=[" tb1 ", "", "tb2"])
    body = capture["ndjson"][1]
    assert {"terms": {"glossary_id": ["tb1", "tb2"]}} in body["query"]["bool"]["filter"]


# ── result assembly ──────────────────────────────────────────────────────────

def test_per_text_mappings_align_with_texts():
    matches = _fetch(_glossary_client())

    assert len(matches.per_text_mappings) == len(TEXTS)
    assert matches.per_text_mappings[0] == [
        {"source_content": "brake pad", "target_content": "plaquette de frein"},
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "engine", "target_content": "bloc moteur"},
    ]
    # Second text matched only the engine concept.
    assert {m["source_content"] for m in matches.per_text_mappings[1]} == {"engine"}


def test_targets_are_deduplicated_per_text():
    percolate = {
        "responses": [
            {"hits": {"hits": [
                {"_source": {"term_text": "engine", "concept_id": "c2"}},
                {"_source": {"term_text": "motor", "concept_id": "c2"}},
            ]}},
            {"hits": {"hits": []}},
        ]
    }
    matches = _fetch(_glossary_client(percolate=percolate))
    # Both source terms resolve to the same concept, so each target appears once.
    assert [m["target_content"] for m in matches.per_text_mappings[0]] == ["moteur", "bloc moteur"]


def test_global_mappings_are_deduplicated_across_texts():
    matches = _fetch(_glossary_client())
    assert [m["target_content"] for m in matches.mappings] == [
        "plaquette de frein", "moteur", "bloc moteur"
    ]


def test_no_percolate_hits_skips_concept_lookup():
    empty = {"responses": [{"hits": {"hits": []}}, {"hits": {"hits": []}}]}
    capture = {}
    matches = _fetch(_glossary_client(percolate=empty, capture=capture))

    assert matches.mappings == []
    assert matches.per_text_mappings == [[], []]
    assert "search_body" not in capture  # second query never issued


def test_percolate_error_spares_rest(caplog):
    partial = {
        "responses": [
            {"error": {"type": "search_phase_execution_exception"}},
            {"hits": {"hits": [{"_source": {"term_text": "engine", "concept_id": "c2"}}]}},
        ]
    }
    matches = _fetch(_glossary_client(percolate=partial))

    assert matches.per_text_mappings[0] == []
    assert [m["target_content"] for m in matches.per_text_mappings[1]] == ["moteur", "bloc moteur"]


# ── validation ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"glossary_ids": []}, "No glossary IDs provided"),
        ({"source_language": ""}, "No source language provided"),
        ({"target_language": ""}, "No target language provided"),
        ({"texts": []}, "No texts provided"),
    ],
)
def test_missing_inputs_are_rejected(overrides, message):
    with pytest.raises(ValueError, match=message):
        _fetch(_glossary_client(), **overrides)


# ── term counting, backing the CLI preflight ─────────────────────────────────

def test_count_terms_asks_right_index_for_ids():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"count": 42})

    client = GlossaryClient("http://search.test")
    client._client = httpx.Client(
        base_url="http://search.test", transport=httpx.MockTransport(handler)
    )

    assert client.count_terms(["tb1", "tb2"]) == 42
    assert seen["path"] == "/term-bases/_count"
    assert seen["body"] == {"query": {"terms": {"glossary_id": ["tb1", "tb2"]}}}
    assert client.count_terms(["tb1"], provider="XTM") == 42
    assert seen["path"] == "/xtm-term-bases/_count"

# ══════════════════════════════════════════════════════════════════════════════
# post-mt - which text gets scored, and what would make the number meaningless.
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize(
    "segment,expected",
    [
        ({"ape_results": {"text": "post-edited"}, "target_content": "mt"}, "post-edited"),
        ({"aped_text": "legacy", "target_content": "mt"}, "legacy"),
        # APE legitimately left it alone: fall back to MT rather than biasing the post-edit number.
        ({"ape_results": {"text": None}, "target_content": "mt"}, "mt"),
        ({"target_content": "mt"}, "mt"),
        ({}, ""),
    ],
)
def test_extract_post_edited_handles_every_pipeline_version(segment, expected):
    assert extract_post_edited(segment) == expected


# ── preflight: refuse runs that would cost money and measure nothing ──────────

GOOD = {
    "cat_project_id": "P1",
    "cat_tool_provider": "MemSource",
    "ecosystem_id": "001Aa00000jloVrIAI_XXX",
    "tempo_task_id": "task-1",
}


def test_preflight_passes_on_complete_parameters():
    assert preflight_parameters(GOOD) == []
    assert preflight_parameters({**GOOD, "cat_tool_provider": "XTM"}) == []


def test_preflight_catches_each_silent_skip_condition():
    """Each of these makes post-mt retrieve no glossary at all, without raising an error."""
    missing_ecosystem = preflight_parameters({**GOOD, "ecosystem_id": ""})
    assert any("ecosystem_id" in p for p in missing_ecosystem)

    missing_provider = preflight_parameters({**GOOD, "cat_tool_provider": ""})
    assert any("cat_tool_provider" in p for p in missing_provider)

    unsupported = preflight_parameters({**GOOD, "cat_tool_provider": "Trados"})
    assert any("does not support" in p for p in unsupported)

    missing_project = preflight_parameters({**GOOD, "cat_project_id": ""})
    assert any("cat_project_id" in p for p in missing_project)


def test_preflight_reports_every_problem_at_once():
    problems = preflight_parameters({})
    # tempo_task_id, provider, ecosystem, project — not just the first
    assert len(problems) == 4


def test_preflight_requires_tempo_task_id():
    """Without it post-mt fails every segment before any step runs, returning no post-edited text."""
    problems = preflight_parameters({**GOOD, "tempo_task_id": ""})
    assert any("tempo_task_id" in p for p in problems)


# ── per-segment failures ─────────────────────────────────────────────────────

def test_segment_error_is_found_on_ape_and_aqe():
    assert segment_error({"ape_results": {"text": "", "error": "boom"}}) == "boom"
    assert segment_error({"aqe_results": {"error": "assessment failed"}}) == "assessment failed"
    assert segment_error({"ape_results": {"text": "ok"}}) is None
    assert segment_error({}) is None


def test_failed_ape_looks_untouched():
    """post-mt returns empty APE text beside an error, and the fallback yields the raw MT."""
    failed = {"target_content": "raw mt", "ape_results": {"text": "", "error": "boom"}}
    assert extract_post_edited(failed) == "raw mt"      # identical to an untouched segment...
    assert segment_error(failed) == "boom"              # ...so only this can tell them apart


# ── has_glossary ─────────────────────────────────────────────────────────────

def test_has_glossary_is_read_from_aqe_results():
    """post-mt nests it under aqe_results; the top level left the warning permanently dead."""
    assert reported_has_glossary({"aqe_results": {"has_glossary": True}}) is True
    assert reported_has_glossary({"aqe_results": {"has_glossary": False}}) is False
    assert reported_has_glossary({"has_glossary": True}) is True      # legacy top-level
    assert reported_has_glossary({"aqe_results": {}}) is None
    assert reported_has_glossary({}) is None

# ══════════════════════════════════════════════════════════════════════════════
# The DNT service - request shape, batching, and the normalizers that accept its response.
# ══════════════════════════════════════════════════════════════════════════════

def _dnt_client(handler, api_key="k"):
    """Reach into the private client, as the glossary tests do: the transport is the seam."""
    client = DntClient("http://dnt.test", api_key)
    client._client = httpx.Client(
        base_url="http://dnt.test",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": api_key} if api_key else {},
    )
    return client


PAIR = {"id": "0", "source": "AcoladPro is here.", "target": "Le Pro Acolad est ici."}


# --- the header, which is the likeliest thing to get wrong -------------------------------------

def test_api_key_header_spelling():
    """post-mt sends `X-API-KEY`, DNT wants `X-Api-Key`; invisible until every call 401s."""
    captured = {}

    def handler(request):
        captured["headers"] = dict(request.headers)
        return httpx.Response(200, json={"segments": [{"terms": []}]})

    client = DntClient("http://dnt.test", "secret")
    client._client = httpx.Client(
        base_url="http://dnt.test",
        transport=httpx.MockTransport(handler),
        headers={"X-Api-Key": "secret"},
    )
    client.revert([PAIR], batch_size=10)

    assert captured["headers"]["x-api-key"] == "secret"


# --- request shape ------------------------------------------------------------------------------

def test_revert_posts_source_and_target_pairs_to_v1_revert():
    captured = {}

    def handler(request):
        captured["path"] = request.url.path
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"segments": [{"terms": []}]})

    _dnt_client(handler).revert([PAIR], batch_size=10)

    assert captured["path"] == "/v1/revert"
    assert captured["body"] == {
        "segments": [
            {"id": "0", "source": "AcoladPro is here.", "target": "Le Pro Acolad est ici."}
        ]
    }


def test_segments_are_identified_by_index():
    """mxliff units can have no id and CSV ids can repeat, so the index is the only reliable key."""
    captured = {}

    def handler(request):
        captured["ids"] = [s["id"] for s in json.loads(request.content)["segments"]]
        return httpx.Response(200, json={"segments": [{"terms": []}] * 3})

    pairs = [{"id": str(i), "source": "s", "target": "t"} for i in range(3)]
    _dnt_client(handler).revert(pairs, batch_size=10)

    assert captured["ids"] == ["0", "1", "2"]


def test_batching_splits_call_and_keeps_every_segment():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["segments"]))
        return httpx.Response(200, json={"segments": [{"terms": []}] * len(body["segments"])})

    pairs = [{"id": str(i), "source": "s", "target": "t"} for i in range(7)]
    results = _dnt_client(handler).revert(pairs, batch_size=3)

    assert calls == [3, 3, 1]
    assert len(results) == 7


def test_no_pairs_makes_no_call_at_all():
    def handler(request):
        raise AssertionError("should not have been called")

    assert _dnt_client(handler).revert([], batch_size=10) == []


# --- failure -------------------------------------------------------------------------------------

def test_failed_batch_yields_none():
    """None leaves the denominator, no items scores nothing: the two must stay distinguishable."""
    def handler(request):
        return httpx.Response(500)

    results = _dnt_client(handler).revert([PAIR], batch_size=10)

    assert results == [None]


def test_only_failed_batch_is_lost():
    state = {"calls": 0}

    def handler(request):
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, json={"segments": [{"terms": ["A"], "corrected_text": "x"}]})

    pairs = [{"id": str(i), "source": "s", "target": "t"} for i in range(2)]
    results = _dnt_client(handler).revert(pairs, batch_size=1)

    assert results[0] is None
    assert results[1].items == ["A"]


def test_short_response_realigns():
    def handler(request):
        return httpx.Response(200, json={"segments": [{"terms": ["A"], "corrected_text": "x"}]})

    pairs = [{"id": str(i), "source": "s", "target": "t"} for i in range(3)]
    results = _dnt_client(handler).revert(pairs, batch_size=10)

    assert len(results) == 3
    assert results[0].items == ["A"]
    assert results[1] is None and results[2] is None


# --- health --------------------------------------------------------------------------------------

def test_health_is_false_when_service_cannot_be_reached():
    def handler(request):
        raise httpx.ConnectError("no route", request=request)

    assert _dnt_client(handler).health() is False


def test_health_is_false_on_rejected_key():
    """`/health` takes no key, so probing only that calls a run healthy until every segment 401s."""
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(401)

    assert _dnt_client(handler).health() is False


def test_health_is_false_on_redirect():
    """A redirect downgrades a POST to a GET, so the revert calls would silently do nothing."""
    def handler(request):
        return httpx.Response(301, headers={"location": "https://elsewhere.test/health"})

    assert _dnt_client(handler).health() is False


def test_health_is_true_when_both_probes_pass():
    def handler(request):
        return httpx.Response(200, json={"ok": True})

    assert _dnt_client(handler).health() is True


# --- the response normalizers --------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (["AcoladPro"], ["AcoladPro"]),
    (["  AcoladPro  "], ["AcoladPro"]),
    ([{"text": "AcoladPro"}], []),   # `terms` is an array of strings; anything else is dropped
    ([None], []),
])
def test_items_are_read_as_strings(raw, expected):
    assert dnt.parse_reversion({"terms": raw, "result": "x"}, "sent").items == expected


def test_corrected_text_comes_from_result():
    assert dnt.parse_reversion({"result": "fixed"}, "sent").rev_text == "fixed"


def test_corrected_text_falls_back_to_what_was_sent():
    """An empty string would read as a segment that lost its translation."""
    assert dnt.parse_reversion({}, "sent").rev_text == "sent"


def test_reverted_is_not_corrected_text():
    """`reverted` is a list of item strings; reading it as the text would make reversion a no-op."""
    segment = {"result": "AcoladPro est ici.", "reverted": ["AcoladPro"]}

    assert dnt.parse_reversion(segment, "Le Pro Acolad est ici.").rev_text == "AcoladPro est ici."


def test_repairs_only_list_is_flagged():
    """A repairs-only list has no preserved items, so its rate measures the service's fix rate."""
    reversion = dnt.parse_reversion({"reverted": ["A"], "result": "x"}, "sent")

    assert reversion.items_are_repairs_only is True
    assert reversion.items == ["A"]


def test_full_item_list_is_not_flagged():
    reversion = dnt.parse_reversion({"terms": ["A", "B"], "result": "x"}, "sent")

    assert reversion.items_are_repairs_only is False
    assert reversion.items == ["A", "B"]


def test_repairs_only_response_is_logged_loudly(caplog):
    def handler(request):
        return httpx.Response(200, json={"results": [{"reverted": ["A"], "result": "x"}]})

    _dnt_client(handler).revert([PAIR], batch_size=10)

    assert "only the items the service repaired" in caplog.text


@pytest.mark.parametrize("body, count", [
    ({"segments": [{"terms": []}, {"terms": []}]}, 2),
    ({"results": [{"terms": []}]}, 1),
    ({"data": [{"terms": []}]}, 1),
    ([{"terms": []}, {"terms": []}], 2),
    ({"terms": [], "corrected_text": "x"}, 1),      # a single segment with no envelope
    ({"unrelated": 1}, 0),
])
def test_segments_are_found_in_any_envelope(body, count):
    assert len(dnt.response_segments(body)) == count


# --- what the service's own smoke test pins ------------------------------------------------------
# These follow the service's documented verification calls, so the shapes below are the real ones.

def test_detect_envelope_is_understood():
    """The service answers `results[0].terms`, not `segments[0].terms`."""
    body = {"results": [{"id": "test-1", "terms": ["Microsoft Azure"]}]}

    entries = dnt.response_segments(body)

    assert len(entries) == 1
    assert dnt.parse_reversion(entries[0], "").items == ["Microsoft Azure"]


def test_health_fails_when_llm_gateway_is_down():
    """Finding out per batch would spend the dataset before reporting it."""
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"cache": "connected", "llm_gateway": "disconnected"})
        return httpx.Response(200, json={})

    assert _dnt_client(handler).health() is False


def test_health_fails_when_cache_is_down():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"cache": "disconnected", "llm_gateway": "connected"})
        return httpx.Response(200, json={})

    assert _dnt_client(handler).health() is False


def test_health_passes_when_both_dependencies_are_connected():
    def handler(request):
        if request.url.path == "/health":
            return httpx.Response(200, json={"cache": "connected", "llm_gateway": "connected"})
        return httpx.Response(200, json={})

    assert _dnt_client(handler).health() is True


def test_missing_dependency_fields_pass():
    """Catch a known-bad state, not a schema: a service reporting differently is not broken."""
    def handler(request):
        return httpx.Response(200, json={"status": "ok"})

    assert _dnt_client(handler).health() is True


def test_non_json_health_body_passes():
    def handler(request):
        return httpx.Response(200, text="alive")

    assert _dnt_client(handler).health() is True


def test_documented_revert_response():
    """One RevertSegmentResult as `/openapi.json` describes it, so schema drift surfaces here."""
    def handler(request):
        return httpx.Response(200, json={
            "results": [{
                "id": "0",
                "source": "AcoladPro is here.",
                "target": "Le Pro Acolad est ici.",
                "result": "AcoladPro est ici.",
                "terms": ["AcoladPro"],
                "reverted": ["AcoladPro"],
                "unresolved": [],
                "method": "llm",
                "changed": True,
            }],
            "metadata": {"reversion_prompt_version": "v2", "reversion_prompt_hash": "abc",
                         "model": "azure/gpt-4.1-mini", "processing_time_ms": 812,
                         "stats": {"total": 1, "reverted": 1, "casing": 0, "llm": 1,
                                   "unresolved": 0}},
        })

    [reversion] = _dnt_client(handler).revert([PAIR], batch_size=10)

    assert reversion.rev_text == "AcoladPro est ici."
    assert reversion.items == ["AcoladPro"]
    assert reversion.items_are_repairs_only is False


def test_language_pair_drops_region():
    """The service names languages without a region, as its own examples do."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"terms": [], "result": "x"}]})

    _dnt_client(handler).revert(
        [PAIR], batch_size=10, source_language="en-gb", target_language="fr-fr"
    )

    assert captured["body"]["options"] == {"source_language": "en", "target_language": "fr"}


def test_no_language_configured_sends_no_options_block():
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"terms": [], "result": "x"}]})

    _dnt_client(handler).revert([PAIR], batch_size=10)

    assert "options" not in captured["body"]


# ================================================================================================
# the run
# ================================================================================================

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
    monkeypatch.setattr(run, "PostMtClient", lambda *a, **k: stub)
    return stub


class _StubGlossary:
    """Reachable term-bases index that records the ids it was asked for."""


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


class _StubDnt:
    """A reachable DNT service naming one item per segment, with the health probes the CLI needs."""


    def __init__(self, items=("TimberLine",)):
        self.items = list(items)
        self.authenticated = True

    def health(self):
        return True

    def revert(self, pairs, *, batch_size, source_language=None, target_language=None):
        return [Reversion(rev_text=pair["target"], items=self.items) for pair in pairs]

    def close(self):
        pass


@pytest.fixture(autouse=True)
def stub_stanza(monkeypatch):
    """Every CLI run building a glossary client builds a Stanza one, so stub it module-wide."""
    monkeypatch.setattr(run, "StanzaClient", lambda *a, **k: _StubStanza())


@pytest.fixture(autouse=True)
def measure_glossary_only(monkeypatch):
    """The default for this module, so no test reaches a service it did not ask for."""
    monkeypatch.setenv("BENCH_COMPONENT", "glossary")


@pytest.fixture
def stub_glossary(monkeypatch):
    """The term-bases index is the only glossary source, so every run needs a reachable one."""
    glossary = _StubGlossary()
    monkeypatch.setenv("SEARCH_ENGINE_URL", "http://search.test")
    monkeypatch.setattr(run, "GlossaryClient", lambda *a, **k: glossary)
    return glossary


@pytest.fixture
def stub_dnt(monkeypatch):
    dnt = _StubDnt()
    monkeypatch.setenv("DNT_BASE_URL", "http://dnt.test")
    monkeypatch.setattr(run, "DntClient", lambda *a, **k: dnt)
    return dnt


SEGMENT = {
    "source_segment_id": "1",
    "source_content": "The TimberLine engine.",
    "target_content": "Le bloc TimberLine.",
    "reference_content": "Le moteur TimberLine.",
}


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
    body = {"name": path.stem, "parameters": parameters, "segments": [dict(SEGMENT)]}
    # A dnt dataset pins no ids: its items come from the service, and carrying them is an error.
    if glossary_ids is not None:
        body["glossary_ids"] = list(glossary_ids)

    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _report(folder, stem):
    """The one report a run left behind. Its name carries a timestamp, so it is found, not built."""
    written = sorted(folder.glob(f"{stem}_*.md"))
    assert len(written) == 1, f"expected one {stem} report, found {[p.name for p in written]}"
    return written[0]


@pytest.fixture
def configure(monkeypatch, tmp_path):
    """Write a dataset, point GLOSSARY_PATH at it and run from tmp_path, as a real run would."""
    def _configure(**overrides):
        path = _write_dataset(tmp_path, **overrides)
        monkeypatch.setenv("GLOSSARY_PATH", str(path))
        monkeypatch.chdir(tmp_path)
        return path
    return _configure


def test_preflight_blocks_skipped_glossary(
    configure, stub_postmt, stub_glossary, caplog
):
    """A real run with no tempo_task_id/cat_tool_provider would cost money and measure nothing."""
    configure(tempo_task_id=None, cat_tool_provider=None)

    code = run.main([])

    assert code == 1
    assert stub_postmt.submitted is False
    assert "Preflight failed" in caplog.text


def test_dry_run_skips_preflight_and_never_touches_postmt(configure, stub_postmt, stub_glossary):
    configure(tempo_task_id=None, cat_tool_provider=None)

    assert run.main(["--dry-run"]) == 0
    assert stub_postmt.submitted is False


def test_missing_search_engine_url_is_config(
    configure, stub_postmt, monkeypatch, capsys
):
    """An unset URL must say so rather than report the empty string as unreachable."""
    configure()
    monkeypatch.setenv("SEARCH_ENGINE_URL", "")

    code = run.main(["--dry-run"])

    assert code == 2
    assert "SEARCH_ENGINE_URL" in capsys.readouterr().err


def test_pinned_ids_are_ones_queried(configure, stub_postmt, stub_glossary):
    """Ids come from the dataset, so a run is a fixed experiment rather than a live lookup."""
    configure(glossary_ids=["tb1", "tb2"])

    assert run.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == ["tb1", "tb2"]


def test_glossary_ids_absent_from_index_stop_run(
    configure, stub_postmt, monkeypatch, caplog
):
    """An id from another system matches nothing and would score a clean-looking zero."""
    configure(glossary_ids=["041cf63c-3f16-4d79-a386-35cf7688faf0"])
    monkeypatch.setattr(run, "GlossaryClient", lambda *a, **k: _StubGlossary(term_count=0))
    monkeypatch.setenv("SEARCH_ENGINE_URL", "http://search.test")

    code = run.main(["--dry-run"])

    assert code == 1
    assert "None of the glossary ids" in caplog.text
    assert "cluster post-mt queries" in caplog.text


def test_configured_run_needs_no_arguments_at_all(
    configure, stub_postmt, stub_glossary, tmp_path
):
    """Both ends come from .env, so the command line carries only behaviour flags."""
    configure()

    assert run.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == ["tb1"]
    assert _report(tmp_path / "reports", "glossary_dry-run").is_file()


def test_no_dataset_names_variable(
    stub_postmt, stub_glossary, monkeypatch, tmp_path, caplog
):
    """Scoring nothing must not look like a clean run, and the error has to say how to fix it."""
    monkeypatch.setenv("GLOSSARY_PATH", "")
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 1
    assert "GLOSSARY_PATH" in caplog.text


def test_folder_scores_every_dataset_in_and_pools(
    monkeypatch, tmp_path, stub_postmt, stub_glossary
):
    """GLOSSARY_PATH may name a folder; each dataset inside is scored and pooled by stratum."""
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="a.json")
    _write_dataset(folder, name="b.json")
    monkeypatch.setenv("GLOSSARY_PATH", str(folder))
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    report = _report(tmp_path / "reports", "glossary_dry-run").read_text(encoding="utf-8")
    assert "## By stratum" in report
    assert "en-gb->fr-fr" in report


# --- BENCH_COMPONENT ------------------------------------------------------------------------------


def test_no_component_configured_says_what_choices_are(monkeypatch, tmp_path, capsys):
    """An empty setting must not run nothing and exit 0 as though everything passed."""
    monkeypatch.setenv("BENCH_COMPONENT", "")
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 2
    error = capsys.readouterr().err
    assert "BENCH_COMPONENT" in error and "glossary, dnt" in error


def test_unknown_component_is_rejected(monkeypatch, tmp_path, capsys):
    """A typo would otherwise measure less than was asked for and still report success."""
    monkeypatch.setenv("BENCH_COMPONENT", "glossary,dtn")
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 2
    assert "dtn" in capsys.readouterr().err


@pytest.mark.parametrize("written", ['["glossary", "dnt"]', "glossary,dnt"])
def test_both_setting_spellings_agree(
    written, monkeypatch, tmp_path, stub_postmt, stub_glossary, stub_dnt
):
    """`.env` has no notion of a list, so the setting is written both ways in the wild."""
    monkeypatch.setenv("BENCH_COMPONENT", written)
    monkeypatch.setenv("GLOSSARY_PATH", str(_write_dataset(tmp_path, name="g.json")))
    monkeypatch.setenv(
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json", glossary_ids=None))
    )
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    report = _report(tmp_path / "reports", "glossary+dnt_dry-run").read_text(encoding="utf-8")
    assert "# Terminology adherence" in report
    assert "# DNT preservation" in report


def test_every_component_prints_results(
    monkeypatch, tmp_path, stub_postmt, stub_glossary, stub_dnt, capsys
):
    """The file is read afterwards; the console is what a run says while someone is watching it."""
    monkeypatch.setenv("BENCH_COMPONENT", "glossary,dnt")
    monkeypatch.setenv("GLOSSARY_PATH", str(_write_dataset(tmp_path, name="g.json")))
    monkeypatch.setenv(
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json", glossary_ids=None))
    )
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "Adherence MT" in out       # the terminology scorecard
    assert "Per-term adherence" in out  # and the terms it came from
    assert "Preservation" in out       # the DNT scorecard


def test_console_and_file_carry_same_numbers(
    configure, stub_postmt, stub_glossary, tmp_path, capsys
):
    """One scorecard renders both, so a run cannot say one thing and file another."""
    configure()

    assert run.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    report = _report(tmp_path / "reports", "glossary_dry-run").read_text(encoding="utf-8")
    # The facts are identical; only the decoration around them differs.
    for fact in (line[2:] for line in report.splitlines() if line.startswith("- ")):
        assert fact in out


def test_detected_items_are_printed(
    monkeypatch, tmp_path, stub_postmt, stub_dnt, capsys
):
    """A detector that named nothing writes a report of clean-looking zeroes, so it says so here."""
    monkeypatch.setenv("BENCH_COMPONENT", "dnt")
    monkeypatch.setenv(
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json", glossary_ids=None))
    )
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "DNT items detected" in out
    assert "TimberLine" in out


def test_missing_dnt_url_is_config(
    monkeypatch, tmp_path, stub_postmt, capsys
):
    """Dev is VPN-only and prod is not deployed, so an unset URL is the likeliest mistake."""
    monkeypatch.setenv("BENCH_COMPONENT", "dnt")
    monkeypatch.setenv("DNT_BASE_URL", "")
    monkeypatch.setenv("DNT_PATH", str(tmp_path))
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 2
    assert "DNT_BASE_URL" in capsys.readouterr().err
