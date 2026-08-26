"""Term-bases provider — the one that reproduces how post-mt retrieves terms.

The queries here are pinned deliberately: if they drift from the ones production sends, the
benchmark scores against a glossary the pipeline was never shown.
"""

import json

import httpx
import pytest

from sourcecode.glossary import GlossaryClient

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


def _client(percolate=PERCOLATE, concepts=CONCEPTS, capture=None):
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
    _fetch(_client(capture=capture))

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
    _fetch(_client(capture=capture))

    body = capture["search_body"]
    assert capture["search_path"] == "/term-bases/_search"
    assert body["size"] == 1000
    filters = body["query"]["bool"]["filter"]
    assert {"terms": {"concept_id": ["c1", "c2"]}} in filters
    assert {"terms": {"language": ["fr-fr", "fr"]}} in filters


def test_xtm_provider_uses_the_xtm_index():
    capture = {}
    _fetch(_client(capture=capture), provider="XTM")
    assert capture["ndjson"][0] == {"index": "xtm-term-bases"}
    assert capture["search_path"] == "/xtm-term-bases/_search"


def test_non_xtm_providers_use_the_default_index():
    capture = {}
    _fetch(_client(capture=capture), provider="MemSource")
    assert capture["ndjson"][0] == {"index": "term-bases"}


def test_comma_separated_glossary_ids_are_split():
    capture = {}
    _fetch(_client(capture=capture), glossary_ids=" tb1 , tb2 ")
    body = capture["ndjson"][1]
    assert {"terms": {"glossary_id": ["tb1", "tb2"]}} in body["query"]["bool"]["filter"]


# ── result assembly ──────────────────────────────────────────────────────────

def test_per_text_mappings_align_with_texts():
    matches = _fetch(_client())

    assert len(matches.per_text_mappings) == len(TEXTS)
    assert matches.per_text_mappings[0] == [
        {"source_content": "brake pad", "target_content": "plaquette de frein"},
        {"source_content": "engine", "target_content": "moteur"},
        {"source_content": "engine", "target_content": "bloc moteur"},
    ]
    # Second text matched only the engine concept.
    assert {m["source_content"] for m in matches.per_text_mappings[1]} == {"engine"}


def test_targets_are_deduplicated_per_text_by_target_content():
    percolate = {
        "responses": [
            {"hits": {"hits": [
                {"_source": {"term_text": "engine", "concept_id": "c2"}},
                {"_source": {"term_text": "motor", "concept_id": "c2"}},
            ]}},
            {"hits": {"hits": []}},
        ]
    }
    matches = _fetch(_client(percolate=percolate))
    # Both source terms resolve to the same concept, so each target appears once.
    assert [m["target_content"] for m in matches.per_text_mappings[0]] == ["moteur", "bloc moteur"]


def test_global_mappings_are_deduplicated_across_texts():
    matches = _fetch(_client())
    assert [m["target_content"] for m in matches.mappings] == [
        "plaquette de frein", "moteur", "bloc moteur"
    ]


def test_no_percolate_hits_skips_the_concept_lookup():
    empty = {"responses": [{"hits": {"hits": []}}, {"hits": {"hits": []}}]}
    capture = {}
    matches = _fetch(_client(percolate=empty, capture=capture))

    assert matches.mappings == []
    assert matches.per_text_mappings == [[], []]
    assert "search_body" not in capture  # second query never issued


def test_a_percolate_error_on_one_text_does_not_lose_the_others(caplog):
    partial = {
        "responses": [
            {"error": {"type": "search_phase_execution_exception"}},
            {"hits": {"hits": [{"_source": {"term_text": "engine", "concept_id": "c2"}}]}},
        ]
    }
    matches = _fetch(_client(percolate=partial))

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
        _fetch(_client(), **overrides)


# ── term counting, backing the CLI preflight ─────────────────────────────────

def test_count_terms_asks_the_right_index_for_the_ids():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"count": 42})

    client = GlossaryClient("http://search.test")
    client._client = httpx.Client(
        base_url="http://search.test", transport=httpx.MockTransport(handler)
    )

    assert client.count_terms("tb1, tb2") == 42
    assert seen["path"] == "/term-bases/_count"
    assert seen["body"] == {"query": {"terms": {"glossary_id": ["tb1", "tb2"]}}}
    assert client.count_terms("tb1", provider="XTM") == 42
    assert seen["path"] == "/xtm-term-bases/_count"
