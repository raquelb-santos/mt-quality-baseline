"""The parts both components share: matching, dataset loading, the HTTP clients, and the run."""

import io
import json
import zipfile
from functools import partial

import httpx
import pytest

from sourcecode import config
from sourcecode import run
from sourcecode import dnt
from sourcecode.text_processing import (
    Dataset,
    Task,
    count_lemma,
    find_datasets,
    count_occurrences,
    count_surface,
    is_unspaced_language,
    in_languages,
    load,
    normalize_language,
    parse_language_pairs,
    normalize_text,
    parse_csv,
)
from sourcecode.dnt import DntClient, Reversion
from sourcecode.pipeline import run_pipeline
from sourcecode.postmt import RunResult
from sourcecode.postmt import (
    extract_post_edited,
    preflight_parameters,
    preflight_tasks,
    reported_has_glossary,
    segment_error,
)


@pytest.fixture(autouse=True)
def _no_standing_language(monkeypatch):
    """`.env` may narrow every run to one pair; a test says for itself which pairs it means."""
    monkeypatch.delenv("BENCH_LANGUAGE", raising=False)


# matching and counting

# does the term appear at all

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


# how many times it appears
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


# languages

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


# verbatim counting, which DNT asks for

def test_count_surface_can_be_case_sensitive():
    assert count_surface("Le iPhone est ici.", "iPhone", "fr-fr", casefold=False) == 1
    assert count_surface("Le iphone est ici.", "iPhone", "fr-fr", casefold=False) == 0
    assert count_surface("Le iphone est ici.", "iPhone", "fr-fr") == 1


def test_count_surface_inherits_underscore_boundary():
    """`_` is excluded from the boundary class, so it does not separate words."""
    assert count_surface("foo_BAR here", "BAR", "en-gb") == 1


# dataset loading

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


def test_steps_come_from_the_dataset(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "parameters": PARAMETERS,
                "steps": ["AQE"],
                "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}],
            }
        ),
    )
    assert load(path, component="glossary").steps == ["AQE"]


def test_blank_segment_fields_are_rejected(tmp_path):
    path = _write(
        tmp_path, "d.json",
        json.dumps(
            {
                "parameters": PARAMETERS,
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
                "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}],
            }
        ),
    )
    with pytest.raises(ValueError, match="source_language"):
        load(path, component="glossary")


def test_parse_csv_numbers_rows_that_carry_no_id():
    rows = parse_csv("source_content,target_content\nhello,bonjour\n")
    assert rows[0]["source_segment_id"] == "0"
    assert rows[0]["source_content"] == "hello"
    assert rows[0]["target_content"] == "bonjour"

    rows = parse_csv("source_segment_id,source_content,target_content\n7,hello,bonjour\n")
    assert rows[0]["source_segment_id"] == "7"


def test_parse_csv_handles_quoted_fields_with_commas():
    rows = parse_csv('source_content,target_content\n"a, b","c, d"\n')
    assert rows[0]["source_content"] == "a, b"
    assert rows[0]["target_content"] == "c, d"


def test_a_csv_in_the_canonical_names_carries_no_parameters(tmp_path):
    """The rows parse, but nothing in the file names a language, so it cannot be scored."""
    path = _write(tmp_path, "d.csv",
                  "source_content,target_content,reference_content\nhello,bonjour,salut\n")
    with pytest.raises(ValueError, match="source_language"):
        load(path, component="glossary")


def test_only_csv_and_json_are_datasets(tmp_path):
    """A folder may hold working files beside its datasets, and only the two formats are read."""
    for name in ("a.json", "b.csv", "notes.txt", "job.mxliff", "job.xliff", "sheet.xlsx"):
        _write(tmp_path, name, "{}")

    found = find_datasets(str(tmp_path), variable="GLOSSARY_PATH")

    assert [path.name for path in found] == ["a.json", "b.csv"]


def test_another_format_named_outright_is_refused(tmp_path):
    path = _write(tmp_path, "job.mxliff", "<xliff/>")

    with pytest.raises(ValueError, match="Unsupported dataset format"):
        load(path, component="glossary")


# BENCH_LANGUAGE - scoring one pair out of an export that covers many.

EXPORT = (
    "SEGMENTID,SOURCECONTENT,TARGETCONTENT,HUMAN_TARGET,ISOSOURCELANGUAGE,ISOTARGETLANGUAGE\n"
    "1,a,b,c,en-GB,es-ES\n"
    "2,d,e,f,en-US,es-MX\n"
    "3,g,h,i,en-GB,fr-FR\n"
)


def test_a_pair_without_a_region_matches_every_region():
    """`en_es` is how a pair is named day to day; the export spells out the locales."""
    assert parse_language_pairs("en_es") == [("en", "es")]
    pairs = parse_language_pairs("en_es")
    assert in_languages({"source_language": "en-GB", "target_language": "es-ES"}, pairs)
    assert in_languages({"source_language": "en-US", "target_language": "es-419"}, pairs)
    assert not in_languages({"source_language": "en-GB", "target_language": "fr-FR"}, pairs)


def test_a_language_spelled_as_a_name_still_matches():
    """A dataset may name a language rather than code it, so the clean code is what is matched."""
    spelled = normalize_language(
        {"source_language": "English (United Kingdom)", "target_language": "French (France)"}
    )

    assert in_languages(spelled, parse_language_pairs("en_fr"))
    assert not in_languages(spelled, parse_language_pairs("en_es"))


def test_a_pinned_region_matches_only_itself():
    pairs = parse_language_pairs("en-gb_es-es")
    assert in_languages({"source_language": "en-GB", "target_language": "es-ES"}, pairs)
    assert not in_languages({"source_language": "en-US", "target_language": "es-ES"}, pairs)


def test_several_pairs_are_comma_separated():
    assert parse_language_pairs("en_es, en_fr") == [("en", "es"), ("en", "fr")]


@pytest.mark.parametrize("given", ["enes", "en_", "_es", ""])
def test_a_malformed_pair_is_rejected(given):
    with pytest.raises(ValueError, match="SOURCE_TARGET"):
        parse_language_pairs(given)


def test_loading_keeps_only_the_tasks_asked_for(tmp_path):
    path = _write(tmp_path, "export.csv", EXPORT)

    whole = load(path, component="tags")
    assert len(whole.tasks) == 3

    only = load(path, component="tags", languages=parse_language_pairs("en_es"))
    assert [s["source_segment_id"] for s in only.segments] == ["1", "2"]

    # The file's shared parameters are recomputed over the kept tasks, not the whole file.
    pinned = load(path, component="tags", languages=parse_language_pairs("en-gb_es-es"))
    assert pinned.parameters["target_language"] == "es-ES"


def test_a_file_holding_none_of_the_pairs_loads_empty(tmp_path):
    """A folder is scored file by file, so one that matches nothing is skipped, not an error."""
    path = _write(tmp_path, "export.csv", EXPORT)

    data = load(path, component="tags", languages=parse_language_pairs("en_ja"))

    assert data.tasks == [] and data.segments == []


def test_an_invalid_file_still_fails_when_a_pair_is_asked_for(tmp_path):
    """Filtering happens after validation, so a broken file cannot hide behind BENCH_LANGUAGE."""
    path = _write(tmp_path, "export.csv", EXPORT.replace(",c,", ",,"))

    with pytest.raises(ValueError, match="reference_content"):
        load(path, component="tags", languages=parse_language_pairs("en_es"))


# the HTTP boundary

# post-mt - which text gets scored, and what would make the number meaningless.

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


# preflight: refuse runs that would cost money and measure nothing

GOOD = {
    "cat_project_id": "P1",
    "cat_tool_provider": "MemSource",
    "tempo_task_id": "task-1",
}


def test_preflight_passes_on_complete_parameters():
    assert preflight_parameters(GOOD) == []
    assert preflight_parameters({**GOOD, "cat_tool_provider": "XTM"}) == []


def test_preflight_catches_each_silent_skip_condition():
    """Each of these makes post-mt retrieve no glossary at all, without raising an error."""
    missing_provider = preflight_parameters({**GOOD, "cat_tool_provider": ""})
    assert any("cat_tool_provider" in p for p in missing_provider)

    unsupported = preflight_parameters({**GOOD, "cat_tool_provider": "Trados"})
    assert any("does not support" in p for p in unsupported)

    missing_project = preflight_parameters({**GOOD, "cat_project_id": ""})
    assert any("cat_project_id" in p for p in missing_project)


def test_preflight_reports_every_problem_at_once():
    problems = preflight_parameters({})
    # tempo_task_id, provider, project — not just the first
    assert len(problems) == 3


def test_preflight_names_the_tasks_a_problem_is_in():
    """A file covering many jobs would otherwise repeat one problem once per task."""
    tasks = [Task(GOOD, []), Task({**GOOD, "cat_tool_provider": ""}, [])]
    problems = preflight_tasks(tasks, preflight_parameters)

    assert len(problems) == 1
    assert "cat_tool_provider" in problems[0] and "in 1 of 2 tasks" in problems[0]


def test_preflight_leaves_a_problem_every_task_shares_unqualified():
    tasks = [Task({**GOOD, "cat_tool_provider": ""}, []) for _ in range(3)]
    assert preflight_tasks(tasks, preflight_parameters) == preflight_parameters(
        {**GOOD, "cat_tool_provider": ""}
    )


def test_preflight_requires_tempo_task_id():
    """Without it post-mt fails every segment before any step runs, returning no APE text."""
    problems = preflight_parameters({**GOOD, "tempo_task_id": ""})
    assert any("tempo_task_id" in p for p in problems)


# per-segment failures

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


# has_glossary

def test_has_glossary_is_read_beside_aqe_results():
    """post-mt writes it as a sibling of aqe_results, so reading inside it is always None."""
    assert reported_has_glossary({"has_glossary": True}) is True
    assert reported_has_glossary({"has_glossary": False}) is False
    assert reported_has_glossary({"aqe_results": {"has_glossary": True}}) is None
    # A task that ran no AQE step carries no flag, which is not the same as a negative.
    assert reported_has_glossary({"aqe_results": {}}) is None
    assert reported_has_glossary({}) is None

# The DNT service - request shape, batching, and the normalizers that accept its response.

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


# the header, which is the likeliest thing to get wrong

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


# request shape

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
    """A dataset may carry no id and CSV ids can repeat, so the index is the only reliable key."""
    captured = {}

    def handler(request):
        captured["ids"] = [s["id"] for s in json.loads(request.content)["segments"]]
        return httpx.Response(200, json={"segments": [{"terms": []}] * 3})

    pairs = [{"id": str(i), "source_content": "s", "target": "t"} for i in range(3)]
    _dnt_client(handler).revert(pairs, batch_size=10)

    assert captured["ids"] == ["0", "1", "2"]


def test_batching_splits_call_and_keeps_every_segment():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(len(body["segments"]))
        return httpx.Response(200, json={"segments": [{"terms": []}] * len(body["segments"])})

    pairs = [{"id": str(i), "source_content": "s", "target": "t"} for i in range(7)]
    results = _dnt_client(handler).revert(pairs, batch_size=3)

    assert calls == [3, 3, 1]
    assert len(results) == 7


def test_no_pairs_makes_no_call_at_all():
    def handler(request):
        raise AssertionError("should not have been called")

    assert _dnt_client(handler).revert([], batch_size=10) == []


# failure

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

    pairs = [{"id": str(i), "source_content": "s", "target": "t"} for i in range(2)]
    results = _dnt_client(handler).revert(pairs, batch_size=1)

    assert results[0] is None
    assert results[1].items == ["A"]


def test_short_response_realigns():
    def handler(request):
        return httpx.Response(200, json={"segments": [{"terms": ["A"], "corrected_text": "x"}]})

    pairs = [{"id": str(i), "source_content": "s", "target": "t"} for i in range(3)]
    results = _dnt_client(handler).revert(pairs, batch_size=10)

    assert len(results) == 3
    assert results[0].items == ["A"]
    assert results[1] is None and results[2] is None


# health

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


# the response normalizers

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


# what the service's own smoke test pins
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
                "source_content": "AcoladPro is here.",
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


# the run

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
    """The CAT tool's terms, recording the language they were asked for."""
    def __init__(self, *args, **kwargs):
        self.asked_for = None

    def fetch_matches(self, *, glossary_ids, source_language, texts, **kwargs):
        from sourcecode.glossary import GlossaryMatches

        self.asked_for = source_language
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


def test_settings_file_defaults_to_dotenv(monkeypatch):
    """No ENV_FILE leaves dotenv to find `.env` itself - the behaviour every run had before."""
    monkeypatch.delenv("ENV_FILE", raising=False)
    assert config._settings_file() is None


def test_settings_file_named_by_env_file_is_used(monkeypatch, tmp_path):
    settings = tmp_path / ".env.tm"
    settings.write_text("BENCH_COMPONENT=[\"tm\"]\n", encoding="utf-8")
    monkeypatch.setenv("ENV_FILE", str(settings))
    assert config._settings_file() == str(settings)


def test_env_file_that_is_not_there_stops_the_run(monkeypatch, tmp_path):
    """Falling back to `.env` would score the run against settings nobody asked for."""
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "absent"))
    with pytest.raises(RuntimeError, match="which is not a file"):
        config._settings_file()


@pytest.fixture(autouse=True)
def measure_glossary_only(monkeypatch):
    """The default for this module, so no test reaches a service it did not ask for."""
    monkeypatch.setenv("BENCH_COMPONENT", "glossary")


class _StubPhrase:
    """The CAT tool, answering that every project has one term base holding one term."""

    def __init__(self, *args, term_base_ids=("tb-1",), **kwargs):
        self.asked = []
        self._ids = list(term_base_ids)

    def term_base_ids(self, project_id):
        self.asked.append(project_id)
        return list(self._ids)

    def terms(self, term_base_id):
        from sourcecode.cat_tool import Term

        return [Term("c1", "en-gb", "engine"), Term("c1", "fr-fr", "moteur")]

    def close(self):
        pass


@pytest.fixture
def stub_glossary(monkeypatch):
    """The term-bases index is the only glossary source, so every run needs a reachable one, and
    a CAT tool to say which term bases each project has."""
    glossary = _StubGlossary()
    monkeypatch.setenv("SEARCH_ENGINE_URL", "http://search.test")
    monkeypatch.setenv("PHRASE_BASE_URL", "http://phrase.test")
    monkeypatch.setenv("PHRASE_USERNAME", "u")
    monkeypatch.setenv("PHRASE_PASSWORD", "p")
    monkeypatch.setattr(run, "CatToolGlossary", lambda *a, **k: glossary)
    monkeypatch.setattr(run, "PhraseClient", _StubPhrase)
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


def _write_dataset(folder, name="d.json", **overrides):
    """A dataset. Pass ``key=None`` to omit a parameter and trip the preflight."""
    parameters = {
        "cat_project_id": "P1",
        "cat_tool_provider": "MemSource",
        "tempo_task_id": "T1",
        "source_language": "en-gb",
        "target_language": "fr-fr",
    }
    parameters.update(overrides)
    parameters = {k: v for k, v in parameters.items() if v is not None}

    path = folder / name
    body = {"name": path.stem, "parameters": parameters, "segments": [dict(SEGMENT)]}
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
    # tempo_task_id is preflight-only; dropping the CAT tool or the project instead would trip the
    # term-base guard, which a dry run still applies because a run with no terms measures nothing.
    configure(tempo_task_id=None)

    assert run.main(["--dry-run"]) == 0
    assert stub_postmt.submitted is False


def test_no_cat_tool_is_config(
    configure, stub_postmt, stub_glossary, monkeypatch, capsys
):
    """Terms come from the CAT tool, so neither one configured is a setting to fix, not a failure."""
    configure()
    for name in ("PHRASE_BASE_URL", "PHRASE_USERNAME", "PHRASE_PASSWORD"):
        monkeypatch.setenv(name, "")

    code = run.main(["--dry-run"])

    assert code == 2
    assert "PHRASE_BASE_URL" in capsys.readouterr().err


def test_run_percolates_the_whole_index_for_the_source_language(
    configure, stub_postmt, stub_glossary, tmp_path
):
    """Nothing names a term base, so the source language alone decides what is retrieved."""
    configure()

    assert run.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == "en-gb"
    assert _report(tmp_path / "reports", "glossary_dry-run").is_file()


def test_projects_without_a_term_base_stop_the_run(
    configure, stub_postmt, stub_glossary, monkeypatch, caplog
):
    """A project with nothing attached retrieves nothing and would score a clean-looking zero."""
    configure()
    monkeypatch.setattr(run, "PhraseClient", partial(_StubPhrase, term_base_ids=[]))

    code = run.main(["--dry-run"])

    assert code == 1
    assert "has a term base attached" in caplog.text


def test_a_dataset_without_term_bases_is_warned_about_and_skipped(
    monkeypatch, tmp_path, stub_postmt, stub_glossary, caplog
):
    """One unusable file must not cost the run the files beside it."""
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="a.json", cat_project_id="HAS-ONE")
    _write_dataset(folder, name="b.json", cat_project_id="HAS-NONE")

    class _SomeProjects(_StubPhrase):
        def term_base_ids(self, project_id):
            return ["tb-1"] if project_id == "HAS-ONE" else []

    monkeypatch.setattr(run, "PhraseClient", _SomeProjects)
    monkeypatch.setenv("GLOSSARY_PATH", str(folder))
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0
    assert "has a term base attached" in caplog.text

    report = _report(tmp_path / "reports", "glossary_dry-run").read_text(encoding="utf-8")
    assert "a" in report and "b.json" not in report


def test_configured_run_needs_no_arguments_at_all(
    configure, stub_postmt, stub_glossary, tmp_path
):
    """Both ends come from .env, so the command line carries only behaviour flags."""
    configure()

    assert run.main(["--dry-run"]) == 0
    assert stub_glossary.asked_for == "en-gb"
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
    """GLOSSARY_PATH may name a folder; each dataset inside is scored and pooled by language pair."""
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="a.json")
    _write_dataset(folder, name="b.json")
    monkeypatch.setenv("GLOSSARY_PATH", str(folder))
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    report = _report(tmp_path / "reports", "glossary_dry-run").read_text(encoding="utf-8")
    assert "## By language pair" in report
    assert "en-gb->fr-fr" in report


# BENCH_LANGUAGE, over a whole run


def test_a_pair_no_dataset_holds_stops_the_run(configure, stub_postmt, stub_glossary, monkeypatch, capsys):
    """Scoring nothing would write an empty report and exit 0, reading like a clean result."""
    configure()
    monkeypatch.setenv("BENCH_LANGUAGE", "en_es")

    code = run.main(["--dry-run"])

    assert code == 1
    assert "No glossary dataset could be scored" in capsys.readouterr().err


def test_bench_language_narrows_the_run(
    monkeypatch, tmp_path, stub_postmt, stub_glossary
):
    """A folder is scored file by file, so the pairs not asked for are skipped, not failed."""
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="fr.json")
    _write_dataset(folder, name="es.json", target_language="es-es")
    monkeypatch.setenv("GLOSSARY_PATH", str(folder))
    monkeypatch.setenv("BENCH_LANGUAGE", "en_es")
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    report = _report(tmp_path / "reports", "glossary_dry-run").read_text(encoding="utf-8")
    assert "en-gb->es-es" in report and "en-gb->fr-fr" not in report


def test_a_blank_slot_scores_every_pair(monkeypatch, tmp_path, stub_postmt, stub_glossary, stub_dnt):
    """Each component gets its own slot, so narrowing one leaves the others measuring everything."""
    monkeypatch.setenv("BENCH_COMPONENT", "dnt,glossary")
    folder = tmp_path / "many"
    folder.mkdir()
    _write_dataset(folder, name="fr.json")
    _write_dataset(folder, name="es.json", target_language="es-es")
    monkeypatch.setenv("GLOSSARY_PATH", str(folder))
    monkeypatch.setenv("DNT_PATH", str(_write_dataset(tmp_path, name="n.json")))
    # DNT is asked for every pair, terminology only for en_es.
    monkeypatch.setenv("BENCH_LANGUAGE", "[ , en_es]")
    monkeypatch.chdir(tmp_path)

    assert run.main(["--dry-run"]) == 0

    report = _report(tmp_path / "reports", "dnt+glossary_dry-run").read_text(encoding="utf-8")
    # The DNT dataset is en-gb->fr-fr, which only an unfiltered slot keeps.
    assert "en-gb->fr-fr" in report and "en-gb->es-es" in report


def test_more_slots_than_components_is_a_usage_error(configure, monkeypatch, capsys):
    """A slot list that does not line up would silently narrow the wrong component."""
    configure()
    monkeypatch.setenv("BENCH_LANGUAGE", "[en_es, en_fr]")

    assert run.main(["--dry-run"]) == 2
    assert "line up slot by slot" in capsys.readouterr().err


def test_a_malformed_language_is_a_usage_error(configure, monkeypatch, capsys):
    configure()
    monkeypatch.setenv("BENCH_LANGUAGE", "enes")

    assert run.main(["--dry-run"]) == 2
    assert "SOURCE_TARGET" in capsys.readouterr().err


# BENCH_COMPONENT


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
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json"))
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
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json"))
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
        "DNT_PATH", str(_write_dataset(tmp_path, name="n.json"))
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


# a dataset covering several jobs

def _two_task_dataset():
    first = {"source_language": "en-gb", "target_language": "fr-fr", "cat_project_id": "P1"}
    second = {**first, "target_language": "fr-ca", "cat_project_id": "P2"}
    return Dataset(
        name="mixed",
        component="tags",
        parameters={"source_language": "en-gb"},
        tasks=[
            Task(normalize_language(first), [{"source_segment_id": "a"}]),
            Task(normalize_language(second), [{"source_segment_id": "b"}, {"source_segment_id": "c"}]),
        ],
    )


def test_segments_read_in_task_order():
    dataset = _two_task_dataset()
    assert [s["source_segment_id"] for s in dataset.segments] == ["a", "b", "c"]


def test_a_parameter_is_carried_per_segment_from_its_own_task():
    """Scoring reads the language of the task a segment was sent in, not the file's."""
    assert _two_task_dataset().per_segment("clean_target_language_code") == [
        "fr-fr", "fr-ca", "fr-ca",
    ]


def test_each_task_is_submitted_on_its_own_parameters():
    """post-mt finds the term bases through these, so one submission cannot carry two sets."""
    submitted = []

    class Recorder:
        def run(self, *, parameters, segments, steps, on_progress):
            submitted.append((parameters["cat_project_id"], len(segments)))
            return RunResult(task_id="t", segments=list(segments), error=None)

    run_pipeline(Recorder(), _two_task_dataset(), batch_size=50)
    assert submitted == [("P1", 1), ("P2", 2)]


def test_a_batch_never_spans_two_tasks():
    submitted = []

    class Recorder:
        def run(self, *, parameters, segments, steps, on_progress):
            submitted.append((parameters["cat_project_id"], len(segments)))
            return RunResult(task_id="t", segments=list(segments), error=None)

    run_pipeline(Recorder(), _two_task_dataset(), batch_size=2)
    assert submitted == [("P1", 1), ("P2", 2)]


# resolving a CAT project's term bases, the step post-mt runs before any retrieval

from sourcecode.cat_tool import PhraseClient, Term, TermBaseResolver, XtmClient   # noqa: E402


class _CountingCat:
    def __init__(self, ids=("tb-1",)):
        self.ids = list(ids)
        self.calls = []

    def term_base_ids(self, project_id):
        self.calls.append(project_id)
        return list(self.ids)

    def close(self):
        pass


def test_memsource_and_phrase_are_one_tool():
    """post-mt routes both names to the Phrase API, so a Memsource project must resolve."""
    cat = _CountingCat()
    resolver = TermBaseResolver(phrase=cat)

    assert resolver.ids_for("P1", "MemSource") == ["tb-1"]
    assert resolver.ids_for("P1", "Phrase") == ["tb-1"]


def test_an_unsupported_cat_tool_resolves_nothing():
    """Trados has no term-base lookup in post-mt, so its segments get no glossary rather than all."""
    resolver = TermBaseResolver(phrase=_CountingCat(), xtm=_CountingCat())

    assert resolver.ids_for("P1", "Trados") == []


def test_a_project_is_asked_about_once():
    cat = _CountingCat()
    resolver = TermBaseResolver(phrase=cat)

    assert resolver.ids_for("P1", "MemSource") == resolver.ids_for("P1", "MemSource")
    assert cat.calls == ["P1"]


def test_an_empty_answer_is_not_cached():
    """A project mid-ingest answers with nothing; caching that would settle the whole run."""
    cat = _CountingCat(ids=[])
    resolver = TermBaseResolver(phrase=cat)

    assert resolver.ids_for("P1", "MemSource") == []
    assert resolver.ids_for("P1", "MemSource") == []
    assert cat.calls == ["P1", "P1"]


def test_a_failing_cat_tool_leaves_the_task_without_glossary():
    """Retrieval degrades rather than stopping the run, and the error says which project."""
    class _Broken:
        def term_base_ids(self, project_id):
            raise httpx.ConnectError("no route")

        def close(self):
            pass

    assert TermBaseResolver(phrase=_Broken()).ids_for("P1", "MemSource") == []


def test_phrase_reads_the_term_base_uids():
    def handler(request):
        if request.url.path.endswith("/v3/auth/login"):
            return httpx.Response(200, json={"token": "t"})
        assert request.headers["Authorization"] == "ApiToken t"
        assert request.url.path.endswith("/web/api2/v1/projects/P1/termBases")
        return httpx.Response(200, json={"termBases": [
            {"termBase": {"uid": "tb-1"}}, {"termBase": {"uid": "tb-2"}},
        ]})

    client = PhraseClient("http://phrase.test", "u", "p")
    client._client = httpx.Client(base_url="http://phrase.test/web/api2",
                                  transport=httpx.MockTransport(handler))

    assert client.term_base_ids("P1") == ["tb-1", "tb-2"]


TBX = """<?xml version='1.0' encoding='UTF-8'?>
<martif xml:lang="en" type="TBX"><text><body>
  <termEntry>
    <descrip type="conceptId">c1</descrip>
    <langSet xml:lang="en-gb">
      <tig><term>gearbox</term><termNote type="forbidden">false</termNote></tig>
    </langSet>
    <langSet xml:lang="fr-fr">
      <tig><term>boite de vitesses</term><termNote type="forbidden">false</termNote></tig>
      <tig><term>transmission</term><termNote type="forbidden">true</termNote></tig>
    </langSet>
  </termEntry>
</body></text></martif>"""


def _exporting_phrase(body, status=200):
    def handler(request):
        if request.url.path.endswith("/v3/auth/login"):
            return httpx.Response(200, json={"token": "t"})
        assert request.headers["Authorization"] == "ApiToken t"
        assert request.url.path.endswith("/web/api2/v1/termBases/tb-1/export")
        return httpx.Response(status, content=body.encode())

    client = PhraseClient("http://phrase.test", "u", "p")
    client._client = httpx.Client(base_url="http://phrase.test/web/api2",
                                  transport=httpx.MockTransport(handler))
    return client


def test_phrase_terms_are_read_from_the_tbx_export():
    """Phrase serves no term listing, so the terms can only come from the exported term base."""
    assert _exporting_phrase(TBX).terms("tb-1") == [
        Term("c1", "en-gb", "gearbox"),
        Term("c1", "fr-fr", "boite de vitesses"),
        Term("c1", "fr-fr", "transmission", forbidden=True),
    ]


def test_an_unreadable_export_fails_the_way_the_glossary_expects():
    """Retrieval degrades on RuntimeError; a raw XML fault would stop the run instead."""
    with pytest.raises(RuntimeError, match="valid TBX"):
        _exporting_phrase("<martif>truncated").terms("tb-1")


def test_xtm_term_bases_are_the_projects_term_customer_ids():
    """XTM has no term-base endpoint, so post-mt reads them off the project itself."""
    def handler(request):
        if request.url.path.endswith("/auth/token"):
            return httpx.Response(200, json={"token": "t"})
        assert request.headers["Authorization"] == "XTM-Basic t"
        # The endpoint answers with a list holding the project.
        return httpx.Response(200, json=[{"termCustomerIds": [11, 12]}])

    client = XtmClient("http://xtm.test", "acolad", "7", "p")
    client._client = httpx.Client(base_url="http://xtm.test/project-manager-api-rest",
                                  transport=httpx.MockTransport(handler))

    assert client.term_base_ids("P1") == ["11", "12"]


XTM_TBX = """<?xml version="1.0" encoding="UTF-8"?>
<martif type="TBX-Basic" xml:lang="en"><text><body>
  <termEntry>
    <langSet xml:lang="en-US"><ntig><termGrp>
      <term>gearbox</term><termNote type="status">VALID</termNote>
    </termGrp></ntig></langSet>
    <langSet xml:lang="fr-FR"><ntig><termGrp>
      <term>transmission</term><termNote type="status">FORBIDDEN</termNote>
    </termGrp></ntig></langSet>
  </termEntry>
  <termEntry>
    <langSet xml:lang="en-US"><ntig><termGrp><term>engine</term></termGrp></ntig></langSet>
  </termEntry>
</body></text></martif>"""


def test_xtm_terms_are_read_from_the_customers_terminology_export(monkeypatch):
    """XTM serves no term listing, so the terms come from an export job, polled until it finishes."""
    monkeypatch.setattr("sourcecode.cat_tool.time.sleep", lambda seconds: None)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("terms.tbx", XTM_TBX)
    statuses = iter(["IN_PROGRESS", "FINISHED"])

    def handler(request):
        path = request.url.path.removeprefix("/project-manager-api-rest")
        if path == "/auth/token":
            return httpx.Response(200, json={"token": "t"})
        assert request.headers["Authorization"] == "XTM-Basic t"
        if path == "/terminology/files/export":
            assert json.loads(request.content)["filter"] == {"customerIds": [11]}
            return httpx.Response(201, json={"fileId": 5})
        if path == "/terminology/files/export/5/status":
            return httpx.Response(200, json={"status": next(statuses)})
        assert path == "/terminology/files/export/5/download"
        return httpx.Response(200, content=archive.getvalue())

    client = XtmClient("http://xtm.test", "acolad", "7", "p")
    client._client = httpx.Client(base_url="http://xtm.test/project-manager-api-rest",
                                  transport=httpx.MockTransport(handler))

    # XTM names no concept, so each entry is one, kept apart from other customers' entries.
    assert client.terms("11") == [
        Term("11#0", "en-US", "gearbox"),
        Term("11#0", "fr-FR", "transmission", forbidden=True),
        Term("11#1", "en-US", "engine"),
    ]


def test_a_refused_token_is_earned_again_once():
    seen = []
    logins = []

    def handler(request):
        if request.url.path.endswith("/v3/auth/login"):
            logins.append(f"t{len(logins) + 1}")
            seen.append("login")
            return httpx.Response(200, json={"token": logins[-1]})

        seen.append(request.headers["Authorization"])
        # The first token is refused, so the client earns a second and retries with it.
        if len(logins) == 1:
            return httpx.Response(401, json={})
        return httpx.Response(200, json={"termBases": [{"termBase": {"uid": "tb-1"}}]})

    client = PhraseClient("http://phrase.test", "u", "p")
    client._client = httpx.Client(base_url="http://phrase.test/web/api2",
                                  transport=httpx.MockTransport(handler))

    assert client.term_base_ids("P1") == ["tb-1"]
    assert seen == ["login", "ApiToken t1", "login", "ApiToken t2"]


# reading the terms themselves from the CAT tool, in place of the term-bases index

from sourcecode.glossary import CatToolGlossary   # noqa: E402


class _TermCat:
    def __init__(self, terms):
        self.terms_by_base = terms
        self.asked = []

    def terms(self, term_base_id):
        self.asked.append(term_base_id)
        return list(self.terms_by_base.get(term_base_id, []))

    def close(self):
        pass


ENGINE = [Term("c1", "en-gb", "engine"), Term("c1", "fr-fr", "moteur"),
          Term("c2", "en-gb", "brake pad"), Term("c2", "fr-fr", "plaquette de frein")]


def _cat_glossary(terms=None, provider="phrase"):
    cat = _TermCat(terms if terms is not None else {"tb-1": ENGINE})
    kwargs = {"phrase": cat} if provider == "phrase" else {"xtm": cat}
    return CatToolGlossary(_StubStanza(), **kwargs), cat


def test_terms_come_from_the_named_cat_tool():
    """CATTOOL picks the client, so a Memsource task never reads XTM's term bases."""
    glossary, cat = _cat_glossary()

    glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                           target_language="fr-fr", texts=["the engine"], provider="MemSource")

    assert cat.asked == ["tb-1"]


def test_an_xtm_task_reads_the_xtm_client():
    glossary, cat = _cat_glossary(provider="xtm")

    glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                           target_language="fr-fr", texts=["the engine"], provider="XTM")

    assert cat.asked == ["tb-1"]


def test_a_tool_with_no_client_reads_nothing():
    glossary, cat = _cat_glossary()

    matches = glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                                     target_language="fr-fr", texts=["the engine"],
                                     provider="XTM")

    assert (matches.mappings, cat.asked) == ([], [])


def test_a_matched_term_carries_its_target_wording():
    glossary, _ = _cat_glossary()

    matches = glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                                     target_language="fr-fr",
                                     texts=["the engine is electric"], provider="phrase")

    assert matches.mappings == [{"source_content": "engine", "target_content": "moteur"}]
    assert matches.per_text_mappings == [matches.mappings]


def test_a_term_the_text_never_uses_is_not_matched():
    glossary, _ = _cat_glossary()

    matches = glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                                     target_language="fr-fr", texts=["the cable is loose"],
                                     provider="phrase")

    assert matches.per_text_mappings == [[]]


def test_the_language_filter_is_permissive():
    """A term base written in `en` must still serve an `en-gb` task, as the index filter does."""
    terms = {"tb-1": [Term("c1", "en", "engine"), Term("c1", "fr", "moteur")]}
    glossary, _ = _cat_glossary(terms)

    matches = glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                                     target_language="fr-fr", texts=["the engine"],
                                     provider="phrase")

    assert matches.mappings == [{"source_content": "engine", "target_content": "moteur"}]


def test_a_term_base_is_read_once_across_tasks():
    glossary, cat = _cat_glossary()
    for _ in range(3):
        glossary.fetch_matches(glossary_ids=["tb-1"], source_language="en-gb",
                               target_language="fr-fr", texts=["the engine"], provider="phrase")

    assert cat.asked == ["tb-1"]


def test_several_term_bases_are_all_read():
    terms = {"tb-1": ENGINE[:2], "tb-2": ENGINE[2:]}
    glossary, cat = _cat_glossary(terms)

    matches = glossary.fetch_matches(glossary_ids=["tb-1", "tb-2"], source_language="en-gb",
                                     target_language="fr-fr",
                                     texts=["the engine and the brake pad"], provider="phrase")

    assert cat.asked == ["tb-1", "tb-2"]
    assert {m["target_content"] for m in matches.mappings} == {"moteur", "plaquette de frein"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"glossary_ids": []}, "No glossary IDs provided"),
        ({"source_language": ""}, "No source language provided"),
        ({"texts": []}, "No texts provided"),
    ],
)
def test_the_cat_glossary_rejects_the_same_missing_inputs(overrides, message):
    glossary, _ = _cat_glossary()
    kwargs = dict(glossary_ids=["tb-1"], source_language="en-gb", target_language="fr-fr",
                  texts=["the engine"], provider="phrase")
    kwargs.update(overrides)

    with pytest.raises(ValueError, match=message):
        glossary.fetch_matches(**kwargs)


def test_stanza_splits_texts_into_batches():
    """A whole term base in one request is rejected as 413 Payload Too Large."""
    from sourcecode.postmt import StanzaClient

    sizes = []

    def handler(request):
        texts = json.loads(request.content)["texts"]
        sizes.append(len(texts))
        return httpx.Response(200, json={"lemmatized_texts": [t.upper() for t in texts]})

    client = StanzaClient("http://stanza.test", 5, batch_size=2)
    client._client = httpx.Client(base_url="http://stanza.test", transport=httpx.MockTransport(handler))

    assert client.lemmatize_batch_safe(["a", "b", "c", "d", "e"], "en") == ["A", "B", "C", "D", "E"]
    assert sizes == [2, 2, 1]
