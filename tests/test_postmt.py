"""post-mt client: which text gets scored, and the run conditions that would make the
number meaningless without saying so."""

import pytest

from sourcecode.postmt import (
    extract_post_edited,
    preflight_parameters,
    reported_has_glossary,
    segment_error,
)


@pytest.mark.parametrize(
    "segment,expected",
    [
        ({"ape_results": {"text": "post-edited"}, "target_content": "mt"}, "post-edited"),
        ({"aped_text": "legacy", "target_content": "mt"}, "legacy"),
        # APE legitimately left it alone: fall back to MT rather than dropping the segment,
        # which would bias the post-edit number.
        ({"ape_results": {"text": None}, "target_content": "mt"}, "mt"),
        ({"target_content": "mt"}, "mt"),
        ({}, ""),
    ],
)
def test_extract_post_edited_handles_every_pipeline_version(segment, expected):
    assert extract_post_edited(segment) == expected


# ── preflight: refuse runs that would cost money and measure nothing ──────────

GOOD = {
    "cat_project_id": "01nwBAb8IMKg8QsJp5vyU1",
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


def test_failed_ape_is_indistinguishable_from_untouched_text():
    """post-mt returns empty APE text beside an error, and the fallback yields the raw MT."""
    failed = {"target_content": "raw mt", "ape_results": {"text": "", "error": "boom"}}
    assert extract_post_edited(failed) == "raw mt"      # identical to an untouched segment...
    assert segment_error(failed) == "boom"              # ...so only this can tell them apart


# ── has_glossary ─────────────────────────────────────────────────────────────

def test_has_glossary_is_read_from_aqe_results():
    """post-mt nests it under aqe_results; reading the top level made the 'pipeline was never
    shown the glossary' warning permanently dead."""
    assert reported_has_glossary({"aqe_results": {"has_glossary": True}}) is True
    assert reported_has_glossary({"aqe_results": {"has_glossary": False}}) is False
    assert reported_has_glossary({"has_glossary": True}) is True      # legacy top-level
    assert reported_has_glossary({"aqe_results": {}}) is None
    assert reported_has_glossary({}) is None
