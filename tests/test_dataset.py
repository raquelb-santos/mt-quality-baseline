"""Dataset loading, parsing and validation."""

import json

import pytest

from sourcecode.dataset import DatasetError, load, parse_csv, parse_mxliff

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
    dataset = load(path)
    assert dataset.name == "demo"
    assert dataset.parameters["clean_source_language_code"] == "en-gb"
    assert dataset.parameters["clean_target_language_code"] == "fr-fr"
    assert dataset.steps == ["AQE", "APE"]


def test_overrides_take_precedence_over_file(tmp_path):
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
    dataset = load(path, {"glossary_ids": ["from-cli"], "steps": ["AQE"]})
    assert dataset.glossary_ids == ["from-cli"]
    assert dataset.steps == ["AQE"]


def test_missing_glossary_ids_is_rejected_with_a_pointer_to_the_readme(tmp_path):
    """The term-bases index is the only glossary source, so a dataset without ids cannot be
    scored at all; it must say so rather than resolve nothing and report a clean zero."""
    path = _write(
        tmp_path, "d.json",
        json.dumps({"parameters": PARAMETERS, "segments": [{"source_content": "a", "target_content": "b", "reference_content": "c"}]}),
    )
    with pytest.raises(DatasetError, match="glossary_ids"):
        load(path)


def test_the_readme_section_that_error_points_at_still_exists():
    """The pointer went stale once already, when the section was renamed during a refactor.
    A dangling reference is invisible at runtime, so pin it here instead."""
    import re
    from pathlib import Path

    from sourcecode import dataset as dataset_module

    source = Path(dataset_module.__file__).read_text(encoding="utf-8")
    referenced = set(re.findall(r'see README "([^"]+)"', source))
    assert referenced, "expected dataset.py to point at a README section"

    readme = Path(__file__).resolve().parents[1] / "README.md"
    headings = {line.lstrip("#").strip() for line in readme.read_text(encoding="utf-8").splitlines()
                if line.startswith("#")}

    missing = referenced - headings
    assert not missing, f"dataset.py points at README sections that do not exist: {sorted(missing)}"


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
    with pytest.raises(DatasetError, match="target_content"):
        load(path)


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
    with pytest.raises(DatasetError, match="source_language"):
        load(path)


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


def test_load_csv_requires_parameters_via_overrides(tmp_path):
    path = _write(tmp_path, "d.csv", "source,target,reference\nhello,bonjour,salut\n")
    dataset = load(path, {"parameters": PARAMETERS, "glossary_ids": ["tb1"]})
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


def test_parse_mxliff_reads_the_machine_proposal_beside_the_delivered_target():
    segments = parse_mxliff(MXLIFF)
    # Unit 2 has an empty target, matching post-mt's own mxliff filter; unit 4 was typed rather
    # than post-edited, so it carries no machine proposal and there is nothing to score.
    assert [s["source_segment_id"] for s in segments] == ["1", "3"]
    assert segments[0]["target_content"] == "La plaquette."
    assert segments[0]["reference_content"] == "La plaquette de frein."
    assert segments[1]["target_content"] == "Cafe"
    assert segments[1]["reference_content"] == "Café"


def test_load_mxliff_end_to_end(tmp_path):
    path = _write(tmp_path, "job.mxliff", MXLIFF)
    dataset = load(path, {"parameters": PARAMETERS, "glossary_ids": ["tb1"]})
    assert dataset.name == "job"
    assert len(dataset.segments) == 2
