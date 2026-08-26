"""Dataset loading and validation; glossary ids are pinned, not resolved live, so a run stays a fixed experiment."""

from __future__ import annotations

import csv
import json
import io
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .normalize import normalize_language


class DatasetError(ValueError):
    pass


@dataclass
class Dataset:
    name: str
    parameters: dict[str, Any]
    glossary_ids: list[str]
    segments: list[dict[str, Any]]
    steps: list[str] = field(default_factory=lambda: ["AQE", "APE"])
    #: How ``glossary_ids`` was obtained, recorded in the report so a run is self-describing.
    glossary_ids_source: str = "dataset (pinned)"


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_mxliff(xml_string: str) -> list[dict[str, Any]]:
    """Namespace-agnostic: mxliff files declare the XLIFF namespace inconsistently, and a qualified search silently finds nothing.

    A trans-unit carries both columns the benchmark needs: `<target>` is the translation the job
    was delivered with — the human answer key — and the `<alt-trans>` beside it is the machine
    proposal that target was produced from. A unit holding only one of the two cannot be scored,
    so it is dropped rather than compared against itself.
    """
    root = ET.fromstring(xml_string)
    segments: list[dict[str, Any]] = []

    for element in root.iter():
        if _strip_namespace(element.tag) != "trans-unit":
            continue

        source = reference = machine = ""
        for child in element:
            name = _strip_namespace(child.tag)
            if name == "source":
                source = "".join(child.itertext())
            elif name == "target":
                reference = "".join(child.itertext())
            elif name == "alt-trans" and not machine:
                for proposal in child:
                    if _strip_namespace(proposal.tag) == "target":
                        machine = "".join(proposal.itertext())
                        break

        if source.strip() and machine.strip() and reference.strip():
            segments.append(
                {
                    "source_segment_id": element.get("id"),
                    "source_content": source,
                    "target_content": machine,
                    "reference_content": reference,
                }
            )

    return segments


def parse_csv(text: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(text))
    segments: list[dict[str, Any]] = []

    for index, row in enumerate(reader):
        row = {(key or "").strip(): (value or "") for key, value in row.items()}
        segment = {
            "source_segment_id": row.get("source_segment_id") or row.get("segment_id") or str(index),
            "source_content": row.get("source_content") or row.get("source") or "",
            "target_content": row.get("target_content") or row.get("mt") or row.get("target") or "",
        }

        reference = (
            row.get("reference_content") or row.get("reference")
            or row.get("corrected_content") or row.get("corrected")
            or row.get("post_edited") or row.get("human") or ""
        )
        if reference.strip():
            segment["reference_content"] = reference

        segments.append(segment)

    return segments


def validate(dataset: Dataset, *, require_glossary_ids: bool = True) -> None:
    errors: list[str] = []

    if not dataset.parameters:
        errors.append("missing `parameters`")
    if not dataset.parameters.get("source_language"):
        errors.append("missing `parameters.source_language`")
    if not dataset.parameters.get("target_language"):
        errors.append("missing `parameters.target_language`")

    if require_glossary_ids and not dataset.glossary_ids:
        errors.append('missing `glossary_ids` — see README "Terminology adherence"')

    if not dataset.segments:
        errors.append("no segments")

    for index, segment in enumerate(dataset.segments):
        # The reference is the metric's denominator, not an optional extra column: without it a
        # segment has nothing to be scored against — see README "The metric".
        for field_name in ("source_content", "target_content", "reference_content"):
            value = segment.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"segment[{index}] missing `{field_name}`")

    if errors:
        listed = "\n  - ".join(errors[:12])
        raise DatasetError(f'Invalid dataset "{dataset.name}":\n  - {listed}')


def load(
    path: str | Path,
    overrides: dict[str, Any] | None = None,
    *,
    require_glossary_ids: bool = True,
) -> Dataset:
    overrides = overrides or {}
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()

    if suffix == ".json":
        body = json.loads(raw)
    elif suffix == ".csv":
        body = {"segments": parse_csv(raw)}
    elif suffix in {".mxliff", ".xliff", ".xlf"}:
        body = {"segments": parse_mxliff(raw)}
    else:
        raise DatasetError(f"Unsupported dataset format: {suffix} (expected .json, .csv or .mxliff)")

    parameters = {**body.get("parameters", {}), **overrides.get("parameters", {})}

    dataset = Dataset(
        name=overrides.get("name") or body.get("name") or path.stem,
        parameters=normalize_language(parameters),
        glossary_ids=list(overrides.get("glossary_ids") or body.get("glossary_ids") or []),
        segments=list(overrides.get("segments") or body.get("segments") or []),
        steps=list(overrides.get("steps") or body.get("steps") or ["AQE", "APE"]),
    )
    validate(dataset, require_glossary_ids=require_glossary_ids)
    return dataset
