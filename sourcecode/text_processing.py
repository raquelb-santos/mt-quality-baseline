"""Dataset loading and validation, the language normalization it applies, and term matching."""

import csv
import json
import io
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


COMPONENTS = ("glossary", "dnt")

DATA_TYPES = (".json", ".csv", ".mxliff", ".xliff", ".xlf")

PARAMS_SUFFIX = ".params.json"


def find_datasets(configured: str, *, variable: str) -> list[Path]:
    """The datasets one component's setting names: a file, or a folder's direct children, sorted."""
    if not configured.strip():
        raise ValueError(
            f"No dataset to score. Set {variable} in .env to a dataset file, or to a folder "
            f"to score every dataset inside it."
        )

    candidate = Path(configured)
    if candidate.is_dir():
        found = sorted(
            child for child in candidate.iterdir()
            if child.is_file() and child.suffix.lower() in DATA_TYPES
            and not child.name.endswith(PARAMS_SUFFIX)
        )
        if not found:
            raise ValueError(f"No dataset files in {candidate} (looked for {', '.join(DATA_TYPES)}).")
        return found
    if candidate.is_file():
        return [candidate]

    raise ValueError(f"{variable} points at nothing: {candidate}")


@dataclass
class Dataset:
    name: str
    parameters: dict[str, Any]
    glossary_ids: list[str]
    segments: list[dict[str, Any]]
    # Which component scores this dataset.
    component: str
    steps: list[str] = field(default_factory=lambda: ["AQE", "APE"])


def _strip_namespace(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def parse_mxliff(xml_string: str) -> list[dict[str, Any]]:
    """Namespace-agnostic; `<target>` is the human reference, the `<alt-trans>` beside it the MT."""
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
            row.get("reference_content") or row.get("reference") or row.get("corrected_content")
            or row.get("corrected") or row.get("post_edited") or row.get("human") or ""
        )
        if reference.strip():
            segment["reference_content"] = reference

        segments.append(segment)

    return segments


def validate(dataset: Dataset) -> None:
    errors: list[str] = []

    if not dataset.parameters:
        errors.append("missing `parameters`")
    if not dataset.parameters.get("source_language"):
        errors.append("missing `parameters.source_language`")
    if not dataset.parameters.get("target_language"):
        errors.append("missing `parameters.target_language`")

    if dataset.component == "glossary" and not dataset.glossary_ids:
        errors.append('missing `glossary_ids`')
    if not dataset.segments:
        errors.append("no segments")

    for index, segment in enumerate(dataset.segments):
        # The reference is the metric's denominator.
        for field_name in ("source_content", "target_content", "reference_content"):
            value = segment.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"segment[{index}] missing `{field_name}`")

    if errors:
        listed = "\n  - ".join(errors[:12])
        raise ValueError(f'Invalid dataset "{dataset.name}":\n  - {listed}')


def read_params(path: Path) -> dict[str, Any]:
    params_file = path.with_name(path.stem + PARAMS_SUFFIX)
    return json.loads(params_file.read_text(encoding="utf-8")) if params_file.is_file() else {}


def load(path: str | Path, *, component: str) -> Dataset:
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
        raise ValueError(f"Unsupported dataset format: {suffix} (expected one of {', '.join(DATA_TYPES)})")

    params = read_params(path)
    dataset = Dataset(
        name=params.get("name") or body.get("name") or path.stem,
        parameters=normalize_language(
            {**body.get("parameters", {}), **params.get("parameters", {})}
        ),
        glossary_ids=list(params.get("glossary_ids") or body.get("glossary_ids") or []),
        segments=list(body.get("segments") or []),
        steps=list(params.get("steps") or body.get("steps") or ["AQE", "APE"]),
        component=component,
    )
    validate(dataset)
    return dataset


LANGUAGE_MAPPING: dict[str, str] = {
    "af-za": "Afrikaans (South Africa)",
    "sq-al": "Albanian (Albania)",
    "ar-sa": "Arabic (Saudi Arabia)",
    "hy-ma": "Armenian (Armenia)",
    "bn-bd": "Bengali (Bangladesh)",
    "bs-ba": "Bosnian (Bosnia and Herzegovina)",
    "ca-es": "Catalan (Spain)",
    "hr-hr": "Croatian (Croatia)",
    "cs-cz": "Czech (Czech Republic)",
    "da-dk": "Danish (Denmark)",
    "nl-be": "Dutch (Belgium)",
    "nl-nl": "Dutch (Netherlands)",
    "en-us": "English (United States)",
    "en-gb": "English (United Kingdom)",
    "eo": "Esperanto",
    "et-ee": "Estonian (Estonia)",
    "tl-ph": "Filipino (Philippines)",
    "fi-fi": "Finnish (Finland)",
    "fr-fr": "French (France)",
    "fr-be": "French (Belgium)",
    "fr-ca": "French (Canada)",
    "de-de": "German (Germany)",
    "el-gr": "Greek (Greece)",
    "gu-in": "Gujarati (India)",
    "hi-in": "Hindi (India)",
    "hu-hu": "Hungarian (Hungary)",
    "is-is": "Icelandic (Iceland)",
    "id-id": "Indonesian (Indonesia)",
    "it-it": "Italian (Italy)",
    "ja-jp": "Japanese (Japan)",
    "jw-id": "Javanese (Indonesia)",
    "ko-kr": "Korean (South Korea)",
    "la-la": "Latin",
    "mr-in": "Marathi (India)",
    "pl-pl": "Polish (Poland)",
    "pt-br": "Portuguese (Brazil)",
    "pt-pt": "Portuguese (Portugal)",
    "ro-ro": "Romanian (Romania)",
    "ru-ru": "Russian (Russia)",
    "sr-rs": "Serbian (Serbia)",
    "si-lk": "Sinhala (Sri Lanka)",
    "sk-sk": "Slovak (Slovakia)",
    "sl-si": "Slovenian (Slovenia)",
    "es-419": "Spanish (Latin America)",
    "es-es": "Spanish (Spain)",
    "es-mx": "Spanish (Mexico)",
    "es-ar": "Spanish (Argentina)",
    "sv-fi": "Swedish (Finland)",
    "sv-se": "Swedish (Sweden)",
    "ta-in": "Tamil (India)",
    "te-in": "Telugu (India)",
    "th-th": "Thai (Thailand)",
    "tr-tr": "Turkish (Turkey)",
    "uk-ua": "Ukrainian (Ukraine)",
    "vi-vn": "Vietnamese (Vietnam)",
    "cy-gb": "Welsh (United Kingdom)",
    "zh-cn": "Chinese (Simplified, China)",
    "zh-tw": "Chinese (Traditional, Taiwan)",
    "zh-hk": "Chinese (Traditional, Hong Kong)",
}

LANGUAGE_REVERSE_MAPPING: dict[str, str] = {name.lower(): code for code, name in LANGUAGE_MAPPING.items()}


def normalize(lang: str) -> tuple[str, str]:
    """Any input (code or display name) to (code, name); unknowns pass through."""
    value = str(lang).lower()
    if value in LANGUAGE_MAPPING:
        return value, LANGUAGE_MAPPING[value]
    return LANGUAGE_REVERSE_MAPPING.get(value, value), lang


def normalize_language(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters with clean_*_language_code/_name injected."""
    output = dict(parameters)
    for side in ("source", "target"):
        if parameters.get(f"{side}_language"):
            code, name = normalize(parameters[f"{side}_language"])
            output[f"clean_{side}_language_code"] = code
            output[f"clean_{side}_language_name"] = name.lower()
    return output


# Word-boundary matching is meaningless for languages written without inter-word spacing.
UNSPACED_LANGUAGES = frozenset({"ja", "zh", "ko", "th", "lo", "km", "my"})


def is_unspaced_language(language_code: str | None) -> bool:
    return str(language_code or "").split("-")[0].lower() in UNSPACED_LANGUAGES


def normalize_text(text: object, *, casefold: bool = True) -> str:
    """NFC-normalize, collapse whitespace and casefold; do-not-translate turns casefold off."""
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    if casefold:
        normalized = normalized.casefold()
    return re.sub(r"\s+", " ", normalized).strip()


# \b is a \w/\W transition, so a term ending in "+" gets no boundary; [^\W_] = \p{L}\p{N}.
_BOUNDARY = r"[^\W_]"


@lru_cache(maxsize=4096)
def bounded_pattern(term: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!{_BOUNDARY}){re.escape(term)}(?!{_BOUNDARY})", re.UNICODE)


def tokenize(text: object) -> list[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def count_surface(text: object, term: object, language_code: str | None) -> int:
    haystack = normalize_text(text)
    needle = normalize_text(term)
    if not haystack or not needle:
        return 0
    if is_unspaced_language(language_code):
        return haystack.count(needle)
    return len(bounded_pattern(needle).findall(haystack))


def count_lemma(text_lemmas: Sequence[str] | str, term_lemmas: Sequence[str] | str) -> int:
    haystack = list(text_lemmas) if isinstance(text_lemmas, (list, tuple)) else tokenize(text_lemmas)
    needle = list(term_lemmas) if isinstance(term_lemmas, (list, tuple)) else tokenize(term_lemmas)
    if not needle or len(needle) > len(haystack):
        return 0

    count = index = 0
    while index <= len(haystack) - len(needle):
        if haystack[index : index + len(needle)] == needle:
            count += 1
            index += len(needle)  # non-overlapping
        else:
            index += 1
    return count


def count_occurrences(
    *,
    text: object,
    term: object,
    language_code: str | None,
    text_lemmas: str | None = None,
    term_lemmas: str | None = None,
) -> int:
    """Surface first, lemma second; never summed, or uninflected matches count twice."""
    surface = count_surface(text, term, language_code)
    if surface:
        return surface

    if text_lemmas and term_lemmas and not is_unspaced_language(language_code):
        return count_lemma(text_lemmas, term_lemmas)

    return 0
