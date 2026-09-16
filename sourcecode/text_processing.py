"""Dataset loading and validation, the language normalization it applies, and term matching."""

import csv
import json
import io
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence


def find_datasets(configured: str, *, variable: str) -> list[Path]:
    """A file, or a folder's direct children, sorted."""
    types = (".json", ".csv")
    if not configured.strip():
        raise ValueError(
            f"No dataset to score. Set {variable} in .env to a dataset file, or to a folder "
            f"to score every dataset inside it."
        )

    candidate = Path(configured)
    if candidate.is_dir():
        found = sorted(
            child for child in candidate.iterdir()
            if child.is_file() and child.suffix.lower() in types
        )
        if not found:
            raise ValueError(f"No dataset files in {candidate} (looked for {', '.join(types)}).")
        return found
    if candidate.is_file():
        return [candidate]

    raise ValueError(f"{variable} points at nothing: {candidate}")


@dataclass
class Task:
    """The segments a dataset holds under one set of post-mt parameters."""

    parameters: dict[str, Any]
    segments: list[dict[str, Any]]


@dataclass
class Dataset:
    name: str
    # What the tasks agree on; a parameter that varies belongs to its task, not to the file.
    parameters: dict[str, Any]
    # post-mt takes one parameter set per task, so a file covering several jobs is sent as several.
    tasks: list[Task]
    # Which component scores this dataset.
    component: str
    steps: list[str] = field(default_factory=lambda: ["AQE", "APE"])

    @property
    def segments(self) -> list[dict[str, Any]]:
        """Every segment the file holds, in task order."""
        return [segment for task in self.tasks for segment in task.segments]

    def per_segment(self, name: str) -> list[Any]:
        """One parameter value per segment, from the task that segment is sent in."""
        return [task.parameters.get(name) for task in self.tasks for _ in task.segments]


    def parameters_per_segment(self) -> list[dict[str, Any]]:
        """The whole parameter set each segment is sent under."""
        return [task.parameters for task in self.tasks for _ in task.segments]


def shared_parameters(tasks: Sequence[Task]) -> dict[str, Any]:
    first, *rest = [task.parameters for task in tasks]
    return {
        name: value for name, value in first.items()
        if all(other.get(name) == value for other in rest)
    }


def parse_csv(text: str) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []

    for index, row in enumerate(csv.DictReader(io.StringIO(text))):
        row = {(key or "").strip(): (value or "") for key, value in row.items()}
        segment = {
            "source_segment_id": row.get("source_segment_id") or str(index),
            "source_content": row.get("source_content") or "",
            "target_content": row.get("target_content") or "",
        }

        reference = row.get("reference_content") or ""
        if reference.strip():
            segment["reference_content"] = reference

        segments.append(segment)

    return segments


# CAT export text columns; other columns are ignored, so exports score without trimming.
EXPORT_COLUMNS = {
    "SEGMENTID": "source_segment_id",
    "SOURCECONTENT": "source_content",
    "TARGETCONTENT": "target_content",
    "HUMAN_TARGET": "reference_content",
}

# Per-row job columns; rows sharing their values become one post-mt task.
EXPORT_PARAMETERS = {
    "ISOSOURCELANGUAGE": "source_language",
    "ISOTARGETLANGUAGE": "target_language",
    "CATTOOL": "cat_tool_provider",
    "CATPROJECTID": "cat_project_id",
    "TEMPOTASKCODE": "tempo_task_id",
    "DOMAIN": "domain",
    "OPERATION": "operation",
}


def is_export_header(names: Sequence[str]) -> bool:
    """Whether a header is a CAT export rather than a dataset written in the canonical names."""
    return "SOURCECONTENT" in {str(name or "").strip().upper() for name in names}


def parse_csv_export(text: str, name: str) -> dict[str, Any]:
    """A CAT segment export, one row per segment, read by the columns it writes."""
    reader = csv.DictReader(io.StringIO(text))
    header = {(column or "").strip().upper() for column in (reader.fieldnames or ())}
    if missing := [column for column in EXPORT_COLUMNS if column not in header]:
        raise ValueError(f"{name} has no {' column, no '.join(missing)} column.")

    tasks: dict[tuple[tuple[str, str], ...], Task] = {}
    kept = 0

    for row in reader:
        values = {(key or "").strip().upper(): (value or "") for key, value in row.items()}
        if not values["SOURCECONTENT"].strip():
            continue

        parameters = {
            parameter: value
            for column, parameter in EXPORT_PARAMETERS.items()
            if (value := values.get(column, "").strip())
        }
        task = tasks.setdefault(tuple(sorted(parameters.items())), Task(parameters, []))

        segment = {field: values[column] for column, field in EXPORT_COLUMNS.items()}
        segment["source_segment_id"] = segment["source_segment_id"].strip() or str(kept)
        task.segments.append(segment)
        kept += 1

    return {"tasks": list(tasks.values())}


def validate(dataset: Dataset) -> None:
    errors: list[str] = []

    for index, task in enumerate(dataset.tasks):
        where = "parameters" if len(dataset.tasks) == 1 else f"tasks[{index}].parameters"
        if not task.parameters:
            errors.append(f"missing `{where}`")
        if not task.parameters.get("source_language"):
            errors.append(f"missing `{where}.source_language`")
        if not task.parameters.get("target_language"):
            errors.append(f"missing `{where}.target_language`")

    if not dataset.segments:
        errors.append("no segments")

    for index, segment in enumerate(dataset.segments):
        for field_name in ("source_content", "target_content", "reference_content"):
            value = segment.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"segment[{index}] missing `{field_name}`")

    if errors:
        listed = "\n  - ".join(errors[:12])
        raise ValueError(f'Invalid dataset "{dataset.name}":\n  - {listed}')


def load(path: str | Path, *, component: str) -> Dataset:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        body = json.loads(path.read_text(encoding="utf-8"))
    elif suffix == ".csv":
        # Two formats share the extension, so the header says which one this is.
        text = path.read_text(encoding="utf-8-sig")
        header = next(csv.reader(io.StringIO(text)), [])
        body = (
            parse_csv_export(text, path.name) if is_export_header(header)
            else {"segments": parse_csv(text)}
        )
    else:
        raise ValueError(f"Unsupported dataset format: {suffix}")

    # A format that describes one job parses to one task; only an export can hold several.
    tasks = body.get("tasks") or [Task(body.get("parameters") or {}, list(body.get("segments") or []))]
    tasks = [Task(normalize_language(task.parameters), task.segments) for task in tasks]

    dataset = Dataset(
        name=body.get("name") or path.stem,
        parameters=shared_parameters(tasks),
        tasks=tasks,
        steps=list(body.get("steps") or ["AQE", "APE"]),
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


def normalize_language(parameters: dict[str, Any]) -> dict[str, Any]:
    """Return parameters with clean_*_language_code/_name injected."""
    output = dict(parameters)
    for side in ("source", "target"):
        given = parameters.get(f"{side}_language")
        if not given:
            continue
        value = str(given).lower()
        code, name = (
            (value, LANGUAGE_MAPPING[value]) if value in LANGUAGE_MAPPING
            else (LANGUAGE_REVERSE_MAPPING.get(value, value), given)
        )
        output[f"clean_{side}_language_code"] = code
        output[f"clean_{side}_language_name"] = name.lower()
    return output


# Word-boundary matching is meaningless for languages written without inter-word spacing.
def is_unspaced_language(language_code: str | None) -> bool:
    return str(language_code or "").split("-")[0].lower() in {
        "ja", "zh", "ko", "th", "lo", "km", "my"
    }


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


def count_surface(text: object, term: object, language_code: str | None, *, casefold: bool = True) -> int:
    haystack = normalize_text(text, casefold=casefold)
    needle = normalize_text(term, casefold=casefold)
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
