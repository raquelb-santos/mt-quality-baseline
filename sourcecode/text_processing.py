"""Dataset loading and validation, the language normalization it applies, and term matching."""

import csv
import json
import io
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)


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

    @property
    def is_gold_set(self) -> bool:
        """A gold set names the terms that must survive, so it needs neither post-mt nor a
        reference — it is scored against the terms themselves."""
        return bool(self.segments) and "expected_terms" in self.segments[0]

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


def is_gold_set(body: Any) -> bool:
    """A DNT gold set names the terms that must survive, so it carries no human reference."""
    segments = body.get("segments") if isinstance(body, dict) else None
    return bool(segments) and isinstance(segments[0], dict) and "expected_terms" in segments[0]


def parse_gold_set(body: dict[str, Any], name: str) -> dict[str, Any]:
    """The reversion gold set, one entry per source segment and target language."""
    first = body["segments"][0]
    if not ("source" in first and "target" in first):
        raise ValueError(
            f"{name} names expected_terms but carries no source and target, so it is a detection "
            f"gold set. Point DNT_PATH at the reversion gold set instead."
        )

    tasks: dict[tuple[str, str], Task] = {}
    untranslated = 0
    for entry in body["segments"]:
        # Empty MT has nothing to revert, so skip it rather than score every term lost.
        if not str(entry.get("target") or "").strip():
            untranslated += 1
            continue

        # The gold set spells a language `en_US`, the rest of the benchmark `en-us`.
        languages = (
            str(entry.get("source_language") or "").replace("_", "-"),
            str(entry.get("target_language") or "").replace("_", "-"),
        )
        # post-mt requires a task id; it stays stable so post-mt's cache makes reruns free.
        task_id = f"mt-dnt-bench-{Path(name).stem}-{languages[0]}_{languages[1]}".lower()
        task = tasks.setdefault(languages, Task(
            {
                "tempo_task_id": task_id,
                "source_language": languages[0],
                "target_language": languages[1],
            }, [],
        ))
        segment = {
            "source_segment_id": str(entry.get("id") or len(task.segments)),
            "source_content": entry.get("source") or "",
            "target_content": entry.get("target") or "",
            # The gold: the terms the reverted target has to carry, whatever MT did to them.
            "expected_terms": list(entry.get("expected_terms") or []),
        }
        # A gold set needs no reference, but one written against a human keeps it for analysis.
        if str(entry.get("reference_content") or "").strip():
            segment["reference_content"] = entry["reference_content"]
        task.segments.append(segment)

    if untranslated:
        logger.warning(
            "[GOLD] %s: %d of %d pairs carry no target at all and are left out",
            name, untranslated, len(body["segments"]),
        )

    return {"tasks": list(tasks.values())}


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
        # A gold set states the terms outright, so it needs no reference to read them off.
        required = ("source_content", "target_content")
        if "expected_terms" not in segment:
            required += ("reference_content",)

        for field_name in required:
            value = segment.get(field_name)
            if not (isinstance(value, str) and value.strip()):
                errors.append(f"segment[{index}] missing `{field_name}`")

    if errors:
        listed = "\n  - ".join(errors[:12])
        raise ValueError(f'Invalid dataset "{dataset.name}":\n  - {listed}')


def parse_language_pairs(given: str) -> list[tuple[str, str]]:
    """`en_es`, or `en-gb_es-es` to pin the regions; several are separated by commas."""
    pairs = []
    for item in str(given).split(","):
        source, separator, target = item.strip().lower().partition("_")
        if not (source and separator and target):
            raise ValueError(
                f"BENCH_LANGUAGE wants SOURCE_TARGET, like en_es or en-gb_es-es; got {item.strip()!r}"
            )
        pairs.append((source, target))
    return pairs


def _speaks(given: Any, wanted: str) -> bool:
    """A wanted code that names no region matches every region of that language."""
    value = str(given or "").lower()
    return value == wanted or value.startswith(f"{wanted}-")


def in_languages(parameters: dict[str, Any], pairs: Sequence[tuple[str, str]]) -> bool:
    """Matched on the clean codes, since a dataset may spell a language `English (United Kingdom)`."""
    spoken = tuple(
        parameters.get(f"clean_{side}_language_code") or parameters.get(f"{side}_language")
        for side in ("source", "target")
    )
    return any(
        _speaks(spoken[0], source) and _speaks(spoken[1], target) for source, target in pairs
    )


def load(path: str | Path, *, component: str, languages: Sequence[tuple[str, str]] = ()) -> Dataset:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        body = json.loads(path.read_text(encoding="utf-8"))
        if is_gold_set(body):
            body = {"name": body.get("name"), **parse_gold_set(body, path.name)}
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

    # Filtered after validation, so the whole file is still checked when one pair is scored.
    if languages:
        kept = [task for task in dataset.tasks if in_languages(task.parameters, languages)]
        if len(kept) != len(dataset.tasks):
            logger.info("[BENCH] %s: %d of %d task(s) match the language filter",
                        dataset.name, len(kept), len(dataset.tasks))
        dataset.tasks = kept
        # An empty dataset has nothing to agree on, and the caller skips it.
        dataset.parameters = shared_parameters(kept) if kept else {}

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


def _lemma_starts(haystack: list[str], needle: list[str]) -> list[int]:
    starts: list[int] = []
    if not needle or len(needle) > len(haystack):
        return starts

    index = 0
    while index <= len(haystack) - len(needle):
        if haystack[index : index + len(needle)] == needle:
            starts.append(index)
            index += len(needle)  # non-overlapping
        else:
            index += 1
    return starts


def count_lemma(text_lemmas: Sequence[str] | str, term_lemmas: Sequence[str] | str) -> int:
    haystack = list(text_lemmas) if isinstance(text_lemmas, (list, tuple)) else tokenize(text_lemmas)
    needle = list(term_lemmas) if isinstance(term_lemmas, (list, tuple)) else tokenize(term_lemmas)
    return len(_lemma_starts(haystack, needle))


def _pairing_cost(word: str, lemma: str) -> float:
    shared = next((k for k, (a, b) in enumerate(zip(word, lemma)) if a != b), min(len(word), len(lemma)))
    return 1 - shared / max(len(word), len(lemma))


@lru_cache(maxsize=4096)
def _words_of_lemmas(text: str, text_lemmas: str) -> tuple[str, list[tuple[int, int]], dict[int, int]]:
    """Each lemma token's word, paired on shared prefixes, a contraction's two lemmas sharing one word; punctuation ignored."""
    normalized = normalize_text(text)
    words = [match.span() for match in re.finditer(r"\w+", normalized)]
    surface = [normalized[start:end] for start, end in words]
    lemmas = [(index, token) for index, token in enumerate(tokenize(text_lemmas)) if re.search(r"\w", token)]

    # Cheapest path from no words and no lemmas to all of both; a skipped word or lemma costs 1.
    cost = {(0, 0): 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    for i in range(len(surface) + 1):
        for j in range(len(lemmas) + 1):
            here = cost.get((i, j))
            if here is None:
                continue
            moves = [(i + 1, j, 1.0), (i, j + 1, 1.0)]
            if i < len(surface) and j < len(lemmas):
                moves.append((i + 1, j + 1, _pairing_cost(surface[i], lemmas[j][1])))
            if i < len(surface) and j + 1 < len(lemmas):
                moves.append((i + 1, j + 2, 1.0))  # "du" → "de le"
            for to_i, to_j, step in moves:
                if to_i <= len(surface) and to_j <= len(lemmas) and here + step < cost.get((to_i, to_j), float("inf")):
                    cost[(to_i, to_j)] = here + step
                    came_from[(to_i, to_j)] = (i, j)

    word_of: dict[int, int] = {}
    at = (len(surface), len(lemmas))
    while at in came_from:
        before = came_from[at]
        if at[0] - before[0] == 1:
            word_of.update({lemmas[j][0]: before[0] for j in range(before[1], at[1])})
        at = before
    return normalized, words, word_of


def find_occurrences(
    *,
    text: object,
    term: object,
    language_code: str | None,
    text_lemmas: str | None = None,
    term_lemmas: str | None = None,
) -> list[str | None]:
    """Each occurrence as worded in `text`; None where a lemma match's words cannot be pinned down."""
    surface = count_surface(text, term, language_code)
    if surface:
        return [normalize_text(term)] * surface

    if not (text_lemmas and term_lemmas) or is_unspaced_language(language_code):
        return []

    needle = tokenize(term_lemmas)
    starts = _lemma_starts(tokenize(text_lemmas), needle)
    if not starts:
        return []

    normalized, words, word_of = _words_of_lemmas(str(text), text_lemmas)
    wordings: list[str | None] = []
    for start in starts:
        indices = [word_of.get(start + k) for k, token in enumerate(needle) if re.search(r"\w", token)]
        if not indices or None in indices or any(b - a not in (0, 1) for a, b in zip(indices, indices[1:])):
            wordings.append(None)
        else:
            wordings.append(normalized[words[indices[0]][0] : words[indices[-1]][1]])
    return wordings


def count_occurrences(
    *,
    text: object,
    term: object,
    language_code: str | None,
    text_lemmas: str | None = None,
    term_lemmas: str | None = None,
) -> int:
    """Surface first, lemma second; never summed, or uninflected matches count twice."""
    return len(find_occurrences(
        text=text, term=term, language_code=language_code,
        text_lemmas=text_lemmas, term_lemmas=term_lemmas,
    ))
