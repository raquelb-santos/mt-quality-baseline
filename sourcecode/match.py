from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from .normalize import is_unspaced_language


def normalize_text(text: object, casefold: bool = True) -> str:
    """Casefold, NFC-normalize and collapse whitespace."""
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


def surface_match(text: object, term: object, language_code: str | None) -> bool:
    return count_surface(text, term, language_code) > 0


def tokenize(text: object) -> list[str]:
    return [token for token in normalize_text(text).split(" ") if token]


def lemma_match(text_lemmas: Sequence[str] | str, term_lemmas: Sequence[str] | str) -> bool:
    return count_lemma(text_lemmas, term_lemmas) > 0


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


@dataclass(frozen=True)
class MatchResult:
    found: bool
    via: str | None = None


def term_present(
    *,
    text: object,
    term: object,
    language_code: str | None,
    text_lemmas: str | None = None,
    term_lemmas: str | None = None,
) -> MatchResult:
    if surface_match(text, term, language_code):
        return MatchResult(True, "surface")

    # Unspaced languages have no tokens to align.
    if text_lemmas and term_lemmas and not is_unspaced_language(language_code):
        if lemma_match(text_lemmas, term_lemmas):
            return MatchResult(True, "lemma")

    return MatchResult(False, None)
