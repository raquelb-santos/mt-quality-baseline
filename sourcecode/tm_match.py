"""The band a candidate falls in, and what a version did with the entry it used."""

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, Sequence

from .text_processing import normalize_text

EXACT, FUZZY, SEMANTIC = 'exact', 'fuzzy', 'semantic'
BANDS = (EXACT, FUZZY, SEMANTIC)
UNBANDED = 'unbanded'
REPORTED_BANDS = BANDS + (UNBANDED,)

# Which test answered "was the entry used".
TIER_EXACT, TIER_FUZZY, TIER_SEMANTIC = 'exact', 'fuzzy', 'semantic'
# The entry was found inside the version rather than as the whole of it.
TIER_PARTIAL = 'partial'
# The cascade could not be finished.
TIER_UNAVAILABLE, TIER_NONE = 'unavailable', 'none'

# What a version did with the entry.
APPLIED, NEAR_VERBATIM, ADAPTED, NOT_USED = 'applied', 'near_verbatim', 'adapted', 'not_used'

Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


def char_score(left: object, right: object) -> float:
    """Character similarity of two texts on the normalized pair. The band is decided on this."""
    first, second = normalize_text(left), normalize_text(right)
    if not first or not second:
        return 0.0
    if first == second:
        return 1.0
    return SequenceMatcher(None, first, second).ratio()


def _best_window(text: object, other: object) -> tuple[float, tuple[int, int] | None, bool]:
    """The best window of `text` matching `other`, with its score, bounds in the normalized text,
    and whether it is `other` with case and punctuation intact. Only a shorter `other` is sought."""
    haystack, needle = normalize_text(text, casefold=False), normalize_text(other, casefold=False)
    if not needle or len(needle) >= len(haystack):
        return 0.0, None, False

    limit = len(haystack) - len(needle)
    starts = {
        max(0, min(start - offset, limit))
        for offset, start, size in SequenceMatcher(None, needle, haystack).get_matching_blocks()
        if size
    }

    best, span = 0.0, None
    for start in starts:
        window = haystack[start : start + len(needle)]
        score = SequenceMatcher(None, needle.casefold(), window.casefold()).ratio()
        if score > best:
            best, span = score, (start, start + len(needle))

    return best, span, span is not None and haystack[slice(*span)] == needle


def normalized_length(text: object) -> int:
    """The length of a text in window-span coordinates, so whole-text and window claims compare."""
    return len(normalize_text(text, casefold=False))


def contained_score(text: object, other: object) -> float:
    """How well `other` matches the part of `text` it covers, which bands a fragment-sized entry."""
    return _best_window(text, other)[0]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def classify(
    *,
    char: float,
    semantic: float | None,
    fuzzy_floor: float,
    semantic_floor: float,
) -> str | None:
    """The band a candidate falls in, or None where it reached no floor. Semantic is the residual
    band: a candidate the character channel already reaches is banded on characters."""
    if char >= 1.0:
        return EXACT
    if char >= fuzzy_floor:
        return FUZZY
    if semantic is not None and semantic >= semantic_floor:
        return SEMANTIC
    return None


@dataclass(frozen=True)
class MatchEvidence:
    """Whether a version used the entry, which test said so, and how strongly."""

    matched: bool
    tier: str
    score: float
    identical: bool = False
    # The window of the normalized version text the entry was found in
    span: tuple[int, int] | None = None

    @property
    def conclusive(self) -> bool:
        """False where no embedder ran, so a rewritten use cannot be told from a non-use."""
        return self.matched or self.tier != TIER_UNAVAILABLE

    @property
    def verdict(self) -> str:
        if not self.matched:
            return NOT_USED
        if self.identical:
            return APPLIED
        return NEAR_VERBATIM if self.tier == TIER_EXACT else ADAPTED


NO_MATCH = MatchEvidence(matched=False, tier=TIER_NONE, score=0.0)


def match_evidence(
    text: object,
    entry_target: object,
    *,
    reference_floor: float,
    semantic_floor: float,
    embedder: Embedder | None = None,
) -> MatchEvidence:
    """Normalized equality first, then edit distance, then meaning, and the first tier to answer
    wins. With no embedder the last tier cannot run, so an unmatched pair reports `unavailable`."""
    first = normalize_text(text, casefold=False)
    second = normalize_text(entry_target, casefold=False)
    if not first or not second:
        return NO_MATCH

    if first == second:
        return MatchEvidence(matched=True, tier=TIER_EXACT, score=1.0, identical=True)

    if normalize_text(text) == normalize_text(entry_target):
        return MatchEvidence(matched=True, tier=TIER_EXACT, score=1.0)

    proximity = char_score(text, entry_target)
    if proximity >= reference_floor:
        return MatchEvidence(matched=True, tier=TIER_FUZZY, score=proximity)

    if embedder is None:
        return MatchEvidence(matched=False, tier=TIER_UNAVAILABLE, score=proximity)

    query, target = embedder([str(text), str(entry_target)])
    similarity = cosine(query, target)
    if similarity >= semantic_floor:
        return MatchEvidence(matched=True, tier=TIER_SEMANTIC, score=similarity)
    return MatchEvidence(matched=False, tier=TIER_NONE, score=similarity)


def contained_evidence(text: object, entry_target: object, *, floor: float) -> MatchEvidence:
    """The entry found inside the version, not as the whole of it, which is what a segment served
    by more than one entry looks like. Characters only, since a fragment has no meaning alone."""
    score, span, identical = _best_window(text, entry_target)
    if span is None or score < floor:
        return MatchEvidence(matched=False, tier=TIER_NONE, score=score)
    return MatchEvidence(
        matched=True, tier=TIER_PARTIAL, score=score, identical=identical, span=span,
    )


def is_compliant(band: str, verdict: str) -> bool:
    """An exact match is to be applied, any other band adapted. `near_verbatim` counts as applied -
    it is reported on its own line, never lost inside a pass."""
    if band == EXACT:
        return verdict in (APPLIED, NEAR_VERBATIM)
    return verdict == ADAPTED


VIOLATIONS = {EXACT: 'edited_exact', FUZZY: 'unedited_fuzzy', SEMANTIC: 'unedited_semantic'}


def violation(band: str, verdict: str) -> str | None:
    """The name of what went wrong, for the counts the report prints beside the rate."""
    if verdict == NOT_USED:
        return 'not_used'
    if is_compliant(band, verdict):
        return None
    return VIOLATIONS.get(band, 'unedited_unbanded')
