"""Extracting the tags and placeholders a segment carries, in the notations the CAT tools emit."""

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence


PAIRED_OPEN = "paired_open"      # {1>  {b>  {b^>
PAIRED_CLOSE = "paired_close"    # <1}  <b}  <b^}
DOUBLE_BRACE = "double_brace"    # {{var}}
STANDALONE = "standalone"        # {1}  {2}  {name}
XML = "xml"                      # <ph x="1"/>  <span ...>  </span>
PRINTF = "printf"                # %s  %d  %1$s  %.2f
ENTITY = "entity"                # &brand;  &#160;

# In the order the pattern tries them, which is also the order they are reported in.
KINDS = (PAIRED_OPEN, PAIRED_CLOSE, DOUBLE_BRACE, STANDALONE, XML, PRINTF, ENTITY)

# Phrase ids are digits, letters and `_`/`^` suffixed forms; `^` is outside `\w`.
_PHRASE_ID = r"[0-9a-zA-Z_^]{1,8}"

# Ordered: a decoded `<b}` is a Phrase closer before XML can read it as an opener, and `{{var}}`
# is taken whole before the single-brace branch could split it.
TAG_PATTERN = re.compile("|".join((
    r"(?P<paired_open>\{" + _PHRASE_ID + r">)",
    r"(?P<paired_close><" + _PHRASE_ID + r"\})",
    r"(?P<double_brace>\{\{[^{}]{1,64}\}\})",
    r"(?P<standalone>\{[0-9a-zA-Z_^.\-]{1,64}\})",
    r"(?P<xml></?[a-zA-Z][\w:.\-]*(?:\s[^>]*?)?/?>)",
    r"(?P<printf>%(?:\d+\$)?[-+0#]?\d*(?:\.\d+)?[sdifFeEgGxX%])",
    r"(?P<entity>&(?:#\d+|#x[0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]{1,31});)",
)))

_XML_NAME = re.compile(r"</?([a-zA-Z][\w:.\-]*)")

# Elements that never take a closer, so a lone one is not a broken pair.
VOID_ELEMENTS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr",
})


@dataclass(frozen=True)
class Tag:
    text: str    # the token as written, whitespace collapsed — this is what has to survive
    kind: str
    tag_id: str  # the id the CAT tool keys its stored markup by


def _tag_id(kind: str, text: str) -> str:
    if kind in (PAIRED_OPEN, PAIRED_CLOSE, STANDALONE):
        return text[1:-1]
    if kind == DOUBLE_BRACE:
        return text[2:-2]
    if kind == XML:
        return _XML_NAME.match(text).group(1)
    return text


def extract_tags(text: str) -> list[Tag]:
    """Every tag in the order it appears; whatever lies between them is translatable text."""
    return [
        Tag(
            text=(token := re.sub(r"\s+", " ", found.group())),
            kind=found.lastgroup,
            tag_id=_tag_id(found.lastgroup, token),
        )
        for found in TAG_PATTERN.finditer(text)
    ]


def _pairing_role(tag: Tag) -> tuple[str, str] | None:
    """The side a tag takes and the id it pairs on; None where it owes no counterpart."""
    if tag.kind == PAIRED_OPEN:
        return "open", f"phrase:{tag.tag_id}"
    if tag.kind == PAIRED_CLOSE:
        return "close", f"phrase:{tag.tag_id}"
    if tag.kind == XML and tag.tag_id.lower() not in VOID_ELEMENTS:
        if tag.text.startswith("</"):
            return "close", f"xml:{tag.tag_id}"
        if not tag.text.endswith("/>"):
            return "open", f"xml:{tag.tag_id}"
    return None


def unpaired(tags: Sequence[Tag]) -> set[str]:
    """Ids whose openers and closers do not match up, a closer standing before its opener too."""
    opened: Counter[str] = Counter()
    closed: Counter[str] = Counter()
    broken: set[str] = set()

    for tag in tags:
        role = _pairing_role(tag)
        if role is None:
            continue
        side, key = role
        if side == "open":
            opened[key] += 1
        else:
            closed[key] += 1
            if closed[key] > opened[key]:
                broken.add(key)

    broken.update(key for key in opened.keys() | closed.keys() if opened[key] != closed[key])
    return broken
