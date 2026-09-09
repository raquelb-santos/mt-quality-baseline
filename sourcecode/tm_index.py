"""The TM index: fetching the entries a gold-set row names, by id."""

import logging
from dataclasses import dataclass
from typing import Sequence

from .search_engine import SearchClient

logger = logging.getLogger(__name__)

# How many ids one _mget carries
BATCH_SIZE = 200


@dataclass(frozen=True)
class TmFields:
    """The index's own field names, so a schema change is configuration and not a code change."""

    source: str = 'source_text'
    target: str = 'target_text'
    source_language: str = 'source_lang'
    target_language: str = 'target_lang'


@dataclass(frozen=True)
class Entry:
    entry_id: str
    source: str
    target: str
    source_language: str = ''
    target_language: str = ''

    @property
    def usable(self) -> bool:
        return bool(self.source.strip()) and bool(self.target.strip())


@dataclass
class FetchReport:
    """What the fetch could and could not deliver - the `declared -> usable` stage."""

    requested: int = 0
    found: int = 0
    wrong_language: int = 0
    empty_text: int = 0

    @property
    def missing(self) -> int:
        return self.requested - self.found

    def add(self, other: 'FetchReport') -> None:
        self.requested += other.requested
        self.found += other.found
        self.wrong_language += other.wrong_language
        self.empty_text += other.empty_text


def _same_language(left: str, right: str) -> bool:
    """`en-US` matches `en-us` and `en`"""
    if not left or not right:
        return True
    return str(left).lower().split('-')[0] == str(right).lower().split('-')[0]


class TmIndexClient:
    def __init__(
        self,
        search: SearchClient,
        index: str,
        fields: TmFields | None = None,
        batch_size: int = BATCH_SIZE,
    ) -> None:
        self.search = search
        self.index = index
        self.fields = fields or TmFields()
        self.batch_size = batch_size

    def fetch_entries_by_id(
        self,
        entry_ids: Sequence[str],
        *,
        source_language: str = '',
        target_language: str = '',
    ) -> tuple[dict[str, Entry], FetchReport]:
        """The entries behind a set of ids, keyed by id. A wrong-language or empty-side entry is
        left out rather than returned broken - the caller counts it as not usable."""
        wanted = list(entry_ids)
        fields = self.fields
        report = FetchReport(requested=len(wanted))
        entries: dict[str, Entry] = {}

        for start in range(0, len(wanted), self.batch_size):
            batch = wanted[start : start + self.batch_size]
            for document in self.search.mget(self.index, batch):
                if not document.get('found'):
                    continue
                report.found += 1

                source = document.get('_source') or {}
                entry = Entry(
                    entry_id=str(document.get('_id')),
                    source=str(source.get(fields.source) or ''),
                    target=str(source.get(fields.target) or ''),
                    source_language=str(source.get(fields.source_language) or ''),
                    target_language=str(source.get(fields.target_language) or ''),
                )

                if not (
                    _same_language(entry.source_language, source_language)
                    and _same_language(entry.target_language, target_language)
                ):
                    report.wrong_language += 1
                elif not entry.usable:
                    report.empty_text += 1
                else:
                    entries[entry.entry_id] = entry

        logger.info(
            '[TM] %s: %d/%d entries fetched (%d wrong language, %d empty, %d missing)',
            self.index, len(entries), report.requested,
            report.wrong_language, report.empty_text, report.missing,
        )
        return entries, report
