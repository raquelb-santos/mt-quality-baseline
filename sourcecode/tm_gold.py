"""The TM gold set: one query per line, its graded candidate entries and its hard negatives."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .text_processing import normalize_language, read_params

logger = logging.getLogger(__name__)

# Relevance grades run from 4, the most relevant entry, down to 1.
TOP_GRADE = 4


@dataclass(frozen=True)
class GoldRow:
    query_id: str
    source: str
    raw_mt: str
    reference: str
    source_language: str
    target_language: str
    cat_project_id: str = ''
    domain: str = ''
    # Entry id -> relevance grade: 4 is the most relevant entry, 1 the least.
    tm_match: dict[str, int] = field(default_factory=dict)
    hard_negatives: tuple[str, ...] = ()

    @property
    def declared(self) -> bool:
        """Eligible: the gold set listed at least one relevant entry (the floors decide the band)."""
        return bool(self.tm_match)

    @property
    def candidate_ids(self) -> list[str]:
        """Most relevant first: the grade ranks them, and equal grades keep the written order."""
        return sorted(self.tm_match, key=lambda entry_id: -self.tm_match[entry_id])

    def grade(self, entry_id: str) -> int | None:
        return self.tm_match.get(entry_id)

    @property
    def pair(self) -> tuple[str, str]:
        return self.source_language, self.target_language


@dataclass(frozen=True)
class Submission:
    """Rows that can go to post-mt in one call: one language pair, one CAT project."""

    name: str
    rows: list[GoldRow]
    parameters: dict[str, Any]
    steps: list[str]


@dataclass(frozen=True)
class GoldSet:
    name: str
    rows: list[GoldRow]
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def declared(self) -> list[GoldRow]:
        return [row for row in self.rows if row.declared]

    def submissions(self) -> list[Submission]:
        """Grouped by (pair, project): post-mt takes one `parameters` block per call."""
        grouped: dict[tuple[str, str, str], list[GoldRow]] = {}
        for row in self.rows:
            grouped.setdefault((*row.pair, row.cat_project_id), []).append(row)

        steps = list(self.params.get('steps') or ['AQE', 'APE'])
        configured = dict(self.params.get('parameters') or {})

        return [
            Submission(
                name=f'{source_language}->{target_language} {project}'.strip(),
                rows=rows,
                parameters=normalize_language({
                    **configured,
                    'source_language': source_language,
                    'target_language': target_language,
                    'domain': rows[0].domain,
                    **({'cat_project_id': project} if project else {}),
                }),
                steps=steps,
            )
            for (source_language, target_language, project), rows in grouped.items()
        ]


def _as_grades(value: Any) -> dict[str, int]:
    """`tm_match` maps an entry id to its grade, 4 to 1; a bare list is read as ungraded."""
    if isinstance(value, dict):
        return {str(key): int(grade or 0) for key, grade in value.items()}
    if isinstance(value, (list, tuple)):
        return {str(key): 0 for key in value}
    return {}


def parse_rows(text: str) -> list[GoldRow]:
    rows: list[GoldRow] = []
    errors: list[str] = []
    seen: dict[str, int] = {}

    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            body = json.loads(line)
        except ValueError as error:
            errors.append(f'line {number} is not JSON: {error}')
            continue
        if not isinstance(body, dict):
            errors.append(f'line {number} is not an object')
            continue

        missing = [
            name for name in
            ('source', 'raw_mt', 'reference', 'source_language', 'target_language')
            if not str(body.get(name) or '').strip()
        ]
        if missing:
            errors.append(f'line {number} missing {", ".join(missing)}')
            continue

        # A duplicate id collides in the post-edit table and would score two rows against one
        # translation, so it is rejected rather than resolved.
        query_id = str(body.get('query_id') or number)
        if query_id in seen:
            errors.append(
                f'line {number} repeats the query_id {query_id!r} from line {seen[query_id]}'
            )
            continue
        seen[query_id] = number

        rows.append(GoldRow(
            query_id=query_id,
            source=str(body['source']),
            raw_mt=str(body['raw_mt']),
            reference=str(body['reference']),
            source_language=str(body['source_language']),
            target_language=str(body['target_language']),
            cat_project_id=str(body.get('cat_project_id') or ''),
            domain=str(body.get('domain') or ''),
            tm_match=_as_grades(body.get('tm_match')),
            hard_negatives=tuple(str(item) for item in (body.get('hard_negatives') or [])),
        ))

    if errors:
        listed = '\n  - '.join(errors[:12])
        raise ValueError(f'Invalid TM gold set:\n  - {listed}')

    raise_for_mixed_strata(rows)
    return rows


def raise_for_mixed_strata(rows: list[GoldRow]) -> None:
    """One file, one stratum. Counts from different language pairs or domains do not add, and a
    result carries a single stratum, so a mixed file reports every row under the first one's."""
    strata = {(*row.pair, row.domain) for row in rows}
    if len(strata) > 1:
        listed = ', '.join(
            f'{source}->{target} {domain or "(no domain)"}'
            for source, target, domain in sorted(strata)
        )
        raise ValueError(
            f'The gold set spans {len(strata)} strata ({listed}); counts from different strata '
            f'do not add, so each one needs its own file.'
        )


def load_gold_set(path: str | Path) -> GoldSet:
    path = Path(path)
    if path.suffix.lower() != '.jsonl':
        raise ValueError(f'A TM gold set is a .jsonl, not {path.suffix}: {path}')

    gold = GoldSet(
        name=path.stem,
        rows=parse_rows(path.read_text(encoding='utf-8')),
        params=read_params(path),
    )
    if not gold.rows:
        raise ValueError(f'No rows in {path}')

    declared = len(gold.declared)
    logger.info(
        '[TM] %s: %d rows, %d with a listed entry, %d submission group(s)',
        gold.name, len(gold.rows), declared, len(gold.submissions()),
    )
    if not declared:
        logger.error(
            '[TM] %s lists no relevant entry on any row, so eligibility is 0 and no reference rate '
            'has a denominator.', gold.name,
        )
    return gold
