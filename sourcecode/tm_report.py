"""Rendering for the TM component."""

from typing import Any, Sequence

from . import tm_match, tm_score
from .report import Scorecard, by_stratum, cell, pct, rate, table
from .tm_benchmark import TmResult
from .tm_gold import TOP_GRADE

# A band row here is segments, but in the compliance table it is the entries a version used.
BAND_COLUMNS = ('Eligible', 'REF', 'APE', 'Carry-over', 'Lesser', f'Grade {TOP_GRADE}')
COMPLIANCE_COLUMNS = ('Entries', 'REF', 'APE', 'Applied', 'Adapted', 'Violations')
SEGMENT_COLUMNS = ('band', 'match', 'entry', 'candidates', 'ref', 'ape', 'flags')

UNMEASURED = 'unmeasured'

NO_TM_IN_PIPELINE = 'post-mt retrieves no TM, so REF-minus-APE is the headroom TM would recover.'

ICE_NOT_REPORTED = 'The gold set is segment-level, so ICE is unreported and 100% matches exact.'

SEMANTIC_NOT_MEASURED = 'No embedder, so nothing reached the semantic band and it reads unmeasured.'

MATCH_TEST_UNFINISHED = 'no embedder, so a rewritten use reads as non-use and REF is a lower bound'

NOT_FETCHED = 'segments name an entry the index lacks, so all rates drop them as a data fault.'
BELOW_RELEVANCE = 'segments list only entries under TM_MIN_RELEVANCE, so all rates drop them.'
REF_MATCHED_NONE = 'segments where REF matched no listed entry, by bad labels or an ignored match.'

USED_UNBANDED = (
    'Nothing reached a floor here yet an entry was matched, so TM_FUZZY_FLOOR is tight or the test'
    ' is loose, and the hard negatives say which.'
)


def _unmeasured(result: TmResult, band: str) -> bool:
    return band == tm_match.SEMANTIC and not result.semantic_measured


def tm_scorecard(result: TmResult) -> Scorecard:
    """One gold set's headline results, in the form both destinations render from."""
    totals, aggregate, floors = result.totals, result.aggregate, result.floors

    subheading = f"{result.parameters['source_language']} → {result.parameters['target_language']}"
    if result.parameters.get('domain'):
        subheading += f"  ·  {result.parameters['domain']}"

    warnings = [NO_TM_IN_PIPELINE, ICE_NOT_REPORTED]
    if not result.semantic_measured:
        warnings.append(SEMANTIC_NOT_MEASURED)

    if result.failed_segments:
        share = f"{result.failed_segments}/{totals['segments']}"
        warnings.insert(0, f'{share} segments failed inside post-mt\n{result.failure_reason}')
    if aggregate.not_fetched:
        warnings.insert(0, f'{aggregate.not_fetched}/{aggregate.declared} {NOT_FETCHED}')
    if aggregate.below_relevance:
        warnings.insert(0, f'{aggregate.below_relevance}/{aggregate.fetched} {BELOW_RELEVANCE}')
    inconclusive = aggregate.inconclusive(tm_score.REF)
    if inconclusive:
        warnings.append(f'{inconclusive}/{aggregate.usable} REF {MATCH_TEST_UNFINISHED}')
    if aggregate.ref_matched_none:
        warnings.append(f'{aggregate.ref_matched_none}/{aggregate.usable} {REF_MATCHED_NONE}')

    used_unbanded = aggregate.bands[tm_match.UNBANDED].used[tm_score.REF]
    if used_unbanded:
        share = f'{used_unbanded}/{aggregate.unbanded}'
        warnings.append(f'{share} unbanded segments used an entry. {USED_UNBANDED}')

    # A version that used nothing has no compliance denominator, so it reads as silence not failure.
    entries = sum(tally.entries[tm_score.REF] for tally in aggregate.bands.values())
    compliant = sum(tally.compliant[tm_score.REF] for tally in aggregate.bands.values())
    metrics = [
        ('Eligibility', f'{pct(aggregate.eligibility_rate)} · {aggregate.declared}/{aggregate.sampled} segments'),
        ('Reference REF', f'{pct(aggregate.reference_rate(tm_score.REF))} over {aggregate.usable} eligible'),
        ('Compliance REF', f'{pct(rate(compliant, entries))} · {compliant}/{entries} entries used'),
    ]
    if tm_score.APE in result.scored_versions:
        metrics.append(
            ('Reference APE', f'{pct(aggregate.reference_rate(tm_score.APE))} · carry-over'
                              f' {pct(aggregate.carry_over_rate)}')
        )
    grades = aggregate.grade_rates(tm_score.REF)
    if len(grades) > 1:
        metrics.append(
            ('By grade REF', ' · '.join(
                f'{grade} {pct(value)} ({eligible})' for grade, value, eligible in grades
            ))
        )
    width = max(len(label) for label, _ in metrics)

    facts = [
        'Bands · ' + ' · '.join(
            f'{band} {UNMEASURED}' if _unmeasured(result, band)
            else f'{band} {aggregate.bands[band].eligible}'
            for band in tm_match.REPORTED_BANDS
        ),
        *(f'{label.ljust(width)}  {value}' for label, value in metrics),
        f"Floors · fuzzy {pct(floors['fuzzy'])} · semantic {pct(floors['semantic'])}"
        f" · reference {pct(floors['reference'])} · partial {pct(floors['partial'])}",
        f"Index · {result.index} · {totals['entries_fetched']} entries fetched",
    ]

    stitched = aggregate.stitched(tm_score.REF)
    if stitched:
        facts.append(
            f'Stitched · {stitched}/{aggregate.usable} segments where REF used more than one entry'
        )

    if result.calibration.tested:
        facts.append(
            f'Hard negatives · {pct(result.calibration.false_match_rate)} false matches'
            f' over {result.calibration.tested} tested'
        )

    if result.usage.cost or result.usage.tokens:
        facts.append(f'LLM spend ${result.usage.cost:.4f} · {result.usage.tokens:,} tokens')

    return Scorecard(heading='tm', subheading=subheading, facts=facts, warnings=warnings)


def band_cells(result: TmResult) -> list[tuple[str, list[str]]]:
    aggregate = result.aggregate
    rows = []
    for band in tm_match.REPORTED_BANDS:
        tally = aggregate.bands[band]
        rows.append((band, [UNMEASURED] * len(BAND_COLUMNS) if _unmeasured(result, band) else [
            str(tally.eligible),
            pct(tally.reference_rate(tm_score.REF)),
            pct(tally.reference_rate(tm_score.APE)),
            pct(tally.carry_over_rate),
            str(tally.matched_lesser[tm_score.REF]),
            str(tally.grades.get(TOP_GRADE, 0)),
        ]))

    rows.append(('ALL', [
        str(aggregate.usable),
        pct(aggregate.reference_rate(tm_score.REF)),
        pct(aggregate.reference_rate(tm_score.APE)),
        pct(aggregate.carry_over_rate),
        str(sum(t.matched_lesser[tm_score.REF] for t in aggregate.bands.values())),
        str(sum(t.grades.get(TOP_GRADE, 0) for t in aggregate.bands.values())),
    ]))
    return rows


def compliance_cells(result: TmResult) -> list[tuple[str, list[str]]]:
    rows = []
    for band in tm_match.REPORTED_BANDS:
        tally = result.aggregate.bands[band]
        if _unmeasured(result, band):
            rows.append((band, [UNMEASURED] * len(COMPLIANCE_COLUMNS)))
            continue
        violations = ' '.join(
            f'{name} {count}'
            for name, count in sorted(tally.violations[tm_score.REF].items())
            if name != 'not_used'
        )
        rows.append((band, [
            str(tally.entries[tm_score.REF]),
            pct(tally.compliance_rate(tm_score.REF)),
            pct(tally.compliance_rate(tm_score.APE)),
            str(tally.verdicts[tm_score.REF][tm_match.APPLIED]
                + tally.verdicts[tm_score.REF][tm_match.NEAR_VERBATIM]),
            str(tally.verdicts[tm_score.REF][tm_match.ADAPTED]),
            violations or '-',
        ]))
    return rows


def render_tm_bands(result: TmResult, *, console: bool = False) -> str:
    title = f'Reference rate by band — {result.dataset}' if console else 'Reference rate by band'
    return table(title, 'band' if console else 'Band', BAND_COLUMNS, band_cells(result),
                 console=console)


def render_tm_compliance(result: TmResult, *, console: bool = False) -> str:
    """Exact matches are to be applied, fuzzy and semantic ones adapted. A row counts entries used
    from that band, crediting a stitched segment per entry, while declining one costs REF."""
    title = f'Applied versus adapted — {result.dataset}' if console else 'Applied versus adapted'
    return table(title, 'band' if console else 'Band', COMPLIANCE_COLUMNS,
                 compliance_cells(result), console=console)


def render_tm_calibration(result: TmResult) -> str:
    """What the match test does on entries the gold set says are not relevant."""
    calibration = result.calibration
    if not calibration.tested:
        return ''

    rows = [
        (tier, [str(tally.tested), str(tally.matched), pct(tally.false_match_rate)])
        for tier, tally in sorted(calibration.by_tier.items())
    ]
    rows.append(('ALL', [str(calibration.tested), str(calibration.matched),
                         pct(calibration.false_match_rate)]))
    return table('Match test against the hard negatives', 'Tier',
                 ('Tested', 'Matched', 'False match'), rows)


def _severity(segment: tm_score.SegmentScore) -> tuple[int, int, int]:
    """Worst first: a violation, then a flag, then the rest."""
    ref = segment.versions.get(tm_score.REF)
    if not segment.eligible or ref is None:
        return 0, 0, 0
    flagged = segment.ref_matched_none or ref.matched_lesser or ref.length_flagged
    return -int(bool(ref.violation)), -int(flagged), -1


def _outcome(version: tm_score.VersionScore | None) -> str:
    """Every entry's verdict, in the order the entry column lists them."""
    if version is None:
        return ''
    verdicts = '+'.join(use.verdict for use in version.uses)
    return f'{verdicts or version.verdict} ({version.evidence.tier})'


def segment_rows(result: TmResult) -> list[dict[str, Any]]:
    """One row per segment, worst first, so the head of the table is the worklist."""
    rows = []
    for segment in sorted(result.segments, key=_severity):
        ref = segment.versions.get(tm_score.REF)
        ape = segment.versions.get(tm_score.APE)
        uses = ref.uses if ref is not None else ()
        rows.append({
            'query_id': segment.query_id,
            'band': segment.band or '(not listed)',
            'match': pct(segment.best.score) if segment.best else 'n/a',
            'entry': '+'.join(use.candidate.entry_id[:8] for use in uses),
            'candidates': str(len(segment.candidates)),
            'ref': _outcome(ref),
            'ape': _outcome(ape),
            'flags': ' '.join(filter(None, [
                'stitched' if ref and ref.stitched else '',
                # An entry covered part of the segment only, on the source side, the target side
                # or both.
                'partial' if any(
                    use.candidate.partial or use.evidence.tier == tm_match.TIER_PARTIAL
                    for use in uses
                ) else '',
                'lesser' if ref and ref.matched_lesser else '',
                'length' if ref and ref.length_flagged else '',
                'no-match' if segment.ref_matched_none else '',
            ])) or '-',
        })
    return rows


def render_tm_segments(result: TmResult) -> str:
    """Every segment, worst first. Not a top-N: the file is where a reviewer works."""
    rows = segment_rows(result)
    if not rows:
        return ''
    lines = [
        '### Segments (worst first)',
        '',
        f"| Query | {' | '.join(SEGMENT_COLUMNS)} |",
        f"| --- | {' | '.join('---' for _ in SEGMENT_COLUMNS)} |",
        *(
            f"| {cell(row['query_id'])} | "
            + ' | '.join(cell(row[column]) for column in SEGMENT_COLUMNS) + ' |'
            for row in rows
        ),
        '',
    ]
    return '\n'.join(lines)


def render_tm_segments_console(result: TmResult) -> str:
    """The worst 20 only - the console is a summary, and the report file holds them all."""
    rows = segment_rows(result)
    shown = rows[:20]
    title = f'Segments — {result.dataset} (worst first)'
    if len(rows) > len(shown):
        title += f', {len(shown)} of {len(rows)} — the rest are in the report'
    return table(
        title, 'query', SEGMENT_COLUMNS,
        [(row['query_id'], [row[column] for column in SEGMENT_COLUMNS]) for row in shown],
        console=True,
    )


def render_tm_strata(results: Sequence[TmResult], *, console: bool = False) -> str:
    rows = []
    for (pair, domain), grouped in by_stratum(results).items():
        pooled = tm_score.pool(result.aggregate for result in grouped)
        rows.append((f'{pair} · {domain}', [
            str(pooled.sampled),
            pct(pooled.eligibility_rate),
            pct(pooled.reference_rate(tm_score.REF)),
            pct(pooled.reference_rate(tm_score.APE)),
            pct(pooled.carry_over_rate),
        ]))
    return table('Pooled by stratum', 'stratum' if console else 'Stratum',
                 ('Segments', 'Eligibility', 'REF', 'APE', 'Carry-over'), rows, console=console)
