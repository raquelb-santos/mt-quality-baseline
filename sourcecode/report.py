"""Report rendering. A run says everything it has to say on the console and writes nothing:
the numbers belong to the run that produced them, not to a directory that accumulates them."""

from __future__ import annotations

from typing import Any, Sequence

from . import score
from .benchmark import BenchmarkResult


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"



def render_summary(result: BenchmarkResult) -> str:
    mt, ape, totals, delta = result.mt_baseline, result.post_edited, result.totals, result.delta

    header = f"{result.parameters['source_language']} → {result.parameters['target_language']}"
    if result.parameters.get("domain"):
        header += f"  ·  {result.parameters['domain']}"
    header += f"  ·  steps: {'+'.join(result.steps)}"

    def moved(before: Any, after: Any) -> str:
        """One metric as `MT x → APE y`, so the two versions need no table to sit in."""
        return f"MT {before} → APE {after}"

    lines = [
        "",
        "═" * 78,
        f"  TERMINOLOGY ADHERENCE — {result.dataset}",
        f"  {header}",
        "═" * 78,
        "",
    ]

    if result.failed_segments:
        lines += [
            f"  ⚠ {result.failed_segments}/{totals['segments']} segments failed inside post-mt",
            f"    {result.failure_reason}",
            "",
        ]

    lines += [
        f"  Segments {totals['segments']} · with glossary terms {totals['segments_with_glossary']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · reference instances {mt.expected}",
        "",
        f"  Adherence      {moved(pct(mt.adherence_rate), pct(ape.adherence_rate))}"
        f"  ({signed_pct(delta.adherence_rate)})",
        f"  Violations     {moved(mt.misses.substituted, ape.misses.substituted)}"
        f"  ·  Omissions {moved(mt.misses.omitted, ape.misses.omitted)}",
        f"  Strict         {moved(pct(mt.strict.adherence_rate), pct(ape.strict.adherence_rate))}"
        f"  ({mt.strict.expected} inst.)"
        f"  ·  Permissive {moved(pct(mt.permissive.adherence_rate), pct(ape.permissive.adherence_rate))}"
        f"  ({mt.permissive.expected} inst.)",
        f"  Segment-level  {moved(pct(mt.segment_adherence_rate), pct(ape.segment_adherence_rate))}",
        "",
        # Four pairs on one line, so the arrow carries what `moved` spells out above.
        f"  Terms {mt.terms.distinct_terms} distinct"
        f" · matched {mt.terms.used_everywhere} → {ape.terms.used_everywhere}"
        f" · partly {mt.terms.used_partly} → {ape.terms.used_partly}"
        f" · never {mt.terms.never_used} → {ape.terms.never_used}"
        f" · over-used {mt.terms.over_used} → {ape.terms.over_used}",
        f"  APE repaired {delta.terms_fixed_by_ape} · broke {delta.terms_regressed_by_ape}",
    ]

    # Over-use is adherent by the cap, so without saying so the count above reads as a failure the
    # rate forgot to charge for. It is a worklist item, not a miss.
    if mt.terms.over_used or ape.terms.over_used:
        lines.append("  over-used terms are flagged for review, never counted as violations")

    if result.usage.cost or result.usage.tokens:
        lines.append(
            f"  LLM spend ${result.usage.cost:.4f} · {result.usage.tokens:,} tokens "
            f"({result.usage.prompt_tokens:,} prompt / {result.usage.completion_tokens:,} completion)"
        )

    if not result.config.get("lemma_matching"):
        lines.append("")
        lines.append("  ⚠ Lemma matching disabled")

    lines.append("")
    return "\n".join(line.rstrip() for line in lines)


def term_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    """One row per distinct glossary term, pooled over every segment it was matched in.

    The finest grain at which a rate means anything: a term's own adherence across the dataset,
    which the per-segment and per-stratum rows then roll up. Rates are recomputed from the pooled
    counts rather than averaged over segments, on the same principle as `score.pool` - a term
    matched once in a short segment must not outweigh the same term matched forty times.
    """
    identity = _identity_columns(result)
    pooled: dict[str, dict[str, Any]] = {}

    for segment in result.segments:
        columns = {
            "mt": {t.source_content: t for t in segment.mt.term_scores},
            "ape": {t.source_content: t for t in segment.ape.term_scores},
        }
        # The union, not MT alone: a term the human never used is scored only in the column that
        # used it, so keying off one column would drop the other's over-use entirely.
        for source in dict.fromkeys([*columns["mt"], *columns["ape"]]):
            term = columns["mt"].get(source) or columns["ape"][source]
            entry = pooled.setdefault(source, {
                "expected_targets": " | ".join(term.expected_targets),
                "strictness": term.strictness,
                "segments": 0,
                **{f"{c}_{f}": 0 for c in ("mt", "ape")
                   for f in ("expected", "adherent", "rendered", "violations", "omissions")},
            })
            entry["segments"] += 1
            for column, scores in columns.items():
                scored = scores.get(source)
                if scored is not None:
                    entry[f"{column}_expected"] += scored.expected
                    entry[f"{column}_adherent"] += scored.adherent
                    entry[f"{column}_rendered"] += scored.rendered
                    entry[f"{column}_violations"] += scored.violations
                    entry[f"{column}_omissions"] += scored.omissions

    rows = []
    for source, entry in pooled.items():
        row = {
            **identity,
            "source_term": source,
            "expected_targets": entry["expected_targets"],
            "strictness": entry["strictness"],
            "segments": entry["segments"],
            "expected_instances": entry["mt_expected"],
        }
        for column in ("mt", "ape"):
            expected, adherent = entry[f"{column}_expected"], entry[f"{column}_adherent"]
            # A zero denominator is a term the human never used: the renderings are still shown,
            # so the row reads as a review item rather than as a silently perfect one.
            row[f"{column}_adherent"] = adherent if expected else ""
            row[f"{column}_rendered"] = entry[f"{column}_rendered"]
            # Against the uncapped count, so the term the cap folded away is still reviewable.
            row[f"{column}_over_used"] = entry[f"{column}_rendered"] > expected
            # Substitutions only, as everywhere else. The two sum to `expected - adherent`, so a
            # term whose adherent count and violation count do not reconcile was omitted.
            row[f"{column}_violations"] = entry[f"{column}_violations"] if expected else ""
            row[f"{column}_omissions"] = entry[f"{column}_omissions"] if expected else ""
            row[f"{column}_adherence_rate"] = score.rate(adherent, expected)
        rows.append(row)

    # Worst first, so the head of the file and of the console table is the worklist.
    rows.sort(
        key=lambda r: (-((r["ape_violations"] or 0) + (r["ape_omissions"] or 0)), r["source_term"])
    )
    return rows


def render_term_adherence(result: BenchmarkResult) -> str:
    """Every glossary term the run matched, worst first.

    Not a top-N: a table that stops at some row silently tells a reader they have seen the whole
    worklist. The rows are the terms the dataset actually matched, so the table is as long as the
    run has something to say and no longer.
    """
    rows = term_rows(result)
    if not rows:
        return "  No glossary terms matched.\n"

    lines = [
        "",
        "  PER-TERM ADHERENCE (post-edited output, worst first)",
        f"  {'─' * 90}",
        f"  {'Source term':28}{'Expected':24}{'R':>4}{'T':>5}{'Adherent':>10}{'Viol.':>7}"
        f"{'Adherence':>11}  {'Kind':13}Flag",
    ]
    for row in rows:
        lines.append(
            f"  {row['source_term'][:26]:28}{row['expected_targets'][:22]:24}"
            f"{row['expected_instances']:>4}{row['ape_rendered']:>5}"
            f"{row['ape_adherent']:>10}{row['ape_violations']:>7}"
            f"{pct(row['ape_adherence_rate']):>11}  {row['strictness']:13}"
            f"{'review' if row['ape_over_used'] else ''}"
        )
    lines.append("")

    return "\n".join(lines)


def render_comparison(results: Sequence[BenchmarkResult]) -> str:
    if len(results) < 2:
        return ""

    lines = ["", "  ACROSS DATASETS", ""]
    for result in results:
        pair = f"{result.parameters['source_language']}>{result.parameters['target_language']}"
        lines.append(
            f"  {result.dataset[:28]:30}{pair:14}{result.mt_baseline.expected:>5} inst"
            f"  ·  MT {pct(result.mt_baseline.adherence_rate)}"
            f" → APE {pct(result.post_edited.adherence_rate)}"
            f"  ({signed_pct(result.delta.adherence_rate)})"
        )
    lines.append("")
    return "\n".join(lines)


def stratum_of(result: BenchmarkResult) -> tuple[str, str]:
    """A stratum is one language pair in one domain — the cell a result is reported in."""
    parameters = result.parameters
    pair = f"{parameters.get('source_language', '?')}->{parameters.get('target_language', '?')}"
    return pair, str(parameters.get("domain") or "(no domain)")


def by_stratum(results: Sequence[BenchmarkResult]) -> dict[tuple[str, str], list[BenchmarkResult]]:
    """Group results into strata, preserving the order each stratum was first seen."""
    grouped: dict[tuple[str, str], list[BenchmarkResult]] = {}
    for result in results:
        grouped.setdefault(stratum_of(result), []).append(result)
    return grouped


def stratum_rows(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    """One row per stratum, with the pooled counts the rates were computed from.

    The counts travel with the rates deliberately: a 100% stratum standing on 2 instances and one
    standing on 400 look identical otherwise, and only one of them is evidence.
    """
    rows = []
    for (pair, domain), group in by_stratum(results).items():
        mt = score.pool([r.mt_baseline for r in group])
        ape = score.pool([r.post_edited for r in group])

        rows.append({
            "language_pair": pair,
            "domain": domain,
            "datasets": len(group),
            "segments": sum(r.totals.get("segments", 0) for r in group),
            "expected_instances": mt.expected,
            "mt_adherence_rate": mt.adherence_rate,
            "ape_adherence_rate": ape.adherence_rate,
            # `used_everywhere` is the remainder, so it is not shipped as its own column.
            "distinct_terms": mt.terms.distinct_terms,
            "mt_terms_never_used": mt.terms.never_used,
            "ape_terms_never_used": ape.terms.never_used,
            "mt_terms_partly_used": mt.terms.used_partly,
            "ape_terms_partly_used": ape.terms.used_partly,
            "mt_terms_over_used": mt.terms.over_used,
            "ape_terms_over_used": ape.terms.over_used,
            "delta_pct": (None if mt.adherence_rate is None or ape.adherence_rate is None
                         else (ape.adherence_rate - mt.adherence_rate) * 100),
            "mt_violations": mt.misses.substituted,
            "ape_violations": ape.misses.substituted,
            "mt_omissions": mt.misses.omitted,
            "ape_omissions": ape.misses.omitted,
        })
    return rows


def render_strata(results: Sequence[BenchmarkResult]) -> str:
    """Pooled adherence per language pair and domain, when there is something to pool.

    A single dataset is a stratum of one, and every figure in that row is already on its own
    scorecard - the pair, the domain, the instance count and both rates - so it is not printed.
    """
    if len(results) < 2:
        return ""

    rows = stratum_rows(results)
    if not rows:
        return ""

    lines = ["", "  BY STRATUM", ""]
    for row in rows:
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.2f}%"
        lines.append(
            f"  {row['language_pair'][:14]:16}{row['domain'][:20]:22}"
            f"{row['expected_instances']:>5} inst"
            f"  ·  MT {pct(row['mt_adherence_rate'])} → APE {pct(row['ape_adherence_rate'])}"
            f"  ({delta})"
        )

    if len(rows) > 1:
        overall_mt = score.pool([r.mt_baseline for r in results])
        overall_ape = score.pool([r.post_edited for r in results])
        lines.append(
            f"  {'ALL':16}{'':22}{overall_mt.expected:>5} inst"
            f"  ·  MT {pct(overall_mt.adherence_rate)} → APE {pct(overall_ape.adherence_rate)}"
        )

    lines.append("")
    return "\n".join(lines)


def _identity_columns(result: BenchmarkResult) -> dict[str, str]:
    return {
        "dataset": result.dataset,
        "language_pair": f"{result.parameters['source_language']}>{result.parameters['target_language']}",
        "domain": result.parameters.get("domain") or "",
    }


def violation_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    identity = _identity_columns(result)
    rows: list[dict[str, Any]] = []

    for segment in result.segments:
        mt_violated = {v.source_content for v in segment.mt.violations}
        for violation in segment.ape.violations:
            already_wrong = violation.source_content in mt_violated
            rows.append(
                {
                    **identity,
                    "segment_id": segment.source_segment_id,
                    "source_term": violation.source_content,
                    "expected_targets": " | ".join(violation.expected_targets),
                    "strictness": violation.strictness,
                    # Fewer missed than expected is a partial rendering — used somewhere in the
                    # segment but not everywhere — which reads identically to a term that never
                    # appeared unless both counts travel with the row.
                    "missed_occurrences": violation.missed_occurrences,
                    "expected_occurrences": violation.expected_occurrences,
                    # Substitutions are a terminology worklist; omissions belong to whoever owns
                    # the empty segment, and mixing them wastes a reviewer's pass.
                    "miss_kind": violation.miss_kind,
                    "already_wrong_in_mt": "yes" if already_wrong else "no",
                    "introduced_by_ape": "no" if already_wrong else "yes",
                    "source_content": segment.source_content,
                    "post_edited_text": segment.post_edited_text,
                    # Every violation is a term the human did use: the row carries the wording
                    # that expected it, so the miss can be judged without opening the dataset.
                    "reference_text": segment.reference_content,
                }
            )

    return rows


def segment_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    identity = _identity_columns(result)
    return [
        {
            **identity,
            "segment_id": segment.source_segment_id,
            "has_glossary_resolved": segment.has_glossary_resolved,
            "has_glossary_reported": segment.has_glossary_reported,
            "changed_by_ape": segment.changed_by_ape,
            "expected_terms": segment.mt.expected,
            "mt_adherent": segment.mt.adherent,
            "ape_adherent": segment.ape.adherent,
            "mt_violations": segment.mt.misses.substituted,
            "ape_violations": segment.ape.misses.substituted,
            "mt_omissions": segment.mt.misses.omitted,
            "ape_omissions": segment.ape.misses.omitted,
            "mt_adherence_rate": score.rate(segment.mt.adherent, segment.mt.expected),
            "ape_adherence_rate": score.rate(segment.ape.adherent, segment.ape.expected),
            # The lenient reading of the same segment, counted per distinct term rather than per
            # occurrence. `used_everywhere` is the remainder and is not shipped as its own column.
            # `expected_terms` is the reference's own count, so both columns share it.
            "distinct_terms": segment.mt.terms.distinct_terms,
            "mt_terms_never_used": segment.mt.terms.never_used,
            "ape_terms_never_used": segment.ape.terms.never_used,
            "mt_terms_partly_used": segment.mt.terms.used_partly,
            "ape_terms_partly_used": segment.ape.terms.used_partly,
            "mt_terms_over_used": segment.mt.terms.over_used,
            "ape_terms_over_used": segment.ape.terms.over_used,
        }
        for segment in result.segments
    ]
