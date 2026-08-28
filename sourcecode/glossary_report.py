"""Rendering for the terminology component: one row builder feeds the file and the terminal."""

from typing import Any, Sequence

from . import glossary_score
from .report import Scorecard, by_stratum, cell, pct, rate, signed_pct
from .glossary_benchmark import BenchmarkResult


def glossary_scorecard(result: BenchmarkResult) -> Scorecard:
    """One terminology dataset's headline results, in the form both destinations render from."""
    mt, ape = result.mt, result.ape
    totals, delta = result.totals, result.delta
    mt_v, ape_v = result.mt_violations, result.ape_violations

    # What was measured and the stratum it was measured in - the rest is on the tables below.
    subheading = f"{result.parameters['source_language']} → {result.parameters['target_language']}"
    if result.parameters.get("domain"):
        subheading += f"  ·  {result.parameters['domain']}"

    def moved(before: Any, after: Any) -> str:
        return f"MT {before} → APE {after}"

    warnings = []
    if result.failed_segments:
        warnings.append(
            f"{result.failed_segments}/{totals['segments']} segments failed inside post-mt"
            f"\n{result.failure_reason}"
        )
    never_shown = totals.get("segments_glossary_never_shown", 0)
    if never_shown:
        warnings.append(
            f"post-mt was shown no glossary on {never_shown}/{totals['segments_with_glossary']}"
            " segments where terms were resolved, so the APE column is not a measurement of"
            " terminology adherence. Check cat_project_id, cat_tool_provider and ecosystem_id."
        )
    if not result.config.get("lemma_matching"):
        warnings.append("Lemma matching disabled")

    facts = [
        f"Segments {totals['segments']} · with glossary terms {totals['segments_with_glossary']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · REF instances {mt.expected}",
        f"Adherence {moved(*(pct(a.adherence_rate) for a in (mt, ape)))}"
        f" ({signed_pct(delta.adherence_rate)})",
        f"Violations {moved(*(a.violations for a in (mt, ape)))}",
        f"Strict {moved(*(pct(a.strict.adherence_rate) for a in (mt, ape)))}"
        f" ({mt.strict.expected} inst.)"
        f" · Permissive {moved(*(pct(a.permissive.adherence_rate) for a in (mt, ape)))}"
        f" ({mt.permissive.expected} inst.)",
        f"Segment-level {moved(*(pct(a.segment_adherence_rate) for a in (mt, ape)))}",
        f"Corpus violations {moved(*(v.total for v in (mt_v, ape_v)))}"
        f" · segments affected {moved(*(pct(v.violation_rate) for v in (mt_v, ape_v)))}",
        f"Violation kinds · miss {moved(*(v.miss for v in (mt_v, ape_v)))}"
        f" · inconsistency {moved(*(v.inconsistency for v in (mt_v, ape_v)))}"
        f" · over-application {moved(*(v.over_application for v in (mt_v, ape_v)))}",
        f"Terms {mt.terms.distinct_terms} distinct"
        f" · matched {mt.terms.used_everywhere} → {ape.terms.used_everywhere}"
        f" · partly {mt.terms.used_partly} → {ape.terms.used_partly}"
        f" · never {mt.terms.never_used} → {ape.terms.never_used}"
        f" · over-used {mt.terms.over_used} → {ape.terms.over_used}",
        f"APE repaired {delta.terms_fixed_by_ape} · broke {delta.terms_regressed_by_ape}",
    ]

    # Over-use is adherent by the cap, so say so: it is a worklist item, not a miss.
    if mt.terms.over_used or ape.terms.over_used:
        facts.append("over-used terms are flagged for review, never counted as violations")

    if result.usage.cost or result.usage.tokens:
        facts.append(
            f"LLM spend ${result.usage.cost:.4f} · {result.usage.tokens:,} tokens "
            f"({result.usage.prompt_tokens:,} prompt / {result.usage.completion_tokens:,} completion)"
        )

    return Scorecard(heading="glossary", subheading=subheading, facts=facts, warnings=warnings)


def _bucket(rendered: int, expected: int) -> str:
    """The scorecard's four buckets, decided per row on the pooled counts."""
    if rendered == 0:
        # No denominator either: REF never used the term, so there is nothing to bucket.
        return "never" if expected else ""
    if rendered < expected:
        return "partly"
    return "matched" if rendered == expected else "over-used"


def term_rows(result: BenchmarkResult) -> list[dict[str, Any]]:
    """One row per distinct term, with rates recomputed from the pooled counts, not averaged."""
    pooled: dict[str, dict[str, Any]] = {}

    for segment in result.segments:
        columns = {
            "mt": {t.source_content: t for t in segment.mt.term_scores},
            "ape": {t.source_content: t for t in segment.ape.term_scores},
        }
        # The union, not MT alone: keying off one column would drop the other's over-use.
        for source in dict.fromkeys([*columns["mt"], *columns["ape"]]):
            term = columns["mt"].get(source) or columns["ape"][source]
            entry = pooled.setdefault(source, {
                "expected_targets": " | ".join(term.expected_targets),
                "strictness": term.strictness,
                "segments": 0,
                **{f"{c}_{f}": 0 for c in ("mt", "ape")
                   for f in ("expected", "adherent", "rendered", "violations")},
            })
            entry["segments"] += 1
            for column, scores in columns.items():
                scored = scores.get(source)
                if scored is not None:
                    entry[f"{column}_expected"] += scored.expected
                    entry[f"{column}_adherent"] += scored.adherent
                    entry[f"{column}_rendered"] += scored.rendered
                    entry[f"{column}_violations"] += scored.violations

    rows = []
    for source, entry in pooled.items():
        expected = entry["ape_expected"]
        rows.append({
            "source_term": source,
            "expected_targets": entry["expected_targets"],
            "strictness": entry["strictness"],
            "segments": entry["segments"],
            "ref_rendered": entry["mt_expected"],
            "mt_rendered": entry["mt_rendered"],
            "mt_adherent": entry["mt_adherent"] if entry["mt_expected"] else "",
            # A zero denominator is a term REF never used, so the row reads as a review item.
            "ape_adherent": entry["ape_adherent"] if expected else "",
            "ape_rendered": entry["ape_rendered"],
            # Against the uncapped count, so the term the cap folded away is still reviewable.
            "ape_bucket": _bucket(entry["ape_rendered"], expected),
            "ape_violations": entry["ape_violations"] if expected else "",
            "ape_adherence_rate": rate(entry["ape_adherent"], expected),
        })

    # Worst first, so the head of the file is the worklist.
    rows.sort(key=lambda r: (-(r["ape_violations"] or 0), r["source_term"]))
    return rows


def render_term_adherence(result: BenchmarkResult) -> str:
    """Every term matched, worst first. Not a top-N and never truncated."""
    rows = term_rows(result)
    if not rows:
        return "No glossary terms matched.\n"

    lines = [
        "### Per-term adherence (adherence is APE against REF, worst first)",
        "",
        "| Source term | Targets | MT | APE | REF | Violations | Adherence | Bucket | Kind |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {cell(row['source_term'])} | {cell(row['expected_targets'])}"
            f" | {row['mt_rendered']} | {row['ape_rendered']}"
            f" | {row['ref_rendered']} | {row['ape_violations']}"
            f" | {pct(row['ape_adherence_rate'])} | {row['ape_bucket']}"
            f" | {row['strictness']} |"
        )
    lines.append("")

    return "\n".join(lines)


def render_term_adherence_console(result: BenchmarkResult) -> str:
    """The same rows, narrowed to the columns the rate is made of."""
    rows = term_rows(result)
    if not rows:
        # Silence here would leave a scorecard of zeroes looking like a clean result.
        return "\n".join(
            [f"Per-term adherence — {result.dataset}", "", "  No glossary terms matched.", ""]
        )

    width = max(max(len(row["source_term"]) for row in rows), len("term"))
    lines = [
        f"Per-term adherence — {result.dataset} (adherence is APE against REF, worst first)",
        "",
        f"  {'term'.ljust(width)}  {'MT':>4} {'APE':>4} {'REF':>4}"
        f" {'violations':>10}  adherence",
    ]
    for row in rows:
        lines.append(
            f"  {row['source_term'].ljust(width)}"
            f"  {row['mt_rendered']:>4} {row['ape_rendered']:>4}"
            f" {row['ref_rendered']:>4} {str(row['ape_violations']):>10}"
            f"  {pct(row['ape_adherence_rate'])}"
        )
    lines.append("")

    return "\n".join(lines)


def render_comparison(results: Sequence[BenchmarkResult]) -> str:
    if len(results) < 2:
        return ""

    lines = ["## Across datasets", ""]
    for result in results:
        pair = f"{result.parameters['source_language']}>{result.parameters['target_language']}"
        lines.append(
            f"- {result.dataset} · {pair} · {result.mt.expected} inst"
            f"  ·  MT {pct(result.mt.adherence_rate)}"
            f" → APE {pct(result.ape.adherence_rate)}"
            f"  ({signed_pct(result.delta.adherence_rate)})"
        )
    lines.append("")
    return "\n".join(lines)


def stratum_rows(results: Sequence[BenchmarkResult]) -> list[dict[str, Any]]:
    """One row per stratum, with the pooled counts its rates were computed from."""
    rows = []
    for (pair, domain), group in by_stratum(results).items():
        mt = glossary_score.pool([r.mt for r in group])
        ape = glossary_score.pool([r.ape for r in group])
        mt_v = glossary_score.pool_violations([r.mt_violations for r in group])
        ape_v = glossary_score.pool_violations([r.ape_violations for r in group])

        rows.append({
            "language_pair": pair,
            "domain": domain,
            "datasets": len(group),
            "segments": sum(r.totals.get("segments", 0) for r in group),
            "expected_instances": mt.expected,
            "mt_adherence_rate": mt.adherence_rate,
            "ape_adherence_rate": ape.adherence_rate,
            "delta_pct": (None if mt.adherence_rate is None or ape.adherence_rate is None
                          else (ape.adherence_rate - mt.adherence_rate) * 100),
            "mt_violations": mt_v.total,
            "ape_violations": ape_v.total,
            "mt_violation_rate": mt_v.violation_rate,
            "ape_violation_rate": ape_v.violation_rate,
        })
    return rows


def _stratum_lines(results: Sequence[BenchmarkResult]) -> list[str]:
    """Built once so the report's bullets and the console's list cannot disagree."""
    lines = []
    for row in stratum_rows(results):
        delta = "n/a" if row["delta_pct"] is None else f"{row['delta_pct']:+.2f}%"
        lines.append(
            f"{row['language_pair']} · {row['domain']} · {row['expected_instances']} inst"
            f"  ·  MT {pct(row['mt_adherence_rate'])} → APE {pct(row['ape_adherence_rate'])}"
            f"  ({delta})  ·  violations {row['mt_violations']} → {row['ape_violations']}"
            f" · segments affected {pct(row['mt_violation_rate'])}"
            f" → {pct(row['ape_violation_rate'])}"
        )

    if len(lines) > 1:
        mt = glossary_score.pool([r.mt for r in results])
        ape = glossary_score.pool([r.ape for r in results])
        mt_v = glossary_score.pool_violations([r.mt_violations for r in results])
        ape_v = glossary_score.pool_violations([r.ape_violations for r in results])
        lines.append(
            f"ALL · {mt.expected} inst"
            f"  ·  MT {pct(mt.adherence_rate)} → APE {pct(ape.adherence_rate)}"
            f"  ·  violations {mt_v.total} → {ape_v.total}"
            f" · segments affected {pct(mt_v.violation_rate)} → {pct(ape_v.violation_rate)}"
        )

    return lines


def render_strata(results: Sequence[BenchmarkResult]) -> str:
    """Pooled adherence per language pair and domain."""
    lines = _stratum_lines(results)
    if not lines:
        return ""
    return "\n".join(["## By stratum", "", *(f"- {line}" for line in lines), ""])


def render_strata_console(results: Sequence[BenchmarkResult]) -> str:
    """The same lines, indented for a terminal."""
    lines = _stratum_lines(results)
    if not lines:
        return ""
    return "\n".join(["Adherence by stratum", "", *(f"  {line}" for line in lines), ""])
