"""Rendering for the terminology component: one row builder feeds the file and the terminal."""

from typing import Any, Sequence

from .report import Scorecard, by_stratum, cell, delta as delta_of, failure_warning, pct, rate, scope_note, signed_pct, spend_fact, subheading_of, table
from .glossary_benchmark import Result
from .glossary_score import bucket_of, pool, pool_violations

VERSIONS = ("mt", "ape")


def _moved(values: Sequence[Any]) -> str:
    return " → ".join(f"{v.upper()} {value}" for v, value in zip(VERSIONS, values))


def scorecard(result: Result) -> Scorecard:
    mt, ape = result.mt, result.ape
    totals, delta = result.totals, result.delta
    mt_v, ape_v = result.mt_violations, result.ape_violations
    check = result.ref_check

    warnings = failure_warning(result)
    never_shown = totals["segments_glossary_never_shown"]
    if never_shown:
        warnings.append(
            f"post-mt was shown no glossary on {never_shown}/{totals['segments_with_glossary']}"
            " segments where terms were resolved, so the APE column is not a measurement of"
            " terminology adherence. Check cat_project_id, cat_tool_provider and ecosystem_id."
        )
    if not result.config["lemma_matching"]:
        warnings.append("Lemma matching disabled")

    facts = [
        f"Segments {totals['segments']} · with glossary terms {totals['segments_with_glossary']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · REF instances {mt.expected}",
        f"Adherence {_moved([pct(a.adherence_rate) for a in (mt, ape)])}"
        f" ({signed_pct(delta.ape_adherence_rate)})",
        f"Violations {_moved([a.violations for a in (mt, ape)])}",
        f"Strict {_moved([pct(a.strict.adherence_rate) for a in (mt, ape)])}"
        f" ({mt.strict.expected} inst.)"
        f" · Permissive {_moved([pct(a.permissive.adherence_rate) for a in (mt, ape)])}"
        f" ({mt.permissive.expected} inst.)",
        f"Segment-level {_moved([pct(a.segment_adherence_rate) for a in (mt, ape)])}",
        f"Corpus violations {_moved([v.total for v in (mt_v, ape_v)])}"
        f" · segments affected {_moved([pct(v.violation_rate) for v in (mt_v, ape_v)])}",
        f"Violation kinds · miss {_moved([v.miss for v in (mt_v, ape_v)])}"
        f" · inconsistency {_moved([v.inconsistency for v in (mt_v, ape_v)])}"
        f" · over-application {_moved([v.over_application for v in (mt_v, ape_v)])}",
        f"Reference check · REF rendered {check.terms_rendered}/{check.terms_checked}"
        " glossary terms"
        + (f" · {check.to_review} to review in {check.segments_to_review} segments"
           if check.items else ""),
        f"Terms {mt.terms.distinct_terms} distinct"
        f" · matched {mt.terms.matched_ref} → {ape.terms.matched_ref}"
        f" · partly {mt.terms.used_partly} → {ape.terms.used_partly}"
        f" · never {mt.terms.never_used} → {ape.terms.never_used}"
        f" · over-used {mt.terms.over_used} → {ape.terms.over_used}",
        f"APE repaired {delta.terms_fixed_by_ape} · broke {delta.terms_broken_by_ape}",
        *scope_note(mt.segments_scored, totals["segments_with_glossary"],
                    "REF rendered no glossary term in the rest"),
    ]

    # Over-use is adherent by the cap, so say so: it is a worklist item, not a miss.
    if mt.terms.over_used or ape.terms.over_used:
        facts.append("over-used terms are flagged for review, never counted as violations")

    facts += spend_fact(result.usage)

    return Scorecard(heading="glossary", subheading=subheading_of(result), facts=facts, warnings=warnings)


def _bucket(rendered: int, expected: int) -> str:
    """The scorer's ladder, relabelled - it runs on the pooled counts, not the per-segment ones."""
    return {"never_used": "never", "used_partly": "partly", "matched_ref": "matched",
            "over_used": "over-used", "": ""}[bucket_of(rendered, expected)]


def term_rows(result: Result) -> list[dict[str, Any]]:
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
                # REF's own count: the denominator, identical for every version scored.
                "expected": 0,
                **{f"{c}_{f}": 0 for c in ("mt", "ape")
                   for f in ("adherent", "rendered", "violations")},
            })
            entry["segments"] += 1
            entry["expected"] += term.expected
            for column, scores in columns.items():
                scored = scores.get(source)
                if scored is not None:
                    entry[f"{column}_adherent"] += scored.adherent
                    entry[f"{column}_rendered"] += scored.rendered
                    entry[f"{column}_violations"] += scored.violations

    rows = []
    for source, entry in pooled.items():
        expected = entry["expected"]
        rows.append({
            "source_term": source,
            "expected_targets": entry["expected_targets"],
            # The entry as the glossary states it, so a row is readable without the two columns.
            "entry": f"{source} → {entry['expected_targets']}",
            "strictness": entry["strictness"],
            "segments": entry["segments"],
            "ref_rendered": expected,
            "mt_rendered": entry["mt_rendered"],
            # A zero denominator is a term REF never used, so the row reads as a review item.
            "mt_adherent": entry["mt_adherent"] if expected else "",
            "ape_adherent": entry["ape_adherent"] if expected else "",
            "ape_rendered": entry["ape_rendered"],
            "mt_violations": entry["mt_violations"] if expected else "",
            # Against the uncapped count, so the term the cap folded away is still reviewable.
            "ape_bucket": _bucket(entry["ape_rendered"], expected),
            "ape_violations": entry["ape_violations"] if expected else "",
            "ape_adherence_rate": rate(entry["ape_adherent"], expected),
        })

    # Worst first, so the head of the file is the worklist.
    rows.sort(key=lambda r: (-(r["ape_violations"] or 0), r["source_term"]))
    return rows


def render_terms(result: Result) -> str:
    """Every term matched, worst first. Not a top-N and never truncated."""
    rows = [
        (
            row["entry"],
            [str(row["mt_rendered"]), str(row["ape_rendered"]), str(row["ref_rendered"]),
             str(row["mt_violations"]), str(row["ape_violations"]),
             pct(row["ape_adherence_rate"]), row["ape_bucket"], row["strictness"]],
        )
        for row in term_rows(result)
    ]
    if not rows:
        return "No glossary terms matched.\n"
    return table(
        "Per-term adherence (adherence is APE against REF, worst first)", "Glossary entry",
        ("MT", "APE", "REF", "Violations MT", "Violations APE", "Adherence", "Bucket", "Kind"),
        rows,
    )


def render_terms_console(result: Result) -> str:
    """The same rows, narrowed to the columns the rate is made of."""
    rows = [
        (
            row["entry"],
            [str(row["mt_rendered"]), str(row["ape_rendered"]), str(row["ref_rendered"]),
             str(row["ape_violations"]), pct(row["ape_adherence_rate"])],
        )
        for row in term_rows(result)
    ]
    if not rows:
        # Silence here would leave a scorecard of zeroes looking like a clean result.
        return "\n".join(
            [f"Per-term adherence — {result.dataset}", "", "  No glossary terms matched.", ""]
        )
    # The counts need no more room than their headers; only the rate column is wide.
    return table(
        f"Per-term adherence — {result.dataset} (adherence is APE against REF, worst first)",
        "entry", ("MT", "APE", "REF", "violations", "adherence"), rows, console=True, width=4,
    )


def render_comparison(results: Sequence[Result]) -> str:
    if len(results) < 2:
        return ""

    lines = ["## Across datasets", ""]
    for result in results:
        lines.append(
            f"- {result.dataset} · {result.parameters['source_language']}"
            f">{result.parameters['target_language']} · {result.mt.expected} inst"
            f"  ·  MT {pct(result.mt.adherence_rate)}"
            f" → APE {pct(result.ape.adherence_rate)}"
            f"  ({signed_pct(result.delta.ape_adherence_rate)})"
        )
    return "\n".join([*lines, ""])


def _pooled(group: Sequence[Result]) -> tuple[Any, Any, Any, Any]:
    return (
        pool([r.mt for r in group]),
        pool([r.ape for r in group]),
        pool_violations([r.mt_violations for r in group]),
        pool_violations([r.ape_violations for r in group]),
    )


def stratum_rows(results: Sequence[Result]) -> list[dict[str, Any]]:
    """One row per stratum, with the pooled counts its rates were computed from."""
    rows = []
    for (pair, domain), group in by_stratum(results).items():
        mt, ape, mt_v, ape_v = _pooled(group)
        rows.append({
            "language_pair": pair,
            "domain": domain,
            "datasets": len(group),
            "segments": sum(r.totals["segments"] for r in group),
            "expected_instances": mt.expected,
            "mt_adherence_rate": mt.adherence_rate,
            "ape_adherence_rate": ape.adherence_rate,
            "delta_rate": delta_of(mt.adherence_rate, ape.adherence_rate),
            "mt_violations": mt_v.total,
            "ape_violations": ape_v.total,
            "mt_violation_rate": mt_v.violation_rate,
            "ape_violation_rate": ape_v.violation_rate,
        })
    return rows


def _stratum_lines(results: Sequence[Result]) -> list[str]:
    """Built once so the report's bullets and the console's list cannot disagree."""
    lines = []
    for row in stratum_rows(results):
        lines.append(
            f"{row['language_pair']} · {row['domain']} · {row['expected_instances']} inst"
            f"  ·  MT {pct(row['mt_adherence_rate'])} → APE {pct(row['ape_adherence_rate'])}"
            f"  ({signed_pct(row['delta_rate'])})  ·  violations {row['mt_violations']} → {row['ape_violations']}"
            f" · segments affected {pct(row['mt_violation_rate'])}"
            f" → {pct(row['ape_violation_rate'])}"
        )

    if len(lines) > 1:
        mt, ape, mt_v, ape_v = _pooled(results)
        lines.append(
            f"ALL · {mt.expected} inst"
            f"  ·  MT {pct(mt.adherence_rate)} → APE {pct(ape.adherence_rate)}"
            f"  ·  violations {mt_v.total} → {ape_v.total}"
            f" · segments affected {pct(mt_v.violation_rate)} → {pct(ape_v.violation_rate)}"
        )

    return lines


def render_strata(results: Sequence[Result]) -> str:
    """Pooled adherence per language pair and domain."""
    lines = _stratum_lines(results)
    return "" if not lines else "\n".join(["## By stratum", "", *(f"- {l}" for l in lines), ""])


def render_strata_console(results: Sequence[Result]) -> str:
    lines = _stratum_lines(results)
    return "" if not lines else "\n".join(["Adherence by stratum", "", *(f"  {l}" for l in lines), ""])
