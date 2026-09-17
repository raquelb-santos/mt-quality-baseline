"""Rendering for the DNT component: leaks and over-keeps get separate lines, never netted."""

from collections import defaultdict
from typing import Any, Sequence

from .dnt_benchmark import Result
from .dnt_score import aggregate, aggregate_reversion
from .report import Scorecard, arrow, cell, delta as delta_of, failure_warning, pct, rate, scope_note, signed_pct, strata, strata_rates, subheading_of, table
from .text_processing import count_surface

VERSIONS = ("mt", "ape", "rev")

ITEM_COLUMNS = ("MT", "APE", "REV", "REF", "Preservation")


def _moved(values: Sequence[Any]) -> str:
    return " → ".join(f"{v.upper()} {value}" for v, value in zip(VERSIONS, values))


def scorecard(result: Result) -> Scorecard:
    mt = result.mt
    versions = (mt, result.ape, result.rev)
    totals, delta = result.totals, result.delta

    warnings = failure_warning(result)
    if mt.segments_unread:
        warnings.append(
            f"{mt.segments_unread}/{totals['segments']} segments came back from no revert batch"
            " and are excluded from every rate below, rather than counted as having nothing to"
            " preserve."
        )

    # Padded to a common width so the three versions read as columns without being a table.
    metrics = [
        ("Preservation", [pct(a.preservation_rate) for a in versions]),
        ("Leaked", [a.leaked for a in versions]),
        ("Over-kept", [a.over_kept for a in versions]),
        ("Segments clean", [pct(a.segment_preservation_rate) for a in versions]),
        ("Kept from SRC", [pct(a.src_retention_rate) for a in versions]),
    ]
    width = max(len(label) for label, _ in metrics)

    facts = [
        f"Segments {totals['segments']} · read {totals['segments_read']}"
        f" · with items {totals['segments_with_items']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · by REV {totals['segments_changed_by_rev']}",
        f"Items {mt.items.distinct_items} scored · {mt.expected} REF instances"
        f" · excluded: not in REF {mt.not_in_ref}"
        f" · not in SRC {mt.not_in_src} · fingerprint {result.fingerprint}",
        f"SRC instances {mt.in_src} · REF kept {pct(result.ref.src_retention_rate)} of them,"
        f" so the rest are items the human chose to translate",
        *(f"{label.ljust(width)}  {_moved(values)}" for label, values in metrics),
        *scope_note(mt.segments_scored, totals["segments_with_items"],
                    "the rest name no item that both SRC and REF carry"),
    ]

    detail = [
        f"Preservation delta APE {signed_pct(delta.ape_preservation_rate)}"
        f" · REV {signed_pct(delta.rev_preservation_rate)}",
        f"Leak kinds · translated {arrow([a.leaks.translated for a in versions])}"
        f" · case drift {arrow([a.leaks.case_drift for a in versions])}",
        f"Item outcomes"
        f" · matched {arrow([a.items.matched_ref for a in versions])}"
        f" · partly {arrow([a.items.kept_partly for a in versions])}"
        f" · never {arrow([a.items.never_kept for a in versions])}"
        f" · over-kept {arrow([a.items.over_kept for a in versions])}",
        f"APE repaired {delta.items_fixed_by_ape} · broke {delta.items_broken_by_ape}"
        f" · REV repaired {delta.items_fixed_by_rev}"
        f" · broke {delta.items_broken_by_rev}",
    ]

    return Scorecard(
        heading="dnt", dataset=result.dataset, subheading=subheading_of(result), facts=facts,
        warnings=warnings, detail=detail,
    )


def item_rows(result: Result) -> list[dict[str, Any]]:
    """One row per distinct DNT item."""
    pooled: dict[str, defaultdict[str, int]] = {}

    for segment in result.segments:
        columns = {c: {i.text: i for i in getattr(segment, c).item_scores} for c in VERSIONS}
        # One column's keys are all of them: an item is either scored in every column or in none.
        for text, mt_score in columns["mt"].items():
            entry = pooled.setdefault(text, defaultdict(int))
            # Once per segment: the count is a SRC property, so per-column adds would treble it.
            entry["in_src"] += mt_score.in_src
            # REF's own count: the denominator, identical for every version scored.
            entry["expected"] += mt_score.expected
            for column, scores in columns.items():
                scored = scores[text]
                entry[f"{column}_preserved"] += scored.preserved
                entry[f"{column}_found"] += scored.found
                entry[f"{column}_over_kept"] += scored.over_kept

    rows = [
        {
            "item": text,
            "in_src": entry["in_src"],
            "ref_kept": entry["expected"],
            "mt_over_kept": entry["mt_over_kept"],
            "rev_over_kept": entry["rev_over_kept"],
            **{f"{c}_found": entry[f"{c}_found"] for c in VERSIONS},
            **{f"{c}_preservation_rate": rate(entry[f"{c}_preserved"], entry["expected"])
               for c in VERSIONS},
        }
        for text, entry in pooled.items()
    ]

    # Worst as delivered first, then worst as the MT had it, so a rescued item still sorts high.
    rows.sort(key=lambda r: (r["rev_preservation_rate"], r["mt_preservation_rate"], r["item"]))
    return rows


def item_cells(result: Result) -> list[tuple[str, list[str]]]:
    """Each item against what every version kept, REF included, and REV's ratio."""
    return [
        (
            row["item"],
            [
                *(str(row[f"{column}_found"]) for column in VERSIONS),
                str(row["ref_kept"]),
                pct(row["rev_preservation_rate"]),
            ],
        )
        for row in item_rows(result)
    ]


def render_items(result: Result) -> str:
    rows = item_cells(result)
    if not rows:
        return "No DNT items were reported.\n"
    return table("Preservation by DNT item (preservation is REV against REF, worst first)",
                 "DNT item", ITEM_COLUMNS, rows)


def render_items_console(result: Result) -> str:
    rows = item_cells(result)
    if not rows:
        return "\n".join(
            [f"Preservation by DNT item — {result.dataset}", "", "  No DNT items were reported.", ""]
        )
    # The counts need no more room than their headers; only the rate column is wide.
    return table(
        f"Preservation by DNT item — {result.dataset}"
        " (preservation is REV against REF, worst first)",
        "DNT item", ITEM_COLUMNS, rows, console=True, width=4,
    )


def detection_rows(result: Result) -> list[dict[str, Any]]:
    """One row per item per segment: the grain the scope gates work at. Empty segments get a row."""
    source_language = result.parameters.get("source_language")
    target_language = result.parameters.get("target_language")

    rows = []
    for segment in result.segments:
        items = [] if segment.unread else list(dict.fromkeys(segment.items))
        if not items:
            rows.append({
                "segment_id": segment.source_segment_id,
                "item": "(no response)" if segment.unread else "(none)",
                "scored": False, "flag": "", "counted": False,
                "in_src": "", "in_ref": "",
                **{f"in_{column}": "" for column in VERSIONS},
                "preserved": 0, "expected": 0,
            })
            continue

        for item in items:
            in_src = count_surface(segment.src_text, item, source_language, casefold=False)
            in_ref = count_surface(segment.ref_text, item, target_language, casefold=False)
            kept = {
                column: count_surface(
                    getattr(segment, f"{column}_text"), item, target_language, casefold=False
                )
                for column in VERSIONS
            }

            # The same gates `score_dnt` applies, so the row and the totals agree.
            flag = "not in SRC" if not in_src else "not in REF" if not in_ref else ""

            rows.append({
                "segment_id": segment.source_segment_id,
                "item": item,
                "scored": not flag,
                "flag": flag,
                "counted": True,
                "in_src": in_src,
                "in_ref": in_ref,
                **{f"in_{column}": value for column, value in kept.items()},
                "expected": 0 if flag else in_ref,
                # The delivered version, so the ratio here is the one the report headlines.
                "preserved": 0 if flag else min(kept["rev"], in_ref),
            })

    return rows


def _preservation(row: dict[str, Any]) -> str:
    """REV against REF, or a dash where REF set no expectation to meet."""
    if not row["scored"]:
        return "-" if row["counted"] else ""
    return pct(rate(row["preserved"], row["expected"]))


def render_detection(result: Result) -> str:
    """What the service named per segment, including the items no column scores."""
    rows = detection_rows(result)
    if not rows:
        return ""

    lines = [
        "#### DNT items detected (from /v1/revert)",
        "",
        "| Segment | DNT item | SRC | MT | APE | REV | REF | Preservation | Flag |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {cell(row['segment_id'])} | {cell(row['item'])} | {row['in_src']}"
            f" | {row['in_mt']} | {row['in_ape']} | {row['in_rev']} | {row['in_ref']}"
            f" | {_preservation(row)} | {row['flag']} |"
        )
    lines.append("")

    return "\n".join(lines)


def render_detection_console(result: Result) -> str:
    rows = detection_rows(result)
    if not rows:
        return ""

    seg_width = max(max(len(str(row["segment_id"])) for row in rows), len("SEG"))
    item_width = max(max(len(row["item"]) for row in rows), len("DNT item"))
    lines = [
        f"DNT items detected — {result.dataset} (from /v1/revert)",
        "",
        f"  {'SEG'.rjust(seg_width)}  {'DNT item'.ljust(item_width)}"
        f"  {'SRC':>4} {'MT':>4} {'APE':>4} {'REV':>4} {'REF':>4}  preservation",
    ]
    for row in rows:
        lines.append(
            f"  {str(row['segment_id']).rjust(seg_width)}  {row['item'].ljust(item_width)}"
            f"  {row['in_src']:>4} {row['in_mt']:>4} {row['in_ape']:>4}"
            f" {row['in_rev']:>4} {row['in_ref']:>4}"
            f"  {_preservation(row):>12}  {row['flag']}".rstrip()
        )
    lines.append("")

    return "\n".join(lines)


def defect_rows(result: Result) -> list[dict[str, Any]]:
    """One row per item a version failed, which is the worklist tracing numbers to segments."""
    rows = []
    for segment in result.segments:
        scores = {version: {i.text: i for i in getattr(segment, version).item_scores}
                  for version in VERSIONS}
        for scored in segment.ref.item_scores:
            faults = []
            for version in VERSIONS:
                against_ref = scores[version][scored.text]
                if against_ref.leaked:
                    faults.append(f"{version.upper()} {against_ref.leak_kind.replace('_', ' ')}")
                elif against_ref.over_kept:
                    faults.append(f"{version.upper()} over-kept")
            if not faults:
                continue

            rows.append({
                "segment_id": segment.source_segment_id,
                "item": scored.text,
                "ref_kept": scored.expected,
                **{f"{version}_found": scores[version][scored.text].found for version in VERSIONS},
                "faults": ", ".join(faults),
            })
    return rows


def render_defects(result: Result) -> str:
    """Every DNT item a version lost or over-kept, and what each version did to it."""
    rows = [
        (
            str(row["segment_id"]),
            [
                cell(row["item"]),
                str(row["ref_kept"]),
                *(str(row[f"{column}_found"]) for column in VERSIONS),
                cell(row["faults"]),
            ],
        )
        for row in defect_rows(result)
    ]
    return table("DNT items that did not survive", "Segment",
                 ("DNT item", "REF", "MT", "APE", "REV", "What went wrong"), rows)


def render_comparison(results: Sequence[Result]) -> str:
    if len(results) < 2:
        return ""

    lines = ["### Across datasets", ""]
    for result in results:
        pair = f"{result.parameters['source_language']}>{result.parameters['target_language']}"
        lines.append(
            f"- {result.dataset} · {pair} · {result.mt.expected} inst"
            f"  ·  MT {pct(result.mt.preservation_rate)}"
            f" → APE {pct(result.ape.preservation_rate)}"
            f" → REV {pct(result.rev.preservation_rate)}"
            f"  ·  over-kept {result.mt.over_kept} → {result.ape.over_kept}"
            f" → {result.rev.over_kept}"
        )
    lines.append("")
    return "\n".join(lines)


def _measured(segments: Sequence[Any]) -> dict[str, Any]:
    """One group's counts from its segments; unread ones leave the denominator, as on the scorecard."""
    unread = sum(1 for s in segments if s.unread)
    scored = [s for s in segments if not s.unread]
    mt, ape, rev = (
        aggregate([getattr(s, column) for s in scored], segments_unread=unread)
        for column in VERSIONS
    )
    return {
        "segments": len(segments),
        "segments_unread": unread,
        "expected_instances": mt.expected,
        "mt_preservation_rate": mt.preservation_rate,
        "ape_preservation_rate": ape.preservation_rate,
        "rev_preservation_rate": rev.preservation_rate,
        "mt_leaked": mt.leaked,
        "mt_over_kept": mt.over_kept,
    }


def stratum_rows(results: Sequence[Result]) -> list[dict[str, Any]]:
    return strata(results, _measured)


def stratum_rate_rows(results: Sequence[Result]) -> list[tuple[str, list[str]]]:
    return strata_rates(results, _measured, "inst", [f"{c}_preservation_rate" for c in VERSIONS])


def render_strata(results: Sequence[Result]) -> str:
    """Preservation per language pair, split by the domains inside it."""
    return table("Preservation by language pair", "Language pair", [v.upper() for v in VERSIONS],
                 stratum_rate_rows(results), heading="###")


def render_strata_console(results: Sequence[Result]) -> str:
    return table("Preservation by language pair", "language pair", [v.upper() for v in VERSIONS],
                 stratum_rate_rows(results), console=True)


def reversion_scorecard(result: Any) -> Scorecard:
    scored, totals = result.scored, result.totals

    warnings = failure_warning(result)
    if scored.segments_unread:
        warnings.append(
            f"{scored.segments_unread}/{totals['segments']} pairs came back from no revert batch"
            " and are excluded from every rate below, rather than counted as having lost"
            " every term."
        )
    if not scored.expected:
        warnings.append(
            "No gold term on any pair scored, so every rate below is over an empty corpus."
        )

    facts = [
        f"Pairs {totals['segments']} · with gold terms {totals['segments_with_terms']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · by REV {totals['segments_changed_by_rev']}"
        f" · by REV on APE {totals['segments_changed_by_rev_ape']}"
        f" · gold terms {scored.expected}",
        f"Terms carried  MT {pct(scored.mt_rate)} → REV {pct(scored.rev_rate)}"
        f" ({signed_pct(delta_of(scored.mt_rate, scored.rev_rate))})"
        f"  ·  APE {pct(scored.ape_rate)} → REV {pct(scored.rev_ape_rate)}"
        f" ({signed_pct(delta_of(scored.ape_rate, scored.rev_ape_rate))})",
        f"Counts  MT {scored.in_mt}/{scored.expected} → REV {scored.in_rev}/{scored.expected}"
        f"  ·  APE {scored.in_ape}/{scored.expected}"
        f" → REV {scored.in_rev_ape}/{scored.expected}",
        f"REV on MT restored {scored.repaired} · broke {scored.broken}"
        f"  ·  on APE restored {scored.repaired_from_ape} · broke {scored.broken_from_ape}",
        f"Pairs fully carried  from MT {pct(scored.segment_rate)}"
        f" ({scored.segments_clean}/{scored.segments_scored})"
        f"  ·  from APE {pct(scored.segment_rate_from_ape)}"
        f" ({scored.segments_clean_from_ape}/{scored.segments_scored})",
        *scope_note(scored.segments_scored, totals["segments_with_terms"],
                    "a pair with no gold term sets no expectation"),
    ]

    return Scorecard(
        heading="dnt reversion", dataset=result.dataset, subheading=subheading_of(result),
        facts=facts, warnings=warnings,
    )


def _arm(carried: bool, broken: bool) -> str:
    return "broken by REV" if broken else ("carried" if carried else "not restored")


def reversion_term_rows(result: Any) -> list[dict[str, Any]]:
    """A term is listed when either arm lost it, so the two arms can be read against each other."""
    rows = []
    for segment in result.segments:
        for term in segment.score.terms:
            if term.in_rev_mt and term.in_rev_ape:
                continue
            rows.append({
                "segment_id": segment.source_segment_id,
                "term": term.text,
                "in_mt": "yes" if term.in_mt else "no",
                "in_rev": "yes" if term.in_rev_mt else "no",
                "in_ape": "yes" if term.in_ape else "no",
                "in_rev_ape": "yes" if term.in_rev_ape else "no",
                "outcome": _arm(term.in_rev_mt, term.broken),
                "outcome_ape": _arm(term.in_rev_ape, term.broken_from_ape),
            })
    return rows


_TERM_COLUMNS = ["In MT", "In REV", "In APE", "In REV(APE)", "From MT", "From APE"]


def _term_cells(row: dict[str, Any]) -> list[str]:
    return [row["in_mt"], row["in_rev"], row["in_ape"], row["in_rev_ape"],
            row["outcome"], row["outcome_ape"]]


def render_reversion_terms(result: Any) -> str:
    rows = reversion_term_rows(result)
    return table(
        f"Gold terms reversion did not carry — {result.dataset}", "Segment",
        ["Term", *_TERM_COLUMNS],
        [(row["segment_id"], [cell(row["term"]), *_term_cells(row)]) for row in rows],
    )


def render_reversion_terms_console(result: Any) -> str:
    rows = reversion_term_rows(result)
    return table(
        f"Gold terms reversion did not carry — {result.dataset}", "segment",
        ["term", *(name.lower() for name in _TERM_COLUMNS)],
        [(str(row["segment_id"]), [row["term"], *_term_cells(row)]) for row in rows[:40]],
        console=True,
    )


def _reversion_measured(segments: Sequence[Any]) -> dict[str, Any]:
    """Aggregated from the segments themselves, so a pair drawn from several files is one row."""
    pooled = aggregate_reversion([s.score for s in segments if not s.unread])
    return {"expected_instances": pooled.expected, "mt_rate": pooled.mt_rate,
            "rev_rate": pooled.rev_rate, "ape_rate": pooled.ape_rate, "rev_ape_rate": pooled.rev_ape_rate}


def reversion_rate_rows(results: Sequence[Any]) -> list[tuple[str, list[str]]]:
    return strata_rates(results, _reversion_measured, "terms",
                        ["mt_rate", "rev_rate", "ape_rate", "rev_ape_rate"])


def render_reversion_strata(results: Sequence[Any]) -> str:
    return table("Gold terms carried by language pair", "Language pair",
                 ["MT", "REV", "APE", "REV(APE)"], reversion_rate_rows(results), heading="###")


def render_reversion_strata_console(results: Sequence[Any]) -> str:
    return table("Gold terms carried by language pair", "language pair",
                 ["MT", "REV", "APE", "REV(APE)"], reversion_rate_rows(results), console=True)


def render_reversion_comparison(results: Sequence[Any]) -> str:
    if len(results) < 2:
        return ""

    lines = ["### Across datasets", ""]
    for result in results:
        lines.append(
            f"- {result.dataset} · {result.parameters['source_language']}"
            f">{result.parameters['target_language']} · {result.scored.expected} terms"
            f"  ·  MT {pct(result.scored.mt_rate)} → REV {pct(result.scored.rev_rate)}"
            f"  ·  APE {pct(result.scored.ape_rate)} → REV {pct(result.scored.rev_ape_rate)}"
        )
    return "\n".join([*lines, ""])
