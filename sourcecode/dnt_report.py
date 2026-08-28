"""Rendering for the DNT component: leaks and over-keeps get separate lines, never netted."""

from typing import Any, Sequence

from . import dnt_score
from .dnt_benchmark import DntResult
from .report import Scorecard, by_stratum, cell, pct, rate, signed_pct

# The versions a run scores, in delivery order.
PIPELINE = (("mt", "MT"), ("ape", "APE"), ("rev", "REV"))

# The per-item worklist counts instead of rating: what each version kept, REF too, and REV's ratio.
ITEM_COLUMNS = ("MT", "APE", "REV", "REF", "Preservation")


def _arrow(values: Sequence[Any]) -> str:
    return " → ".join(str(value) for value in values)


def _moved(values: Sequence[Any]) -> str:
    return " → ".join(f"{label} {value}" for (_, label), value in zip(PIPELINE, values))


def dnt_scorecard(result: DntResult) -> Scorecard:
    """One DNT dataset's headline results, in the form both destinations render from."""
    mt, ape, rev = result.mt, result.ape, result.rev
    versions = (mt, ape, rev)
    totals, delta = result.totals, result.delta

    subheading = f"{result.parameters['source_language']} → {result.parameters['target_language']}"
    if result.parameters.get("domain"):
        subheading += f"  ·  {result.parameters['domain']}"

    warnings = []
    if result.failed_segments:
        warnings.append(
            f"{result.failed_segments}/{totals['segments']} segments failed inside post-mt"
            f"\n{result.failure_reason}"
        )
    if mt.segments_unread:
        warnings.append(
            f"{mt.segments_unread}/{totals['segments']} segments came back from no revert batch"
            " and are excluded from every count below, rather than counted as having nothing to"
            " preserve."
        )

    # Padded to a common width so the three versions read as columns without being a table.
    metrics = [
        ("Preservation", [pct(a.preservation_rate) for a in versions]),
        ("Leaked", [a.leaked for a in versions]),
        ("Over-kept", [a.over_kept for a in versions]),
        ("Segments clean", [pct(a.segment_preservation_rate) for a in versions]),
    ]
    width = max(len(label) for label, _ in metrics)

    facts = [
        f"Segments {totals['segments']} · read {totals['segments_read']}"
        f" · with items {totals['segments_with_items']}"
        f" · changed by APE {totals['segments_changed_by_ape']}"
        f" · by REV {totals['segments_changed_by_rev']}",
        f"Items {mt.distinct_items} scored · {mt.expected} REF instances"
        f" · excluded: not in REF {mt.not_in_ref}"
        f" · not in SRC {mt.not_in_src} · fingerprint {result.fingerprint}",
        *(f"{label.ljust(width)}  {_moved(values)}" for label, values in metrics),
    ]

    if result.usage.cost or result.usage.tokens:
        facts.append(
            f"LLM spend ${result.usage.cost:.4f} · {result.usage.tokens:,} tokens "
            f"({result.usage.prompt_tokens:,} prompt / {result.usage.completion_tokens:,} completion)"
        )

    detail = [
        f"Preservation delta APE {signed_pct(delta.ape_preservation_rate)}"
        f" · REV {signed_pct(delta.rev_preservation_rate)}",
        f"Leak kinds · translated {_arrow([a.leaks.translated for a in versions])}"
        f" · case drift {_arrow([a.leaks.case_drift for a in versions])}",
        f"Item outcomes"
        f" · matched {_arrow([a.items.matched_ref for a in versions])}"
        f" · partly {_arrow([a.items.kept_partly for a in versions])}"
        f" · never {_arrow([a.items.never_kept for a in versions])}"
        f" · over-kept {_arrow([a.items.over_kept for a in versions])}",
        f"APE repaired {delta.items_fixed_by_ape} · broke {delta.items_broken_by_ape}"
        f" · REV repaired {delta.items_fixed_by_rev}"
        f" · broke {delta.items_broken_by_rev}",
    ]

    return Scorecard(
        heading="dnt", subheading=subheading, facts=facts, warnings=warnings, detail=detail
    )


def item_rows(result: DntResult) -> list[dict[str, Any]]:
    """One row per distinct DNT item, with rates recomputed from the pooled counts, not averaged."""
    pooled: dict[str, dict[str, Any]] = {}

    for segment in result.segments:
        columns = {
            "mt": {i.text: i for i in segment.mt.item_scores},
            "ape": {i.text: i for i in segment.ape.item_scores},
            "rev": {i.text: i for i in segment.rev.item_scores},
        }
        # One column's keys are all of them: an item is either scored in every column or in none.
        for text, mt_score in columns["mt"].items():
            entry = pooled.setdefault(text, {
                "in_src": 0,
                **{f"{c}_{f}": 0 for c, _ in PIPELINE
                   for f in ("expected", "preserved", "kept", "over_kept")},
            })
            # Once per segment: the count is a SRC property, so per-column adds would treble it.
            entry["in_src"] += mt_score.in_src
            for column, scores in columns.items():
                scored = scores[text]
                entry[f"{column}_expected"] += scored.expected
                entry[f"{column}_preserved"] += scored.preserved
                entry[f"{column}_kept"] += scored.kept
                entry[f"{column}_over_kept"] += scored.over_kept

    rows = [
        {
            "item": text,
            "in_src": entry["in_src"],
            "ref_kept": entry["mt_expected"],
            "mt_over_kept": entry["mt_over_kept"],
            "rev_over_kept": entry["rev_over_kept"],
            "rev_preserved": entry["rev_preserved"],
            **{f"{column}_kept": entry[f"{column}_kept"] for column, _ in PIPELINE},
            **{
                f"{column}_preservation_rate":
                    rate(entry[f"{column}_preserved"], entry[f"{column}_expected"])
                for column, _ in PIPELINE
            },
        }
        for text, entry in pooled.items()
    ]

    # Worst as delivered first, then worst as the MT had it, so a rescued item still sorts high.
    rows.sort(key=lambda r: (r["rev_preservation_rate"], r["mt_preservation_rate"], r["item"]))
    return rows


def item_cells(result: DntResult) -> list[tuple[str, list[str]]]:
    """Each item against what every version kept, REF included, and REV's ratio."""
    return [
        (
            row["item"],
            [
                str(row["mt_kept"]), str(row["ape_kept"]), str(row["rev_kept"]),
                str(row["ref_kept"]),
                pct(row["rev_preservation_rate"]),
            ],
        )
        for row in item_rows(result)
    ]


def render_dnt_items(result: DntResult) -> str:
    """Every item weighed, worst first; REV alone hides the leaks reversion repaired."""
    rows = item_cells(result)
    if not rows:
        return "No DNT items were reported.\n"

    lines = [
        "### Preservation by DNT item (preservation is REV against REF, worst first)",
        "",
        f"| DNT item | {' | '.join(ITEM_COLUMNS)} |",
        f"| --- | {' | '.join('---:' for _ in ITEM_COLUMNS)} |",
    ]
    lines += [f"| {cell(item)} | {' | '.join(rates)} |" for item, rates in rows]
    lines.append("")

    return "\n".join(lines)


def render_dnt_items_console(result: DntResult) -> str:
    """The same rows, aligned for a terminal."""
    rows = item_cells(result)
    if not rows:
        return "\n".join(
            [f"Preservation by DNT item — {result.dataset}", "", "  No DNT items were reported.", ""]
        )

    width = max(max(len(item) for item, _ in rows), len("DNT item"))
    # The counts need no more room than their headers; only the rate column is wide.
    columns = [(label, max(len(label), 4)) for label in ITEM_COLUMNS]
    lines = [
        f"Preservation by DNT item — {result.dataset}"
        " (preservation is REV against REF, worst first)",
        "",
        f"  {'DNT item'.ljust(width)}  {' '.join(f'{label:>{w}}' for label, w in columns)}",
    ]
    for item, cells in rows:
        lines.append(
            f"  {item.ljust(width)}"
            f"  {' '.join(f'{value:>{w}}' for value, (_, w) in zip(cells, columns))}"
        )
    lines.append("")

    return "\n".join(lines)


def detection_rows(result: DntResult) -> list[dict[str, Any]]:
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
                **{f"in_{column}": "" for column, _ in PIPELINE},
                "preserved": 0, "expected": 0,
            })
            continue

        for item in items:
            in_src = dnt_score.count_item(segment.src_text, item, source_language)
            in_ref = dnt_score.count_item(segment.ref_text, item, target_language)
            kept = {
                column: dnt_score.count_item(text, item, target_language)
                for column, text in (
                    ("mt", segment.mt_text),
                    ("ape", segment.ape_text),
                    ("rev", segment.rev_text),
                )
            }

            # The same gates `score_dnt` applies, so the row and the totals agree.
            if not in_src:
                flag = "not in SRC"
            elif not in_ref:
                flag = "not in REF"
            else:
                flag = ""

            rows.append({
                "segment_id": segment.source_segment_id,
                "item": item,
                "scored": not flag,
                "flag": flag,
                "counted": True,
                "in_src": in_src,
                "in_ref": in_ref,
                **{f"in_{column}": value for column, value in kept.items()},
                "expected": in_ref if not flag else 0,
                # The delivered version, so the ratio here is the one the report headlines.
                "preserved": min(kept["rev"], in_ref) if not flag else 0,
            })

    return rows


def _preservation(row: dict[str, Any]) -> str:
    """REV against REF, or a dash where REF set no expectation to meet."""
    if not row["scored"]:
        return "-" if row["counted"] else ""
    return pct(rate(row["preserved"], row["expected"]))


def render_dnt_detection(result: DntResult) -> str:
    """What the service named per segment, including the items no column scores."""
    rows = detection_rows(result)
    if not rows:
        return ""

    lines = [
        "### DNT items detected (from /v1/revert)",
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


def render_dnt_detection_console(result: DntResult) -> str:
    """The same rows, aligned for a terminal."""
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


def render_dnt_comparison(results: Sequence[DntResult]) -> str:
    if len(results) < 2:
        return ""

    lines = ["## Across datasets", ""]
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


def dnt_stratum_rows(results: Sequence[DntResult]) -> list[dict[str, Any]]:
    """One row per stratum, with the pooled counts its rates were computed from."""
    rows = []
    for (pair, domain), group in by_stratum(results).items():
        mt = dnt_score.pool([r.mt for r in group])
        ape = dnt_score.pool([r.ape for r in group])
        rev = dnt_score.pool([r.rev for r in group])

        rows.append({
            "language_pair": pair,
            "domain": domain,
            "expected_instances": mt.expected,
            "mt_preservation_rate": mt.preservation_rate,
            "ape_preservation_rate": ape.preservation_rate,
            "rev_preservation_rate": rev.preservation_rate,
            "mt_leaked": mt.leaked,
            "mt_over_kept": mt.over_kept,
        })
    return rows


def stratum_rate_rows(results: Sequence[DntResult]) -> list[tuple[str, list[str]]]:
    """Each stratum against its pooled preservation, the instance count in the label."""
    rows = [
        (
            f"{row['language_pair']} · {row['domain']} · {row['expected_instances']} inst",
            [pct(row[f"{column}_preservation_rate"]) for column, _ in PIPELINE],
        )
        for row in dnt_stratum_rows(results)
    ]

    if len(rows) > 1:
        pooled = [dnt_score.pool([getattr(r, attr) for r in results])
                  for attr in ("mt", "ape", "rev")]
        rows.append(
            (f"ALL · {pooled[0].expected} inst", [pct(a.preservation_rate) for a in pooled])
        )

    return rows


def render_dnt_strata(results: Sequence[DntResult]) -> str:
    """Pooled preservation per language pair and domain."""
    rows = stratum_rate_rows(results)
    if not rows:
        return ""

    lines = [
        "## Preservation by stratum",
        "",
        f"| Stratum | {' | '.join(label for _, label in PIPELINE)} |",
        f"| --- | {' | '.join('---:' for _ in PIPELINE)} |",
    ]
    lines += [f"| {cell(label)} | {' | '.join(rates)} |" for label, rates in rows]
    lines.append("")

    return "\n".join(lines)


def render_dnt_strata_console(results: Sequence[DntResult]) -> str:
    """The same rows, aligned for a terminal."""
    rows = stratum_rate_rows(results)
    if not rows:
        return ""

    width = max(max(len(label) for label, _ in rows), len("stratum"))
    lines = [
        "Preservation by stratum",
        "",
        f"  {'stratum'.ljust(width)}  {' '.join(f'{name:>8}' for _, name in PIPELINE)}",
    ]
    for label, rates in rows:
        lines.append(f"  {label.ljust(width)}  {' '.join(f'{value:>8}' for value in rates)}")
    lines.append("")

    return "\n".join(lines)
