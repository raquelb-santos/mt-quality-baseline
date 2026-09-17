"""Rendering for the tags component: drops, duplications and hallucinations stay separate lines."""

from collections import defaultdict
from typing import Any, Sequence

from .report import Scorecard, arrow, cell, failure_warning, pct, rate, signed_pct, strata, strata_rates, subheading_of, table
from .tags import KINDS
from .tags_benchmark import Result
from .tags_score import Score, aggregate

# The versions a run scores, in delivery order.
VERSIONS = ("mt", "ape", "ref")

TAG_COLUMNS = ("MT", "APE", "REF", "SRC", "Integrity")


def _moved(values: Sequence[Any]) -> str:
    return " → ".join(f"{v.upper()} {value}" for v, value in zip(VERSIONS, values))


def scorecard(result: Result) -> Scorecard:
    mt = result.mt
    versions = (mt, result.ape, result.ref)
    totals, delta = result.totals, result.delta

    warnings = failure_warning(result)
    if not totals["segments_with_tags"]:
        warnings.append(
            "No segment carries a tag, so every rate below is over an empty corpus. Check that "
            "the dataset kept its markup rather than reading a tag-stripped export."
        )
    if mt.segments_source_unpaired:
        warnings.append(
            f"{mt.segments_source_unpaired} segments arrive with an unpaired tag already in SRC,"
            " so those ids are left out of the unpaired count."
        )

    # Padded to a common width so the three versions read as columns without being a table.
    metrics = [
        ("Integrity", [pct(a.integrity_rate) for a in versions]),
        ("Dropped", [a.errors.dropped for a in versions]),
        ("Duplicated", [a.errors.duplicated for a in versions]),
        ("Hallucinated", [a.errors.hallucinated for a in versions]),
        ("Segments clean", [pct(a.segment_integrity_rate) for a in versions]),
    ]
    width = max(len(label) for label, _ in metrics)

    facts = [
        f"Segments {totals['segments']} · with tags {totals['segments_with_tags']}"
        f" · changed by APE {totals['segments_changed_by_ape']}",
        f"Tags {mt.tags.distinct_tags} scored · {mt.expected} SRC instances"
        f" · families {', '.join(sorted({r['family'] for r in family_rows(result)})) or 'none'}",
        *(f"{label.ljust(width)}  {_moved(values)}" for label, values in metrics),
    ]

    detail = [
        f"Integrity delta APE {signed_pct(delta.ape_integrity_rate)}",
        f"Well-formedness · segments unpaired {arrow([a.segments_unpaired for a in versions])}"
        f" · mis-ordered {arrow([a.segments_mis_ordered for a in versions])}",
        f"Tag outcomes"
        f" · matched {arrow([a.tags.matched_source for a in versions])}"
        f" · partly {arrow([a.tags.carried_partly for a in versions])}"
        f" · never {arrow([a.tags.never_carried for a in versions])}"
        f" · duplicated {arrow([a.tags.over_carried for a in versions])}",
        f"APE repaired {delta.tags_fixed_by_ape} · broke {delta.tags_broken_by_ape}",
    ]

    return Scorecard(
        heading="tags", dataset=result.dataset, subheading=subheading_of(result), facts=facts,
        warnings=warnings, detail=detail,
    )


def family_rows(result: Result) -> list[dict[str, Any]]:
    """One row per tag family."""
    pooled: dict[str, defaultdict[str, int]] = {}

    for segment in result.segments:
        columns = {c: {t.text: t for t in getattr(segment, c).tag_scores} for c in VERSIONS}
        for text, mt_score in columns["mt"].items():
            if not mt_score.expected:
                continue
            entry = pooled.setdefault(mt_score.kind, defaultdict(int))
            # Once per tag: the count is a SRC property, so per-column adds would treble it.
            entry["expected"] += mt_score.expected
            for column, scores in columns.items():
                entry[f"{column}_present"] += scores[text].present

    rows = [
        {
            "family": family,
            "expected": entry["expected"],
            **{f"{c}_integrity_rate": rate(entry[f"{c}_present"], entry["expected"])
               for c in VERSIONS},
        }
        for family, entry in pooled.items()
    ]

    rows.sort(key=lambda r: KINDS.index(r["family"]))
    return rows


def family_cells(result: Result) -> list[tuple[str, list[str]]]:
    return [
        (
            f"{row['family']} · {row['expected']} inst",
            [pct(row[f"{column}_integrity_rate"]) for column in VERSIONS],
        )
        for row in family_rows(result)
    ]


def render_families(result: Result) -> str:
    return table("Integrity by tag family", "Family", [v.upper() for v in VERSIONS], family_cells(result)) or "No tags were found in the source segments.\n"


def render_families_console(result: Result) -> str:
    title = f"Integrity by tag family — {result.dataset}"
    return table(title, "family", [v.upper() for v in VERSIONS], family_cells(result),
                 console=True) or f"{title}\n\n  No tags were found in the source segments.\n"


def tag_rows(result: Result) -> list[dict[str, Any]]:
    """One row per distinct tag."""
    pooled: dict[str, defaultdict[str, int]] = {}

    for segment in result.segments:
        columns = {c: {t.text: t for t in getattr(segment, c).tag_scores} for c in VERSIONS}
        # One column's keys are not all of them: a hallucination shows up in its column alone.
        for text in {t for scores in columns.values() for t in scores}:
            entry = pooled.setdefault(text, defaultdict(int))
            for column, scores in columns.items():
                scored = scores.get(text)
                if scored is None:
                    continue
                entry[f"{column}_present"] += scored.present
                entry[f"{column}_found"] += scored.found
                # SRC's own count, read off MT, whose scores cover every SRC tag.
                if column == "mt":
                    entry["expected"] += scored.expected

    rows = [
        {
            "tag": text,
            "expected": entry["expected"],
            **{f"{c}_found": entry[f"{c}_found"] for c in VERSIONS},
            **{f"{c}_integrity_rate": rate(entry[f"{c}_present"], entry["expected"])
               for c in VERSIONS},
        }
        for text, entry in pooled.items()
    ]

    # Worst as delivered first, then worst as the MT had it, so a rescued tag still sorts high.
    rows.sort(key=lambda r: (
        r["ape_integrity_rate"] is not None, r["ape_integrity_rate"] or 0.0,
        r["mt_integrity_rate"] or 0.0, r["tag"],
    ))
    return rows


def tag_cells(result: Result) -> list[tuple[str, list[str]]]:
    """Each tag against what every version carried, SRC included, and APE's ratio."""
    return [
        (
            row["tag"],
            [
                *(str(row[f"{column}_found"]) for column in VERSIONS),
                str(row["expected"]),
                pct(row["ape_integrity_rate"]),
            ],
        )
        for row in tag_rows(result)
    ]


def render_tags(result: Result) -> str:
    return table("Integrity by tag (integrity is APE against SRC, worst first)", "Tag",
                 TAG_COLUMNS, tag_cells(result)) or "No tags were found in any version.\n"


def render_tags_console(result: Result) -> str:
    # Silence on no rows would leave a scorecard of zeroes looking like a clean result.
    # The counts need no more room than their headers; only the rate column is wide.
    return table(
        f"Integrity by tag — {result.dataset} (integrity is APE against SRC, worst first)",
        "tag", TAG_COLUMNS, tag_cells(result), console=True, width=4,
    ) or f"Integrity by tag — {result.dataset}\n\n  No tags were found in any version.\n"


def _defects(score: Score) -> str:
    return ", ".join(
        f"{count} {label}"
        for label, count in (
            ("dropped", score.errors.dropped),
            ("duplicated", score.errors.duplicated),
            ("hallucinated", score.errors.hallucinated),
            ("unpaired", score.unpaired),
            ("mis-ordered", int(score.mis_ordered)),
        )
        if count
    )


def defect_rows(result: Result) -> list[dict[str, Any]]:
    """One row per segment any version broke, which is the worklist tracing numbers to segments."""
    rows = []
    for segment in result.segments:
        named = {column: _defects(getattr(segment, column)) for column in VERSIONS}
        if not any(named.values()):
            continue

        rows.append({
            "segment_id": segment.source_segment_id,
            "tags": " ".join(segment.tags) or "(none)",
            **{f"{column}_defects": text for column, text in named.items()},
        })
    return rows


def render_defects(result: Result) -> str:
    """Every segment whose tags came through wrong, and what went wrong in each version."""
    rows = defect_rows(result)
    if not rows:
        return ""

    lines = [
        "#### Segments with a tag defect",
        "",
        "| Segment | SRC tags | MT | APE | REF |",
        "| --- | --- | --- | --- | --- |",
    ]
    lines += [
        f"| {cell(row['segment_id'])} | {cell(row['tags'])} | {cell(row['mt_defects'])}"
        f" | {cell(row['ape_defects'])} | {cell(row['ref_defects'])} |"
        for row in rows
    ]
    lines.append("")

    return "\n".join(lines)


def render_comparison(results: Sequence[Result]) -> str:
    if len(results) < 2:
        return ""

    lines = ["### Across datasets", ""]
    for result in results:
        pair = f"{result.parameters['source_language']}>{result.parameters['target_language']}"
        lines.append(
            f"- {result.dataset} · {pair} · {result.mt.expected} inst"
            f"  ·  MT {pct(result.mt.integrity_rate)}"
            f" → APE {pct(result.ape.integrity_rate)}"
            f"  ·  dropped {result.mt.errors.dropped} → {result.ape.errors.dropped}"
        )
    lines.append("")
    return "\n".join(lines)


def _measured(segments: Sequence[Any]) -> dict[str, Any]:
    """One group's counts, aggregated from the segments themselves rather than from datasets."""
    mt, ape, ref = (aggregate([getattr(s, column) for s in segments]) for column in VERSIONS)
    return {
        "segments": len(segments),
        "expected_instances": mt.expected,
        "mt_integrity_rate": mt.integrity_rate,
        "ape_integrity_rate": ape.integrity_rate,
        "ref_integrity_rate": ref.integrity_rate,
    }


def stratum_rows(results: Sequence[Result]) -> list[dict[str, Any]]:
    return strata(results, _measured)


def stratum_rate_rows(results: Sequence[Result]) -> list[tuple[str, list[str]]]:
    return strata_rates(results, _measured, "inst", [f"{c}_integrity_rate" for c in VERSIONS])


def render_strata(results: Sequence[Result]) -> str:
    """Integrity per language pair, split by the domains inside it."""
    return table("Integrity by language pair", "Language pair", [v.upper() for v in VERSIONS],
                 stratum_rate_rows(results), heading="###")


def render_strata_console(results: Sequence[Result]) -> str:
    return table("Integrity by language pair", "language pair", [v.upper() for v in VERSIONS],
                 stratum_rate_rows(results), console=True)
