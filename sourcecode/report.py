"""Report primitives every component formats its numbers with, and the file a run writes."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Per component: the heading it is filed under, and what renders its parts.
Sections = Mapping[str, tuple[str, Callable[[Sequence[Any]], list[str]]]]


def rate(numerator: int, denominator: int) -> float | None:
    """None, never 0, with no denominator: a misconfigured run is not total failure."""
    return None if denominator == 0 else numerator / denominator


def delta(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def signed_pct(value: float | None) -> str:
    """A difference of two rates, so percentage points, and no sign on no change."""
    if value is None:
        return "n/a"
    points = round(value * 100, 2)
    return f"{points:+.2f} pp" if points else "0.00 pp"


def arrow(values: Sequence[Any]) -> str:
    return " → ".join(str(value) for value in values)


def cell(value: Any) -> str:
    """A pipe inside a term or one of its targets would otherwise start a column."""
    return str(value).replace("|", "\\|")


def _numeric(value: str) -> bool:
    """Right-aligned only when the whole column is counts and rates."""
    return re.fullmatch(r"[-+]?[\d,.]*%?( pp)?|n/a|-", value) is not None


def table(
    title: str,
    first: str,
    columns: Sequence[str],
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    console: bool = False,
    heading: str = "####",
    width: int = 8,
) -> str:
    """One table, as a Markdown grid or as columns aligned for a terminal."""
    if not rows:
        return ""

    if not console:
        aligns = ["---:" if all(_numeric(values[i]) for _, values in rows) else "---"
                  for i in range(len(columns))]
        lines = [
            f"{heading} {title}",
            "",
            f"| {first} | {' | '.join(columns)} |",
            f"| --- | {' | '.join(aligns)} |",
            *(f"| {cell(name)} | {' | '.join(values)} |" for name, values in rows),
        ]
    else:
        label_width = max(max(len(name) for name, _ in rows), len(first))
        sized = [(label, max(len(label), width)) for label in columns]
        lines = [
            title,
            "",
            f"  {first.ljust(label_width)}  {' '.join(f'{c:>{w}}' for c, w in sized)}",
            *(
                f"  {name.ljust(label_width)}"
                f"  {' '.join(f'{v:>{w}}' for v, (_, w) in zip(values, sized))}"
                for name, values in rows
            ),
        ]

    return "\n".join([*lines, ""])


@dataclass(frozen=True)
class Scorecard:
    """One dataset's headline results for both destinations; `detail` goes to the file only."""

    heading: str
    dataset: str
    subheading: str
    facts: list[str]
    warnings: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = [f"### {self.dataset}", "", self.subheading, ""]

        for warning in self.warnings:
            first, *rest = warning.splitlines()
            lines += [f"> ⚠ {first}"] + [f"> {line}" for line in rest] + [""]

        # A list, because Markdown runs consecutive lines into one paragraph.
        lines += [f"- {fact}" for fact in [*self.facts, *self.detail]]
        lines.append("")
        return "\n".join(line.rstrip() for line in lines)

    def as_console(self) -> str:
        lines = [f"{self.heading}  -  {self.subheading}", ""]

        for warning in self.warnings:
            first, *rest = warning.splitlines()
            lines.append(f"  ! {first}")
            lines += [f"    {line}" for line in rest]

        lines += [f"  {fact}" for fact in self.facts]
        lines.append("")
        return "\n".join(line.rstrip() for line in lines)


def report_parameters(dataset: Any) -> dict[str, Any]:
    """Canonical codes, so every component files one dataset under one stratum. A file covering
    several post-mt tasks names every value they carry, so the stratum hides none of them."""
    def across(name: str) -> str | None:
        values = dict.fromkeys(
            str(task.parameters[name]) for task in dataset.tasks if task.parameters.get(name)
        )
        return ", ".join(values) or None

    return {
        "source_language": across("clean_source_language_code"),
        "target_language": across("clean_target_language_code"),
        "domain": across("domain"),
        "cat_tool_provider": across("cat_tool_provider"),
        "cat_project_id": across("cat_project_id"),
    }


def subheading_of(result: Any) -> str:
    """The language pair a dataset was scored on, and its domain where it names one."""
    pair = f"{result.parameters['source_language']} → {result.parameters['target_language']}"
    return f"{pair}  ·  {result.parameters['domain']}" if result.parameters.get("domain") else pair


def failure_warning(result: Any) -> list[str]:
    """The one warning every component raises: post-mt failed inside some segments."""
    if not result.failed_segments:
        return []
    return [
        f"{result.failed_segments}/{result.totals['segments']} segments failed inside post-mt"
        f"\n{result.failure_reason}"
    ]


def scope_note(scored: int, named: int, reason: str) -> list[str]:
    """Every rate is against REF, so it is over the segments REF set an expectation in, not all."""
    if scored == named:
        return []
    return [f"Scored against REF {scored} of {named} segments - {reason}"]


def stratum_of(parameters: Mapping[str, Any]) -> tuple[str, str]:
    """A stratum is one language pair in one domain — the cell a segment is reported in."""
    pair = (f"{parameters.get('clean_source_language_code') or '?'}"
            f"->{parameters.get('clean_target_language_code') or '?'}")
    return pair, str(parameters.get("domain") or "(no domain)")


def by_language_pair(results: Sequence[Any]) -> dict[str, dict[str, list[Any]]]:
    """Every scored segment under its language pair, then under its domain. One dataset can
    span several of both, so this groups segments rather than datasets."""
    grouped: dict[str, dict[str, list[Any]]] = {}
    for result in results:
        for segment in result.segments:
            pair, domain = segment.stratum
            grouped.setdefault(pair, {}).setdefault(domain, []).append(segment)

    # Widest evidence first, and a stable order inside each pair.
    return {
        pair: dict(sorted(domains.items()))
        for pair, domains in sorted(
            grouped.items(), key=lambda item: (-sum(len(s) for s in item[1].values()), item[0])
        )
    }


def render_report(
    results_by_component: Mapping[str, Sequence[Any]],
    sections: Sections,
    *,
    dry_run: bool,
    now: datetime,
) -> str:
    parts = [
        f"# Quality baseline — {' + '.join(results_by_component) or 'nothing'}\n\n"
        f"{'dry run' if dry_run else 'full run'} · {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
    ]

    for component, results in results_by_component.items():
        if not results:
            continue
        heading, render = sections[component]
        parts.append(f"## {heading}\n")
        parts.extend(render(results))

    return "\n".join(part for part in parts if part).rstrip() + "\n"


def report_path(components: Sequence[str], *, dry_run: bool, now: datetime) -> Path:
    stem = "+".join(components) or "baseline"
    suffix = "_dry-run" if dry_run else ""
    return Path("reports") / f"{stem}{suffix}_{now.strftime('%Y%m%d-%H%M%S')}.md"


def write_report(
    results_by_component: Mapping[str, Sequence[Any]],
    sections: Sections,
    *,
    dry_run: bool,
) -> Path:
    now = datetime.now(timezone.utc)
    path = report_path(list(results_by_component), dry_run=dry_run, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_report(results_by_component, sections, dry_run=dry_run, now=now), encoding="utf-8"
    )
    return path
