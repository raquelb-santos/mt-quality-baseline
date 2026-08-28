"""The report: the primitives a component formats its numbers with, and the file a run writes."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Per component: the heading it is filed under, and what renders its parts.
Sections = Mapping[str, tuple[str, Callable[[Sequence[Any]], list[str]]]]

REPORTS_DIR = Path("reports")


def rate(numerator: int, denominator: int) -> float | None:
    """None, never 0, with no denominator: a misconfigured run is not total failure."""
    return None if denominator == 0 else numerator / denominator


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.2f}%"


def signed_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def cell(value: Any) -> str:
    """A pipe inside a term or one of its targets would otherwise start a column."""
    return str(value).replace("|", "\\|")


@dataclass(frozen=True)
class Scorecard:
    """One dataset's headline results for both destinations; `detail` goes to the file only."""

    heading: str
    subheading: str
    facts: list[str]
    warnings: list[str] = field(default_factory=list)
    detail: list[str] = field(default_factory=list)

    def as_markdown(self) -> str:
        lines = [f"## {self.heading}", "", self.subheading, ""]

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


def stratum_of(result: Any) -> tuple[str, str]:
    """A stratum is one language pair in one domain — the cell a result is reported in."""
    parameters = result.parameters
    pair = f"{parameters.get('source_language', '?')}->{parameters.get('target_language', '?')}"
    return pair, str(parameters.get("domain") or "(no domain)")


def by_stratum(results: Sequence[Any]) -> dict[tuple[str, str], list[Any]]:
    """Group results into strata, preserving the order each stratum was first seen."""
    grouped: dict[tuple[str, str], list[Any]] = {}
    for result in results:
        grouped.setdefault(stratum_of(result), []).append(result)
    return grouped


def render_report(
    results_by_component: Mapping[str, Sequence[Any]],
    sections: Sections,
    *,
    dry_run: bool,
    now: datetime,
) -> str:
    measured = " + ".join(results_by_component) or "nothing"
    parts = [
        f"# Quality baseline — {measured}\n\n"
        f"{'dry run' if dry_run else 'full run'} · {now.strftime('%Y-%m-%d %H:%M UTC')}\n"
    ]

    for component, results in results_by_component.items():
        if not results:
            continue
        heading, render = sections[component]
        parts.append(f"# {heading}\n")
        parts.extend(render(results))

    return "\n".join(part for part in parts if part).rstrip() + "\n"


def report_path(components: Sequence[str], *, dry_run: bool, now: datetime) -> Path:
    stem = "+".join(components) or "baseline"
    suffix = "_dry-run" if dry_run else ""
    return REPORTS_DIR / f"{stem}{suffix}_{now.strftime('%Y%m%d-%H%M%S')}.md"


def write_report(
    results_by_component: Mapping[str, Sequence[Any]],
    sections: Sections,
    *,
    dry_run: bool,
) -> Path:
    """Write the run's report and return where it went, for the caller to name on the console."""
    now = datetime.now(timezone.utc)
    path = report_path(list(results_by_component), dry_run=dry_run, now=now)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_report(results_by_component, sections, dry_run=dry_run, now=now), encoding="utf-8"
    )
    return path
