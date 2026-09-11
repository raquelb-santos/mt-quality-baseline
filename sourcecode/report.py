"""Report primitives every component formats its numbers with, and the file a run writes."""

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
    return "n/a" if value is None else f"{value * 100:+.2f}%"


def arrow(values: Sequence[Any]) -> str:
    return " → ".join(str(value) for value in values)


def cell(value: Any) -> str:
    """A pipe inside a term or one of its targets would otherwise start a column."""
    return str(value).replace("|", "\\|")


def table(
    title: str,
    first: str,
    columns: Sequence[str],
    rows: Sequence[tuple[str, Sequence[str]]],
    *,
    console: bool = False,
    heading: str = "###",
    width: int = 8,
) -> str:
    """One table, as a Markdown grid or as columns aligned for a terminal."""
    if not rows:
        return ""

    if not console:
        lines = [
            f"{heading} {title}",
            "",
            f"| {first} | {' | '.join(columns)} |",
            f"| --- | {' | '.join('---:' for _ in columns)} |",
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


def report_parameters(dataset: Any) -> dict[str, Any]:
    """Canonical codes, so every component files one dataset under one stratum."""
    parameters = dataset.parameters
    return {
        "source_language": parameters.get("clean_source_language_code"),
        "target_language": parameters.get("clean_target_language_code"),
        "domain": parameters.get("domain"),
        "cat_tool_provider": parameters.get("cat_tool_provider"),
        "cat_project_id": parameters.get("cat_project_id"),
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


def spend_fact(usage: Any) -> list[str]:
    if not (usage.cost or usage.tokens):
        return []
    return [
        f"LLM spend ${usage.cost:.4f} · {usage.tokens:,} tokens "
        f"({usage.prompt_tokens:,} prompt / {usage.completion_tokens:,} completion)"
    ]


def stratum_of(result: Any) -> tuple[str, str]:
    """A stratum is one language pair in one domain — the cell a result is reported in."""
    parameters = result.parameters
    pair = f"{parameters.get('source_language', '?')}->{parameters.get('target_language', '?')}"
    return pair, str(parameters.get("domain") or "(no domain)")


def by_stratum(results: Sequence[Any]) -> dict[tuple[str, str], list[Any]]:
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
    parts = [
        f"# Quality baseline — {' + '.join(results_by_component) or 'nothing'}\n\n"
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
