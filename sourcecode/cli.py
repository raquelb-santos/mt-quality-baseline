"""Command line entry point.
What gets scored comes from `BENCH_DATASET` in `.env`, so a configured run takes no arguments.
The report is printed and not kept; `--help` lists the invocations.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Running this file directly leaves the package unimported, so the imports below would fail.
# Putting the repo root on sys.path makes `python sourcecode/cli.py` work from a plain clone.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcecode import dataset as dataset_module
from sourcecode import report as report_module
from sourcecode.benchmark import Benchmark, BenchmarkResult
from sourcecode.config import load_config
from sourcecode.glossary import GlossaryClient
from sourcecode.postmt import PostMtClient, preflight_parameters
from sourcecode.stanza import StanzaClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mt-quality-baseline",
        description="Measure terminology adherence and glossary violations for the post-mt pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "the dataset comes from BENCH_DATASET in .env (a file, or a folder of them).\n"
            "\n"
            "examples:\n"
            "  # resolve the glossary and score the MT baseline; never calls post-mt\n"
            "  python sourcecode/cli.py --dry-run\n"
            "\n"
            "  # full run: post-mt post-edits the segments, MT vs post-edited is scored\n"
            "  python sourcecode/cli.py\n"
        ),
    )
    parser.add_argument("--params", help="JSON parameters block, for .csv/.mxliff inputs")
    parser.add_argument("--source-lang", help="source language code or name, e.g. en-gb")
    parser.add_argument("--target-lang", help="target language code or name, e.g. fr-fr")
    parser.add_argument("--domain", help="domain label, recorded in reports")
    parser.add_argument("--glossary-ids", help="comma-separated glossary ids (overrides the dataset)")
    parser.add_argument("--steps", help="pipeline steps, e.g. AQE,APE (default: AQE,APE)")
    parser.add_argument("--batch-size", type=int, help="segments per post-mt task")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="resolve glossary and score the MT baseline only; never calls post-mt",
    )
    return parser


# Kept in step with the suffixes `dataset.load` accepts: a folder scan that skipped a format
# the loader supports would report "no dataset files" for a folder full of them.
DATASET_SUFFIXES = (".json", ".csv", ".mxliff", ".xliff", ".xlf")


def resolve_datasets(configured: str) -> list[Path]:
    """Expand `BENCH_DATASET` into the datasets to score.

    It is the single source of what gets measured: one place to look when a number is surprising,
    and a run that is reproducible from the `.env` alone rather than from a shell history.

    The value may be a single file or a folder. A folder contributes every dataset directly inside
    it, sorted, and is not searched recursively — a nested folder is usually a different
    experiment, and silently absorbing it would change what a run means without saying so.
    """
    if not configured.strip():
        raise ValueError(
            "No dataset to score. Set BENCH_DATASET in .env to a dataset file, or to a folder "
            "to score every dataset inside it."
        )

    candidate = Path(configured)
    if candidate.is_dir():
        found = sorted(
            child for child in candidate.iterdir()
            if child.is_file() and child.suffix.lower() in DATASET_SUFFIXES
        )
        if not found:
            raise ValueError(
                f"No dataset files in {candidate} (looked for {', '.join(DATASET_SUFFIXES)})."
            )
        return found
    if candidate.is_file():
        return [candidate]

    raise ValueError(f"BENCH_DATASET points at nothing: {candidate}")


def _force_utf8_output() -> None:
    """Windows consoles default to a code page that cannot encode translated content."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # detached or already-wrapped stream
                pass


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    _force_utf8_output()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    # httpx logs a line per request, which buries this tool's own progress output.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = load_config()
    if args.batch_size:
        config.benchmark.batch_size = args.batch_size

    overrides: dict = {}
    if args.params:
        overrides["parameters"] = json.loads(Path(args.params).read_text(encoding="utf-8"))

    inline_parameters = {
        key: value
        for key, value in (
            ("source_language", args.source_lang),
            ("target_language", args.target_lang),
            ("domain", args.domain),
        )
        if value
    }
    if inline_parameters:
        overrides["parameters"] = {**overrides.get("parameters", {}), **inline_parameters}
    if args.glossary_ids:
        overrides["glossary_ids"] = [gid.strip() for gid in args.glossary_ids.split(",") if gid.strip()]
    if args.steps:
        overrides["steps"] = [step.strip().upper() for step in args.steps.split(",") if step.strip()]

    # An unreachable Stanza degrades on its own: lemmatize_batch_safe warns and the run falls
    # back to surface forms, so there is no flag to turn it off deliberately. Scoring surface
    # forms on purpose is BENCH_LEMMA_MATCHING in `.env`, not a flag: the report records the
    # config it ran under, and a flag would make that record disagree with the checked-in one.
    stanza = StanzaClient(config.stanza.base_url, config.stanza.timeout)

    if not config.search_engine.node:
        print(
            "No glossary source. Set SEARCH_ENGINE_URL to the term-bases index post-mt "
            "queries (it lives in Parameter Store; see the README).",
            file=sys.stderr,
        )
        return 2
    if config.search_engine.aws_sigv4 and not config.search_engine.aws_region:
        print("ES_AWS_SIGV4_ENABLED is set but AWS_REGION is not.", file=sys.stderr)
        return 2

    glossary = GlossaryClient(
        config.search_engine.node,
        config.search_engine.username,
        config.search_engine.password,
        config.search_engine.timeout,
        aws_region=config.search_engine.aws_region if config.search_engine.aws_sigv4 else None,
        aws_profile=config.search_engine.aws_profile,
    )
    postmt = PostMtClient(
        config.postmt.base_url,
        config.postmt.poll_interval,
        config.postmt.timeout,
        config.postmt.api_key,
    )

    try:
        if not glossary.ping():
            print(
                f"Search engine unreachable at {config.search_engine.node}. Set SEARCH_ENGINE_URL.",
                file=sys.stderr,
            )
            return 1
        if not args.dry_run and not postmt.health():
            # health() has already logged the specific cause (401 / redirect / transport).
            hint = (
                "Check POSTMT_API_KEY." if postmt.authenticated
                else "Set POSTMT_API_KEY, or check POSTMT_BASE_URL."
            )
            print(f"Cannot use post-mt at {config.postmt.base_url}. {hint}", file=sys.stderr)
            return 1

        benchmark = Benchmark(postmt=postmt, stanza=stanza, glossary=glossary, config=config)

        datasets = resolve_datasets(config.benchmark.dataset)
        logging.info("[BENCH] BENCH_DATASET: %s", config.benchmark.dataset)
        if len(datasets) > 1:
            logging.info("[BENCH] %d datasets to score", len(datasets))

        results: list[BenchmarkResult] = []
        for path in datasets:
            logging.info("[BENCH] loading %s", path)
            data = dataset_module.load(path, overrides, require_glossary_ids=True)

            # An id that is not in this cluster does not fail: the percolate simply matches
            # nothing, and the run reports a clean-looking 0-instance scorecard.
            if not glossary.count_terms(
                data.glossary_ids, data.parameters.get("cat_tool_provider")
            ):
                print(
                    f"None of the glossary ids ({', '.join(data.glossary_ids)}) exist in the "
                    f"term-bases index at {config.search_engine.node}. The run would score "
                    f"0 expected instances and read like a clean result. "
                    f"Check the ids are CAT term-base uids rather than another system's "
                    f"and that this is the cluster post-mt queries.",
                    file=sys.stderr,
                )
                return 1

            # For some parameter sets post-mt silently retrieves no glossary, at full LLM cost.
            if not args.dry_run:
                problems = preflight_parameters(data.parameters)
                if problems:
                    for problem in problems:
                        logging.error("[PREFLIGHT] %s", problem)
                    print(
                        "\nPreflight failed: post-mt would run these segments but "
                        "retrieve no glossary, so the post-edited column would "
                        "measure nothing - at full LLM cost.\n"
                        "Fix the parameters above.",
                        file=sys.stderr,
                    )
                    return 1

            result = benchmark.run(data, skip_pipeline=args.dry_run)
            results.append(result)

            print(report_module.render_summary(result))
            print(report_module.render_term_adherence(result))

        if len(results) > 1:
            print(report_module.render_comparison(results))

        print(report_module.render_strata(results))

        return 0

    except KeyboardInterrupt:
        # A queued task keeps running server-side and keeps billing unless it is cancelled.
        if postmt.cancel_active():
            print("\nInterrupted - cancelled the in-flight post-mt task.", file=sys.stderr)
        else:
            print("\nInterrupted.", file=sys.stderr)
        return 130

    except (dataset_module.DatasetError, OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    finally:
        stanza.close()
        glossary.close()
        postmt.close()


if __name__ == "__main__":
    raise SystemExit(main())
