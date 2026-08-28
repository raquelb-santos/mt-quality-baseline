"""Drives a whole run - what `.env` says to measure, the clients for it, what each component
renders, and the exit code - and is the only module aware of every component."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Running this file directly leaves the package unimported, so the imports below would fail.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcecode import dnt_benchmark, dnt_report, glossary_benchmark, glossary_report, report
from sourcecode import text_processing
from sourcecode.config import PATH_VARIABLES, Config
from sourcecode.dnt import DntClient
from sourcecode.glossary import GlossaryClient
from sourcecode.postmt import PostMtClient, StanzaClient


# Each component's heading and the parts filed under it.
COMPONENT_SECTIONS = {
    "glossary": (
        "Terminology adherence",
        lambda results: [
            *(part for result in results
              for part in (glossary_report.glossary_scorecard(result).as_markdown(),
                           glossary_report.render_term_adherence(result))),
            glossary_report.render_comparison(results),
            glossary_report.render_strata(results),
        ],
    ),
    "dnt": (
        "DNT preservation",
        lambda results: [
            *(part for result in results
              for part in (dnt_report.dnt_scorecard(result).as_markdown(),
                           dnt_report.render_dnt_detection(result),
                           dnt_report.render_dnt_items(result))),
            dnt_report.render_dnt_comparison(results),
            dnt_report.render_dnt_strata(results),
        ],
    ),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mt-quality-baseline",
        description="Measure the translation quality of the post-mt pipeline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="score the MT baseline only; never calls post-mt",
    )
    return parser


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
    logging.getLogger("httpx").setLevel(logging.WARNING)

    config = Config()

    components = config.benchmark.components
    if not components:
        print(f"Set BENCH_COMPONENT to one or more of: {', '.join(text_processing.COMPONENTS)}.",
              file=sys.stderr)
        return 2
    unknown = [c for c in components if c not in text_processing.COMPONENTS]
    if unknown:
        print(f"BENCH_COMPONENT names {', '.join(unknown)}; expected one or more of: "
              f"{', '.join(text_processing.COMPONENTS)}.", file=sys.stderr)
        return 2

    stanza = glossary = dnt = postmt = None

    try:
        if "glossary" in components:
            if not config.search_engine.node:
                print("Set SEARCH_ENGINE_URL to the term-bases index post-mt queries.",
                      file=sys.stderr)
                return 2
            if config.search_engine.aws_sigv4 and not config.search_engine.aws_region:
                print("ES_AWS_SIGV4_ENABLED is set but AWS_REGION is not.", file=sys.stderr)
                return 2

            stanza = StanzaClient(config.stanza.base_url, config.stanza.timeout)
            glossary = GlossaryClient(
                config.search_engine.node,
                config.search_engine.username,
                config.search_engine.password,
                config.search_engine.timeout,
                aws_region=config.search_engine.aws_region if config.search_engine.aws_sigv4 else None,
                aws_profile=config.search_engine.aws_profile,
            )
            if not glossary.ping():
                print(f"Search engine unreachable at {config.search_engine.node}.", file=sys.stderr)
                return 1

        if "dnt" in components:
            if not config.dnt.base_url:
                print("Set DNT_BASE_URL to the DNT service.", file=sys.stderr)
                return 2

            dnt = DntClient(config.dnt.base_url, config.dnt.api_key, config.dnt.timeout)
            if not dnt.health():
                fix = "Check DNT_API_KEY." if dnt.authenticated else "Set DNT_API_KEY."
                print(f"Cannot use the DNT service at {config.dnt.base_url}. {fix}", file=sys.stderr)
                return 1

        if not args.dry_run:
            postmt = PostMtClient(
                config.postmt.base_url,
                config.postmt.poll_interval,
                config.postmt.timeout,
                config.postmt.api_key,
            )
            if not postmt.health():
                fix = "Check POSTMT_API_KEY." if postmt.authenticated else "Set POSTMT_API_KEY."
                print(f"Cannot use post-mt at {config.postmt.base_url}. {fix}", file=sys.stderr)
                return 1

        logging.info("Measuring: %s", ", ".join(components))

        results_by_component: dict[str, list[Any]] = {}
        for component in components:
            configured = config.benchmark.data_path(component)
            datasets = text_processing.find_datasets(configured, variable=PATH_VARIABLES[component])
            logging.info("[BENCH] %s: %s", PATH_VARIABLES[component], configured)
            if len(datasets) > 1:
                logging.info("[BENCH] %d %s datasets to score", len(datasets), component)

            results = []
            for path in datasets:
                logging.info("[BENCH] loading %s", path)
                if component == "glossary":
                    data = glossary_benchmark.load_dataset(
                        path, glossary=glossary,
                        node=config.search_engine.node, dry_run=args.dry_run,
                    )
                    scored = glossary_benchmark.Benchmark(
                        postmt=postmt, stanza=stanza, glossary=glossary, config=config
                    ).run(data, skip_pipeline=args.dry_run)
                    blocks = [
                        glossary_report.glossary_scorecard(scored).as_console(),
                        glossary_report.render_term_adherence_console(scored),
                    ]
                if component == "dnt":
                    data = dnt_benchmark.load_dataset(path, dry_run=args.dry_run)
                    scored = dnt_benchmark.DntBenchmark(postmt=postmt, dnt=dnt, config=config).run(
                        data, skip_pipeline=args.dry_run
                    )
                    blocks = [
                        dnt_report.dnt_scorecard(scored).as_console(),
                        dnt_report.render_dnt_detection_console(scored),
                        dnt_report.render_dnt_items_console(scored),
                    ]

                for block in blocks:
                    if block:
                        print(block)
                results.append(scored)
            results_by_component[component] = results

            # After the datasets: pools every dataset that shares a stratum
            if component == "glossary" and results:
                print(glossary_report.render_strata_console(results))
            if component == "dnt" and results:
                print(dnt_report.render_dnt_strata_console(results))

        report_file = report.write_report(
            results_by_component,
            COMPONENT_SECTIONS,
            dry_run=args.dry_run,
        )
        logging.info("[BENCH] report: %s", report_file)

        return 0

    except KeyboardInterrupt:
        # A queued task keeps running server-side and keeps billing unless it is cancelled.
        if postmt is not None and postmt.cancel_active():
            print("\nInterrupted - cancelled the in-flight post-mt task.", file=sys.stderr)
        else:
            print("\nInterrupted.", file=sys.stderr)
        return 130

    except (OSError, ValueError, RuntimeError) as error:
        logging.error("%s", error)
        return 1
    finally:
        for client in (stanza, glossary, dnt, postmt):
            if client is not None:
                client.close()


if __name__ == "__main__":
    raise SystemExit(main())
