"""Drives a whole run - the only module aware of every component - and sets the exit code."""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

# Running this file directly leaves the package unimported, so the imports below would fail.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sourcecode import dnt_benchmark, dnt_report, glossary_benchmark, glossary_report, pipeline, report, tags_benchmark, tags_report, text_processing
from sourcecode.config import PATH_VARIABLES, Config
from sourcecode.dnt import DntClient
from sourcecode.glossary import GlossaryClient
from sourcecode.postmt import PostMtClient, StanzaClient


COMPONENTS = {
    "glossary": ("Terminology adherence", glossary_report,
                 (glossary_report.render_terms,),
                 (glossary_report.render_terms_console,)),
    "dnt": ("DNT preservation", dnt_report,
            (dnt_report.render_detection, dnt_report.render_items,
             dnt_report.render_defects),
            (dnt_report.render_detection_console, dnt_report.render_items_console)),
    "tags": ("Tag and placeholder integrity", tags_report,
             (tags_report.render_families, tags_report.render_tags, tags_report.render_defects),
             (tags_report.render_families_console, tags_report.render_tags_console)),
}


def _section(module: Any, own: tuple[Any, ...]) -> Any:
    """The scorecard and each component's own parts per dataset, then the corpus-wide tables."""
    return lambda results: [
        *(part for result in results
          for part in (module.scorecard(result).as_markdown(),
                       *(render(result) for render in own))),
        module.render_comparison(results),
        module.render_strata(results),
    ]


COMPONENT_SECTIONS = {
    component: (heading, _section(module, own))
    for component, (heading, module, own, _) in COMPONENTS.items()
}


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
    parser = argparse.ArgumentParser(
        prog="mt-quality-baseline",
        description="Measure the translation quality of the post-mt pipeline.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="score the MT baseline only; never calls post-mt",
    )
    args = parser.parse_args(argv)

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
        print(f"Set BENCH_COMPONENT to one or more of: {', '.join(COMPONENTS)}.",
              file=sys.stderr)
        return 2
    unknown = [c for c in components if c not in COMPONENTS]
    if unknown:
        print(f"BENCH_COMPONENT names {', '.join(unknown)}; expected one or more of: "
              f"{', '.join(COMPONENTS)}.", file=sys.stderr)
        return 2

    stanza = glossary = dnt = postmt = None

    try:
        if "glossary" in components:
            search = config.search_engine
            if search.aws_sigv4 and not search.aws_region:
                print("ES_AWS_SIGV4_ENABLED is set but AWS_REGION is not.", file=sys.stderr)
                return 2

            if not search.node:
                print("Set SEARCH_ENGINE_URL to the term-bases index post-mt queries.",
                      file=sys.stderr)
                return 2

            if not config.stanza.base_url:
                print("Set STANZA_BASE_URL to the lemmatizer.", file=sys.stderr)
                return 2

            stanza = StanzaClient(config.stanza.base_url, config.stanza.timeout)
            glossary = GlossaryClient(
                search.node, search.username, search.password, search.timeout,
                search.aws_region if search.aws_sigv4 else None, search.aws_profile,
            )
            if not glossary.ping():
                print(f"Search engine unreachable at {search.node}.", file=sys.stderr)
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
            if not config.postmt.base_url:
                print("Set POSTMT_BASE_URL to the post-mt instance to drive.", file=sys.stderr)
                return 2

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
            configured = config.benchmark.paths[component]
            datasets = text_processing.find_datasets(
                configured, variable=PATH_VARIABLES[component],
            )
            logging.info("[BENCH] %s: %s", PATH_VARIABLES[component], configured)
            if len(datasets) > 1:
                logging.info("[BENCH] %d %s datasets to score", len(datasets), component)

            _, module, _, console = COMPONENTS[component]
            results = []
            for path in datasets:
                logging.info("[BENCH] loading %s", path)
                if component == "glossary":
                    data = glossary_benchmark.load_dataset(
                        path, glossary=glossary, node=config.search_engine.node,
                        dry_run=args.dry_run,
                    )
                    scored = glossary_benchmark.run_benchmark(
                        data, postmt=postmt, stanza=stanza, glossary=glossary, config=config,
                        skip_pipeline=args.dry_run,
                    )
                elif component == "dnt":
                    data = pipeline.load_dataset(path, component="dnt", dry_run=args.dry_run)
                    scored = dnt_benchmark.run_benchmark(
                        data, postmt=postmt, dnt=dnt, config=config, skip_pipeline=args.dry_run
                    )
                elif component == "tags":
                    data = pipeline.load_dataset(path, component="tags", dry_run=args.dry_run)
                    scored = tags_benchmark.run_benchmark(
                        data, postmt=postmt, config=config, skip_pipeline=args.dry_run
                    )
                else:
                    raise ValueError(f"{component} is in COMPONENTS but has no branch here.")

                for block in (module.scorecard(scored).as_console(),
                              *(render(scored) for render in console)):
                    if block:
                        print(block)
                results.append(scored)

            results_by_component[component] = results
            if results:
                print(module.render_strata_console(results))

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
