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
from sourcecode.cat_tool import PhraseClient, TermBaseResolver, XtmClient
from sourcecode.dnt import DntClient
from sourcecode.glossary import CatToolGlossary
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


def _language_slots(configured: list[str], components: list[str]) -> dict[str, str]:
    """`BENCH_LANGUAGE` lines up slot by slot with `BENCH_COMPONENT`, a blank slot scoring every pair."""
    if len(configured) > len(components):
        raise ValueError(
            f"BENCH_LANGUAGE has {len(configured)} slots but BENCH_COMPONENT names "
            f"{len(components)}; they line up slot by slot."
        )
    slots = dict(zip(components, configured + [""] * (len(components) - len(configured))))

    # Parsed now so a typo stops the run before it measures anything.
    for given in slots.values():
        if given:
            text_processing.parse_language_pairs(given)
    return slots


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

    try:
        slots = _language_slots(config.benchmark.languages, components)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    stanza = glossary = term_bases = dnt = postmt = None

    try:
        if "glossary" in components:
            if not config.stanza.base_url:
                print("Set STANZA_BASE_URL to the lemmatizer.", file=sys.stderr)
                return 2

            # Terms are read from the CAT tool the task names, so without one there is no glossary.
            if not (config.phrase.configured or config.xtm.configured):
                print(
                    "Set PHRASE_BASE_URL/PHRASE_USERNAME/PHRASE_PASSWORD or "
                    "XTM_BASE_URL/XTM_CLIENT/XTM_USER_ID/XTM_PASSWORD. Terminology reads its "
                    "terms from the CAT tool that holds the project's term bases.",
                    file=sys.stderr,
                )
                return 2

            stanza = StanzaClient(config.stanza.base_url, config.stanza.timeout, config.stanza.batch_size)
            phrase = PhraseClient(
                config.phrase.base_url, config.phrase.username, config.phrase.password,
                config.phrase.timeout,
            ) if config.phrase.configured else None
            xtm = XtmClient(
                config.xtm.base_url, config.xtm.client, config.xtm.user_id,
                config.xtm.password, config.xtm.timeout,
            ) if config.xtm.configured else None

            term_bases = TermBaseResolver(phrase=phrase, xtm=xtm)
            glossary = CatToolGlossary(stanza, phrase=phrase, xtm=xtm)

            for name, configured in (("Phrase/Memsource", config.phrase.configured),
                                     ("XTM", config.xtm.configured)):
                if not configured:
                    logging.warning("[BENCH] %s is not configured - its tasks retrieve no glossary", name)

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

            wanted = slots[component]
            languages = text_processing.parse_language_pairs(wanted) if wanted else ()

            _, module, _, console = COMPONENTS[component]
            results = []
            for path in datasets:
                logging.info("[BENCH] loading %s", path)
                data = (
                    glossary_benchmark.load_dataset(
                        path, term_bases=term_bases, dry_run=args.dry_run, languages=languages,
                    )
                    if component == "glossary" else
                    pipeline.load_dataset(
                        path, component=component, dry_run=args.dry_run, languages=languages,
                    )
                )
                if not data.tasks:
                    logging.info("[BENCH] nothing to score in %s - skipped", path)
                    continue

                if component == "glossary":
                    scored = glossary_benchmark.run_benchmark(
                        data, postmt=postmt, stanza=stanza, glossary=glossary,
                        term_bases=term_bases, config=config, skip_pipeline=args.dry_run,
                    )
                elif component == "dnt":
                    scored = dnt_benchmark.run_benchmark(
                        data, postmt=postmt, dnt=dnt, config=config, skip_pipeline=args.dry_run
                    )
                elif component == "tags":
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

            if not results:
                print(f"No {component} dataset could be scored - see the warnings above. Check "
                      f"{PATH_VARIABLES[component]}, and BENCH_LANGUAGE if it is set.",
                      file=sys.stderr)
                return 1

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
        for client in (stanza, glossary, term_bases, dnt, postmt):
            if client is not None:
                client.close()


if __name__ == "__main__":
    raise SystemExit(main())
