"""Glossary resolution against the term-bases index, sending the same queries post-mt sends."""

import logging
from dataclasses import dataclass
from typing import Sequence

from .search_engine import SearchClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlossaryMatches:
    mappings: list[dict[str, str]]
    # Aligned with the texts queried: per_text_mappings[i] belongs to texts[i].
    per_text_mappings: list[list[dict[str, str]]]


def language_variants(language: str) -> list[str]:
    """Full code and base code (en-us -> [en-us, en])."""
    return list(dict.fromkeys([language, str(language).split("-")[0]]))


def as_id_list(glossary_ids: Sequence[str]) -> list[str]:
    return [str(gid).strip() for gid in glossary_ids if str(gid).strip()]


def _term_index(provider: str | None) -> str:
    return "xtm-term-bases" if str(provider or "").lower() == "xtm" else "term-bases"


class GlossaryClient:

    def __init__(
        self,
        node: str,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 120.0,
        aws_region: str | None = None,
        aws_profile: str | None = None,
    ) -> None:
        self.search = SearchClient(node, username, password, timeout, aws_region, aws_profile)

    def close(self) -> None:
        self.search.close()

    def ping(self) -> bool:
        return self.search.ping()

    def count_terms(self, glossary_ids: Sequence[str], provider: str | None = None) -> int:
        """Documents the index holds for these ids — 0 means the ids are not in this cluster."""
        return self.search.count(
            _term_index(provider),
            {"query": {"terms": {"glossary_id": as_id_list(glossary_ids)}}},
        )

    def fetch_matches(
        self,
        *,
        glossary_ids: Sequence[str],
        source_language: str,
        target_language: str,
        texts: Sequence[str],
        provider: str | None = None,
    ) -> GlossaryMatches:
        """Resolve glossary matches for a batch of lemmatized texts."""
        ids = as_id_list(glossary_ids)
        for value, label in (
            (ids, "glossary IDs"), (source_language, "source language"),
            (target_language, "target language"), (texts, "texts"),
        ):
            if not value:
                raise ValueError(f"No {label} provided")

        index = _term_index(provider)
        per_text_source_terms = [[] for _ in texts]
        concept_ids: set[str] = set()

        responses = self.search.msearch(index, [
            {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"glossary_id": ids}},
                            {"terms": {"language": language_variants(source_language)}},
                        ],
                        "must": [{"percolate": {"field": "query", "document": {"content": str(text)}}}],
                    }
                },
                "size": 50,
                "sort": ["_score"],
            }
            for text in texts
        ])

        for i, response in enumerate(responses):
            if response.get("error"):
                logger.warning("[GLOSSARY] percolate error on text %d: %s", i, response["error"])
                continue
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                per_text_source_terms[i].append(
                    {"term_text": source.get("term_text"), "concept_id": source.get("concept_id")}
                )
                concept_ids.add(source.get("concept_id"))

        targets_by_concept: dict[str, list[str]] = {}
        if concept_ids:
            response = self.search.search(index, {
                "query": {
                    "bool": {
                        "filter": [
                            {"terms": {"concept_id": sorted(concept_ids)}},
                            {"terms": {"language": language_variants(target_language)}},
                        ]
                    }
                },
                "size": 1000,
            })
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                targets_by_concept.setdefault(source.get("concept_id"), []).append(
                    source.get("term_text")
                )

        per_text_mappings = []
        for source_terms in per_text_source_terms:
            mappings: list[dict[str, str]] = []
            seen: set[str] = set()
            for source in source_terms:
                for target_text in targets_by_concept.get(source["concept_id"], []):
                    if target_text not in seen:
                        seen.add(target_text)
                        mappings.append({"source_content": source["term_text"], "target_content": target_text})
            per_text_mappings.append(mappings)

        # Keyed on the pair: two source terms sharing a target are two mappings, not one.
        flat: dict[tuple[str, str], dict[str, str]] = {}
        for mappings in per_text_mappings:
            for mapping in mappings:
                flat.setdefault((mapping["source_content"], mapping["target_content"]), mapping)

        return GlossaryMatches(mappings=list(flat.values()), per_text_mappings=per_text_mappings)
