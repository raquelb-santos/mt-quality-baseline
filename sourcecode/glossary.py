"""Glossary terms, read from the CAT tool that holds the project's term bases."""

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import httpx

from .cat_tool import PHRASE_PROVIDERS, Term, XTM_PROVIDERS
from .text_processing import count_surface

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlossaryMatches:
    mappings: list[dict[str, str]]
    # Aligned with the texts queried: per_text_mappings[i] belongs to texts[i].
    per_text_mappings: list[list[dict[str, str]]]


def language_variants(language: str) -> list[str]:
    """Full code and base code (en-us -> [en-us, en])."""
    return list(dict.fromkeys([language, str(language).split("-")[0]]))


class CatToolGlossary:
    """Glossary terms read from the CAT tool that holds the project's term bases.

    Terms and segments are both matched on their lemmas, so an inflected wording still counts as
    the term having been used.
    """

    def __init__(self, stanza: Any, phrase: Any = None, xtm: Any = None) -> None:
        self._stanza = stanza
        self._clients = {
            **{provider: phrase for provider in PHRASE_PROVIDERS if phrase is not None},
            **{provider: xtm for provider in XTM_PROVIDERS if xtm is not None},
        }
        self._terms: dict[tuple[str, str], list[Term]] = {}
        self._lemmas: dict[tuple[str, str], str] = {}

    def close(self) -> None:
        """The CAT clients belong to the resolver that shares them, and are closed there."""

    def terms_in(self, term_base_id: str, provider: str) -> list[Term]:
        provider = str(provider or "").strip().lower()
        client = self._clients.get(provider)
        if client is None:
            return []

        key = (provider, str(term_base_id))
        if key not in self._terms:
            try:
                self._terms[key] = client.terms(str(term_base_id))
            except (httpx.HTTPError, RuntimeError, ValueError) as error:
                logger.error("[GLOSSARY] %s could not read term base %s: %s", provider, term_base_id, error)
                return []
            logger.debug("[GLOSSARY] term base %s holds %d terms", term_base_id, len(self._terms[key]))

        return self._terms[key]

    def _lemmatize(self, texts: Sequence[str], language: str) -> list[str]:
        """Cached across term bases and tasks, since the same term recurs throughout a run."""
        unknown = [text for text in dict.fromkeys(texts) if (language, text) not in self._lemmas]
        if unknown:
            lemmas = self._stanza.lemmatize_batch_safe(unknown, language)
            # Degrade to surface forms rather than abort, as retrieval does elsewhere.
            if lemmas is None:
                logger.warning("[GLOSSARY] terms not lemmatized for %s - matching surface forms", language)
                lemmas = unknown
            self._lemmas.update(zip(((language, text) for text in unknown), lemmas))

        return [self._lemmas[(language, text)] for text in texts]

    def fetch_matches(
        self,
        *,
        glossary_ids: Sequence[str],
        source_language: str,
        target_language: str,
        texts: Sequence[str],
        provider: str | None = None,
    ) -> GlossaryMatches:
        """Resolve glossary matches for a batch of lemmatized texts, against the terms the CAT
        tool holds in the project's own term bases."""
        for value, label in (
            (glossary_ids, "glossary IDs"),
            (source_language, "source language"),
            (target_language, "target language"), (texts, "texts"),
        ):
            if not value:
                raise ValueError(f"No {label} provided")

        terms = [term for identifier in glossary_ids for term in self.terms_in(identifier, provider)]

        # Permissive language matching, over the full code and the base code.
        wanted_source = set(language_variants(source_language))
        wanted_target = set(language_variants(target_language))

        source_terms = [term for term in terms if term.language.lower() in wanted_source]
        targets_by_concept: dict[str, list[str]] = {}
        for term in terms:
            if term.language.lower() in wanted_target:
                targets_by_concept.setdefault(term.concept_id, []).append(term.text)

        source_lemmas = self._lemmatize([term.text for term in source_terms], source_language)

        per_text_mappings = []
        for text in texts:
            mappings: list[dict[str, str]] = []
            seen: set[str] = set()
            for term, lemma in zip(source_terms, source_lemmas):
                if not count_surface(text, lemma, source_language):
                    continue
                for target_text in targets_by_concept.get(term.concept_id, []):
                    if target_text not in seen:
                        seen.add(target_text)
                        mappings.append({"source_content": term.text, "target_content": target_text})
            per_text_mappings.append(mappings)

        # Keyed on the pair, so a target two source terms reach in different texts is kept once each.
        flat: dict[tuple[str, str], dict[str, str]] = {}
        for mappings in per_text_mappings:
            for mapping in mappings:
                flat.setdefault((mapping["source_content"], mapping["target_content"]), mapping)

        return GlossaryMatches(mappings=list(flat.values()), per_text_mappings=per_text_mappings)
