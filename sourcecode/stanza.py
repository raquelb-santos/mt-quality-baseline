"""Client for the Stanza lemmatization service."""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)


class StanzaClient:
    def __init__(self, base_url: str, timeout: float = 120.0) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def lemmatize_batch(self, texts: list[str], language: str) -> list[str]:
        if not texts:
            return []
        response = self._client.post("/lemmatize/batch", json={"texts": texts, "language": language})
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload.get("lemmatized_texts") or []
        return payload or []

    def lemmatize_batch_safe(self, texts: list[str], language: str) -> list[str] | None:
        try:
            lemmas = self.lemmatize_batch(texts, language)
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("[STANZA] lemmatization unavailable for %s: %s", language, error)
            return None

        if len(lemmas) != len(texts):
            logger.warning(
                "[STANZA] returned %d lemmas for %d texts (%s) - ignoring",
                len(lemmas), len(texts), language,
            )
            return None

        return lemmas
