"""Client for the DNT service: one `/v1/revert` yields both the corrected text and the items."""

import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import httpx

logger = logging.getLogger(__name__)

# The keys a per-segment entry is recognised by when the response carries no envelope.
SEGMENT_KEYS = ("terms", "reverted", "result")


def base_language(code: str | None) -> str:
    """`en-gb` -> `en`. The service names languages without a region."""
    return str(code or "").split("-")[0].lower()


def item_list(raw: Any) -> list[str] | None:
    """Arrays of strings; anything else is dropped rather than inflating the denominator."""
    if not isinstance(raw, list):
        return None
    return [item.strip() for item in raw if isinstance(item, str) and item.strip()]


@dataclass(frozen=True)
class Reversion:
    rev_text: str
    items: list[str] = field(default_factory=list)
    # True when ``items`` holds only what the service repaired, so it cannot carry a denominator.
    items_are_repairs_only: bool = False


def parse_reversion(segment: Mapping[str, Any], sent_target: str) -> Reversion:
    # An empty result means the text we sent, not a segment that lost its translation.
    result = segment.get("result")
    rev_text = result if isinstance(result, str) and result else sent_target

    items = item_list(segment.get("terms"))
    if items is not None:
        return Reversion(rev_text=rev_text, items=items)

    return Reversion(
        rev_text=rev_text,
        items=item_list(segment.get("reverted")) or [],
        items_are_repairs_only=True,
    )


def response_segments(body: Any) -> list[Mapping[str, Any]]:
    """The per-segment entries, however the response wraps them."""
    if isinstance(body, list):
        return [entry for entry in body if isinstance(entry, Mapping)]

    if isinstance(body, Mapping):
        for key in ("segments", "results", "data"):
            raw = body.get(key)
            if isinstance(raw, list):
                return [entry for entry in raw if isinstance(entry, Mapping)]
        # A single-segment body with no envelope around it.
        if any(key in body for key in SEGMENT_KEYS):
            return [body]

    return []


class DntClient:

    def __init__(self, base_url: str, api_key: str | None = None, timeout: float = 120.0) -> None:
        headers = {"X-Api-Key": api_key} if api_key else {}

        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, headers=headers)
        self.base_url = base_url
        self.authenticated = bool(api_key)

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        """`/health` needs no key, so a network problem and a bad credential report differently."""
        try:
            response = self._client.get("/health")
        except httpx.HTTPError as error:
            logger.error("[DNT] unreachable at %s: %s (on the VPN?)", self.base_url, error)
            return False

        # Redirects are deliberately not followed: HTTP downgrades a redirected POST to GET.
        if response.is_redirect:
            logger.error(
                "[DNT] %s redirected to %s - set DNT_BASE_URL to that URL "
                "(redirects are not followed, as they would downgrade POST to GET)",
                self.base_url, response.headers.get("location", "?"),
            )
            return False

        if response.is_error:
            logger.error(
                "[DNT] %s returned HTTP %d for /health. Every path 404ing usually means the URL "
                "reaches a gateway rather than the service - check DNT_BASE_URL, and that you are "
                "on the VPN.",
                self.base_url, response.status_code,
            )
            return False

        if not self._dependencies_are_up(response):
            return False

        try:
            keyed = self._client.get("/v1/prompts")
        except httpx.HTTPError as error:
            logger.error("[DNT] /v1/prompts unreachable at %s: %s", self.base_url, error)
            return False

        if keyed.status_code in (401, 403):
            logger.error(
                "[DNT] %d at %s - %s",
                keyed.status_code, self.base_url,
                "DNT_API_KEY is set but rejected" if self.authenticated
                else "this service requires an API key; set DNT_API_KEY",
            )
            return False

        return True

    def _dependencies_are_up(self, response: httpx.Response) -> bool:
        """A disconnected LLM gateway fails every revert; unknown fields are not a failure."""
        try:
            body = response.json()
        except ValueError:
            return True
        if not isinstance(body, dict):
            return True

        for name in ("llm_gateway", "cache"):
            status = body.get(name)
            if isinstance(status, str) and status.strip().lower() not in {"connected", "ok", "up", "healthy"}:
                logger.error(
                    "[DNT] %s reports %s = %r. Reversion runs an LLM call through that gateway, so "
                    "every batch would fail.",
                    self.base_url, name, status,
                )
                return False

        return True

    def revert(
        self,
        pairs: Sequence[Mapping[str, str]],
        *,
        batch_size: int,
        source_language: str | None = None,
        target_language: str | None = None,
    ) -> list[Reversion | None]:
        """One entry per input index; `None` is a failed batch, which leaves the denominator."""
        if not pairs:
            return []

        results: list[Reversion | None] = []
        batches = [pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)]
        logger.info("[DNT] %d segments in %d batch(es)", len(pairs), len(batches))

        for number, batch in enumerate(batches, start=1):
            try:
                returned = self._revert_batch(batch, source_language, target_language)
            except (httpx.HTTPError, ValueError) as error:
                logger.error("[DNT] batch %d/%d failed: %s", number, len(batches), error)
                results.extend([None] * len(batch))
                continue

            if len(returned) != len(batch):
                logger.warning(
                    "[DNT] batch %d returned %d segments for %d inputs - realigning by index",
                    number, len(returned), len(batch),
                )

            for i, pair in enumerate(batch):
                entry = returned[i] if i < len(returned) else None
                results.append(
                    None if entry is None else parse_reversion(entry, pair.get("target", ""))
                )

            logger.info("[DNT] batch %d/%d done", number, len(batches))

        repairs_only = sum(1 for r in results if r is not None and r.items_are_repairs_only)
        if repairs_only:
            logger.error(
                "[DNT] %d/%d segments reported only the items the service repaired, not every item "
                "it weighed. Preserved items are then missing from the denominator, so the "
                "preservation rate would measure the service's fix rate instead. Check the "
                "response against /openapi.json before trusting these numbers.",
                repairs_only, len(results),
            )

        return results

    def _revert_batch(
        self,
        batch: Sequence[Mapping[str, str]],
        source_language: str | None,
        target_language: str | None,
    ) -> list[Mapping[str, Any] | None]:
        payload: dict[str, Any] = {
            "segments": [
                {"id": pair["id"], "source": pair.get("source", ""), "target": pair.get("target", "")}
                for pair in batch
            ]
        }

        # The service detects better for knowing the pair, sent as the base code its examples use.
        options = {
            key: base_language(value)
            for key, value in (
                ("source_language", source_language),
                ("target_language", target_language),
            )
            if value
        }
        if options:
            payload["options"] = options

        response = self._client.post("/v1/revert", json=payload)
        response.raise_for_status()
        return list(response_segments(response.json()))
