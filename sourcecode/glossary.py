"""Glossary resolution against the term-bases index, sending the same queries post-mt sends."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import httpx

from .normalize import language_variants

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlossaryMatches:
    """``per_text_mappings[i]`` corresponds to ``texts[i]``."""
    mappings: list[dict[str, str]]
    per_text_mappings: list[list[dict[str, str]]]


def as_id_list(glossary_ids: str | Sequence[str]) -> list[str]:
    if isinstance(glossary_ids, str):
        return [part.strip() for part in glossary_ids.split(",") if part.strip()]
    return [str(gid).strip() for gid in glossary_ids if str(gid).strip()]


def dedup_mappings(per_text_mappings: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    flat: list[dict[str, Any]] = []
    for mappings in per_text_mappings:
        for mapping in mappings:
            if mapping["target_content"] not in seen:
                seen.add(mapping["target_content"])
                flat.append(mapping)
    return flat


class SigV4Auth(httpx.Auth):
    """Sign requests for an AWS-managed OpenSearch/Elasticsearch domain."""

    requires_request_body = True

    def __init__(self, region: str, profile: str | None = None, service: str = "es") -> None:
        try:
            from botocore.session import Session
        except ImportError as error:  # pragma: no cover - depends on the install extra
            raise RuntimeError(
                "AWS request signing needs botocore: pip install 'mt-quality-baseline[aws]'"
            ) from error

        credentials = Session(profile=profile).get_credentials()
        if credentials is None:
            raise RuntimeError(
                f"No AWS credentials for profile {profile!r}. Run `aws sso login --profile {profile}`."
            )
        self._credentials = credentials
        self.region = region
        self.service = service

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth as _SigV4Auth
        from botocore.awsrequest import AWSRequest

        # Sign a copy carrying only the headers that are part of the signature; botocore returns
        # Authorization plus the x-amz-* headers to copy back onto the real request.
        signable = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Host": request.url.netloc.decode("ascii")},
        )
        if "content-type" in request.headers:
            signable.headers["Content-Type"] = request.headers["content-type"]

        _SigV4Auth(self._credentials.get_frozen_credentials(), self.service, self.region).add_auth(signable)
        for header, value in signable.headers.items():
            request.headers[header] = value
        yield request


class GlossaryClient:
    source_label = "term-bases-index"

    def __init__(
        self,
        node: str,
        username: str | None = None,
        password: str | None = None,
        timeout: float = 120.0,
        aws_region: str | None = None,
        aws_profile: str | None = None,
    ) -> None:
        if aws_region:
            auth: Any = SigV4Auth(aws_region, aws_profile)
        elif username and password:
            auth = (username, password)
        else:
            auth = None
        self._client = httpx.Client(base_url=node.rstrip("/"), timeout=timeout, auth=auth)
        self.node = node

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        try:
            self._client.get("/").raise_for_status()
            return True
        except httpx.HTTPError as error:
            logger.error("[GLOSSARY] cannot reach search engine at %s: %s", self.node, error)
            return False

    def _msearch(self, index: str, bodies: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        lines: list[str] = []
        for body in bodies:
            lines.append(json.dumps({"index": index}))
            lines.append(json.dumps(body))
        payload = "\n".join(lines) + "\n"

        response = self._client.post(
            "/_msearch",
            content=payload.encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        return response.json().get("responses", [])

    def count_terms(self, glossary_ids: str | Sequence[str], provider: str | None = None) -> int:
        """Documents the index holds for these ids — 0 means the ids are not in this cluster."""
        index = "xtm-term-bases" if str(provider or "").lower() == "xtm" else "term-bases"
        response = self._client.post(
            f"/{index}/_count", json={"query": {"terms": {"glossary_id": as_id_list(glossary_ids)}}}
        )
        response.raise_for_status()
        return int(response.json().get("count", 0))

    def fetch_matches(
        self,
        *,
        glossary_ids: str | Sequence[str],
        source_language: str,
        target_language: str,
        texts: Sequence[str],
        provider: str | None = None,
    ) -> GlossaryMatches:
        """Resolve glossary matches for a batch of lemmatized texts."""
        ids = as_id_list(glossary_ids)
        if not ids:
            raise ValueError("No glossary IDs provided")
        if not source_language:
            raise ValueError("No source language provided")
        if not target_language:
            raise ValueError("No target language provided")
        if not texts:
            raise ValueError("No texts provided")

        index = "xtm-term-bases" if str(provider or "").lower() == "xtm" else "term-bases"

        term_filter = [
            {"terms": {"glossary_id": ids}},
            {"terms": {"language": language_variants(source_language)}},
        ]
        bodies = [
            {
                "query": {
                    "bool": {
                        "filter": term_filter,
                        "must": [{"percolate": {"field": "query", "document": {"content": str(text)}}}],
                    }
                },
                "size": 50,
                "sort": ["_score"],
            }
            for text in texts
        ]

        per_text_source_terms: list[list[dict[str, str]]] = [[] for _ in texts]
        all_concept_ids: set[str] = set()

        for i, response in enumerate(self._msearch(index, bodies)):
            if response.get("error"):
                logger.warning("[GLOSSARY] percolate error on text %d: %s", i, response["error"])
                continue
            for hit in response.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                per_text_source_terms[i].append(
                    {"term_text": source.get("term_text"), "concept_id": source.get("concept_id")}
                )
                all_concept_ids.add(source.get("concept_id"))

        if not all_concept_ids:
            return GlossaryMatches(mappings=[], per_text_mappings=[[] for _ in texts])

        target_body = {
            "query": {
                "bool": {
                    "filter": [
                        {"terms": {"concept_id": sorted(all_concept_ids)}},
                        {"terms": {"language": language_variants(target_language)}},
                    ]
                }
            },
            "size": 1000,
        }
        response = self._client.post(f"/{index}/_search", json=target_body)
        response.raise_for_status()

        targets_by_concept: dict[str, list[str]] = {}
        for hit in response.json().get("hits", {}).get("hits", []):
            source = hit.get("_source", {})
            targets_by_concept.setdefault(source.get("concept_id"), []).append(source.get("term_text"))

        per_text_mappings: list[list[dict[str, str]]] = []
        for source_terms in per_text_source_terms:
            mappings: list[dict[str, str]] = []
            seen: set[str] = set()
            for source in source_terms:
                for target_text in targets_by_concept.get(source["concept_id"], []):
                    if target_text not in seen:
                        seen.add(target_text)
                        mappings.append({"source_content": source["term_text"], "target_content": target_text})
            per_text_mappings.append(mappings)

        return GlossaryMatches(
            mappings=dedup_mappings(per_text_mappings),
            per_text_mappings=per_text_mappings,
        )
