"""The OpenSearch/Elasticsearch client the glossary queries through."""

import json
import logging
from typing import Any, Iterable

import httpx

logger = logging.getLogger(__name__)


class SigV4Auth(httpx.Auth):
    """Sign requests for an AWS-managed OpenSearch/Elasticsearch domain."""

    requires_request_body = True

    def __init__(self, region: str, profile: str | None = None) -> None:
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

    def auth_flow(self, request: httpx.Request):
        from botocore.auth import SigV4Auth as _SigV4Auth
        from botocore.awsrequest import AWSRequest

        # Sign a copy carrying only the signed headers, then copy botocore's result back on.
        signable = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers={"Host": request.url.netloc.decode("ascii")},
        )
        if "content-type" in request.headers:
            signable.headers["Content-Type"] = request.headers["content-type"]

        _SigV4Auth(self._credentials.get_frozen_credentials(), "es", self.region).add_auth(signable)
        for header, value in signable.headers.items():
            request.headers[header] = value
        yield request


class SearchClient:
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

        self.node = node
        self._client = httpx.Client(base_url=node.rstrip("/"), timeout=timeout, auth=auth)

    def close(self) -> None:
        self._client.close()

    def ping(self) -> bool:
        try:
            self._client.get("/").raise_for_status()
            return True
        except httpx.HTTPError as error:
            logger.error("[SEARCH] cannot reach search engine at %s: %s", self.node, error)
            return False

    def search(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(f"/{index}/_search", json=body)
        response.raise_for_status()
        return response.json()

    def count(self, index: str, body: dict[str, Any]) -> int:
        response = self._client.post(f"/{index}/_count", json=body)
        response.raise_for_status()
        return int(response.json().get("count", 0))

    def msearch(self, index: str, bodies: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        lines: list[str] = []
        for body in bodies:
            lines.append(json.dumps({"index": index}))
            lines.append(json.dumps(body))

        response = self._client.post(
            "/_msearch",
            content=("\n".join(lines) + "\n").encode("utf-8"),
            headers={"Content-Type": "application/x-ndjson"},
        )
        response.raise_for_status()
        return response.json().get("responses", [])

