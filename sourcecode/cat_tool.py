"""The CAT tools, which say which term bases a project has and what terms are in them."""

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from typing import Any
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)


def clients_by_provider(phrase: Any, xtm: Any) -> dict[str, Any]:
    return {name: client for name, client in (("memsource", phrase), ("xtm", xtm)) if client is not None}


@dataclass(frozen=True)
class Term:
    """One term in one language, tied to the concept its translations share."""

    concept_id: str
    language: str
    text: str
    forbidden: bool = False


def read_tbx(content: bytes, term_base_id: str) -> list[Term]:
    """The terms of a TBX export, from Phrase's `tig` or XTM's `ntig/termGrp` layout."""
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as error:
        raise RuntimeError(f"term base {term_base_id} did not export valid TBX: {error}") from error

    found: list[Term] = []
    for position, entry in enumerate(root.iter("termEntry")):
        # Phrase names the concept in a descrip; XTM names none, so the entry is the concept.
        concept_id = (entry.findtext('descrip[@type="conceptId"]') or entry.get("id")
                      or f"{term_base_id}#{position}")
        for language_set in entry.iter("langSet"):
            language = language_set.get("{http://www.w3.org/XML/1998/namespace}lang") or ""
            for group in [*language_set.iter("tig"), *language_set.iter("ntig")]:
                text = (group.findtext(".//term") or "").strip()
                notes = {note.get("type"): (note.text or "").strip().lower()
                         for note in group.iter("termNote")}
                forbidden = notes.get("forbidden") == "true" or notes.get("status") == "forbidden"
                if language and text:
                    found.append(Term(concept_id, language, text, forbidden))

    return found


class _TokenClient:
    """A token that outlives one project, so it is re-earned only once it is refused."""

    tool: str
    scheme: str

    def __init__(self, base_url: str, timeout: float) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._token: str | None = None

    def close(self) -> None:
        self._client.close()

    def _fetch(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in (1, 2):
            if self._token is None:
                self._token = self._authenticate()

            response = self._client.request(
                method, path, headers={"Authorization": f"{self.scheme} {self._token}"}, **kwargs
            )
            if response.status_code == 401 and attempt == 1:
                self._token = None
                continue

            response.raise_for_status()
            return response

        raise RuntimeError(f"{self.tool} refused the token twice for {path}.")


class PhraseClient(_TokenClient):
    """A token from `v3/auth/login`, sent back as `ApiToken`."""

    tool, scheme = "Phrase", "ApiToken"

    def __init__(self, base_url: str, username: str, password: str, timeout: float = 120.0) -> None:
        super().__init__(f"{base_url.rstrip('/')}/web/api2", timeout)
        self._username = username
        self._password = password

    def _authenticate(self) -> str:
        response = self._client.post(
            "/v3/auth/login", json={"userName": self._username, "password": self._password}
        )
        response.raise_for_status()
        token = (response.json() or {}).get("token")
        if not token:
            raise RuntimeError("Phrase returned no token; check PHRASE_USERNAME and PHRASE_PASSWORD.")
        return str(token)

    def term_base_ids(self, project_id: str) -> list[str]:
        body = self._fetch("GET", f"/v1/projects/{project_id}/termBases").json() or {}
        return [
            uid for entry in (body.get("termBases") or [])
            if (uid := str(((entry or {}).get("termBase") or {}).get("uid") or ""))
        ]

    def terms(self, term_base_id: str) -> list[Term]:
        return read_tbx(self._fetch("GET", f"/v1/termBases/{term_base_id}/export").content, term_base_id)


class XtmClient(_TokenClient):
    """A token from `/auth/token`, sent back as `XTM-Basic`; term bases are `termCustomerIds`."""

    tool, scheme = "XTM", "XTM-Basic"

    def __init__(
        self, base_url: str, client: str, user_id: str, password: str, timeout: float = 120.0
    ) -> None:
        super().__init__(f"{base_url.rstrip('/')}/project-manager-api-rest", timeout)
        self._name = client
        self._user_id = user_id
        self._password = password

    def _authenticate(self) -> str:
        response = self._client.post(
            "/auth/token",
            json={"client": self._name, "userId": int(self._user_id), "password": self._password},
        )
        response.raise_for_status()
        token = (response.json() or {}).get("token")
        if not token:
            raise RuntimeError("XTM returned no token; check XTM_CLIENT, XTM_USER_ID and XTM_PASSWORD.")
        return str(token)

    def term_base_ids(self, project_id: str) -> list[str]:
        body = self._fetch("GET", f"/projects/{project_id}").json()
        # The endpoint answers with the project, or with a list holding it.
        project = body[0] if isinstance(body, list) and body else body
        if not isinstance(project, dict):
            return []
        return [str(identifier) for identifier in (project.get("termCustomerIds") or [])]

    def terms(self, term_base_id: str) -> list[Term]:
        """XTM serves no term listing, so the customer's terminology is exported as TBX."""
        export = self._fetch("POST", "/terminology/files/export", json={
            "fileExtensionType": "TBX", "filter": {"customerIds": [int(term_base_id)]},
        }).json() or {}
        file_id = export.get("fileId")
        if not file_id:
            raise RuntimeError(f"XTM started no terminology export for customer {term_base_id}.")

        deadline = time.monotonic() + 300
        while (status := (self._fetch("GET", f"/terminology/files/export/{file_id}/status").json()
                          or {}).get("status")) != "FINISHED":
            if status == "ERROR" or time.monotonic() > deadline:
                raise RuntimeError(f"XTM terminology export {file_id} ended {status or 'without a status'}.")
            time.sleep(3)

        archive = self._fetch("GET", f"/terminology/files/export/{file_id}/download").content
        try:
            with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
                name = next(name for name in bundle.namelist() if name.lower().endswith(".tbx"))
                return read_tbx(bundle.read(name), term_base_id)
        except (zipfile.BadZipFile, StopIteration) as error:
            raise RuntimeError(f"XTM terminology export {file_id} holds no TBX file.") from error


class TermBaseResolver:
    """Which term bases a CAT project has attached, so retrieval reads the same ones post-mt does."""

    def __init__(self, phrase: Any = None, xtm: Any = None) -> None:
        self._clients = clients_by_provider(phrase, xtm)
        self._cache: dict[tuple[str, str], list[str]] = {}

    def close(self) -> None:
        for client in dict.fromkeys(self._clients.values()):
            client.close()

    def ids_for(self, project_id: Any, provider: Any) -> list[str]:
        """The ids, or none — which tells the caller to skip the glossary step for this task."""
        if not project_id or not provider:
            return []

        provider = str(provider).strip().lower()
        client = self._clients.get(provider)
        if client is None:
            logger.error(
                "[TERMBASE] %s for project %s: %s", provider, project_id,
                "post-mt has no term-base lookup for this CAT tool"
                if provider not in ("memsource", "xtm")
                else "no credentials configured for this CAT tool",
            )
            return []

        key = (provider, str(project_id))
        if key in self._cache:
            return self._cache[key]

        try:
            ids = list(dict.fromkeys(identifier for identifier in client.term_base_ids(str(project_id)) if identifier))
        except (httpx.HTTPError, RuntimeError, ValueError) as error:
            logger.error("[TERMBASE] %s could not list the term bases of %s: %s", provider, project_id, error)
            return []

        # Only a non-empty answer is kept, so a transient failure cannot settle the run's retrieval.
        if ids:
            self._cache[key] = ids
        else:
            logger.warning("[TERMBASE] %s project %s has no term base attached", provider, project_id)

        return ids
