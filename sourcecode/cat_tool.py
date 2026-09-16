"""The CAT tools, which say which term bases a project has and what terms are in them."""

import io
import logging
import time
import zipfile
from dataclasses import dataclass
from typing import Any, Sequence
from xml.etree import ElementTree

import httpx

logger = logging.getLogger(__name__)

PHRASE_PROVIDERS = {"memsource"}
XTM_PROVIDERS = {"xtm"}


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


class PhraseClient:
    """A token from `v3/auth/login`, sent back as `ApiToken`."""

    def __init__(self, base_url: str, username: str, password: str, timeout: float = 120.0) -> None:
        self._client = httpx.Client(base_url=f"{base_url.rstrip('/')}/web/api2", timeout=timeout)
        self._username = username
        self._password = password
        self._token: str | None = None

    def close(self) -> None:
        self._client.close()

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
        body = self._get(f"/v1/projects/{project_id}/termBases")
        return [
            uid for entry in (body.get("termBases") or [])
            if (uid := str(((entry or {}).get("termBase") or {}).get("uid") or ""))
        ]

    def terms(self, term_base_id: str) -> list[Term]:
        return read_tbx(self._fetch(f"/v1/termBases/{term_base_id}/export").content, term_base_id)

    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._fetch(path, params).json() or {}

    def _fetch(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """A token outlives one project, so it is re-earned only once it is refused."""
        for attempt in (1, 2):
            if self._token is None:
                self._token = self._authenticate()

            response = self._client.get(
                path, params=params, headers={"Authorization": f"ApiToken {self._token}"}
            )
            if response.status_code == 401 and attempt == 1:
                self._token = None
                continue

            response.raise_for_status()
            return response

        raise RuntimeError(f"Phrase refused the token twice for {path}.")


class XtmClient:
    """A token from `/auth/token`, sent back as `XTM-Basic`; term bases are `termCustomerIds`."""

    def __init__(
        self, base_url: str, client: str, user_id: str, password: str, timeout: float = 120.0
    ) -> None:
        self._client = httpx.Client(
            base_url=f"{base_url.rstrip('/')}/project-manager-api-rest", timeout=timeout
        )
        self._name = client
        self._user_id = user_id
        self._password = password
        self._token: str | None = None

    def close(self) -> None:
        self._client.close()

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
        body = self._get(f"/projects/{project_id}")
        # The endpoint answers with the project, or with a list holding it.
        project = body[0] if isinstance(body, list) and body else body
        if not isinstance(project, dict):
            return []
        return [str(identifier) for identifier in (project.get("termCustomerIds") or [])]

    def terms(self, term_base_id: str) -> list[Term]:
        """Every term the customer holds. XTM serves no term listing, so the customer's
        terminology is exported as TBX, the way post-mt exports a TM."""
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

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return self._fetch("GET", path, params=params).json()

    def _fetch(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        for attempt in (1, 2):
            if self._token is None:
                self._token = self._authenticate()

            response = self._client.request(
                method, path, headers={"Authorization": f"XTM-Basic {self._token}"}, **kwargs
            )
            if response.status_code == 401 and attempt == 1:
                self._token = None
                continue

            response.raise_for_status()
            return response

        raise RuntimeError(f"XTM refused the token twice for {path}.")


class TermBaseResolver:
    """Which term bases a CAT project has attached, so retrieval reads the same ones post-mt does."""

    def __init__(self, phrase: Any = None, xtm: Any = None) -> None:
        self._clients = {
            **{provider: phrase for provider in PHRASE_PROVIDERS if phrase is not None},
            **{provider: xtm for provider in XTM_PROVIDERS if xtm is not None},
        }
        self._cache: dict[tuple[str, str], list[str]] = {}

    def close(self) -> None:
        for client in dict.fromkeys(self._clients.values()):
            client.close()

    @property
    def providers(self) -> set[str]:
        return set(self._clients)

    def ids_for(self, project_id: Any, provider: Any) -> list[str]:
        """The ids, or none — which tells the caller to skip the glossary step for this task."""
        if not project_id or not provider:
            return []

        provider = str(provider).strip().lower()
        client = self._clients.get(provider)
        if client is None:
            known = PHRASE_PROVIDERS | XTM_PROVIDERS
            logger.error(
                "[TERMBASE] %s for project %s: %s", provider, project_id,
                "post-mt has no term-base lookup for this CAT tool" if provider not in known
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
