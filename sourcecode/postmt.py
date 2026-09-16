"""Clients for the post-mt async workflow API (``/api/workflow/async``) and for Stanza."""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import httpx

logger = logging.getLogger(__name__)


def extract_post_edited(segment: dict[str, Any]) -> str:
    """The fallback makes a failed APE look untouched; `segment_error` tells them apart."""
    ape_results = segment.get("ape_results") or {}
    return ape_results.get("text") or segment.get("aped_text") or segment.get("target_content") or ""


def segment_id(returned: dict[str, Any], original: dict[str, Any], index: int) -> str:
    """post-mt echoes the id back, but a realigned segment may arrive without one."""
    return str(returned.get("source_segment_id") or original.get("source_segment_id") or index)


def reported_has_glossary(segment: dict[str, Any]) -> bool | None:
    """Read at the top level, beside ``aqe_results``; None means the task ran no AQE step."""
    return segment.get("has_glossary")


def segment_error(segment: dict[str, Any]) -> str | None:
    """Errors are per segment, not per task: ``task["error"]`` stays null and the status "done"."""
    for key in ("ape_results", "aqe_results"):
        error = (segment.get(key) or {}).get("error")
        if error:
            return str(error)
    return None


class PostMtError(RuntimeError):
    pass


SUPPORTED_CAT_TOOLS = {"memsource", "phrase", "xtm"}


def preflight_submission(parameters: dict[str, Any]) -> list[str]:
    """Reasons post-mt would reject the task outright, whatever is being measured."""
    return [
        f"`{name}` is missing — post-mt rejects every segment with "
        f"'Missing required parameters fields: {name}' and returns no APE text"
        for name in ("tempo_task_id", "cat_project_id")
        if not str(parameters.get(name) or "").strip()
    ]


def preflight_parameters(parameters: dict[str, Any]) -> list[str]:
    """Reasons post-mt would reject the task or silently retrieve no glossary, at full LLM cost."""
    problems = preflight_submission(parameters)

    provider = str(parameters.get("cat_tool_provider") or "").strip()
    if not provider:
        problems.append("`cat_tool_provider` is missing — post-mt skips glossary retrieval entirely")
    elif provider.lower() not in SUPPORTED_CAT_TOOLS:
        problems.append(
            f"`cat_tool_provider` is {provider!r}, which post-mt does not support "
            f"(expected one of: {', '.join(sorted(SUPPORTED_CAT_TOOLS))})"
        )

    return problems


def preflight_tasks(tasks: Sequence[Any], check: Callable[[dict[str, Any]], list[str]]) -> list[str]:
    """Each task is submitted on its own parameters, so each is checked and named once."""
    counts: dict[str, int] = {}
    for task in tasks:
        for problem in check(task.parameters):
            counts[problem] = counts.get(problem, 0) + 1

    return [
        problem if count == len(tasks) else f"{problem} - in {count} of {len(tasks)} tasks"
        for problem, count in counts.items()
    ]


def raise_for_preflight(problems: list[str], message: str) -> None:
    if problems:
        for problem in problems:
            logger.error("[PREFLIGHT] %s", problem)
        raise RuntimeError(message)


@dataclass(frozen=True)
class Usage:
    cost: float = 0.0
    tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            self.cost + other.cost,
            self.tokens + other.tokens,
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
        )

    @classmethod
    def from_task(cls, task: dict[str, Any]) -> Usage:
        return cls(
            cost=float(task.get("totalCost") or 0),
            tokens=int(task.get("totalTokens") or 0),
            prompt_tokens=int(task.get("totalPromptTokens") or 0),
            completion_tokens=int(task.get("totalCompletionTokens") or 0),
        )


@dataclass(frozen=True)
class RunResult:
    task_id: str
    segments: list[dict[str, Any]]
    error: str | None
    usage: Usage = field(default_factory=Usage)


class PostMtClient:
    def __init__(self, base_url: str, poll_interval: float, timeout: float, api_key: str | None) -> None:
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=120.0,
            headers={"X-API-KEY": api_key} if api_key else {},
        )
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.authenticated = bool(api_key)
        self.active_task_id: str | None = None

    def close(self) -> None:
        self._client.close()

    def health(self) -> bool:
        try:
            response = self._client.get("/api/workflow/async", params={"limit": 1})
        except httpx.HTTPError as error:
            logger.error("[POST-MT] unreachable at %s: %s", self.base_url, error)
            return False

        if response.is_redirect:
            logger.error(
                "[POST-MT] %s redirected to %s - set POSTMT_BASE_URL to that URL "
                "(redirects are not followed, as they would downgrade POST to GET)",
                self.base_url,
                response.headers.get("location", "?"),
            )
            return False

        if response.status_code == 401:
            logger.error(
                "[POST-MT] 401 Unauthorized at %s - %s",
                self.base_url,
                "POSTMT_API_KEY is set but rejected" if self.authenticated
                else "this instance requires an API key; set POSTMT_API_KEY",
            )
            return False

        if response.is_error:
            logger.error("[POST-MT] %s returned HTTP %d", self.base_url, response.status_code)
            return False

        return True

    def submit(
        self, *, parameters: dict[str, Any], steps: Sequence[str], segments: Sequence[dict[str, Any]]
    ) -> str:
        response = self._client.post(
            "/api/workflow/async",
            json={"parameters": parameters, "steps": list(steps), "segments": list(segments)},
        )
        response.raise_for_status()
        task_id = response.json().get("taskId")
        if not task_id:
            raise PostMtError("submit returned no taskId")
        return task_id

    def get_task(self, task_id: str) -> dict[str, Any]:
        response = self._client.get(f"/api/workflow/async/{task_id}")
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        task_id: str,
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout
        last_percent = -1

        while time.monotonic() < deadline:
            response = self._client.get(f"/api/workflow/async/{task_id}/status")
            response.raise_for_status()
            status_body = response.json()
            status = status_body.get("status")

            percent = (status_body.get("progress") or {}).get("percent")
            if percent is not None and percent != last_percent:
                last_percent = percent
                if on_progress:
                    on_progress(status_body)

            if status == "done":
                return self.get_task(task_id)
            if status in {"failed", "canceled"}:
                task = self.get_task(task_id)
                raise PostMtError(f"task {task_id} {status}: {task.get('error') or 'no error detail'}")

            time.sleep(self.poll_interval)

        raise PostMtError(f"task {task_id} timed out after {self.timeout}s")

    def cancel(self, task_id: str) -> bool:
        try:
            response = self._client.post(f"/api/workflow/async/{task_id}/cancel")
            if response.status_code == 400:
                return False
            response.raise_for_status()
            logger.info("[POST-MT] cancelled task %s", task_id)
            return True
        except httpx.HTTPError as error:
            logger.warning("[POST-MT] could not cancel task %s: %s", task_id, error)
            return False

    def cancel_active(self) -> bool:
        return self.cancel(self.active_task_id) if self.active_task_id else False

    def run(
        self,
        *,
        parameters: dict[str, Any],
        segments: Sequence[dict[str, Any]],
        steps: Sequence[str],
        on_progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> RunResult:
        task_id = self.submit(parameters=parameters, steps=steps, segments=segments)
        logger.info("[POST-MT] submitted task %s (%d segments, steps=%s)", task_id, len(segments), "+".join(steps))

        self.active_task_id = task_id
        try:
            task = self.wait_for_completion(task_id, on_progress)
        finally:
            self.active_task_id = None

        error = task.get("error")
        if error:
            logger.warning("[POST-MT] task %s completed with errors: %s", task_id, error)

        usage = Usage.from_task(task)
        if usage.cost:
            logger.info("[POST-MT] task %s cost $%.4f (%s tokens)", task_id, usage.cost, f"{usage.tokens:,}")

        return RunResult(task_id=task_id, segments=task.get("segments") or [], error=error, usage=usage)


class StanzaClient:
    def __init__(self, base_url: str, timeout: float, batch_size: int) -> None:
        self._client = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout)
        self._batch_size = batch_size

    def close(self) -> None:
        self._client.close()

    def lemmatize_batch_safe(self, texts: list[str], language: str) -> list[str] | None:
        """None where lemmatization is unavailable or came back misaligned, so callers skip it."""
        if not texts:
            return []

        lemmas: list[str] = []
        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            try:
                response = self._client.post(
                    "/lemmatize/batch", json={"texts": batch, "language": language}
                )
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as error:
                logger.warning("[STANZA] lemmatization unavailable for %s: %s", language, error)
                return None

            returned = payload.get("lemmatized_texts") or [] if isinstance(payload, dict) else payload or []
            if len(returned) != len(batch):
                logger.warning(
                    "[STANZA] returned %d lemmas for %d texts (%s) - ignoring",
                    len(returned), len(batch), language,
                )
                return None
            lemmas.extend(returned)

        return lemmas
