"""Runtime configuration, read from `.env`."""

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv


def _settings_file() -> str | None:
    """`ENV_FILE` points a run at a settings file other than `.env` - one per component, say, so a
    TM run needs no edit to the file a glossary run reads. None leaves dotenv to find `.env` itself.

    A named file that is not there stops the run rather than falling back to `.env`: the fallback
    would score the run against settings the operator did not ask for, and every typed setting
    would then fail one at a time as though it were unset."""
    named = (os.getenv("ENV_FILE") or "").strip()
    if not named:
        return None
    if not os.path.isfile(named):
        raise RuntimeError(f"ENV_FILE names {named}, which is not a file.")
    return named


load_dotenv(_settings_file(), override=True)


def parse_list(raw: str) -> list[str]:
    parts = raw.strip().strip("[]").split(",")
    return [item for item in (part.strip().strip("\"'") for part in parts) if item]


def _required(name: str) -> str:
    """A typed setting has no fallback: unset stops the run rather than scoring against a guess."""
    value = os.getenv(name) or ""
    if not value.strip():
        raise RuntimeError(f"{name} is not set in .env.")
    return value.strip()


def _env(name: str) -> Any:
    return field(default_factory=lambda: os.getenv(name))


def _env_list(name: str) -> Any:
    return field(default_factory=lambda: parse_list(os.getenv(name) or ""))




def _env_bool(name: str) -> Any:
    return field(default_factory=lambda: _required(name).lower() in {"1", "true", "yes", "on"})


PATH_VARIABLES = {"glossary": "GLOSSARY_PATH", "dnt": "DNT_PATH"}


@dataclass(frozen=True)
class PostMtConfig:
    base_url: str = _env("POSTMT_BASE_URL")
    poll_interval: float = 3.0
    timeout: float = 30 * 60.0
    api_key: str | None = _env("POSTMT_API_KEY")


@dataclass(frozen=True)
class StanzaConfig:
    base_url: str = _env("STANZA_BASE_URL")
    timeout: float = 120.0


@dataclass(frozen=True)
class SearchEngineConfig:
    node: str = _env("SEARCH_ENGINE_URL")
    username: str = _env("SEARCH_ENGINE_USERNAME")
    password: str = _env("SEARCH_ENGINE_PASSWORD")
    timeout: float = 120.0
    # AWS-managed domains reject basic auth; requests must be SigV4-signed.
    aws_sigv4: bool = _env_bool("ES_AWS_SIGV4_ENABLED")
    aws_region: str | None = _env("AWS_REGION")
    aws_profile: str | None = _env("AWS_PROFILE")


@dataclass(frozen=True)
class DntConfig:
    base_url: str | None = _env("DNT_BASE_URL")
    api_key: str | None = _env("DNT_API_KEY")
    timeout: float = 120.0
    batch_size: int = 25


@dataclass
class BenchmarkConfig:
    batch_size: int = 50
    lemma_matching: bool = True
    # The components this run measures, in reporting order.
    components: list[str] = _env_list("BENCH_COMPONENT")
    # Per component: a file, or a folder to score every dataset inside it.
    paths: dict[str, str] = field(
        default_factory=lambda: {
            component: os.getenv(variable) or "" for component, variable in PATH_VARIABLES.items()
        }
    )


@dataclass
class Config:
    postmt: PostMtConfig = field(default_factory=PostMtConfig)
    stanza: StanzaConfig = field(default_factory=StanzaConfig)
    search_engine: SearchEngineConfig = field(default_factory=SearchEngineConfig)
    dnt: DntConfig = field(default_factory=DntConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
