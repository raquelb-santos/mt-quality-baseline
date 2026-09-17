"""Runtime configuration, read from `.env`."""

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv


def _settings_file() -> str | None:
    """`ENV_FILE` names a settings file other than `.env`; a missing one stops the run."""
    named = (os.getenv("ENV_FILE") or "").strip()
    if named and not os.path.isfile(named):
        raise RuntimeError(f"ENV_FILE names {named}, which is not a file.")
    return named or None


load_dotenv(_settings_file(), override=True)


def _env(name: str) -> Any:
    return field(default_factory=lambda: os.getenv(name))


def _env_list(name: str, *, keep_blanks: bool = False) -> Any:
    """A comma-separated setting, brackets optional; `keep_blanks` keeps slots lined up."""
    def parse() -> list[str]:
        raw = (os.getenv(name) or "").strip()
        items = [part.strip().strip("\"'") for part in raw.strip("[]").split(",")] if raw else []
        return items if keep_blanks else [item for item in items if item]
    return field(default_factory=parse)


def _env_bool(name: str) -> Any:
    return field(default_factory=lambda: (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"})


PATH_VARIABLES = {"glossary": "GLOSSARY_PATH", "dnt": "DNT_PATH", "tags": "TAGS_PATH"}


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
    batch_size: int = 200


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
class PhraseConfig:
    base_url: str | None = _env("PHRASE_BASE_URL")
    username: str | None = _env("PHRASE_USERNAME")
    password: str | None = _env("PHRASE_PASSWORD")
    timeout: float = 120.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.username and self.password)


@dataclass(frozen=True)
class XtmConfig:
    base_url: str | None = _env("XTM_BASE_URL")
    client: str | None = _env("XTM_CLIENT")
    user_id: str | None = _env("XTM_USER_ID")
    password: str | None = _env("XTM_PASSWORD")
    timeout: float = 120.0

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.client and self.user_id and self.password)


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
    # One slot per component, in `components` order; a blank slot scores every language pair.
    languages: list[str] = _env_list("BENCH_LANGUAGE", keep_blanks=True)
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
    phrase: PhraseConfig = field(default_factory=PhraseConfig)
    xtm: XtmConfig = field(default_factory=XtmConfig)
    dnt: DntConfig = field(default_factory=DntConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
