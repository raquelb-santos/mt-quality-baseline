"""Runtime configuration, read from `.env`."""

import json
import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)


def _env(name: str, default: Any = None) -> Any:
    return field(default_factory=lambda: os.getenv(name, default))


def parse_list(raw: str) -> list[str]:
    value = raw.strip()

    if value.startswith("["):
        try:
            parsed = json.loads(value)
        except ValueError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        value = value.strip("[]")

    return [item for item in (part.strip().strip("\"'") for part in value.split(",")) if item]


def _env_list(name: str) -> Any:
    return field(default_factory=lambda: parse_list(os.getenv(name, "")))


def _env_bool(name: str, default: bool) -> Any:
    return field(
        default_factory=lambda: os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class PostMtConfig:
    base_url: str = _env("POSTMT_BASE_URL", "http://localhost:3000")
    poll_interval: float = 3.0
    timeout: float = 30 * 60.0
    api_key: str | None = _env("POSTMT_API_KEY")


@dataclass(frozen=True)
class StanzaConfig:
    base_url: str = _env("STANZA_BASE_URL", "http://localhost:8000")
    timeout: float = 120.0


@dataclass(frozen=True)
class SearchEngineConfig:
    node: str = _env("SEARCH_ENGINE_URL", "")
    username: str | None = _env("SEARCH_ENGINE_USERNAME")
    password: str | None = _env("SEARCH_ENGINE_PASSWORD")
    timeout: float = 120.0
    # AWS-managed domains reject basic auth; requests must be SigV4-signed.
    aws_sigv4: bool = _env_bool("ES_AWS_SIGV4_ENABLED", False)
    aws_region: str | None = _env("AWS_REGION")
    aws_profile: str | None = _env("AWS_PROFILE")


PATH_VARIABLES = {"glossary": "GLOSSARY_PATH", "dnt": "DNT_PATH"}


@dataclass(frozen=True)
class DntConfig:
    base_url: str = _env("DNT_BASE_URL", "")
    api_key: str | None = _env("DNT_API_KEY")
    timeout: float = 120.0
    batch_size: int = 25


@dataclass
class BenchmarkConfig:
    batch_size: int = 50
    lemma_matching: bool = True
    # The components this run measures, in reporting order.
    components: list[str] = _env_list("BENCH_COMPONENT")
    # A file, or a folder to score every dataset inside it.
    glossary_path: str = _env("GLOSSARY_PATH", "")
    dnt_path: str = _env("DNT_PATH", "")

    def data_path(self, component: str) -> str:
        return {"glossary": self.glossary_path, "dnt": self.dnt_path}[component]


@dataclass
class Config:
    postmt: PostMtConfig = field(default_factory=PostMtConfig)
    stanza: StanzaConfig = field(default_factory=StanzaConfig)
    search_engine: SearchEngineConfig = field(default_factory=SearchEngineConfig)
    dnt: DntConfig = field(default_factory=DntConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
