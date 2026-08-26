"""Runtime configuration, read from the environment (optionally via a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _env(name: str, default: Any = None) -> Any:
    return field(default_factory=lambda: os.getenv(name, default))


def _env_int(name: str, default: int) -> Any:
    return field(default_factory=lambda: int(os.getenv(name, default)))


def _env_seconds(name: str, default_ms: int) -> Any:
    return field(default_factory=lambda: int(os.getenv(name, default_ms)) / 1000)


def _env_bool(name: str, default: bool) -> Any:
    return field(
        default_factory=lambda: os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}
    )


@dataclass(frozen=True)
class PostMtConfig:
    base_url: str = _env("POSTMT_BASE_URL", "http://localhost:3000")
    poll_interval: float = _env_seconds("POSTMT_POLL_INTERVAL_MS", 3000)
    timeout: float = _env_seconds("POSTMT_TIMEOUT_MS", 30 * 60 * 1000)
    api_key: str | None = _env("POSTMT_API_KEY")


@dataclass(frozen=True)
class StanzaConfig:
    base_url: str = _env("STANZA_BASE_URL", "http://localhost:8000")
    timeout: float = 120.0


@dataclass(frozen=True)
class SearchEngineConfig:
    # OPENSEARCH_URL / ELASTICSEARCH_URL are accepted as aliases; sibling services name it either way.
    node: str = field(
        default_factory=lambda: os.getenv("SEARCH_ENGINE_URL")
        or os.getenv("OPENSEARCH_URL")
        or os.getenv("ELASTICSEARCH_URL")
        or ""
    )
    username: str | None = _env("SEARCH_ENGINE_USERNAME")
    password: str | None = _env("SEARCH_ENGINE_PASSWORD")
    timeout: float = 120.0
    #: AWS-managed domains reject basic auth; requests must be SigV4-signed instead.
    aws_sigv4: bool = _env_bool("ES_AWS_SIGV4_ENABLED", False)
    aws_region: str | None = _env("AWS_REGION")
    aws_profile: str | None = _env("AWS_PROFILE")


@dataclass
class BenchmarkConfig:
    batch_size: int = _env_int("BENCH_BATCH_SIZE", 50)
    lemma_matching: bool = _env_bool("BENCH_LEMMA_MATCHING", True)
    #: The single source of what gets scored: a dataset file, or a folder to run every
    #: dataset inside. There is no command-line equivalent.
    dataset: str = _env("BENCH_DATASET", "")


@dataclass
class Config:
    postmt: PostMtConfig = field(default_factory=PostMtConfig)
    stanza: StanzaConfig = field(default_factory=StanzaConfig)
    search_engine: SearchEngineConfig = field(default_factory=SearchEngineConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)


def load_config() -> Config:
    return Config()
