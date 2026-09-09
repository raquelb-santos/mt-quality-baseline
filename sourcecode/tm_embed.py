"""The embedding model behind the semantic band, called through the LiteLLM proxy."""

import logging
from typing import Sequence

logger = logging.getLogger(__name__)


class TmEmbedder:
    """An `Embedder`: texts in, one vector each, in the order they were given. Vectors are cached
    per text because one version's text is embedded again for every candidate it is tested against."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self._cache: dict[str, list[float]] = {}

    def ping(self) -> bool:
        try:
            self._embed(['ping'])
            return True
        except Exception as error:
            logger.error('[TM] cannot embed with %s at %s: %s', self.model, self.base_url, error)
            return False

    def __call__(self, texts: Sequence[str]) -> list[list[float]]:
        wanted = [str(text) for text in texts]
        missing = [text for text in dict.fromkeys(wanted) if text not in self._cache]
        if missing:
            self._cache.update(zip(missing, self._embed(missing)))
        return [self._cache[text] for text in wanted]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from litellm import embedding
        except ImportError as error:  # pragma: no cover - depends on the install extra
            raise RuntimeError(
                "The semantic band needs litellm: pip install 'mt-quality-baseline[semantic]'"
            ) from error

        response = embedding(
            model=self.model,
            input=list(texts),
            api_key=self.api_key,
            api_base=self.base_url,
            timeout=self.timeout,
            num_retries=2,
        )
        # The proxy is not guaranteed to answer in order; `index` is what pairs a vector to its text.
        data = sorted(response['data'], key=lambda item: item.get('index', 0))
        vectors = [[float(value) for value in item.get('embedding') or []] for item in data]
        if len(vectors) != len(texts) or not all(vectors):
            raise ValueError(
                f'{self.model} returned {len(vectors)} usable vectors for {len(texts)} texts'
            )
        return vectors
