"""OpenAI embeddings (default: text-embedding-3-small, 1536 dims)."""

from __future__ import annotations

from typing import Any

import numpy as np

from filingqa.embeddings.base import Embedder, EmbeddingBatch, l2_normalize
from filingqa.openai_common import make_client, retry_after_seconds, retryable_errors
from filingqa.retry import retry_call

KNOWN_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}
MAX_INPUT_CHARS = 24_000  # well under the 8,191-token input limit for normal prose


class OpenAIEmbedder(Embedder):
    provider = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        batch_size: int = 128,
        max_retries: int = 5,
        client: Any = None,
        sleep: Any = None,
    ) -> None:
        if dimensions is None and model not in KNOWN_DIMS:
            raise ValueError(f"unknown dimension for {model}; set embedding.dimensions")
        self.model_name = model
        self.dimensions = dimensions
        self.dim = int(dimensions or KNOWN_DIMS[model])
        # Different output sizes are different vector spaces, so they get different keys.
        self.model = model if dimensions is None else f"{model}@{dimensions}"
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = client
        self._sleep = sleep

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client()
        return self._client

    def _call(self, inputs: list[str]) -> tuple[np.ndarray, int]:
        def go() -> Any:
            kwargs: dict[str, Any] = {"model": self.model_name, "input": inputs}
            if self.dimensions is not None:
                kwargs["dimensions"] = self.dimensions
            return self.client.embeddings.create(**kwargs)

        extra = {"sleep": self._sleep} if self._sleep else {}
        resp = retry_call(
            go,
            retry_on=retryable_errors(),
            max_attempts=self.max_retries,
            base_delay=1.0,
            max_delay=60.0,
            retry_after=retry_after_seconds,
            **extra,
        )
        data = sorted(resp.data, key=lambda d: d.index)
        vecs = np.asarray([d.embedding for d in data], dtype=np.float32)
        usage = getattr(resp, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0)
        return vecs, tokens

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, self.dim), dtype=np.float32), 0)
        out: list[np.ndarray] = []
        tokens = 0
        for i in range(0, len(texts), self.batch_size):
            batch = [(t or " ")[:MAX_INPUT_CHARS] for t in texts[i : i + self.batch_size]]
            vecs, n = self._call(batch)
            out.append(vecs)
            tokens += n
        return EmbeddingBatch(l2_normalize(np.concatenate(out)), tokens)
