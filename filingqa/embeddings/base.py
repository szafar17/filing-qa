"""Embedding interface. Vectors are L2-normalized float32 so cosine == dot product."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class EmbeddingBatch:
    vectors: np.ndarray  # shape (n, dim)
    tokens: int = 0  # billed input tokens (0 for local models)


class Embedder(ABC):
    provider: str
    model: str  # key stored in chunk_embeddings.model
    dim: int

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> EmbeddingBatch: ...

    def embed_query(self, text: str) -> EmbeddingBatch:
        return self.embed_documents([text])


def l2_normalize(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim == 1:
        m = m[None, :]
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms
