"""Deterministic feature-hashing embedder for tests and CI (no downloads, no API).

Each text becomes a bag of stemmed unigrams and bigrams hashed into ``dim`` buckets
with a sign bit (the "hashing trick"), weighted by 1 + log(tf), then L2-normalized.
It captures lexical overlap only, so it is a stand-in for a real embedding model,
never a substitute when measuring retrieval quality.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter

import numpy as np

from filingqa.embeddings.base import Embedder, EmbeddingBatch, l2_normalize
from filingqa.textutil import content_tokens


class HashingEmbedder(Embedder):
    provider = "hashing"

    def __init__(self, dim: int = 512) -> None:
        self.dim = int(dim)
        self.model = f"hashing-{self.dim}"

    def _vector(self, text: str) -> np.ndarray:
        toks = content_tokens(text)
        feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:], strict=False)]
        v = np.zeros(self.dim, dtype=np.float32)
        for feat, count in Counter(feats).items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
            sign = 1.0 if (h >> 63) & 1 else -1.0
            v[h % self.dim] += sign * (1.0 + math.log(count))
        return v

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, self.dim), dtype=np.float32), 0)
        return EmbeddingBatch(l2_normalize(np.stack([self._vector(t) for t in texts])), 0)
