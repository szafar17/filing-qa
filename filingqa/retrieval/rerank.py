"""Rerankers score (query, passage) pairs; scores are probabilities in [0, 1].

CrossEncoderReranker runs a sentence-transformers cross-encoder (default
cross-encoder/ms-marco-MiniLM-L-6-v2): the query and passage are encoded together, so
the model can attend across both, which is more accurate than comparing two independent
embeddings, but too slow to run over the whole corpus. Hence: retrieve ~30 candidates
cheaply, then rerank them. Raw logits are mapped to probabilities with a sigmoid so a
refusal threshold has a stable meaning.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import numpy as np

from filingqa.config import RerankerConfig
from filingqa.textutil import content_tokens


class Reranker(ABC):
    provider: str
    model: str

    @abstractmethod
    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance probability for each passage, same order as given."""


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class CrossEncoderReranker(Reranker):
    provider = "cross-encoder"

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        *,
        batch_size: int = 32,
        max_length: int = 512,
        device: str | None = None,
        ce_model: Any = None,
    ) -> None:
        if ce_model is None:
            from sentence_transformers import CrossEncoder

            ce_model = CrossEncoder(model, max_length=max_length, device=device)
        self._ce = ce_model
        self.model = model
        self.batch_size = batch_size

    def _identity_kwargs(self) -> dict[str, Any]:
        """Ask for raw logits. The keyword changed name across sentence-transformers versions."""
        try:
            params = inspect.signature(self._ce.predict).parameters
        except (TypeError, ValueError):
            return {}
        for name in ("activation_fn", "activation_fct"):
            if name in params:
                try:
                    import torch

                    return {name: torch.nn.Identity()}
                except ImportError:  # test doubles without torch
                    return {name: None}
        return {}

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [(query, p) for p in passages]
        raw = self._ce.predict(
            pairs,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            **self._identity_kwargs(),
        )
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[1] == 2:  # two-class heads: P(relevant) via softmax
            e = np.exp(arr - arr.max(axis=1, keepdims=True))
            probs = e[:, 1] / e.sum(axis=1)
        else:
            probs = _sigmoid(arr.reshape(len(pairs)))
        return [float(p) for p in probs]


class LexicalReranker(Reranker):
    """Offline, deterministic stand-in for the cross-encoder (tests/CI only).

    score = coverage * (0.6 + 0.25 * bm25_norm + 0.15 * bigram_share), in [0, 1], where
      coverage     = IDF-weighted share of query terms present in the passage (IDF over the
                     candidate set, so a term every candidate shares counts for little and a
                     term no candidate has, e.g. "2030", pulls every score down),
      bm25_norm    = BM25 of the passage within the candidate set / best candidate's BM25,
      bigram_share = share of the query's adjacent word pairs found in the passage.
    """

    provider = "lexical"

    def __init__(self, model: str = "lexical-overlap", k1: float = 1.2, b: float = 0.75) -> None:
        self.model = model
        self.k1, self.b = k1, b

    def score(self, query: str, passages: list[str]) -> list[float]:
        q_seq = content_tokens(query)
        q = set(q_seq)
        if not q or not passages:
            return [0.0] * len(passages)
        q_bigrams = set(zip(q_seq, q_seq[1:], strict=False))
        docs = [content_tokens(p) for p in passages]
        n = len(docs)
        avgdl = (sum(len(d) for d in docs) / n) or 1.0
        df = Counter(t for d in docs for t in set(d) if t in q)
        w_cov = {t: math.log(1 + n / (1 + df[t])) for t in q}
        idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in q}
        total_w = sum(w_cov.values()) or 1.0
        covs, bms, bigs = [], [], []
        for d in docs:
            tf = Counter(d)
            covs.append(sum(w_cov[t] for t in q if tf[t]) / total_w)
            norm = self.k1 * (1 - self.b + self.b * len(d) / avgdl)
            bms.append(sum(idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in q if tf[t]))
            d_bigrams = set(zip(d, d[1:], strict=False))
            bigs.append(len(q_bigrams & d_bigrams) / len(q_bigrams) if q_bigrams else 0.0)
        top = max(bms) or 1.0
        return [c * (0.6 + 0.25 * (bm / top) + 0.15 * bg) for c, bm, bg in zip(covs, bms, bigs, strict=True)]


def make_reranker(cfg: RerankerConfig) -> Reranker | None:
    if cfg.provider == "none":
        return None
    if cfg.provider == "lexical":
        return LexicalReranker(cfg.model)
    if cfg.provider == "cross-encoder":
        return CrossEncoderReranker(cfg.model, batch_size=cfg.batch_size, max_length=cfg.max_length)
    raise ValueError(f"unknown reranker provider {cfg.provider}")
