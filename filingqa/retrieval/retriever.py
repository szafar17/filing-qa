"""Retrieval modes, selectable per call so the eval can compare them on equal footing.

vector         pgvector cosine search (baseline)
bm25           Okapi BM25 over the Postgres inverted index
hybrid         both, fused with Reciprocal Rank Fusion
hybrid_rerank  hybrid candidates rescored by a cross-encoder
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from filingqa.config import ALL_MODES, RetrievalConfig
from filingqa.cost import PriceTable
from filingqa.embeddings.base import Embedder
from filingqa.retrieval.fusion import reciprocal_rank_fusion
from filingqa.retrieval.rerank import Reranker
from filingqa.store.repository import ChunkRecord, ChunkStore, SearchFilters
from filingqa.trace import Trace


@dataclass
class QueryEmbedding:
    vector: np.ndarray
    ms: float
    tokens: int


@dataclass
class Hit:
    chunk: ChunkRecord
    rank: int
    score: float
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    mode: str
    hits: list[Hit]
    timings_ms: dict[str, float]
    embed_tokens: int = 0
    evidence: float | None = None  # best rerank probability or best cosine similarity
    evidence_kind: str | None = None  # "rerank" | "vector" | "bm25" | None

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())


class Retriever:
    def __init__(
        self,
        store: ChunkStore,
        embedder: Embedder | None,
        reranker: Reranker | None,
        cfg: RetrievalConfig,
        prices: PriceTable | None = None,
    ) -> None:
        self.store = store
        self.embedder = embedder
        self.reranker = reranker
        self.cfg = cfg
        self.prices = prices or PriceTable()

    def embed_query(self, query: str) -> QueryEmbedding:
        if self.embedder is None:
            raise RuntimeError("vector retrieval needs an embedder")
        t0 = time.perf_counter()
        batch = self.embedder.embed_query(query)
        return QueryEmbedding(batch.vectors[0], (time.perf_counter() - t0) * 1000, batch.tokens)

    def retrieve(
        self,
        query: str,
        *,
        mode: str | None = None,
        k: int | None = None,
        filters: SearchFilters | None = None,
        query_embedding: QueryEmbedding | None = None,
        trace: Trace | None = None,
    ) -> RetrievalResult:
        mode = mode or self.cfg.mode
        if mode not in ALL_MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; choose from {ALL_MODES}")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank mode needs a reranker (reranker.provider is 'none')")
        k = k or self.cfg.top_k
        timings: dict[str, float] = {}
        tokens = 0
        qe: QueryEmbedding | None = None
        vec: list[tuple[int, float]] = []
        bm: list[tuple[int, float]] = []
        n_cand = max(k, self.cfg.candidates)

        def timed(name: str, fn):  # noqa: ANN001, ANN202
            t0 = time.perf_counter()
            out = fn()
            timings[name] = timings.get(name, 0.0) + (time.perf_counter() - t0) * 1000
            return out

        if mode in ("vector", "hybrid", "hybrid_rerank"):
            qe = query_embedding or self.embed_query(query)
            timings["embed_query"] = qe.ms
            tokens = qe.tokens
            assert self.embedder is not None
            emb = self.embedder
            vec = timed(
                "vector_search",
                lambda: self.store.vector_search(
                    emb.model,
                    emb.dim,
                    qe.vector,
                    k if mode == "vector" else n_cand,
                    filters,
                    self.cfg.hnsw_ef_search,
                ),
            )
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            bm = timed(
                "bm25_search",
                lambda: self.store.bm25_search(
                    query,
                    k if mode == "bm25" else n_cand,
                    filters,
                    self.cfg.bm25_k1,
                    self.cfg.bm25_b,
                ),
            )

        vec_scores = dict(vec)
        bm_scores = dict(bm)
        if mode == "vector":
            ranked = vec[:k]
            rank_scores = {cid: {"vector": s} for cid, s in ranked}
        elif mode == "bm25":
            ranked = bm[:k]
            rank_scores = {cid: {"bm25": s} for cid, s in ranked}
        else:
            fused = timed(
                "fuse",
                lambda: reciprocal_rank_fusion([[c for c, _ in vec], [c for c, _ in bm]], k=self.cfg.rrf_k),
            )
            pool = fused[: (self.cfg.rerank_candidates if mode == "hybrid_rerank" else k)]
            ranked = [(int(cid), s) for cid, s in pool]
            rank_scores = {}
            for cid, s in ranked:
                sc = {"rrf": s}
                if cid in vec_scores:
                    sc["vector"] = vec_scores[cid]
                if cid in bm_scores:
                    sc["bm25"] = bm_scores[cid]
                rank_scores[cid] = sc

        ids = [cid for cid, _ in ranked]
        records = timed("fetch", lambda: self.store.get_chunks(ids))
        ranked = [(cid, s) for cid, s in ranked if cid in records]

        if mode == "hybrid_rerank":
            assert self.reranker is not None
            rr = self.reranker
            passages = [records[cid].display_text for cid, _ in ranked]
            probs = timed("rerank", lambda: rr.score(query, passages))
            order = sorted(range(len(ranked)), key=lambda i: (-probs[i], i))
            new_ranked = []
            for i in order[:k]:
                cid = ranked[i][0]
                rank_scores[cid]["rerank"] = probs[i]
                new_ranked.append((cid, probs[i]))
            ranked = new_ranked

        hits = [
            Hit(chunk=records[cid], rank=r, score=s, scores=rank_scores.get(cid, {}))
            for r, (cid, s) in enumerate(ranked, start=1)
        ]

        evidence, kind = None, None
        if mode == "hybrid_rerank":
            evidence, kind = (hits[0].score if hits else 0.0), "rerank"
        elif mode in ("vector", "hybrid"):
            evidence, kind = (max(vec_scores.values()) if vec_scores else 0.0), "vector"
        elif mode == "bm25" and bm:
            evidence, kind = bm[0][1], "bm25"

        if trace is not None:
            for name, ms in timings.items():
                trace.add_stage(name, ms)
            if qe is not None and self.embedder is not None:
                trace.add_usage(
                    "embed_query",
                    model=self.embedder.model,
                    provider=self.embedder.provider,
                    input_tokens=tokens,
                    cost_usd=self.prices.cost(
                        getattr(self.embedder, "model_name", self.embedder.model),
                        tokens,
                        provider=self.embedder.provider,
                    ),
                )
        return RetrievalResult(
            mode=mode,
            hits=hits,
            timings_ms=timings,
            embed_tokens=tokens,
            evidence=evidence,
            evidence_kind=kind,
        )
