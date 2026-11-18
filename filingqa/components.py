"""Build the runtime object graph (store, embedder, reranker, retriever, LLM) from config."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filingqa.answer.service import AnswerService
from filingqa.config import AppConfig
from filingqa.cost import PriceTable
from filingqa.embeddings import make_embedder
from filingqa.embeddings.base import Embedder
from filingqa.llm import make_llm
from filingqa.llm.base import LLM
from filingqa.retrieval.rerank import Reranker, make_reranker
from filingqa.retrieval.retriever import Retriever
from filingqa.store.db import connect, make_pool
from filingqa.store.repository import ChunkStore


@dataclass
class Components:
    cfg: AppConfig
    store: ChunkStore
    embedder: Embedder
    reranker: Reranker | None
    retriever: Retriever
    llm: LLM
    prices: PriceTable
    service: AnswerService
    conn: Any = None

    def close(self) -> None:
        if self.conn is not None:
            self.conn.close()


def load_prices(cfg: AppConfig) -> PriceTable:
    p = Path(cfg.prices_file)
    return PriceTable.load(p) if p.is_file() else PriceTable()


def build_components(cfg: AppConfig, *, pooled: bool = False, need_reranker: bool = True) -> Components:
    conn = make_pool(cfg.database.url) if pooled else connect(cfg.database.url)
    store = ChunkStore(conn)
    embedder = make_embedder(cfg.embedding)
    reranker = make_reranker(cfg.reranker) if need_reranker else None
    prices = load_prices(cfg)
    retriever = Retriever(store, embedder, reranker, cfg.retrieval, prices)
    llm = make_llm(cfg.llm)
    service = AnswerService(retriever, llm, cfg.refusal, prices)
    return Components(cfg, store, embedder, reranker, retriever, llm, prices, service, conn)
