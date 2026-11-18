from __future__ import annotations

from filingqa.config import EmbeddingConfig
from filingqa.embeddings.base import Embedder, EmbeddingBatch


def make_embedder(cfg: EmbeddingConfig) -> Embedder:
    if cfg.provider == "hashing":
        from filingqa.embeddings.hashing import HashingEmbedder

        return HashingEmbedder(dim=cfg.dimensions or 512)
    if cfg.provider == "openai":
        from filingqa.embeddings.openai_embedder import OpenAIEmbedder

        return OpenAIEmbedder(
            cfg.model,
            dimensions=cfg.dimensions,
            batch_size=cfg.batch_size,
            max_retries=cfg.max_retries,
        )
    if cfg.provider == "sentence-transformers":
        from filingqa.embeddings.local import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(
            cfg.model,
            query_prefix=cfg.query_prefix,
            document_prefix=cfg.document_prefix,
            batch_size=cfg.batch_size,
        )
    raise ValueError(f"unknown embedding provider {cfg.provider}")


__all__ = ["EmbeddingBatch", "Embedder", "make_embedder"]
