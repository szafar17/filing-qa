"""Local sentence-transformers embeddings (pip install -e ".[local]")."""

from __future__ import annotations

from typing import Any

import numpy as np

from filingqa.embeddings.base import Embedder, EmbeddingBatch, l2_normalize


class SentenceTransformerEmbedder(Embedder):
    provider = "sentence-transformers"

    def __init__(
        self,
        model: str = "BAAI/bge-small-en-v1.5",
        *,
        query_prefix: str = "",
        document_prefix: str = "",
        batch_size: int = 64,
        device: str | None = None,
        st_model: Any = None,
    ) -> None:
        if st_model is None:
            from sentence_transformers import SentenceTransformer

            st_model = SentenceTransformer(model, device=device)
        self._st = st_model
        self.model = model
        getter = (
            getattr(st_model, "get_embedding_dimension", None) or st_model.get_sentence_embedding_dimension
        )
        self.dim = int(getter())
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.batch_size = batch_size

    def _encode(self, texts: list[str]) -> np.ndarray:
        vecs = self._st.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return l2_normalize(np.asarray(vecs, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> EmbeddingBatch:
        if not texts:
            return EmbeddingBatch(np.zeros((0, self.dim), dtype=np.float32), 0)
        return EmbeddingBatch(self._encode([self.document_prefix + t for t in texts]), 0)

    def embed_query(self, text: str) -> EmbeddingBatch:
        return EmbeddingBatch(self._encode([self.query_prefix + text]), 0)
