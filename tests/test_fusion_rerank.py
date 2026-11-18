from __future__ import annotations

import importlib.util
import math

import numpy as np
import pytest

from filingqa.retrieval.fusion import reciprocal_rank_fusion
from filingqa.retrieval.rerank import CrossEncoderReranker, LexicalReranker, Reranker

# --- RRF ---------------------------------------------------------------------------------


def test_rrf_matches_hand_computation():
    vector = ["a", "b", "c"]
    bm25 = ["c", "a", "d"]
    fused = dict(reciprocal_rank_fusion([vector, bm25], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)
    order = [doc for doc, _ in reciprocal_rank_fusion([vector, bm25], k=60)]
    assert order == ["a", "c", "b", "d"]


def test_rrf_weights_and_duplicates():
    fused = dict(reciprocal_rank_fusion([["x", "x", "y"], ["y"]], k=1, weights=[2.0, 1.0]))
    assert fused["x"] == pytest.approx(2.0 / 2)  # duplicate "x" counted once
    assert fused["y"] == pytest.approx(2.0 / 4 + 1.0 / 2)
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_rrf_ties_are_deterministic():
    a = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    b = reciprocal_rank_fusion([["q", "p"], ["p", "q"]])
    assert [d for d, _ in a] == [d for d, _ in b]


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []]) == []


# --- rerankers ---------------------------------------------------------------------------


class FakeCrossEncoder:
    """Mimics sentence_transformers.CrossEncoder.predict's signature."""

    def __init__(self, logits, two_class: bool = False):
        self.logits = logits
        self.two_class = two_class
        self.calls = []

    def predict(
        self, inputs, batch_size=32, show_progress_bar=None, activation_fn=None, convert_to_numpy=True, **kw
    ):
        self.calls.append({"n": len(inputs), "activation_fn": activation_fn, "batch_size": batch_size})
        out = np.array(self.logits[: len(inputs)], dtype=float)
        return np.stack([-out, out], axis=1) if self.two_class else out


def test_cross_encoder_maps_logits_through_sigmoid():
    ce = FakeCrossEncoder([2.0, -1.0, 0.0])
    rr = CrossEncoderReranker("fake", ce_model=ce, batch_size=8)
    scores = rr.score("q", ["p1", "p2", "p3"])
    assert scores == pytest.approx([1 / (1 + math.exp(-2)), 1 / (1 + math.exp(1)), 0.5])
    assert ce.calls[0]["n"] == 3 and ce.calls[0]["batch_size"] == 8
    # with torch available we ask for raw logits (identity activation) so the sigmoid is
    # applied exactly once; without torch (no real CrossEncoder possible) nothing is passed
    if importlib.util.find_spec("torch") is not None:
        assert type(ce.calls[0]["activation_fn"]).__name__ == "Identity"
    else:
        assert ce.calls[0]["activation_fn"] is None


def test_cross_encoder_two_class_head_uses_softmax():
    rr = CrossEncoderReranker("fake", ce_model=FakeCrossEncoder([1.0], two_class=True))
    (p,) = rr.score("q", ["p"])
    assert p == pytest.approx(math.exp(1) / (math.exp(1) + math.exp(-1)))


def test_cross_encoder_empty():
    assert CrossEncoderReranker("fake", ce_model=FakeCrossEncoder([])).score("q", []) == []


def test_lexical_reranker_prefers_passages_covering_rare_query_terms():
    rr: Reranker = LexicalReranker()
    passages = [
        "Total revenue increased 13.6% to $1,284.6 million in 2024.",
        "Our employees are not represented by a labor union.",
        "Revenue recognition policies are described in Note 2.",
    ]
    scores = rr.score("What was total revenue in 2024?", passages)
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] == max(scores)
    assert scores[1] == min(scores)


def test_lexical_reranker_penalizes_missing_terms():
    rr = LexicalReranker()
    passages = ["Acme revenue was $10 million in fiscal 2024.", "Acme revenue grew."]
    full = rr.score("Acme revenue fiscal 2024", passages)[0]
    missing = rr.score("Acme revenue forecast fiscal 2030", passages)[0]
    assert missing < full


def test_real_sentence_transformers_cross_encoder_api(tmp_path):
    """Runs the real CrossEncoder class on a tiny randomly initialised local model, so the
    adapter is checked against the installed library's API (no download needed)."""
    st = pytest.importorskip("sentence_transformers")
    transformers = pytest.importorskip("transformers")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "revenue", "total", "2024", "the", "was"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        num_labels=1,
    )
    model = transformers.BertForSequenceClassification(config)
    model.save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    ce = st.CrossEncoder(str(tmp_path), max_length=32, device="cpu")
    scores = CrossEncoderReranker(str(tmp_path), ce_model=ce).score(
        "total revenue 2024", ["the revenue was 2024", "the"]
    )
    assert len(scores) == 2 and all(0.0 < s < 1.0 for s in scores)


def test_real_sentence_transformers_embedder_api(tmp_path):
    st = pytest.importorskip("sentence_transformers")
    transformers = pytest.importorskip("transformers")
    from filingqa.embeddings.local import SentenceTransformerEmbedder

    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "revenue", "total", "2024"]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab))
    tok = transformers.BertTokenizerFast(vocab_file=str(tmp_path / "vocab.txt"))
    config = transformers.BertConfig(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
    )
    transformers.BertModel(config).save_pretrained(tmp_path)
    tok.save_pretrained(tmp_path)
    # A plain transformers checkpoint: SentenceTransformer adds mean pooling on load.
    model = st.SentenceTransformer(str(tmp_path), device="cpu")
    emb = SentenceTransformerEmbedder("tiny", st_model=model, query_prefix="q: ")
    assert emb.dim == 16
    docs = emb.embed_documents(["total revenue", "2024"])
    assert docs.vectors.shape == (2, 16)
    assert np.allclose(np.linalg.norm(docs.vectors, axis=1), 1.0, atol=1e-5)
    assert emb.embed_query("revenue").vectors.shape == (1, 16)
