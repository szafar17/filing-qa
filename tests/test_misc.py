from __future__ import annotations

import json
import logging

import numpy as np
import pytest

from filingqa.config import AppConfig, IngestConfig, load_config
from filingqa.cost import PriceTable
from filingqa.embeddings.hashing import HashingEmbedder
from filingqa.logging_utils import JsonFormatter
from filingqa.retry import retry_call
from filingqa.textutil import content_tokens, locate_quote, norm_for_match, sentence_spans
from filingqa.trace import Trace


def test_price_table_math_and_snapshots():
    prices = PriceTable.load("config/prices.yaml")
    assert prices.as_of and prices.source
    # 1,000 input + 200 output tokens on gpt-4.1-mini at $0.40 / $1.60 per 1M
    assert prices.cost("gpt-4.1-mini", 1000, 200) == pytest.approx(0.00072)
    # dated snapshot names fall back to the base model's price
    assert prices.cost("gpt-4.1-mini-2025-04-14", 1000, 200) == pytest.approx(0.00072)
    # cached input tokens are billed at the cached rate
    assert prices.cost("gpt-4.1-mini", 1000, 0, cached_input_tokens=1000) == pytest.approx(0.0001)
    assert prices.cost("text-embedding-3-small", 1_000_000) == pytest.approx(0.02)
    assert prices.cost("anything", 10_000, 10_000, provider="sentence-transformers") == 0.0


def test_unknown_models_are_reported_not_hidden():
    prices = PriceTable.load("config/prices.yaml")
    assert prices.cost("mystery-model", 1000, 1000) == 0.0
    assert prices.unpriced == {"mystery-model"}


def test_locate_quote_variants():
    text = "Net sales were $ 1,842.5 million\nin fiscal 2024 — the company’s best."
    assert locate_quote(text, "Net sales were") == (0, 14)
    s, e = locate_quote(text, "net sales were $1,842.5 million in fiscal 2024 - the company's best")
    assert text[s:e].startswith("Net sales") and text[s:e].endswith("best")
    assert locate_quote(text, "revenue") is None
    assert locate_quote(text, "   ") is None


def test_norm_for_match():
    assert norm_for_match("  A  “B” (5 )  $ 10 ") == 'a "b" (5) $10'


def test_sentence_spans_respect_abbreviations():
    text = "Acme Inc. sells robots. Revenue rose 5%. See Note 2 for U.S. details.\nNext line"
    pieces = [text[s:e] for s, e in sentence_spans(text)]
    assert pieces == [
        "Acme Inc. sells robots.",
        "Revenue rose 5%.",
        "See Note 2 for U.S. details.",
        "Next line",
    ]


def test_content_tokens():
    assert content_tokens("What were the revenues in 2024?") == ["revenue", "2024"]


def test_hashing_embedder_is_deterministic_and_lexical():
    e = HashingEmbedder(dim=256)
    a, b, c = e.embed_documents(["total revenue 2024", "total revenue 2024", "employees union"]).vectors
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)
    assert float(a @ b) > float(a @ c)
    assert e.model == "hashing-256"


def test_retry_call_only_retries_listed_errors():
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise TimeoutError
        return "ok"

    assert retry_call(flaky, retry_on=(TimeoutError,), sleep=lambda s: None) == "ok"
    with pytest.raises(KeyError):
        retry_call(lambda: (_ for _ in ()).throw(KeyError()), retry_on=(TimeoutError,), sleep=lambda s: None)


def test_trace_records_stages_usage_and_cost():
    t = Trace()
    with t.stage("rerank"):
        pass
    t.add_stage("llm", 120.0)
    t.add_usage("answer", model="m", provider="openai", input_tokens=10, output_tokens=5, cost_usd=0.001)
    d = t.to_dict()
    assert set(d["stages_ms"]) == {"rerank", "llm"} and d["cost_usd"] == pytest.approx(0.001)
    assert d["tokens"]["answer"] == {"model": "m", "input": 10, "output": 5}


def test_json_log_formatter():
    rec = logging.LogRecord("x", logging.INFO, __file__, 1, "ask", None, None)
    rec.data = {"event": "ask", "total_ms": 12.5}
    out = json.loads(JsonFormatter().format(rec))
    assert out["event"] == "ask" and out["total_ms"] == 12.5 and out["level"] == "INFO"


def test_config_loading_and_env_override(monkeypatch):
    monkeypatch.setenv("FILINGQA_DATABASE_URL", "postgresql://u:p@h:1/db")
    cfg = load_config("config/offline.yaml")
    assert cfg.embedding.provider == "hashing" and cfg.llm.provider == "fake"
    assert cfg.database.url == "postgresql://u:p@h:1/db"
    default = load_config("config/default.yaml", use_env=False)
    assert default.embedding.model == "text-embedding-3-small"
    assert default.retrieval.mode == "hybrid_rerank" and default.eval.min_verified == 100
    with pytest.raises(FileNotFoundError):
        load_config("config/missing.yaml")


def test_sec_rate_limit_validated():
    with pytest.raises(ValueError):
        IngestConfig(max_requests_per_second=12)
    assert AppConfig().ingest.max_requests_per_second <= 10
