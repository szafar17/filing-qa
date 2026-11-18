from __future__ import annotations

from datetime import date
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from filingqa.answer.prompts import ANSWER_SCHEMA, build_answer_user, parse_answer_user
from filingqa.answer.schema import AskResponse, LLMAnswer, Usage
from filingqa.answer.service import AnswerService, evidence_too_weak
from filingqa.api.app import create_app
from filingqa.config import RefusalConfig
from filingqa.eval.draft import DRAFT_SCHEMA, UNANSWERABLE_SCHEMA
from filingqa.eval.faithfulness import JUDGE_SCHEMA
from filingqa.llm.base import LLM, LLMOutputError, LLMResult
from filingqa.llm.fake import FakeLLM
from filingqa.retrieval.retriever import Hit, RetrievalResult
from filingqa.store.repository import ChunkRecord

TEXT = (
    "Total revenue increased 13.6% to $1,284.6 million in 2024 from $1,130.5 million in 2023. "
    "Gross margin improved to 38.2%."
)


def record(i: int = 1, text: str = TEXT, start: int = 5000) -> ChunkRecord:
    return ChunkRecord(
        id=i,
        chunk_uid=f"0000000001-25-000001:{i}",
        accession="0000000001-25-000001",
        company="Example Corp",
        ticker="EXM",
        cik=1,
        form_type="10-K",
        fiscal_year=2024,
        fiscal_period="FY",
        filing_date=date(2025, 2, 1),
        period_of_report=date(2024, 12, 31),
        section_id="7",
        section_title="Item 7. MD&A",
        chunk_index=i,
        char_start=start,
        char_end=start + len(text),
        text=text,
        context_header="Example Corp (EXM) | 10-K | fiscal FY2024\nSection: Item 7. MD&A",
        has_table=False,
        source_url="https://www.sec.gov/Archives/edgar/data/1/x/doc.htm",
    )


class StubRetriever:
    def __init__(self, hits: list[ChunkRecord], evidence: float | None = 0.9, kind: str | None = "rerank"):
        self.hits, self.evidence, self.kind = hits, evidence, kind
        self.calls: list[dict] = []

    def retrieve(self, question: str, **kw: Any) -> RetrievalResult:
        self.calls.append({"question": question, **kw})
        hits = [Hit(chunk=c, rank=i + 1, score=0.9) for i, c in enumerate(self.hits)]
        return RetrievalResult(
            mode=kw.get("mode") or "hybrid_rerank",
            hits=hits,
            timings_ms={"rerank": 1.0},
            evidence=self.evidence,
            evidence_kind=self.kind,
        )


class ScriptedLLM(LLM):
    provider = "openai"

    def __init__(self, outputs: list[Any], model: str = "gpt-4.1-mini"):
        self.outputs = list(outputs)
        self.model = model
        self.calls = 0

    def complete_json(self, *, system, user, schema, schema_name) -> LLMResult:
        self.calls += 1
        out = self.outputs.pop(0)
        if isinstance(out, Exception):
            raise out
        return LLMResult(
            data=out, raw="", model=self.model, input_tokens=1000, output_tokens=100, latency_ms=5.0
        )


def service(llm: LLM, retriever: StubRetriever | None = None, **refusal: Any) -> AnswerService:
    from filingqa.cost import ModelPrice, PriceTable

    prices = PriceTable(models={"gpt-4.1-mini": ModelPrice(input=0.40, output=1.60)})
    return AnswerService(retriever or StubRetriever([record()]), llm, RefusalConfig(**refusal), prices)


GOOD = {
    "refused": False,
    "refusal_reason": None,
    "answer": "Total revenue was $1,284.6 million in 2024 [1].",
    "confidence": 0.9,
    "citations": [{"passage": 1, "quote": "Total revenue increased 13.6% to $1,284.6 million in 2024"}],
}


def test_answer_with_grounded_citation_and_offsets():
    out = service(ScriptedLLM([GOOD])).ask("What was total revenue in 2024?")
    r = out.response
    assert not r.refused and r.answer.startswith("Total revenue")
    (c,) = r.citations
    assert c.chunk_id == "0000000001-25-000001:1" and c.section == "Item 7. MD&A"
    assert c.char_start == 5000 and c.char_end == 5000 + len(c.quote)
    assert c.source_url.startswith("https://www.sec.gov/")
    # cost = 1000 * 0.40/1M + 100 * 1.60/1M
    assert r.cost_usd == pytest.approx(0.00056)
    assert r.usage.input_tokens == 1000 and r.usage.output_tokens == 100
    assert out.schema_valid is True and out.refusal_stage is None


def test_weak_evidence_refuses_without_calling_llm():
    llm = ScriptedLLM([GOOD])
    out = service(llm, StubRetriever([record()], evidence=0.02), min_rerank_score=0.1).ask("q?")
    assert out.response.refused and "Evidence too weak" in out.response.refusal_reason
    assert llm.calls == 0 and out.refusal_stage == "evidence" and not out.llm_called
    assert out.response.citations == [] and out.response.answer is None


def test_vector_evidence_threshold():
    rr = StubRetriever([record()], evidence=0.1, kind="vector").retrieve("q")
    assert evidence_too_weak(rr, RefusalConfig(min_vector_similarity=0.25))
    assert evidence_too_weak(rr, RefusalConfig(min_vector_similarity=0.05)) is None


def test_no_hits_refuses():
    out = service(ScriptedLLM([GOOD]), StubRetriever([])).ask("q?")
    assert out.response.refused and out.refusal_stage == "evidence"


def test_model_abstention_is_a_refusal():
    abstain = {
        "refused": True,
        "refusal_reason": "Not in passages.",
        "answer": None,
        "confidence": 0.0,
        "citations": [],
    }
    out = service(ScriptedLLM([abstain])).ask("q?")
    assert out.response.refused and out.response.refusal_reason == "Not in passages."
    assert out.refusal_stage == "model"


def test_unverifiable_quote_turns_answer_into_refusal():
    bad = {**GOOD, "citations": [{"passage": 1, "quote": "Revenue was $9 trillion"}]}
    out = service(ScriptedLLM([bad])).ask("q?")
    assert out.response.refused and out.refusal_stage == "grounding"
    assert out.dropped_citations == 1


def test_out_of_range_passage_is_dropped_but_valid_one_kept():
    mixed = {**GOOD, "citations": [{"passage": 7, "quote": "x"}, GOOD["citations"][0]]}
    out = service(ScriptedLLM([mixed])).ask("q?")
    assert not out.response.refused and len(out.response.citations) == 1
    assert out.dropped_citations == 1


def test_quote_matching_tolerates_whitespace_and_quote_style():
    rec = record(text="Net sales were $ 1,842.5 million\nin fiscal 2024, the company’s best year.")
    ans = {
        **GOOD,
        "citations": [
            {"passage": 1, "quote": "net sales were $1,842.5 million in fiscal 2024, the company's best year"}
        ],
    }
    out = service(ScriptedLLM([ans]), StubRetriever([rec])).ask("q?")
    assert not out.response.refused
    assert out.response.citations[0].quote.startswith("Net sales were $ 1,842.5 million")


def test_schema_invalid_output_is_retried_then_refused():
    invalid = {"refused": False, "answer": None, "confidence": 0.5, "citations": []}
    out = service(ScriptedLLM([invalid, GOOD])).ask("q?")
    assert not out.response.refused and out.schema_valid is False  # valid on 2nd try only
    out2 = service(ScriptedLLM([invalid, LLMOutputError("bad json")])).ask("q?")
    assert out2.response.refused and out2.refusal_stage == "schema"


def test_llm_answer_schema_rules():
    LLMAnswer.model_validate(GOOD)
    with pytest.raises(ValidationError):
        LLMAnswer.model_validate({**GOOD, "answer": "  "})
    LLMAnswer.model_validate({**GOOD, "refused": True, "answer": None, "citations": []})


def test_ask_response_consistency_rules():
    base = dict(
        question="q", confidence=0.5, mode="vector", trace_id="t", latency_ms=1.0, cost_usd=0.0, usage=Usage()
    )
    with pytest.raises(ValidationError):
        AskResponse(answer="x", citations=[], refused=False, **base)  # answer without citation
    with pytest.raises(ValidationError):
        AskResponse(answer=None, citations=[], refused=True, refusal_reason=None, **base)
    with pytest.raises(ValidationError):
        AskResponse(answer=None, citations=[], refused=False, **{**base, "confidence": 1.5})
    AskResponse(answer=None, citations=[], refused=True, refusal_reason="weak", **base)


@pytest.mark.parametrize("schema", [ANSWER_SCHEMA, JUDGE_SCHEMA, DRAFT_SCHEMA, UNANSWERABLE_SCHEMA])
def test_structured_output_schemas_are_strict_compatible(schema):
    """OpenAI strict mode: every object lists all properties as required, no extras."""

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node["required"]) == set(node["properties"])
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(schema)


def test_prompt_roundtrip():
    user = build_answer_user("What?", [("Hdr A\nSection: 7", "Text one."), ("Hdr B", "Text two.")])
    q, passages = parse_answer_user(user)
    assert q == "What?"
    assert passages == [(1, "Hdr A / Section: 7", "Text one."), (2, "Hdr B", "Text two.")]


def test_fake_llm_extracts_and_refuses():
    svc = service(FakeLLM())
    ok = svc.ask("What was total revenue in 2024?").response
    assert not ok.refused and "$1,284.6 million" in ok.citations[0].quote
    no = svc.ask("Who is the chief executive officer's favorite composer?").response
    assert no.refused and no.refusal_reason


# --- API -------------------------------------------------------------------------------------


@pytest.fixture()
def client():
    svc = service(ScriptedLLM([GOOD] * 10))
    app = create_app(service_factory=lambda: (svc, None))
    with TestClient(app) as c:
        c.service = svc  # type: ignore[attr-defined]
        yield c


def test_api_healthz(client):
    assert client.get("/healthz").json() == {"status": "ok"}


def test_api_ask_returns_schema_valid_json(client):
    resp = client.post(
        "/ask",
        json={
            "question": "What was total revenue in 2024?",
            "mode": "hybrid",
            "top_k": 3,
            "filters": {"tickers": ["exm"], "fiscal_years": [2024]},
        },
    )
    assert resp.status_code == 200
    body = AskResponse.model_validate(resp.json())
    assert not body.refused and body.citations[0].quote
    call = client.service.retriever.calls[-1]
    assert call["mode"] == "hybrid" and call["k"] == 3
    assert call["filters"].tickers == ["exm"] and call["filters"].fiscal_years == [2024]


def test_api_validates_requests(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "mode": "magic"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 0}).status_code == 422


def test_api_refusal_path():
    svc = service(ScriptedLLM([GOOD]), StubRetriever([record()], evidence=0.01), min_rerank_score=0.1)
    with TestClient(create_app(service_factory=lambda: (svc, None))) as c:
        body = c.post("/ask", json={"question": "Unrelated question?"}).json()
    assert body["refused"] is True and body["answer"] is None and body["citations"] == []
