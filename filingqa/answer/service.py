"""Question -> retrieve -> (refuse | LLM with numbered passages) -> verified, cited answer."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError

from filingqa.answer.prompts import ANSWER_SCHEMA, ANSWER_SYSTEM, build_answer_user
from filingqa.answer.schema import AskResponse, Citation, LLMAnswer, Usage
from filingqa.config import RefusalConfig
from filingqa.cost import PriceTable
from filingqa.llm.base import LLM, LLMOutputError
from filingqa.logging_utils import log_event
from filingqa.retrieval.retriever import RetrievalResult, Retriever
from filingqa.store.repository import ChunkRecord, SearchFilters
from filingqa.textutil import locate_quote
from filingqa.trace import Trace

log = logging.getLogger("filingqa.ask")


@dataclass
class AnswerOutcome:
    response: AskResponse
    retrieval: RetrievalResult | None
    passages: list[ChunkRecord] = field(default_factory=list)
    llm_called: bool = False
    schema_valid: bool | None = None  # None when the LLM was not called
    dropped_citations: int = 0
    refusal_stage: str | None = None  # "evidence" | "model" | "grounding" | "schema"


def evidence_too_weak(retrieval: RetrievalResult, cfg: RefusalConfig) -> str | None:
    """Return a refusal reason if the best retrieved evidence is below threshold."""
    if not retrieval.hits:
        return "No passages were retrieved for this question."
    ev, kind = retrieval.evidence, retrieval.evidence_kind
    if ev is None:
        return None
    if kind == "rerank" and ev < cfg.min_rerank_score:
        return f"Evidence too weak: best reranker score {ev:.3f} < {cfg.min_rerank_score}."
    if kind == "vector" and ev < cfg.min_vector_similarity:
        return f"Evidence too weak: best vector similarity {ev:.3f} < {cfg.min_vector_similarity}."
    if kind == "bm25" and cfg.min_bm25_score is not None and ev < cfg.min_bm25_score:
        return f"Evidence too weak: best BM25 score {ev:.2f} < {cfg.min_bm25_score}."
    return None


class AnswerService:
    def __init__(
        self,
        retriever: Retriever,
        llm: LLM,
        refusal: RefusalConfig,
        prices: PriceTable | None = None,
        schema_retries: int = 1,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.refusal = refusal
        self.prices = prices or PriceTable()
        self.schema_retries = schema_retries

    def _call_llm(
        self, question: str, passages: list[ChunkRecord], trace: Trace
    ) -> tuple[LLMAnswer | None, bool]:
        user = build_answer_user(question, [(p.context_header, p.text) for p in passages])
        for attempt in range(self.schema_retries + 1):
            try:
                with trace.stage("llm"):
                    res = self.llm.complete_json(
                        system=ANSWER_SYSTEM, user=user, schema=ANSWER_SCHEMA, schema_name="answer"
                    )
            except LLMOutputError as exc:
                log_event(log, "llm_output_invalid", logging.WARNING, error=str(exc), attempt=attempt)
                continue
            trace.add_usage(
                "answer",
                model=res.model,
                provider=self.llm.provider,
                input_tokens=res.input_tokens,
                output_tokens=res.output_tokens,
                cost_usd=self.prices.cost(
                    res.model,
                    res.input_tokens,
                    res.output_tokens,
                    provider=self.llm.provider,
                    cached_input_tokens=res.cached_input_tokens,
                ),
            )
            try:
                return LLMAnswer.model_validate(res.data), attempt == 0
            except ValidationError as exc:
                log_event(
                    log,
                    "llm_schema_invalid",
                    logging.WARNING,
                    error=str(exc)[:500],
                    attempt=attempt,
                )
        return None, False

    def _ground(self, parsed: LLMAnswer, passages: list[ChunkRecord]) -> tuple[list[Citation], int]:
        """Keep only citations whose quote really occurs in the cited passage's text."""
        out: list[Citation] = []
        dropped = 0
        seen: set[tuple[int, int, int]] = set()
        for c in parsed.citations:
            if not 1 <= c.passage <= len(passages):
                dropped += 1
                continue
            p = passages[c.passage - 1]
            span = locate_quote(p.text, c.quote)
            if span is None:
                dropped += 1
                continue
            start, end = p.char_start + span[0], p.char_start + span[1]
            key = (c.passage, start, end)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                Citation(
                    passage=c.passage,
                    chunk_id=p.chunk_uid,
                    company=p.company,
                    ticker=p.ticker,
                    form_type=p.form_type,
                    fiscal_year=p.fiscal_year,
                    fiscal_period=p.fiscal_period,
                    filing_date=p.filing_date,
                    section=p.section_title,
                    source_url=p.source_url,
                    quote=p.text[span[0] : span[1]],
                    char_start=start,
                    char_end=end,
                )
            )
        return out, dropped

    def ask(
        self,
        question: str,
        *,
        mode: str | None = None,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
    ) -> AnswerOutcome:
        trace = Trace()
        t0 = time.perf_counter()
        retrieval = self.retriever.retrieve(question, mode=mode, k=top_k, filters=filters, trace=trace)
        passages = [h.chunk for h in retrieval.hits]
        llm_called = False
        schema_valid: bool | None = None
        dropped = 0
        answer: str | None = None
        citations: list[Citation] = []
        confidence = 0.0
        refusal_reason = evidence_too_weak(retrieval, self.refusal)
        stage = "evidence" if refusal_reason else None

        if refusal_reason is None:
            llm_called = True
            parsed, schema_valid = self._call_llm(question, passages, trace)
            if parsed is None:
                refusal_reason = "The model did not return valid structured output."
                stage = "schema"
                schema_valid = False
            elif parsed.refused:
                refusal_reason = parsed.refusal_reason or "The model found no supporting evidence."
                stage = "model"
            else:
                citations, dropped = self._ground(parsed, passages)
                if self.refusal.require_citations and not citations:
                    refusal_reason = (
                        "The answer could not be grounded: no cited quote was found in the "
                        "retrieved passages."
                    )
                    stage = "grounding"
                else:
                    answer = parsed.answer
                    confidence = min(max(parsed.confidence, 0.0), 1.0)

        latency_ms = (time.perf_counter() - t0) * 1000
        usage = Usage(
            input_tokens=sum(u.input_tokens for k, u in trace.usage.items() if k != "embed_query"),
            output_tokens=sum(u.output_tokens for u in trace.usage.values()),
            embedding_tokens=retrieval.embed_tokens,
        )
        response = AskResponse(
            question=question,
            answer=None if refusal_reason else answer,
            citations=[] if refusal_reason else citations,
            confidence=0.0 if refusal_reason else confidence,
            evidence_score=retrieval.evidence,
            evidence_kind=retrieval.evidence_kind,
            refused=refusal_reason is not None,
            refusal_reason=refusal_reason,
            mode=retrieval.mode,
            trace_id=trace.trace_id,
            latency_ms=round(latency_ms, 2),
            cost_usd=round(trace.cost_usd, 8),
            usage=usage,
        )
        trace.meta.update(
            {
                "mode": retrieval.mode,
                "n_passages": len(passages),
                "refused": response.refused,
                "refusal_stage": stage,
                "n_citations": len(response.citations),
                "dropped_citations": dropped,
                "evidence": retrieval.evidence,
                "evidence_kind": retrieval.evidence_kind,
            }
        )
        log_event(log, "ask", **trace.to_dict())
        return AnswerOutcome(
            response=response,
            retrieval=retrieval,
            passages=passages,
            llm_called=llm_called,
            schema_valid=schema_valid,
            dropped_citations=dropped,
            refusal_stage=stage,
        )
