"""Pydantic models for the model's raw output and for the public /ask API."""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RetrievalModeField = Literal["vector", "bm25", "hybrid", "hybrid_rerank"]


class LLMCitation(BaseModel):
    passage: int
    quote: str


class LLMAnswer(BaseModel):
    """What the LLM must return (mirrors prompts.ANSWER_SCHEMA)."""

    refused: bool
    refusal_reason: str | None = None
    answer: str | None = None
    confidence: float
    citations: list[LLMCitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> LLMAnswer:
        if not self.refused and not (self.answer and self.answer.strip()):
            raise ValueError("answer is required unless refused")
        return self


class AskFilters(BaseModel):
    tickers: list[str] | None = None
    form_types: list[str] | None = None
    fiscal_years: list[int] | None = None


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: RetrievalModeField | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    filters: AskFilters | None = None


class Citation(BaseModel):
    passage: int = Field(ge=1)
    chunk_id: str
    company: str
    ticker: str | None
    form_type: str
    fiscal_year: int | None
    fiscal_period: str | None
    filing_date: date | None
    section: str
    source_url: str
    quote: str = Field(min_length=1)
    # Offsets of the quote inside the full filing text (filing_texts.text).
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    embedding_tokens: int = 0


class AskResponse(BaseModel):
    question: str
    answer: str | None
    citations: list[Citation]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence_score: float | None = None
    evidence_kind: str | None = None
    refused: bool
    refusal_reason: str | None = None
    mode: str
    trace_id: str
    latency_ms: float
    cost_usd: float
    usage: Usage

    @model_validator(mode="after")
    def _consistent(self) -> AskResponse:
        if self.refused:
            if self.answer is not None or self.citations:
                raise ValueError("a refusal carries no answer and no citations")
            if not self.refusal_reason:
                raise ValueError("a refusal needs a reason")
        else:
            if not (self.answer and self.answer.strip()):
                raise ValueError("an answer is required when not refused")
            if not self.citations:
                raise ValueError("an answer needs at least one citation")
        return self
