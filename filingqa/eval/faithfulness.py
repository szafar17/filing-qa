"""Faithfulness: are the answer's claims supported by the passages it cites?

LLM-as-judge with a fixed rubric (also documented in the README):
  1. Split the answer into atomic factual claims.
  2. A claim is supported only if the cited passages state it explicitly, or it follows by
     trivial arithmetic or rewording. Missing, contradicted, wrong period/segment/company,
     or added specifics => not supported.
  3. Judge only against the cited passages, never against outside knowledge.
Per-answer score = supported claims / total claims. The eval reports the mean over
answered (non-refused) questions, plus the share of answers with every claim supported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from filingqa.answer.prompts import format_passages, parse_passages
from filingqa.cost import PriceTable
from filingqa.llm.base import LLM, LLMOutputError

JUDGE_SYSTEM = """You grade whether an answer is faithful to the source passages it cites.
These are passages from SEC 10-K and 10-Q filings. Apply this rubric exactly:
1. Split the answer into atomic factual claims (one fact each: a number, a date, a name, a
   direction of change, a cause). Ignore citation markers like [1] and pure hedging.
2. A claim is supported (true) only if the cited passages state it explicitly, or it follows
   by trivial arithmetic or rewording (e.g. "$1,284.6 million" = "about $1.28 billion").
3. A claim is NOT supported (false) if it is absent from the passages, contradicts them,
   refers to a different period, segment or company, or adds specifics (numbers, causes,
   comparisons) that the passages do not contain.
4. Judge only against the passages shown, never against your own knowledge, even if the
   claim is true in reality.
Give a short reason for each claim."""

JUDGE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "claim": {"type": "string"},
                    "supported": {"type": "boolean"},
                    "reason": {"type": "string"},
                },
                "required": ["claim", "supported", "reason"],
            },
        }
    },
    "required": ["claims"],
}


def build_judge_user(question: str, answer: str, passages: list[tuple[str, str]]) -> str:
    return f"Question: {question}\n\nAnswer: {answer}\n\nCited passages:\n{format_passages(passages)}"


def parse_judge_user(user: str) -> tuple[str, list[tuple[int, str, str]]]:
    m = re.match(r"Question: .*?\n\nAnswer: (.*?)\n\nCited passages:\n(.*)\Z", user, re.S)
    if not m:
        return "", []
    return m.group(1), parse_passages(m.group(2))


@dataclass
class FaithfulnessResult:
    claims: list[dict[str, Any]] = field(default_factory=list)
    supported: int = 0
    total: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    model: str = ""
    error: str | None = None

    @property
    def score(self) -> float | None:
        return self.supported / self.total if self.total else None


def judge_faithfulness(
    llm: LLM,
    question: str,
    answer: str,
    cited: list[tuple[str, str]],
    prices: PriceTable | None = None,
) -> FaithfulnessResult:
    prices = prices or PriceTable()
    try:
        res = llm.complete_json(
            system=JUDGE_SYSTEM,
            user=build_judge_user(question, answer, cited),
            schema=JUDGE_SCHEMA,
            schema_name="faithfulness_judgment",
        )
    except LLMOutputError as exc:
        return FaithfulnessResult(error=str(exc), model=llm.model)
    claims = [c for c in res.data.get("claims", []) if isinstance(c, dict)]
    supported = sum(1 for c in claims if c.get("supported") is True)
    return FaithfulnessResult(
        claims=claims,
        supported=supported,
        total=len(claims),
        input_tokens=res.input_tokens,
        output_tokens=res.output_tokens,
        cost_usd=prices.cost(res.model, res.input_tokens, res.output_tokens, provider=llm.provider),
        model=res.model,
    )
