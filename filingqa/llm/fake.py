"""Deterministic offline LLM used by tests and CI. NOT a model: a test double.

It understands the three prompt formats this project sends (answer, faithfulness judge,
golden-set drafting) and produces schema-valid JSON:
  * answer: picks the sentence/table row that covers the most question terms and quotes it
    verbatim; refuses when coverage is low.
  * judge: a claim is "supported" if >= 80% of its content words occur in the cited passages.
  * drafts: template questions over the most number-dense sentence of the passage.
Numbers it produces measure plumbing, never model quality.
"""

from __future__ import annotations

import json
import re
from typing import Any

from filingqa.answer.prompts import parse_answer_user
from filingqa.llm.base import LLM, LLMResult
from filingqa.textutil import content_tokens, sentence_spans

_MARKER = re.compile(r"\s*\[\d+\]")


def _units(text: str) -> list[str]:
    return [text[s:e] for s, e in sentence_spans(text)]


class FakeLLM(LLM):
    provider = "fake"

    def __init__(self, model: str = "fake-extractive", min_coverage: float = 0.6) -> None:
        self.model = model
        self.min_coverage = min_coverage
        self.calls: list[str] = []

    def complete_json(self, *, system: str, user: str, schema: dict[str, Any], schema_name: str) -> LLMResult:
        self.calls.append(schema_name)
        handler = {
            "answer": self._answer,
            "faithfulness_judgment": self._judge,
            "golden_draft": self._draft,
            "golden_unanswerable": self._unanswerable,
        }.get(schema_name)
        if handler is None:
            raise ValueError(f"FakeLLM has no handler for schema {schema_name!r}")
        data = handler(user)
        raw = json.dumps(data)
        return LLMResult(
            data=data,
            raw=raw,
            model=self.model,
            input_tokens=(len(system) + len(user)) // 4,
            output_tokens=len(raw) // 4,
            latency_ms=0.0,
        )

    # --- answer -------------------------------------------------------------------

    def _answer(self, user: str) -> dict[str, Any]:
        question, passages = parse_answer_user(user)
        q_terms = set(content_tokens(question))
        refuse = {
            "refused": True,
            "refusal_reason": "The passages do not cover the question.",
            "answer": None,
            "confidence": 0.0,
            "citations": [],
        }
        if not q_terms or not passages:
            return refuse
        best: tuple[float, float, int, str] | None = None
        for n, header, text in passages:
            header_terms = set(content_tokens(header))
            for unit in _units(text):
                unit_terms = set(content_tokens(unit))
                in_unit = q_terms & unit_terms
                if not in_unit:
                    continue
                coverage = len(q_terms & (unit_terms | header_terms)) / len(q_terms)
                key = (coverage, len(in_unit) / len(q_terms), -n, unit)
                if best is None or key[:3] > best[:3]:
                    best = key
        if best is None or best[0] < self.min_coverage:
            return refuse
        coverage, _, neg_n, unit = best
        return {
            "refused": False,
            "refusal_reason": None,
            "answer": f"{unit} [{-neg_n}]",
            "confidence": round(coverage, 3),
            "citations": [{"passage": -neg_n, "quote": unit}],
        }

    # --- faithfulness judge -------------------------------------------------------

    def _judge(self, user: str) -> dict[str, Any]:
        from filingqa.eval.faithfulness import parse_judge_user

        answer, passages = parse_judge_user(user)
        support_terms: set[str] = set()
        for _, header, text in passages:
            support_terms |= set(content_tokens(header + " " + text))
        claims = []
        for claim in _units(_MARKER.sub("", answer)):
            toks = content_tokens(claim)
            if not toks:
                continue
            frac = sum(t in support_terms for t in toks) / len(toks)
            claims.append(
                {
                    "claim": claim,
                    "supported": frac >= 0.8,
                    "reason": f"{frac:.0%} of content words appear in the cited passages",
                }
            )
        return {"claims": claims}

    # --- golden drafting ----------------------------------------------------------

    def _draft(self, user: str) -> dict[str, Any]:
        from filingqa.eval.draft import parse_draft_user

        meta, text = parse_draft_user(user)
        units = [u for u in _units(text) if len(u) >= 40]
        if not units:
            return {"usable": False, "question": "", "answer": "", "quote": ""}
        unit = max(units, key=lambda u: (sum(ch.isdigit() for ch in u), -len(u)))
        topic = " ".join(re.findall(r"[A-Za-z][A-Za-z'-]+", unit)[:8]).lower()
        company = meta.get("company", "the company")
        period = meta.get("period", "the period")
        return {
            "usable": True,
            "question": f"For {company} ({period}), what does the filing say about {topic}?",
            "answer": unit,
            "quote": unit,
        }

    def _unanswerable(self, user: str) -> dict[str, Any]:
        from filingqa.eval.draft import parse_draft_user

        meta, _ = parse_draft_user(user)
        company = meta.get("company", "the company")
        return {
            "question": f"What revenue will {company} report for fiscal 2035?",
            "why_unanswerable": "Filings report historical results, not results 10 years out.",
        }
