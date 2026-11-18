"""Answer prompt, its strict JSON schema, and a parser for the prompt format."""

from __future__ import annotations

import re

ANSWER_SYSTEM = """You answer questions about SEC 10-K and 10-Q filings for financial analysts.
Use ONLY the numbered passages provided. Rules:
1. Every factual statement in the answer must be supported by a passage. Put the passage
   number in square brackets after the statement, e.g. "Net sales were $383.3 billion [2]."
2. For each passage you rely on, add a citation with the passage number and a quote copied
   character-for-character from that passage's text (between <<< and >>>, not from its
   header line). Keep each quote short: one sentence or one table row.
3. If the passages do not contain the information needed, set "refused" to true, "answer" to
   null, "citations" to [], and say briefly in "refusal_reason" what is missing. Do not guess
   and do not use outside knowledge.
4. Copy numbers exactly as written, with units, currency and the period they refer to. If the
   company or period in the question does not match the passages, refuse.
5. "confidence" is your probability (0 to 1) that the answer is fully supported by the quotes."""

# Strict structured-output schema: every property required, no extras, nullables via type lists.
ANSWER_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "refused": {"type": "boolean"},
        "refusal_reason": {"type": ["string", "null"]},
        "answer": {"type": ["string", "null"]},
        "confidence": {"type": "number"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "passage": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["passage", "quote"],
            },
        },
    },
    "required": ["refused", "refusal_reason", "answer", "confidence", "citations"],
}

_PASSAGE_RE = re.compile(r"\[(\d+)\] (.*?)\n<<<\n(.*?)\n>>>", re.S)


def format_passages(passages: list[tuple[str, str]]) -> str:
    """passages: (header, text) pairs, numbered from 1."""
    return "\n\n".join(
        f"[{i}] {header.replace(chr(10), ' / ')}\n<<<\n{text}\n>>>"
        for i, (header, text) in enumerate(passages, start=1)
    )


def build_answer_user(question: str, passages: list[tuple[str, str]]) -> str:
    return f"Question: {question}\n\nPassages:\n{format_passages(passages)}"


def parse_passages(block: str) -> list[tuple[int, str, str]]:
    return [(int(n), h, t) for n, h, t in _PASSAGE_RE.findall(block)]


def parse_answer_user(user: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Inverse of build_answer_user (used by the offline fake LLM)."""
    m = re.match(r"Question: (.*?)\n\nPassages:\n(.*)\Z", user, re.S)
    if not m:
        return user, []
    return m.group(1).strip(), parse_passages(m.group(2))
