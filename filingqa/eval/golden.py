"""Golden set and draft records (JSONL), with the labeling rules enforced in code.

Rules:
  * Only items with labeled_by="human", verified=true and a verified_at timestamp count
    toward the official eval. The `make label` tool is the only thing in this repo that
    sets those fields, and only after a person accepts the item.
  * LLM drafts (labeled_by="llm-draft") live in a separate drafts file and can never be
    verified.
  * Synthetic fixture items (labeled_by="synthetic-fixture", synthetic=true) are test
    data and can never be verified.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

LabeledBy = Literal["human", "llm-draft", "synthetic-fixture"]


class GoldPassage(BaseModel):
    accession: str
    quote: str = Field(min_length=1)
    chunk_id: str | None = None  # chunk the passage came from when labeled (informational)
    char_start: int | None = None  # offsets into the filing's normalized text
    char_end: int | None = None
    source_url: str | None = None
    section: str | None = None


class GoldenItem(BaseModel):
    id: str
    question: str = Field(min_length=3)
    answerable: bool = True
    gold: list[GoldPassage] = Field(default_factory=list)
    reference_answer: str | None = None
    company: str | None = None
    ticker: str | None = None
    form_type: str | None = None
    fiscal_period: str | None = None
    labeled_by: LabeledBy
    verified: bool = False
    verified_at: datetime | None = None
    reviewer: str | None = None
    synthetic: bool = False
    notes: str | None = None
    # Provenance: who wrote the question ("llm-draft" = LLM-drafted then human-reviewed).
    origin: Literal["llm-draft", "human-written", "synthetic"] | None = None
    edited: bool = False  # reviewer changed the drafted question/answer/quote
    draft_id: str | None = None
    draft_model: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> GoldenItem:
        if self.answerable and not self.gold:
            raise ValueError(f"{self.id}: answerable items need at least one gold passage")
        if not self.answerable and self.gold:
            raise ValueError(f"{self.id}: unanswerable items must not have gold passages")
        if self.labeled_by == "synthetic-fixture" and not self.synthetic:
            raise ValueError(f"{self.id}: synthetic-fixture items must set synthetic=true")
        if self.verified:
            if self.labeled_by != "human":
                raise ValueError(f"{self.id}: only labeled_by='human' items can be verified")
            if self.verified_at is None:
                raise ValueError(f"{self.id}: verified items need verified_at")
            if self.synthetic:
                raise ValueError(f"{self.id}: synthetic fixture items can never be verified")
        return self

    def counts_as_verified(self) -> bool:
        return (
            self.verified
            and self.labeled_by == "human"
            and self.verified_at is not None
            and not self.synthetic
        )


class DraftItem(BaseModel):
    """An LLM-drafted candidate awaiting human review (eval/golden_drafts.jsonl)."""

    draft_id: str
    question: str
    answerable: bool = True
    reference_answer: str | None = None
    gold: list[GoldPassage] = Field(default_factory=list)
    why_unanswerable: str | None = None
    company: str | None = None
    ticker: str | None = None
    form_type: str | None = None
    fiscal_period: str | None = None
    context_header: str | None = None
    context_text: str | None = None  # the chunk the draft was written from, for review
    context_char_start: int | None = None  # offset of context_text in the filing text
    draft_model: str
    labeled_by: Literal["llm-draft"] = "llm-draft"
    verified: Literal[False] = False
    status: Literal["pending", "accepted", "rejected"] = "pending"
    reviewed_at: datetime | None = None
    reject_reason: str | None = None


def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def _read_jsonl(path: str | Path) -> Iterable[tuple[int, dict]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows = []
    for n, line in enumerate(p.read_text().splitlines(), start=1):
        if line.strip():
            rows.append((n, json.loads(line)))
    return rows


def load_golden(path: str | Path) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    for n, row in _read_jsonl(path):
        try:
            items.append(GoldenItem.model_validate(row))
        except ValidationError as exc:
            raise ValueError(f"{path}:{n}: invalid golden item: {exc}") from exc
    ids = [i.id for i in items]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate ids {sorted(dupes)}")
    return items


def load_drafts(path: str | Path) -> list[DraftItem]:
    return [DraftItem.model_validate(row) for _, row in _read_jsonl(path)]


def write_jsonl(path: str | Path, rows: Iterable[BaseModel]) -> None:
    """Atomic rewrite: write a temp file, then rename over the original."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(r.model_dump_json(exclude_none=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(p)


def verified_count(items: Iterable[GoldenItem]) -> int:
    return sum(1 for i in items if i.counts_as_verified())
