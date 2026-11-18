"""make draft-golden: LLM-drafted golden-set CANDIDATES from the real index.

Samples chunks spread across filings (prose and table chunks), asks the LLM for one
analyst question per chunk with a reference answer and a verbatim supporting quote,
checks the quote really occurs in the chunk, and appends candidates to
eval/golden_drafts.jsonl with labeled_by="llm-draft", verified=false. It also drafts
unanswerable questions for the refusal metric. Nothing here can mark an item verified:
a person does that with `make label`.

  python -m filingqa.eval.draft --n 140 --unanswerable 25
"""

from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys

from filingqa.config import load_config
from filingqa.eval.golden import DraftItem, GoldPassage, load_drafts, write_jsonl
from filingqa.ingest.fiscal import period_label
from filingqa.llm.base import LLM, LLMOutputError
from filingqa.logging_utils import setup_logging
from filingqa.store.repository import ChunkRecord
from filingqa.textutil import locate_quote

DRAFT_SYSTEM = """You write evaluation questions for a question-answering system over SEC filings.
Given one passage from a filing, write ONE question that a financial analyst might ask and
that this passage alone answers.
Requirements:
- Name the company and the fiscal period or filing in the question (e.g. "In Apple's fiscal
  2023 10-K, ..."), so the question makes sense without seeing the passage.
- Ask about a specific fact: an amount, percentage, date, count, name, risk or policy. No
  yes/no questions and no questions about "this passage".
- "answer": a short reference answer (one or two sentences) using only the passage.
- "quote": copy, character-for-character, the shortest span of the passage that supports the
  answer (one sentence or one table row).
- If the passage has no specific answerable fact (boilerplate, table of contents, signature
  blocks, incorporation by reference), set "usable" to false and leave the other fields empty."""

DRAFT_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "usable": {"type": "boolean"},
        "question": {"type": "string"},
        "answer": {"type": "string"},
        "quote": {"type": "string"},
    },
    "required": ["usable", "question", "answer", "quote"],
}

UNANSWERABLE_SYSTEM = """Write ONE question that an analyst might plausibly ask about the company
below but that its SEC filing for the stated period would NOT answer. Good kinds: forecasts or
targets the company does not publish, facts about a fiscal period far outside the stated one,
private or personal details, or details about a competitor. The question must sound
realistic, name the company, and not be answerable from the passage shown. Also give a
one-line reason it is unanswerable from the filing."""

UNANSWERABLE_SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "question": {"type": "string"},
        "why_unanswerable": {"type": "string"},
    },
    "required": ["question", "why_unanswerable"],
}

DEFAULT_SECTIONS = [
    "1",
    "1A",
    "1C",
    "2",
    "3",
    "5",
    "7",
    "7A",
    "8",
    "9A",
    "I-1",
    "I-2",
    "I-3",
    "I-4",
    "II-1",
    "II-1A",
]


def build_draft_user(chunk: ChunkRecord, text: str | None = None) -> str:
    period = period_label(chunk.fiscal_year, chunk.fiscal_period) or "unknown period"
    return (
        f"Company: {chunk.company}\nTicker: {chunk.ticker}\nForm: {chunk.form_type}\n"
        f"Period: {period}\nSection: {chunk.section_title}\n\nPassage:\n<<<\n"
        f"{chunk.text if text is None else text}\n>>>"
    )


def parse_draft_user(user: str) -> tuple[dict[str, str], str]:
    meta = dict(re.findall(r"^(Company|Ticker|Form|Period|Section): (.*)$", user, re.M))
    m = re.search(r"Passage:\n<<<\n(.*)\n>>>", user, re.S)
    return {k.lower(): v for k, v in meta.items()}, (m.group(1) if m else "")


def _draft_id(question: str) -> str:
    return "draft-" + hashlib.sha1(question.strip().lower().encode()).hexdigest()[:10]


def draft_from_chunk(llm: LLM, chunk: ChunkRecord) -> tuple[DraftItem | None, str]:
    """Returns (draft or None, status) where status explains a skip."""
    try:
        res = llm.complete_json(
            system=DRAFT_SYSTEM,
            user=build_draft_user(chunk),
            schema=DRAFT_SCHEMA,
            schema_name="golden_draft",
        )
    except LLMOutputError as exc:
        return None, f"llm_error: {exc}"
    d = res.data
    if not d.get("usable") or not d.get("question") or not d.get("quote"):
        return None, "not_usable"
    span = locate_quote(chunk.text, d["quote"])
    if span is None:
        return None, "quote_not_in_passage"
    s, e = span
    gold = GoldPassage(
        accession=chunk.accession,
        quote=chunk.text[s:e],
        chunk_id=chunk.chunk_uid,
        char_start=chunk.char_start + s,
        char_end=chunk.char_start + e,
        source_url=chunk.source_url,
        section=chunk.section_title,
    )
    return (
        DraftItem(
            draft_id=_draft_id(d["question"]),
            question=d["question"].strip(),
            answerable=True,
            reference_answer=(d.get("answer") or "").strip() or None,
            gold=[gold],
            company=chunk.company,
            ticker=chunk.ticker,
            form_type=chunk.form_type,
            fiscal_period=period_label(chunk.fiscal_year, chunk.fiscal_period),
            context_header=chunk.context_header,
            context_text=chunk.text,
            context_char_start=chunk.char_start,
            draft_model=res.model,
        ),
        "ok",
    )


def draft_unanswerable(llm: LLM, chunk: ChunkRecord) -> DraftItem | None:
    try:
        res = llm.complete_json(
            system=UNANSWERABLE_SYSTEM,
            user=build_draft_user(chunk, chunk.text[:600]),
            schema=UNANSWERABLE_SCHEMA,
            schema_name="golden_unanswerable",
        )
    except LLMOutputError:
        return None
    q = (res.data.get("question") or "").strip()
    if not q:
        return None
    return DraftItem(
        draft_id=_draft_id(q),
        question=q,
        answerable=False,
        why_unanswerable=res.data.get("why_unanswerable"),
        company=chunk.company,
        ticker=chunk.ticker,
        form_type=chunk.form_type,
        fiscal_period=period_label(chunk.fiscal_year, chunk.fiscal_period),
        context_header=chunk.context_header,
        draft_model=res.model,
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.eval.draft")
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=140, help="answerable candidates to draft")
    ap.add_argument("--unanswerable", type=int, default=25)
    ap.add_argument("--table-share", type=float, default=0.3, help="share drafted from table chunks")
    ap.add_argument("--out", default="eval/golden_drafts.jsonl")
    ap.add_argument("--seed", type=float, default=0.42)
    ap.add_argument("--include-synthetic", action="store_true", help="for tests only")
    args = ap.parse_args(argv)
    setup_logging("WARNING")
    cfg = load_config(args.config)

    from filingqa.llm import make_llm
    from filingqa.store.db import connect
    from filingqa.store.repository import ChunkStore

    store = ChunkStore(connect(cfg.database.url))
    llm = make_llm(cfg.llm)
    existing = load_drafts(args.out)
    seen_q = {d.question.strip().lower() for d in existing}
    used_uids = [g.chunk_id for d in existing for g in d.gold if g.chunk_id]

    n_table = round(args.n * args.table_share)
    pool = store.sample_chunks(
        (args.n - n_table) * 2,
        seed=args.seed,
        section_ids=DEFAULT_SECTIONS,
        include_synthetic=args.include_synthetic,
        has_table=False,
        exclude_uids=used_uids,
    ) + store.sample_chunks(
        n_table * 2,
        seed=args.seed,
        include_synthetic=args.include_synthetic,
        has_table=True,
        exclude_uids=used_uids,
    )
    new: list[DraftItem] = []
    skipped: dict[str, int] = {}
    want = {"text": args.n - n_table, "table": n_table}
    got = {"text": 0, "table": 0}
    for chunk in pool:
        kind = "table" if chunk.has_table else "text"
        if got[kind] >= want[kind]:
            continue
        draft, status = draft_from_chunk(llm, chunk)
        if draft is None or draft.question.lower() in seen_q:
            key = status if draft is None else "duplicate"
            skipped[key] = skipped.get(key, 0) + 1
            continue
        seen_q.add(draft.question.lower())
        new.append(draft)
        got[kind] += 1
        print(f"drafted {len(new)}: {draft.question[:100]}", file=sys.stderr)

    rng = random.Random(int(args.seed * 1e6))
    sources = store.sample_chunks(
        args.unanswerable * 2, seed=args.seed / 2, include_synthetic=args.include_synthetic
    )
    rng.shuffle(sources)
    n_un = 0
    for chunk in sources:
        if n_un >= args.unanswerable:
            break
        d = draft_unanswerable(llm, chunk)
        if d and d.question.lower() not in seen_q:
            seen_q.add(d.question.lower())
            new.append(d)
            n_un += 1

    write_jsonl(args.out, [*existing, *new])
    print(
        f"wrote {len(new)} new drafts ({got['text']} prose, {got['table']} table, {n_un} unanswerable) "
        f"to {args.out}; skipped: {skipped or 'none'}.\n"
        f"These are UNVERIFIED. Review them with: make label"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
