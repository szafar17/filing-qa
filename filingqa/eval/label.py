"""make label: terminal review loop that turns LLM drafts into human-verified golden items.

For each pending draft you see the question, the draft answer, and the gold passage
highlighted inside the chunk it came from. Then:

  a  accept   -> appended to eval/golden.jsonl with labeled_by="human", verified=true,
                 verified_at=<now, UTC>, origin="llm-draft"
  e  edit     -> change question / answer / quote (the quote must be copied from the
                 passage), or mark the item unanswerable; then accept or reject
  w  write    -> write your own question for this passage (origin="human-written")
  r  reject   -> marked rejected in the drafts file, never used
  s  skip     -> stays pending for a later session
  q  quit     -> progress is already saved

Every decision is saved immediately (atomic file rewrite), so you can stop any time and
resume with the same command.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from collections.abc import Callable

from filingqa.eval.golden import (
    DraftItem,
    GoldenItem,
    GoldPassage,
    load_drafts,
    load_golden,
    utcnow,
    verified_count,
    write_jsonl,
)
from filingqa.textutil import locate_quote

WIDTH = 100


class Reviewer:
    def __init__(
        self,
        drafts_path: str,
        golden_path: str,
        *,
        input_fn: Callable[[str], str] = input,
        print_fn: Callable[[str], None] = print,
        reviewer: str | None = None,
        retrieval_fn: Callable[[str], list[tuple[str, str]]] | None = None,
        min_verified: int = 100,
        color: bool = False,
    ) -> None:
        self.drafts_path = drafts_path
        self.golden_path = golden_path
        self.input = input_fn
        self.print = print_fn
        self.reviewer = reviewer
        self.retrieval_fn = retrieval_fn
        self.min_verified = min_verified
        self.color = color
        self.drafts = load_drafts(drafts_path)
        self.golden = load_golden(golden_path)

    # --- display ------------------------------------------------------------------

    def _wrap(self, text: str, indent: str = "    ") -> str:
        return "\n".join(
            textwrap.fill(line, WIDTH, initial_indent=indent, subsequent_indent=indent) or indent
            for line in text.splitlines()
        )

    def _highlight(self, context: str, quote: str) -> str:
        span = locate_quote(context, quote)
        if span is None:
            return context
        s, e = span
        on, off = ("\033[1;33m", "\033[0m") if self.color else (">>>", "<<<")
        return context[:s] + on + context[s:e] + off + context[e:]

    def show(self, d: DraftItem, idx: int, total: int) -> None:
        p = self.print
        p("=" * WIDTH)
        kind = "ANSWERABLE" if d.answerable else "UNANSWERABLE (should be refused)"
        p(
            f"[{idx}/{total}] {d.draft_id}  {kind}  |  verified so far: {verified_count(self.golden)}"
            f"/{self.min_verified}"
        )
        if d.context_header:
            p(self._wrap(d.context_header.replace("\n", " | "), ""))
        p("QUESTION:")
        p(self._wrap(d.question))
        if d.answerable:
            p("DRAFT ANSWER:")
            p(self._wrap(d.reference_answer or "(none)"))
            if d.context_text and d.gold:
                p("GOLD PASSAGE (quote marked inside its chunk):")
                p(self._wrap(self._highlight(d.context_text, d.gold[0].quote)))
        else:
            p("WHY UNANSWERABLE (LLM's claim, check it):")
            p(self._wrap(d.why_unanswerable or "(none)"))
            if self.retrieval_fn is not None:
                p("TOP RETRIEVED PASSAGES (none of these should answer it):")
                for i, (header, text) in enumerate(self.retrieval_fn(d.question)[:3], start=1):
                    p(self._wrap(f"[{i}] {header.splitlines()[0]}", "  "))
                    p(self._wrap(text[:600]))

    # --- persistence --------------------------------------------------------------

    def _save(self) -> None:
        write_jsonl(self.golden_path, self.golden)
        write_jsonl(self.drafts_path, self.drafts)

    def _to_golden(self, d: DraftItem, *, origin: str, edited: bool) -> GoldenItem:
        prefix = "q-" if origin == "llm-draft" else "h-"
        base = prefix + d.draft_id.removeprefix("draft-")
        ids = {g.id for g in self.golden}
        gid, n = base, 1
        while gid in ids:
            n += 1
            gid = f"{base}-{n}"
        return GoldenItem(
            id=gid,
            question=d.question,
            answerable=d.answerable,
            gold=d.gold if d.answerable else [],
            reference_answer=d.reference_answer if d.answerable else None,
            company=d.company,
            ticker=d.ticker,
            form_type=d.form_type,
            fiscal_period=d.fiscal_period,
            labeled_by="human",
            verified=True,
            verified_at=utcnow(),
            reviewer=self.reviewer,
            origin=origin,  # type: ignore[arg-type]
            edited=edited,
            draft_id=d.draft_id,
            draft_model=d.draft_model,
        )

    # --- editing ------------------------------------------------------------------

    def _ask_quote(self, d: DraftItem) -> GoldPassage | None:
        """Prompt until the quote is found in the passage (or the user gives up)."""
        if not d.context_text or not d.gold:
            return None
        while True:
            q = self.input("New gold quote, copied from the passage (Enter = keep): ").strip()
            if not q:
                return d.gold[0]
            span = locate_quote(d.context_text, q)
            if span is None:
                self.print("  That text is not in the passage. Copy it exactly, or press Enter.")
                continue
            s, e = span
            base = d.context_char_start if d.context_char_start is not None else 0
            old = d.gold[0]
            return GoldPassage(
                accession=old.accession,
                quote=d.context_text[s:e],
                chunk_id=old.chunk_id,
                char_start=base + s if d.context_char_start is not None else None,
                char_end=base + e if d.context_char_start is not None else None,
                source_url=old.source_url,
                section=old.section,
            )

    def edit(self, d: DraftItem) -> DraftItem:
        upd = d.model_copy(deep=True)
        q = self.input("New question (Enter = keep): ").strip()
        if q:
            upd.question = q
        if upd.answerable:
            if self.input("Mark as UNANSWERABLE instead? [y/N]: ").strip().lower() == "y":
                upd.answerable = False
                upd.gold = []
                upd.reference_answer = None
                upd.why_unanswerable = self.input("Why is it unanswerable? ").strip() or None
                return upd
            a = self.input("New reference answer (Enter = keep): ").strip()
            if a:
                upd.reference_answer = a
            gp = self._ask_quote(upd)
            if gp is not None:
                upd.gold = [gp]
        return upd

    # --- loop ---------------------------------------------------------------------

    def run(self, limit: int | None = None) -> dict[str, int]:
        pending = [d for d in self.drafts if d.status == "pending"]
        counts = {"accepted": 0, "written": 0, "rejected": 0, "skipped": 0}
        self.print(
            f"{len(pending)} pending drafts; {verified_count(self.golden)} human-verified items in "
            f"{self.golden_path} (official eval needs {self.min_verified})."
        )
        for idx, d in enumerate(pending[:limit] if limit else pending, start=1):
            current, edited = d, False
            self.show(current, idx, len(pending))
            while True:
                choice = self.input("[a]ccept [e]dit [w]rite own [r]eject [s]kip [q]uit > ")
                choice = choice.strip().lower()[:1]
                if choice == "a":
                    self.golden.append(self._to_golden(current, origin="llm-draft", edited=edited))
                    d.status, d.reviewed_at = "accepted", utcnow()
                    self._save()
                    counts["accepted"] += 1
                    break
                if choice == "e":
                    current, edited = self.edit(current), True
                    self.show(current, idx, len(pending))
                    continue
                if choice == "w":
                    if not d.answerable or not d.context_text:
                        self.print("  [w] needs an answerable draft with a passage.")
                        continue
                    own = current.model_copy(deep=True)
                    own.question = ""
                    while not own.question:
                        own.question = self.input("Your question: ").strip()
                    own.reference_answer = self.input("Your reference answer: ").strip() or None
                    gp = self._ask_quote(own)
                    if gp is not None:
                        own.gold = [gp]
                    self.golden.append(self._to_golden(own, origin="human-written", edited=False))
                    d.status, d.reviewed_at = "accepted", utcnow()
                    self._save()
                    counts["written"] += 1
                    break
                if choice == "r":
                    d.reject_reason = self.input("Reason (optional): ").strip() or None
                    d.status, d.reviewed_at = "rejected", utcnow()
                    self._save()
                    counts["rejected"] += 1
                    break
                if choice == "s":
                    counts["skipped"] += 1
                    break
                if choice == "q":
                    self.print(f"Saved. {verified_count(self.golden)} human-verified items.")
                    return counts
                self.print("  Please type a, e, w, r, s or q.")
        self.print(f"Done. {verified_count(self.golden)} human-verified items in {self.golden_path}.")
        return counts


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.eval.label")
    ap.add_argument("--config", default=None)
    ap.add_argument("--drafts", default="eval/golden_drafts.jsonl")
    ap.add_argument("--golden", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--show-retrieval",
        action="store_true",
        help="for unanswerable drafts, show top retrieved passages (needs the DB)",
    )
    args = ap.parse_args(argv)

    from filingqa.config import load_config

    cfg = load_config(args.config)
    retrieval_fn = None
    if args.show_retrieval:
        from filingqa.components import build_components

        comps = build_components(cfg, need_reranker=False)

        def retrieval_fn(q: str) -> list[tuple[str, str]]:
            rr = comps.retriever.retrieve(q, mode="hybrid", k=3)
            return [(h.chunk.context_header, h.chunk.text) for h in rr.hits]

    reviewer = Reviewer(
        args.drafts,
        args.golden or cfg.eval.golden_path,
        reviewer=os.environ.get("FILINGQA_REVIEWER") or None,
        retrieval_fn=retrieval_fn,
        min_verified=cfg.eval.min_verified,
        color=sys.stdout.isatty(),
    )
    try:
        reviewer.run(limit=args.limit)
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted; every decision so far is saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
