"""Run the eval: retrieval metrics per mode, then end-to-end answer metrics.

Relevance: a retrieved chunk is relevant to a gold passage when it is from the same
filing and covers at least ``relevance_overlap`` (default 50%) of the gold span. Gold
spans are character ranges in the filing text; at the start of a run each one is checked
against the current index (and relocated by its quote if the parser changed), so
re-chunking never silently invalidates labels. Gold passages that cannot be found are
reported and their items are skipped, never counted as hits or misses.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from filingqa.components import Components
from filingqa.eval.faithfulness import judge_faithfulness
from filingqa.eval.golden import GoldenItem
from filingqa.eval.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    span_overlap_fraction,
)
from filingqa.llm.base import LLM
from filingqa.retrieval.retriever import Hit
from filingqa.textutil import locate_quote, norm_for_match

log = logging.getLogger("filingqa.eval")


@dataclass
class ResolvedGold:
    accession: str
    start: int
    end: int


@dataclass
class EvalOptions:
    modes: list[str]
    answer_modes: list[str]
    ks: list[int]
    official: bool
    relevance_overlap: float = 0.5
    judge: bool = True
    progress_every: int = 10


@dataclass
class _ModeAcc:
    recall: dict[int, list[float]] = field(default_factory=dict)
    hit: dict[int, list[float]] = field(default_factory=dict)
    rr: list[float] = field(default_factory=list)
    latency: list[float] = field(default_factory=list)


def git_commit() -> str | None:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def resolve_gold(item: GoldenItem, filing_text) -> list[ResolvedGold] | None:  # noqa: ANN001
    """Validate/relocate each gold span in the current filing text; None if any is missing."""
    out: list[ResolvedGold] = []
    for gp in item.gold:
        text = filing_text(gp.accession)
        if text is None:
            return None
        s, e = gp.char_start, gp.char_end
        ok = (
            s is not None
            and e is not None
            and 0 <= s < e <= len(text)
            and norm_for_match(text[s:e]) == norm_for_match(gp.quote)
        )
        if not ok:
            span = locate_quote(text, gp.quote)
            if span is None:
                return None
            s, e = span
        out.append(ResolvedGold(gp.accession, int(s), int(e)))
    return out


def match_hits(hits: list[Hit], golds: list[ResolvedGold], min_overlap: float) -> list[set[int]]:
    return [
        {
            gi
            for gi, g in enumerate(golds)
            if h.chunk.accession == g.accession
            and span_overlap_fraction(h.chunk.char_start, h.chunk.char_end, g.start, g.end) >= min_overlap
        }
        for h in hits
    ]


def _progress(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _threshold_sweep(rows: list[tuple[bool, float | None]], kind: str | None) -> dict[str, Any]:
    """Refusal trade-off of the evidence pre-check alone (ignores model abstention)."""
    ans = [ev for a, ev in rows if a and ev is not None]
    una = [ev for a, ev in rows if not a and ev is not None]
    grid = [round(0.05 * i, 2) for i in range(0, 20)]
    table = []
    for t in grid:
        table.append(
            {
                "threshold": t,
                "unanswerable_refused": (sum(ev < t for ev in una) / len(una)) if una else None,
                "answerable_refused": (sum(ev < t for ev in ans) / len(ans)) if ans else None,
            }
        )
    return {"evidence_kind": kind, "rows": table}


def run_eval(
    comps: Components,
    items: list[GoldenItem],
    opts: EvalOptions,
    judge_llm: LLM | None,
    golden_meta: dict[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    store, retriever, service = comps.store, comps.retriever, comps.service
    text_cache: dict[str, str | None] = {}

    def filing_text(acc: str) -> str | None:
        if acc not in text_cache:
            text_cache[acc] = store.filing_text(acc)
        return text_cache[acc]

    resolved: dict[str, list[ResolvedGold]] = {}
    unresolved: list[str] = []
    for it in items:
        if not it.answerable:
            continue
        r = resolve_gold(it, filing_text)
        if r is None:
            unresolved.append(it.id)
        else:
            resolved[it.id] = r
    usable = [it for it in items if (not it.answerable) or it.id in resolved]
    answerable = [it for it in usable if it.answerable]
    unanswerable = [it for it in usable if not it.answerable]
    if unresolved:
        _progress(
            f"WARNING: {len(unresolved)} item(s) skipped: gold passage not found in the "
            f"current index ({', '.join(unresolved[:5])}{'...' if len(unresolved) > 5 else ''})"
        )

    per_item: dict[str, dict[str, Any]] = {
        it.id: {
            "id": it.id,
            "question": it.question,
            "answerable": it.answerable,
            "retrieval": {},
            "answer": {},
        }
        for it in usable
    }

    # ---- retrieval ---------------------------------------------------------------
    ks = sorted(set(opts.ks))
    max_k = max(ks)
    accs = {m: _ModeAcc({k: [] for k in ks}, {k: [] for k in ks}) for m in opts.modes}
    needs_vec = any(m in ("vector", "hybrid", "hybrid_rerank") for m in opts.modes)
    for n, it in enumerate(answerable, start=1):
        golds = resolved[it.id]
        qe = retriever.embed_query(it.question) if needs_vec else None
        for mode in opts.modes:
            rr = retriever.retrieve(it.question, mode=mode, k=max_k, query_embedding=qe)
            matches = match_hits(rr.hits, golds, opts.relevance_overlap)
            acc = accs[mode]
            for k in ks:
                acc.recall[k].append(recall_at_k(matches, len(golds), k))
                acc.hit[k].append(hit_at_k(matches, k))
            acc.rr.append(reciprocal_rank(matches))
            acc.latency.append(rr.total_ms)
            per_item[it.id]["retrieval"][mode] = {
                "first_relevant_rank": first_relevant_rank(matches),
                "top_chunk_ids": [h.chunk.chunk_uid for h in rr.hits[:5]],
                "latency_ms": round(rr.total_ms, 2),
            }
        if n % opts.progress_every == 0 or n == len(answerable):
            _progress(f"retrieval: {n}/{len(answerable)} questions")

    retrieval: dict[str, dict[str, Any]] = {}
    for mode, acc in accs.items():
        m: dict[str, Any] = {"n": len(acc.rr)}
        for k in ks:
            m[f"recall@{k}"] = mean(acc.recall[k])
            m[f"hit@{k}"] = mean(acc.hit[k])
        m["mrr"] = mean(acc.rr)
        m["latency_ms_p50"] = percentile(acc.latency, 50)
        m["latency_ms_p95"] = percentile(acc.latency, 95)
        retrieval[mode] = m

    # ---- answers -----------------------------------------------------------------
    answer: dict[str, dict[str, Any]] = {}
    sweeps: dict[str, Any] = {}
    judge_cost_total = 0.0
    for mode in opts.answer_modes:
        lat: list[float] = []
        cost: list[float] = []
        faith: list[float] = []
        fully: list[float] = []
        cite_overlap: list[float] = []
        schema_ok: list[float] = []
        stages: dict[str, int] = {}
        false_ref = correct_ref = answered = no_claims = judge_errors = 0
        sweep_rows: list[tuple[bool, float | None]] = []
        ev_kind: str | None = None
        for n, it in enumerate(usable, start=1):
            out = service.ask(it.question, mode=mode)
            resp = out.response
            lat.append(resp.latency_ms)
            cost.append(resp.cost_usd)
            ev_kind = resp.evidence_kind or ev_kind
            sweep_rows.append((it.answerable, resp.evidence_score))
            if out.schema_valid is not None:
                schema_ok.append(1.0 if out.schema_valid else 0.0)
            if out.refusal_stage:
                stages[out.refusal_stage] = stages.get(out.refusal_stage, 0) + 1
            rec: dict[str, Any] = {
                "refused": resp.refused,
                "refusal_reason": resp.refusal_reason,
                "answer": resp.answer,
                "citations": [
                    {
                        "chunk_id": c.chunk_id,
                        "quote": c.quote,
                        "char_start": c.char_start,
                        "char_end": c.char_end,
                    }
                    for c in resp.citations
                ],
                "evidence": resp.evidence_score,
                "latency_ms": resp.latency_ms,
                "cost_usd": resp.cost_usd,
            }
            if it.answerable and resp.refused:
                false_ref += 1
            if not it.answerable and resp.refused:
                correct_ref += 1
            if not resp.refused:
                answered += 1
                if it.answerable:
                    golds = resolved[it.id]
                    hit = any(
                        c.chunk_id.split(":")[0] == g.accession
                        and span_overlap_fraction(c.char_start, c.char_end, g.start, g.end) > 0
                        for c in resp.citations
                        for g in golds
                    )
                    cite_overlap.append(1.0 if hit else 0.0)
                    rec["citation_overlaps_gold"] = hit
                if opts.judge and judge_llm is not None:
                    cited_nums = sorted({c.passage for c in resp.citations})
                    cited = [
                        (out.passages[i - 1].context_header, out.passages[i - 1].text) for i in cited_nums
                    ]
                    fr = judge_faithfulness(judge_llm, it.question, resp.answer or "", cited, comps.prices)
                    judge_cost_total += fr.cost_usd
                    if fr.error:
                        judge_errors += 1
                    elif fr.score is None:
                        no_claims += 1
                    else:
                        faith.append(fr.score)
                        fully.append(1.0 if fr.supported == fr.total else 0.0)
                    rec["faithfulness"] = {
                        "score": fr.score,
                        "supported": fr.supported,
                        "total": fr.total,
                        "claims": fr.claims,
                        "error": fr.error,
                    }
            per_item[it.id]["answer"][mode] = rec
            if n % opts.progress_every == 0 or n == len(usable):
                _progress(f"answers[{mode}]: {n}/{len(usable)} questions")
        answer[mode] = {
            "n": len(usable),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "faithfulness": mean(faith),
            "fully_faithful_rate": mean(fully),
            "n_judged": len(faith),
            "n_judge_no_claims": no_claims,
            "n_judge_errors": judge_errors,
            "citation_gold_overlap": mean(cite_overlap),
            "refusal_accuracy": (correct_ref / len(unanswerable)) if unanswerable else None,
            "false_refusal_rate": (false_ref / len(answerable)) if answerable else None,
            "answered_rate": answered / len(usable) if usable else None,
            "schema_valid_rate": mean(schema_ok),
            "latency_ms_p50": percentile(lat, 50),
            "latency_ms_p95": percentile(lat, 95),
            "cost_usd_per_query": mean(cost),
            "cost_usd_total": sum(cost),
            "refusal_stages": stages,
        }
        sweeps[mode] = _threshold_sweep(sweep_rows, ev_kind)

    cfg = comps.cfg
    return {
        "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "official": opts.official,
        "git_commit": git_commit(),
        "config_path": cfg.config_path,
        "config": {
            "embedding": cfg.embedding.model_dump(),
            "llm": cfg.llm.model_dump(),
            "reranker": cfg.reranker.model_dump(),
            "retrieval": cfg.retrieval.model_dump(),
            "refusal": cfg.refusal.model_dump(),
            "judge": cfg.eval.judge.model_dump() if opts.judge else None,
            "relevance_overlap": opts.relevance_overlap,
        },
        "golden": {
            **golden_meta,
            "n_evaluated": len(usable),
            "n_answerable": len(answerable),
            "n_unanswerable": len(unanswerable),
            "n_skipped_unresolved_gold": len(unresolved),
            "skipped_ids": unresolved,
        },
        "corpus": store.stats(),
        "prices": {
            "as_of": comps.prices.as_of,
            "source": comps.prices.source,
            "unpriced_models": sorted(comps.prices.unpriced),
        },
        "retrieval": retrieval,
        "answer": answer,
        "judge_cost_usd_total": judge_cost_total,
        "threshold_sweep": sweeps,
        "elapsed_s": round(time.perf_counter() - started, 1),
        "per_item": list(per_item.values()),
    }
