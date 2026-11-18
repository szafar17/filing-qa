"""One-command eval:  make eval  ==  python -m filingqa.eval

Official run (default):
  * evaluates only human-verified golden items, and refuses to run if there are fewer
    than eval.min_verified (default 100);
  * refuses offline stand-ins (fake LLM/judge, hashing embedder): those numbers are not
    model quality;
  * writes results/eval_<run>.json, results/latest.json and results/latest.md;
  * exits 1 if any metric breaks eval/thresholds.json (the regression gate).

Smoke run (--smoke): any items (including synthetic fixtures), any components; results
are marked official=false and written under <results_dir>/smoke_*. CI uses this with the
synthetic mini corpus and eval/thresholds.offline.json.

Exit codes: 0 ok, 1 regression gate failed, 2 refused to run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from filingqa.config import ALL_MODES, load_config
from filingqa.logging_utils import setup_logging


def _modes(s: str | None, default: list[str]) -> list[str]:
    if not s:
        return list(default)
    out = [m.strip() for m in s.split(",") if m.strip()]
    bad = [m for m in out if m not in ALL_MODES]
    if bad:
        raise SystemExit(f"unknown mode(s) {bad}; choose from {ALL_MODES}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.eval")
    ap.add_argument("--config", default=None)
    ap.add_argument("--golden", default=None, help="golden JSONL (default: eval.golden_path)")
    ap.add_argument("--modes", default=None, help="retrieval modes to compare, comma-separated")
    ap.add_argument("--answer-modes", default=None, help="modes to run end-to-end answers for")
    ap.add_argument("--smoke", action="store_true", help="non-official run (see module doc)")
    ap.add_argument("--min-verified", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="evaluate only the first N items")
    ap.add_argument("--no-judge", action="store_true", help="skip the faithfulness judge")
    ap.add_argument("--retrieval-only", action="store_true", help="skip answers entirely")
    ap.add_argument("--thresholds", default=None)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args(argv)

    setup_logging("WARNING")
    cfg = load_config(args.config)
    from filingqa.eval.golden import load_golden, verified_count

    golden_path = args.golden or cfg.eval.golden_path
    items = load_golden(golden_path)
    n_verified = verified_count(items)
    min_verified = cfg.eval.min_verified if args.min_verified is None else args.min_verified
    print(
        f"golden set {golden_path}: {len(items)} items, {n_verified} human-verified "
        f"(official minimum: {min_verified})"
    )

    official = not args.smoke
    if official:
        if n_verified < min_verified:
            print(
                f"REFUSING to run the official eval: only {n_verified} human-verified items, "
                f"need at least {min_verified}. Draft candidates with `make draft-golden` and "
                "review them with `make label`, or use --smoke for a non-official run."
            )
            return 2
        stand_ins = []
        if cfg.llm.provider == "fake":
            stand_ins.append("llm=fake")
        if cfg.eval.judge.provider == "fake" and not (args.no_judge or args.retrieval_only):
            stand_ins.append("judge=fake")
        if cfg.embedding.provider == "hashing":
            stand_ins.append("embedding=hashing")
        if cfg.reranker.provider == "lexical":
            stand_ins.append("reranker=lexical")
        if stand_ins:
            print(f"REFUSING official eval with offline stand-ins ({', '.join(stand_ins)}); use --smoke.")
            return 2
        items = [i for i in items if i.counts_as_verified()]
    if args.limit:
        items = items[: args.limit]

    from filingqa.components import build_components
    from filingqa.eval.gate import check, flatten_metrics, load_thresholds
    from filingqa.eval.report import render_markdown
    from filingqa.eval.runner import EvalOptions, run_eval
    from filingqa.llm import make_llm

    modes = _modes(args.modes, cfg.eval.modes)
    answer_modes = [] if args.retrieval_only else _modes(args.answer_modes, cfg.eval.answer_modes)
    need_rerank = "hybrid_rerank" in modes or "hybrid_rerank" in answer_modes
    comps = build_components(cfg, need_reranker=need_rerank)
    if official:
        n_synth = comps.store.stats()["synthetic_filings"]
        if n_synth:
            print(
                f"REFUSING official eval: the index at {cfg.database.url.rsplit('@', 1)[-1]} contains "
                f"{n_synth} synthetic test filing(s), which could be retrieved for real questions. "
                "Keep the synthetic corpus in its own database (see config/offline.yaml)."
            )
            comps.close()
            return 2
    if comps.store.bm25_stats_stale():
        print("WARNING: BM25 statistics are stale; run `python -m filingqa.ingest index` first.")
    judge_llm = None if (args.no_judge or args.retrieval_only) else make_llm(cfg.eval.judge)
    opts = EvalOptions(
        modes=modes,
        answer_modes=answer_modes,
        ks=cfg.eval.ks,
        official=official,
        relevance_overlap=cfg.eval.relevance_overlap,
        judge=judge_llm is not None,
    )
    all_items = load_golden(golden_path)
    verified = [i for i in all_items if i.counts_as_verified()]
    provenance = {
        "llm_drafted_accepted_unchanged": sum(i.origin == "llm-draft" and not i.edited for i in verified),
        "llm_drafted_edited_by_human": sum(i.origin == "llm-draft" and i.edited for i in verified),
        "written_by_human": sum(i.origin == "human-written" for i in verified),
        "unanswerable": sum(not i.answerable for i in verified),
    }
    results = run_eval(
        comps,
        items,
        opts,
        judge_llm,
        {
            "path": golden_path,
            "n_items_total": len(all_items),
            "n_human_verified": n_verified,
            "verified_provenance": provenance,
        },
    )
    comps.close()

    out_dir = Path(args.results_dir or cfg.eval.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "eval" if official else "smoke"
    run_path = out_dir / f"{prefix}_{results['run_id']}.json"
    latest = out_dir / ("latest.json" if official else "smoke_latest.json")
    md_path = out_dir / ("latest.md" if official else "smoke_latest.md")
    payload = json.dumps(results, indent=2, default=str)
    run_path.write_text(payload)
    latest.write_text(payload)
    md = render_markdown(results)
    md_path.write_text(md)
    print(md)
    print(f"results: {run_path} (also {latest}, {md_path})")

    if args.no_gate:
        return 0
    th_path = args.thresholds or cfg.eval.thresholds_path
    thresholds = load_thresholds(th_path)
    if not thresholds:
        print(f"gate: no thresholds in {th_path}; regression gate is INACTIVE for this run.")
        return 0
    violations = check(flatten_metrics(results), thresholds)
    for v in violations:
        print(f"REGRESSION: {v}")
    print(f"gate ({th_path}): {'FAIL' if violations else 'PASS'}")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
