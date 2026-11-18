"""Render eval results as a markdown report."""

from __future__ import annotations

from typing import Any


def _f(v: Any, pct: bool = False, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{100 * v:.1f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_markdown(r: dict[str, Any]) -> str:
    g, c = r["golden"], r["corpus"]
    cfg = r["config"]
    lines = [
        f"# Filing Q&A eval, run {r['run_id']}",
        "",
        f"- official: **{'yes' if r['official'] else 'NO (smoke run: not quotable)'}**",
        f"- git commit: {r.get('git_commit')}",
        f"- golden set: `{g.get('path')}`; human-verified items: {g.get('n_human_verified')}; "
        f"evaluated: {g['n_evaluated']} ({g['n_answerable']} answerable, "
        f"{g['n_unanswerable']} unanswerable); skipped (gold not found): "
        f"{g['n_skipped_unresolved_gold']}",
        f"- corpus: {c.get('filings')} filings from {c.get('companies')} companies "
        f"(+{c.get('synthetic_filings')} synthetic), {c.get('chunks')} chunks "
        f"({c.get('table_chunks')} with tables)",
        f"- embedding: {cfg['embedding']['provider']}:{cfg['embedding']['model']}; "
        f"reranker: {cfg['reranker']['provider']}:{cfg['reranker']['model']}; "
        f"LLM: {cfg['llm']['provider']}:{cfg['llm']['model']}; "
        f"judge: {(cfg.get('judge') or {}).get('provider')}:{(cfg.get('judge') or {}).get('model')}",
        f"- prices as of {r['prices'].get('as_of')}; unpriced models: "
        f"{r['prices'].get('unpriced_models') or 'none'}",
        "",
        "## Retrieval",
        "",
        "| mode | n | recall@1 | recall@5 | recall@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for mode, m in r["retrieval"].items():
        lines.append(
            f"| {mode} | {m['n']} | {_f(m.get('recall@1'), True)} | {_f(m.get('recall@5'), True)} "
            f"| {_f(m.get('recall@10'), True)} | {_f(m.get('mrr'))} | "
            f"{_f(m.get('latency_ms_p50'), digits=1)} | {_f(m.get('latency_ms_p95'), digits=1)} |"
        )
    for mode, a in r["answer"].items():
        lines += [
            "",
            f"## End-to-end answers ({mode})",
            "",
            "| metric | value |",
            "|---|---|",
            f"| faithfulness (mean supported-claim share, answered items) | {_f(a['faithfulness'], True)} |",
            f"| answers with every claim supported | {_f(a['fully_faithful_rate'], True)} |",
            f"| answers judged | {a['n_judged']} (no claims: {a['n_judge_no_claims']}, "
            f"judge errors: {a['n_judge_errors']}) |",
            f"| citation overlaps a gold passage (answered, answerable) | "
            f"{_f(a['citation_gold_overlap'], True)} |",
            f"| refusal accuracy (unanswerable refused) | {_f(a['refusal_accuracy'], True)} |",
            f"| false refusal rate (answerable refused) | {_f(a['false_refusal_rate'], True)} |",
            f"| first-try schema-valid LLM output | {_f(a['schema_valid_rate'], True)} |",
            f"| end-to-end latency p50 / p95 | {_f(a['latency_ms_p50'], digits=0)} ms / "
            f"{_f(a['latency_ms_p95'], digits=0)} ms |",
            f"| cost per query (embedding + answer LLM) | ${_f(a['cost_usd_per_query'], digits=5)} |",
            f"| refusals by stage | {a['refusal_stages'] or 'none'} |",
        ]
        sweep = r.get("threshold_sweep", {}).get(mode)
        if sweep and sweep["rows"]:
            lines += [
                "",
                f"Refusal threshold sweep on `{sweep['evidence_kind']}` evidence (pre-LLM check only; "
                "pick a threshold that refuses unanswerable questions without refusing answerable ones):",
                "",
                "| threshold | unanswerable refused | answerable refused |",
                "|---|---|---|",
            ]
            for row in sweep["rows"]:
                lines.append(
                    f"| {row['threshold']:.2f} | {_f(row['unanswerable_refused'], True)} | "
                    f"{_f(row['answerable_refused'], True)} |"
                )
    lines += ["", f"Judge cost (eval overhead, not per-query cost): ${r['judge_cost_usd_total']:.4f}", ""]
    return "\n".join(lines)
