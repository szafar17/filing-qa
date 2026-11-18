"""Regression gate: compare flattened eval metrics with eval/thresholds*.json.

thresholds.json:
    {"metrics": {"retrieval.hybrid_rerank.recall@5": {"min": 0.80},
                 "answer.hybrid_rerank.latency_ms_p95": {"max": 3000}}}

A metric listed in thresholds but missing from the results is a failure, so a broken
run cannot pass by producing nothing.

    python -m filingqa.eval.gate results/latest.json eval/thresholds.json
    python -m filingqa.eval.gate results/latest.json eval/thresholds.json --write   # re-baseline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HIGHER_IS_BETTER = (
    "recall@",
    "hit@",
    "mrr",
    "faithfulness",
    "fully_faithful_rate",
    "refusal_accuracy",
    "citation_gold_overlap",
    "schema_valid_rate",
    "answered_rate",
)
LOWER_IS_BETTER = ("latency_ms", "cost_usd", "false_refusal_rate")


@dataclass
class Violation:
    metric: str
    value: float | None
    bound: float
    kind: str  # "min" | "max" | "missing"

    def __str__(self) -> str:
        if self.kind == "missing":
            return f"{self.metric}: missing from results (threshold {self.bound})"
        op = ">=" if self.kind == "min" else "<="
        return f"{self.metric} = {self.value:.4f}, required {op} {self.bound}"


def flatten_metrics(results: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for section in ("retrieval", "answer"):
        for mode, metrics in (results.get(section) or {}).items():
            for name, value in metrics.items():
                if isinstance(value, int | float) and not isinstance(value, bool):
                    flat[f"{section}.{mode}.{name}"] = float(value)
    return flat


def load_thresholds(path: str | Path) -> dict[str, dict[str, float]]:
    p = Path(path)
    if not p.is_file():
        return {}
    return (json.loads(p.read_text()) or {}).get("metrics", {}) or {}


def check(flat: dict[str, float], thresholds: dict[str, dict[str, float]]) -> list[Violation]:
    out: list[Violation] = []
    for metric, bounds in sorted(thresholds.items()):
        value = flat.get(metric)
        for kind in ("min", "max"):
            if kind not in bounds or bounds[kind] is None:
                continue
            bound = float(bounds[kind])
            if value is None:
                out.append(Violation(metric, None, bound, "missing"))
            elif (kind == "min" and value < bound) or (kind == "max" and value > bound):
                out.append(Violation(metric, value, bound, kind))
    return out


def thresholds_from_results(
    flat: dict[str, float],
    *,
    abs_margin: float = 0.03,
    rel_margin: float = 0.25,
    include: tuple[str, ...] = (
        "recall@5",
        "mrr",
        "faithfulness",
        "refusal_accuracy",
        "latency_ms_p95",
        "cost_usd_per_query",
    ),
) -> dict[str, dict[str, float]]:
    """Baseline thresholds from a run: quality may drop by abs_margin, latency/cost may
    rise by rel_margin, before the gate fails."""
    out: dict[str, dict[str, float]] = {}
    for metric, value in sorted(flat.items()):
        name = metric.rsplit(".", 1)[-1]
        if not any(name.startswith(i) or name == i for i in include):
            continue
        if any(name.startswith(h) for h in HIGHER_IS_BETTER):
            out[metric] = {"min": round(max(0.0, value - abs_margin), 4)}
        elif any(name.startswith(lo) for lo in LOWER_IS_BETTER):
            out[metric] = {"max": round(value * (1 + rel_margin), 6)}
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m filingqa.eval.gate")
    ap.add_argument("results")
    ap.add_argument("thresholds")
    ap.add_argument("--write", action="store_true", help="write thresholds baselined on results")
    ap.add_argument("--abs-margin", type=float, default=0.03)
    ap.add_argument("--rel-margin", type=float, default=0.25)
    ap.add_argument(
        "--metrics",
        default=None,
        help="comma-separated metric names to baseline (default: recall@5, mrr, faithfulness, "
        "refusal_accuracy, latency_ms_p95, cost_usd_per_query)",
    )
    args = ap.parse_args(argv)
    results = json.loads(Path(args.results).read_text())
    flat = flatten_metrics(results)
    if args.write:
        extra = {"include": tuple(m.strip() for m in args.metrics.split(","))} if args.metrics else {}
        th = thresholds_from_results(flat, abs_margin=args.abs_margin, rel_margin=args.rel_margin, **extra)
        payload = {
            "_comment": (
                f"Baselined from {args.results} (run {results.get('run_id')}, official="
                f"{results.get('official')}): quality metrics may drop by {args.abs_margin}, "
                f"latency/cost may rise by {args.rel_margin:.0%}."
            ),
            "metrics": th,
        }
        Path(args.thresholds).write_text(json.dumps(payload, indent=2) + "\n")
        print(f"wrote {len(th)} thresholds to {args.thresholds}")
        return 0
    violations = check(flat, load_thresholds(args.thresholds))
    for v in violations:
        print(f"REGRESSION: {v}")
    print("gate: FAIL" if violations else "gate: PASS")
    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
