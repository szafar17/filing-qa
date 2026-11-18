from __future__ import annotations

import json

import numpy as np
import pytest

from filingqa.eval.gate import check, flatten_metrics, load_thresholds, main, thresholds_from_results
from filingqa.eval.metrics import (
    first_relevant_rank,
    hit_at_k,
    mean,
    percentile,
    recall_at_k,
    reciprocal_rank,
    span_overlap_fraction,
)

# matches[i] = gold passages matched by the i-th retrieved chunk
M = [set(), {0}, set(), {1}, set(), {0}]


def test_recall_at_k_counts_distinct_gold_passages():
    assert recall_at_k(M, n_gold=2, k=1) == 0.0
    assert recall_at_k(M, n_gold=2, k=2) == 0.5
    assert recall_at_k(M, n_gold=2, k=5) == 1.0
    assert recall_at_k(M, n_gold=2, k=100) == 1.0
    with pytest.raises(ValueError):
        recall_at_k(M, n_gold=0, k=5)


def test_single_gold_recall_equals_hit():
    one = [set(), set(), {0}]
    for k in (1, 2, 3, 10):
        assert recall_at_k(one, 1, k) == hit_at_k(one, k)


def test_hit_rr_and_rank():
    assert hit_at_k(M, 1) == 0.0 and hit_at_k(M, 2) == 1.0
    assert reciprocal_rank(M) == 0.5
    assert reciprocal_rank([set(), set()]) == 0.0
    assert first_relevant_rank(M) == 2
    assert first_relevant_rank([]) is None


def test_mean_ignores_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([]) is None


@pytest.mark.parametrize("q", [0, 25, 50, 90, 95, 99, 100])
def test_percentile_matches_numpy_linear(q):
    vals = [120.0, 80.0, 3000.0, 450.0, 95.0, 101.0, 2500.0]
    assert percentile(vals, q) == pytest.approx(np.percentile(vals, q))


def test_percentile_edge_cases():
    assert percentile([], 95) is None
    assert percentile([7.0], 95) == 7.0
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_span_overlap_fraction():
    assert span_overlap_fraction(0, 100, 50, 150) == 0.5
    assert span_overlap_fraction(0, 10, 20, 30) == 0.0
    assert span_overlap_fraction(0, 100, 10, 20) == 1.0
    assert span_overlap_fraction(0, 100, 5, 5) == 0.0


# --- gate ----------------------------------------------------------------------------------

RESULTS = {
    "run_id": "r1",
    "official": True,
    "retrieval": {
        "vector": {"recall@5": 0.55, "mrr": 0.5, "n": 100},
        "hybrid_rerank": {"recall@5": 0.72, "mrr": 0.6, "n": 100},
    },
    "answer": {
        "hybrid_rerank": {
            "faithfulness": 0.81,
            "latency_ms_p95": 1900.0,
            "cost_usd_per_query": 0.002,
            "refusal_stages": {"model": 3},
        }
    },
}
# (arbitrary example inputs for the gate math; not measured results)


def test_flatten_skips_non_numeric():
    flat = flatten_metrics(RESULTS)
    assert flat["retrieval.hybrid_rerank.recall@5"] == 0.72
    assert flat["answer.hybrid_rerank.latency_ms_p95"] == 1900.0
    assert not any("refusal_stages" in k for k in flat)


def test_check_min_max_and_missing():
    flat = flatten_metrics(RESULTS)
    th = {
        "retrieval.hybrid_rerank.recall@5": {"min": 0.70},
        "answer.hybrid_rerank.latency_ms_p95": {"max": 2000},
    }
    assert check(flat, th) == []
    th["retrieval.hybrid_rerank.recall@5"] = {"min": 0.75}
    th["answer.hybrid_rerank.cost_usd_per_query"] = {"max": 0.001}
    th["answer.hybrid_rerank.not_a_metric"] = {"min": 0.1}
    kinds = sorted((v.metric.rsplit(".", 1)[-1], v.kind) for v in check(flat, th))
    assert kinds == [("cost_usd_per_query", "max"), ("not_a_metric", "missing"), ("recall@5", "min")]


def test_thresholds_from_results_directions():
    th = thresholds_from_results(flatten_metrics(RESULTS), abs_margin=0.03, rel_margin=0.25)
    assert th["retrieval.hybrid_rerank.recall@5"] == {"min": pytest.approx(0.69)}
    assert th["answer.hybrid_rerank.latency_ms_p95"] == {"max": pytest.approx(2375.0)}
    assert th["answer.hybrid_rerank.cost_usd_per_query"] == {"max": pytest.approx(0.0025)}


def test_gate_cli_exit_codes(tmp_path):
    res = tmp_path / "res.json"
    res.write_text(json.dumps(RESULTS))
    th = tmp_path / "th.json"
    assert main([str(res), str(th), "--write"]) == 0
    assert load_thresholds(th)
    assert main([str(res), str(th)]) == 0
    worse = json.loads(json.dumps(RESULTS))
    worse["retrieval"]["hybrid_rerank"]["recall@5"] = 0.60
    res.write_text(json.dumps(worse))
    assert main([str(res), str(th)]) == 1


def test_missing_thresholds_file_means_no_thresholds(tmp_path):
    assert load_thresholds(tmp_path / "nope.json") == {}
