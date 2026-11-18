"""Metric math, kept free of I/O so it is easy to test.

For one question, ``matches[i]`` is the set of gold passages matched by the i-th retrieved
chunk (empty set = not relevant).

  recall@k    = |gold passages matched by the top k| / |gold passages|
  hit@k       = 1 if any of the top k matches any gold passage else 0
  RR          = 1 / rank of the first relevant chunk (0 if none retrieved); MRR = mean RR
  percentile  = linear interpolation between closest ranks (numpy's default method)

With one gold passage per question (what `make draft-golden` produces), recall@k == hit@k.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def recall_at_k(matches: Sequence[set[int]], n_gold: int, k: int) -> float:
    if n_gold <= 0:
        raise ValueError("recall is undefined without gold passages")
    found: set[int] = set()
    for m in matches[:k]:
        found |= m
    return len(found) / n_gold


def hit_at_k(matches: Sequence[set[int]], k: int) -> float:
    return 1.0 if any(m for m in matches[:k]) else 0.0


def reciprocal_rank(matches: Sequence[set[int]]) -> float:
    for i, m in enumerate(matches, start=1):
        if m:
            return 1.0 / i
    return 0.0


def first_relevant_rank(matches: Sequence[set[int]]) -> int | None:
    for i, m in enumerate(matches, start=1):
        if m:
            return i
    return None


def mean(values: Sequence[float]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def percentile(values: Sequence[float], q: float) -> float | None:
    """q in [0, 100]. Linear interpolation, matching numpy.percentile(method='linear')."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if not 0 <= q <= 100:
        raise ValueError("q must be in [0, 100]")
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(vals[lo])
    return float(vals[lo] + (vals[hi] - vals[lo]) * (pos - lo))


def span_overlap_fraction(a_start: int, a_end: int, b_start: int, b_end: int) -> float:
    """Fraction of span B covered by span A (0..1)."""
    length = b_end - b_start
    if length <= 0:
        return 0.0
    inter = max(0, min(a_end, b_end) - max(a_start, b_start))
    return inter / length
