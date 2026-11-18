"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    rrf(d) = sum over rankings r of  w_r / (k + rank_r(d))

Only ranks matter, so BM25 scores and cosine similarities (different scales) can be
combined without normalization. k = 60 is the value from the paper.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[Hashable, float]]:
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    for w, ranking in zip(weights, rankings, strict=True):
        seen: set[Hashable] = set()
        for rank, doc in enumerate(ranking, start=1):
            if doc in seen:  # ignore duplicates within one ranking
                continue
            seen.add(doc)
            scores[doc] = scores.get(doc, 0.0) + w / (k + rank)
            best_rank[doc] = min(best_rank.get(doc, rank), rank)
    # ties: better best-rank first, then stable by id for determinism
    return sorted(scores.items(), key=lambda kv: (-kv[1], best_rank[kv[0]], str(kv[0])))
