"""Evaluation metrics: NDCG@K and HR@K under FULL ranking.

Every non-interacted item is a candidate; no negative sampling in
evaluation.  Deterministic given (model, data, split).
"""

from typing import Dict, List

import numpy as np


def rank_of_target(scores: np.ndarray, seen: List[int], target: int) -> int:
    """1-based rank of the target among non-seen items (full ranking)."""
    masked = scores.copy()
    for i in seen:
        if 0 <= i < len(masked):
            masked[i] = -np.inf
    order = np.argsort(-masked)
    return int(np.where(order == target)[0][0]) + 1


def evaluate(rank_fn, train: Dict[int, List[int]],
             test: Dict[int, int], k: int = 10) -> Dict[str, float]:
    """Mean NDCG@K / HR@K over users with a held-out item."""
    ndcg, hits, n = 0.0, 0.0, 0
    for u, target in sorted(test.items()):
        seen = train[u]
        scores = rank_fn(u, seen)
        r = rank_of_target(scores, seen, target)
        if r <= k:
            hits += 1
            ndcg += 1.0 / np.log2(r + 1)
        n += 1
    return {"ndcg@10": ndcg / n, "hr@10": hits / n, "n_eval_users": n}


def paired_t_test(a: List[float], b: List[float]) -> Dict[str, float]:
    """Two-sided paired t-test over per-seed metric values."""
    from scipy import stats
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    t, p = stats.ttest_rel(a, b)
    return {"mean_diff": float((a - b).mean()), "t": float(t),
            "p": float(p), "n": len(a)}
