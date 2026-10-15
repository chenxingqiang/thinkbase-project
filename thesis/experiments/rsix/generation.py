"""Self-generation mechanisms for the RSI loop.

Two generators are compared in the thesis:

* ``topk_candidates`` -- the naive generator: the model's own top-k
  predictions per user.  Training on argmax-like self-output is
  self-reinforcing and collapses even under a popularity-budget gate
  (the motivation experiment in Chapter 4).
* ``consistency_candidates`` -- FixMatch-style pseudo-labeling
  (Sohn et al., NeurIPS 2020) adapted to sequential recommendation:
  per user, K augmented views of the history (item dropout + tail
  crop) are scored by the current model; a synthetic interaction is
  admitted only if the views AGREE on the same top item with
  confidence >= tau and margin >= m.  Pseudo-labels that survive
  augmentation-consistency carry information about robust beliefs
  rather than amplifying the model's head bias.
"""

import random
from typing import Dict, List, Tuple

import numpy as np

View = List[int]


def augment_views(seq: List[int], k: int, rng: random.Random,
                  dropout: float = 0.1, crop: float = 0.9) -> List[View]:
    """K augmented views: random item dropout then tail crop."""
    views = []
    for _ in range(k):
        v = [i for i in seq if rng.random() >= dropout] or seq[-1:]
        keep = max(1, int(len(v) * crop))
        views.append(v[-keep:])
    return views


def topk_candidates(model, train: Dict[int, List[int]],
                    topk: int = 10) -> List[Tuple[int, int, float, float]]:
    """Naive generator: per-user top-k unseen items (margin vs history mean)."""
    out = []
    for u, seen in sorted(train.items()):
        if not seen:
            continue
        scores = model.rank(u, seen)
        masked = scores.copy()
        for i in seen:
            masked[i] = -np.inf
        top = np.argsort(-masked)[:topk]
        hist_mean = float(np.mean([scores[i] for i in seen]))
        for i in top:
            if np.isfinite(masked[i]):
                out.append((u, int(i), float(masked[i] - hist_mean),
                            float(masked[i])))
    return out


def consistency_candidates(model, train: Dict[int, List[int]],
                           n_views: int = 3, tau: float = 0.01,
                           margin: float = 0.002,
                           rng: random.Random = None
                           ) -> Tuple[List[Tuple[int, int, float, float]], Dict]:
    """FixMatch-style generator: augmentation-consistent pseudo-labels.

    For every user: score K augmented views; the pseudo-label is the
    view-agreed top item among non-seen items, admitted when (a) all
    views agree, (b) mean top probability >= tau, and (c) the top-1 vs
    top-2 probability margin >= margin.  Returns candidates plus
    per-user admission statistics.
    """
    rng = rng or random.Random(0)
    out: List[Tuple[int, int, float, float]] = []
    stats = {"users": 0, "users_consistent": 0, "candidates": 0}
    for u, seen in sorted(train.items()):
        if len(seen) < 2:
            continue
        stats["users"] += 1
        views = augment_views(seen, n_views, rng)
        probs, top_items = [], []
        for v in views:
            scores = model.rank(u, v)
            for i in set(v):
                scores[i] = -np.inf
            for i in seen:  # never recommend the true history
                scores[i] = -np.inf
            p = np.exp(scores - scores.max())
            p = p / p.sum()
            order = np.argsort(-p)
            probs.append(p)
            top_items.append(int(order[0]))
        if len(set(top_items)) != 1:
            continue  # views disagree -> no pseudo-label
        item = top_items[0]
        mean_p = float(np.mean([p[item] for p in probs]))
        second = float(np.mean([np.sort(p)[-2] for p in probs]))
        if mean_p < tau or (mean_p - second) < margin:
            continue
        stats["users_consistent"] += 1
        stats["candidates"] += 1
        out.append((u, item, mean_p - second, mean_p))
    return out, stats
