"""MovieLens-100K loading, leave-one-out split, sequence construction.

Deterministic given the seed: user order, interaction order (by
timestamp then id), and every random choice derive from one RNG.
"""

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

RATING_THRESHOLD = 4  # rating >= 4 is a positive interaction


def load_interactions(path: Path) -> List[Tuple[int, int, int]]:
    """Return (user, item, timestamp) triples, sorted deterministically."""
    triples = []
    with open(path) as fh:
        for line_no, line in enumerate(fh):
            parts = line.split()
            if len(parts) != 4:
                continue
            user, item, rating, ts = (int(parts[0]), int(parts[1]),
                                      int(parts[2]), int(parts[3]))
            if rating >= RATING_THRESHOLD:
                triples.append((user, item, ts, line_no))
    triples.sort(key=lambda t: (t[0], t[2], t[3]))
    return [(u, i) for u, i, _, _ in triples]


def remap(triples: List[Tuple[int, int]]):
    """Compact user/item ids to [0, n)."""
    users = sorted({u for u, _ in triples})
    items = sorted({i for _, i in triples})
    umap = {u: k for k, u in enumerate(users)}
    imap = {i: k for k, i in enumerate(items)}
    return [(umap[u], imap[i]) for u, i in triples], len(users), len(items)


def leave_one_out(
        triples: List[Tuple[int, int]], n_users: int
) -> Tuple[Dict[int, List[int]], Dict[int, int]]:
    """Per user: train sequence = all but the last interaction; test =
    the last interaction.  Returns (train_seqs, test_item)."""
    seqs: Dict[int, List[int]] = {u: [] for u in range(n_users)}
    for u, i in triples:
        seqs[u].append(i)
    train, test = {}, {}
    for u, items in seqs.items():
        if len(items) < 2:
            train[u] = items[:]
            continue  # cold user: nothing held out
        train[u] = items[:-1]
        test[u] = items[-1]
    return train, test


def popularity(train: Dict[int, List[int]], n_items: int) -> np.ndarray:
    """Item popularity counts over the real training set."""
    pop = np.zeros(n_items, dtype=np.int64)
    for items in train.values():
        for i in items:
            pop[i] += 1
    return pop


def dataset(path: Path):
    """Full deterministic pipeline: file -> (train, test, n_users, n_items)."""
    triples = load_interactions(path)
    compact, n_users, n_items = remap(triples)
    train, test = leave_one_out(compact, n_users)
    return train, test, n_users, n_items


def sparsify(train: Dict[int, List[int]], keep: float,
             rng: random.Random) -> Dict[int, List[int]]:
    """Simulate a sparser regime by keeping a fraction of train events."""
    out = {}
    for u, items in train.items():
        k = max(1, int(len(items) * keep))
        out[u] = sorted(rng.sample(items, k)) if len(items) > k else items[:]
    return out
