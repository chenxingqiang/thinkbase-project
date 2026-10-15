"""The RSI closed loop: generate -> gate -> mix -> retrain -> evaluate.

One round of recursive self-improvement.  Variants:

* ``no_rsi``     -- no self-generated data at all (the base model)
* ``naive``      -- accept every generated sample, fixed mix (the
                     classic self-training trap)
* ``conf_only``  -- margin gate without distribution control
* ``gated``      -- the full fidelity gate (ours)
* ``no_adaptive``-- fixed 50% admission without the JS budget
"""

import random
from typing import Callable, Dict, List

import numpy as np

from . import fidelity as fid
from .metrics import evaluate


def generate_candidates(model, train: Dict[int, List[int]],
                        topk: int = 10) -> List[tuple]:
    """Model self-generation: per user, its top-k unseen items.

    Returns (user, item, margin, score) where margin = score minus the
    mean score over the user's history items (a per-user calibrated
    confidence).
    """
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


def run_rsi_loop(model_factory: Callable, train: Dict[int, List[int]],
                 test: Dict[int, int], n_items: int,
                 rounds: int = 5, variant: str = "gated",
                 seed: int = 0, epochs_init: int = 20, epochs_round: int = 8,
                 topk: int = 10, eps: float = 0.08, z_min: float = 0.0,
                 mix_lambda: float = 0.5,
                 gate: fid.FidelityGate = None) -> Dict:
    """Run the closed loop and record every round's measurements.

    Deterministic: one RNG for the whole run, single-thread torch,
    seeds threaded into the model factory.
    """
    rng = random.Random(seed)
    pop = np.zeros(n_items, dtype=np.int64)
    for items in train.values():
        for i in items:
            pop[i] += 1
    if gate is None:
        gate = fid.FidelityGate(eps=eps, z_min=z_min,
                               mix_lambda=mix_lambda).fit(pop)
    real_n = int(pop.sum())

    model = model_factory(seed=seed)
    model.fit(train, epochs=epochs_init, rng=rng)
    history = {"rounds": [{
        "round": 0,
        "metrics": evaluate(model.rank, train, test),
        "synthetic": {"n_admitted": 0, "js": 0.0},
    }]}

    for r in range(1, rounds + 1):
        candidates = generate_candidates(model, train, topk=topk)
        if variant == "no_rsi":
            admitted, stats = [], {"n_admitted": 0}
        elif variant == "naive":
            admitted = [(u, i) for u, i, _, _ in candidates]
            dist = fid.stratum_distribution(
                [i for _, i in admitted], gate.head_of)
            stats = {"n_admitted": len(admitted),
                     "synthetic_js_distance":
                         fid.js_distance(dist, gate.real_dist)}
        elif variant == "conf_only":
            admitted, stats = gate.filter_confidence_only(
                candidates, real_n)
        elif variant == "no_adaptive":
            admitted, stats = gate.filter_no_adaptive(candidates, real_n)
        else:  # gated (ours)
            admitted, stats = gate.score_and_filter(candidates, real_n)

        # mix: real + lambda-weighted synthetic.  Weighting is realized
        # by repetition (integer replication keeps CPU training simple
        # and deterministic).
        mixed = {u: list(items) for u, items in train.items()}
        reps = max(1, round(mix_lambda))
        for u, i in admitted:
            for _ in range(reps):
                mixed.setdefault(u, []).append(i)

        if variant == "no_rsi":
            model.fit(train, epochs=epochs_round, rng=rng)
            mixed_train = train
        else:
            model.fit(mixed, epochs=epochs_round, rng=rng)
            mixed_train = mixed
        history["rounds"].append({
            "round": r,
            "metrics": evaluate(model.rank, mixed_train, test),
            "synthetic": stats,
        })
    return history
