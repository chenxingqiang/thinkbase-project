"""Fidelity gating: the thesis's core innovation module.

Self-generated samples drift the training distribution: naive RSI
loops accept whatever the model itself ranks highest, so head items
amplify, coverage collapses, and error accumulates over rounds.  The
fidelity gate scores every candidate synthetic interaction along two
complementary axes and admits only samples that keep the *mixed*
distribution within a budget:

* axis 1 -- calibrated margin: the model's score margin between the
  candidate and the user's history-mean score, z-normalized per item
  popularity stratum so that tail items are not structurally
  disadvantaged (confidence alone prefers the head);
* axis 2 -- distribution distance: the JS divergence between the
  synthetic popularity distribution and the real one, measured on
  popularity strata (coarse bins), which is robust to noise and cheap.

An adaptive threshold keeps the mixed-distribution JS distance under a
per-round budget eps: if the current synthetic pool drifts too far,
the gate tightens; when it is healthy, it admits more.  This is the
mechanism that (in the experiments) turns the collapse of naive RSI
into a monotone, stable improvement curve.
"""

from typing import Dict, List, Tuple

import numpy as np

N_STRATA = 5  # popularity strata: quintiles of real popularity


def stratum_of(item: int, boundaries: np.ndarray) -> int:
    """Which popularity stratum an item belongs to (0 = least popular)."""
    return int(np.searchsorted(boundaries, item, side="right"))


def popularity_strata(pop: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Stratum boundaries (item ids sorted by popularity) and the
    stratum of every item."""
    order = np.argsort(-pop)                    # most popular first
    n = len(order)
    bounds = [order[int(k * n / N_STRATA) - 1] for k in range(1, N_STRATA)]
    boundaries = np.array(sorted(bounds))
    item_stratum = np.searchsorted(boundaries, np.arange(n), side="right")
    # reverse: high popularity -> low stratum id is *head*; remap so
    # stratum 0 = head quintile
    head_of = np.empty(n, dtype=np.int64)
    for s, items in enumerate([order[int(k * n / N_STRATA):
                                     int((k + 1) * n / N_STRATA)]
                               for k in range(N_STRATA)]):
        head_of[items] = s
    return boundaries, head_of


def stratum_distribution(items: List[int],
                          head_of: np.ndarray) -> np.ndarray:
    """Fraction of samples per popularity stratum (length N_STRATA)."""
    counts = np.zeros(N_STRATA)
    for i in items:
        counts[head_of[i]] += 1
    total = counts.sum()
    return counts / total if total else counts


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence), symmetric."""
    p = p + 1e-12
    q = q + 1e-12
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        return float(np.sum(a * np.log(a / b)))

    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def calibrate_margins(candidates: List[Tuple[int, int, float, float]],
                       head_of: np.ndarray) -> List[Tuple[int, int, float, float]]:
    """Z-normalize margins per stratum: (user, item, margin, raw) ->
    (user, item, z_margin, raw).  Confidence alone favors the head;
    per-stratum normalization removes that structural bias."""
    if not candidates:
        return []
    arr = np.array([c[2] for c in candidates])
    out = []
    for s in range(N_STRATA):
        idx = [k for k, c in enumerate(candidates) if head_of[c[1]] == s]
        if len(idx) < 2:
            for k in idx:
                out.append((candidates[k][0], candidates[k][1],
                            0.0 if len(idx) < 2 else 0.0, candidates[k][3]))
            continue
        vals = arr[idx]
        mu, sigma = vals.mean(), vals.std() + 1e-9
        for k in idx:
            u, i, m, raw = candidates[k]
            out.append((u, i, float((m - mu) / sigma), raw))
    return out


class FidelityGate:
    """Adaptive per-round gate over synthetic candidate interactions.

    Parameters
    ----------
    eps : float
        JS-distance budget for the mixed (real + synthetic) popularity
        distribution per round.
    z_min : float
        Floor on the calibrated margin: candidates below it never pass,
        however healthy the distribution looks.
    mix_lambda : float
        Synthetic samples are weighted by this factor at training time.
    """

    def __init__(self, eps: float = 0.08, z_min: float = 0.0,
                 mix_lambda: float = 0.5):
        self.eps, self.z_min, self.mix_lambda = eps, z_min, mix_lambda
        self.boundaries: np.ndarray = None
        self.head_of: np.ndarray = None
        self.real_dist: np.ndarray = None

    def fit(self, pop: np.ndarray) -> "FidelityGate":
        self.boundaries, self.head_of = popularity_strata(pop)
        self.real_dist = stratum_distribution(
            np.where(pop > 0)[0].repeat(
                np.maximum(pop[np.where(pop > 0)], 1).astype(int)),
            self.head_of) if pop.any() else None
        # simpler + cheaper: distribution over *events* by stratum
        counts = np.zeros(N_STRATA)
        for item in range(len(pop)):
            counts[self.head_of[item]] += pop[item]
        total = counts.sum()
        self.real_dist = counts / total if total else counts
        return self

    def score_and_filter(
            self, candidates: List[Tuple[int, int, float, float]],
            real_n: int, verbose: bool = False) -> Tuple[List, Dict]:
        """Admit candidates that keep the mixed JS distance under eps.

        Returns (admitted, stats).  Candidates are processed in
        calibrated-margin order (best first); admission stops at the
        first candidate whose addition would breach the budget, then
        continues scanning (the budget check is per-insertion).
        """
        cal = calibrate_margins(candidates, self.head_of)
        cal = [c for c in cal if c[2] >= self.z_min]
        cal.sort(key=lambda c: -c[2])

        admitted: List[Tuple[int, int]] = []
        for u, i, z, raw in cal:
            proj = stratum_distribution(
                [i for _, i in admitted] + [i], self.head_of)
            mixed = ((real_n) * self.real_dist
                    + self.mix_lambda * len(admitted + [(u, i)]) * proj) \
                / (real_n + self.mix_lambda * len(admitted + [(u, i)]))
            if js_distance(mixed, self.real_dist) > self.eps:
                continue
            admitted.append((u, i))
        # report the JS of the ACTUAL admitted mixture, not of the
        # last candidate checked (which may have been rejected)
        if admitted:
            proj = stratum_distribution(
                [i for _, i in admitted], self.head_of)
            mixed = ((real_n) * self.real_dist
                    + self.mix_lambda * len(admitted) * proj) \
                / (real_n + self.mix_lambda * len(admitted))
            js = js_distance(mixed, self.real_dist)
        else:
            js = 0.0
        stats = {
            "n_candidates": len(candidates),
            "n_admitted": len(admitted),
            "admission_rate": len(admitted) / max(1, len(candidates)),
            "mixed_js_distance": js,
            "thresholds": {"eps": self.eps, "z_min": self.z_min},
        }
        return admitted, stats

    # -- ablation variants ------------------------------------------------
    def filter_confidence_only(
            self, candidates, real_n: int) -> Tuple[List, Dict]:
        """Ablation A: margin-only gate (no distribution budget)."""
        cal = calibrate_margins(candidates, self.head_of)
        cal = [c for c in cal if c[2] >= self.z_min]
        cal.sort(key=lambda c: -c[2])
        admitted = [(u, i) for u, i, z, _ in cal]
        dist = stratum_distribution([i for _, i in admitted], self.head_of)
        return admitted, {
            "n_candidates": len(candidates), "n_admitted": len(admitted),
            "admission_rate": len(admitted) / max(1, len(candidates)),
            "synthetic_js_distance": js_distance(dist, self.real_dist),
            "thresholds": {"eps": None, "z_min": self.z_min}}

    def filter_no_adaptive(
            self, candidates, real_n: int) -> Tuple[List, Dict]:
        """Ablation B: fixed 50% admission by margin, no JS budget."""
        cal = calibrate_margins(candidates, self.head_of)
        cal.sort(key=lambda c: -c[2])
        keep = len(cal) // 2
        admitted = [(u, i) for u, i, _, _ in cal[:keep]]
        dist = stratum_distribution([i for _, i in admitted], self.head_of)
        return admitted, {
            "n_candidates": len(candidates), "n_admitted": len(admitted),
            "admission_rate": len(admitted) / max(1, len(candidates)),
            "synthetic_js_distance": js_distance(dist, self.real_dist),
            "thresholds": {"eps": "fixed-50%", "z_min": None}}
