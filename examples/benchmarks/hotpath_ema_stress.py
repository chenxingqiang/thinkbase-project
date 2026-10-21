"""Hot-path detector stress: exact matching vs EMA on diverse trajectory streams.

Compares the shipped exact-identity counter/threshold detector
(``DynamicOptimizer(detection_mode="exact")`` -- supplementary D.4's
pre-remedy baseline, discussed in AgentISA.md Section 4.2.2) against the
implemented windowed EMA detector (``detection_mode="ema"``,
supplementary D.4: per-path EMA counter with decay gamma = 2/(window+1),
hot iff recent execution share counter[h]/sum(counter) exceeds theta and
stability clears the gate), across six seeded trajectory-stream shapes
with a brute-force ORACLE detector providing the ground truth of which
paths ought to be detected:

* ``concentrated``     one dominant path (30 of every 32 recordings);
* ``diverse_rotated``  three variants of one family in round robin, 40
                       recordings each -- far below the absolute
                       threshold 100; the core Section 4.2.2 scenario;
* ``diverse_zipf``     10 distinct variants with Zipf(1.1)-shaped
                       occurrence counts;
* ``cyclic_shift``     K=4 near-clone variants (family-shared names,
                       variant-distinct ids), cycled 40 each;
* ``bursty``           two phases: 80 diverse one-offs, then a hot
                       burst of 80 on one path;
* ``pure_diversity``   160 distinct one-off trajectories -- the
                       false-positive control (nothing may be hot).

Per seed (rng 1000+k, default 5 seeds), per scenario, and per detector
(exact, ema), the harness records every promotion event: the 1-based
recording index at which each path first enters ``hot_paths``, plus how
many recordings of that path that took.  Because the manager serves the
re-optimized trace from promotion on, the promotion index doubles as
the adoption latency (full-form executions = promotion index - 1); the
bursty scenario additionally measures adoption end-to-end through
``OptimizationManager.record_execution`` (detection -> trace
re-optimization -> first shrunken-form serve).  All aggregates (means,
standard deviations, recalls, ratios) are computed by this script from
those raw events; nothing is hand-computed.

A cap sweep (on by default) measures LRU eviction under
``max_tracked_paths`` caps 2000/1000/500/200/100 on a larger stress
stream (a Zipf(1.1) mix of ~1900 recordings with 120 one-off
trajectories interleaved -- more distinct paths than the smallest cap,
so the <= cap bounds are non-vacuous): final size of every per-path
mapping (counts, EMA counters, stats, hot paths, recency), eviction
counts, and which oracle-hot paths lost hot status through eviction.

Deterministic oracle rule (no free parameters): a path is hot at
recording n iff it has >= 2 recordings so far (repeat -- matching the
EMA detector's cold-start guard) and its cumulative share of all
recordings so far exceeds theta.  Both detectors' sensitivity/recall
and false promotion counts are reported against it.  The oracle uses
cumulative share where the EMA detector uses recent share, so early-
stream share inflation affects both; differences are reported honestly.

Honesty rules: every number in the output JSON is computed by this
script from recorded events; no number is hand-entered.  Wall-clock is
reported once, machine-dependent, as run metadata only.

Usage:
    python hotpath_ema_stress.py                     # full run, 5 seeds
    python hotpath_ema_stress.py --seeds 1 --quick   # smoke run
    python hotpath_ema_stress.py --output hotpath_ema_results.json
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.instruction import Instruction, InstructionType
from agentflow.core.isa.instruction_optimization import (
    DynamicOptimizer,
    OptimizationManager,
    OptimizationMetrics,
)

# ---------------------------------------------------------------------------
# Fixed detector parameters (identical for both arms of every comparison).
# ---------------------------------------------------------------------------

HOT_PATH_THRESHOLD = 100   # shipped default: exact detector needs > 100
STABILITY_THRESHOLD = 0.9  # shipped default, shared by both detectors
EMA_WINDOW = 10            # D.4 window -> gamma = 2/11
EMA_THRESHOLD = 0.3        # theta: recent execution share must exceed this
N_SEEDS_DEFAULT = 5        # the mandated seed count
SEED_BASE = 1000           # rng seeds are 1000 + k, k in range(n_seeds)
CAPS: Tuple[int, ...] = (2000, 1000, 500, 200, 100)
CAP_RESCUE_CAPS: Tuple[int, ...] = (1000, 100, 50, 20, 15, 12, 10)
DEFAULT_CAP = 1000         # shipped default max_tracked_paths
TRACE_LEN = 4              # instructions per trajectory variant


class _TraceInstr(Instruction):
    """Minimal concrete instruction for trace formation/adoption.

    Side-effect-free with a declared output (so dead-code elimination
    keeps it) and an empty, pairwise-compatible parameter schema (so the
    combining pass can actually fuse it -- adoption is observable).
    """

    def __init__(self, id: str, name: str):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Stress instruction {name}",
            side_effects=False,
            outputs=[f"{name}_out"],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"status": "executed", "id": self.id, f"{self.name}_out": True}


def variant_instructions(family: str, variant: str,
                         length: int = TRACE_LEN) -> List[Instruction]:
    """A distinct instruction sequence for one trajectory variant.

    All variants of a family share instruction NAMES (the family's
    common shape) but carry variant-distinct ids, so their path_ids --
    the exact-sequence identity key -- differ while the name-level
    instruction identity stays family-shared: exactly the "similar but
    non-identical sequences" diversity of Section 4.2.2.
    """
    names = [f"{family}_s{i}" for i in range(length - 1)] + [f"{family}_tail"]
    return [
        _TraceInstr(id=f"{variant}_{i}", name=names[i])
        for i in range(length)
    ]


def stable_metrics() -> OptimizationMetrics:
    return OptimizationMetrics(
        execution_time=1.0, memory_usage=1.0, api_calls=0, success_rate=1.0,
    )


# ---------------------------------------------------------------------------
# Seeded scenario streams.  Each returns a list of path_ids.
# ---------------------------------------------------------------------------

def stream_concentrated(rng: random.Random) -> List[str]:
    """One dominant path (30 of every 32) + light background diversity."""
    background = [f"bg{i}" for i in range(30)]
    stream: List[str] = []
    for i in range(512):
        if i % 32 < 30:
            stream.append("dom")
        else:
            stream.append(background[rng.randrange(len(background))])
    return stream


def stream_diverse_rotated(rng: random.Random) -> List[str]:
    """Three family variants in round robin, 40 recordings each."""
    return [f"v{(i % 3) + 1}" for i in range(120)]


def _zipf_counts(n: int, n_ranks: int = 10, s: float = 1.1) -> Dict[str, int]:
    """Zipf(s) occurrence counts over n_ranks variant labels, summing ~n."""
    weights = [1.0 / (r ** s) for r in range(1, n_ranks + 1)]
    total = sum(weights)
    return {
        f"z{r}": int(round(n * w / total))
        for r, w in zip(range(1, n_ranks + 1), weights)
    }


def stream_diverse_zipf(rng: random.Random) -> List[str]:
    """10 variants with Zipf(1.1)-shaped occurrence counts, shuffled."""
    counts = _zipf_counts(160)
    stream = [p for p, c in counts.items() for _ in range(c)]
    rng.shuffle(stream)
    return stream


def stream_cyclic_shift(rng: random.Random) -> List[str]:
    """K=4 near-clone variants, cycled (40 recordings each)."""
    return [f"c{i % 4}" for i in range(160)]


def stream_bursty(rng: random.Random) -> List[str]:
    """Diverse warm-up (8 one-offs), then a hot burst (80 on one path)."""
    return [f"w{i}" for i in range(8)] + ["burst"] * 80


def stream_warmup_heavy(rng: random.Random) -> List[str]:
    """Heavy diverse warm-up (80 one-offs), then a hot burst (80).

    Under the implemented per-path EMA (decay applies only when a path
    is itself recorded; the denominator sums every tracked path's
    counter), the 80 warm-up paths freeze at counter ~1.0 and hold the
    share denominator up: the burst's share saturates far below theta,
    so the EMA detector stays cold unless the tracked-paths cap evicts
    the stale warm-up and shrinks the denominator.  This scenario
    measures that floor effect and its cap remedy (cap_rescue block).
    """
    return [f"u{i}" for i in range(80)] + ["burst"] * 80


def stream_pure_diversity(rng: random.Random) -> List[str]:
    """160 distinct one-off trajectories: the false-positive control."""
    return [f"p{i}" for i in range(160)]


SCENARIOS: Tuple[Tuple[str, Callable[[random.Random], List[str]]], ...] = (
    ("concentrated", stream_concentrated),
    ("diverse_rotated", stream_diverse_rotated),
    ("diverse_zipf", stream_diverse_zipf),
    ("cyclic_shift", stream_cyclic_shift),
    ("bursty", stream_bursty),
    ("warmup_heavy_burst", stream_warmup_heavy),
    ("pure_diversity", stream_pure_diversity),
)


def stream_cap_stress(rng: random.Random, zipf_budget: int = 800,
                      n_oneoffs: int = 2200) -> List[str]:
    """Stress stream with far more distinct paths than every swept cap.

    ``n_oneoffs`` one-off trajectories are interleaved into a
    Zipf(1.1) mix of ``zipf_budget`` recordings, so LRU evictions hit
    while the Zipf paths are still being re-recorded (churn, not just
    FIFO tail shedding).  Distinct paths = n_oneoffs + 10 > 2000, so
    every swept cap (down to 100) genuinely evicts.
    """
    oneoffs = [f"o{i}" for i in range(n_oneoffs)]
    counts = _zipf_counts(zipf_budget)
    zipf_part = [p for p, c in counts.items() for _ in range(c)]
    rng.shuffle(zipf_part)
    chunk = max(1, zipf_budget // n_oneoffs)
    stream: List[str] = []
    zi = 0
    for oneoff in oneoffs:
        stream.extend(zipf_part[zi:zi + chunk])
        zi += chunk
        stream.append(oneoff)
    stream.extend(zipf_part[zi:])
    return stream


# ---------------------------------------------------------------------------
# Detectors and the brute-force oracle.
# ---------------------------------------------------------------------------

def _detector_kwargs(mode: str, cap: int) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = dict(
        hot_path_threshold=HOT_PATH_THRESHOLD,
        stability_threshold=STABILITY_THRESHOLD,
        detection_mode=mode,
        max_tracked_paths=cap,
    )
    if mode == "ema":
        kwargs.update(ema_window=EMA_WINDOW, ema_threshold=EMA_THRESHOLD)
    return kwargs


def make_detector(mode: str, cap: int = DEFAULT_CAP) -> DynamicOptimizer:
    """Construct one detector arm with the shared fixed parameters."""
    return DynamicOptimizer(**_detector_kwargs(mode, cap))


def make_manager(mode: str, cap: int = DEFAULT_CAP) -> OptimizationManager:
    """Manager wiring the same detector parameters (adoption path)."""
    return OptimizationManager(**_detector_kwargs(mode, cap))


class OracleDetector:
    """Brute-force ground truth: which paths ought to be detected.

    Deterministic, parameter-free rule: a path is hot at recording n iff
    it has at least 2 recordings so far (repeat -- matching the EMA
    detector's cold-start guard) and its cumulative share of ALL
    recordings so far exceeds theta.  The oracle reads only the stream,
    never the detectors; it defines the sensitivity/recall and false-
    promotion reference for both detectors.
    """

    def __init__(self, threshold: float = EMA_THRESHOLD):
        self.threshold = threshold
        self.counts: Dict[str, int] = {}
        self.total = 0
        self._hot: set = set()

    @property
    def hot_paths(self) -> set:
        """Set-shaped alias so membership checks mirror the detectors."""
        return self._hot

    def record(self, path_id: str) -> bool:
        self.counts[path_id] = self.counts.get(path_id, 0) + 1
        self.total += 1
        if self.counts[path_id] >= 2 and (
                self.counts[path_id] / self.total) > self.threshold:
            self._hot.add(path_id)
            return True
        return False


# ---------------------------------------------------------------------------
# Detection cell: both detectors + oracle over one (seed, scenario) stream.
# ---------------------------------------------------------------------------

def run_detection_cell(stream: Sequence[str],
                       instr_for: Dict[str, List[Instruction]],
                       ) -> Dict[str, Any]:
    """Run exact + ema + oracle over one stream; return raw promotion events.

    An event is (path_id, 1-based recording index of first promotion,
    recordings of that path up to and including the promotion).
    """
    detectors: Dict[str, Any] = {
        "exact": make_detector("exact"),
        "ema": make_detector("ema"),
        "oracle": OracleDetector(),
    }
    events: Dict[str, List[Tuple[str, int, int]]] = {
        "exact": [], "ema": [], "oracle": [],
    }
    per_path_recordings: Dict[str, int] = {}
    for idx, path_id in enumerate(stream):
        per_path_recordings[path_id] = per_path_recordings.get(path_id, 0) + 1
        instrs = instr_for[path_id]
        detectors["exact"].record_execution(path_id, instrs, stable_metrics())
        detectors["ema"].record_execution(path_id, instrs, stable_metrics())
        detectors["oracle"].record(path_id)
        for who in ("exact", "ema", "oracle"):
            if path_id in detectors[who].hot_paths:
                if not any(e[0] == path_id for e in events[who]):
                    events[who].append(
                        (path_id, idx + 1, per_path_recordings[path_id])
                    )
    return {
        "stream_len": len(stream),
        "distinct_paths": len(per_path_recordings),
        "events": events,
    }


# ---------------------------------------------------------------------------
# Adoption cell: detection -> shrunken-trace serve through the manager.
# ---------------------------------------------------------------------------

def run_adoption_cell(stream: Sequence[str],
                      instr_for: Dict[str, List[Instruction]],
                      mode: str) -> Dict[str, Any]:
    """Measure adoption latency (first shrunken serve) for one mode.

    ``OptimizationManager.record_execution`` re-optimizes the hot path's
    trace and serves it from promotion on; adoption latency is the
    1-based recording index of the first serve that is strictly shorter
    than the recorded full form.  Full-form executions before adoption
    are that index minus one.
    """
    manager = make_manager(mode)
    # The manager keys hot paths by the joined instruction ids, not by
    # the stream's path label -- mirror OptimizationManager's own
    # convention so membership checks are against real keys.
    burst_instrs = instr_for.get("burst")
    burst_key = ("_".join(i.id for i in burst_instrs)
                 if burst_instrs is not None else None)
    detection_index: Optional[int] = None
    first_adoption: Optional[int] = None
    full_len: Optional[int] = None
    served_len_at_adoption: Optional[int] = None
    for idx, path_id in enumerate(stream):
        instrs = instr_for[path_id]
        if path_id == "burst" and full_len is None:
            full_len = len(instrs)
        result = manager.record_execution(instrs, stable_metrics())
        if (detection_index is None and path_id == "burst"
                and burst_key is not None
                and burst_key in manager.dynamic_optimizer.hot_paths):
            detection_index = idx + 1
        if (first_adoption is None and path_id == "burst"
                and full_len is not None and len(result) < full_len):
            first_adoption = idx + 1
            served_len_at_adoption = len(result)
    adopted: Optional[int] = None
    if burst_key is not None:
        optimized = manager.dynamic_optimizer.optimize_hot_path(burst_key)
        adopted = len(optimized) if optimized is not None else None
    return {
        "mode": mode,
        "detection_index": detection_index,
        "first_adoption_index": first_adoption,
        "full_form_executions_before_adoption": (
            first_adoption - 1 if first_adoption is not None else None),
        "full_trace_len": full_len,
        "adopted_trace_len": adopted,
        "served_len_at_adoption": served_len_at_adoption,
    }


# ---------------------------------------------------------------------------
# Cap sweep cell: LRU eviction under max_tracked_paths.
# ---------------------------------------------------------------------------

def run_cap_cell(stream: Sequence[str], cap: int,
                 instr_for: Dict[str, List[Instruction]]) -> Dict[str, Any]:
    """One (cap) cell: both detector modes share the stream and the cap.

    Records, per mode: final size of every per-path mapping, eviction
    count (distinct recorded - retained), whether every mapping stayed
    within the cap, and for ema the oracle-hot paths that lost hot
    status through eviction.
    """
    detectors = {
        "exact": make_detector("exact", cap=cap),
        "ema": make_detector("ema", cap=cap),
    }
    oracle = OracleDetector()
    per_path_recordings: Dict[str, int] = {}
    ever_hot: Dict[str, set] = {"exact": set(), "ema": set()}
    for path_id in stream:
        per_path_recordings[path_id] = per_path_recordings.get(path_id, 0) + 1
        instrs = instr_for[path_id]
        detectors["exact"].record_execution(path_id, instrs, stable_metrics())
        detectors["ema"].record_execution(path_id, instrs, stable_metrics())
        oracle.record(path_id)
        for mode in ("exact", "ema"):
            if path_id in detectors[mode].hot_paths:
                ever_hot[mode].add(path_id)
    results: Dict[str, Any] = {
        "stream_len": len(stream),
        "oracle_hot": sorted(oracle.hot_paths),
    }
    for mode in ("exact", "ema"):
        det = detectors[mode]
        retained = set(det.execution_counts)
        row: Dict[str, Any] = {
            "cap": cap,
            "distinct_paths_recorded": len(per_path_recordings),
            "final_counts_size": len(det.execution_counts),
            "final_ema_counters_size": len(det.ema_counters),
            "final_stats_size": len(det.performance_stats),
            "final_hot_paths_size": len(det.hot_paths),
            "final_recency_size": len(det._path_recency),
            "ever_hot_size": len(ever_hot[mode]),
            "hot_lost_after_eviction": sorted(ever_hot[mode] - set(det.hot_paths)),
            "evictions": len(per_path_recordings) - len(retained),
            "bound_respected": (
                len(det.execution_counts) <= cap
                and len(det.ema_counters) <= cap
                and len(det.performance_stats) <= cap
                and len(det.hot_paths) <= cap
                and len(det._path_recency) <= cap
            ),
        }
        if mode == "ema":
            row["oracle_hot_missing_from_final"] = sorted(
                oracle.hot_paths - set(det.hot_paths))
            row["oracle_hot_never_promoted"] = sorted(
                oracle.hot_paths - ever_hot[mode])
            row["final_hot_set"] = sorted(det.hot_paths)
        results[mode] = row
    return results


# ---------------------------------------------------------------------------
# Cap-rescue cell: eviction shrinking the share denominator restores
# detection on the warmup_heavy_burst stream.
# ---------------------------------------------------------------------------

def run_cap_rescue_cell(stream: Sequence[str], cap: int,
                        instr_for: Dict[str, List[Instruction]],
                        ) -> Dict[str, Any]:
    """One (cap) cell of the cap-rescue measurement (EMA focus).

    Under the shipped default cap the warm-up paths stay tracked and
    the burst's recent share saturates below theta; a small cap evicts
    the stale warm-up LRU-first, shrinking the denominator so the burst
    can cross theta.  Records the burst's first EMA promotion index
    (None if never), the final denominator, and eviction count.
    """
    ema = make_detector("ema", cap=cap)
    exact = make_detector("exact", cap=cap)
    burst_detection: Optional[int] = None
    per_path_recordings: Dict[str, int] = {}
    for idx, path_id in enumerate(stream):
        per_path_recordings[path_id] = per_path_recordings.get(path_id, 0) + 1
        instrs = instr_for[path_id]
        ema.record_execution(path_id, instrs, stable_metrics())
        exact.record_execution(path_id, instrs, stable_metrics())
        if (burst_detection is None and path_id == "burst"
                and "burst" in ema.hot_paths):
            burst_detection = idx + 1
    return {
        "cap": cap,
        "stream_len": len(stream),
        "distinct_paths_recorded": len(per_path_recordings),
        "ema_burst_detection_index": burst_detection,
        "ema_final_denominator": sum(ema.ema_counters.values()),
        "ema_final_hot_set": sorted(ema.hot_paths),
        "exact_promoted_count": len(exact.hot_paths),
        "evictions": len(per_path_recordings) - len(ema.execution_counts),
    }


# ---------------------------------------------------------------------------
# Aggregation: everything computed from raw events by this script.
# ---------------------------------------------------------------------------

def _ev_dicts(events: List[Tuple[str, int, int]]) -> List[Dict[str, Any]]:
    return [
        {"path": p, "at_recording": i, "recordings_of_path": c}
        for p, i, c in events
    ]


def _mean_stdev(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "stdev": None, "n": 0}
    if len(values) == 1:
        return {"mean": statistics.fmean(values), "stdev": 0.0, "n": 1}
    return {
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values),
        "n": len(values),
    }


def _per_path_first_promotions(
        events_per_cell: List[List[Tuple[str, int, int]]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean/stdev of each promoted path's first promotion index."""
    merged: Dict[str, List[int]] = {}
    for events in events_per_cell:
        for p, i, _ in events:
            merged.setdefault(p, []).append(i)
    return {p: _mean_stdev([float(x) for x in v])
            for p, v in sorted(merged.items())}


def summarize_scenario(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed summary of one scenario from its per-seed raw cells."""
    per_seed: Dict[str, Any] = {}
    for cell in cells:
        raw = cell["raw"]
        per_seed[str(cell["seed"])] = {
            "stream_len": raw["stream_len"],
            "distinct_paths": raw["distinct_paths"],
            "exact_events": _ev_dicts(raw["events"]["exact"]),
            "ema_events": _ev_dicts(raw["events"]["ema"]),
            "oracle_events": _ev_dicts(raw["events"]["oracle"]),
        }
    exact_promoted = [float(len(c["raw"]["events"]["exact"])) for c in cells]
    ema_promoted = [float(len(c["raw"]["events"]["ema"])) for c in cells]
    oracle_promoted = [float(len(c["raw"]["events"]["oracle"])) for c in cells]
    exact_first = [float(c["raw"]["events"]["exact"][0][1]) for c in cells
                   if c["raw"]["events"]["exact"]]
    ema_first = [float(c["raw"]["events"]["ema"][0][1]) for c in cells
                 if c["raw"]["events"]["ema"]]

    recalls_exact: List[float] = []
    recalls_ema: List[float] = []
    fp_exact: List[float] = []
    fp_ema: List[float] = []
    for c in cells:
        oracle_set = {e[0] for e in c["raw"]["events"]["oracle"]}
        exact_set = {e[0] for e in c["raw"]["events"]["exact"]}
        ema_set = {e[0] for e in c["raw"]["events"]["ema"]}
        if oracle_set:
            recalls_exact.append(len(exact_set & oracle_set) / len(oracle_set))
            recalls_ema.append(len(ema_set & oracle_set) / len(oracle_set))
        else:
            recalls_exact.append(0.0)
            recalls_ema.append(0.0)
        fp_exact.append(float(len(exact_set - oracle_set)))
        fp_ema.append(float(len(ema_set - oracle_set)))

    return {
        "n_seeds": len(cells),
        "stream_len": cells[0]["raw"]["stream_len"] if cells else None,
        "distinct_paths": cells[0]["raw"]["distinct_paths"] if cells else None,
        "exact_promoted": _mean_stdev(exact_promoted),
        "ema_promoted": _mean_stdev(ema_promoted),
        "oracle_promoted": _mean_stdev(oracle_promoted),
        "exact_first_promotion_recording": _mean_stdev(exact_first),
        "ema_first_promotion_recording": _mean_stdev(ema_first),
        "exact_recall_vs_oracle": _mean_stdev(recalls_exact),
        "ema_recall_vs_oracle": _mean_stdev(recalls_ema),
        "exact_false_promotions_vs_oracle": _mean_stdev(fp_exact),
        "ema_false_promotions_vs_oracle": _mean_stdev(fp_ema),
        "exact_promotion_latency_per_path": _per_path_first_promotions(
            [c["raw"]["events"]["exact"] for c in cells]),
        "ema_promotion_latency_per_path": _per_path_first_promotions(
            [c["raw"]["events"]["ema"] for c in cells]),
        "per_seed": per_seed,
    }


def summarize_adoption(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed adoption latency per mode (bursty scenario)."""
    out: Dict[str, Any] = {}
    for mode in ("exact", "ema"):
        rows = [c["raw"][mode] for c in cells if mode in c["raw"]]
        det_idx = [float(r["detection_index"]) for r in rows
                   if r["detection_index"] is not None]
        ado_idx = [float(r["first_adoption_index"]) for r in rows
                   if r["first_adoption_index"] is not None]
        full_before = [float(r["full_form_executions_before_adoption"])
                       for r in rows
                       if r["full_form_executions_before_adoption"] is not None]
        out[mode] = {
            "n_seeds": len(rows),
            "detected_seeds": len(det_idx),
            "detection_index": _mean_stdev(det_idx),
            "adopted_seeds": len(ado_idx),
            "first_adoption_index": _mean_stdev(ado_idx),
            "full_form_executions_before_adoption": _mean_stdev(full_before),
            "full_trace_len": rows[0]["full_trace_len"] if rows else None,
            "adopted_trace_len": rows[0]["adopted_trace_len"] if rows else None,
            "per_seed": {
                str(c["seed"]): dict(c["raw"][mode])
                for c in cells if mode in c["raw"]
            },
        }
    return out


def summarize_cap(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed cap-sweep summary: sizes, evictions, hot-path loss."""
    by_cap: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_cap.setdefault(cell["cap"], []).append(cell)
    out: Dict[str, Any] = {}
    for cap in sorted(by_cap):
        cap_cells = by_cap[cap]
        first_raw = cap_cells[0]["raw"]
        entry: Dict[str, Any] = {
            "n_seeds": len(cap_cells),
            "stream_len": first_raw["stream_len"],
            "distinct_paths_recorded": first_raw["exact"][
                "distinct_paths_recorded"],
            "oracle_hot": first_raw["oracle_hot"],
        }
        for mode in ("exact", "ema"):
            rows = [c["raw"][mode] for c in cap_cells]
            entry[mode] = {
                "final_counts_size": _mean_stdev(
                    [float(r["final_counts_size"]) for r in rows]),
                "final_ema_counters_size": _mean_stdev(
                    [float(r["final_ema_counters_size"]) for r in rows]),
                "final_stats_size": _mean_stdev(
                    [float(r["final_stats_size"]) for r in rows]),
                "final_hot_paths_size": _mean_stdev(
                    [float(r["final_hot_paths_size"]) for r in rows]),
                "final_recency_size": _mean_stdev(
                    [float(r["final_recency_size"]) for r in rows]),
                "ever_hot_size": _mean_stdev(
                    [float(r["ever_hot_size"]) for r in rows]),
                "hot_lost_after_eviction": _mean_stdev(
                    [float(len(r["hot_lost_after_eviction"])) for r in rows]),
                "evictions": _mean_stdev(
                    [float(r["evictions"]) for r in rows]),
                "bound_respected_all_seeds": all(
                    r["bound_respected"] for r in rows),
                "per_seed": {
                    str(c["seed"]): dict(c["raw"][mode]) for c in cap_cells},
            }
        never_prom = [float(len(c["raw"]["ema"]["oracle_hot_never_promoted"]))
                      for c in cap_cells]
        entry["ema_oracle_hot_never_promoted"] = _mean_stdev(never_prom)
        out[str(cap)] = entry
    return out


def summarize_cap_rescue(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Cross-seed cap-rescue summary: detection vs cap on the heavy stream."""
    by_cap: Dict[int, List[Dict[str, Any]]] = {}
    for cell in cells:
        by_cap.setdefault(cell["cap"], []).append(cell)
    out: Dict[str, Any] = {}
    for cap in sorted(by_cap):
        cap_cells = by_cap[cap]
        det_idx = [float(c["raw"]["ema_burst_detection_index"])
                   for c in cap_cells
                   if c["raw"]["ema_burst_detection_index"] is not None]
        out[str(cap)] = {
            "n_seeds": len(cap_cells),
            "stream_len": cap_cells[0]["raw"]["stream_len"],
            "distinct_paths_recorded": cap_cells[0]["raw"][
                "distinct_paths_recorded"],
            "ema_burst_detected_seeds": len(det_idx),
            "ema_burst_detection_index": _mean_stdev(det_idx),
            "ema_final_denominator": _mean_stdev(
                [float(c["raw"]["ema_final_denominator"]) for c in cap_cells]),
            "evictions": _mean_stdev(
                [float(c["raw"]["evictions"]) for c in cap_cells]),
            "exact_promoted_count": _mean_stdev(
                [float(c["raw"]["exact_promoted_count"]) for c in cap_cells]),
            "per_seed": {
                str(c["seed"]): dict(c["raw"]) for c in cap_cells},
        }
    return out


# ---------------------------------------------------------------------------
# Invariant checks (hard assertions on the raw results; abort on failure).
# ---------------------------------------------------------------------------

def check_invariants(scenario_results: Dict[str, Any],
                     adoption: Dict[str, Any],
                     cap_results: Dict[str, Any],
                     cap_rescue: Dict[str, Any]) -> List[str]:
    """Sanity invariants the harness asserts before writing results.

    Returns the list of passed invariant descriptions; raises on any
    violation so a broken run can never produce a results file.
    """
    checks: List[str] = []

    def require(cond: bool, desc: str) -> None:
        if not cond:
            raise RuntimeError(f"INVARIANT VIOLATED: {desc}")
        checks.append(desc)

    rot = scenario_results["diverse_rotated"]
    require(
        rot["ema_promoted"]["mean"] == 3.0
        and rot["exact_promoted"]["mean"] == 0.0,
        "diverse_rotated: EMA promotes all 3 variants, exact promotes none",
    )
    pure = scenario_results["pure_diversity"]
    require(
        pure["ema_promoted"]["mean"] == 0.0
        and pure["exact_promoted"]["mean"] == 0.0,
        "pure_diversity: no detector fires on 160 one-off trajectories",
    )
    # Denominator floor: 80 stale warm-up paths keep the burst's recent
    # share below theta under the default cap (no eviction pressure).
    heavy = scenario_results["warmup_heavy_burst"]
    require(
        heavy["ema_promoted"]["mean"] == 0.0,
        "warmup_heavy_burst: EMA stays cold under the default cap "
        "(share denominator floor)",
    )
    # Cap rescue: the smallest cap evicts the stale warm-up, shrinks the
    # denominator, and the burst gets detected in every seed.
    smallest = str(min(int(c) for c in cap_rescue)) if cap_rescue else None
    if smallest is not None:
        require(
            cap_rescue[smallest]["ema_burst_detected_seeds"]
            == cap_rescue[smallest]["n_seeds"],
            f"cap_rescue (cap={smallest}): eviction restores burst detection "
            "in every seed",
        )
    ema_ad = adoption["ema"]
    exact_ad = adoption["exact"]
    require(
        ema_ad["detected_seeds"] == ema_ad["n_seeds"]
        and ema_ad["adopted_seeds"] == ema_ad["n_seeds"],
        "bursty: EMA detects and adopts the burst in every seed",
    )
    require(
        exact_ad["detected_seeds"] == 0,
        "bursty: exact detector stays cold through 80 burst recordings (<100)",
    )
    for cap, entry in sorted(cap_results.items()):
        require(
            entry["exact"]["bound_respected_all_seeds"]
            and entry["ema"]["bound_respected_all_seeds"],
            f"cap={cap}: all per-path mappings stay within the cap",
        )
    conc = scenario_results["concentrated"]
    require(
        conc["exact_promoted"]["mean"] >= 1.0
        and conc["ema_promoted"]["mean"] >= 1.0,
        "concentrated: both detectors promote the dominant path",
    )
    return checks


# ---------------------------------------------------------------------------
# Headline numbers (script-extracted from the computed structures).
# ---------------------------------------------------------------------------

def build_headline(scenario_results: Dict[str, Any],
                   adoption_results: Dict[str, Any],
                   cap_results: Dict[str, Any],
                   cap_rescue_results: Dict[str, Any],
                   invariants: List[str]) -> Dict[str, Any]:
    """Pull the headline comparison numbers from computed results.

    Every value here is a reference into (or arithmetic over) the
    aggregates computed above; nothing is hand-entered.
    """
    rot = scenario_results["diverse_rotated"]
    heavy = scenario_results["warmup_heavy_burst"]
    ema_ad = adoption_results["ema"]
    exact_ad = adoption_results["exact"]
    rescue = cap_rescue_results
    detecting_caps = sorted(
        (int(c) for c, e in rescue.items() if e["ema_burst_detected_seeds"] > 0),
        reverse=True)
    non_detecting_caps = sorted(
        int(c) for c, e in rescue.items() if e["ema_burst_detected_seeds"] == 0)
    smallest_cap = str(min(int(c) for c in cap_results)) if cap_results else None
    max_final_size = max(
        (e[mode]["final_counts_size"]["mean"]
         for e in cap_results.values() for mode in ("exact", "ema")),
        default=None)
    return {
        "diverse_rotated_exact_promoted_mean": rot["exact_promoted"]["mean"],
        "diverse_rotated_ema_promoted_mean": rot["ema_promoted"]["mean"],
        "diverse_rotated_ema_first_promotion_recording_mean": (
            rot["ema_first_promotion_recording"]["mean"]),
        "bursty_ema_detection_index_mean": ema_ad["detection_index"]["mean"],
        "bursty_ema_full_form_executions_before_adoption_mean": (
            ema_ad["full_form_executions_before_adoption"]["mean"]),
        "bursty_exact_detected_seeds": exact_ad["detected_seeds"],
        "warmup_heavy_burst_ema_promoted_mean": heavy["ema_promoted"]["mean"],
        "cap_rescue_detecting_caps": [str(c) for c in detecting_caps],
        "cap_rescue_largest_detecting_cap": (
            str(detecting_caps[0]) if detecting_caps else None),
        "cap_rescue_largest_detecting_cap_detection_index_mean": (
            rescue[str(detecting_caps[0])]["ema_burst_detection_index"]["mean"]
            if detecting_caps else None),
        "cap_rescue_smallest_non_detecting_cap": (
            str(non_detecting_caps[0]) if non_detecting_caps else None),
        "cap_sweep_smallest_cap": smallest_cap,
        "cap_sweep_max_final_counts_size_mean": max_final_size,
        "cap_sweep_all_bounds_respected": all(
            e[mode]["bound_respected_all_seeds"]
            for e in cap_results.values() for mode in ("exact", "ema")),
        "cap_sweep_exact_hot_lost_after_eviction_at_smallest_cap": (
            cap_results[smallest_cap]["exact"]["hot_lost_after_eviction"]["mean"]
            if smallest_cap else None),
        "cap_sweep_ema_hot_lost_after_eviction_at_smallest_cap": (
            cap_results[smallest_cap]["ema"]["hot_lost_after_eviction"]["mean"]
            if smallest_cap else None),
        "invariants_passed": len(invariants),
    }


# ---------------------------------------------------------------------------
# Orchestration.
# ---------------------------------------------------------------------------

def build_instr_for(stream: Sequence[str],
                    family: str = "fam") -> Dict[str, List[Instruction]]:
    """One instruction sequence per distinct path in the stream."""
    return {
        path: variant_instructions(family, path)
        for path in sorted(set(stream))
    }


def collect_git_rev() -> str:
    """Read-only git rev for provenance (allowed by the task rules)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception as exc:  # pragma: no cover - environment-dependent
        return f"unavailable: {exc.__class__.__name__}"


def run_all(seeds: int, quick: bool, cap_sweep: bool) -> Dict[str, Any]:
    """Run the full stress matrix and assemble the report."""
    wall_start = time.perf_counter()
    rng_seeds = [SEED_BASE + k for k in range(seeds)]

    scenario_cells: Dict[str, List[Dict[str, Any]]] = {
        name: [] for name, _ in SCENARIOS
    }
    adoption_cells: List[Dict[str, Any]] = []
    cap_cells: List[Dict[str, Any]] = []
    cap_rescue_cells: List[Dict[str, Any]] = []

    detection_recordings = 0
    cap_recordings = 0
    cap_rescue_recordings = 0

    for seed in rng_seeds:
        for name, gen in SCENARIOS:
            stream = gen(random.Random(seed))
            instr_for = build_instr_for(stream)
            cell_raw = run_detection_cell(stream, instr_for)
            detection_recordings += cell_raw["stream_len"]
            scenario_cells[name].append({"seed": seed, "raw": cell_raw})
            if name == "bursty":
                adoption_raw = {
                    mode: run_adoption_cell(stream, instr_for, mode)
                    for mode in ("exact", "ema")
                }
                adoption_cells.append({"seed": seed, "raw": adoption_raw})
        if cap_sweep:
            cap_kwargs = ({"zipf_budget": 400, "n_oneoffs": 120}
                          if quick else {})
            cap_stream = stream_cap_stress(random.Random(seed), **cap_kwargs)
            cap_instr = build_instr_for(cap_stream)
            for cap in CAPS:
                raw = run_cap_cell(cap_stream, cap, cap_instr)
                cap_recordings += raw["stream_len"]
                cap_cells.append({"seed": seed, "cap": cap, "raw": raw})
            rescue_stream = stream_warmup_heavy(random.Random(seed))
            rescue_instr = build_instr_for(rescue_stream)
            for cap in CAP_RESCUE_CAPS:
                raw = run_cap_rescue_cell(rescue_stream, cap, rescue_instr)
                cap_rescue_recordings += raw["stream_len"]
                cap_rescue_cells.append(
                    {"seed": seed, "cap": cap, "raw": raw})

    scenario_results = {
        name: summarize_scenario(cells)
        for name, cells in scenario_cells.items()
    }
    adoption_results = summarize_adoption(adoption_cells)
    cap_results = summarize_cap(cap_cells) if cap_cells else {}
    cap_rescue_results = (summarize_cap_rescue(cap_rescue_cells)
                          if cap_rescue_cells else {})
    invariants = check_invariants(
        scenario_results, adoption_results, cap_results, cap_rescue_results)

    # Every detection recording feeds BOTH detectors + the oracle; the
    # manager-based adoption cells replay the bursty stream once per
    # mode; cap and rescue cells feed both detectors per recording.
    bursty_len = len(stream_bursty(random.Random(rng_seeds[0])))
    n_adoption_recordings = bursty_len * len(rng_seeds) * 2

    return {
        "scenarios": scenario_results,
        "adoption_bursty": adoption_results,
        "cap_sweep": cap_results,
        "cap_rescue": cap_rescue_results,
        "invariants_passed": invariants,
        "headline": build_headline(
            scenario_results, adoption_results, cap_results,
            cap_rescue_results, invariants),
        "run_meta": {
            "wall_seconds": round(time.perf_counter() - wall_start, 3),
            "n_seeds": seeds,
            "rng_seeds": rng_seeds,
            "detection_cell_recordings": detection_recordings,
            "adoption_cell_recordings": n_adoption_recordings,
            "cap_cell_recordings": cap_recordings,
            "cap_rescue_cell_recordings": cap_rescue_recordings,
            "detector_record_calls_total": (
                2 * detection_recordings + n_adoption_recordings
                + 2 * cap_recordings + 2 * cap_rescue_recordings),
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seeds", type=int, default=N_SEEDS_DEFAULT,
                        help=f"number of rng seeds (default {N_SEEDS_DEFAULT})")
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: 1 seed, shrunk cap-stress stream")
    parser.add_argument("--no-cap-sweep", action="store_true",
                        help="skip the max_tracked_paths cap sweep")
    parser.add_argument("--output", default=None,
                        help="output JSON path (default: alongside script)")
    args = parser.parse_args(argv)

    if args.quick and args.seeds == N_SEEDS_DEFAULT:
        args.seeds = 1

    report = run_all(args.seeds, args.quick, cap_sweep=not args.no_cap_sweep)

    report["provenance"] = {
        "git_rev": collect_git_rev(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "rng_seeds": report["run_meta"]["rng_seeds"],
        "scale": {
            "n_seeds": args.seeds,
            "scenario_stream_lens": {
                "concentrated": 512,
                "diverse_rotated": 120,
                "diverse_zipf": "Zipf(1.1) counts over 10 ranks, ~160",
                "cyclic_shift": 160,
                "bursty": 88,
                "warmup_heavy_burst": 160,
                "pure_diversity": 160,
                "cap_stress": (
                    "Zipf(1.1) mix (~800 recordings) with 2200 interleaved "
                    "one-offs (~3000 recordings, 2210 distinct paths)"
                    if not args.quick
                    else "quick mode: shrunk (~480 recordings, 130 distinct)"),
            },
            "caps_swept": list(CAPS) if not args.no_cap_sweep else [],
            "cap_rescue_caps": (list(CAP_RESCUE_CAPS)
                                if not args.no_cap_sweep else []),
            "trace_len_per_variant": TRACE_LEN,
            "detection_cells_cap": DEFAULT_CAP,
            "quick_mode": args.quick,
        },
        "detector_config": {
            "hot_path_threshold": HOT_PATH_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
            "ema_window": EMA_WINDOW,
            "ema_gamma": round(2.0 / (EMA_WINDOW + 1), 6),
            "ema_threshold": EMA_THRESHOLD,
        },
    }
    report["coverage"] = {
        "i3_hotpath_ema": "shipped",
        "scenarios_run": [name for name, _ in SCENARIOS],
        "cap_sweep_run": not args.no_cap_sweep,
        "cap_rescue_run": not args.no_cap_sweep,
        "adoption_measured": True,
        "oracle": (
            "deterministic cumulative-share rule (>=2 recordings and "
            "share > theta), brute-force over the stream"
        ),
        "notes": (
            "All aggregates computed by this script from raw recorded "
            "events; no hand-entered numbers. Detection cells compare "
            "exact vs ema vs oracle over identical seeded streams at the "
            "shipped default cap; adoption measured through "
            "OptimizationManager; cap sweep bounds every per-path mapping "
            "under each cap; cap_rescue measures eviction shrinking the "
            "EMA share denominator (per-path decay only on own "
            "recordings) and restoring burst detection."
        ),
    }

    out_path = Path(args.output) if args.output else (
        Path(__file__).resolve().parent / "hotpath_ema_results.json")
    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out_path}")
    print(json.dumps({
        "run_meta": report["run_meta"],
        "invariants_passed": len(report["invariants_passed"]),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
