#!/usr/bin/env python
"""E3-selector-benchmark: heuristic vs DP selector on random IR trees.

Compares the shipped heuristic tree cover (``_greedy_tree_cover``: the
name-overlap matcher applied per node in isolation, single-node patterns
only) against the MATCH-TREE dynamic program (supplementary Appendix
D.1, ``InstructionSelector(selection_strategy="dp")``) on randomly
generated IR trees of sizes {20, 50, 100, 200} nodes x 5 seeds.

Design mirrors ``tests/core/isa/test_instruction_selector_dp.py``:
random trees over a fixed op alphabet; catalogues containing a base
single-node instruction per op (so both arms can always cover any
tree), costlier "premium" variants (only the cost-blind heuristic can
ever prefer them), and fused patterns that declare
``params["spanned_child_ops"]`` at a discounted cost (only the DP has
the cross-node horizon to exploit them).

Aggregation discipline: every aggregate in the JSON artifact is
computed by this script from the raw per-instance records via the
``statistics`` module -- nothing is hand-computed or hand-filled.

Coverage honesty: if any instance fails, it is excluded from the
aggregates, recorded under ``coverage.failed_instances``, and the run
is marked ``coverage.status = "partial"`` with exit code 1.  All
reported numbers come from actually-run instances only.

Provenance: git rev (read-only via ``git rev-parse --short HEAD``),
UTC timestamp, seeds, sizes, RNG formula, and the pre-bench status of
the implemented DP-selector component (I2-dp-selector) are embedded in
the artifact.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import subprocess  # noqa: S403  (read-only rev parse, no user input)
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Make the agentflow package importable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.instruction_selector import (  # noqa: E402
    InstructionSelector,
    IRTreeNode,
    TreeSelectionResult,
)
from agentflow.core.isa.isa_manager import (  # noqa: E402
    Instruction,
    InstructionType,
)

#: Node counts of the random IR trees.
SIZES: Tuple[int, ...] = (20, 50, 100, 200)
#: Seeds per size: the run uses the full 5.
SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
#: Op alphabet of the random trees (mirrors the DP unit tests' `ops`).
OPS: Tuple[str, ...] = ("a", "b", "c", "d")
#: Timing repetitions per arm; the median is reported.
REPS: int = 5
#: Cost comparison tolerance.
COST_EPS: float = 1e-9
#: Number of premium variants in a catalogue.
PREMIUM_OPS_COUNT: int = 2
#: Fused-pattern count range in a catalogue.
FUSED_COUNT_RANGE: Tuple[int, int] = (3, 6)
#: Fused-pattern discount range, relative to the sum of base costs of the
#: root op plus the spanned child ops it subsumes.
FUSED_DISCOUNT_RANGE: Tuple[float, float] = (0.70, 0.95)
#: Premium cost multiplier range, relative to the base instruction.
PREMIUM_COST_RANGE: Tuple[float, float] = (1.3, 2.0)

DESCRIPTION = (
    "Heuristic vs DP instruction selector on random IR trees: MATCH-TREE "
    "DP (supplementary Appendix D.1) vs the shipped greedy heuristic tree "
    "cover. Sizes 20/50/100/200 nodes x 5 seeds; metrics are covering cost "
    "and wall-clock selection time."
)


# ---------------------------------------------------------------------------
# RNG plumbing
# ---------------------------------------------------------------------------

def rng_seed_for(size: int, seed: int) -> int:
    """Deterministic master seed for one (size, seed) instance.

    Incorporates the size so every (size, seed) pair gets its own
    catalogue/tree stream instead of sharing catalogues across sizes.
    """
    return size * 1000 + seed


def _round_cost(value: float) -> float:
    """Round a random cost the way the DP unit tests do (3 decimals)."""
    return round(value, 3)


# ---------------------------------------------------------------------------
# Random IR tree generation
# ---------------------------------------------------------------------------

def generate_ir_tree(rng: random.Random, ops: Sequence[str], node_count: int,
                     payload_prob: float = 0.3,
                     payload_value: Dict[str, Any] = None) -> IRTreeNode:
    """Generate a random IR tree with exactly ``node_count`` nodes.

    Grows the tree one node at a time: each new node is attached to a
    uniformly chosen existing node that still has room for children
    (fanout <= 3) and is below the depth cap.  This guarantees the exact
    node count (the "length" of the benchmark sizes) while keeping the
    shape random and the recursion depth bounded for the DP.

    Payloads make the tree a fair battlefield for the heuristic's
    name-overlap scorer: nodes carrying the premium param can look
    attractive to the cost-blind heuristic.
    """
    if node_count < 1:
        raise ValueError(f"node_count must be >= 1, got {node_count}")
    payload_value = payload_value if payload_value is not None else {
        "urgency": "high"}
    # Depth cap generous enough that the attachment process never runs
    # out of candidate parents (capacity is exponential in the cap).
    max_depth = max(4, int(math.ceil(math.log2(node_count + 1))) + 3)
    max_fanout = 3

    def make_node() -> IRTreeNode:
        return IRTreeNode(
            op=rng.choice(list(ops)),
            children=[],
            payload=dict(payload_value) if rng.random() < payload_prob
            else {},
        )

    root = make_node()
    nodes = [root]
    depth = {id(root): 1}
    while len(nodes) < node_count:
        candidates = [n for n in nodes
                      if len(n.children) < max_fanout
                      and depth[id(n)] < max_depth]
        if not candidates:  # defensive: capacity far exceeds node_count
            raise RuntimeError(
                f"no candidate parents left after {len(nodes)} nodes")
        parent = rng.choice(candidates)
        child = make_node()
        parent.children.append(child)
        nodes.append(child)
        depth[id(child)] = depth[id(parent)] + 1
    return root


def iter_tree(node: IRTreeNode) -> Iterable[IRTreeNode]:
    """Yield every node of the subtree rooted at ``node``."""
    yield node
    for child in node.children:
        yield from iter_tree(child)


def tree_stats(node: IRTreeNode) -> Dict[str, Any]:
    """Structural stats of a tree (node count, depth, branching, payloads)."""
    depths: Dict[int, int] = {}
    branch_counts: List[int] = []
    n_internal = 0
    n_leaf = 0
    n_payload = 0
    # Iterative post-order walk so deep trees cannot overflow the stack.
    stack: List[Tuple[IRTreeNode, bool]] = [(node, False)]
    while stack:
        current, expanded = stack.pop()
        if not expanded:
            stack.append((current, True))
            for child in reversed(current.children):
                stack.append((child, False))
            continue
        if not current.children:
            depths[id(current)] = 1
            n_leaf += 1
        else:
            depths[id(current)] = 1 + max(depths[id(c)]
                                          for c in current.children)
            branch_counts.append(len(current.children))
            n_internal += 1
        if current.payload:
            n_payload += 1
    return {
        "node_count": n_internal + n_leaf,
        "depth": depths[id(node)],
        "internal_nodes": n_internal,
        "leaf_nodes": n_leaf,
        "max_fanout": max(branch_counts) if branch_counts else 0,
        "payload_nodes": n_payload,
    }


def tree_signature(node: IRTreeNode) -> Tuple:
    """A structural signature for determinism assertions."""
    return (node.op, tuple(tree_signature(c) for c in node.children))


# ---------------------------------------------------------------------------
# Random instruction catalogue
# ---------------------------------------------------------------------------

def generate_catalogue(rng: random.Random, ops: Sequence[str]
                        ) -> Dict[str, Instruction]:
    """Build the random catalogue for one instance.

    * one base single-node instruction per op (cost in [0.5, 5.0]) --
      guarantees both arms can cover any tree over ``ops``;
    * ``PREMIUM_OPS_COUNT`` costlier variants of randomly chosen ops
      carrying the ``urgency`` param (cost * PREMIUM_COST_RANGE) -- only
      the cost-blind heuristic can ever pick them;
    * ``rng.randint(*FUSED_COUNT_RANGE)`` fused patterns: root op plus
      1-2 spanned child ops at a discounted cost -- only the DP has the
      horizon to exploit them.
    """
    catalogue: Dict[str, Instruction] = {}

    def add(key: str, name: str, cost: float, params: Dict[str, Any]) -> None:
        catalogue[key] = Instruction(
            id=f"instr-{key}",
            name=name,
            type=InstructionType.COMPUTATION,
            description=f"{name} operation (bench catalogue)",
            params=params,
            cost=cost,
        )

    for op in ops:
        add(op, op, _round_cost(rng.uniform(0.5, 5.0)), {})

    premium_ops = rng.sample(list(ops), PREMIUM_OPS_COUNT)
    for op in premium_ops:
        base = catalogue[op]
        add(f"{op}_premium", op,
            _round_cost(base.cost * rng.uniform(*PREMIUM_COST_RANGE)),
            {"urgency": "high"})

    # Base costs only: premium/fused entries share the base ops' names but
    # must not leak their (higher) costs into the fused discount baseline.
    base_costs = {instr.name: instr.cost for instr in catalogue.values()
                  if not instr.params}
    fused_counter = 0
    for _ in range(rng.randint(*FUSED_COUNT_RANGE)):
        root_op = rng.choice(ops)
        spanned = tuple(rng.choice(ops) for _ in range(rng.randint(1, 2)))
        parts = base_costs[root_op] + sum(base_costs[op] for op in spanned)
        add(f"{root_op}_fused_{fused_counter}", root_op,
            _round_cost(parts * rng.uniform(*FUSED_DISCOUNT_RANGE)),
            {"spanned_child_ops": list(spanned)})
        fused_counter += 1

    return catalogue


def catalogue_summary(catalogue: Dict[str, Instruction]) -> Dict[str, Any]:
    """JSON-serialisable summary of a catalogue (base/premium/fused)."""
    base_costs: Dict[str, float] = {}
    premium: List[Dict[str, Any]] = []
    fused: List[Dict[str, Any]] = []
    for key in sorted(catalogue):
        instr = catalogue[key]
        entry = {"key": key, "name": instr.name, "root_op": instr.name,
                 "cost": instr.cost, "params": instr.params}
        if "spanned_child_ops" in instr.params:
            entry["spanned_child_ops"] = list(
                instr.params["spanned_child_ops"])
            fused.append(entry)
        elif instr.params.get("urgency") == "high":
            premium.append(entry)
        else:
            base_costs[instr.name] = instr.cost
    return {
        "n_instructions": len(catalogue),
        "n_single": len(base_costs),
        "n_premium": len(premium),
        "n_fused": len(fused),
        "base_costs": base_costs,
        "premium": premium,
        "fused": fused,
    }


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def time_select_ms(selector: InstructionSelector, tree: IRTreeNode,
                   catalogue: Dict[str, Instruction], reps: int
                   ) -> Tuple[TreeSelectionResult, List[float]]:
    """Run ``select_for_tree`` ``reps`` times, timing each rep in ms.

    Returns the final result plus all recorded rep timings; the median
    is the reported timing so one-off scheduler hiccups do not dominate.
    """
    timings: List[float] = []
    result: Optional[TreeSelectionResult] = None
    for _ in range(reps):
        start = time.perf_counter()
        result = selector.select_for_tree(
            tree, available_instructions=catalogue)
        timings.append((time.perf_counter() - start) * 1000.0)
    assert result is not None
    return result, timings


# ---------------------------------------------------------------------------
# Covering validation (mirrors the DP unit tests' assert_valid_tiling)
# ---------------------------------------------------------------------------

def check_tiling(result: TreeSelectionResult, root: IRTreeNode) -> bool:
    """Whether the covering tiles the tree: each node covered exactly once
    and the reported cost equals the sum of chosen pattern costs."""
    covered: List[int] = []
    for node, pattern in result.covering:
        covered.append(id(node))
        for child in pattern.spanned_children(node):
            covered.extend(id(n) for n in iter_tree(child))
    all_ids = [id(n) for n in iter_tree(root)]
    if sorted(covered) != sorted(all_ids):
        return False
    expected_cost = sum(p.cost for _, p in result.covering)
    return abs(result.cost - expected_cost) <= COST_EPS


def count_sites(result: TreeSelectionResult, marker: str) -> int:
    """Count covering sites whose chosen pattern carries ``marker``."""
    return sum(1 for _, pattern in result.covering
               if marker in getattr(pattern.instruction, "id", ""))


# ---------------------------------------------------------------------------
# Per-instance run
# ---------------------------------------------------------------------------

def run_instance(size: int, seed: int, ops: Sequence[str] = OPS,
                 reps: int = REPS) -> Dict[str, Any]:
    """Run one (size, seed) instance through both selector arms."""
    master = rng_seed_for(size, seed)
    tree_rng = random.Random(master)
    tree = generate_ir_tree(tree_rng, ops, size)
    catalogue = generate_catalogue(random.Random(master), ops)

    dp_selector = InstructionSelector(selection_strategy="dp")
    heur_selector = InstructionSelector(selection_strategy="heuristic")

    dp_result, dp_ms = time_select_ms(dp_selector, tree, catalogue, reps)
    heur_result, heur_ms = time_select_ms(heur_selector, tree, catalogue, reps)

    record: Dict[str, Any] = {
        "size": size,
        "seed": seed,
        "rng_seed": master,
        "tree": tree_stats(tree),
        "catalogue": catalogue_summary(catalogue),
        "dp": _arm_record(dp_result, dp_ms, tree),
        "heuristic": _arm_record(heur_result, heur_ms, tree),
    }
    dp_cost = record["dp"]["cost"]
    heur_cost = record["heuristic"]["cost"]
    record["cost_gap_heuristic_minus_dp"] = round(heur_cost - dp_cost, 9)
    record["cost_ratio_dp_over_heuristic"] = round(dp_cost / heur_cost, 6)
    record["dp_cost_le_heuristic"] = dp_cost <= heur_cost + COST_EPS
    record["time_ratio_dp_over_heuristic_median"] = round(
        record["dp"]["time_ms_median"]
        / record["heuristic"]["time_ms_median"], 6)
    return record


def _arm_record(result: TreeSelectionResult, timings: List[float],
                tree: IRTreeNode) -> Dict[str, Any]:
    """Per-arm record; incomplete covers get null cost/timing comparisons."""
    complete = result.complete
    return {
        "complete": complete,
        "tiling_valid": check_tiling(result, tree) if complete else False,
        "cost": round(result.cost, 6) if complete else None,
        "covering_size": len(result.covering),
        "fused_sites_used": count_sites(result, "_fused"),
        "premium_sites_used": count_sites(result, "_premium"),
        "time_ms_reps": [round(t, 6) for t in timings],
        "time_ms_median": round(statistics.median(timings), 6),
        "time_ms_min": round(min(timings), 6),
        "time_ms_max": round(max(timings), 6),
    }


# ---------------------------------------------------------------------------
# Aggregation (script-computed from raw records; nothing hand-filled)
# ---------------------------------------------------------------------------

def mean_stdev(values: Sequence[float]) -> Tuple[float, float]:
    """Mean and stdev of a non-empty list (single value -> stdev 0.0)."""
    mean = statistics.mean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, stdev


def aggregate_instances(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate raw per-instance records into the summary block."""
    dp_costs = [r["dp"]["cost"] for r in records]
    heur_costs = [r["heuristic"]["cost"] for r in records]
    dp_ms = [r["dp"]["time_ms_median"] for r in records]
    heur_ms = [r["heuristic"]["time_ms_median"] for r in records]
    gaps = [r["cost_gap_heuristic_minus_dp"] for r in records]
    ratios = [r["cost_ratio_dp_over_heuristic"] for r in records]
    time_ratios = [r["time_ratio_dp_over_heuristic_median"] for r in records]
    depths = [r["tree"]["depth"] for r in records]
    eps = COST_EPS
    wins = sum(1 for r in records
               if r["dp"]["cost"] < r["heuristic"]["cost"] - eps)
    ties = sum(1 for r in records
               if abs(r["cost_gap_heuristic_minus_dp"]) <= eps)
    losses = sum(1 for r in records
                 if r["dp"]["cost"] > r["heuristic"]["cost"] + eps)

    dp_cost_mean, dp_cost_stdev = mean_stdev(dp_costs)
    heur_cost_mean, heur_cost_stdev = mean_stdev(heur_costs)
    dp_time_mean, dp_time_stdev = mean_stdev(dp_ms)
    heur_time_mean, heur_time_stdev = mean_stdev(heur_ms)
    gap_mean, gap_stdev = mean_stdev(gaps)
    ratio_mean, _ = mean_stdev(ratios)
    time_ratio_mean, _ = mean_stdev(time_ratios)
    depth_mean, _ = mean_stdev(depths)

    return {
        "n_instances": len(records),
        "dp_cost_mean": round(dp_cost_mean, 6),
        "dp_cost_stdev": round(dp_cost_stdev, 6),
        "heuristic_cost_mean": round(heur_cost_mean, 6),
        "heuristic_cost_stdev": round(heur_cost_stdev, 6),
        "cost_gap_mean": round(gap_mean, 6),
        "cost_gap_stdev": round(gap_stdev, 6),
        "cost_ratio_dp_over_heuristic_mean": round(ratio_mean, 6),
        "cost_ratio_dp_over_heuristic_min": round(min(ratios), 6),
        "cost_ratio_dp_over_heuristic_max": round(max(ratios), 6),
        "dp_wins": wins,
        "cost_ties": ties,
        "dp_cost_losses": losses,
        "dp_cost_le_heuristic_all": losses == 0,
        "dp_time_ms_mean": round(dp_time_mean, 6),
        "dp_time_ms_stdev": round(dp_time_stdev, 6),
        "heuristic_time_ms_mean": round(heur_time_mean, 6),
        "heuristic_time_ms_stdev": round(heur_time_stdev, 6),
        "time_ratio_dp_over_heuristic_mean": round(time_ratio_mean, 6),
        "time_ratio_dp_over_heuristic_min": round(min(time_ratios), 6),
        "time_ratio_dp_over_heuristic_max": round(max(time_ratios), 6),
        "dp_fused_sites_total": sum(
            r["dp"]["fused_sites_used"] for r in records),
        "dp_premium_sites_total": sum(
            r["dp"]["premium_sites_used"] for r in records),
        "heuristic_fused_sites_total": sum(
            r["heuristic"]["fused_sites_used"] for r in records),
        "heuristic_premium_sites_total": sum(
            r["heuristic"]["premium_sites_used"] for r in records),
        "tree_depth_mean": round(depth_mean, 6),
        "tree_depth_max": max(depths),
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def read_git_rev() -> str:
    """Read-only ``git rev-parse --short HEAD`` for provenance."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=15,
        )
        return out.stdout.strip()
    except Exception as exc:  # noqa: BLE001  (provenance robustness)
        return f"unavailable ({exc.__class__.__name__})"


def build_provenance(sizes: Sequence[int], seeds: Sequence[int], reps: int,
                     component_check: str) -> Dict[str, Any]:
    """Assemble the provenance block of the artifact."""
    return {
        "git_rev": read_git_rev(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "seeds": list(seeds),
        "sizes": list(sizes),
        "rng_seed_formula": "random.Random(size * 1000 + seed) per "
                            "(size, seed); catalogue and tree use "
                            "independent streams from the same seed",
        "timing": {
            "reps": reps,
            "reported_statistic": "median of reps, per arm",
            "clock": "time.perf_counter",
        },
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "component": {
            "job": "I2-dp-selector",
            "status": "shipped",
            "pre_bench_component_check": component_check,
            "dp_arm": "InstructionSelector(selection_strategy='dp')",
            "heuristic_arm": (
                "InstructionSelector(selection_strategy='heuristic')"),
            "module": "agentflow/core/isa/instruction_selector.py",
        },
    }


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

def run_benchmark(sizes: Sequence[int] = SIZES, seeds: Sequence[int] = SEEDS,
                  ops: Sequence[str] = OPS, reps: int = REPS,
                  component_check: str = "") -> Dict[str, Any]:
    """Run the full benchmark grid; failed instances degrade coverage."""
    instances: List[Dict[str, Any]] = []
    failed: List[Dict[str, Any]] = []
    planned = [(size, seed) for size in sizes for seed in seeds]
    for size, seed in planned:
        try:
            instances.append(
                run_instance(size=size, seed=seed, ops=ops, reps=reps))
        except Exception as exc:  # noqa: BLE001  (coverage honesty)
            failed.append({
                "size": size,
                "seed": seed,
                "detail": f"{exc.__class__.__name__}: {exc}",
            })

    by_size = {str(size): aggregate_instances(
        [r for r in instances if r["size"] == size]) for size in sizes}
    overall = aggregate_instances(instances) if instances else {}
    status = "complete" if not failed else "partial"
    return {
        "benchmark": "selector_dp_bench",
        "description": DESCRIPTION,
        "provenance": build_provenance(sizes, seeds, reps, component_check),
        "coverage": {
            "status": status,
            "instances_planned": len(planned),
            "instances_completed": len(instances),
            "failed_instances": failed,
            "note": ("All aggregates are computed only over completed "
                     "instances; failures are listed above.") if failed
            else "All planned instances completed.",
        },
        "instances": instances,
        "by_size": by_size,
        "overall": overall,
        "notes": [
            "Costs are catalogue instruction costs summed over the covering "
            "(MATCH-TREE total cost); lower is better.",
            "The invariant under test is DP cost <= heuristic cost "
            "(structural: the DP minimises over a covering set that "
            "includes the greedy single-node covering); the ABSOLUTE gap "
            "depends on this synthetic catalogue's pricing model.",
            "Catalogue pricing: fused patterns cost 0.70-0.95x the base "
            "cost of (root op + spanned child ops) while covering the "
            "spanned children's whole subtrees, so one fused site near "
            "the root can subsume a large subtree -- the cost gap grows "
            "with tree size by construction of the model, not by magic.",
            "Premium variants (1.3-2.0x base cost, carrying the urgency "
            "param) are only ever chosen by the score-first heuristic; "
            "the cost-minimising DP never selects them.",
            "Timing is wall-clock selection time (median of reps); DP "
            "being slower is allowed and reported honestly.",
            "The heuristic arm is the shipped greedy tree cover "
            "(single-node patterns only, no cross-node horizon); the DP "
            "arm is MATCH-TREE (Appendix D.1).",
            "Aggregates are computed by this script from the raw "
            "per-instance records via the statistics module; nothing is "
            "hand-filled.",
        ],
    }


def summarize(report: Dict[str, Any]) -> str:
    """Human-readable summary of a run (numbers straight from the report)."""
    lines = [f"benchmark: {report['benchmark']}"]
    cov = report["coverage"]
    lines.append(
        f"coverage: {cov['status']} "
        f"({cov['instances_completed']}/{cov['instances_planned']} instances)")
    by_size = sorted(report["by_size"].items(), key=lambda kv: int(kv[0]))
    for size, agg in by_size:
        lines.append(
            f"size {size:>3}: n={agg['n_instances']}  "
            f"dp_cost={agg['dp_cost_mean']:.3f}±{agg['dp_cost_stdev']:.3f}  "
            f"heur_cost={agg['heuristic_cost_mean']:.3f}"
            f"±{agg['heuristic_cost_stdev']:.3f}  "
            f"ratio_dp/heur={agg['cost_ratio_dp_over_heuristic_mean']:.4f}  "
            f"wins={agg['dp_wins']} ties={agg['cost_ties']} "
            f"losses={agg['dp_cost_losses']}  "
            f"dp_ms={agg['dp_time_ms_mean']:.3f} "
            f"heur_ms={agg['heuristic_time_ms_mean']:.3f} "
            f"(x{agg['time_ratio_dp_over_heuristic_mean']:.2f})")
    overall = report.get("overall") or {}
    if overall:
        lines.append(
            f"overall: dp_cost={overall['dp_cost_mean']:.3f}"
            f"±{overall['dp_cost_stdev']:.3f}  "
            f"heur_cost={overall['heuristic_cost_mean']:.3f}"
            f"±{overall['heuristic_cost_stdev']:.3f}  "
            f"dp_cost_losses={overall['dp_cost_losses']}  "
            f"time_ratio="
            f"{overall['time_ratio_dp_over_heuristic_mean']:.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(items: Sequence[str]) -> List[int]:
    """Parse one or more comma-separated integer lists (``--sizes 20,50``
    and ``--sizes 20 50`` both work)."""
    values: List[int] = []
    for item in items:
        values.extend(int(part) for part in str(item).split(",")
                      if part.strip())
    return values


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", type=str, nargs="+",
                        default=[str(s) for s in SIZES],
                        help="tree node counts (default: "
                             f"{' '.join(map(str, SIZES))})")
    parser.add_argument("--seeds", type=str, nargs="+",
                        default=[str(s) for s in SEEDS],
                        help="seeds (default: 0-4, i.e. 5 seeds per size)")
    parser.add_argument("--reps", type=int, default=REPS,
                        help=f"timing reps per arm (default: {REPS})")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().parent
                                    / "selector_dp_results.json"),
                        help="output JSON path")
    parser.add_argument("--component-check", type=str, default="",
                        help="pre-bench status of the DP-selector component")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    sizes = tuple(_parse_int_list(args.sizes))
    seeds = tuple(_parse_int_list(args.seeds))
    if not sizes or not seeds:
        print("--sizes and --seeds must be non-empty", file=sys.stderr)
        return 2
    if args.reps < 1:
        print("--reps must be >= 1", file=sys.stderr)
        return 2

    report = run_benchmark(sizes=sizes, seeds=seeds, reps=args.reps,
                           component_check=args.component_check)
    output = Path(args.output)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarize(report))
    print(f"wrote {output}")
    return 0 if report["coverage"]["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
