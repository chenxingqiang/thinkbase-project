"""Chain-enriched fusion benchmark: optimization-level differentiation (E8).

Responds to the ``"optimization gains are small"`` criticism.  The
published compiler ladder (``isa_microbench_results.json``, experiment
``compiler``) shows BASIC == ADVANCED == AGGRESSIVE on effective
instruction-count reduction and ADVANCED == AGGRESSIVE on dispatch
count, because its random-slot workload under-represents chain
structure.  This suite generates *chain-enriched* workloads -- long
runs of pairwise-fusible pure instructions, independent parallel
batches, producer->consumer dataflow chains (with declared memory),
dead runs and effectful anchors -- and quantifies, at
``len(LEVELS) x seeds`` granularity, where the optimization ladder
actually differentiates:

* NONE        -> nothing applied;
* BASIC       -> instruction_fusion + dead_code_elimination;
* ADVANCED    -> + parallel_execution + resource_optimization;
* AGGRESSIVE  -> + memory_optimization (metadata-only plan).

A random-slot control arm reuses the published generator
(``isa_microbench.synthetic_formal_sequence``, same seeds) so the two
arms are measured by the identical harness; the control reproduces
the published numbers exactly.

All aggregates (means / stdevs / deltas / verdicts) are computed by
this script from the raw per-seed records written into the JSON --
nothing is hand-computed or hand-filled.

Usage
-----
    python fusion_chain_bench.py --seeds 5 --lengths 50,200,1000 \
        --output fusion_chain_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.compiler import (  # noqa: E402
    CompilationContext,
    InstructionCompiler,
    OptimizationLevel,
)
from agentflow.core.isa.formal import (  # noqa: E402
    FormalInstruction,
    InstructionType,
)

import isa_microbench as mb  # noqa: E402  (sibling module in examples/benchmarks)

LEVELS = ["NONE", "BASIC", "ADVANCED", "AGGRESSIVE"]

#: Segment mix of the chain-enriched workload (must sum to 1.0).
CHAIN_MIX: Dict[str, float] = {
    "fusion_chain": 0.35,      # runs of pure same-type no-dep instructions
    "parallel_batch": 0.30,    # runs of pure alternating-type instructions
    "dataflow_chain": 0.15,    # producer -> consumer chains, declared memory
    "dead_chain": 0.08,        # pure instructions with no outputs (dead)
    "effectful_anchor": 0.12,  # single side-effectful instructions
}

SEGMENT_ORDER = tuple(CHAIN_MIX)

#: Run-length ranges (inclusive) per segment type.
SEGMENT_RUN_RANGES: Dict[str, Tuple[int, int]] = {
    "fusion_chain": (4, 12),
    "parallel_batch": (4, 10),
    "dataflow_chain": (3, 8),
    "dead_chain": (2, 6),
    "effectful_anchor": (1, 1),
}

#: Declared-cost model draws.
COST_RANGE = (0.5, 2.0)
MEMORY_SIZES = (64, 128)
DATAFLOW_MEMORY_SIZES = (256, 512, 1024, 2048)

#: Seed bases.  The control arm mirrors the published microbench
#: compiler experiment (seed base 2000) so it reproduces those numbers
#: exactly; the chain arm uses a distinct base.
CONTROL_SEED_BASE = 2000
CHAIN_SEED_BASE_DEFAULT = 5000

#: Metrics that describe the *structure* of the compiled program.  Two
#: levels are structurally identical when these match per seed.
STRUCTURAL_METRIC_KEYS = (
    "effective_instruction_count",
    "instruction_count_reduction",
    "dispatch_count",
    "declared_cost_makespan",
    "fused_instruction_count",
    "parallel_block_wrappers",
    "max_parallel_width",
)

MEMORY_METRIC_KEYS = (
    "peak_live_outputs",
    "distinct_tracked_outputs",
    "buffer_reuse_potential",
    "total_declared_memory_bytes",
)

WORKLOAD_TYPES = (InstructionType.LLM, InstructionType.STATE,
                  InstructionType.CONTROL)


# ---------------------------------------------------------------------------
# Chain-enriched workload generation
# ---------------------------------------------------------------------------

@dataclass
class Workload:
    """A generated workload: instructions plus structural census."""

    instructions: List[FormalInstruction]
    census: Dict[str, int]
    runs: List[Tuple[str, int]] = field(default_factory=list)


def _pick_segment(rng: random.Random, weights: Dict[str, float]) -> str:
    """Weighted segment choice over SEGMENT_ORDER (deterministic)."""
    r = rng.random()
    acc = 0.0
    for name in SEGMENT_ORDER:
        weight = weights.get(name, 0.0)
        if weight <= 0.0:
            continue
        acc += weight
        if r < acc:
            return name
    return "effectful_anchor"  # numeric fallback for r >= sum(weights)


def generate_chain_workload(rng: random.Random, length: int,
                            weights: Dict[str, float]) -> Workload:
    """Build a chain-enriched formal-instruction sequence.

    The body (``length - 1`` instructions) is assembled from segments
    drawn by ``weights`` until exactly filled:

    * ``fusion_chain``   -- run of pure, same-type, output-declaring
      instructions with no dependencies: every adjacent pair satisfies
      ``InstructionCompiler._are_fusible``;
    * ``parallel_batch`` -- run of pure, dependency-free instructions
      with alternating types (blocks pairwise fusion, keeps the run
      parallelizable);
    * ``dataflow_chain``  -- each link consumes the previous link's
      output (producer->consumer edges block fusion and
      parallelization, create live ranges for the memory plan) and
      declares a large memory footprint;
    * ``dead_chain``      -- pure instructions with no outputs and no
      consumers (removed by dead-code elimination);
    * ``effectful_anchor``-- a single side-effectful instruction
      (always live, breaks fusion and parallel groups).

    The final instruction is an effectful sink depending on exactly
    the *leaf* outputs (outputs no other instruction consumes), so the
    sequence has a well-defined live sink while chain-internal
    intermediates stay internal.  The sink is counted in the census as
    an effectful anchor, so the per-segment counts sum to the total.
    """
    # 1. Segment plan until the body is exactly filled.
    plan: List[Tuple[str, int]] = []
    built = 0
    while built < length - 1:
        seg = _pick_segment(rng, weights)
        low, high = SEGMENT_RUN_RANGES[seg]
        run = rng.randint(low, high)
        run = min(run, length - 1 - built)
        plan.append((seg, run))
        built += run

    # 2. Materialize instructions from the plan.
    instructions: List[FormalInstruction] = []
    consumed: set = set()
    census: Dict[str, int] = {f"{seg}_instructions": 0 for seg in SEGMENT_ORDER}
    idx = 0
    for seg, run in plan:
        census[f"{seg}_instructions"] += run
        if seg == "effectful_anchor":
            for _ in range(run):
                instructions.append(FormalInstruction(
                    name=f"anchor_{idx}",
                    type=rng.choice(WORKLOAD_TYPES),
                    metadata={
                        "side_effects": True,
                        "cost": rng.uniform(*COST_RANGE),
                        "memory": rng.choice(MEMORY_SIZES),
                    }))
                idx += 1
            continue
        seg_type = rng.choice(WORKLOAD_TYPES)
        prev_out: Optional[str] = None
        for _ in range(run):
            if seg == "dead_chain":
                instructions.append(FormalInstruction(
                    name=f"dead_{idx}",
                    type=seg_type,
                    metadata={
                        "side_effects": False,
                        "outputs": [],
                        "cost": rng.uniform(*COST_RANGE),
                        "memory": rng.choice(MEMORY_SIZES),
                    }))
            else:
                out = f"v{idx}"
                deps: List[str] = []
                if seg == "dataflow_chain" and prev_out is not None:
                    deps = [prev_out]
                    consumed.add(prev_out)
                memory = (rng.choice(DATAFLOW_MEMORY_SIZES)
                          if seg == "dataflow_chain"
                          else rng.choice(MEMORY_SIZES))
                instr = FormalInstruction(
                    name=f"{seg}_{idx}",
                    type=seg_type,
                    metadata={
                        "side_effects": False,
                        "outputs": [out],
                        "cost": rng.uniform(*COST_RANGE),
                        "memory": memory,
                    })
                instr.dependencies = deps
                instructions.append(instr)
                prev_out = out
                if seg == "parallel_batch":
                    others = [t for t in WORKLOAD_TYPES if t != seg_type]
                    seg_type = rng.choice(others)
            idx += 1

    # 3. Effectful sink consuming exactly the leaf outputs.
    all_outputs = [out for instr in instructions
                   for out in instr.metadata.get("outputs", [])]
    leaves = [out for out in dict.fromkeys(all_outputs)
              if out not in consumed]
    sink = FormalInstruction(
        name="final_sink",
        type=InstructionType.CONTROL,
        metadata={
            "side_effects": True,
            "cost": rng.uniform(*COST_RANGE),
            "memory": rng.choice(MEMORY_SIZES),
        })
    sink.dependencies = leaves
    instructions.append(sink)

    # The final sink is itself an effectful anchor for census purposes,
    # so the per-segment counts sum to the workload total.
    census["effectful_anchor_instructions"] += 1
    census["total"] = len(instructions)
    census["declared_outputs"] = len(set(all_outputs))
    census["leaf_outputs"] = len(leaves)
    return Workload(instructions=instructions, census=census, runs=plan)


def chain_workload_factory(
        weights: Dict[str, float]) -> Callable[[random.Random, int],
                                                List[FormalInstruction]]:
    """Factory adapter: (rng, length) -> chain-enriched instructions."""
    def factory(rng: random.Random, length: int) -> List[FormalInstruction]:
        return generate_chain_workload(rng, length, weights).instructions
    return factory


def control_workload_factory() -> (
        Callable[[random.Random, int], List[FormalInstruction]]):
    """Control arm: the published random-slot generator (same seeds)."""
    def factory(rng: random.Random,
                length: int) -> List[FormalInstruction]:
        return mb.synthetic_formal_sequence(rng, length,
                                            p_dead=0.25, p_pure=0.55)
    return factory


def control_census(instructions: List[FormalInstruction]) -> Dict[str, int]:
    """Census for control workloads (roles by declared metadata)."""
    body = instructions[:-1]
    dead = pure = effectful = 0
    for instr in body:
        effectful_flag = bool(instr.metadata.get("side_effects", True))
        outputs = instr.metadata.get("outputs") or []
        if effectful_flag:
            effectful += 1
        elif outputs:
            pure += 1
        else:
            dead += 1
    return {
        "dead_instructions": dead,
        "pure_instructions": pure,
        "effectful_instructions": effectful,
        "total": len(instructions),
    }


# ---------------------------------------------------------------------------
# Compile wrapper with IR capture
# ---------------------------------------------------------------------------

def compile_and_capture_ir(
        compiler: InstructionCompiler,
        instructions: List[FormalInstruction],
        context: CompilationContext):
    """Run ``compiler.compile`` while capturing the final IR node.

    The wrapper swaps ``code_generator.generate`` for a pass-through
    closure that records the IR handed to code generation -- the
    post-optimization-pass IR whose metadata carries the
    ``resource_plan`` (ADVANCED+) and ``memory_plan`` (AGGRESSIVE) the
    passes attach.  The output is the genuine ``compile`` result (the
    wrapper never touches it), and the patched method is fully removed
    afterwards.
    """
    generator = compiler.code_generator
    captured: Dict[str, Any] = {}
    original = generator.generate

    def generate(ir, ctx):
        captured["ir"] = ir
        return original(ir, ctx)

    generator.generate = generate
    try:
        output = compiler.compile(instructions, context)
    finally:
        # Remove the instance-attribute shadow entirely; lookups then
        # fall through to the class-bound method again.
        generator.__dict__.pop("generate", None)
    return output, captured.get("ir")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: Reused from the published microbench so wrapper-unwrapping cannot
#: drift between the two suites.
effective_instruction_count = mb.effective_instruction_count


def dispatch_count(output: List[Any]) -> int:
    """Number of top-level dispatch units in a compiled program."""
    return len(output)


def _parallel_members(item: Any) -> Optional[List[Any]]:
    """Members of a parallel_block wrapper, or None if not a wrapper."""
    if hasattr(item, "metadata"):
        inner = item.metadata.get("parallel_instructions")
        if inner:
            return inner
    return None


def _declared_cost(instr: Any) -> float:
    return float(instr.metadata.get("cost", 1.0))


def declared_cost_makespan(output: List[Any]) -> float:
    """Declared-cost makespan proxy.

    Sequential dispatch units contribute their declared cost; a
    parallel_block wrapper contributes the *maximum* member cost (its
    members run concurrently).  Costs come from
    ``metadata['cost']`` (default 1.0).
    """
    total = 0.0
    for item in output:
        members = _parallel_members(item)
        if members is not None:
            total += max(_declared_cost(m) for m in members)
        else:
            total += _declared_cost(item)
    return total


def fused_instruction_count(output: List[Any]) -> int:
    """Surviving top-level fused artifacts (``fused_*`` names)."""
    return sum(1 for item in output
               if getattr(item, "name", "").startswith("fused_"))


def parallel_block_stats(output: List[Any]) -> Tuple[int, int]:
    """(wrapper count, max width) over the compiled program."""
    wrappers = 0
    max_width = 0
    for item in output:
        members = _parallel_members(item)
        if members is not None:
            wrappers += 1
            max_width = max(max_width, len(members))
    return wrappers, max_width


def memory_plan_metrics(plan: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Derive peak-liveness metrics from a compiler memory plan.

    ``peak_live_outputs`` is the maximum number of simultaneously-live
    outputs over the flattened program (conservative tie-break: at
    equal positions, production is counted before release).
    ``buffer_reuse_potential`` is ``distinct_tracked_outputs -
    peak_live_outputs``: the buffers a disjoint-reuse allocator could
    recycle.  Returns ``None`` when the compiler emitted no plan.
    """
    if not plan:
        return None
    live_ranges = plan.get("lifecycle", {}).get("live_ranges", {})
    events: List[Tuple[int, int]] = []
    for production, last_use in live_ranges.values():
        events.append((production, 1))
        events.append((last_use + 1, -1))
    events.sort(key=lambda event: (event[0], -event[1]))
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    distinct = len(live_ranges)
    allocation = plan.get("allocation", {})
    return {
        "peak_live_outputs": peak,
        "distinct_tracked_outputs": distinct,
        "buffer_reuse_potential": distinct - peak,
        "total_declared_memory_bytes": allocation.get("total_bytes"),
        "optimizations": list(plan.get("optimization", [])),
    }


# ---------------------------------------------------------------------------
# Experiment execution
# ---------------------------------------------------------------------------

def compile_record(compiler: InstructionCompiler,
                   instructions: List[FormalInstruction],
                   level: str, length: int, seed: int,
                   arm: str) -> Dict[str, Any]:
    """Compile one workload at one level and record raw metrics."""
    context = CompilationContext({"optimization_level": level})
    output, ir = compile_and_capture_ir(compiler, instructions, context)
    effective = effective_instruction_count(output)
    wrappers, max_width = parallel_block_stats(output)
    plan = ir.metadata.get("memory_plan") if ir is not None else None
    memory = memory_plan_metrics(plan)
    total = len(instructions)
    record: Dict[str, Any] = {
        "arm": arm,
        "length": length,
        "level": level,
        "seed": seed,
        "original_count": total,
        "effective_instruction_count": effective,
        "instruction_count_reduction": 1.0 - effective / total,
        "dispatch_count": dispatch_count(output),
        "declared_cost_makespan": declared_cost_makespan(output),
        "fused_instruction_count": fused_instruction_count(output),
        "parallel_block_wrappers": wrappers,
        "max_parallel_width": max_width,
        "memory_plan_present": memory is not None,
    }
    for key in MEMORY_METRIC_KEYS:
        record[key] = memory[key] if memory is not None else None
    record["memory_plan_optimizations"] = (memory["optimizations"]
                                           if memory is not None else None)
    return record


def run_arm(compiler: InstructionCompiler,
            workload_factory: Callable[[random.Random, int],
                                       List[FormalInstruction]],
            lengths: List[int], levels: List[str], seeds: int,
            seed_base: int, arm: str) -> List[Dict[str, Any]]:
    """Run one workload arm over lengths x seeds x levels.

    The same seeded workload is compiled at every level (paired per
    seed), so level comparisons are within-seed.
    """
    records: List[Dict[str, Any]] = []
    for length in lengths:
        for seed in range(seeds):
            rng = random.Random(seed_base + seed)
            instructions = workload_factory(rng, length)
            for level in levels:
                records.append(compile_record(
                    compiler, instructions, level, length, seed, arm))
    return records


def aggregate_records(
        records: List[Dict[str, Any]]) -> Dict[Tuple[str, str],
                                               Dict[str, Any]]:
    """Aggregate raw records by (length, level) via statistics.

    Every aggregate is ``statistics.mean``/``statistics.stdev`` applied
    to the raw per-seed values; a single-seed cell gets stdev 0.0.
    Memory metrics aggregate only over seeds where the plan exists and
    stay ``None`` otherwise.
    """
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["length"]), record["level"]),
                           []).append(record)
    aggregates: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for (length, level), cell_records in grouped.items():
        cell: Dict[str, Any] = {
            "length": int(length),
            "level": level,
        }
        for key in STRUCTURAL_METRIC_KEYS:
            values = [r[key] for r in cell_records]
            cell[key] = {
                "mean": statistics.mean(values),
                "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
            }
        for key in MEMORY_METRIC_KEYS:
            values = [r[key] for r in cell_records if r[key] is not None]
            cell[key] = ({"mean": statistics.mean(values),
                          "stdev": statistics.stdev(values)
                          if len(values) > 1 else 0.0}
                         if values else None)
        with_plan = [r for r in cell_records if r["memory_plan_present"]]
        cell["memory_plan_present"] = {
            "seeds_with_plan": len(with_plan),
            "seeds_total": len(cell_records),
        }
        unions = sorted({opt for r in with_plan
                          for opt in (r["memory_plan_optimizations"] or [])})
        cell["memory_plan_optimizations"] = {"union": unions} if unions \
            else None
        aggregates[(length, level)] = cell
    return aggregates


def _mean(values: List[float]) -> Optional[float]:
    return statistics.mean(values) if values else None


def _pair_records(records: List[Dict[str, Any]], length: int,
                  lower: str, upper: str) -> List[Tuple[Dict, Dict]]:
    by_level: Dict[str, Dict[int, Dict]] = {"lower": {}, "upper": {}}
    for record in records:
        if record["length"] != length:
            continue
        if record["level"] == lower:
            by_level["lower"][record["seed"]] = record
        elif record["level"] == upper:
            by_level["upper"][record["seed"]] = record
    return [(by_level["lower"][seed], by_level["upper"][seed])
            for seed in sorted(set(by_level["lower"])
                               & set(by_level["upper"]))]


def differentiate(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-length level-ladder differentiation from raw records.

    For each consecutive level pair, reports mean deltas (upper minus
    lower) for every structural metric, whether the two levels are
    structurally identical per seed, and the memory-plan
    differentiation between them.
    """
    entries: List[Dict[str, Any]] = []
    lengths = sorted({record["length"] for record in records})
    for length in lengths:
        pairs: List[Dict[str, Any]] = []
        for lower, upper in zip(LEVELS, LEVELS[1:]):
            matched = _pair_records(records, length, lower, upper)
            entry: Dict[str, Any] = {"pair": [lower, upper]}
            if not matched:
                entry["deltas"] = None
                entry["structural_metrics_identical_per_seed"] = False
                entry["memory_plan_differentiation"] = None
                pairs.append(entry)
                continue
            deltas = {}
            identical = True
            for key in STRUCTURAL_METRIC_KEYS:
                deltas[key] = (_mean([u[key] for _, u in matched])
                               - _mean([l[key] for l, _ in matched]))
                if any(l[key] != u[key] for l, u in matched):
                    identical = False
            entry["deltas"] = deltas
            entry["structural_metrics_identical_per_seed"] = identical
            upper_reuse = [u["buffer_reuse_potential"] for _, u in matched
                           if u["buffer_reuse_potential"] is not None]
            entry["memory_plan_differentiation"] = {
                "lower_seeds_with_plan": sum(
                    1 for l, _ in matched if l["memory_plan_present"]),
                "upper_seeds_with_plan": sum(
                    1 for _, u in matched if u["memory_plan_present"]),
                "upper_mean_buffer_reuse_potential": (
                    _mean(upper_reuse) if upper_reuse else None),
            }
            pairs.append(entry)
        entries.append({"length": length, "pairs": pairs})
    return entries


def verdict(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Script-computed headline verdict for one arm's records."""
    index = {(r["length"], r["seed"], r["level"]): r for r in records}
    lengths = sorted({r["length"] for r in records})
    seeds = sorted({r["seed"] for r in records})

    structurally_identical = all(
        index[(length, seed, "ADVANCED")][key]
        == index[(length, seed, "AGGRESSIVE")][key]
        for length in lengths for seed in seeds
        for key in STRUCTURAL_METRIC_KEYS)

    aggressive_plan_all = all(
        index[(length, seed, "AGGRESSIVE")]["memory_plan_present"]
        for length in lengths for seed in seeds)
    advanced_plan_any = any(
        index[(length, seed, "ADVANCED")]["memory_plan_present"]
        for length in lengths for seed in seeds)
    reuse_values = [
        index[(length, seed, "AGGRESSIVE")]["buffer_reuse_potential"]
        for length in lengths for seed in seeds
        if index[(length, seed, "AGGRESSIVE")]["buffer_reuse_potential"]
        is not None]
    memory_differentiates = bool(
        aggressive_plan_all and not advanced_plan_any and reuse_values
        and _mean(reuse_values) > 0)

    max_length = max(lengths)
    levels_present = [level for level in LEVELS
                      if any((max_length, seed, level) in index
                             for seed in seeds)]
    ladder = {}
    for level in levels_present:
        cell = [index[(max_length, seed, level)] for seed in seeds]
        ladder[level] = {
            key: _mean([r[key] for r in cell])
            for key in ("effective_instruction_count",
                        "instruction_count_reduction", "dispatch_count",
                        "declared_cost_makespan")
        }
    matched = _pair_records(records, max_length, "ADVANCED", "AGGRESSIVE")
    aggr_adv: Dict[str, Any] = {}
    if matched:
        for key in STRUCTURAL_METRIC_KEYS:
            aggr_adv[key] = (_mean([u[key] for _, u in matched])
                             - _mean([l[key] for l, _ in matched]))
        upper_reuse = [u["buffer_reuse_potential"] for _, u in matched
                       if u["buffer_reuse_potential"] is not None]
        aggr_adv["aggressive_mean_buffer_reuse_potential"] = (
            _mean(upper_reuse) if upper_reuse else None)

    return {
        "aggressive_structurally_identical_to_advanced":
            structurally_identical,
        "aggressive_memory_plan_differentiates": memory_differentiates,
        "max_length": max_length,
        "ladder_at_max_length": ladder,
        "aggressive_vs_advanced_at_max_length": aggr_adv,
    }


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------

def get_git_rev() -> str:
    """Short HEAD rev via read-only ``git rev-parse --short HEAD``."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True)
        return result.stdout.strip()
    except Exception:  # pragma: no cover - environment-dependent
        return "unavailable"


def _census_block(lengths: List[int], seeds: int, seed_base: int,
                   weights: Dict[str, float]) -> List[Dict[str, Any]]:
    entries = []
    for length in lengths:
        per_seed = []
        for seed in range(seeds):
            rng = random.Random(seed_base + seed)
            per_seed.append(
                generate_chain_workload(rng, length, weights).census)
        keys = sorted(per_seed[0])
        entries.append({
            "length": length,
            "per_seed": per_seed,
            "mean": {key: _mean([c[key] for c in per_seed])
                     for key in keys},
        })
    return entries


def _control_census_block(lengths: List[int], seeds: int,
                          seed_base: int) -> List[Dict[str, Any]]:
    factory = control_workload_factory()
    entries = []
    for length in lengths:
        per_seed = []
        for seed in range(seeds):
            rng = random.Random(seed_base + seed)
            per_seed.append(control_census(factory(rng, length)))
        keys = sorted(per_seed[0])
        entries.append({
            "length": length,
            "per_seed": per_seed,
            "mean": {key: _mean([c[key] for c in per_seed])
                     for key in keys},
        })
    return entries


def _aggregates_as_list(aggregates: Dict[Tuple[str, str],
                                          Dict[str, Any]]) -> List[Dict]:
    level_rank = {level: rank for rank, level in enumerate(LEVELS)}
    return sorted(aggregates.values(),
                  key=lambda cell: (cell["length"],
                                   level_rank[cell["level"]]))


def build_results(lengths: List[int], seeds: int,
                  seed_base: int = CHAIN_SEED_BASE_DEFAULT,
                  git_rev: Optional[str] = None,
                  timestamp: Optional[str] = None) -> Dict[str, Any]:
    """Run both arms and assemble the full results document."""
    compiler = InstructionCompiler({})

    started = time.perf_counter()
    chain_records = run_arm(
        compiler, chain_workload_factory(CHAIN_MIX), lengths, LEVELS,
        seeds, seed_base, "chain_enriched")
    chain_wall = time.perf_counter() - started

    started = time.perf_counter()
    control_records = run_arm(
        compiler, control_workload_factory(), lengths, LEVELS, seeds,
        CONTROL_SEED_BASE, "random_slots_control")
    control_wall = time.perf_counter() - started

    chain_aggregates = aggregate_records(chain_records)
    control_aggregates = aggregate_records(control_records)

    passes_present = {
        name: callable(fn)
        for name, fn in compiler.optimization_passes.items()}
    passes_by_level = {}
    for level in LEVELS:
        context = CompilationContext({"optimization_level": level})
        passes_by_level[level] = sorted(
            name for name in compiler.optimization_passes
            if compiler._should_apply_pass(name, context))

    cells_planned = len(lengths) * len(LEVELS) * seeds * 2
    cells_run = len(chain_records) + len(control_records)
    coverage = {
        "status": ("complete" if cells_run == cells_planned
                   and all(passes_present.values()) else "partial"),
        "cells_planned": cells_planned,
        "cells_run": cells_run,
        "compiler_passes_present": passes_present,
        "passes_applied_by_level": passes_by_level,
        "limitations": [
            "AGGRESSIVE's memory_optimization pass only annotates the "
            "IR metadata; no executor consumes the plan, so memory "
            "metrics are plan-derived analyses, not measured runtime "
            "savings.",
            "declared_cost_makespan is a metadata cost model (a "
            "parallel_block contributes the max member cost), not "
            "wall-clock time.",
            "peak_live_outputs uses a conservative tie-break: at equal "
            "positions production is counted before release.",
        ],
    }

    results = {
        "experiment": "fusion_chain_level_differentiation",
        "purpose": (
            "Respond to the 'optimization gains are small' criticism by "
            "quantifying optimization-level differentiation under "
            "chain-enriched workloads (4 levels x 5 seeds), against a "
            "random-slot control arm that reproduces the published "
            "microbench compiler numbers."),
        "provenance": {
            "git_rev": git_rev if git_rev is not None else get_git_rev(),
            "timestamp": (timestamp if timestamp is not None else
                           datetime.now(timezone.utc).isoformat()),
            "timezone": "UTC",
            "seeds": list(range(seeds)),
            "seed_base": {
                "chain_enriched": seed_base,
                "random_slots_control": CONTROL_SEED_BASE,
            },
            "scale": {
                "lengths": list(lengths),
                "seeds_per_length": seeds,
                "levels": list(LEVELS),
                "arms": 2,
            },
            "python": platform.python_version(),
            "determinism": (
                "workloads and compiles are fully seeded; only the "
                "timestamp and git rev vary between runs of this script"),
            "compiler": "agentflow/core/isa/compiler.py",
            "chain_generator": (
                "examples/benchmarks/fusion_chain_bench.py::"
                "generate_chain_workload"),
            "control_generator": (
                "examples/benchmarks/isa_microbench.py::"
                "synthetic_formal_sequence"),
        },
        "config": {
            "chain_mix": dict(CHAIN_MIX),
            "segment_run_ranges": {
                k: list(v) for k, v in SEGMENT_RUN_RANGES.items()},
            "cost_range": list(COST_RANGE),
            "memory_sizes": list(MEMORY_SIZES),
            "dataflow_memory_sizes": list(DATAFLOW_MEMORY_SIZES),
            "control_config": {"p_dead": 0.25, "p_pure": 0.55},
        },
        "metric_definitions": {
            "effective_instruction_count": (
                "instruction count after compilation, unwrapping "
                "parallel_block wrappers (reused from isa_microbench)"),
            "instruction_count_reduction": (
                "1 - effective_instruction_count / original_count"),
            "dispatch_count": (
                "number of top-level dispatch units in the compiled "
                "program; a parallel_block wrapper is one dispatch"),
            "declared_cost_makespan": (
                "sum of declared costs over dispatch units; a "
                "parallel_block contributes max(member costs) since "
                "members run concurrently; metadata['cost'] default 1.0"),
            "fused_instruction_count": (
                "surviving top-level instructions named 'fused_*'"),
            "peak_live_outputs": (
                "from the AGGRESSIVE memory plan lifecycle: max "
                "simultaneously-live outputs (conservative tie-break)"),
            "buffer_reuse_potential": (
                "distinct_tracked_outputs - peak_live_outputs: buffers "
                "a disjoint-reuse allocator could recycle, from the "
                "AGGRESSIVE memory plan"),
            "structural_metrics_identical_per_seed": (
                "all STRUCTURAL_METRIC_KEYS match per seed between the "
                "two levels"),
            "aggressive_memory_plan_differentiates": (
                "AGGRESSIVE emits a memory plan on every seed, ADVANCED "
                "on none, and the mean buffer_reuse_potential is > 0"),
        },
        "coverage": coverage,
        "workload_census": {
            "chain_enriched": _census_block(lengths, seeds, seed_base,
                                           CHAIN_MIX),
            "random_slots_control": _control_census_block(
                lengths, seeds, CONTROL_SEED_BASE),
        },
        "data": {
            "chain_enriched": chain_records,
            "random_slots_control": control_records,
        },
        "aggregates": {
            "chain_enriched": _aggregates_as_list(chain_aggregates),
            "random_slots_control": _aggregates_as_list(control_aggregates),
        },
        "differentiation": {
            "chain_enriched": differentiate(chain_records),
            "random_slots_control": differentiate(control_records),
        },
        "verdicts": {
            "chain_enriched": verdict(chain_records),
            "random_slots_control": verdict(control_records),
        },
        "wall_time_s": {
            "chain_enriched": chain_wall,
            "random_slots_control": control_wall,
        },
    }
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Chain-enriched fusion benchmark: 4 optimization "
                    "levels x N seeds; quantifies level differentiation "
                    "against the published random-slot control.")
    parser.add_argument("--output", default=str(
        Path(__file__).with_name("fusion_chain_results.json")),
        help="output JSON path (default: alongside this script)")
    parser.add_argument("--seeds", type=int, default=5,
                        help="seeds per length per arm (default: 5)")
    parser.add_argument("--lengths", default="50,200,1000",
                        help="comma-separated workload lengths")
    parser.add_argument("--seed-base", type=int,
                        default=CHAIN_SEED_BASE_DEFAULT,
                        help="rng seed base for the chain-enriched arm")
    args = parser.parse_args(argv)

    lengths = [int(part) for part in args.lengths.split(",") if part.strip()]
    results = build_results(lengths, seeds=args.seeds,
                            seed_base=args.seed_base)

    output_path = Path(args.output)
    with output_path.open("w") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")

    for arm in ("chain_enriched", "random_slots_control"):
        verdicts = results["verdicts"][arm]
        ladder = verdicts["ladder_at_max_length"]
        print(f"[{arm}] ladder at length {verdicts['max_length']} "
              "(mean dispatch / makespan / reduction):")
        for level in LEVELS:
            cell = ladder[level]
            print(f"  {level:<10} dispatch={cell['dispatch_count']:.1f} "
                  f"makespan={cell['declared_cost_makespan']:.1f} "
                  f"reduction={cell['instruction_count_reduction']:.3f}")
        print(f"  aggressive structurally identical to advanced: "
              f"{verdicts['aggressive_structurally_identical_to_advanced']}"
              f"; memory plan differentiates: "
              f"{verdicts['aggressive_memory_plan_differentiates']}")
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
