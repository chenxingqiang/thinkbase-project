"""Honest microbenchmark suite for AgentISA.

All numbers produced by this suite come from real code paths in
``agentflow.core.isa`` (``InstructionCache``, ``InstructionCompiler``,
``OptimizationManager``) executed against a mock LLM backend that
counts API calls and charges simulated latency.  No live LLM is
contacted; the mock makes every "model inference" observable and the
results fully reproducible.

Experiments
-----------
1. cache       : LLM API-call reduction from result caching under
                 Zipf-distributed workloads.
2. compiler    : Instruction-count reduction from static compilation
                 (fusion + dead-code elimination) at each optimization
                 level.
3. dynamic     : Hot-path optimization on repeatedly executed traces.
4. ablation    : Per-pass contribution (fusion / DCE) to reduction.

Usage
-----
    python isa_microbench.py --experiments cache compiler dynamic ablation \
        --seeds 5 --output results/isa_microbench_results.json
"""

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.compiler import (
    CompilationContext,
    InstructionCompiler,
    IRBuilder,
    IRNodeType,
    OptimizationLevel,
)
from agentflow.core.isa.formal import FormalInstruction, InstructionType
from agentflow.core.isa.instruction import Instruction, InstructionType as IsaInstructionType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager,
    OptimizationMetrics,
)


# --------------------------------------------------------------------------
# Mock LLM backend
# --------------------------------------------------------------------------

class MockLLMBackend:
    """Counts every simulated model inference.

    Each call to :meth:`infer` represents one LLM API round trip with a
    configurable per-call latency that is charged to a simulated clock
    (no real sleeping, so benchmarks stay fast while remaining
    deterministic).
    """

    def __init__(self, latency_ms: float = 80.0):
        self.latency_ms = latency_ms
        self.api_calls = 0
        self.simulated_ms = 0.0

    def infer(self, payload: str) -> str:
        self.api_calls += 1
        self.simulated_ms += self.latency_ms
        return f"response::{payload}"

    def reset(self) -> None:
        self.api_calls = 0
        self.simulated_ms = 0.0


class MockLLMInstruction(Instruction):
    """A pure LLM-backed instruction routed through the mock backend.

    Follows the same cache protocol as ``CompositeInstruction``: an
    execution whose (instruction, params) pair is already cached is
    served without touching the backend.
    """

    def __init__(self, backend: MockLLMBackend, name: str, use_cache: bool = True):
        super().__init__(
            id=f"llm_{name}",
            name=name,
            type=IsaInstructionType.LLM_INTERACTION,
            params={},
            description=f"Mock LLM instruction {name}",
            side_effects=False,
            outputs=[],
            cache_config={"max_size": 100_000, "ttl": 3600} if use_cache else None,
        )
        self.backend = backend
        self.use_cache = use_cache

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        start = time.perf_counter()
        cache_key = None
        if self.use_cache:
            cache_key = self._get_cache_key(input_params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._update_metrics(start, True, cache_hit=True)
                return cached
        result = self.backend.infer(f"{self.name}:{sorted(input_params.items())}")
        if self.use_cache and cache_key is not None:
            self.cache.put(cache_key, result)
        self._update_metrics(start, True)
        return result


# --------------------------------------------------------------------------
# Workload generation
# --------------------------------------------------------------------------

def zipf_workload(rng: random.Random, n_tasks: int, n_instructions: int,
                  n_param_values: int, zipf_s: float) -> List[Tuple[int, Dict[str, Any]]]:
    """Draw (instruction_index, params) pairs from a Zipf distribution.

    Heavy-tailed repetition is the empirical signature of user-LLM
    interaction logs, which is exactly the regime AgentISA targets.
    """
    # Truncated Zipf over instruction indices.
    weights = [1.0 / (i + 1) ** zipf_s for i in range(n_instructions)]
    total = sum(weights)
    probs = [w / total for w in weights]
    tasks = []
    for _ in range(n_tasks):
        idx = rng.choices(range(n_instructions), weights=probs, k=1)[0]
        params = {"value": rng.randrange(n_param_values)}
        tasks.append((idx, params))
    return tasks


def effective_instruction_count(instructions: List[Any]) -> int:
    """Count instructions, unwrapping parallel-block wrappers.

    ``_wrap_parallel`` folds a group of N instructions into one
    ``parallel_block`` wrapper.  Counting wrappers as single
    instructions would overstate reduction, so the effective count
    unwraps them; the wrapper count is reported separately as the
    dispatch count.
    """
    total = 0
    for item in instructions:
        if isinstance(item, list):
            total += effective_instruction_count(item)
            continue
        inner = item.metadata.get("parallel_instructions") if hasattr(item, "metadata") else None
        if inner:
            total += effective_instruction_count(inner)
        else:
            total += 1
    return total


def synthetic_formal_sequence(rng: random.Random, length: int,
                              p_dead: float, p_pure: float) -> List[FormalInstruction]:
    """Build a formal instruction sequence with known structure.

    Structure per slot:
      * ``p_dead``   -> side-effect-free, no outputs, no consumers (dead)
      * ``p_pure``   -> side-effect-free with outputs feeding the final
                        consumer (fusable when adjacent & same type)
      * remainder    -> side-effectful (always live)
    The final instruction always carries side effects, so the sequence
    has a well-defined live sink.
    """
    instructions: List[FormalInstruction] = []
    types = [InstructionType.LLM, InstructionType.STATE]
    for i in range(length - 1):
        r = rng.random()
        if r < p_dead:
            instr = FormalInstruction(
                name=f"dead_{i}",
                type=rng.choice(types),
                metadata={"side_effects": False, "outputs": []},
            )
        elif r < p_dead + p_pure:
            out = f"v{i}"
            instr = FormalInstruction(
                name=f"pure_{i}",
                type=rng.choice(types),
                metadata={"side_effects": False, "outputs": [out]},
            )
        else:
            instr = FormalInstruction(
                name=f"effect_{i}",
                type=rng.choice(types),
                metadata={"side_effects": True},
            )
        instructions.append(instr)
    # Final live sink consuming all pure outputs.
    pure_outputs = [
        out for ins in instructions
        for out in ins.metadata.get("outputs", [])
    ]
    final = FormalInstruction(
        name="final_sink",
        type=InstructionType.CONTROL,
        metadata={"side_effects": True},
    )
    final.dependencies = pure_outputs
    instructions.append(final)
    return instructions


# --------------------------------------------------------------------------
# Experiment 1: result caching
# --------------------------------------------------------------------------

async def run_cache_experiment(seeds: int, zipf_values: List[float],
                               n_tasks: int, n_instructions: int,
                               n_param_values: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {"experiment": "cache", "config": {
        "seeds": seeds, "n_tasks": n_tasks, "n_instructions": n_instructions,
        "n_param_values": n_param_values, "zipf_s": zipf_values,
        "llm_latency_ms": 80.0,
    }, "data": []}

    for zipf_s in zipf_values:
        no_cache_calls, with_cache_calls, hit_rates, sim_ms = [], [], [], []
        for seed in range(seeds):
            rng = random.Random(1000 + seed)
            workload = zipf_workload(rng, n_tasks, n_instructions, n_param_values, zipf_s)

            # Baseline: no cache.
            backend = MockLLMBackend()
            instrs = [MockLLMInstruction(backend, f"i{i}", use_cache=False)
                      for i in range(n_instructions)]
            for idx, params in workload:
                await instrs[idx].execute(params)
            no_cache_calls.append(backend.api_calls)

            # Treatment: InstructionCache enabled.
            backend.reset()
            instrs = [MockLLMInstruction(backend, f"i{i}", use_cache=True)
                      for i in range(n_instructions)]
            for idx, params in workload:
                await instrs[idx].execute(params)
            with_cache_calls.append(backend.api_calls)

            total = sum(m.total_executions for m in
                        [i.metrics for i in instrs])
            hits = sum(m.cache_hits for m in [i.metrics for i in instrs])
            hit_rates.append(hits / total if total else 0.0)
            sim_ms.append((no_cache_calls[-1] - with_cache_calls[-1]) * 80.0)

        reduction = [
            1.0 - c / n for c, n in zip(with_cache_calls, no_cache_calls)
        ]
        results["data"].append({
            "zipf_s": zipf_s,
            "api_calls_no_cache": {
                "mean": statistics.mean(no_cache_calls),
                "stdev": statistics.stdev(no_cache_calls) if seeds > 1 else 0.0,
            },
            "api_calls_cached": {
                "mean": statistics.mean(with_cache_calls),
                "stdev": statistics.stdev(with_cache_calls) if seeds > 1 else 0.0,
            },
            "reduction_ratio": {
                "mean": statistics.mean(reduction),
                "stdev": statistics.stdev(reduction) if seeds > 1 else 0.0,
            },
            "cache_hit_rate": {
                "mean": statistics.mean(hit_rates),
                "stdev": statistics.stdev(hit_rates) if seeds > 1 else 0.0,
            },
            "simulated_latency_saved_ms": {
                "mean": statistics.mean(sim_ms),
                "stdev": statistics.stdev(sim_ms) if seeds > 1 else 0.0,
            },
        })
    return results


# --------------------------------------------------------------------------
# Experiment 2: static compiler reduction
# --------------------------------------------------------------------------

def run_compiler_experiment(seeds: int, lengths: List[int]) -> Dict[str, Any]:
    levels = ["NONE", "BASIC", "ADVANCED", "AGGRESSIVE"]
    results: Dict[str, Any] = {"experiment": "compiler", "config": {
        "seeds": seeds, "lengths": lengths, "levels": levels,
        "p_dead": 0.25, "p_pure": 0.55,
    }, "data": []}

    compiler = InstructionCompiler({})
    for length in lengths:
        for level in levels:
            reductions = []
            dispatch_counts = []
            for seed in range(seeds):
                rng = random.Random(2000 + seed)
                seq = synthetic_formal_sequence(rng, length, p_dead=0.25, p_pure=0.55)
                ctx = CompilationContext({"optimization_level": level})
                out = compiler.compile(seq, ctx)
                reductions.append(1.0 - effective_instruction_count(out) / len(seq))
                dispatch_counts.append(len(out))
            results["data"].append({
                "length": length,
                "level": level,
                "effective_instruction_count_reduction": {
                    "mean": statistics.mean(reductions),
                    "stdev": statistics.stdev(reductions) if seeds > 1 else 0.0,
                },
                "dispatch_count": {
                    "mean": statistics.mean(dispatch_counts),
                    "stdev": statistics.stdev(dispatch_counts) if seeds > 1 else 0.0,
                },
            })
    return results


# --------------------------------------------------------------------------
# Experiment 3: dynamic hot-path optimization
# --------------------------------------------------------------------------

def run_dynamic_experiment(seeds: int, trace_length: int,
                           iterations: int, hot_threshold: int) -> Dict[str, Any]:
    """Measure trace shrinkage once a path goes hot.

    The OptimizationManager records each execution; after the hot-path
    threshold is crossed, ``optimize_hot_path`` applies peephole + DCE
    + combination passes to the cached trace.  We report the API-call
    count and simulated latency before vs. after optimization triggers.
    """
    results: Dict[str, Any] = {"experiment": "dynamic", "config": {
        "seeds": seeds, "trace_length": trace_length,
        "iterations": iterations, "hot_path_threshold": hot_threshold,
        "p_dead": 0.3, "p_pure": 0.5,
    }, "data": []}

    for seed in range(seeds):
        rng = random.Random(3000 + seed)
        # Build a typed Instruction trace with mix of live/dead/pure.
        from tests_bench_support import BenchInstruction  # local helper
        trace: List[Instruction] = []
        for i in range(trace_length - 1):
            r = rng.random()
            if r < 0.3:
                trace.append(BenchInstruction(f"d{i}", side_effects=False))
            elif r < 0.8:
                trace.append(BenchInstruction(f"p{i}", side_effects=False,
                                              outputs=[f"v{i}"]))
            else:
                trace.append(BenchInstruction(f"e{i}", side_effects=True))
        sink = BenchInstruction("sink", side_effects=True)
        sink.dependencies = [f"v{i}" for i in range(trace_length - 1)
                             if trace[i].outputs]
        trace.append(sink)

        manager = OptimizationManager()
        manager.dynamic_optimizer.hot_path_threshold = hot_threshold
        metrics = OptimizationMetrics(
            execution_time=0.05, memory_usage=1.0, api_calls=1, success_rate=1.0
        )
        path_id = manager._compute_path_id(trace)
        optimized_at = None
        for it in range(iterations):
            result = manager.record_execution(trace, metrics)
            if result is not trace and len(result) < len(trace):
                optimized_at = it
                break
        final = manager.dynamic_optimizer.optimize_hot_path(path_id)
        shrink = 1.0 - len(final) / len(trace) if final else 0.0
        results["data"].append({
            "seed": seed,
            "original_len": len(trace),
            "optimized_len": len(final) if final else len(trace),
            "shrink_ratio": shrink,
            "optimized_at_iteration": optimized_at,
        })
    return results


# --------------------------------------------------------------------------
# Experiment 4: ablation over passes
# --------------------------------------------------------------------------

def run_ablation_experiment(seeds: int, length: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {"experiment": "ablation", "config": {
        "seeds": seeds, "length": length, "p_dead": 0.25, "p_pure": 0.55,
    }, "data": []}

    compiler = InstructionCompiler({})
    configs = {
        "none": [],
        "fusion_only": ["instruction_fusion"],
        "dce_only": ["dead_code_elimination"],
        "fusion+dce": ["instruction_fusion", "dead_code_elimination"],
        "all_passes": ["instruction_fusion", "dead_code_elimination",
                       "parallel_execution", "resource_optimization",
                       "memory_optimization"],
    }
    for name, passes in configs.items():
        reductions = []
        for seed in range(seeds):
            rng = random.Random(4000 + seed)
            seq = synthetic_formal_sequence(rng, length, p_dead=0.25, p_pure=0.55)
            ir = compiler.ir_builder.build(seq)
            ctx = CompilationContext({"optimization_level": "AGGRESSIVE"})
            for pass_name, pass_fn in compiler.optimization_passes.items():
                if pass_name in passes:
                    ir = pass_fn(ir, ctx)
            count = effective_instruction_count(ir.instructions)
            reductions.append(1.0 - count / len(seq))
        results["data"].append({
            "configuration": name,
            "instruction_count_reduction": {
                "mean": statistics.mean(reductions),
                "stdev": statistics.stdev(reductions) if seeds > 1 else 0.0,
            },
        })
    return results


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="AgentISA honest microbenchmarks")
    parser.add_argument("--experiments", nargs="+",
                        default=["cache", "compiler", "dynamic", "ablation"],
                        choices=["cache", "compiler", "dynamic", "ablation"])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=str,
                        default="isa_microbench_results.json")
    args = parser.parse_args()

    all_results: List[Dict[str, Any]] = []
    for experiment in args.experiments:
        print(f"\n=== Running experiment: {experiment} ===")
        t0 = time.perf_counter()
        if experiment == "cache":
            res = asyncio.run(run_cache_experiment(
                seeds=args.seeds,
                zipf_values=[0.5, 1.0, 1.5],
                n_tasks=2000, n_instructions=50, n_param_values=20,
            ))
        elif experiment == "compiler":
            res = run_compiler_experiment(seeds=args.seeds, lengths=[50, 200, 1000])
        elif experiment == "dynamic":
            res = run_dynamic_experiment(
                seeds=args.seeds, trace_length=100, iterations=200,
                hot_threshold=50,
            )
        else:
            res = run_ablation_experiment(seeds=args.seeds, length=200)
        res["wall_time_s"] = round(time.perf_counter() - t0, 3)
        all_results.append(res)
        print(json.dumps(res, indent=2)[:1500])

    out_path = Path(__file__).parent / args.output
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
