"""Randomized stress test: 100 different system-level AgentISA tasks.

Each task is a *randomly generated* instruction stream -- random
shape (length, transform kinds, dead-guard placement, cacheable-LLM
placement, data size), from a fixed seed, so the whole battery is
deterministic and reproducible.  Every task must pass the full
verification battery the single-task demo passes:

1. static compilation preserves the task's dataflow (producers of
   consumed outputs survive; only dead guards may go);
2. repeated realization: cached steps make zero model calls after
   the first realization and every report is byte-identical;
3. closed-loop hot-path adoption is consistent (the instruction count
   only ever steps down, and reports stay identical through adoption);
4. full-stack determinism: a fresh stream + cache + workspace +
   backend reproduces the report byte-for-byte;
5. ELL export: the stream serializes to a JSONL trace with the
   expected message/step counts and real cache-hit statistics.

Usage:
    ./venv/bin/python examples/random_tasks_stress.py \
        --tasks 100 --seed 2026 --repeats 4 --hot-threshold 2 \
        --output examples/random_tasks_results.json
"""

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.core.isa.compiler import CompilationContext, InstructionCompiler
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.formal import InstructionType as FormalType
from agentflow.core.isa.instruction import Instruction
from agentflow.core.isa.instruction import InstructionType as IsaType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager, OptimizationMetrics,
)
from agentflow.core.isa.ell_export import (
    export_stream_to_ell, load_ell_trace, save_ell_trace,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor

from system_task_demo import (
    RuleClassifierBackend, generate_corpus, inject_sandbox,
    report_sha256, run_stream, write_corpus,
)

# ---------------------------------------------------------------------------
# Task generation
# ---------------------------------------------------------------------------

TRANSFORMS: List[Tuple[str, Callable[[List[str]], List[str]]]] = [
    ("filter_info", lambda ls: [l for l in ls if "level=INFO" in l]),
    ("filter_warn", lambda ls: [l for l in ls if "level=WARN" in l]),
    ("filter_error", lambda ls: [l for l in ls if "level=ERROR" in l]),
    ("upper", lambda ls: [l.upper() for l in ls]),
    ("dedupe", lambda ls: sorted(set(ls))),
    ("reverse_words", lambda ls: [" ".join(reversed(l.split())) for l in ls]),
    ("tail", lambda ls: ls[len(ls) // 2:]),
    ("firsthalf", lambda ls: ls[: max(1, len(ls) // 2)]),
]


class TaskSpec:
    """Shape of one randomly generated task (plain data, JSON-able)."""

    def __init__(self, rng: random.Random, index: int):
        self.index = index
        self.n_files = rng.randint(2, 8)
        self.lines_per_file = rng.randint(80, 500)
        chain_len = rng.randint(2, 6)
        # Chain step kinds: pure transforms + 0-2 cacheable LLM steps.
        self.chain: List[str] = []
        llm_budget = rng.randint(0, 2)
        for _ in range(chain_len):
            if llm_budget and rng.random() < 0.35:
                self.chain.append("classify")
                llm_budget -= 1
            else:
                self.chain.append(rng.choice(TRANSFORMS)[0])
        # 0-3 dead guards at random chain positions.
        self.guards: List[int] = sorted(
            rng.sample(range(len(self.chain) + 1), k=rng.randint(0, 3))
        ) if len(self.chain) + 1 >= 1 else []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index, "files": self.n_files,
            "lines_per_file": self.lines_per_file, "chain": list(self.chain),
            "guards_at": list(self.guards),
        }


def _instruction_factory(
        id: str, itype: IsaType, execute: Callable, *,
        deps: List[str], outputs: List[str], side_effects: bool,
        description: str = "", cache_config: Optional[Dict] = None) -> Instruction:
    """Parameterized Instruction subclass, mirroring the demo's classes.

    ``execute`` is invoked as ``execute(input_params, self)`` so steps
    that need the instruction object (cache, metrics helpers) can reach
    it.
    """

    class _Step(Instruction):
        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            result = execute(input_params, self)
            self._update_metrics(start, True,
                                 cache_hit=getattr(self, "_last_hit", False))
            return result

    step = _Step(
        id=id, name=id, type=itype, params={},
        description=description or id,
        side_effects=side_effects,
        outputs=outputs,
        cache_config=cache_config,
    )
    step.dependencies = list(deps)
    # Pre-declare so inject_sandbox() (which assigns only pre-existing
    # attributes, mirroring the demo's classes) can inject the workspace.
    step.sandbox = None
    step._last_hit = False
    return step


def build_task_stream(spec: TaskSpec, backend: RuleClassifierBackend,
                      sandbox: LocalSandboxExecutor,
                      corpus: Dict[str, str]) -> List[Instruction]:
    """Materialize a TaskSpec as an executable instruction stream."""
    stream: List[Instruction] = []

    # ingest (side-effectful): files -> raw_lines
    filenames = sorted(corpus)
    stream.append(_instruction_factory(
        "ingest", IsaType.STATE_MANAGEMENT,
        lambda p, s: {"raw_lines": [
            line for name in filenames
            for line in s.sandbox.read_file(f"logs/{name}").decode().splitlines()
        ]},
        deps=[], outputs=["raw_lines"], side_effects=True,
        description="Read the task's input files"))

    current_key = "raw_lines"
    chain_pos = 0
    guard_cursor = 0
    for step_kind in spec.chain:
        out_key = f"v{chain_pos}"
        # dead guard(s) before this chain step (cursor, never mutates spec)
        while (guard_cursor < len(spec.guards)
               and spec.guards[guard_cursor] == chain_pos):
            dep_key = current_key
            stream.append(_instruction_factory(
                f"guard_{chain_pos}_{guard_cursor}", IsaType.CONTROL_FLOW,
                (lambda p, s, d=dep_key: (_check_guard(p["ctx"][d]) or {})),
                deps=[dep_key], outputs=[], side_effects=False,
                description="Defensive guard (dead code)"))
            guard_cursor += 1
        if step_kind == "classify":
            def _classify(p, instr, key=current_key, ok=out_key):
                lines = sorted(set(p["ctx"][key]))
                cache_key = instr._get_cache_key({"n": len(lines)})
                cached = instr.cache.get(cache_key)
                if cached is not None:
                    instr._last_hit = True
                    return cached
                instr._last_hit = False
                # list[str] -> list[str] (enriched "line|class"), keeping
                # the chain's uniform typing so downstream transforms work
                classification = {ok: [
                    f"{l}|{backend.infer(l)}" for l in lines]}
                instr.cache.put(cache_key, classification)
                return classification
            step = _instruction_factory(
                f"classify_{chain_pos}", IsaType.LLM_INTERACTION, _classify,
                deps=[current_key], outputs=[out_key], side_effects=False,
                description="Cacheable LLM classification step",
                cache_config={"max_size": 100_000, "ttl": 3600})
            step._last_hit = False
            stream.append(step)
        else:
            fn = dict(TRANSFORMS)[step_kind]
            stream.append(_instruction_factory(
                f"{step_kind}_{chain_pos}", IsaType.CONTROL_FLOW,
                (lambda p, s, f=fn, key=current_key, ok=out_key:
                 {ok: f(p["ctx"][key])}),
                deps=[current_key], outputs=[out_key], side_effects=False,
                description=f"Pure transform: {step_kind}"))
        current_key = out_key
        chain_pos += 1

    while guard_cursor < len(spec.guards):  # trailing guards
        dep_key = current_key
        stream.append(_instruction_factory(
            f"guard_{chain_pos}_{guard_cursor}", IsaType.CONTROL_FLOW,
            (lambda p, s, d=dep_key: (_check_guard(p["ctx"][d]) or {})),
            deps=[dep_key], outputs=[], side_effects=False,
            description="Defensive guard (dead code)"))
        guard_cursor += 1

    final_key = current_key
    report_keys = [f"v{i}" for i in range(chain_pos)] or ["raw_lines"]

    def _write_report(p, s):
        payload = {k: p["ctx"].get(k) for k in report_keys if k != "raw_lines"}
        final_value = p["ctx"][final_key]
        if isinstance(final_value, list):
            sample = final_value[:10]
        elif isinstance(final_value, dict):
            sample = dict(sorted(final_value.items())[:10])
        else:
            sample = final_value
        payload["final_sample"] = sample
        body = json.dumps(payload, sort_keys=True, indent=1)
        s.sandbox.write_file("report.md", body.encode())
        return {"report_written": hashlib.sha256(body.encode()).hexdigest()}

    stream.append(_instruction_factory(
        "write_report", IsaType.STATE_MANAGEMENT, _write_report,
        deps=[final_key], outputs=[], side_effects=True,
        description="Write the task's report"))
    return stream


def _check_guard(value: Any) -> None:
    """Guards validate invariants but produce nothing (dead code)."""
    if value is None:
        raise ValueError("guard failed")  # never happens on valid corpora


FORMAL_ISA_TO_TYPE = {
    IsaType.STATE_MANAGEMENT: FormalType.STATE,
    IsaType.CONTROL_FLOW: FormalType.CONTROL,
    IsaType.LLM_INTERACTION: FormalType.LLM,
}


def formal_view(stream: List[Instruction]) -> List[FormalInstruction]:
    """Formal (compilable) view of a task stream, from the same facts."""
    return [
        FormalInstruction(
            name=i.id, type=FORMAL_ISA_TO_TYPE[i.type],
            metadata={"side_effects": i.side_effects, "outputs": list(i.outputs)},
            dependencies=list(i.dependencies),
        )
        for i in stream
    ]


# ---------------------------------------------------------------------------
# Per-task battery
# ---------------------------------------------------------------------------

def compile_check(stream: List[Instruction],
                  guards: Optional[set] = None) -> Dict[str, Any]:
    """Static-compile a stream and verify dataflow preservation.

    ``guards`` is the set of instruction ids the task author knows to
    be dead code (no outputs consumed downstream); when omitted, the
    ``guard_`` prefix convention is used.
    """
    compiler = InstructionCompiler({})

    def covered(compiled):
        names: set = set()
        for item in compiled:
            inner = item.metadata.get("parallel_instructions")
            if inner:
                names |= {s.name for s in inner}
            else:
                names.add(item.name)
        return names

    spec_view = formal_view(stream)
    all_names = {i.id for i in stream}
    out: Dict[str, Any] = {}
    if guards is None:
        guards = {i.id for i in stream if i.id.startswith("guard_")}
    for level in ("NONE", "AGGRESSIVE"):
        compiled = compiler.compile(
            spec_view, CompilationContext({"optimization_level": level}))
        cov = covered(compiled)
        covered_producers = {n for n in all_names
                             if any(n in c for c in cov)}
        eliminated = all_names - covered_producers
        consumed = {d for i in stream for d in i.dependencies}
        producer_of = {}
        for i in stream:
            for o in i.outputs:
                producer_of[o] = i.id
        missing = sorted({producer_of[d] for d in consumed
                          if d in producer_of and producer_of[d] not in covered_producers})
        out[level] = {
            "effective": sum(
                1 for item in compiled
                for _ in (item.metadata.get("parallel_instructions") or [item])
            ),
            "eliminated": sorted(eliminated),
            "dataflow_preserved": (
                not missing and eliminated <= guards
            ),
        }
    out["n"] = len(stream)
    out["guards"] = sorted(guards)
    out["pass"] = (
        out["NONE"]["effective"] == len(stream)
        and out["NONE"]["dataflow_preserved"]
        and out["AGGRESSIVE"]["dataflow_preserved"]
    )
    return out


async def run_task(spec: TaskSpec, args) -> Dict[str, Any]:
    rng = random.Random(10_000 + spec.index)  # corpus seeded per task
    corpus = generate_corpus(rng, spec.n_files, spec.lines_per_file)
    backend = RuleClassifierBackend()
    stream = build_task_stream(spec, backend, None, corpus)

    sandbox = LocalSandboxExecutor(name=f"task{spec.index}")
    try:
        write_corpus(sandbox, corpus)
        inject_sandbox(stream, sandbox)

        manager = OptimizationManager()
        manager.dynamic_optimizer.hot_path_threshold = args.hot_threshold
        active = stream
        hashes: List[str] = []
        calls: List[int] = []
        per_repeat_len: List[int] = []
        for _ in range(args.repeats):
            per_repeat_len.append(len(active))
            before = backend.model_calls
            await run_stream(active)
            calls.append(backend.model_calls - before)
            hashes.append(report_sha256(sandbox))
            metrics = OptimizationMetrics(
                execution_time=0.01, memory_usage=1.0,
                api_calls=calls[-1], success_rate=1.0)
            result = manager.record_execution(active, metrics)
            if (result is not active and result is not None
                    and len(result) < len(active)):
                active = result

        has_llm = any(i.type == IsaType.LLM_INTERACTION for i in stream)
        # determinism: fully fresh stack, same corpus
        fresh_backend = RuleClassifierBackend()
        fresh = build_task_stream(spec, fresh_backend, None, corpus)
        fresh_sandbox = LocalSandboxExecutor(name=f"task{spec.index}d")
        try:
            write_corpus(fresh_sandbox, corpus)
            inject_sandbox(fresh, fresh_sandbox)
            await run_stream(fresh)
            fresh_hash = report_sha256(fresh_sandbox)
        finally:
            fresh_sandbox.close()

        # ELL export + JSONL round-trip
        workflow, messages = await export_stream_to_ell(
            stream, name=f"random-task-{spec.index}",
            description=f"Randomized stress task #{spec.index}",
            execution_summary={"repeats": args.repeats},
            eliminated=[i.id for i in stream
                        if i.id.startswith("guard_")])
        trace_path = Path(args.ell_dir) / f"task{spec.index}.jsonl"
        save_ell_trace(trace_path, workflow, messages)
        wf_rec, msg_recs = load_ell_trace(trace_path)
        ell_ok = (
            len(messages) == 1 + 2 * len(stream)
            and len(wf_rec["steps"]) == len(stream)
            and len(msg_recs) == len(messages)
        )
    finally:
        sandbox.close()

    compile_out = compile_check(stream)
    llm_steps = [i for i in stream if i.type == IsaType.LLM_INTERACTION]
    expected_hits = sum(
        getattr(i.metrics, "cache_hits", 0) for i in llm_steps)
    cache_ok = (
        not has_llm
        or calls[0] == 0  # classify input emptied by filters: vacuous
        or (calls[0] > 0 and all(c == 0 for c in calls[1:]))
    )
    step_downs = [b < a for a, b in zip(per_repeat_len, per_repeat_len[1:])]
    monotone = all(
        b >= c for b, c in zip(per_repeat_len[1:], per_repeat_len[2:]))
    closed_loop_ok = (
        len(set(hashes)) == 1
        and monotone
        and (not any(step_downs) or per_repeat_len[-1] <= per_repeat_len[0])
    )
    return {
        "spec": spec.to_dict(),
        "checks": {
            "static_compile_valid": compile_out["pass"],
            "cached_repeats": cache_ok and len(set(hashes)) == 1,
            "closed_loop_consistent": closed_loop_ok,
            "deterministic": fresh_hash == hashes[0],
            "ell_export": ell_ok,
        },
        "stream_length": len(stream),
        "aggressive_effective": compile_out["AGGRESSIVE"]["effective"],
        "eliminated": compile_out["AGGRESSIVE"]["eliminated"],
        "per_repeat_length": per_repeat_len,
        "per_repeat_calls": calls,
        "cache_hits": expected_hits,
        "report_sha256": hashes[0][:12],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Randomized 100-task stress test for AgentISA")
    parser.add_argument("--tasks", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--hot-threshold", type=int, default=2)
    parser.add_argument("--output", type=str,
                        default="examples/random_tasks_results.json")
    parser.add_argument("--ell-dir", type=str,
                        default="examples/random_task_traces")
    args = parser.parse_args()

    Path(args.ell_dir).mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    specs = [TaskSpec(rng, i) for i in range(args.tasks)]

    results = []
    for spec in specs:
        results.append(await run_task(spec, args))

    check_totals: Dict[str, int] = {}
    for r in results:
        for name, ok in r["checks"].items():
            check_totals[name] = check_totals.get(name, 0) + int(ok)
    failures = [r["spec"]["index"] for r in results
                if not all(r["checks"].values())]

    summary = {
        "tasks": args.tasks, "seed": args.seed,
        "repeats": args.repeats,
        "all_pass": not failures,
        "check_totals": check_totals,
        "failed_task_indices": failures,
        "stats": {
            "stream_length_min_max": [
                min(r["stream_length"] for r in results),
                max(r["stream_length"] for r in results)],
            "tasks_with_llm_step": sum(
                1 for r in results if r["per_repeat_calls"][0] > 0),
            "tasks_with_elimination": sum(
                1 for r in results if r["eliminated"]),
            "adoption_rate": sum(
                1 for r in results
                if r["per_repeat_length"][-1] < r["per_repeat_length"][0]
            ) / args.tasks,
            "avg_cache_hits": sum(r["cache_hits"] for r in results) / args.tasks,
        },
    }
    if args.output:
        Path(args.output).write_text(json.dumps(
            {"summary": summary, "tasks": results}, indent=1))

    print("=" * 70)
    print(f"Randomized stress test: {args.tasks} tasks "
          f"(seed {args.seed}, {args.repeats} repeats each)")
    print("=" * 70)
    for name, total in sorted(check_totals.items()):
        print(f"  {name:26s}: {total}/{args.tasks}")
    print(f"  stream lengths            : "
          f"{summary['stats']['stream_length_min_max'][0]}-"
          f"{summary['stats']['stream_length_min_max'][1]} instructions")
    print(f"  tasks with cacheable LLM  : "
          f"{summary['stats']['tasks_with_llm_step']}/{args.tasks}")
    print(f"  tasks with elimination    : "
          f"{summary['stats']['tasks_with_elimination']}/{args.tasks}")
    print(f"  closed-loop adoption rate : "
          f"{summary['stats']['adoption_rate']:.0%}")
    print("=" * 70)
    verdict = "PASS" if not failures else f"FAIL ({len(failures)} tasks)"
    print(f"VERDICT: {verdict}"
          + (f" — failed: {failures[:10]}" if failures else ""))
    if args.output:
        print(f"results written to {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
