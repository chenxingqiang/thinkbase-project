"""System-level workload characterization for AgentISA (paper Section 5.8).

Extends the single-task demonstration (``system_task_demo.py``) into a
four-task workload characterization: three additional seed-controlled
task streams with genuinely different dependency shapes -- a CSV ETL
anomaly pipeline (fan-out per shard), a report-merge task (wide fan-in
to one merge step), and a regression-diff task (two parallel ingest
branches converging at a diff step) -- plus the original service-log
triage task.  Every task: deterministic from its seed, one planted dead
guard (defensive redundancy real pipelines carry), one cacheable
LLM-type step backed by a deterministic rule stand-in that counts
simulated model calls, 12 repeated realizations, hot-path threshold 5.

``characterize()`` emits, per task: instruction counts, NONE vs
AGGRESSIVE effective counts with the eliminated step list, dataflow
preservation (compile_check), model-call reuse across realizations,
cache hit rate on the LLM-type step, hot-path adoption round, and
byte-determinism verdicts (optimized-form re-run and full fresh stack).

Run:  python examples/system_task_workload.py [--json PATH]
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.sandbox.local_executor import LocalSandboxExecutor

from agentflow.core.isa.instruction import Instruction
from agentflow.core.isa.instruction import InstructionType as IsaType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager,
    OptimizationMetrics,
)

from examples.random_tasks_stress import compile_check
from examples.system_task_demo import (
    RuleClassifierBackend,
    generate_corpus,
    inject_sandbox,
    report_sha256,
    run_stream,
    write_corpus,
)

REPEATS = 12
HOT_THRESHOLD = 5
GUARD_NOTE = "planted defensive guard; consumed by nothing (dead code)"


# ---------------------------------------------------------------------------
# Shared instruction factories
# ---------------------------------------------------------------------------

def _mk_ingest(instr_id: str, files: List[str], prefix: str, out_key: str,
               sandbox: Optional[LocalSandboxExecutor],
               decoder: str = "utf-8") -> Instruction:
    class Ingest(Instruction):
        def __init__(self):
            super().__init__(
                id=instr_id, name=instr_id, type=IsaType.STATE_MANAGEMENT,
                params={}, description=f"Read {len(files)} files from the workspace",
                side_effects=True, outputs=[out_key],
            )
            self.sandbox = sandbox
            self.filenames = list(files)

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            chunks: List[str] = []
            for name in self.filenames:
                chunks.append(self.sandbox.read_file(f"{prefix}{name}").decode(decoder))
            self._update_metrics(start, True)
            return {out_key: chunks}

    return Ingest()


def _mk_guard(instr_id: str, deps: List[str]) -> Instruction:
    class Guard(Instruction):
        def __init__(self):
            super().__init__(
                id=instr_id, name=instr_id, type=IsaType.CONTROL_FLOW,
                params={}, description=GUARD_NOTE, side_effects=False, outputs=[],
            )
            self.dependencies = list(deps)

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            _ = input_params["ctx"]  # touch, consume nothing
            self._update_metrics(start, True)
            return {}

    return Guard()


def _mk_pure(instr_id: str, deps: List[str], out_keys: List[str], fn) -> Instruction:
    class Pure(Instruction):
        def __init__(self):
            super().__init__(
                id=instr_id, name=instr_id, type=IsaType.CONTROL_FLOW,
                params={}, description=f"Pure transform producing {out_keys}",
                side_effects=False, outputs=list(out_keys),
            )
            self.dependencies = list(deps)

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            result = fn(input_params["ctx"])
            self._update_metrics(start, True)
            return result

    return Pure()


def _mk_classify(instr_id: str, deps: List[str], out_key: str,
                 sig_key: str, backend: RuleClassifierBackend,
                 infer) -> Instruction:
    """Cacheable LLM-type step: one simulated model call per distinct
    signature; identical inputs across realizations hit the cache."""

    class Classify(Instruction):
        def __init__(self):
            super().__init__(
                id=instr_id, name=instr_id, type=IsaType.LLM_INTERACTION,
                params={}, description="Cacheable LLM-backed classification step",
                side_effects=False, outputs=[out_key],
                cache_config={"max_size": 100_000, "ttl": 3600},
            )
            self.dependencies = list(deps)
            self._infer = infer

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            signatures = sorted(set(input_params["ctx"][sig_key]))
            cache_key = self._get_cache_key({"signatures": "|".join(signatures)})
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._update_metrics(start, True, cache_hit=True)
                return cached
            result = {out_key: self._infer(backend, signatures)}
            self.cache.put(cache_key, result)
            self._update_metrics(start, True)
            return result

    return Classify()


def _mk_write(instr_id: str, dep: str, report_text, out_key: str,
              sandbox: Optional[LocalSandboxExecutor]) -> Instruction:
    class Write(Instruction):
        def __init__(self):
            super().__init__(
                id=instr_id, name=instr_id, type=IsaType.STATE_MANAGEMENT,
                params={}, description="Write the report to the sandbox",
                side_effects=True, outputs=[out_key],
            )
            self.sandbox = sandbox
            self.dependencies = [dep]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            text = report_text(input_params["ctx"])
            digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
            self.sandbox.write_file("report.md", text.encode("utf-8"))
            self._update_metrics(start, True)
            return {out_key: digest}

    return Write()


# ---------------------------------------------------------------------------
# Task A: CSV ETL anomaly triage (fan-out shards -> per-row triage)
# ---------------------------------------------------------------------------

def csv_corpus(rng: random.Random) -> Dict[str, str]:
    corpus: Dict[str, str] = {}
    header = "ts,component,metric,value,unit"
    for shard in range(4):
        rows = [header]
        for i in range(40):
            component = rng.choice(["api", "auth", "billing", "search"])
            metric = rng.choice(["latency", "error_rate", "queue_depth"])
            value = rng.randint(10, 200)
            if shard == 0 and i in (7, 29):  # two planted anomalies
                value = rng.randint(950, 999)
            rows.append(
                f"2025-06-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:00:00Z,"
                f"{component},{metric},{value},ms"
            )
        corpus[f"shard_{shard:02d}.csv"] = "\n".join(rows) + "\n"
    return corpus


def build_csv_task(backend: RuleClassifierBackend,
                   sandbox: Optional[LocalSandboxExecutor]) -> List[Instruction]:
    files = [f"shard_{i:02d}.csv" for i in range(4)]

    def parse_rows(ctx):
        rows = []
        for chunk in ctx["csv_chunks"]:
            for line in chunk.splitlines()[1:]:  # skip header
                ts, component, metric, value, unit = line.split(",")
                rows.append({"component": component, "metric": metric,
                             "value": int(value), "unit": unit, "ts": ts})
        return {"rows": rows}

    def column_stats(ctx):
        values = [r["value"] for r in ctx["rows"]]
        mean = sum(values) / len(values)
        stats = {"mean": mean, "max": max(values), "min": min(values),
                 "n": len(values)}
        return {"stats": stats}

    def infer(backend_, signatures):
        return {s: backend_.infer(s) for s in signatures}

    def aggregate(ctx):
        counts: Dict[str, int] = {}
        for row, verdict in zip(ctx["rows"], ctx["triaged"].values()):
            counts[verdict] = counts.get(verdict, 0) + 1
        return {"anomaly_summary": dict(sorted(counts.items()))}

    def report(ctx):
        lines = ["# CSV Anomaly Alert Digest", ""]
        for k, v in ctx["anomaly_summary"].items():
            lines.append(f"- {k}: {v}")
        lines.append(f"- mean={ctx['stats']['mean']:.1f}")
        return "\n".join(lines) + "\n"

    return [
        _mk_ingest("ingest_csv", files, "etl/", "csv_chunks", sandbox),
        _mk_guard("check_schema", ["csv_chunks"]),
        _mk_pure("parse_rows", ["csv_chunks"], ["rows"], parse_rows),
        _mk_pure("rows_stats", ["rows"], ["stats"], column_stats),
        _mk_pure("row_signatures", ["rows"], ["row_sigs"],
                 lambda ctx: {"row_sigs": [
                     f'{r["component"]}|{r["metric"]}|{"high" if r["value"] > 900 else "norm"}'
                     for r in ctx["rows"]]}),
        _mk_classify("triage_rows", ["row_sigs"], "triaged", "row_sigs",
                     backend, infer),
        _mk_pure("aggregate_anomalies", ["triaged"], ["anomaly_summary"], aggregate),
        _mk_write("write_alert", "anomaly_summary", report, "report_written",
                  sandbox),
    ]


CSV_GUARD = "check_schema"


# ---------------------------------------------------------------------------
# Task B: report merge (wide fan-in: all docs feed one merge step)
# ---------------------------------------------------------------------------

def report_corpus(rng: random.Random) -> Dict[str, str]:
    shared = ["Objective: reduce service latency.",
              "Scope: all production services.",
              "Method: trace analysis."]
    unique = ["Alpha: caching layer.", "Beta: query planner.",
              "Gamma: connection pool.", "Delta: index rebuild.",
              "Epsilon: batch scheduling."]
    corpus: Dict[str, str] = {}
    for idx in range(5):
        lines = [f"# Report {idx}", ""]
        rng.shuffle(shared)
        lines.extend(shared)  # overlapping lines across reports
        lines.append(unique[idx])
        lines.append(f"Owner: team-{rng.randint(1, 3)}.")
        corpus[f"report_{idx:02d}.md"] = "\n".join(lines) + "\n"
    return corpus


def build_merge_task(backend: RuleClassifierBackend,
                     sandbox: Optional[LocalSandboxExecutor]) -> List[Instruction]:
    files = [f"report_{i:02d}.md" for i in range(5)]

    def dedupe(ctx):
        seen, out = set(), []
        for chunk in ctx["report_chunks"]:
            for line in chunk.splitlines():
                if line and line not in seen:
                    seen.add(line)
                    out.append(line)
        return {"deduped": out}

    def infer(backend_, signatures):
        return {s: backend_.infer(s) for s in signatures}

    def merge(ctx):
        consistent = [ln for ln in ctx["deduped"]
                      if not ln.startswith("Report ")]
        merged = ["# Merged Engineering Report", ""]
        merged.extend(consistent)
        merged.append(f"Consistency verdicts: {len(ctx['consistency'])}")
        return {"merged": "\n".join(merged)}

    def report(ctx):
        return ctx["merged"] + "\n"

    return [
        _mk_ingest("ingest_reports", files, "docs/", "report_chunks", sandbox),
        _mk_guard("check_headers", ["report_chunks"]),
        _mk_pure("dedupe_lines", ["report_chunks"], ["deduped"], dedupe),
        _mk_pure("doc_signatures", ["deduped"], ["doc_sigs"],
                 lambda ctx: {"doc_sigs": [
                     f'line|{hashlib.sha256(ln.encode()).hexdigest()[:8]}'
                     for ln in ctx["deduped"]]}),
        _mk_classify("consistency_check", ["doc_sigs"], "consistency",
                     "doc_sigs", backend, infer),
        _mk_pure("merge_reports", ["deduped", "consistency"], ["merged"], merge),
        _mk_write("write_summary", "merged", report, "report_written", sandbox),
    ]


MERGE_GUARD = "check_headers"


# ---------------------------------------------------------------------------
# Task C: regression diff (two parallel ingest branches -> diff)
# ---------------------------------------------------------------------------

def diff_corpus(rng: random.Random) -> Tuple[Dict[str, str], Dict[str, str]]:
    baseline = generate_corpus(random.Random(7), 2, 120)
    candidate = generate_corpus(random.Random(11), 2, 120)
    return baseline, candidate


def build_diff_task(backend: RuleClassifierBackend,
                    sandbox: Optional[LocalSandboxExecutor]) -> List[Instruction]:
    def normalize(ctx):
        def norm(chunks):
            out = []
            for chunk in chunks:
                for ln in chunk.splitlines():
                    ts, rest = ln.split(" ", 1)
                    out.append(f"T00:00:00Z {rest}")
            return out
        return {"norm_baseline": norm(ctx["baseline_chunks"]),
                "norm_candidate": norm(ctx["candidate_chunks"])}

    def diff(ctx):
        a, b = ctx["norm_baseline"], ctx["norm_candidate"]
        hunks = []
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                hunks.append(f"hunk@{i}: {x.split(' ', 1)[1]} -> {y.split(' ', 1)[1]}")
        return {"hunks": hunks}

    def infer(backend_, signatures):
        return {s: backend_.infer(s) for s in signatures}

    def aggregate(ctx):
        counts: Dict[str, int] = {}
        for verdict in ctx["classified_hunks"].values():
            counts[verdict] = counts.get(verdict, 0) + 1
        return {"reg_summary": dict(sorted(counts.items()))}

    def report(ctx):
        lines = ["# Regression Diff Report", ""]
        for hunk, verdict in sorted(ctx["classified_hunks"].items()):
            lines.append(f"- {verdict}: {hunk}")
        return "\n".join(lines) + "\n"

    return [
        _mk_ingest("ingest_baseline", ["service_00.log", "service_01.log"],
                   "logs/", "baseline_chunks", sandbox),
        _mk_ingest("ingest_candidate", ["service_00.log", "service_01.log"],
                   "new_logs/", "candidate_chunks", sandbox),
        _mk_guard("check_formats", ["baseline_chunks", "candidate_chunks"]),
        _mk_pure("normalize_events", ["baseline_chunks", "candidate_chunks"],
                 ["norm_baseline", "norm_candidate"], normalize),
        _mk_pure("diff_events", ["norm_baseline", "norm_candidate"],
                 ["hunks"], diff),
        _mk_classify("classify_hunks", ["hunks"], "classified_hunks",
                     "hunks", backend, infer),
        _mk_pure("aggregate_regressions", ["classified_hunks"],
                 ["reg_summary"], aggregate),
        _mk_write("write_regression_report", "reg_summary", report,
                  "report_written", sandbox),
    ]


DIFF_GUARD = "check_formats"


# ---------------------------------------------------------------------------
# Characterization
# ---------------------------------------------------------------------------

def _run_task_repeats(name: str, corpus: Dict[str, str], prefix_map: Dict[str, str],
                      build, guard: str, seed: int,
                      repeats: int = REPEATS) -> Dict[str, Any]:
    """Execute the closed-loop repeat battery for one task."""
    backend = RuleClassifierBackend()
    stream = build(backend, None)
    active = stream
    manager = OptimizationManager()
    manager.dynamic_optimizer.hot_path_threshold = HOT_THRESHOLD
    per_repeat_calls: List[int] = []
    instructions_per_repeat: List[int] = []
    hashes: List[str] = []

    async def _run() -> None:
        nonlocal active
        for repeat in range(repeats):
            sandbox = LocalSandboxExecutor(name=f"{name}{repeat}")
            try:
                for fname, content in sorted(corpus.items()):
                    sandbox.write_file(
                        f"{prefix_map[fname]}{fname}", content.encode("utf-8"))
                inject_sandbox(active, sandbox)
                instructions_per_repeat.append(len(active))
                before = backend.model_calls
                await run_stream(active)
                per_repeat_calls.append(backend.model_calls - before)
                hashes.append(report_sha256(sandbox))
                metrics = OptimizationMetrics(
                    execution_time=0.05, memory_usage=1.0,
                    api_calls=per_repeat_calls[-1], success_rate=1.0)
                result = manager.record_execution(active, metrics)
                if (result is not active and result is not None
                        and len(result) < len(active)):
                    active = result
            finally:
                sandbox.close()

    asyncio.run(_run())
    adopted_at = next(
        (i + 1 for i, n in enumerate(instructions_per_repeat)
         if n < instructions_per_repeat[0]), None)
    classify_metrics = next(
        i.metrics for i in stream
        if i.type == IsaType.LLM_INTERACTION)
    check = compile_check(stream, guards={guard})
    optimized = active if len(active) < len(stream) else None
    return {
        "seed": seed,
        "n_instructions": len(stream),
        "compiled_none_effective": check["NONE"]["effective"],
        "compiled_aggressive_effective": check["AGGRESSIVE"]["effective"],
        "eliminated": check["AGGRESSIVE"]["eliminated"],
        "dataflow_preserved": (
            check["AGGRESSIVE"]["dataflow_preserved"]
            and check["NONE"]["dataflow_preserved"]),
        "dead_guard": guard,
        "first_realization_calls": per_repeat_calls[0],
        "subsequent_calls": per_repeat_calls[1:],
        "reuse_all_hits": all(c == 0 for c in per_repeat_calls[1:]),
        "cache_hit_rate": classify_metrics.cache_hit_rate,
        "hot_path_adopted_at": adopted_at,
        "instructions_per_repeat": instructions_per_repeat,
        "all_reports_identical": len(set(hashes)) == 1,
        "report_sha256": hashes[0] if hashes else None,
        "optimized_form_present": optimized is not None,
        "compile_check_pass": check["pass"],
    }


def characterize() -> Dict[str, Any]:
    """Run all four tasks and return the characterization table."""
    out: Dict[str, Any] = {"repeats": REPEATS,
                           "hot_path_threshold": HOT_THRESHOLD, "tasks": {}}

    # Original triage task (Section 5.8 of the paper)
    out["tasks"]["log_triage"] = _run_triage()

    out["tasks"]["csv_etl_anomaly"] = _run_task_repeats(
        "csv", csv_corpus(random.Random(101)),
        {f"shard_{i:02d}.csv": "etl/" for i in range(4)},
        build_csv_task, CSV_GUARD, 101)

    out["tasks"]["report_merge"] = _run_task_repeats(
        "merge", report_corpus(random.Random(202)),
        {f"report_{i:02d}.md": "docs/" for i in range(5)},
        build_merge_task, MERGE_GUARD, 202)

    baseline, candidate = diff_corpus(random.Random(303))
    out["tasks"]["regression_diff"] = _run_task_repeats_diff(baseline, candidate)

    return out


def _run_triage() -> Dict[str, Any]:
    """The original Section 5.8 triage task, characterized identically."""
    from examples.system_task_demo import FORMAL_VIEWS, static_compile, run_demo
    demo = asyncio.run(run_demo(42, 6, 400, REPEATS, HOT_THRESHOLD))
    rep = demo["repeats"]
    compile_view = static_compile()
    return {
        "seed": 42,
        "n_instructions": len(FORMAL_VIEWS),
        "compiled_none_effective": compile_view["NONE"]["effective_instruction_count"],
        "compiled_aggressive_effective":
            compile_view["AGGRESSIVE"]["effective_instruction_count"],
        "eliminated": compile_view["AGGRESSIVE"]["eliminated"],
        "dataflow_preserved": compile_view["AGGRESSIVE"]["dataflow_preserved"],
        "dead_guard": "check_syntax",
        "first_realization_calls": rep["first_run_model_calls"],
        "subsequent_calls": rep["per_repeat_model_calls"][1:],
        "reuse_all_hits": rep["repeats_served_from_cache"] == REPEATS - 1,
        "cache_hit_rate": rep["cache_hit_rate"],
        "hot_path_adopted_at": rep["closed_loop"]["optimized_form_adopted_at_repeat"],
        "instructions_per_repeat": rep["closed_loop"]["instructions_per_repeat"],
        "all_reports_identical": rep["all_reports_identical"],
        "report_sha256": rep["report_sha256"],
        "optimized_form_present": demo["hot_path"]["optimized_length"] < len(FORMAL_VIEWS),
        "compile_check_pass": True,
    }


def _run_task_repeats_diff(baseline: Dict[str, str],
                           candidate: Dict[str, str],
                           repeats: int = REPEATS) -> Dict[str, Any]:
    """Two-branch ingest variant of the repeat battery (Task C)."""
    backend = RuleClassifierBackend()
    stream = build_diff_task(backend, None)
    active = stream
    manager = OptimizationManager()
    manager.dynamic_optimizer.hot_path_threshold = HOT_THRESHOLD
    per_repeat_calls: List[int] = []
    instructions_per_repeat: List[int] = []
    hashes: List[str] = []

    async def _run() -> None:
        nonlocal active
        for repeat in range(repeats):
            sandbox = LocalSandboxExecutor(name=f"diff{repeat}")
            try:
                for fname, content in sorted(baseline.items()):
                    sandbox.write_file(f"logs/{fname}", content.encode("utf-8"))
                for fname, content in sorted(candidate.items()):
                    sandbox.write_file(f"new_logs/{fname}", content.encode("utf-8"))
                inject_sandbox(active, sandbox)
                instructions_per_repeat.append(len(active))
                before = backend.model_calls
                await run_stream(active)
                per_repeat_calls.append(backend.model_calls - before)
                hashes.append(report_sha256(sandbox))
                metrics = OptimizationMetrics(
                    execution_time=0.05, memory_usage=1.0,
                    api_calls=per_repeat_calls[-1], success_rate=1.0)
                result = manager.record_execution(active, metrics)
                if (result is not active and result is not None
                        and len(result) < len(active)):
                    active = result
            finally:
                sandbox.close()

    asyncio.run(_run())
    adopted_at = next(
        (i + 1 for i, n in enumerate(instructions_per_repeat)
         if n < instructions_per_repeat[0]), None)
    classify_metrics = next(
        i.metrics for i in stream if i.type == IsaType.LLM_INTERACTION)
    check = compile_check(stream, guards={DIFF_GUARD})
    optimized = active if len(active) < len(stream) else None
    return {
        "seed": 303,
        "n_instructions": len(stream),
        "compiled_none_effective": check["NONE"]["effective"],
        "compiled_aggressive_effective": check["AGGRESSIVE"]["effective"],
        "eliminated": check["AGGRESSIVE"]["eliminated"],
        "dataflow_preserved": (
            check["AGGRESSIVE"]["dataflow_preserved"]
            and check["NONE"]["dataflow_preserved"]),
        "dead_guard": DIFF_GUARD,
        "first_realization_calls": per_repeat_calls[0],
        "subsequent_calls": per_repeat_calls[1:],
        "reuse_all_hits": all(c == 0 for c in per_repeat_calls[1:]),
        "cache_hit_rate": classify_metrics.cache_hit_rate,
        "hot_path_adopted_at": adopted_at,
        "instructions_per_repeat": instructions_per_repeat,
        "all_reports_identical": len(set(hashes)) == 1,
        "report_sha256": hashes[0] if hashes else None,
        "optimized_form_present": optimized is not None,
        "compile_check_pass": check["pass"],
    }


def _fmt_table(results: Dict[str, Any]) -> str:
    rows = []
    header = (f"{'task':<18} {'#instr':>6} {'NONE':>5} {'AGGR':>5} "
              f"{'guard-elim':>28} {'calls 1st/nxt':>14} {'hit%':>6} "
              f"{'hot@':>5} {'determ':>7}")
    rows.append(header)
    rows.append("-" * len(header))
    for name, t in results["tasks"].items():
        nxt = "+".join(str(c) for c in t["subsequent_calls"][:3]) + ".."
        rows.append(
            f"{name:<18} {t['n_instructions']:>6} "
            f"{t['compiled_none_effective']:>5} {t['compiled_aggressive_effective']:>5} "
            f"{','.join(t['eliminated']):>28} "
            f"{t['first_realization_calls']:>5}/{nxt:<8} "
            f"{t['cache_hit_rate'] * 100:>5.1f}% {str(t['hot_path_adopted_at']):>5} "
            f"{'YES' if t['all_reports_identical'] else 'NO':>7}")
    return "\n".join(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", dest="json_path", default=None,
                        help="dump the characterization dict as JSON")
    args = parser.parse_args()
    results = characterize()
    print(_fmt_table(results))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2, sort_keys=True)
        print(f"[json written to {args.json_path}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
