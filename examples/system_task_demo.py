#!/usr/bin/env python3
"""Controllable system-level task realized as an AgentISA instruction stream.

Task: *service-log triage*.  An agent instruction stream ingests seeded
service logs from a sandboxed workspace (real file I/O), extracts and
classifies incident signatures (a deterministic rule-based classifier
stands in for the LLM, mirroring the mock-backend protocol of the
microbenchmarks), aggregates the triage, and writes a markdown report
back into the workspace.

The demonstration exercises, end to end on a real system-level task:

1. **Static compilation** — the task's instruction stream is compiled
   by ``InstructionCompiler`` at NONE vs AGGRESSIVE; we report the
   effective-instruction and dispatch counts and *assert the compiled
   stream preserves the task's dataflow* (every step whose output is
   consumed downstream survives; only the dead guard is eliminated).
2. **Execution** — the executable stream runs against a
   ``LocalSandboxExecutor`` workspace; the classifier backend counts
   its (simulated) model calls.
3. **Repeated execution** — the *same* task is realized R times; the
   result cache serves the classification step from cache, so model
   calls drop to zero after the first realization, and every repeat
   must produce a byte-identical report.
4. **Closed-loop dynamic optimization** — the repeated stream goes
   hot; the runtime *adopts* the optimized instruction stream, and all
   later repeats automatically execute the shrunken form (one dead
   guard instruction eliminated).  The optimized form, executed on a
   fresh workspace, must reproduce the report byte-for-byte: the
   optimization is semantics-preserving on the real task.
5. **Determinism** — a fully fresh stack (new instructions, new cache,
   new workspace, same seed) reproduces the report byte-for-byte.

Controllability: the log corpus is generated from a fixed seed; every
threshold is a CLI flag; no wall-clock value enters the report, so the
byte-identity checks are meaningful.

Usage:
    ./venv/bin/python examples/system_task_demo.py \
        --seed 42 --files 6 --lines-per-file 400 --repeats 12
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
from typing import Any, Dict, List, Optional

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentflow.core.isa.compiler import (
    CompilationContext,
    InstructionCompiler,
)
from agentflow.core.isa.formal import FormalInstruction
from agentflow.core.isa.formal import InstructionType as FormalType
from agentflow.core.isa.instruction import Instruction
from agentflow.core.isa.instruction import InstructionType as IsaType
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager,
    OptimizationMetrics,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor


# --------------------------------------------------------------------------
# Seeded, fully synthetic service-log corpus
# --------------------------------------------------------------------------

COMPONENTS = ["auth", "db", "api", "cache", "worker"]
LEVELS = ["INFO", "WARN", "ERROR"]
MESSAGES = [
    "request completed",
    "latency above threshold",
    "connection reset by peer",
    "token refresh failed",
    "disk usage high",
    "timeout waiting for lock",
    "upstream 5xx",
    "schema mismatch on write",
    "cache miss storm",
    "worker heartbeat lost",
]


def generate_corpus(rng: random.Random, files: int, lines_per_file: int
                    ) -> Dict[str, str]:
    """Generate a deterministic log corpus: filename -> content."""
    corpus: Dict[str, str] = {}
    for f in range(files):
        out = []
        for _ in range(lines_per_file):
            ts = (f"2025-06-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:"
                  f"{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z")
            out.append(
                f"{ts} level={rng.choice(LEVELS)} component={rng.choice(COMPONENTS)} "
                f'msg="{rng.choice(MESSAGES)}" req={rng.randint(1000, 9999)}'
            )
        corpus[f"service_{f:02d}.log"] = "\n".join(out) + "\n"
    return corpus


def write_corpus(sandbox: LocalSandboxExecutor, corpus: Dict[str, str]) -> None:
    for name, content in sorted(corpus.items()):
        sandbox.write_file(f"logs/{name}", content.encode("utf-8"))


# --------------------------------------------------------------------------
# Deterministic classifier backend (stands in for the LLM)
# --------------------------------------------------------------------------

class RuleClassifierBackend:
    """Counts every simulated model call, like ``MockLLMBackend``.

    The classification rule is a pure function of the signature (hash
    bucketing), so it is deterministic across runs and stands in for
    the model exactly the way the paper's mock backend does for the
    microbenchmarks: the *cost accounting* is what matters, not the
    intelligence of the classifier.
    """

    SEVERITIES = ["low", "medium", "high", "critical"]

    def __init__(self) -> None:
        self.model_calls = 0

    def infer(self, signature: str) -> str:
        self.model_calls += 1
        # Deterministic severity: weighted by message severity keywords.
        base = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16)
        if "timeout" in signature or "5xx" in signature or "heartbeat" in signature:
            base += 2
        elif "failed" in signature or "mismatch" in signature or "reset" in signature:
            base += 1
        return self.SEVERITIES[base % len(self.SEVERITIES)]


# --------------------------------------------------------------------------
# The instruction stream of the system-level task
# --------------------------------------------------------------------------

def build_stream(backend: RuleClassifierBackend,
                 log_files: List[str],
                 sandbox: Optional[LocalSandboxExecutor] = None) -> List[Instruction]:
    """Build the service-log-triage instruction stream.

    The stream mirrors how an authored agent program looks in
    AgentISA: system-touching steps (ingest, report) are marked
    side-effectful; pure transforms declare their outputs and
    dependencies; the LLM-backed step is cacheable.  ``check_syntax``
    is a defensive guard whose result nothing downstream consumes --
    the kind of stream bloat a real agent program accumulates, and
    exactly what dead-code elimination is for.
    """

    class IngestLogs(Instruction):
        def __init__(self, filenames: List[str]):
            super().__init__(
                id="ingest_logs", name="ingest_logs", type=IsaType.STATE_MANAGEMENT,
                params={}, description="Read all service logs from the workspace",
                side_effects=True,
            )
            self.sandbox = sandbox
            self.filenames = list(filenames)

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            lines: List[str] = []
            for name in self.filenames:
                lines.extend(
                    self.sandbox.read_file(f"logs/{name}").decode("utf-8").splitlines()
                )
            self._update_metrics(start, True)
            return {"raw_lines": lines}

    class CheckSyntax(Instruction):
        """Defensive format guard; result consumed by nothing (dead)."""

        def __init__(self):
            super().__init__(
                id="check_syntax", name="check_syntax", type=IsaType.CONTROL_FLOW,
                params={}, description="Validate log line syntax (guard)",
                side_effects=False, outputs=[],
            )
            self.dependencies = ["ingest_logs"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            bad = sum(
                1 for ln in input_params["ctx"]["raw_lines"]
                if "level=" not in ln
            )
            self._update_metrics(start, True)
            return {"malformed_lines": bad}

    class NormalizeEntries(Instruction):
        def __init__(self):
            super().__init__(
                id="normalize_entries", name="normalize_entries", type=IsaType.CONTROL_FLOW,
                params={}, description="Parse log lines into structured entries",
                side_effects=False, outputs=["entries"],
            )
            self.dependencies = ["ingest_logs"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            entries = []
            for ln in input_params["ctx"]["raw_lines"]:
                parts = dict(p.split("=", 1) for p in ln.split(" ") if "=" in p)
                msg = parts.get("msg", '"?"').strip('"')
                entries.append({
                    "level": parts.get("level", "INFO"),
                    "component": parts.get("component", "unknown"),
                    "signature": f'{parts.get("component", "unknown")}|{msg}',
                })
            self._update_metrics(start, True)
            return {"entries": entries}

    class ExtractIncidents(Instruction):
        def __init__(self):
            super().__init__(
                id="extract_incidents", name="extract_incidents", type=IsaType.CONTROL_FLOW,
                params={}, description="Extract WARN/ERROR incidents",
                side_effects=False, outputs=["incidents"],
            )
            self.dependencies = ["normalize_entries"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            incidents = [
                e for e in input_params["ctx"]["entries"]
                if e["level"] in ("WARN", "ERROR")
            ]
            self._update_metrics(start, True)
            return {"incidents": incidents}

    class ClassifySignature(Instruction):
        """Cacheable LLM-backed step: one model call per distinct signature."""

        def __init__(self):
            super().__init__(
                id="classify_signature", name="classify_signature",
                type=IsaType.LLM_INTERACTION,
                params={}, description="Classify each incident signature",
                side_effects=False, outputs=["classified"],
                cache_config={"max_size": 100_000, "ttl": 3600},
            )
            self.dependencies = ["extract_incidents"]
            self.backend = backend

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            signatures = sorted({i["signature"] for i in input_params["ctx"]["incidents"]})
            cache_key = self._get_cache_key({"signatures": "|".join(signatures)})
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._update_metrics(start, True, cache_hit=True)
                return cached
            classification = {s: self.backend.infer(s) for s in signatures}
            self.cache.put(cache_key, {"classified": classification})
            self._update_metrics(start, True)
            return {"classified": classification}

    class AggregateTriage(Instruction):
        def __init__(self):
            super().__init__(
                id="aggregate_triage", name="aggregate_triage",
                type=IsaType.CONTROL_FLOW,
                params={}, description="Aggregate incident statistics",
                side_effects=False, outputs=["summary"],
            )
            self.dependencies = ["classify_signature"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            ctx = input_params["ctx"]
            by_severity: Dict[str, int] = {}
            by_component: Dict[str, int] = {}
            by_signature: Dict[str, int] = {}
            for incident in ctx["incidents"]:
                sig = incident["signature"]
                sev = ctx["classified"][sig]
                by_severity[sev] = by_severity.get(sev, 0) + 1
                comp = sig.split("|", 1)[0]
                by_component[comp] = by_component.get(comp, 0) + 1
                by_signature[sig] = by_signature.get(sig, 0) + 1
            top = sorted(by_signature.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            self._update_metrics(start, True)
            return {"summary": {
                "total_incidents": len(ctx["incidents"]),
                "by_severity": by_severity,
                "by_component": by_component,
                "top_signatures": top,
            }}

    class WriteReport(Instruction):
        def __init__(self):
            super().__init__(
                id="write_report", name="write_report", type=IsaType.STATE_MANAGEMENT,
                params={}, description="Write the triage report to the workspace",
                side_effects=True,
            )
            self.dependencies = ["aggregate_triage"]
            self.sandbox = sandbox

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()  # matches Instruction._update_metrics clock
            s = input_params["ctx"]["summary"]
            lines = [
                "# Service Log Triage Report",
                "",
                f"- incidents: {s['total_incidents']}",
                "- by severity: "
                + ", ".join(f"{k}={v}" for k, v in sorted(s["by_severity"].items())),
                "- by component: "
                + ", ".join(f"{k}={v}" for k, v in sorted(s["by_component"].items())),
                "- top signatures:",
            ]
            lines += [f"  - {sig} ({count})" for sig, count in s["top_signatures"]]
            report = "\n".join(lines) + "\n"
            self.sandbox.write_file("report.md", report.encode("utf-8"))
            self._update_metrics(start, True)
            return {"report_written": True}

    return [
        IngestLogs(log_files), CheckSyntax(), NormalizeEntries(), ExtractIncidents(),
        ClassifySignature(), AggregateTriage(), WriteReport(),
    ]


def inject_sandbox(instructions: List[Instruction],
                   sandbox: LocalSandboxExecutor) -> None:
    """Point every sandbox-touching instruction at the given workspace.

    Walks composite instructions so a merged trace (hot-path output)
    is handled uniformly.
    """
    for instr in instructions:
        if hasattr(instr, "components"):  # CompositeInstruction
            inject_sandbox(instr.components, sandbox)
        if hasattr(instr, "sandbox"):
            instr.sandbox = sandbox


async def run_stream(stream: List[Instruction]) -> str:
    """Execute an instruction stream sequentially; return report sha256."""
    ctx: Dict[str, Any] = {}
    for instr in stream:
        result = await instr.execute({"ctx": ctx})
        if isinstance(result, dict):
            ctx.update(result)
    return ctx["report_written"]


def report_sha256(sandbox: LocalSandboxExecutor) -> str:
    return hashlib.sha256(sandbox.read_file("report.md")).hexdigest()


# --------------------------------------------------------------------------
# Static compilation view of the same task
# --------------------------------------------------------------------------

# (name, formal type, side effects, outputs, dependencies).
# Dependencies follow the compiler's convention: they reference the
# *output names* consumed (or instruction ids), which is how the DCE
# liveness analysis tracks producers (``any(out in needed)``).
FORMAL_VIEWS = [
    ("ingest_logs", FormalType.STATE, True, ["raw_lines"], []),
    ("check_syntax", FormalType.CONTROL, False, [], ["raw_lines"]),
    ("normalize_entries", FormalType.CONTROL, False, ["entries"], ["raw_lines"]),
    ("extract_incidents", FormalType.CONTROL, False, ["incidents"], ["entries"]),
    ("classify_signature", FormalType.LLM, False, ["classified"], ["incidents"]),
    ("aggregate_triage", FormalType.CONTROL, False, ["summary"], ["classified"]),
    ("write_report", FormalType.STATE, True, [], ["summary"]),
]


def effective_instruction_count(instructions: List[Any]) -> int:
    """Count instructions, unwrapping parallel-block wrappers.

    Mirrors the microbenchmark's counting semantics so the numbers are
    directly comparable (a wrapper counts as one dispatch, its children
    count as effective instructions).
    """
    total = 0
    for item in instructions:
        if isinstance(item, list):
            total += effective_instruction_count(item)
            continue
        inner = (item.metadata.get("parallel_instructions")
                 if hasattr(item, "metadata") else None)
        if inner:
            total += effective_instruction_count(inner)
        else:
            total += 1
    return total


def _covered_names(compiled: List[Any]) -> set:
    """Names of original steps still present in a compiled stream.

    Unwraps parallel-block wrappers; fused instructions carry all
    component names inside their own name (``fused_a_b``), so matching
    is by substring.
    """
    names: set = set()
    for item in compiled:
        if isinstance(item, list):
            names |= _covered_names(item)
            continue
        inner = (item.metadata.get("parallel_instructions")
                 if hasattr(item, "metadata") else None)
        if inner:
            names |= _covered_names(inner)
        else:
            names.add(getattr(item, "name", str(item)))
    return names


def static_compile() -> Dict[str, Any]:
    """Compile the task's formal views and validate the result.

    Beyond counting, we assert the compiled stream preserves the task's
    dataflow: every step whose output is consumed downstream must
    survive compilation (possibly fused), and the only eliminable step
    is the dead guard.  This guards against the subtle failure where a
    dependency-naming mismatch makes DCE drop live producers.
    """
    compiler = InstructionCompiler({})
    out: Dict[str, Any] = {"stream_length": len(FORMAL_VIEWS)}
    for level in ("NONE", "AGGRESSIVE"):
        seq = [
            FormalInstruction(
                name=name, type=type_,
                metadata={"side_effects": side_effects, "outputs": outputs},
                dependencies=list(deps),
            )
            for name, type_, side_effects, outputs, deps in FORMAL_VIEWS
        ]
        compiled = compiler.compile(seq, CompilationContext({"optimization_level": level}))
        covered = _covered_names(compiled)
        all_names = {v[0] for v in FORMAL_VIEWS}
        covered_producers = {
            n for n in all_names if any(n in c for c in covered)
        }
        eliminated = sorted(all_names - covered_producers)
        # Dataflow validity: every consumed output's producer must
        # survive compilation (possibly fused), and the only eliminable
        # step is the dead guard.
        producer_of = {
            output: name
            for name, _, _, outputs, _ in FORMAL_VIEWS
            for output in outputs
        }
        consumed = {dep for _, _, _, _, deps in FORMAL_VIEWS for dep in deps}
        missing = sorted({
            producer_of[d] for d in consumed
            if d in producer_of and producer_of[d] not in covered_producers
        })
        out[level] = {
            "effective_instruction_count": effective_instruction_count(compiled),
            "dispatch_count": len(compiled),
            "eliminated": eliminated,
            "missing_producers": missing,
            "dataflow_preserved": (
                not missing and set(eliminated) <= {"check_syntax"}
            ),
        }
    out["effective_reduction"] = (
        1.0 - out["AGGRESSIVE"]["effective_instruction_count"]
        / out["NONE"]["effective_instruction_count"]
    )
    return out


# --------------------------------------------------------------------------
# Demonstration phases
# --------------------------------------------------------------------------

async def run_demo(seed: int, files: int, lines_per_file: int,
                   repeats: int, hot_threshold: int) -> Dict[str, Any]:
    results: Dict[str, Any] = {
        "task": "service-log-triage",
        "config": {
            "seed": seed, "files": files, "lines_per_file": lines_per_file,
            "repeats": repeats, "hot_path_threshold": hot_threshold,
        },
        "static_compile": static_compile(),
    }
    corpus = generate_corpus(random.Random(seed), files, lines_per_file)

    # --- Phase 1..N: repeated realization of the same system-level task.
    # This is a *closed loop*: once the hot-path threshold is crossed,
    # the runtime adopts the optimized instruction stream, and later
    # repeats execute the shrunken form automatically.
    backend = RuleClassifierBackend()
    log_files = sorted(corpus)
    stream = build_stream(backend, log_files)  # sandbox injected per repeat
    active_stream = stream
    manager = OptimizationManager()
    manager.dynamic_optimizer.hot_path_threshold = hot_threshold

    per_repeat_calls: List[int] = []
    instructions_per_repeat: List[int] = []
    repeat_hashes: List[str] = []

    for repeat in range(repeats):
        sandbox = LocalSandboxExecutor(name=f"demo{repeat}")
        try:
            write_corpus(sandbox, corpus)
            inject_sandbox(active_stream, sandbox)
            instructions_per_repeat.append(len(active_stream))
            calls_before = backend.model_calls
            await run_stream(active_stream)
            per_repeat_calls.append(backend.model_calls - calls_before)
            repeat_hashes.append(report_sha256(sandbox))

            metrics = OptimizationMetrics(
                execution_time=0.05, memory_usage=1.0,
                api_calls=per_repeat_calls[-1], success_rate=1.0,
            )
            result = manager.record_execution(active_stream, metrics)
            if (result is not active_stream and result is not None
                    and len(result) < len(active_stream)):
                active_stream = result  # adopt the optimized form
        finally:
            sandbox.close()

    adopted_at = next(
        (i + 1 for i, n in enumerate(instructions_per_repeat)
         if n < instructions_per_repeat[0]),
        None,
    )
    total_executions = sum(i.metrics.total_executions for i in stream)
    cache_hits = sum(i.metrics.cache_hits for i in stream)
    classify_metrics = next(
        i.metrics for i in stream if i.id == "classify_signature")
    results["repeats"] = {
        "per_repeat_model_calls": per_repeat_calls,
        "first_run_model_calls": per_repeat_calls[0],
        "repeats_served_from_cache": sum(1 for c in per_repeat_calls[1:] if c == 0),
        "cache_hit_rate": classify_metrics.cache_hit_rate,
        "total_instruction_executions": total_executions,
        "all_reports_identical": len(set(repeat_hashes)) == 1,
        "report_sha256": repeat_hashes[0] if repeat_hashes else None,
        "closed_loop": {
            "instructions_per_repeat": instructions_per_repeat,
            "optimized_form_adopted_at_repeat": adopted_at,
            "optimized_stream_executed_repeats": (
                repeats - adopted_at + 1 if adopted_at else 0),
        },
    }

    # --- Semantics check: the adopted (optimized) stream, executed on a
    # fresh workspace, must reproduce the report byte-for-byte.
    optimized = active_stream if len(active_stream) < len(stream) else None
    if optimized is None:
        optimized = manager.dynamic_optimizer.optimize_hot_path(
            manager._compute_path_id(stream))
    hot: Dict[str, Any] = {
        "original_length": len(stream),
        "optimized_length": len(optimized) if optimized else len(stream),
    }
    if optimized:
        hot["shrink_ratio"] = 1.0 - len(optimized) / len(stream)
        hot["eliminated"] = sorted(
            {i.id for i in stream} - {getattr(i, "id", None) for i in optimized})
        sandbox = LocalSandboxExecutor(name="hotpath")
        try:
            write_corpus(sandbox, corpus)
            inject_sandbox(optimized, sandbox)
            await run_stream(optimized)
            hot["optimized_report_matches"] = (
                report_sha256(sandbox) == repeat_hashes[0])
        finally:
            sandbox.close()
    else:
        hot["shrink_ratio"] = 0.0
        hot["optimized_report_matches"] = None
    results["hot_path"] = hot

    # --- Determinism: fully fresh stack, same seed.
    backend2 = RuleClassifierBackend()
    stream2 = build_stream(backend2, log_files)
    sandbox = LocalSandboxExecutor(name="determinism")
    try:
        write_corpus(sandbox, corpus)
        inject_sandbox(stream2, sandbox)
        await run_stream(stream2)
        determinism_hash = report_sha256(sandbox)
    finally:
        sandbox.close()
    results["determinism"] = {
        "run1_sha256": repeat_hashes[0],
        "run2_sha256": determinism_hash,
        "byte_identical": determinism_hash == repeat_hashes[0],
    }

    results["verdict"] = {
        "deterministic": results["determinism"]["byte_identical"],
        "repeatable": results["repeats"]["all_reports_identical"],
        "cache_effective": (per_repeat_calls[0] > 0
                            and all(c == 0 for c in per_repeat_calls[1:])),
        "static_compile_valid": results["static_compile"]["AGGRESSIVE"]["dataflow_preserved"],
        "closed_loop_adopted": adopted_at is not None,
        "hot_path_semantics_preserved": hot.get("optimized_report_matches"),
    }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--files", type=int, default=6)
    parser.add_argument("--lines-per-file", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--hot-threshold", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="optional path for the results JSON")
    args = parser.parse_args()

    results = asyncio.run(run_demo(
        seed=args.seed, files=args.files, lines_per_file=args.lines_per_file,
        repeats=args.repeats, hot_threshold=args.hot_threshold,
    ))

    ok = all(results["verdict"].values())
    print("=" * 64)
    print("AgentISA system-level task demonstration: service-log triage")
    print("=" * 64)
    sc = results["static_compile"]
    print(f"stream length            : {sc['stream_length']}")
    print(f"static compile NONE      : effective={sc['NONE']['effective_instruction_count']}"
          f" dispatch={sc['NONE']['dispatch_count']}")
    print(f"static compile AGGRESSIVE: effective={sc['AGGRESSIVE']['effective_instruction_count']}"
          f" dispatch={sc['AGGRESSIVE']['dispatch_count']}"
          f" (reduction {sc['effective_reduction']:.1%},"
          f" eliminated={sc['AGGRESSIVE']['eliminated']},"
          f" dataflow preserved={sc['AGGRESSIVE']['dataflow_preserved']})")
    rep = results["repeats"]
    print(f"model calls, run 1       : {rep['first_run_model_calls']}")
    print(f"model calls, runs 2..{args.repeats}: "
          f"{rep['per_repeat_model_calls'][1:]}")
    print(f"classifier cache hit rate: {rep['cache_hit_rate']:.1%}")
    cl = rep["closed_loop"]
    print(f"instructions per repeat  : {cl['instructions_per_repeat']}")
    print(f"optimized form adopted   : repeat {cl['optimized_form_adopted_at_repeat']}"
          f" ({cl['optimized_stream_executed_repeats']} repeats on the shrunken form)")
    print(f"all repeats identical    : {rep['all_reports_identical']}")
    hot = results["hot_path"]
    print(f"trace shrink             : {hot['original_length']} -> "
          f"{hot['optimized_length']} instructions"
          f" (eliminated: {hot.get('eliminated')})")
    print(f"optimized report matches : {hot['optimized_report_matches']}")
    det = results["determinism"]
    print(f"determinism              : {det['byte_identical']} ({det['run1_sha256'][:12]}..)")
    print("=" * 64)
    print("VERDICT:", "PASS" if ok else "FAIL", "-", results["verdict"])

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2) + "\n")
        print(f"results written to {args.output}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
