"""Real-trace transfer pilot: does the Section 5.3 reduction survive real agent traces?

Motivation. Section 5.3's headline (34.2% effective-instruction reduction) is
measured on linear sequences whose optimization targets are planted at disclosed
rates (0.25 dead / 0.55 pure). Section I.4.2's non-planted-family control
(19.2%, all dead-guard absorption) narrows it but still uses generator-produced
traces. This script prepares the missing third family: trajectories recorded
from real agent runs, converted to instruction streams under a disclosed,
conservative mapping.

The expected outcome is deliberately not known in advance -- that is the point
of the experiment. What the mapping guarantees is that any reduction reported
here is *earned* under the compiler's own gates, never planted:

  * Effect classification. Unknown tools are treated as effectful (never
    removable, never fusable). Only reads/searches and LLM calls are pure, and
    only they declare outputs.
  * Dataflow. A dependency edge is declared whenever a step's arguments
    reference an earlier step's declared output; when in doubt, we declare it
    (fusion requires PROVEN disjointness, so doubt must resolve to "dependent").
  * Dead code. A step is DCE-removable only if it has no outputs and no side
    effects -- the compiler's own gate, not a heuristic of ours. Real
    trajectories are expected to yield ~0 here; that is a finding, not a bug.

Input adapters (JSONL: one record per line, key = record id):

  swe-agent   {"id": "...", "actions": [{"tool": "...", "args": {...}}, ...]}
  tau-bench   {"id": "...", "trajectory": [{"tool_calls": [{"name": "...",
               "arguments": {...}}]}, ...]}

Both are lossy on purpose: only the ordered (tool, arguments) pairs are used;
model prose is ignored. An adapter is ~20 lines, so adding a corpus family is a
small change.

Measurements per stream, and in aggregate:
  * effective-instruction count at NONE / BASIC / AGGRESSIVE (the shipped
    InstructionCompiler, same counter as Section 5.3);
  * dispatch count (len of the compiled top level; a parallel group = 1);
  * result-cache hits over the trajectory under two keys: the canonical
    (params-only, sorted) key and the raw insertion-order key -- the same
    canonicalization contrast as I.4.1, now on real traces;
  * dead / fusible-pair counts under the compiler's actual gates.

Self-test. ``--self-test`` synthesizes three realistic-shaped trajectories
(coding-agent loop, multi-tool support turn, RAG pipeline) and asserts the
pipeline invariants. ITS NUMBERS ARE NOT RESULTS: they are written to a
self-test path and tagged ``"self_test": true``. Never cite them in the paper.

Stdlib + agentflow only. Usage:
    ./venv/bin/python examples/benchmarks/real_trace_pilot.py --self-test
    ./venv/bin/python examples/benchmarks/real_trace_pilot.py \
        --corpus data/swe_gym.jsonl --adapter swe-agent --output /tmp/pilot.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentflow.core.isa.compiler import (  # noqa: E402
    CompilationContext,
    InstructionCompiler,
)
from agentflow.core.isa.formal import FormalInstruction, InstructionType  # noqa: E402
from agentflow.core.isa.instruction_cache import build_cache_key  # noqa: E402

SCHEMA = "real_trace_pilot/v1"
LEVELS = ["NONE", "BASIC", "AGGRESSIVE"]

# Tool classification. Conservative: anything not listed is EFFECTFUL.
# value = (InstructionType, pure?, declares_output?, output_tag)
TOOL_CLASSES: Dict[str, tuple] = {
    # effectful writers / state changers
    "bash": (InstructionType.STATE, False, False, None),
    "shell": (InstructionType.STATE, False, False, None),
    "terminal": (InstructionType.STATE, False, False, None),
    "str_replace_editor": (InstructionType.STATE, False, False, None),
    "edit": (InstructionType.STATE, False, False, None),
    "write": (InstructionType.STATE, False, False, None),
    "create": (InstructionType.STATE, False, False, None),
    "issue_refund": (InstructionType.STATE, False, False, None),
    "send_email": (InstructionType.STATE, False, False, None),
    "create_task": (InstructionType.STATE, False, False, None),
    "book_reservation": (InstructionType.STATE, False, False, None),
    # pure readers / searchers
    "read_file": (InstructionType.STATE, True, True, "file"),
    "view": (InstructionType.STATE, True, True, "file"),
    "cat": (InstructionType.STATE, True, True, "file"),
    "open": (InstructionType.STATE, True, True, "file"),
    "grep": (InstructionType.STATE, True, True, "search"),
    "search": (InstructionType.STATE, True, True, "search"),
    "find": (InstructionType.STATE, True, True, "search"),
    "get_order": (InstructionType.STATE, True, True, "record"),
    "get_user": (InstructionType.STATE, True, True, "record"),
    "get_product": (InstructionType.STATE, True, True, "record"),
    "retrieve": (InstructionType.STATE, True, True, "record"),
    "embed": (InstructionType.LLM, True, True, "vector"),
    # LLM calls
    "llm": (InstructionType.LLM, True, True, "llm"),
    "chat": (InstructionType.LLM, True, True, "llm"),
    "classify": (InstructionType.LLM, True, True, "llm"),
    "summarize": (InstructionType.LLM, True, True, "llm"),
    "answer": (InstructionType.LLM, True, True, "llm"),
}

EFFECTFUL_DEFAULT = (InstructionType.STATE, False, False, None)


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Adapters: trajectory record -> ordered [(tool, args), ...]
# --------------------------------------------------------------------------
def load_records(corpus: Path, adapter: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def actions_swe_agent(record: Mapping[str, Any]) -> List[List[tuple]]:
    """SWE-agent actions are strictly sequential: one action per turn."""
    return [[(a.get("tool", "unknown"), dict(a.get("args") or {}))]
            for a in record.get("actions", [])]


def actions_tau_bench(record: Mapping[str, Any]) -> List[List[tuple]]:
    """Tool calls sharing an assistant message are issued in parallel: one
    turn per trajectory entry, one call per parallel slot."""
    turns: List[List[tuple]] = []
    for turn in record.get("trajectory", []):
        calls = [(c.get("name", "unknown"), dict(c.get("arguments") or {}))
                 for c in (turn.get("tool_calls") or [])]
        if calls:
            turns.append(calls)
    return turns


ADAPTERS = {"swe-agent": actions_swe_agent, "tau-bench": actions_tau_bench}


# --------------------------------------------------------------------------
# Conservative mapping: actions -> workflow steps (the compiler's input shape)
# --------------------------------------------------------------------------
def _args_text(args: Mapping[str, Any]) -> str:
    try:
        return json.dumps(args, sort_keys=True, default=str)
    except Exception:
        return str(args)


def to_steps(turns: Sequence[Sequence[tuple]]) -> List[Dict[str, Any]]:
    """Map ordered turns of (tool, args) calls to steps.

    Dependency discipline (conservative, and the reason the numbers can be
    read at all). The compiler's fusion gate compares a step's *dependencies*
    against the *outputs* of its predecessor, so the two must share one
    namespace: dependencies are the output tokens this step consumes, never
    internal ids.

      * same turn  -> calls are issued in parallel by construction: no edge;
      * later turn -> an edge to the live handle (the most recent output
        token) unless the arguments name a different earlier output, in which
        case the edge points there. Doubt resolves to "dependent": a step is
        never declared independent merely because its arguments do not spell
        out the handle.
    """
    steps: List[Dict[str, Any]] = []
    handle: Optional[str] = None
    for t, turn in enumerate(turns):
        for s_idx, (tool, args) in enumerate(turn):
            itype, pure, declares, tag = TOOL_CLASSES.get(tool, EFFECTFUL_DEFAULT)
            outputs: List[str] = []
            if pure and declares and tag is not None:
                target = None
                for key in ("path", "file", "filepath", "filename", "id", "query"):
                    if key in args:
                        target = str(args[key])
                        break
                outputs = [f"{tag}:{target}" if target else f"{tag}:{t}.{s_idx}"]
            text = _args_text(args)
            deps: List[str] = []
            # explicit reference to an earlier output wins
            for prev in steps:
                for out in prev["outputs"]:
                    bare = out.split(":", 1)[1] if ":" in out else out
                    if bare and bare in text:
                        deps.append(out)
                        break
            if not deps:
                # same-turn siblings are parallel; later turns chain to the handle
                same_turn = s_idx > 0
                if not same_turn and handle is not None:
                    deps.append(handle)
            steps.append({
                "id": f"s{t}.{s_idx}",
                "name": tool,
                "type": itype.name if hasattr(itype, "name") else str(itype),
                "dependencies": deps,
                "outputs": outputs,
                "side_effects": not pure,
            })
            if outputs and pure:
                handle = outputs[0]
    return steps


def to_formal(step: Mapping[str, Any]) -> FormalInstruction:
    type_map = {"STATE": InstructionType.STATE,
                "CONTROL": InstructionType.CONTROL,
                "LLM": InstructionType.LLM}
    return FormalInstruction(
        name=step["name"],
        type=type_map.get(step["type"], InstructionType.STATE),
        dependencies=list(step.get("dependencies", [])),
        metadata={
            "side_effects": bool(step.get("side_effects", False)),
            "outputs": list(step.get("outputs", [])),
        },
    )


# --------------------------------------------------------------------------
# Measurements (same counters as Section 5.3 / trace_compile_control)
# --------------------------------------------------------------------------
def effective_count(instructions: Iterable[Any]) -> int:
    total = 0
    for item in instructions:
        if isinstance(item, list):
            total += effective_count(item)
        else:
            total += 1
    return total


def dispatch_count(instructions: Sequence[Any]) -> int:
    return len(instructions)


def dead_steps(steps: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for s in steps if not s["outputs"] and not s["side_effects"])


def fusible_pairs_compiler_gate(steps: Sequence[Mapping[str, Any]]) -> int:
    n = 0
    for a, b in zip(steps, steps[1:]):
        if (a["type"] == b["type"] and not a["side_effects"]
                and not b["side_effects"]
                and not (set(b["dependencies"]) & set(a["outputs"]))):
            n += 1
    return n


def cache_measure(actions: Sequence[tuple]) -> Dict[str, Any]:
    """Hit counts under the canonical (params-only, sorted) key and the raw
    insertion-order key, over one trajectory's ordered calls."""
    seen_canon: set = set()
    seen_raw: set = set()
    hits_canon = hits_raw = 0
    for tool, args in actions:
        canon = build_cache_key(tool, dict(args))
        raw = hashlib.sha256(
            f"{tool}|{_args_text_unsorted(args)}".encode("utf-8")
        ).hexdigest()
        if canon in seen_canon:
            hits_canon += 1
        seen_canon.add(canon)
        if raw in seen_raw:
            hits_raw += 1
        seen_raw.add(raw)
    n = len(actions)
    return {
        "calls": n,
        "canonical_hits": hits_canon,
        "canonical_hit_rate": round(hits_canon / n, 4) if n else 0.0,
        "raw_order_hits": hits_raw,
        "raw_order_hit_rate": round(hits_raw / n, 4) if n else 0.0,
    }


def _args_text_unsorted(args: Mapping[str, Any]) -> str:
    return json.dumps(dict(args), default=str)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
def run(records: Sequence[Mapping[str, Any]], adapter: str,
        corpus_label: str, self_test: bool) -> Dict[str, Any]:
    fn = ADAPTERS[adapter]
    compiler = InstructionCompiler({})
    per_stream: List[Dict[str, Any]] = []
    totals = {lvl: 0 for lvl in LEVELS}
    dispatches = {lvl: 0 for lvl in LEVELS}
    total_steps = total_dead = total_fusible = 0
    hits_c = hits_r = calls = 0

    for rec in records:
        turns = fn(rec)
        if not turns:
            continue
        steps = to_steps(turns)
        seq = [to_formal(s) for s in steps]
        cm = cache_measure([c for turn in turns for c in turn])
        row: Dict[str, Any] = {
            "stream": rec.get("id") or rec.get("name") or f"stream{len(per_stream)}",
            "steps": len(steps),
            "dead_steps": dead_steps(steps),
            "fusible_pairs_compiler_gate": fusible_pairs_compiler_gate(steps),
            "cache": cm,
        }
        # Protect the trajectory's final artifact: the terminal turn's declared
        # outputs are required, so backward liveness cannot delete the answer
        # (real traces have no synthetic sink; Section 5.1.2's planted family
        # gets this for free from its final-sink slot).
        required: List[str] = [
            out for st in steps
            if st["id"].startswith(f"s{len(turns) - 1}.")
            for out in st["outputs"]
        ]
        for lvl in LEVELS:
            ctx = CompilationContext({
                "optimization_level": lvl,
                "target_metrics": {"required_outputs": required},
            })
            out = compiler.compile(seq, ctx)
            row[f"effective_{lvl}"] = effective_count(out)
            row[f"dispatch_{lvl}"] = dispatch_count(out)
            totals[lvl] += row[f"effective_{lvl}"]
            dispatches[lvl] += row[f"dispatch_{lvl}"]
        per_stream.append(row)
        total_steps += len(steps)
        total_dead += row["dead_steps"]
        total_fusible += row["fusible_pairs_compiler_gate"]
        hits_c += cm["canonical_hits"]
        hits_r += cm["raw_order_hits"]
        calls += cm["calls"]

    return {
        "experiment": "real_trace_pilot",
        "schema": SCHEMA,
        "self_test": self_test,
        "description": (
            "Static-compilation and result-cache measurements over real agent "
            "trajectories, converted under a conservative disclosed mapping "
            "(unknown tools effectful; doubt resolves to a dataflow edge). "
            "Reduction is earned under the compiler's own gates, never planted."
        ),
        "provenance": {
            "git_rev": git_rev(),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "corpus": corpus_label,
            "adapter": adapter,
            "streams": len(per_stream),
        },
        "aggregate": {
            "total_steps": total_steps,
            "total_dead_steps": total_dead,
            "total_fusible_pairs_compiler_gate": total_fusible,
            "effective": totals,
            "dispatch": dispatches,
            "effective_reduction_vs_none": {
                lvl: round(1.0 - totals[lvl] / total_steps, 4)
                for lvl in LEVELS if lvl != "NONE" and total_steps
            },
            "dispatch_reduction_vs_none": {
                lvl: round(1.0 - dispatches[lvl] / dispatches["NONE"], 4)
                for lvl in LEVELS
                if lvl != "NONE" and dispatches["NONE"]
            },
            "cache": {
                "calls": calls,
                "canonical_hits": hits_c,
                "canonical_hit_rate": round(hits_c / calls, 4) if calls else 0.0,
                "raw_order_hits": hits_r,
                "raw_order_hit_rate": round(hits_r / calls, 4) if calls else 0.0,
            },
        },
        "per_stream": per_stream,
    }


# --------------------------------------------------------------------------
# Self-test: realistic-shaped trajectories (NOT results)
# --------------------------------------------------------------------------
def self_test_records() -> List[Dict[str, Any]]:
    coding = {"id": "selftest-coding-loop", "actions": [
        {"tool": "read_file", "args": {"path": "src/app.py"}},
        {"tool": "grep", "args": {"pattern": "def main"}},
        {"tool": "bash", "args": {"cmd": "pytest -q tests/"}},
        {"tool": "bash", "args": {"cmd": "pytest -q tests/"}},
        {"tool": "read_file", "args": {"path": "src/app.py"}},
        {"tool": "edit", "args": {"path": "src/app.py", "old": "a", "new": "b"}},
        {"tool": "bash", "args": {"cmd": "pytest -q tests/"}},
        {"tool": "bash", "args": {"cmd": "git diff"}},
        {"tool": "read_file", "args": {"path": "src/util.py"}},
        {"tool": "summarize", "args": {"text": "diff summary"}},
    ]}
    support = {"id": "selftest-support-turn", "trajectory": [{"tool_calls": [
        {"name": "get_order", "arguments": {"id": "A100"}},
        {"name": "get_user", "arguments": {"id": "U7"}},
    ]}, {"tool_calls": [
        {"name": "issue_refund", "arguments": {"order": "A100", "amount": 20}},
        {"name": "send_email", "arguments": {"to": "U7", "body": "refund issued"}},
    ]}, {"tool_calls": [
        {"name": "get_order", "arguments": {"id": "A100"}},
    ]}]}
    rag = {"id": "selftest-rag", "actions": [
        {"tool": "embed", "args": {"text": "question"}},
        {"tool": "search", "args": {"query": "policy refund"}},
        {"tool": "read_file", "args": {"path": "docs/policy.md"}},
        {"tool": "summarize", "args": {"text": "policy excerpt"}},
        {"tool": "answer", "args": {"text": "final answer"}},
        {"tool": "write", "args": {"path": "out/report.md"}},
    ]}
    return [coding, support, rag]


def self_test_cases():
    recs = self_test_records()
    return [([recs[0], recs[2]], "swe-agent"), ([recs[1]], "tau-bench")]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, help="JSONL of trajectory records")
    parser.add_argument("--adapter", choices=sorted(ADAPTERS), default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true",
                        help="run the synthetic-shaped pipeline self-test")
    args = parser.parse_args(argv)

    if args.self_test:
        out = args.output or Path("/tmp/real_trace_pilot_selftest.json")
        reports = []
        print("SELF-TEST (not results):")
        for records, adapter in self_test_cases():
            rep = run(records, adapter, f"self-test ({adapter})", True)
            reports.append(rep)
            agg = rep["aggregate"]
            assert agg["total_dead_steps"] == 0, "self-test expects no dead steps"
            assert agg["cache"]["canonical_hits"] > 0, "repeated calls must hit"
            assert agg["cache"]["canonical_hits"] >= agg["cache"]["raw_order_hits"]
            assert agg["effective"]["NONE"] == agg["total_steps"]
            # The terminal-answer protection must hold: a pure final LLM/answer
            # step may not be deleted as dead.
            last_stream = rep["per_stream"][-1]
            assert last_stream["effective_BASIC"] >= last_stream["steps"] - 1, (
                "terminal artifact was eliminated as dead")
            print(f"  [{adapter}] streams={rep['provenance']['streams']} "
                  f"steps={agg['total_steps']} dead={agg['total_dead_steps']} "
                  f"fusible={agg['total_fusible_pairs_compiler_gate']} "
                  f"eff.reduction={agg['effective_reduction_vs_none']} "
                  f"cache={agg['cache']['canonical_hit_rate']:.1%}")
        out.write_text(json.dumps(reports, indent=2), encoding="utf-8")
        print(f"-> {out}")
        return 0

    if not args.corpus or not args.adapter:
        parser.error("--corpus and --adapter are required unless --self-test")
    records = load_records(args.corpus, args.adapter)
    report = run(records, args.adapter,
                 str(args.corpus.relative_to(REPO_ROOT)), False)
    out = args.output or (REPO_ROOT / "examples" / "benchmarks" /
                          "real_trace_pilot_results.json")
    if out.name.endswith("_results.json") and report["provenance"]["streams"] < 20:
        print(f"WARNING: {report['provenance']['streams']} streams is pilot scale; "
              "do not present it as a population result.", file=sys.stderr)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    agg = report["aggregate"]
    print(f"streams: {report['provenance']['streams']} steps: {agg['total_steps']}")
    print(f"effective reduction: {agg['effective_reduction_vs_none']}")
    print(f"dispatch  reduction: {agg['dispatch_reduction_vs_none']}")
    print(f"cache: {agg['cache']['canonical_hits']}/{agg['cache']['calls']} "
          f"({agg['cache']['canonical_hit_rate']:.1%}) canonical, "
          f"{agg['cache']['raw_order_hit_rate']:.1%} raw-order")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
