"""Non-planted control: compile the generated-trace corpus itself.

Review-motivated control for the static-compilation headline (Section
5.3): 34.2% is measured on linear sequences whose optimization targets
are planted at disclosed rates (0.25 dead / 0.55 pure).  This control
runs the SAME shipped compiler (``InstructionCompiler``) at the SAME
levels over the Section 5.10 corpus -- 100 generator-produced
workflows, 792 steps, graph-shaped dependency structure -- where any
dead steps and fusible pairs are planted by a DIFFERENT generator at
its own rates, and the step mix is not chosen to favor the passes.

The corpus is itself generator-produced (the Section 5.10 census's
disclosure applies): this experiment varies the workload family --
graph shapes, different planting rates, the production engine's own
task generator -- it does not remove planting altogether, and the
paper says so.

Discipline: every number in the output is computed here from the
committed corpus; nothing is hand-filled.  The compile is a
deterministic full census of the 100 workflows (no sampling
randomness, so no seeds apply); provenance (git rev, UTC timestamp,
corpus path) is recorded automatically.

Stdlib + agentflow only.  Usage:
    ./venv/bin/python examples/benchmarks/trace_compile_control.py \
        [--trace-dir DIR] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentflow.core.isa.compiler import (  # noqa: E402
    CompilationContext,
    InstructionCompiler,
)
from agentflow.core.isa.formal import (  # noqa: E402
    FormalInstruction,
    InstructionType,
)

DEFAULT_TRACE_DIR = REPO_ROOT / "examples" / "random_task_traces"
DEFAULT_OUTPUT = (
    REPO_ROOT / "examples" / "benchmarks" / "trace_compile_control_results.json"
)
SCHEMA = "trace_compile_control/v1"

# Corpus step types -> formal instruction types (the same enum the
# Section 5.3 sequences use).
TYPE_MAP = {
    "STATE_MANAGEMENT": InstructionType.STATE,
    "CONTROL_FLOW": InstructionType.CONTROL,
    "LLM_INTERACTION": InstructionType.LLM,
}
LEVELS = ["NONE", "BASIC", "AGGRESSIVE"]


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def load_workflows(trace_dir: Path) -> List[Dict[str, Any]]:
    """Load the workflow records from the corpus, in name order."""
    workflows = []
    for path in sorted(trace_dir.glob("task*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "workflow":
                workflows.append(record)
    workflows.sort(key=lambda w: w.get("name", ""))
    return workflows


def to_formal(step: Mapping[str, Any]) -> FormalInstruction:
    """Map one corpus step to the Section 5.3 instruction encoding.

    The encoding matches the microbenchmark's ``synthetic_formal_sequence``
    convention (side_effects / outputs in metadata) plus the corpus's own
    declared dependency edges, which the compiler reads for fusion safety.
    """
    return FormalInstruction(
        name=step["name"],
        type=TYPE_MAP[step["type"]],
        dependencies=list(step.get("dependencies", [])),
        metadata={
            "side_effects": bool(step.get("side_effects", False)),
            "outputs": list(step.get("outputs", [])),
        },
    )


def dead_steps(steps: List[Mapping[str, Any]]) -> int:
    """Steps removable by construction: no outputs, no side effects."""
    return sum(
        1 for s in steps
        if not s.get("outputs") and not s.get("side_effects")
    )


def fusible_pairs_compiler_gate(steps: List[Mapping[str, Any]]) -> int:
    """Adjacent pairs passing the compiler's actual _are_fusible gate:
    same type, both side-effect-free, and NO dataflow from a into b
    (instr2's dependencies disjoint from instr1's outputs) -- the pass
    merges independent neighbors, not dataflow chains.
    """
    n = 0
    for a, b in zip(steps, steps[1:]):
        if (
            a["type"] == b["type"]
            and not a.get("side_effects")
            and not b.get("side_effects")
            and not (set(b.get("dependencies", []))
                     & set(a.get("outputs", [])))
        ):
            n += 1
    return n


def fusible_pairs_loose_gate(steps: List[Mapping[str, Any]]) -> int:
    """Adjacent same-type side-effect-free pairs (type/purity only)."""
    n = 0
    for a, b in zip(steps, steps[1:]):
        if (
            a["type"] == b["type"]
            and not a.get("side_effects")
            and not b.get("side_effects")
        ):
            n += 1
    return n


def fusible_pairs_both_live(steps: List[Mapping[str, Any]]) -> int:
    """Compiler-gate pairs whose members BOTH survive DCE (non-dead).

    Zero here means every fusion the gate permits involves a dead
    step: the pass can only absorb a dead guard, never merge two live
    neighbors (the corpus's pure transforms form dataflow chains,
    which the disjointness condition refuses -- correctly).
    """
    n = 0
    for a, b in zip(steps, steps[1:]):
        if (
            a["type"] == b["type"]
            and not a.get("side_effects")
            and not b.get("side_effects")
            and not (set(b.get("dependencies", []))
                     & set(a.get("outputs", [])))
            and (a.get("outputs") or a.get("side_effects"))
            and (b.get("outputs") or b.get("side_effects"))
        ):
            n += 1
    return n


def effective_count(instructions: List[Any]) -> int:
    """The microbenchmark's counter (parallel blocks unwrapped)."""
    total = 0
    for item in instructions:
        if isinstance(item, list):
            total += effective_count(item)
        else:
            total += 1
    return total


def run(trace_dir: Path) -> Dict[str, Any]:
    workflows = load_workflows(trace_dir)
    if not workflows:
        raise SystemExit(f"no workflow records under {trace_dir}")

    compiler = InstructionCompiler({})
    per_trace: List[Dict[str, Any]] = []
    totals = {lvl: 0 for lvl in LEVELS}
    fused_totals = {lvl: 0 for lvl in LEVELS}
    total_steps = 0
    total_dead = 0
    total_fusible = 0
    total_fusible_loose = 0
    total_fusible_both_live = 0

    for wf in workflows:
        steps = wf["steps"]
        seq = [to_formal(s) for s in steps]
        row: Dict[str, Any] = {
            "workflow": wf.get("name"),
            "steps": len(steps),
            "dead_steps": dead_steps(steps),
            "fusible_compiler_gate": fusible_pairs_compiler_gate(steps),
            "fusible_type_purity_only": fusible_pairs_loose_gate(steps),
            "fusible_both_live": fusible_pairs_both_live(steps),
        }
        for lvl in LEVELS:
            ctx = CompilationContext({"optimization_level": lvl})
            out = compiler.compile(seq, ctx)
            row[f"effective_{lvl}"] = effective_count(out)
            row[f"fused_{lvl}"] = sum(
                1 for item in out
                if not isinstance(item, list)
                and getattr(item, "name", "").startswith("fused_")
            )
            totals[lvl] += effective_count(out)
            fused_totals[lvl] += row[f"fused_{lvl}"]
        per_trace.append(row)
        total_steps += len(steps)
        total_dead += row["dead_steps"]
        total_fusible += row["fusible_compiler_gate"]
        total_fusible_loose += row["fusible_type_purity_only"]
        total_fusible_both_live += row["fusible_both_live"]

    report: Dict[str, Any] = {
        "experiment": "trace_compile_control",
        "schema": SCHEMA,
        "description": (
            "The shipped static compiler at Section 5.3's levels over the "
            "Section 5.10 generated-trace corpus (100 workflows, 792 "
            "steps): a non-planted-family control for the 34.2% "
            "planted-target headline. Corpus is generator-produced; "
            "dead guards and fusible pairs are the generator's own, at "
            "its own rates -- disclosed, not removed."
        ),
        "provenance": {
            "git_rev": git_rev(),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_dir": str(trace_dir.relative_to(REPO_ROOT)),
            "workflows": len(workflows),
            "deterministic_full_census": True,
        },
        "aggregate": {
            "total_steps": total_steps,
            "total_dead_steps": total_dead,
            "total_fusible_pairs_compiler_gate": total_fusible,
            "total_fusible_pairs_type_purity_only": total_fusible_loose,
            "total_fusible_pairs_both_live": total_fusible_both_live,
            "effective": totals,
            "fused_instructions": fused_totals,
            "reduction_vs_none": {
                lvl: round(1.0 - totals[lvl] / total_steps, 4)
                for lvl in LEVELS if lvl != "NONE"
            },
            "decomposition": {
                lvl: {
                    "fused": fused_totals[lvl],
                    "removed_total": total_steps - totals[lvl],
                }
                for lvl in LEVELS if lvl != "NONE"
            },
        },
        "per_workflow": per_trace,
    }
    return report


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = run(args.trace_dir)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    agg = report["aggregate"]
    print(f"workflows: {report['provenance']['workflows']}")
    print(f"steps: {agg['total_steps']} (dead {agg['total_dead_steps']}, "
          f"fusible compiler-gate {agg['total_fusible_pairs_compiler_gate']}, "
          f"type-purity-only {agg['total_fusible_pairs_type_purity_only']})")
    for lvl in LEVELS:
        print(f"  {lvl:>10}: effective {agg['effective'][lvl]}")
    for lvl, red in agg["reduction_vs_none"].items():
        print(f"  reduction {lvl}: {red:.1%}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
