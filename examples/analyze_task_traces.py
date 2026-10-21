"""Workload characterization for the AgentISA task-trace corpus.

A standalone (stdlib-only) analyzer for the JSONL traces produced by
``random_tasks_stress.py`` (``examples/random_task_traces/*.jsonl``).
Its purpose is to characterize the *real* generated workload -- the
actual instruction mix, repetition, and graph structure the pipeline
sees -- so workload-level claims (Zipfian instruction mix, reuse
potential, fusion opportunity) rest on measured corpus statistics
rather than on synthetic construction.

Each trace file is a stream of JSON records; the first ``kind ==
"workflow"`` record carries the step list ({id, name, type,
description, dependencies, outputs, side_effects, step}). Message
records (per-step execution provenance) are not needed here and are
skipped by the loader.

Four analyses, one report:

1. Zipf fit -- instruction *signatures* are (type, params-signature)
   pairs; the signature frequency distribution is fitted with
   least squares on (log rank, log frequency), yielding the Zipf
   exponent ``s`` and R^2.
2. Reuse potential -- the fraction of step instances whose (type,
   params) signature was already seen, i.e. the corpus share that is
   a repeat occurrence and thus a candidate for memoization /
   cross-task cache reuse.
3. Dependency-shape census -- each workflow's step DAG (deps resolve
   to producers via output names or step ids) is classified as
   linear / fan-out / fan-in / branch (both fan-out and fan-in,
   e.g. a diamond), plus dead-end steps (pure steps whose outputs
   no step consumes).
4. Fusable-pair opportunities -- adjacent step pairs (i, i+1) that
   pass the compiler's fusion gate (``agentflow/core/isa/
   compiler.py::_are_fusible``): same type, both side-effect-free,
   and no producer-consumer edge inside the pair.

Usage:
    ./venv/bin/python examples/analyze_task_traces.py \
        [--trace-dir DIR] [--output report.json] [--top N]
"""

import argparse
import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

DEFAULT_TRACE_DIR = Path(__file__).resolve().parent / "random_task_traces"

SHAPE_KEYS = ("linear", "fan-out", "fan-in", "branch")

# Trailing "_<digits>" segments are positional indices the stress
# generator appends to disambiguate repeated operations
# ("filter_warn_5", "guard_0_2"); strip them to recover the
# operation identity.
_INDEX_SUFFIX = re.compile(r"(?:_\d+)+$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_workflow(path: Path) -> Optional[Dict[str, Any]]:
    """Return the first ``kind == "workflow"`` record of a JSONL trace.

    Malformed JSON raises (fail loudly rather than silently
    under-counting a corrupted corpus); blank lines are skipped.
    """
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if isinstance(record, dict) and record.get("kind") == "workflow":
                return record
    return None


# ---------------------------------------------------------------------------
# Signatures
# ---------------------------------------------------------------------------

def params_signature(step: Mapping[str, Any]) -> str:
    """Canonical params signature of a workflow step.

    Traces do not serialize instruction ``params``; when a ``params``
    field is present (and non-empty) it is canonicalized with sorted
    keys. Otherwise the operation identity is recovered from the step
    name by stripping trailing positional index segments.
    """
    params = step.get("params")
    if params:
        return "params=" + json.dumps(params, sort_keys=True, default=str)
    name = step.get("name") or step.get("id") or ""
    return _INDEX_SUFFIX.sub("", str(name))


def instruction_signature(step: Mapping[str, Any]) -> str:
    """Full instruction signature: ``"<TYPE>:<params signature>"``."""
    return f"{step.get('type')}:{params_signature(step)}"


# ---------------------------------------------------------------------------
# (1) Zipf fit
# ---------------------------------------------------------------------------

def fit_zipf(counts: Mapping[str, int]) -> Dict[str, Any]:
    """Least-squares Zipf fit of a frequency distribution.

    Regresses log(frequency) on log(rank) over the categories sorted
    by descending frequency; the negated slope is the Zipf exponent
    ``s`` (frequency ~ rank**-s) and ``r2`` is the coefficient of
    determination. Degenerate cases: fewer than two categories cannot
    be fitted (``s``/``r2`` are None); a zero-variance (uniform)
    distribution reports ``r2 = 0.0`` because the regression has no
    variance to explain.
    """
    freqs = sorted((c for c in counts.values() if c > 0), reverse=True)
    n = len(freqs)
    out: Dict[str, Any] = {
        "n_categories": n, "total": sum(freqs), "s": None, "r2": None,
    }
    if n < 2:
        return out
    xs = [math.log(r) for r in range(1, n + 1)]
    ys = [math.log(f) for f in freqs]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((y - (intercept + slope * x)) ** 2
                 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    out["s"] = -slope + 0.0  # normalizes -0.0 to 0.0
    out["r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return out


# ---------------------------------------------------------------------------
# (2) Reuse potential
# ---------------------------------------------------------------------------

def reuse_potential(workflows: Sequence[Mapping[str, Any]],
                    top: int = 5) -> Dict[str, Any]:
    """How much of the corpus repeats an already-seen signature.

    ``reuse_ratio`` is the corpus-level share of step instances that
    are repeat occurrences of a (type, params) signature (instances
    beyond each signature's first appearance);
    ``per_workflow_mean_reuse_ratio`` is the mean of the per-workflow
    repeat shares.
    """
    total = 0
    distinct: set = set()
    per_workflow: List[float] = []
    counts: Counter = Counter()
    for wf in workflows:
        steps = wf.get("steps") or []
        sigs = [instruction_signature(s) for s in steps]
        counts.update(sigs)
        total += len(sigs)
        distinct.update(sigs)
        if sigs:
            per_workflow.append((len(sigs) - len(set(sigs))) / len(sigs))
    repeat = total - len(distinct)
    return {
        "total_instances": total,
        "distinct_signatures": len(distinct),
        "repeat_instances": repeat,
        "reuse_ratio": (repeat / total) if total else 0.0,
        "per_workflow_mean_reuse_ratio": (
            sum(per_workflow) / len(per_workflow)) if per_workflow else 0.0,
        "top": [[sig, n] for sig, n in counts.most_common(top)],
    }


# ---------------------------------------------------------------------------
# (3) Dependency shapes
# ---------------------------------------------------------------------------

def _graph_stats(steps: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Resolve a step list into graph facts.

    Dependencies resolve to producers by output name first, then by
    step id; unresolvable dependencies are external inputs (no
    edge). Returns per-step in-degrees (distinct resolved
    dependencies), consumer sets (distinct consumer step ids), the
    set of demanded names, and the weakly-connected component count.
    """
    ids = [s.get("id") for s in steps]
    by_id = {s.get("id"): s for s in steps}
    producer: Dict[str, Any] = {}
    for s in steps:
        for out in s.get("outputs") or []:
            producer[out] = s.get("id")
    in_deg = {sid: 0 for sid in ids}
    consumers: Dict[Any, set] = {sid: set() for sid in ids}
    adj: Dict[Any, set] = {sid: set() for sid in ids}
    demands: set = set()
    for s in steps:
        sid = s.get("id")
        resolved = set()
        for dep in s.get("dependencies") or []:
            demands.add(dep)
            pid = producer.get(dep)
            if pid is None and dep in by_id:
                pid = dep
            if pid is None or pid == sid:
                continue  # external input, or degenerate self-edge
            resolved.add(dep)
            consumers[pid].add(sid)
            adj[pid].add(sid)
            adj[sid].add(pid)
        in_deg[sid] = len(resolved)
    # weakly-connected components (isolated steps count as one each)
    seen: set = set()
    n_components = 0
    for sid in ids:
        if sid in seen:
            continue
        n_components += 1
        stack = [sid]
        seen.add(sid)
        while stack:
            for nxt in adj[stack.pop()]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return {"in_deg": in_deg, "consumers": consumers,
            "demands": demands, "n_components": n_components}


def _shape_from_stats(stats: Mapping[str, Any]) -> str:
    fan_out = any(len(c) >= 2 for c in stats["consumers"].values())
    fan_in = any(d >= 2 for d in stats["in_deg"].values())
    if fan_out and fan_in:
        return "branch"
    if fan_out:
        return "fan-out"
    if fan_in:
        return "fan-in"
    return "linear"


def classify_workflow_shape(steps: Sequence[Mapping[str, Any]]) -> str:
    """Classify a workflow's dependency DAG as one of the census keys.

    ``branch`` means both a fan-out and a fan-in node exist (split
    and rejoin, e.g. a diamond); a workflow with neither is
    ``linear`` (no branching at all).
    """
    return _shape_from_stats(_graph_stats(steps))


def _dead_end_steps(steps: Sequence[Mapping[str, Any]],
                    stats: Mapping[str, Any]) -> int:
    """Pure steps whose outputs no step consumes (single-pass, not a
    fixpoint liveness analysis -- the AGGRESSIVE pass may eliminate
    more)."""
    return sum(
        1 for s in steps
        if s.get("side_effects") is False
        and not (set(s.get("outputs") or []) & stats["demands"])
    )


def _shape_census(workflows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = dict.fromkeys(SHAPE_KEYS, 0)
    dead_end = 0
    disconnected = 0
    n_steps = 0
    for wf in workflows:
        steps = wf.get("steps") or []
        n_steps += len(steps)
        stats = _graph_stats(steps)
        counts[_shape_from_stats(stats)] += 1
        dead_end += _dead_end_steps(steps, stats)
        if len(steps) > 1 and stats["n_components"] >= 2:
            disconnected += 1
    return {
        "counts": counts,
        "dead_end_steps": dead_end,
        "workflows_with_multiple_components": disconnected,
        "avg_steps": (n_steps / len(workflows)) if workflows else 0.0,
    }


# ---------------------------------------------------------------------------
# (4) Adjacent fusable pairs
# ---------------------------------------------------------------------------

def _fusable_pair(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    """The compiler's fusion gate, mirrored on trace dicts (see
    ``compiler.py::_are_fusible``): same type, both explicitly pure
    (a missing ``side_effects`` counts as effectful), and ``b`` does
    not consume any of ``a``'s outputs."""
    return (
        a.get("type") == b.get("type")
        and a.get("side_effects") is False
        and b.get("side_effects") is False
        and not (set(b.get("dependencies") or [])
                 & set(a.get("outputs") or []))
    )


def fusion_opportunity(
        workflows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Adjacent (stream-order) step pairs passing the fusion gate."""
    total = 0
    fusable = 0
    by_type: Counter = Counter()
    for wf in workflows:
        steps = wf.get("steps") or []
        for a, b in zip(steps, steps[1:]):
            total += 1
            if _fusable_pair(a, b):
                fusable += 1
                by_type[a.get("type")] += 1
    return {
        "total_adjacent_pairs": total,
        "fusable_pairs": fusable,
        "fusable_ratio": (fusable / total) if total else 0.0,
        "by_type": dict(by_type),
    }


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def analyze(trace_dir: Path | str, top: int = 5) -> Dict[str, Any]:
    """Characterize every ``*.jsonl`` trace under ``trace_dir``.

    The report carries a ``provenance`` section (scanned files,
    workflow count, and a sha256 over the sorted file names+bytes) so
    any published number can be traced back to the exact corpus.
    """
    trace_dir = Path(trace_dir)
    files = sorted(p for p in trace_dir.glob("*.jsonl") if p.is_file())
    workflows: List[Dict[str, Any]] = []
    for path in files:
        workflow = load_workflow(path)
        if workflow is not None:
            workflows.append(workflow)

    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())

    steps = [s for wf in workflows for s in wf.get("steps") or []]
    sig_counts = Counter(instruction_signature(s) for s in steps)
    zipf = fit_zipf(sig_counts)
    zipf["top"] = [[sig, n] for sig, n in sig_counts.most_common(top)]

    return {
        "provenance": {
            "trace_dir": str(trace_dir.resolve()),
            "n_traces": len(workflows),
            "files": [p.name for p in files],
            "corpus_sha256": digest.hexdigest(),
            "n_steps": len(steps),
        },
        "zipf": zipf,
        "reuse": reuse_potential(workflows, top=top),
        "shapes": _shape_census(workflows),
        "fusion": fusion_opportunity(workflows),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _fmt(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _print_summary(report: Mapping[str, Any]) -> None:
    prov = report["provenance"]
    print("AgentISA trace-workload characterization")
    print(f"  corpus: {prov['n_traces']} workflows / "
          f"{prov['n_steps']} steps from {len(prov['files'])} files "
          f"(sha256 {prov['corpus_sha256'][:12]})")

    z = report["zipf"]
    print("\n1. Instruction-signature Zipf fit (log-rank/log-freq OLS)")
    print(f"   categories: {z['n_categories']}, "
          f"s = {_fmt(z['s'])}, R^2 = {_fmt(z['r2'])}")
    print("   top: " + ", ".join(f"{sig}={n}" for sig, n in z["top"]))

    r = report["reuse"]
    print("\n2. (type, params) reuse potential")
    print(f"   {r['total_instances']} instances, "
          f"{r['distinct_signatures']} distinct -> "
          f"{r['reuse_ratio']:.1%} repeat ratio")
    print(f"   per-workflow mean reuse ratio: "
          f"{r['per_workflow_mean_reuse_ratio']:.1%}")

    s = report["shapes"]
    c = s["counts"]
    print("\n3. Dependency-shape census")
    print(f"   linear={c['linear']} fan-out={c['fan-out']} "
          f"fan-in={c['fan-in']} branch={c['branch']} "
          f"(avg {s['avg_steps']:.1f} steps/workflow)")
    print(f"   dead-end (pure, unconsumed) steps: "
          f"{s['dead_end_steps']}; disconnected workflows: "
          f"{s['workflows_with_multiple_components']}")

    f = report["fusion"]
    print("\n4. Adjacent fusable pairs "
          "(same type, pure, no producer-consumer edge)")
    print(f"   {f['fusable_pairs']}/{f['total_adjacent_pairs']} "
          f"({f['fusable_ratio']:.1%})")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Characterize the AgentISA task-trace workload: "
                    "Zipf fit, reuse potential, dependency shapes, "
                    "fusion opportunities.")
    parser.add_argument(
        "--trace-dir", default=str(DEFAULT_TRACE_DIR),
        help=f"directory of *.jsonl traces (default: {DEFAULT_TRACE_DIR})")
    parser.add_argument(
        "--output", default=None,
        help="also write the JSON report to this path")
    parser.add_argument(
        "--top", type=int, default=5,
        help="number of top signatures to list (default: 5)")
    args = parser.parse_args(argv)

    report = analyze(args.trace_dir, top=args.top)
    _print_summary(report)

    if args.output:
        Path(args.output).write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nJSON report written to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
