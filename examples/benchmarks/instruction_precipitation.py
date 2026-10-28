"""Instruction precipitation: does a repertoire actually accumulate and pay off?

The value claim behind instructionization is not that one trajectory can be
optimized, but that instructions *sediment*: a library built from earlier tasks
serves later ones, so per-task work falls as experience grows. That claim has
been implemented (mining -> templates -> parameterized replay -> promotion
gate) but never measured on real traces. This script measures it.

What is measured, and what is deliberately not:

  * **Atomic reuse (earned).** A template is ``(tool, parameter keys)``: one
    instruction with a parameter schema, instantiated per call -- exactly the
    parameterized-replay path the test suite covers
    (``test_replay_parameterization``). A step counts as *served* only if its
    template was promoted into the repertoire strictly before this task, so a
    task never scores off instructions it just created.
  * **Promotion gate.** A template enters the repertoire only after
    ``min_support`` occurrences, mirroring ``AutoPromotionConfig``; until then
    it is a candidate, not a hit. Promotion is the *create* event.
  * **Deprecation (delete).** Templates unused for more than ``max_idle`` tasks
    are retired at task boundaries, mirroring ``deprecate_stale``; retirement
    is the *delete* event, and a retired template must be re-mined to return.
  * **Compound recurrence (upper bound, not earned).** Recurring k-grams of
    templates are reported as an upper bound on what compound instructions
    could cover. Instantiating a compound from the first call's arguments alone
    is not verifiable on inputs-only corpora -- the binding needs observation
    values -- so it is reported separately and never added to the earned
    number.

The payoff metric is *fresh work*: steps per task that the repertoire could not
serve, which must decline if precipitation is real.

Usage:
    ./venv/bin/python examples/benchmarks/instruction_precipitation.py \
        --corpus examples/real_traces/tau_bench_historical.jsonl \
        --param-mode schema \
        --output examples/benchmarks/instruction_precipitation_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA_VERSION = "instruction_precipitation/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

TemplateId = Tuple[str, Tuple[str, ...]]


# -- corpus ---------------------------------------------------------------


def load_tasks(path: Path, first_trial_only: bool = True) -> List[Dict[str, Any]]:
    """Recorded trajectories, one record per task, in a stable order.

    Trial replay would let the repertoire score off a task it was built from,
    so by default only the first trial of each ``(source, task_id)`` is kept.
    """
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    chosen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("source")), str(row.get("task_id")))
        if first_trial_only:
            if key in chosen:
                continue
            chosen[key] = row
        else:
            chosen[(key[0], f"{key[1]}#{row.get('trial')}")] = row
    return [chosen[key] for key in sorted(chosen)]


def calls_of(task: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    out: List[Tuple[str, Dict[str, Any]]] = []
    for turn in task.get("turns") or []:
        for call in turn:
            if isinstance(call, (list, tuple)) and len(call) == 2:
                out.append((str(call[0]), dict(call[1] or {})))
    return out


def template_of(tool: str, args: Dict[str, Any], param_mode: str) -> TemplateId:
    """The instruction a call instantiates: its tool plus parameter shape.

    ``schema`` mode keeps the argument names (a parameterized instruction);
    ``command`` mode keeps the leading command tokens, for shell-shaped calls
    whose payload is a command line rather than named parameters.
    """
    if param_mode == "command":
        head = tuple(str(args.get("cmd", "")).split()[:2])
        return (tool, head)
    return (tool, tuple(sorted(str(key) for key in args)))


def digest(tid: TemplateId) -> str:
    return hashlib.sha256(f"{tid[0]}|{'|'.join(tid[1])}".encode("utf-8")).hexdigest()[:12]


# -- repertoire -----------------------------------------------------------


@dataclass
class Entry:
    """One instruction in the repertoire, with the history that justifies it."""
    tid: TemplateId
    support: int = 0
    promoted_at: int = -1
    last_used: int = -1
    uses: int = 0
    retired_at: int = -1

    @property
    def active(self) -> bool:
        return self.promoted_at >= 0 and self.retired_at < 0


@dataclass
class Repertoire:
    """Support-gated, staleness-retiring instruction library."""
    min_support: int = 2
    max_idle: int = 40
    entries: Dict[TemplateId, Entry] = field(default_factory=dict)
    created: int = 0
    retired: int = 0
    remined: int = 0

    def _entry(self, tid: TemplateId) -> Entry:
        entry = self.entries.get(tid)
        if entry is None:
            entry = self.entries[tid] = Entry(tid=tid)
        return entry

    def observe(self, tid: TemplateId, task_index: int) -> str:
        """Record one occurrence; returns ``hit`` / ``promoted`` / ``candidate``.

        A hit requires an active promotion from an earlier task -- a template
        created during this task cannot serve this task.
        """
        entry = self._entry(tid)
        if entry.active:
            entry.uses += 1
            entry.last_used = task_index
            return "hit"
        if entry.retired_at >= 0:
            self.remined += 1
            entry.retired_at = -1
        entry.support += 1
        if entry.support >= self.min_support and entry.promoted_at < 0:
            entry.promoted_at = task_index
            entry.last_used = task_index
            self.created += 1
            return "promoted"
        return "candidate"

    def retire_stale(self, task_index: int) -> List[TemplateId]:
        """Deprecate active entries idle for more than ``max_idle`` tasks."""
        gone: List[TemplateId] = []
        for entry in self.entries.values():
            if not entry.active:
                continue
            if task_index - max(entry.last_used, entry.promoted_at) > self.max_idle:
                entry.retired_at = task_index
                self.retired += 1
                gone.append(entry.tid)
        return gone

    @property
    def active_count(self) -> int:
        return sum(1 for entry in self.entries.values() if entry.active)


# -- experiment -----------------------------------------------------------


def run_stream(tasks: Sequence[Dict[str, Any]], param_mode: str, min_support: int,
               max_idle: int, compound_k: int, segments: int = 4) -> Dict[str, Any]:
    """One repertoire over one ordered task stream: the unit of measurement.

    A stream must be a coherent workload (one domain, one agent), because a
    repertoire measured across concatenated domains retires instructions the
    moment the traffic changes -- a property of the reporting order, not of
    precipitation.
    """
    repertoire = Repertoire(min_support=min_support, max_idle=max_idle)
    seg_steps = [0] * segments
    seg_served = [0] * segments
    per_task: List[Dict[str, Any]] = []
    seen_windows: set = set()
    compound_windows = compound_repeats = 0

    for index, task in enumerate(tasks):
        repertoire.retire_stale(index)
        calls = calls_of(task)
        tids = [template_of(tool, args, param_mode) for tool, args in calls]
        served = 0
        for tid in tids:
            if repertoire.observe(tid, index) == "hit":
                served += 1
        segment = min(segments - 1, index * segments // max(1, len(tasks)))
        seg_steps[segment] += len(tids)
        seg_served[segment] += served

        if compound_k > 1 and len(tids) >= compound_k:
            for start_at in range(len(tids) - compound_k + 1):
                window = tuple(tids[start_at:start_at + compound_k])
                compound_windows += 1
                if window in seen_windows:
                    compound_repeats += 1
                seen_windows.add(window)

        per_task.append({
            "task": index,
            "task_id": task.get("task_id"),
            "steps": len(tids),
            "served": served,
            "fresh": len(tids) - served,
            "coverage": round(served / len(tids), 4) if tids else 0.0,
            "library_size": repertoire.active_count,
        })

    total_steps = sum(seg_steps)
    total_served = sum(seg_served)
    return {
        "tasks": len(tasks),
        "steps": total_steps,
        "served": total_served,
        "fresh": total_steps - total_served,
        "coverage": round(total_served / total_steps, 4) if total_steps else 0.0,
        "segments": [
            {
                "segment": i,
                "tasks": f"{i * 100 // segments}-{(i + 1) * 100 // segments}%",
                "steps": seg_steps[i],
                "served": seg_served[i],
                "coverage": round(seg_served[i] / seg_steps[i], 4) if seg_steps[i] else 0.0,
            }
            for i in range(segments)
        ],
        "repertoire": {
            "templates_total": len(repertoire.entries),
            "templates_active": repertoire.active_count,
            "created": repertoire.created,
            "retired": repertoire.retired,
            "remined": repertoire.remined,
        },
        "compound": {
            "k": compound_k,
            "windows": compound_windows,
            "repeats": compound_repeats,
            "recurrence": (round(compound_repeats / compound_windows, 4)
                           if compound_windows else 0.0),
        },
        "top_templates": [
            {"tool": entry.tid[0], "shape": list(entry.tid[1]),
             "uses": entry.uses, "support": entry.support, "active": entry.active}
            for entry in sorted(repertoire.entries.values(),
                                key=lambda e: e.uses, reverse=True)[:8]
        ],
        "per_task": per_task,
    }


def interleave(by_source: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Round-robin across sources: a mixed workload, not concatenated domains."""
    ordered: List[Dict[str, Any]] = []
    cursors = {source: 0 for source in by_source}
    while True:
        progressed = False
        for source in sorted(by_source):
            tasks = by_source[source]
            if cursors[source] < len(tasks):
                ordered.append(tasks[cursors[source]])
                cursors[source] += 1
                progressed = True
        if not progressed:
            return ordered


def run(corpus: Path, param_mode: str, min_support: int, max_idle: int,
        first_trial_only: bool = True, compound_k: int = 3) -> Dict[str, Any]:
    tasks = load_tasks(corpus, first_trial_only=first_trial_only)
    by_source: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        by_source.setdefault(str(task.get("source")), []).append(task)

    def stream(ordered: Sequence[Dict[str, Any]], idle: int) -> Dict[str, Any]:
        return run_stream(ordered, param_mode, min_support, idle, compound_k)

    interleaved = stream(interleave(by_source), max_idle)
    return {
        "schema": SCHEMA_VERSION,
        "self_test": False,
        "corpus": str(corpus),
        "param_mode": param_mode,
        "min_support": min_support,
        "max_idle": max_idle,
        "first_trial_only": first_trial_only,
        "tasks": len(tasks),
        "interleaved": interleaved,
        "per_source": {
            source: stream(grouped, max_idle)
            for source, grouped in sorted(by_source.items())
        },
    }


def provenance() -> Dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {"git_rev": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "python": os.environ.get("PYTHON_VERSION", "unknown")}


def summarise(report: Dict[str, Any]) -> str:
    lines = [f"tasks: {report['tasks']}  (param mode: {report['param_mode']})"]
    for name, stream in [("mixed (interleaved)", report["interleaved"]),
                         *report["per_source"].items()]:
        curve = " -> ".join(f"{seg['coverage']:.1%}" for seg in stream["segments"])
        lines.append(
            f"  {name:26s} tasks={stream['tasks']:3d} steps={stream['steps']:6d} "
            f"served={stream['coverage']:6.2%}  [{curve}]  "
            f"templates={stream['repertoire']['templates_active']}"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--param-mode", choices=["schema", "command"], default="schema")
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--max-idle", type=int, default=40)
    parser.add_argument("--compound-k", type=int, default=3)
    parser.add_argument("--all-trials", action="store_true",
                        help="include replay trials (a task then scores off itself)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args.corpus.resolve(), args.param_mode, args.min_support,
                 args.max_idle, first_trial_only=not args.all_trials,
                 compound_k=args.compound_k)
    report["provenance"] = provenance()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarise(report))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
