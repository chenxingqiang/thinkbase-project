"""Compound instructions on real traces: recurrence, bindability, and a sound bound.

An instruction that fuses several calls into one is only an instruction if it can
be *instantiated*: its later steps must take their arguments from the earlier
steps' results, not from values the planner happened to know. Inputs-only corpora
cannot test that, which is why earlier waves reported compound recurrence (75.0%
of three-step windows) as an upper bound and never claimed it. This script tests
the binding against recorded observations.

Definitions, all mechanical:

  * atomic template -- ``(tool, argument keys)``, the identity used throughout;
  * compound -- a contiguous k-gram of atomic templates with support >= 2;
  * bindable occurrence -- for every step after the first, every argument value is
    either already present in an earlier step's arguments (a parameter carried
    through) or appears as a scalar inside an earlier step's observation (an
    observation binding). Anything else is a free value, and the compound is not
    reproducible;
  * bindable compound -- bindable in *every* occurrence recorded;
  * earned reduction -- steps removed by greedily substituting the longest
    bindable compound at each position, longest first, no overlaps.

The reduction is a bound, not an implementation claim: it says how much of the
recorded stream a binder *could* absorb, and it is reported next to the fraction
of recurring compounds that fail to bind, which is the price of the pattern
family that recurrence alone would have advertised.

Usage:
    ./venv/bin/python examples/benchmarks/compound_binding.py \
        --corpus examples/real_traces/tau_bench_observed.jsonl \
        --output examples/benchmarks/compound_binding_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "compound_binding/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

Call = Tuple[str, Dict[str, Any], Optional[str]]
Template = Tuple[str, Tuple[str, ...]]


# -- corpus ---------------------------------------------------------------


def load_tasks(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]


def calls_of(task: Dict[str, Any]) -> List[Call]:
    out: List[Call] = []
    for call in task.get("calls") or []:
        if isinstance(call, (list, tuple)) and len(call) >= 2:
            out.append((str(call[0]), dict(call[1] or {}),
                        None if len(call) < 3 or call[2] is None else str(call[2])))
    return out


def template_of(call: Call) -> Template:
    return (call[0], tuple(sorted(str(key) for key in call[1])))


def _scalars(value: Any, out: Set[str]) -> None:
    """Collect printable scalars from a JSON-ish value (nested or raw text)."""
    if isinstance(value, dict):
        for item in value.values():
            _scalars(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _scalars(item, out)
    elif isinstance(value, bool) or value is None:
        return
    elif isinstance(value, (int, float)):
        out.add(str(value))
    elif isinstance(value, str):
        text = value.strip()
        if text:
            out.add(text)
        for token in text.replace(",", " ").replace(";", " ").split():
            if token:
                out.add(token.strip('"\'.:'))


def scalars_of_args(args: Dict[str, Any]) -> Dict[str, str]:
    """Map normalised scalar -> the argument name that carries it."""
    found: Dict[str, str] = {}
    for key, value in args.items():
        bag: Set[str] = set()
        _scalars(value, bag)
        for scalar in bag:
            found.setdefault(scalar, str(key))
    return found


def scalars_of_observation(obs: Optional[str]) -> Set[str]:
    if not obs:
        return set()
    bag: Set[str] = set()
    try:
        _scalars(json.loads(obs), bag)
    except (json.JSONDecodeError, TypeError):
        _scalars(obs, bag)
    return bag


# -- binding --------------------------------------------------------------


def provenance_of(calls: Sequence[Call]) -> Dict[Tuple[int, str], Tuple[str, int]]:
    """Where each later argument's value comes from, mechanically.

    For every step after the first and every argument, the origin is one of:

      * ``("obs", i)`` -- every scalar in the value appears in step *i*'s
        observation (data dependency; *i* is the latest such step);
      * ``("param", 0)`` -- every scalar appears in the compound's entry
        arguments (a parameter carried through);
      * ``("external", 0)`` -- anything else: a value the environment supplies
        from outside the call sequence (a user-provided order id, a constant).

    Step 0 is the entry point, so its arguments are parameters by definition.
    """
    origins: Dict[Tuple[int, str], Tuple[str, int]] = {}
    known_args: Set[str] = set(scalars_of_args(calls[0][1]))
    observations: List[Set[str]] = []
    for index, (_, args, obs) in enumerate(calls):
        observations.append(scalars_of_observation(obs))
        if index == 0:
            continue
        for key, value in args.items():
            bag: Set[str] = set()
            _scalars(value, bag)
            if not bag:
                origins[(index, key)] = ("constant", 0)
                continue
            source = -1
            for earlier in range(index):
                if bag <= observations[earlier]:
                    source = earlier
            if source >= 0:
                origins[(index, key)] = ("obs", source)
            elif bag <= known_args:
                origins[(index, key)] = ("param", 0)
            else:
                origins[(index, key)] = ("external", 0)
        known_args.update(scalars_of_args(args))
    return origins


def signature(origins: Dict[Tuple[int, str], Tuple[str, int]]) -> Tuple[Any, ...]:
    """The wiring a learner would have to reproduce, with external slots abstracted."""
    return tuple(sorted((index, key, kind, source)
                        for (index, key), (kind, source) in origins.items()))


def bind_occurrence(calls: Sequence[Call]) -> Tuple[bool, Dict[Tuple[int, str], Tuple[str, int]]]:
    """Legacy strict test: every later argument must come from a result or the entry."""
    origins = provenance_of(calls)
    strict = all(kind in ("obs", "param", "constant")
                 for kind, _ in origins.values())
    return strict, origins


def find_paths(value: Any, target: str, path: Optional[List[Any]] = None,
               out: Optional[List[List[Any]]] = None) -> List[List[Any]]:
    """All JSON paths at which ``target`` appears as a scalar (deterministic order)."""
    path = [] if path is None else path
    out = [] if out is None else out
    if isinstance(value, dict):
        for key in sorted(value, key=str):
            find_paths(value[key], target, path + [key], out)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            find_paths(item, target, path + [index], out)
    elif isinstance(value, str):
        if value.strip() == target.strip():
            out.append(list(path))
        else:
            for token in value.replace(",", " ").split():
                if token.strip('"\'.:') == target.strip():
                    out.append(list(path))
                    break
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if str(value) == target:
            out.append(list(path))
    return out


def read_path(value: Any, path: Sequence[Any]) -> Any:
    for step in path:
        try:
            value = value[step]
        except (TypeError, KeyError, IndexError) as error:
            raise ValueError(f"path {list(path)} not readable: {error}") from error
    return value


def learn_spec(calls: Sequence[Call], origins: Dict[Tuple[int, str], Tuple[str, int]]
               ) -> Dict[str, Any]:
    """Turn one occurrence into an executable compound instruction.

    Each argument of each later step becomes either a parameter slot (external or
    carried through) or a read from an earlier step's observation, recorded as a
    JSON path so execution can re-derive it from fresh results.
    """
    steps: List[Dict[str, Any]] = []
    external: List[str] = []
    entry_keys = {key: f"p_{key}" for key in sorted(calls[0][1])}
    for index, (tool, args, _) in enumerate(calls):
        spec_args: Dict[str, Any] = {}
        for key in sorted(args):
            if index == 0:
                slot = entry_keys[key]
                spec_args[key] = {"kind": "param", "from": slot}
                if slot not in external:
                    external.append(slot)
                continue
            kind, source = origins[(index, key)]
            if kind == "constant":
                spec_args[key] = {"kind": "literal", "value": args[key]}
                continue
            if kind == "obs":
                bag: Set[str] = set()
                _scalars(args[key], bag)
                target = sorted(bag)[0] if bag else str(args[key])
                path = find_paths(json.loads(calls[source][2]) if _is_json(calls[source][2])
                                  else (calls[source][2] or ""), target)
                spec_args[key] = {"kind": "path", "step": source,
                                  "path": path[0] if path else []}
            else:
                slot = f"x_{index}_{key}"
                spec_args[key] = {"kind": "param", "from": slot}
                if slot not in external:
                    external.append(slot)
        steps.append({"tool": tool, "args": spec_args})
    return {"steps": steps, "parameters": external}


def _is_json(text: Optional[str]) -> bool:
    if not text:
        return False
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def replay_spec(spec: Dict[str, Any], bindings: Dict[str, Any],
                observations: Sequence[Optional[str]]) -> List[Dict[str, Any]]:
    """Execute a compound spec against recorded observations.

    Returns the argument dict produced for each step. Tool calls themselves are
    not re-run: the recorded observation of each step is supplied by the caller,
    which is exactly the evidence needed to show the *binding* is reproduced.
    """
    produced: List[Dict[str, Any]] = []
    for index, step in enumerate(spec["steps"]):
        args: Dict[str, Any] = {}
        for key, slot in step["args"].items():
            if slot["kind"] == "literal":
                args[key] = slot["value"]
            elif slot["kind"] == "param":
                args[key] = bindings[slot["from"]]
            else:
                source = slot["step"]
                if source >= len(produced) or source >= index:
                    raise ValueError("compound reads a result that has not been produced")
                obs = observations[source]
                payload = json.loads(obs) if _is_json(obs) else (obs or "")
                args[key] = read_path(payload, slot["path"])
        produced.append(args)
    return produced


def verify_learning(tasks: Sequence[Dict[str, Any]],
                    learnable: Sequence[Tuple[Template, ...]]
                    ) -> Tuple[Set[Tuple[Template, ...]], Dict[str, Any]]:
    """For every learnable compound, learn its spec and replay it on new occurrences."""
    verified = checked = mismatched = drifted_count = 0
    passed: Set[Tuple[Template, ...]] = set()
    failures: List[Dict[str, Any]] = []
    for compound in learnable:
        k = len(compound)
        occurrences: List[Tuple[Dict[str, Any], List[Call]]] = []
        for task in tasks:
            calls = calls_of(task)
            templates = [template_of(call) for call in calls]
            for start in range(len(templates) - k + 1):
                if tuple(templates[start:start + k]) == compound:
                    occurrences.append((task, calls[start:start + k]))
        if len(occurrences) < 2:
            continue
        train_task, train = occurrences[0]
        spec = learn_spec(train, provenance_of(train))
        all_ok = True
        for task, window in occurrences[1:]:
            bindings: Dict[str, Any] = {}
            for position, (tool, args, _) in enumerate(window):
                if position == 0:
                    for key, slot in spec["steps"][0]["args"].items():
                        bindings[slot["from"]] = args[key]
                    continue
                origins = provenance_of(window)
                for key, spec_slot in spec["steps"][position]["args"].items():
                    if spec_slot["kind"] == "param" and spec_slot["from"] not in bindings:
                        kind, _source = origins[(position, key)]
                        if kind in ("obs", "constant"):
                            if spec_slot["from"] not in bindings:
                                bindings[spec_slot["from"]] = args[key]
                        else:
                            bindings[spec_slot["from"]] = args[key]
            try:
                produced = replay_spec(spec, bindings, [c[2] for c in window])
            except (ValueError, KeyError, IndexError) as error:
                # the learned path is not valid for this occurrence's observation;
                # count it as drift when the value is still present somewhere
                drifted = False
                for position, spec_step in enumerate(spec["steps"]):
                    if position >= len(window):
                        break
                    for key, slot in spec_step["args"].items():
                        if slot["kind"] != "path":
                            continue
                        obs = window[slot["step"]][2]
                        payload = json.loads(obs) if _is_json(obs) else (obs or "")
                        bag: Set[str] = set()
                        _scalars(window[position][1].get(key), bag)
                        target = sorted(bag)[0] if bag else ""
                        if target and find_paths(payload, target):
                            drifted = True
                if drifted:
                    drifted_count += 1
                else:
                    mismatched += 1
                all_ok = False
                failures.append({"compound": [[t[0], list(t[1])] for t in compound],
                                 "task": task.get("task_id"),
                                 "error": "path drift" if drifted else str(error)})
                continue
            checked += 1
            if all(produced[position] == window[position][1]
                   for position in range(len(window))):
                verified += 1
            else:
                mismatched += 1
                all_ok = False
                failures.append({"compound": [[t[0], list(t[1])] for t in compound],
                                 "task": task.get("task_id"), "error": "argument mismatch"})
        if all_ok and checked:
            passed.add(compound)
    return passed, {"checked": checked, "verified": verified, "mismatched": mismatched,
                    "path_drift": drifted_count, "failures": failures[:6]}


def mine_compounds(tasks: Sequence[Dict[str, Any]], max_k: int = 4,
                   min_support: int = 2) -> Dict[Tuple[Template, ...], List[int]]:
    """Recurring k-grams of atomic templates, keyed by the template sequence."""
    occurrences: Dict[Tuple[Template, ...], List[int]] = {}
    for index, task in enumerate(tasks):
        templates = [template_of(call) for call in calls_of(task)]
        for k in range(2, max_k + 1):
            for start in range(len(templates) - k + 1):
                key = tuple(templates[start:start + k])
                occurrences.setdefault(key, []).append(index)
    return {key: idx for key, idx in occurrences.items() if len(idx) >= min_support}


def per_task_positions(tasks: Sequence[Dict[str, Any]],
                       compounds: Iterable[Tuple[Template, ...]]
                      ) -> Dict[int, List[Tuple[int, int, Tuple[Template, ...]]]]:
    """Where each compound occurs: task -> [(start, k, compound)]."""
    index: Dict[int, List[Tuple[int, int, Tuple[Template, ...]]]] = {}
    for compound in compounds:
        k = len(compound)
        for task_index, task in enumerate(tasks):
            templates = [template_of(call) for call in calls_of(task)]
            for start in range(len(templates) - k + 1):
                if tuple(templates[start:start + k]) == compound:
                    index.setdefault(task_index, []).append((start, k, compound))
    return index


def earned_reduction(tasks: Sequence[Dict[str, Any]],
                     bindable: Set[Tuple[Template, ...]],
                     occurrences: Dict[int, List[Tuple[int, int, Tuple[Template, ...]]]]
                     ) -> Dict[str, Any]:
    """Steps absorbed if bindable compounds are substituted, longest match first."""
    total_steps = sum(len(calls_of(task)) for task in tasks)
    removed = 0
    applied = 0
    for task_index, task in enumerate(tasks):
        used = [False] * len(calls_of(task))
        for start, k, compound in sorted(occurrences.get(task_index, []),
                                         key=lambda item: (-item[1], item[0])):
            if compound not in bindable:
                continue
            if any(used[start:start + k]):
                continue
            for position in range(start, start + k):
                used[position] = True
            removed += k - 1        # k steps become one compound instruction
            applied += 1
    return {
        "steps": total_steps,
        "steps_removed": removed,
        "compounds_applied": applied,
        "reduction": round(removed / total_steps, 4) if total_steps else 0.0,
    }


def run(corpus: Path, max_k: int = 4, min_support: int = 2) -> Dict[str, Any]:
    tasks = load_tasks(corpus)
    compounds = mine_compounds(tasks, max_k=max_k, min_support=min_support)

    learnable: Set[Tuple[Template, ...]] = set()
    strict: Set[Tuple[Template, ...]] = set()
    detail: List[Dict[str, Any]] = []
    internal_slots = external_slots = 0
    for compound, task_indices in sorted(compounds.items(), key=lambda item: len(item[0])):
        signatures: List[Tuple[Any, ...]] = []
        strictly_bound = True
        for task_index in sorted(set(task_indices)):
            calls = calls_of(tasks[task_index])
            templates = [template_of(call) for call in calls]
            k = len(compound)
            for start in range(len(templates) - k + 1):
                if tuple(templates[start:start + k]) != compound:
                    continue
                window = calls[start:start + k]
                origins = provenance_of(window)
                signatures.append(signature(origins))
                if not all(kind in ("obs", "param", "constant")
                           for kind, _ in origins.values()):
                    strictly_bound = False
        # learnable: the wiring is the same in every occurrence, so the compound
        # can be stored as one instruction whose parameters are its external slots
        if signatures and len(set(signatures)) == 1:
            learnable.add(compound)
            for index, key, kind, _ in signatures[0]:
                if kind in ("obs", "constant"):
                    internal_slots += 1
                else:
                    external_slots += 1
        elif signatures:
            detail.append({
                "compound": [[t[0], list(t[1])] for t in compound],
                "occurrences": len(signatures),
                "distinct_wirings": len(set(signatures)),
            })
        if signatures and strictly_bound:
            strict.add(compound)

    occurrences = per_task_positions(tasks, learnable)
    bound_reduction = earned_reduction(tasks, learnable, occurrences)
    verified_set, replay = verify_learning(tasks, sorted(learnable, key=len))
    reduction = earned_reduction(tasks, verified_set,
                                 per_task_positions(tasks, verified_set))
    strict_reduction = earned_reduction(tasks, strict,
                                        per_task_positions(tasks, strict))

    by_k: Dict[int, Dict[str, int]] = {}
    for compound in compounds:
        k = len(compound)
        bucket = by_k.setdefault(k, {"recurring": 0, "signature_consistent": 0,
                                     "replay_verified": 0, "strictly_bound": 0})
        bucket["recurring"] += 1
        if compound in learnable:
            bucket["signature_consistent"] += 1
        if compound in verified_set:
            bucket["replay_verified"] += 1
        if compound in strict:
            bucket["strictly_bound"] += 1

    slots = internal_slots + external_slots
    return {
        "schema": SCHEMA_VERSION,
        "self_test": False,
        "corpus": str(corpus),
        "tasks": len(tasks),
        "steps": reduction["steps"],
        "min_support": min_support,
        "max_k": max_k,
        "compounds": {
            "recurring": len(compounds),
            "signature_consistent": len(learnable),
            "signature_consistent_share": round(len(learnable) / len(compounds), 4) if compounds else 0.0,
            "replay_verified": len(verified_set),
            "replay_verified_share": round(len(verified_set) / len(compounds), 4) if compounds else 0.0,
            "strictly_bound": len(strict),
            "by_k": {str(k): v for k, v in sorted(by_k.items())},
        },
        "wiring": {
            "slots": slots,
            "observation_or_constant": internal_slots,
            "external_parameter": external_slots,
            "internal_share": round(internal_slots / slots, 4) if slots else 0.0,
        },
        "reduction_if_substituted": reduction,
        "upper_bound_signature_consistent": bound_reduction,
        "strict_reduction": strict_reduction,
        "replay_verification": replay,
        "inconsistent_wiring_examples": detail[:8],
        "provenance": provenance(),
    }


def provenance() -> Dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {"git_rev": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain"))}


def summarise(report: Dict[str, Any]) -> str:
    c = report["compounds"]
    r = report["reduction_if_substituted"]
    b = report["upper_bound_signature_consistent"]
    w = report["wiring"]
    v = report["replay_verification"]
    lines = [
        f"tasks: {report['tasks']}  steps: {report['steps']}",
        f"recurring compounds (k<= {report['max_k']}, support >= {report['min_support']}): "
        f"{c['recurring']}",
        f"  signature-consistent wiring: {c['signature_consistent']} "
        f"({c['signature_consistent_share']:.1%}) -> bound {b['reduction']:.2%}",
        f"  replay-verified (spec from one occurrence reproduces the others): "
        f"{c['replay_verified']} ({c['replay_verified_share']:.1%}) "
        f"-> earned {r['reduction']:.2%} ({r['steps_removed']}/{r['steps']} steps, "
        f"{r['compounds_applied']} applications)",
        "  by k (verified/recurring): " + ", ".join(
            f"k={k}: {val['replay_verified']}/{val['recurring']}"
            for k, val in sorted(c["by_k"].items(), key=lambda x: int(x[0]))),
        f"wiring slots: {w['observation_or_constant']} internal / {w['external_parameter']} "
        f"external ({w['internal_share']:.1%} internal)",
        f"replay: {v['verified']}/{v['checked']} exact, {v['path_drift']} path drift, "
        f"{v['mismatched']} mismatch",
        f"strict test (all args obs/entry-bound): {report['strict_reduction']['reduction']:.2%}",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=4)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args.corpus.resolve(), max_k=args.max_k, min_support=args.min_support)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarise(report))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
