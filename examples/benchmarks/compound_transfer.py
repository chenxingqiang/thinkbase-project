"""Cross-model transfer of learned compound instructions on recorded traces.

The compound-binding experiment learns an instruction from one occurrence and
replays it on *other occurrences of the same model's traces*. This script asks
the question instruction memory actually poses: a compound learned from model
A's sessions -- a sequence of atomic templates plus the wiring that says which
argument comes from which earlier result -- is replayed on model B's traces of
the same tasks. What travels with the instruction, and what breaks?

Because tau-bench renders tool results through the same environment simulator
regardless of the policy, the *surface* of a recurring compound cannot vary
across models: a compound is keyed on ``(tool, argument keys)``, so any window
that matches it already has identical argument names. Only the *provenance* of
the values can differ -- model A reads a value out of an earlier result, model B
pastes the same value in from the user request. The test is therefore exactly a
test of whether the learned dependency edges belong to the environment or to
the policy that happened to run.

Definitions, all mechanical:

  * eligibility (learn side only, computed on the learner's source) -- a
    compound is eligible when (i) it is recurring (support >= ``min_support``),
    (ii) its provenance signature is identical in every learner occurrence, so
    exactly one spec exists, and (iii) that spec replays exactly on every
    learner occurrence when only its declared free slots are bound. A spec that
    does not reproduce its own model's traces is not transferred, whatever it
    does elsewhere; the compounds it excludes are reported, with their own
    tester counts, so the filter cannot hide a success;
  * free slots -- the entry arguments of step 0 plus the arguments the learned
    spec marks as external. A tester window is bound by supplying those slots
    from that window's own call arguments (an external value is, by definition,
    something the caller supplies; the instruction claims nothing about it);
  * success (primary, "transferred") -- replaying the spec on the tester window
    reproduces *every* step's argument dict exactly. Per compound that must hold
    for all its tester occurrences; the denominator is the eligible compounds
    that occur at least once in the tester source ("exposed");
  * success (secondary, "faithful") -- transferred, and every non-entry slot is
    wired the same way in the tester window: nothing the spec reads from a
    result was in fact supplied from outside (an *over-claim*, the dangerous
    direction), and nothing the spec calls free was in fact derivable (an
    *under-claim*, which only makes the instruction demand more inputs than the
    trace needed). Exact but unfaithful means the value came out right for a
    reason the instruction does not contain;
  * wiring split -- an eligible compound is *observation-wired* if at least one
    of its non-entry slots is read from an earlier result, and *parameter-only*
    if all of them are free. Only the first kind can save a dispatch; the second
    is a repeated call with the same argument shape. Every transfer rate is
    reported for both, because pooling them hides which half transfers. Note
    that a parameter-only compound replays *by construction*: every later
    argument is re-supplied from the window it runs on, so its exactness is a
    consistency check on the harness, not evidence of transfer. The wired subset
    is the informative one, and it is reported twice -- once over the gated
    repertoire and once over every signature-consistent compound, including the
    ones the gate excluded -- so its denominator is not the gate's own doing;
  * failure modes, first failing step then first failing key:
      - ``unreadable_path``: the recorded JSON path no longer resolves (shape or
        ordering changed);
      - ``ambiguous_read``: the required value is present in the observation the
        path reads, but at another path -- the result offers several scalars and
        the path names the wrong one (positional under-specification);
      - ``wrong_source_step``: the value is derivable, but from a different
        earlier step than the spec reads;
      - ``external_value``: the value is nowhere in the earlier results, so the
        tester obtained it from outside the call sequence (an over-claim);
      - ``constant_differs``: a literal slot carried a different constant;
      - ``missing_param``: a declared free slot has no argument in the window to
        bind it to -- unreachable while compounds are keyed on argument keys, so
        it exists as a defensive check and never fires on this corpus;
      - ``no_exposure``: the compound never occurs in the tester source -- not a
        transfer failure but absence of evidence, reported as a denominator
        (``exposure_share``), never as a zero;
    the five real modes are then grouped for reporting: the three path modes are
    *path drift* (the value exists, the learned path does not name it) and the
    other two are *mismatches*; drift + mismatch + unbound slots partition the
    non-exact replays exactly, and the script asserts that partition;
  * denominators -- every rate carries its own: eligible compounds, exposed
    compounds, occurrence windows, and steps of the tester source absorbed by
    transferred compounds. Nothing is divided by "all compounds";
  * controls -- at a matched learning budget (learner half H1 of the tasks), a
    memory is replayed (a) on the same model's other tasks H2: the within-source
    control, model fixed, task instance varied; (b) on the other model's traces
    of the *same* tasks H1: task fixed, model varied; (c) on the other model's
    traces of H2: both varied. (a) is not a ceiling for (b) -- the two vary
    different things -- so all three are reported per direction and pooled with
    per-cell denominators;
  * cross-domain -- retail to airline is *not* reported as a transfer rate. The
    two ladders share four scaffolding tools (``think``, ``calculate``,
    ``transfer_to_human_agents``, ``get_user_details``) out of 16 and 14; every
    other tool name is disjoint and a compound of length k needs all k
    templates, so 1 of the 49 eligible retail compounds is even expressible in
    the airline vocabulary and none occurs there. The numerator would be zero
    because the denominator is empty, and the task ids collide numerically (50
    of the airline ids exist in retail), so an id join across domains would
    silently pair unrelated tasks. Cross-domain asks whether the *environment's*
    wiring transfers -- a different question from the one this design tests, and
    one this corpus cannot answer; we report the vocabulary overlap instead.

Usage:
    ./venv/bin/python examples/benchmarks/compound_transfer.py \
        --corpus examples/real_traces/tau_bench_observed.jsonl \
        --output examples/benchmarks/compound_transfer_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "compound_transfer/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

Call = Tuple[str, Dict[str, Any], Optional[str]]
Template = Tuple[str, Tuple[str, ...]]
Compound = Tuple[Template, ...]

LEARNER, TESTER = "gpt-4o-retail", "sonnet-35-new-retail"
LEARNER_AIR, TESTER_AIR = "gpt-4o-airline", "sonnet-35-new-airline"

FAILURE_MODES = ("unreadable_path", "ambiguous_read", "wrong_source_step",
                 "external_value", "constant_differs", "missing_param")

# A *drifted* read is one where the wanted value exists somewhere in the earlier
# results but not where the learned spec reads it: the path failed, not the
# value. Everything else that produced a wrong argument is a *mismatch*. The two
# buckets plus ``missing_param`` partition the failed occurrences exactly, which
# is asserted in ``replay_stats``.
DRIFT_MODES = ("unreadable_path", "ambiguous_read", "wrong_source_step")
MISMATCH_MODES = ("external_value", "constant_differs")


# -- shared machinery -----------------------------------------------------


def _binding_module():
    """Reuse ``compound_binding``: one definition of spec, replay and provenance."""
    existing = sys.modules.get("compound_binding")
    if existing is not None:
        return existing
    path = Path(__file__).resolve().parent / "compound_binding.py"
    spec = importlib.util.spec_from_file_location("compound_binding", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["compound_binding"] = module
    spec.loader.exec_module(module)
    return module


CB = _binding_module()


# -- corpus ---------------------------------------------------------------


def load_sources(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Source name -> its tasks, in file order (deterministic)."""
    sources: Dict[str, List[Dict[str, Any]]] = {}
    for task in CB.load_tasks(path):
        sources.setdefault(str(task.get("source")), []).append(task)
    return sources


def task_key(task: Dict[str, Any]) -> Tuple[int, Any]:
    text = str(task.get("task_id"))
    return (0, int(text)) if text.lstrip("-").isdigit() else (1, text)


def split_halves(tasks: Sequence[Dict[str, Any]]
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministic task split: even positions / odd positions of the stable order."""
    ordered = sorted(tasks, key=task_key)
    return ordered[0::2], ordered[1::2]


def windows(tasks: Sequence[Dict[str, Any]], k: int
            ) -> Iterable[Tuple[Dict[str, Any], Tuple[Template, ...], List[Call]]]:
    for task in tasks:
        calls = CB.calls_of(task)
        templates = [CB.template_of(call) for call in calls]
        for start in range(len(templates) - k + 1):
            yield task, tuple(templates[start:start + k]), calls[start:start + k]


def steps_of(tasks: Sequence[Dict[str, Any]]) -> int:
    return sum(len(CB.calls_of(task)) for task in tasks)


# -- replay ---------------------------------------------------------------


def slot_sources(spec: Dict[str, Any]) -> Dict[str, Tuple[int, str]]:
    """Free-slot name -> (step index, argument key) that supplies it."""
    out: Dict[str, Tuple[int, str]] = {}
    for index, step in enumerate(spec["steps"]):
        for key, slot in step["args"].items():
            if slot["kind"] == "param":
                out.setdefault(slot["from"], (index, key))
    return out


def bind_window(spec: Dict[str, Any], window: Sequence[Call]
                ) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Supply the spec's free slots from the window's own call arguments."""
    bindings: Dict[str, Any] = {}
    for slot, (index, key) in slot_sources(spec).items():
        if key in window[index][1]:
            bindings[slot] = window[index][1][key]
        elif key in window[0][1]:
            bindings[slot] = window[0][1][key]
        else:
            return None, "missing_param"
    return bindings, None


def replay_window(spec: Dict[str, Any], window: Sequence[Call]
                  ) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    bindings, error = bind_window(spec, window)
    if error:
        return None, error
    try:
        produced = CB.replay_spec(spec, bindings, [call[2] for call in window])
    except ValueError:
        return None, "unreadable_path"
    except KeyError:                       # pragma: no cover - defensive
        return None, "missing_param"
    return produced, None


def window_exact(spec: Dict[str, Any], window: Sequence[Call]) -> bool:
    produced, error = replay_window(spec, window)
    return error is None and produced is not None and all(
        produced[i] == window[i][1] for i in range(len(window)))


def first_mismatch(spec: Dict[str, Any], window: Sequence[Call]
                   ) -> Optional[Tuple[int, str]]:
    produced, error = replay_window(spec, window)
    if error or produced is None:
        return None
    for index in range(1, len(window)):
        for key in sorted(window[index][1]):
            if produced[index].get(key) != window[index][1][key]:
                return index, key
    return None


def _scalars_of(value: Any) -> Set[str]:
    bag: Set[str] = set()
    CB._scalars(value, bag)
    return bag


def _prefix_scalars(window: Sequence[Call], upto: int) -> Set[str]:
    bag: Set[str] = set()
    for index in range(upto):
        bag |= CB.scalars_of_observation(window[index][2])
    return bag


def failure_mode(spec: Dict[str, Any], window: Sequence[Call],
                 error: Optional[str]) -> str:
    """Classify one failed tester occurrence (deterministic precedence)."""
    if error is not None:
        return error if error in FAILURE_MODES else "unreadable_path"   # defensive
    located = first_mismatch(spec, window)
    if located is None:
        return "constant_differs"
    index, key = located
    slot = spec["steps"][index]["args"][key]
    if slot["kind"] != "path":
        return "constant_differs"
    observation = window[slot["step"]][2]
    payload = json.loads(observation) if CB._is_json(observation) else (observation or "")
    wanted = sorted(_scalars_of(window[index][1][key]))
    target = wanted[0] if wanted else str(window[index][1][key])
    if target and CB.find_paths(payload, target):
        return "ambiguous_read"
    if target and target in _prefix_scalars(window, slot["step"] + 1):
        return "wrong_source_step"
    return "external_value"


def slot_disagreements(spec: Dict[str, Any], window: Sequence[Call]) -> Tuple[int, int]:
    """(over-claimed, under-claimed) non-entry slots for one tester window."""
    provenance = CB.provenance_of(window)
    over = under = 0
    for index in range(1, len(window)):
        for key in sorted(window[index][1]):
            slot = spec["steps"][index]["args"][key]
            kind, _source = provenance[(index, key)]
            if slot["kind"] == "path":
                if kind != "obs":
                    over += 1
            elif kind == "obs":
                under += 1
    return over, under


# -- eligibility (learner side only) --------------------------------------


def compound_spec(tasks: Sequence[Dict[str, Any]], compound: Compound
                  ) -> Tuple[Dict[str, Any], List[List[Call]]]:
    """The one spec for a compound's first learner occurrence, and all occurrences."""
    k = len(compound)
    occurrences = [callsw for _task, templates, callsw in windows(tasks, k)
                   if templates == compound]
    return CB.learn_spec(occurrences[0], CB.provenance_of(occurrences[0])), occurrences


def describe(compound: Compound, spec: Dict[str, Any], occurrences: int
             ) -> Dict[str, Any]:
    reads = sum(1 for step in spec["steps"] for slot in step["args"].values()
                if slot["kind"] == "path")
    return {"spec": spec,
            "occurrences": occurrences,
            "observation_reads": reads,
            "free_slots": len(spec["parameters"]),
            "wired": reads > 0}


def eligible_specs(tasks: Sequence[Dict[str, Any]], max_k: int, min_support: int
                   ) -> Tuple[Dict[Compound, Dict[str, Any]], Dict[str, int],
                              Dict[Compound, Dict[str, Any]]]:
    """Compounds surviving recurrence, wiring consistency and in-source replay.

    Returns (eligible, funnel, excluded), where ``excluded`` holds the
    signature-consistent compounds that failed their own model's replay.
    """
    funnel = {"recurring": 0, "signature_consistent": 0, "in_source_replay": 0}
    rally = CB.mine_compounds(tasks, max_k=max_k, min_support=min_support)
    funnel["recurring"] = len(rally)
    eligible: Dict[Compound, Dict[str, Any]] = {}
    excluded: Dict[Compound, Dict[str, Any]] = {}
    for compound in sorted(rally, key=lambda c: (len(c), c)):
        spec, occurrences = compound_spec(tasks, compound)
        if len(occurrences) < min_support:
            continue
        if len({CB.signature(CB.provenance_of(calls)) for calls in occurrences}) != 1:
            continue
        funnel["signature_consistent"] += 1
        if not all(window_exact(spec, calls) for calls in occurrences):
            excluded[compound] = describe(compound, spec, len(occurrences))
            continue
        funnel["in_source_replay"] += 1
        eligible[compound] = describe(compound, spec, len(occurrences))
    return eligible, funnel, excluded


# -- statistics -----------------------------------------------------------


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    """95% Wilson interval; [0, 0] when there is nothing to divide by."""
    if total <= 0:
        return [0.0, 0.0]
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _rate(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


# -- replay statistics ----------------------------------------------------


def replay_stats(compounds: Iterable[Compound], specs: Dict[Compound, Dict[str, Any]],
                 tester_tasks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Replay a set of learned specs on the tester source; one row of a table."""
    ordered = sorted(compounds, key=lambda c: (len(c), c))
    exposed: List[Compound] = []
    transferred: List[Compound] = []
    faithful: List[Compound] = []
    tested = exact = 0
    over_slots = under_slots = 0
    failures: Dict[str, int] = {mode: 0 for mode in FAILURE_MODES}
    by_k: Dict[str, Dict[str, int]] = {}

    for compound in ordered:
        k = len(compound)
        spec = specs[compound]["spec"]
        bucket = by_k.setdefault(str(k), {"eligible": 0, "exposed": 0,
                                          "transferred": 0, "faithful": 0})
        bucket["eligible"] += 1
        occurrences = [callsw for _task, templates, callsw in windows(tester_tasks, k)
                       if templates == compound]
        if not occurrences:
            continue
        exposed.append(compound)
        bucket["exposed"] += 1
        all_exact = all_faithful = True
        for window in occurrences:
            tested += 1
            over, under = slot_disagreements(spec, window)
            over_slots += over
            under_slots += under
            produced, error = replay_window(spec, window)
            ok = (error is None and produced is not None and
                  all(produced[i] == window[i][1] for i in range(len(window))))
            if ok:
                exact += 1
                if over or under:
                    all_faithful = False
                continue
            all_exact = all_faithful = False
            mode = failure_mode(spec, window, error)
            failures[mode] = failures.get(mode, 0) + 1
        if all_exact:
            transferred.append(compound)
            bucket["transferred"] += 1
        if all_faithful:
            faithful.append(compound)
            bucket["faithful"] += 1

    drifted = sum(failures[mode] for mode in DRIFT_MODES)
    mismatched = sum(failures[mode] for mode in MISMATCH_MODES)
    unbound = failures.get("missing_param", 0)
    failed = tested - exact
    assert drifted + mismatched + unbound == failed, "failure buckets do not partition"
    return {
        "eligible": len(ordered),
        "exposed": len(exposed),
        "exposure_share": _rate(len(exposed), len(ordered)),
        "transferred": len(transferred),
        "transfer_rate": _rate(len(transferred), len(exposed)),
        "transfer_ci95": wilson(len(transferred), len(exposed)),
        "faithful": len(faithful),
        "faithful_rate": _rate(len(faithful), len(exposed)),
        "occurrences": {"tested": tested, "exact": exact,
                        "exact_share": _rate(exact, tested),
                        "failed": failed,
                        "path_drift": drifted,
                        "mismatched": mismatched,
                        "unbound_slots": unbound},
        "slots": {"under_claimed": under_slots, "over_claimed": over_slots,
                  "total": under_slots + over_slots,
                  "under_claimed_share": _rate(under_slots, under_slots + over_slots)},
        "failures": {mode: count for mode, count in failures.items() if count},
        "by_k": {key: value for key, value in sorted(by_k.items()) if value["eligible"]},
        "transferred_compounds": [serialise(compound) for compound in transferred],
    }


def serialise(compound: Compound) -> List[Any]:
    """A compound as plain JSON (stable order, so two runs compare equal)."""
    return [[tool, list(keys)] for tool, keys in compound]


def deserialise(row: Sequence[Any]) -> Compound:
    return tuple((str(tool), tuple(str(key) for key in keys)) for tool, keys in row)


def step_coverage(tasks: Sequence[Dict[str, Any]], compounds: Iterable[Compound]
                  ) -> Dict[str, Any]:
    """Steps of the tester source that the given compounds would absorb."""
    ordered = sorted(compounds, key=lambda c: (-len(c), c))
    total = covered = 0
    for task in tasks:
        calls = CB.calls_of(task)
        templates = [CB.template_of(call) for call in calls]
        used = [False] * len(calls)
        total += len(calls)
        for compound in ordered:
            k = len(compound)
            for start in range(len(templates) - k + 1):
                if tuple(templates[start:start + k]) != compound:
                    continue
                if any(used[start:start + k]):
                    continue
                for position in range(start, start + k):
                    used[position] = True
        covered += sum(used)
    return {"steps": total, "covered": covered, "share": _rate(covered, total)}


# -- one cell of the design -----------------------------------------------


def cell(learner_tasks: Sequence[Dict[str, Any]],
         tester_tasks: Sequence[Dict[str, Any]],
         eligible: Dict[Compound, Dict[str, Any]],
         excluded: Dict[Compound, Dict[str, Any]],
         label: str) -> Dict[str, Any]:
    overall = replay_stats(eligible, eligible, tester_tasks)
    wired = [compound for compound, info in eligible.items() if info["wired"]]
    plain = [compound for compound, info in eligible.items() if not info["wired"]]
    wired_all = wired + [compound for compound, info in excluded.items() if info["wired"]]
    specs_all = dict(eligible)
    specs_all.update(excluded)
    transferred = {deserialise(entry) for entry in overall["transferred_compounds"]}
    overall["wiring"] = {
        "observation_wired": replay_stats(wired, eligible, tester_tasks),
        "parameter_only": replay_stats(plain, eligible, tester_tasks),
        "observation_wired_all_specs": replay_stats(wired_all, specs_all, tester_tasks),
    }
    overall["step_coverage"] = step_coverage(tester_tasks, transferred)
    overall["step_coverage_wiring"] = {
        "observation_wired": step_coverage(tester_tasks, transferred & set(wired)),
        "parameter_only": step_coverage(tester_tasks, transferred & set(plain)),
    }
    overall["observation_wired_all_specs_coverage"] = step_coverage(
        tester_tasks, {deserialise(row) for row in
                       overall["wiring"]["observation_wired_all_specs"]["transferred_compounds"]})
    overall["excluded_by_in_source_replay"] = replay_stats(
        excluded, dict(excluded), tester_tasks)
    shared = {str(task.get("task_id")) for task in learner_tasks} & {
        str(task.get("task_id")) for task in tester_tasks}
    overall.update({
        "label": label,
        "learner_tasks": len(learner_tasks),
        "tester_tasks": len(tester_tasks),
        "shared_task_ids": len(shared),
        "test_task_overlap_with_learner": _rate(len(shared), len(tester_tasks)),
        "learner_steps": steps_of(learner_tasks),
        "tester_steps": steps_of(tester_tasks),
    })
    return overall


# -- the whole design -----------------------------------------------------


def cross_model(sources: Dict[str, List[Dict[str, Any]]], max_k: int,
                min_support: int) -> Dict[str, Any]:
    """Learn on one model's traces, replay on the other's; both directions."""
    cells: Dict[str, Any] = {}
    funnels: Dict[str, Any] = {}
    pairs = {"retail": (LEARNER, TESTER),
             "retail_reverse": (TESTER, LEARNER),
             "airline": (LEARNER_AIR, TESTER_AIR),
             "airline_reverse": (TESTER_AIR, LEARNER_AIR)}
    for name, (learner, tester) in pairs.items():
        if learner not in sources or tester not in sources:
            continue
        eligible, funnel, excluded = eligible_specs(sources[learner], max_k, min_support)
        funnels[name] = dict(funnel,
                             eligible=len(eligible),
                             excluded_by_replay=len(excluded))
        cells[name] = cell(sources[learner], sources[tester], eligible, excluded,
                           f"{learner} -> {tester}")
    return {"cells": cells, "eligibility_funnel": funnels}


def controls(sources: Dict[str, List[Dict[str, Any]]], max_k: int,
             min_support: int) -> Dict[str, Any]:
    """Matched learning budget: learn on H1, vary the model and the tasks."""
    per_direction: Dict[str, Any] = {}
    pooled = {key: {"exposed": 0, "transferred": 0, "faithful": 0,
                    "tested": 0, "exact": 0,
                    "wired_exposed": 0,
                    "plain_exposed": 0, "plain_transferred": 0}
              for key in ("within_source", "same_task", "unseen_tasks")}
    for name, (learner, tester) in {"retail": (LEARNER, TESTER),
                                    "retail_reverse": (TESTER, LEARNER)}.items():
        if learner not in sources or tester not in sources:
            continue
        learn_h1, learn_h2 = split_halves(sources[learner])
        tester_h1, tester_h2 = split_halves(sources[tester])
        eligible, _funnel, excluded = eligible_specs(learn_h1, max_k, min_support)
        conditions = {
            "within_source": cell(learn_h1, learn_h2, eligible, excluded,
                                  f"{learner} H1 -> {learner} H2"),
            "same_task": cell(learn_h1, tester_h1, eligible, excluded,
                              f"{learner} H1 -> {tester} H1"),
            "unseen_tasks": cell(learn_h1, tester_h2, eligible, excluded,
                                 f"{learner} H1 -> {tester} H2"),
        }
        per_direction[name] = {"learn_tasks": len(learn_h1),
                               "eligible": len(eligible),
                               "conditions": conditions}
        for key, value in conditions.items():
            pooled[key]["exposed"] += value["exposed"]
            pooled[key]["transferred"] += value["transferred"]
            pooled[key]["faithful"] += value["faithful"]
            pooled[key]["tested"] += value["occurrences"]["tested"]
            pooled[key]["exact"] += value["occurrences"]["exact"]
            pooled[key]["wired_exposed"] += value["wiring"]["observation_wired"]["exposed"]
            pooled[key]["plain_exposed"] += value["wiring"]["parameter_only"]["exposed"]
            pooled[key]["plain_transferred"] += value["wiring"]["parameter_only"]["transferred"]
    summary = {}
    for key, value in pooled.items():
        summary[key] = {
            "exposed": value["exposed"],
            "transferred": value["transferred"],
            "rate": _rate(value["transferred"], value["exposed"]),
            "ci95": wilson(value["transferred"], value["exposed"]),
            "faithful": value["faithful"],
            "faithful_rate": _rate(value["faithful"], value["exposed"]),
            "occurrences": value["tested"],
            "occurrence_exact_share": _rate(value["exact"], value["tested"]),
            "observation_wired_exposed": value["wired_exposed"],
            "parameter_only": {
                "exposed": value["plain_exposed"],
                "transferred": value["plain_transferred"],
                "rate": _rate(value["plain_transferred"], value["plain_exposed"]),
                "ci95": wilson(value["plain_transferred"], value["plain_exposed"]),
            },
        }
    return {"per_direction": per_direction, "pooled": summary}


def cross_domain(sources: Dict[str, List[Dict[str, Any]]], max_k: int,
                 min_support: int) -> Dict[str, Any]:
    """Why retail -> airline is a vocabulary statement, not a transfer rate."""
    retail, airline = sources.get(LEARNER, []), sources.get(TESTER_AIR, [])
    if not retail or not airline:
        return {}
    retail_templates = {CB.template_of(call) for task in retail for call in CB.calls_of(task)}
    airline_templates = {CB.template_of(call) for task in airline
                         for call in CB.calls_of(task)}
    shared = retail_templates & airline_templates
    eligible, _funnel, _excluded = eligible_specs(retail, max_k, min_support)
    expressible = [compound for compound in eligible
                   if all(template in airline_templates for template in compound)]
    exposed = [compound for compound in eligible
               if any(templates == compound
                      for _task, templates, _calls in windows(airline, len(compound)))]
    retail_ids = {str(task.get("task_id")) for task in retail}
    airline_ids = {str(task.get("task_id")) for task in airline}
    return {
        "learner": LEARNER,
        "tester": TESTER_AIR,
        "learner_atomic_templates": len(retail_templates),
        "tester_atomic_templates": len(airline_templates),
        "shared_atomic_templates": len(shared),
        "shared_tool_names": sorted({tool for tool, _keys in shared}),
        "eligible_compounds": len(eligible),
        "eligible_expressible_in_tester": len(expressible),
        "eligible_exposed_in_tester": len(exposed),
        "colliding_task_ids": len(retail_ids & airline_ids),
        "tester_steps": steps_of(airline),
        "note": ("a compound of length k needs all k templates in the tester "
                 "vocabulary; the only cross-domain overlap is scaffolding "
                 "tools, so exposure is zero by vocabulary, not by transfer "
                 "failure. Task ids also collide, so an id join across domains "
                 "would silently pair unrelated tasks."),
    }


def source_profile(sources: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    profile: Dict[str, Any] = {}
    for name in sorted(sources):
        tasks = sources[name]
        tools = sorted({call[0] for task in tasks for call in CB.calls_of(task)})
        rewards = [float(task.get("reward") or 0.0) for task in tasks]
        profile[name] = {
            "tasks": len(tasks),
            "steps": steps_of(tasks),
            "tools": len(tools),
            "mean_reward": round(sum(rewards) / len(rewards), 4) if rewards else 0.0,
        }
    return profile


# -- entry points ---------------------------------------------------------


MAX_K = 4
MIN_SUPPORT = 2


def run(corpus: Path, max_k: int = MAX_K, min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
    sources = load_sources(corpus)
    return {
        "schema": SCHEMA_VERSION,
        "corpus": str(corpus),
        "max_k": max_k,
        "min_support": min_support,
        "sources": source_profile(sources),
        "cross_model": cross_model(sources, max_k, min_support),
        "controls": controls(sources, max_k, min_support),
        "cross_domain": cross_domain(sources, max_k, min_support),
        "provenance": provenance(corpus),
    }


def provenance(corpus: Path) -> Dict[str, Any]:
    """Revision, tree state, and the corpus identity the numbers are a function of."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"
    return {"git_rev": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
            "corpus_bytes": corpus.stat().st_size}


def summarise(report: Dict[str, Any]) -> str:
    lines = [f"corpus: {Path(report['corpus']).name}  max_k={report['max_k']}  "
             f"min_support={report['min_support']}"]
    for name, info in report["sources"].items():
        lines.append(f"  {name:24s} tasks={info['tasks']:3d} steps={info['steps']:4d} "
                     f"tools={info['tools']:2d} reward={info['mean_reward']:.3f}")
    lines.append("eligibility funnels (recurring -> signature-consistent -> in-source replay):")
    for name, funnel in report["cross_model"]["eligibility_funnel"].items():
        lines.append(f"  {name:16s} {funnel['recurring']:3d} -> "
                     f"{funnel['signature_consistent']:3d} -> "
                     f"{funnel['in_source_replay']:3d} eligible "
                     f"({funnel['excluded_by_replay']} excluded by replay)")
    lines.append("cross-model (learn -> test, same domain):")
    for name, value in report["cross_model"]["cells"].items():
        occ = value["occurrences"]
        cov = value["step_coverage"]
        wired = value["wiring"]["observation_wired"]
        plain = value["wiring"]["parameter_only"]
        lines.append(
            f"  {value['label']:52s} eligible={value['eligible']:3d} "
            f"exposed={value['exposed']:3d} ({value['exposure_share']:.1%}) "
            f"transferred={value['transferred']:3d} ({value['transfer_rate']:.1%} "
            f"[{value['transfer_ci95'][0]:.1%},{value['transfer_ci95'][1]:.1%}]) "
            f"faithful={value['faithful']:3d} ({value['faithful_rate']:.1%}) "
            f"occ={occ['exact']}/{occ['tested']} steps={cov['covered']}/{cov['steps']}"
            f" ({cov['share']:.1%})")
        lines.append(f"      replays: {occ['exact']}/{occ['tested']} exact, "
                     f"{occ['path_drift']} path drift, {occ['mismatched']} mismatch, "
                     f"{occ['unbound_slots']} unbound")
        wall = value["wiring"]["observation_wired_all_specs"]
        lines.append(f"      observation-wired (any spec): {wall['transferred']}/{wall['exposed']}"
                     f" exposed of {wall['eligible']}, "
                     f"occ={wall['occurrences']['exact']}/{wall['occurrences']['tested']}")
        lines.append(f"      observation-wired: {wired['transferred']}/{wired['exposed']} "
                     f"exposed of {wired['eligible']} eligible, "
                     f"occ={wired['occurrences']['exact']}/{wired['occurrences']['tested']}; "
                     f"parameter-only: {plain['transferred']}/{plain['exposed']} "
                     f"occ={plain['occurrences']['exact']}/{plain['occurrences']['tested']}; "
                     f"excluded-by-replay: {value['excluded_by_in_source_replay']['transferred']}"
                     f"/{value['excluded_by_in_source_replay']['exposed']} exposed")
    lines.append("matched-budget controls (learn on H1):")
    for name, value in report["controls"]["per_direction"].items():
        for key, cond in value["conditions"].items():
            lines.append(f"  {name:14s} {key:13s} exposed={cond['exposed']:3d} "
                         f"transferred={cond['transferred']:3d} ({cond['transfer_rate']:.1%}) "
                         f"faithful={cond['faithful']:3d} "
                         f"wired(all specs)="
                         f"{cond['wiring']['observation_wired_all_specs']['transferred']}"
                         f"/{cond['wiring']['observation_wired_all_specs']['exposed']} "
                         f"(occ {cond['wiring']['observation_wired_all_specs']['occurrences']['exact']}"
                         f"/{cond['wiring']['observation_wired_all_specs']['occurrences']['tested']})")
    lines.append("pooled control ladder (learning budget fixed at half the tasks):")
    for key, value in report["controls"]["pooled"].items():
        lines.append(f"  {key:13s} {value['transferred']:3d}/{value['exposed']:3d} "
                     f"({value['rate']:.1%} [{value['ci95'][0]:.1%},{value['ci95'][1]:.1%}]) "
                     f"faithful={value['faithful']} ({value['faithful_rate']:.1%}) "
                     f"param-only={value['parameter_only']['transferred']}"
                     f"/{value['parameter_only']['exposed']}")
    within = report["controls"]["pooled"].get("within_source")
    cells = report["cross_model"]["cells"]
    if within and cells:
        # the control varies the task instance; the cross-source cells vary the
        # model. Reuse is expected to survive the first at least as well as the
        # second, so the control is a reference point, not a ceiling.
        lines.append(f"within-source control {within['rate']:.1%} vs cross-source: " +
                     ", ".join(f"{name} {value['transfer_rate']:.1%}"
                               for name, value in cells.items()))
    domain = report["cross_domain"]
    if domain:
        lines.append(f"cross-domain retail->airline: shared templates "
                     f"{domain['shared_atomic_templates']}/"
                     f"{domain['learner_atomic_templates']}, expressible eligible "
                     f"{domain['eligible_expressible_in_tester']}/"
                     f"{domain['eligible_compounds']}, exposed "
                     f"{domain['eligible_exposed_in_tester']}, colliding task ids "
                     f"{domain['colliding_task_ids']}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args.corpus.resolve(), max_k=args.max_k, min_support=args.min_support)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarise(report))
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
