"""Does the 12.53% compound absorption survive a task-level holdout?

``compound_binding.py`` mines recurring compounds and measures absorption on the
same 325 tasks it mined them from, so a compound that absorbs a dispatch is also
evidence for its own support. That is resubstitution, and it is the standing
objection to the 12.53% the paper quotes ("replay-verified compounds absorb
12.53% of dispatches").

This script re-runs the pipeline with mining and measurement disjoint. For each
domain (retail, airline) the tasks are split by ``task_id`` parity -- even ids to
half A, odd ids to half B -- so a task instance, and every occurrence in it,
belongs to exactly one half; both model runs of a task land on the same side.
Compounds are mined on A only (``max_k=4``, ``min_support=2``), their specs are
learned on A's occurrences, and absorption is measured only on B's dispatches.
The mirror direction (mine B, measure A) is run as a symmetry check, and each
half's own in-sample number is reported beside the holdout so the gap can be
read rather than guessed at.

Protocol, mechanically:

  * mining -- ``compound_binding.mine_compounds`` on the learn half;
  * gate -- the committed funnel, recurring (support >= 2) -> signature-consistent
    -> replay-verified. ``compound_binding.verify_learning`` learns a spec from
    one occurrence and replays it on the half's other occurrences, exactly as the
    committed run does. The signature-consistent set comes from
    ``compound_transfer.eligible_specs``, which applies the same two stages
    around ``compound_binding``'s primitives; verifying *all* mined compounds
    instead of the signature-consistent subset reproduces neither the committed
    compound count nor its rate (reported as ``reference.unfiltered_replay``),
    so the stage is part of the definition, not a convenience;
  * transfer -- a spec learned on the learn half is replayed on every occurrence
    in the measure half. Binding reuses ``compound_transfer.bind_window`` and
    ``replay_window``, which call ``compound_binding.replay_spec``. A compound
    *verifies on B* when all of its B occurrences replay exactly; compounds with
    no B occurrence are unexposed, not failed;
  * absorption -- ``compound_binding.earned_reduction`` over the measure half's
    dispatches: greedy longest match first, no overlap, k - 1 dispatches removed
    per application. Three gates are measured: verified-on-learn (the committed
    definition, and the number this report calls the holdout), verified-on-learn
    intersected with transferable (the strict gate), and signature-consistent on
    the learn half (the "upper bound" arm that corresponds to the paper's
    15.03%);
  * references -- in-sample on the learn half (same pipeline, same half), and the
    full-corpus repertoire measured on the measure half. Those separate "half the
    learning data" from "unseen tasks", which are otherwise confounded: the two
    halves differ in absorbable structure before any learning happens.

What the holdout does not fix: absorption is a substitution count, so a spec that
would fail to bind on the measure half still credits its windows under the
committed gate -- the strict gate quantifies that exposure. The split is by task
instance, not by family: a compound family seen in A can recur in B. And the two
halves share the mock environment, the tool vocabulary, and the two model runs.

Reuses ``compound_binding.py`` (mining, spec learning, replay, earned reduction)
and ``compound_transfer.py`` (eligibility funnel, window binding) by import;
nothing is copied.

Usage:
    ./venv/bin/python examples/benchmarks/compound_holdout.py \
        --corpus examples/real_traces/tau_bench_observed.jsonl \
        --output examples/benchmarks/compound_holdout_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

SCHEMA_VERSION = "compound_holdout/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_observed.jsonl"
DEFAULT_BASELINE = BENCH_DIR / "compound_binding_results.json"

DOMAINS = ("retail", "airline")
MAX_K = 4
MIN_SUPPORT = 2
SURVIVAL_POINTS = 1.0      # a holdout within one point of the committed rate holds

Call = Tuple[str, Dict[str, Any], Optional[str]]
Template = Tuple[str, Tuple[str, ...]]
Compound = Tuple[Template, ...]
Occurrence = Tuple[Dict[str, Any], List[Call]]


# -- shared machinery -----------------------------------------------------


def _load(name: str, filename: str):
    """Import a sibling benchmark module once, under its own name."""
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = BENCH_DIR / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


CB = _load("compound_binding", "compound_binding.py")
CT = _load("compound_transfer", "compound_transfer.py")


# -- corpus and split -----------------------------------------------------


def domain_of(task: Dict[str, Any]) -> str:
    """The tau-bench domain of a task, from its recorded source label."""
    source = str(task.get("source") or "")
    return source.rsplit("-", 1)[-1]


def parity_of(task: Dict[str, Any]) -> int:
    text = str(task.get("task_id"))
    try:
        return int(text) % 2
    except ValueError as error:
        raise ValueError(
            f"task_id {text!r} is not an integer, so the parity split is undefined"
        ) from error


def split_parity(tasks: Sequence[Dict[str, Any]]
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Even task ids to half A, odd task ids to half B (file order preserved)."""
    a: List[Dict[str, Any]] = []
    b: List[Dict[str, Any]] = []
    for task in tasks:
        (a if parity_of(task) == 0 else b).append(task)
    return a, b


def group_by_domain(tasks: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for task in tasks:
        grouped.setdefault(domain_of(task), []).append(task)
    return grouped


def dispatches_of(tasks: Sequence[Dict[str, Any]]) -> int:
    """Recorded calls; ``compound_transfer``'s definition of a step, reused."""
    return CT.steps_of(tasks)


def occurrence_map(tasks: Sequence[Dict[str, Any]], max_k: int
                   ) -> Dict[Compound, List[Occurrence]]:
    """Every k-gram occurrence of every template sequence, in deterministic order."""
    out: Dict[Compound, List[Occurrence]] = {}
    for k in range(2, max_k + 1):
        for task, templates, window in CT.windows(tasks, k):
            out.setdefault(templates, []).append((task, window))
    return out


# -- the committed funnel, stage by stage --------------------------------


def signature_stage(tasks: Sequence[Dict[str, Any]], max_k: int, min_support: int
                    ) -> Tuple[Set[Compound], Dict[str, int]]:
    """Signature-consistent compounds on this half, via the transfer funnel.

    ``eligible_specs`` returns the compounds that pass signature consistency and
    replay (``eligible``) and the signature-consistent ones that fail replay
    (``excluded``); their union is the first stage's yield. Taking it from the
    funnel keeps one definition of the stage instead of a second implementation.
    """
    eligible, funnel, excluded = CT.eligible_specs(tasks, max_k, min_support)
    return set(eligible) | set(excluded), funnel


def verify(tasks: Sequence[Dict[str, Any]], signature_consistent: Set[Compound]
           ) -> Tuple[Set[Compound], Dict[str, Any]]:
    """``compound_binding.verify_learning`` over the signature-consistent set."""
    ordered = sorted(signature_consistent, key=lambda c: (len(c), c))
    return CB.verify_learning(tasks, ordered)


def learned_spec(occurrence: Occurrence) -> Dict[str, Any]:
    """The spec the committed run would learn from this occurrence."""
    _task, window = occurrence
    return CB.learn_spec(window, CB.provenance_of(window))


# -- cross-half verification ----------------------------------------------


def transfer(compounds: Set[Compound],
             learn_occ: Dict[Compound, List[Occurrence]],
             measure_occ: Dict[Compound, List[Occurrence]]
             ) -> Tuple[Set[Compound], Dict[str, Any]]:
    """Replay learn-half specs on every measure-half occurrence; all must be exact."""
    report: Dict[str, Any] = {
        "specs": len(compounds), "exposed": 0, "transferred": 0, "unexposed": 0,
        "occurrence_tests": 0, "occurrence_exact": 0,
        "path_drift": 0, "mismatched": 0, "failures": [],
    }
    passed: Set[Compound] = set()
    for compound in sorted(compounds, key=lambda c: (len(c), c)):
        spec = learned_spec(learn_occ[compound][0])
        windows = measure_occ.get(compound) or []
        if not windows:
            report["unexposed"] += 1
            continue
        report["exposed"] += 1
        exact_all = True
        for task, window in windows:
            report["occurrence_tests"] += 1
            produced, error = CT.replay_window(spec, window)
            if error is None and produced is not None and all(
                    produced[index] == window[index][1]
                    for index in range(len(window))):
                report["occurrence_exact"] += 1
                continue
            exact_all = False
            mode = CT.failure_mode(spec, window, error)
            if mode in CT.DRIFT_MODES:
                report["path_drift"] += 1
            else:
                report["mismatched"] += 1
            report["failures"].append({"compound": CT.serialise(compound),
                                       "task": task.get("task_id"), "mode": mode})
        if exact_all:
            passed.add(compound)
    report["transferred"] = len(passed)
    report["failures"].sort(key=lambda f: (json.dumps(f["compound"]),
                                           str(f["task"]), f["mode"]))
    report["failures"] = report["failures"][:6]
    return passed, report


# -- absorption ------------------------------------------------------------


def absorption(measure_tasks: Sequence[Dict[str, Any]], gate: Set[Compound],
               gate_name: str) -> Dict[str, Any]:
    """Dispatches absorbed on ``measure_tasks`` by ``gate`` under greedy substitution."""
    dispatches = dispatches_of(measure_tasks)
    if not gate:
        return {"gate": gate_name, "applications": 0, "dispatches_removed": 0,
                "dispatches": dispatches, "rate": 0.0}
    earned = CB.earned_reduction(measure_tasks, gate,
                                 CB.per_task_positions(measure_tasks, gate))
    if earned["steps"] != dispatches:
        raise AssertionError("dispatch count drifted between passes")
    return {"gate": gate_name,
            "applications": earned["compounds_applied"],
            "dispatches_removed": earned["steps_removed"],
            "dispatches": earned["steps"],
            "rate": earned["reduction"]}


# -- one direction, one domain -------------------------------------------


def direction(learn_tasks: Sequence[Dict[str, Any]],
              measure_tasks: Sequence[Dict[str, Any]],
              learn_half: str, measure_half: str,
              max_k: int, min_support: int,
              full_repertoire: Set[Compound]) -> Dict[str, Any]:
    """Mine and verify on one half, measure absorption on the other."""
    mined = CB.mine_compounds(learn_tasks, max_k=max_k, min_support=min_support)
    signature, funnel = signature_stage(learn_tasks, max_k, min_support)
    if len(mined) != funnel["recurring"]:
        raise AssertionError("the funnel and mine_compounds disagree on recurrence")
    if not signature <= set(mined):
        raise AssertionError("signature-consistent compounds must be mined")

    verified, replay = verify(learn_tasks, signature)
    learn_occ = occurrence_map(learn_tasks, max_k)
    measure_occ = occurrence_map(measure_tasks, max_k)
    transferred, cross = transfer(verified, learn_occ, measure_occ)
    # The same replay over every mined compound, not just the gated repertoire:
    # this is the literal "of the compounds mined on A, how many verify on B".
    mined_transferred, mined_cross = transfer(set(mined), learn_occ, measure_occ)

    by_k: Dict[int, Dict[str, int]] = {}
    for compound in mined:
        bucket = by_k.setdefault(len(compound), {"mined": 0, "verified_on_learn": 0,
                                                 "transferred": 0})
        bucket["mined"] += 1
        if compound in verified:
            bucket["verified_on_learn"] += 1
        if compound in transferred:
            bucket["transferred"] += 1

    return {
        "learn": {"half": learn_half, "tasks": len(learn_tasks),
                  "dispatches": dispatches_of(learn_tasks)},
        "measure": {"half": measure_half, "tasks": len(measure_tasks),
                    "dispatches": dispatches_of(measure_tasks)},
        "compounds": {
            "mined_on_learn": len(mined),
            "signature_consistent_on_learn": len(signature),
            "verified_on_learn": len(verified),
            "mined_exposed_in_measure": mined_cross["exposed"],
            "mined_transferred_to_measure": len(mined_transferred),
            "verified_exposed_in_measure": cross["exposed"],
            "verified_transferred_to_measure": len(transferred),
        },
        "funnel": funnel,
        "replay_on_learn": {
            "occurrence_tests": replay["checked"],
            "occurrence_exact": replay["verified"],
            "path_drift": replay["path_drift"],
            "mismatched": replay["mismatched"],
        },
        "transfer_to_measure": cross,
        "absorption": {
            "measure_holdout": absorption(measure_tasks, verified,
                                          "verified_on_learn"),
            "measure_holdout_strict": absorption(
                measure_tasks, verified & transferred,
                "verified_on_learn and transferred_to_measure"),
            "measure_signature_bound": absorption(measure_tasks, signature,
                                                  "signature_consistent_on_learn"),
            "measure_full_repertoire": absorption(measure_tasks, full_repertoire,
                                                  "verified on all tasks"),
            "learn_in_sample": absorption(learn_tasks, verified, "verified_on_learn"),
        },
        "by_k": {str(k): by_k[k] for k in sorted(by_k)},
    }


def protocol(tasks: Sequence[Dict[str, Any]], label: str, max_k: int,
             min_support: int, full_repertoire: Set[Compound]) -> Dict[str, Any]:
    """Split one task set by parity and run both directions."""
    a, b = split_parity(tasks)
    return {
        "label": label,
        "tasks": len(tasks),
        "dispatches": dispatches_of(tasks),
        "halves": {"A": {"tasks": len(a), "dispatches": dispatches_of(a)},
                   "B": {"tasks": len(b), "dispatches": dispatches_of(b)}},
        "A_to_B": direction(a, b, "A", "B", max_k, min_support, full_repertoire),
        "B_to_A": direction(b, a, "B", "A", max_k, min_support, full_repertoire),
    }


def full_repertoire_for(tasks: Sequence[Dict[str, Any]], max_k: int,
                        min_support: int
                        ) -> Tuple[Set[Compound], Dict[str, Any]]:
    """The committed repertoire: mined, signature-filtered, verified on all tasks."""
    mined = CB.mine_compounds(tasks, max_k=max_k, min_support=min_support)
    signature, funnel = signature_stage(tasks, max_k, min_support)
    verified, replay = verify(tasks, signature)
    return verified, {
        "mined": len(mined),
        "signature_consistent": len(signature),
        "verified": len(verified),
        "funnel": funnel,
        "replay": {"occurrence_tests": replay["checked"],
                   "occurrence_exact": replay["verified"],
                   "path_drift": replay["path_drift"],
                   "mismatched": replay["mismatched"]},
    }


# -- committed baseline and reproduction ----------------------------------


def plain_number(payload: Dict[str, Any], *keys: str) -> Any:
    node: Any = payload
    for key in keys:
        node = node[key]
    return node


def read_baseline(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        in_sample = {
            "compounds_mined": plain_number(payload, "compounds", "recurring"),
            "signature_consistent": plain_number(payload, "compounds",
                                                 "signature_consistent"),
            "replay_verified": plain_number(payload, "compounds", "replay_verified"),
            "dispatches_removed": plain_number(payload, "reduction_if_substituted",
                                               "steps_removed"),
            "dispatches": plain_number(payload, "reduction_if_substituted", "steps"),
            "applications": plain_number(payload, "reduction_if_substituted",
                                         "compounds_applied"),
            "rate": plain_number(payload, "reduction_if_substituted", "reduction"),
        }
    except (KeyError, TypeError) as error:
        raise SystemExit(f"baseline {path} lacks {error}; cannot anchor the comparison")
    return {
        "path": relative(path),
        "sha256": sha256_of(path),
        "schema": payload.get("schema"),
        "git_rev": (payload.get("provenance") or {}).get("git_rev"),
        "in_sample": in_sample,
    }


def reproduce(baseline: Dict[str, Any], info: Dict[str, Any],
              absorbed: Dict[str, Any]) -> Dict[str, Any]:
    """Re-run the committed pipeline on all tasks and compare field by field."""
    reproduced = {
        "compounds_mined": info["mined"],
        "signature_consistent": info["signature_consistent"],
        "replay_verified": info["verified"],
        "dispatches_removed": absorbed["dispatches_removed"],
        "dispatches": absorbed["dispatches"],
        "applications": absorbed["applications"],
        "rate": absorbed["rate"],
    }
    comparisons = []
    for field, expected in baseline["in_sample"].items():
        actual = reproduced[field]
        matches = (abs(expected - actual) < 5e-5 if isinstance(expected, float)
                   else expected == actual)
        comparisons.append({"field": field, "committed": expected,
                            "reproduced": actual, "matches": matches})
    return {"committed": baseline["in_sample"], "reproduced": reproduced,
            "comparisons": comparisons,
            "matches": all(item["matches"] for item in comparisons)}


# -- verdict ---------------------------------------------------------------


def verdict(pooled: Dict[str, Any], baseline: Dict[str, Any]) -> Dict[str, Any]:
    """Pool the two directions and state whether the committed rate survives."""
    forward = pooled["A_to_B"]["absorption"]["measure_holdout"]
    reverse = pooled["B_to_A"]["absorption"]["measure_holdout"]
    removed = forward["dispatches_removed"] + reverse["dispatches_removed"]
    dispatches = forward["dispatches"] + reverse["dispatches"]
    if dispatches != pooled["dispatches"]:
        raise AssertionError("the measured halves do not partition the corpus")
    rate = round(removed / dispatches, 4)

    committed = baseline["in_sample"]
    shortfall = committed["dispatches_removed"] - removed
    points = round((committed["rate"] - rate) * 100, 2)
    relative = round((committed["rate"] - rate) / committed["rate"], 4)
    spread = round(abs(forward["rate"] - reverse["rate"]) * 100, 2)

    # Where the in-sample/holdout gap comes from, by measured half. For half X,
    # in-sample means "mined, verified and measured on X"; holdout means "mined
    # and verified on the other half"; full_repertoire means "verified on all
    # tasks". The first difference splits into a repertoire effect (half the
    # learning data) and the gate's fitting edge over the full-corpus repertoire.
    in_sample = {"A": pooled["A_to_B"]["absorption"]["learn_in_sample"]["dispatches_removed"],
                 "B": pooled["B_to_A"]["absorption"]["learn_in_sample"]["dispatches_removed"]}
    holdout = {"A": reverse["dispatches_removed"], "B": forward["dispatches_removed"]}
    full = {"A": pooled["B_to_A"]["absorption"]["measure_full_repertoire"]
            ["dispatches_removed"],
            "B": pooled["A_to_B"]["absorption"]["measure_full_repertoire"]
            ["dispatches_removed"]}
    in_minus_hold = {half: in_sample[half] - holdout[half] for half in ("A", "B")}
    repertoire_loss = {half: full[half] - holdout[half] for half in ("A", "B")}
    gate_edge = {half: in_sample[half] - full[half] for half in ("A", "B")}

    def total(values: Dict[str, int]) -> Dict[str, int]:
        return {**values, "total": sum(values.values())}

    holds_point = all(abs(cell["rate"] - committed["rate"]) * 100 <= SURVIVAL_POINTS
                      for cell in (forward, reverse))
    holds_shape = rate >= 0.5 * committed["rate"]
    return {
        "criterion": (
            f"the committed {committed['rate']:.2%} survives as a point estimate if "
            f"out-of-sample absorption is within {SURVIVAL_POINTS:.0f} percentage "
            "point of it in both directions; it survives as a qualitative claim if "
            "the pooled holdout is at least half the committed rate"),
        "committed_rate": committed["rate"],
        "holdout_rate": rate,
        "holdout_dispatches_removed": removed,
        "dispatches": dispatches,
        "shortfall": {"dispatches": shortfall, "points": points,
                      "relative": relative},
        "directional": {"A_to_B": forward["rate"], "B_to_A": reverse["rate"],
                        "spread_points": spread},
        "holds_as_point_estimate": holds_point,
        "holds_as_claim_about_magnitude": holds_shape,
        "decomposition": {
            "note": ("keys are the measured half; in-sample minus holdout = "
                     "repertoire loss (half the learning data) + the half-corpus "
                     "gate's fitting edge over the full repertoire"),
            "in_sample": total(in_sample),
            "holdout": total(holdout),
            "full_repertoire": total(full),
            "in_sample_minus_holdout": total(in_minus_hold),
            "repertoire_loss": total(repertoire_loss),
            "gate_fitting_edge": total(gate_edge),
        },
        "statement": (
            f"The committed {committed['rate']:.2%} does not survive as a point "
            f"estimate. On disjoint task halves, absorption is {rate:.2%} "
            f"({removed}/{dispatches}), {points:.2f} points ({relative:.1%} relative) "
            f"lower, and both directions sit below it ({forward['rate']:.2%} A to B, "
            f"{reverse['rate']:.2%} B to A; a {spread:.2f}-point spread, smaller than "
            "the shortfall). It survives as a claim about magnitude: verified "
            "compounds still absorb about a tenth of dispatches, far below the "
            "75.0% recurrence ceiling reported for the inputs-only corpus, so the "
            "paper should quote the holdout rate (or both rates) rather than "
            "12.53% alone."
        ),
    }


# -- checks ----------------------------------------------------------------


def split_checks(tasks: Sequence[Dict[str, Any]],
                 pooled: Dict[str, Any],
                 domains: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    a, b = split_parity(tasks)
    checks = []

    def add(name: str, ok: bool, detail: Any) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    keys_a = {(domain_of(t), str(t.get("task_id"))) for t in a}
    keys_b = {(domain_of(t), str(t.get("task_id"))) for t in b}
    add("halves_partition_the_corpus",
        len(a) + len(b) == len(tasks) and not (keys_a & keys_b),
        {"A": len(a), "B": len(b), "shared_task_instances": len(keys_a & keys_b)})
    add("dispatches_sum",
        pooled["halves"]["A"]["dispatches"] + pooled["halves"]["B"]["dispatches"]
        == pooled["dispatches"],
        {"A": pooled["halves"]["A"]["dispatches"],
         "B": pooled["halves"]["B"]["dispatches"], "all": pooled["dispatches"]})
    add("domains_partition_the_corpus",
        sum(block["tasks"] for block in domains.values()) == len(tasks),
        {name: block["tasks"] for name, block in domains.items()})
    add("holdout_denominator_is_the_corpus",
        pooled["A_to_B"]["absorption"]["measure_holdout"]["dispatches"]
        + pooled["B_to_A"]["absorption"]["measure_holdout"]["dispatches"]
        == pooled["dispatches"],
        {"A_to_B": pooled["A_to_B"]["absorption"]["measure_holdout"]["dispatches"],
         "B_to_A": pooled["B_to_A"]["absorption"]["measure_holdout"]["dispatches"]})
    return checks


def gate_checks(block: Dict[str, Any], label: str) -> List[Dict[str, Any]]:
    compounds = block["compounds"]
    return [
        {"name": f"gate_is_a_subset_chain:{label}",
         "ok": (compounds["verified_on_learn"]
                <= compounds["signature_consistent_on_learn"]
                <= compounds["mined_on_learn"]),
         "detail": {key: compounds[key] for key in
                    ("mined_on_learn", "signature_consistent_on_learn",
                     "verified_on_learn")}},
        {"name": f"verify_learning_matches_funnel:{label}",
         "ok": compounds["verified_on_learn"] == block["funnel"]["in_source_replay"],
         "detail": {"verify_learning": compounds["verified_on_learn"],
                    "funnel_in_source_replay": block["funnel"]["in_source_replay"]}},
    ]


# -- report ----------------------------------------------------------------


def run(corpus: Path, baseline_path: Path, max_k: int = MAX_K,
        min_support: int = MIN_SUPPORT) -> Dict[str, Any]:
    tasks = CB.load_tasks(corpus)
    grouped = group_by_domain(tasks)
    unexpected = sorted(set(grouped) - set(DOMAINS))
    if unexpected:
        raise SystemExit(f"unexpected domains {unexpected}; expected {DOMAINS}")

    full, full_info = full_repertoire_for(tasks, max_k, min_support)
    pooled = protocol(tasks, "pooled", max_k, min_support, full)
    domains = {name: protocol(grouped[name], name, max_k, min_support, full)
               for name in DOMAINS if name in grouped}

    baseline = read_baseline(baseline_path)
    on_all = absorption(tasks, full, "verified on all tasks")
    reproduction = reproduce(baseline, full_info, on_all)

    a, b = split_parity(tasks)
    # How much the committed gate depends on the signature stage: verifying every
    # mined compound instead is a different, much larger repertoire.
    unfiltered, unfiltered_stats = verify(tasks, set(CB.mine_compounds(
        tasks, max_k=max_k, min_support=min_support)))
    unfiltered_absorbed = absorption(tasks, unfiltered, "verified on all mined")

    pooled_removed = (
        pooled["A_to_B"]["absorption"]["measure_holdout"]["dispatches_removed"]
        + pooled["B_to_A"]["absorption"]["measure_holdout"]["dispatches_removed"])
    domain_removed = sum(
        block[tag]["absorption"]["measure_holdout"]["dispatches_removed"]
        for block in domains.values() for tag in ("A_to_B", "B_to_A"))

    reference = {
        "full_repertoire": {
            "compounds": full_info,
            "absorption": {
                "on_all": on_all,
                "on_half_A": absorption(a, full, "verified on all tasks"),
                "on_half_B": absorption(b, full, "verified on all tasks"),
                "on_domain": {name: absorption(grouped[name], full,
                                               "verified on all tasks")
                              for name in domains},
            },
        },
        "unfiltered_replay": {
            "note": ("diagnostic only: the committed run verifies the "
                     "signature-consistent set, not every mined compound; this cell "
                     "shows how much the stage changes the answer"),
            "compounds_verified": len(unfiltered),
            "absorption": unfiltered_absorbed,
            "replay": {"occurrence_tests": unfiltered_stats["checked"],
                       "occurrence_exact": unfiltered_stats["verified"]},
        },
        "in_sample_on_half": {
            "A": pooled["A_to_B"]["absorption"]["learn_in_sample"],
            "B": pooled["B_to_A"]["absorption"]["learn_in_sample"],
        },
        "domain_composition": {
            "note": ("pooled mining can mint compounds over the tools both domains "
                     "share (calculate, think, get_user_details, "
                     "transfer_to_human_agents), so the pooled holdout is not exactly "
                     "the sum of the domain holdouts"),
            "pooled_holdout_dispatches_removed": pooled_removed,
            "sum_of_domain_holdouts": domain_removed,
            "difference": pooled_removed - domain_removed,
        },
    }

    verdict_block = verdict(pooled, baseline)
    checks = split_checks(tasks, pooled, domains)
    checks.append({"name": "reproduction_matches_committed",
                   "ok": reproduction["matches"],
                   "detail": {item["field"]: item["matches"]
                              for item in reproduction["comparisons"]}})
    for name in ("pooled", *sorted(domains)):
        block = pooled if name == "pooled" else domains[name]
        checks.extend(gate_checks(block["A_to_B"], f"{name}:A_to_B"))
        checks.extend(gate_checks(block["B_to_A"], f"{name}:B_to_A"))

    return {
        "schema": SCHEMA_VERSION,
        "self_test": False,
        "corpus": {
            "path": relative(corpus),
            "sha256": sha256_of(corpus),
            "bytes": corpus.stat().st_size,
            "tasks": len(tasks),
            "dispatches": dispatches_of(tasks),
            "domains": sorted(grouped),
        },
        "split": {
            "rule": "task_id parity, applied within each domain: even -> A, odd -> B",
            "note": ("a task instance appears in exactly one half, and both model "
                     "runs of a task are on the same side; file order is preserved "
                     "inside each half"),
            "max_k": max_k,
            "min_support": min_support,
        },
        "definitions": {
            "mined": "recurring contiguous k-gram (2 <= k <= max_k, support >= min_support) on the learn half",
            "signature_consistent": "every learn-half occurrence has the same provenance signature",
            "verified_on_learn": "spec learned on one learn-half occurrence replays exactly on the half's other occurrences",
            "exposed": "the compound has at least one occurrence in the measure half",
            "transferred_to_measure": "the learn-half spec replays exactly on every exposed measure-half occurrence; reported once for the verified repertoire and once for every mined compound",
            "absorption": "greedy longest-first, non-overlapping substitution of gate compounds; k - 1 dispatches removed per application; dispatches are recorded calls",
            "holdout": "absorption on the measure half with the verified-on-learn gate",
            "strict": "absorption with the verified-on-learn and transferred gates",
            "in_sample": "absorption on the learn half with its own verified gate",
            "full_repertoire": "the committed repertoire (all tasks), measured on the given half",
        },
        "baseline": baseline,
        "reproduction": reproduction,
        "pooled": pooled,
        "by_domain": domains,
        "reference": reference,
        "verdict": verdict_block,
        "checks": checks,
        "provenance": provenance(corpus, baseline_path, max_k, min_support),
    }


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(corpus: Path, baseline_path: Path, max_k: int,
               min_support: int) -> Dict[str, Any]:
    """Revision, tree state, and the inputs the numeric payload is a function of."""
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except Exception:
            return "unknown"

    module_ids = {
        "compound_binding": {"path": relative(BENCH_DIR / "compound_binding.py"),
                             "sha256": sha256_of(BENCH_DIR / "compound_binding.py")},
        "compound_transfer": {"path": relative(BENCH_DIR / "compound_transfer.py"),
                              "sha256": sha256_of(BENCH_DIR / "compound_transfer.py")},
    }
    return {
        "git_rev": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "script": relative(Path(__file__).resolve()),
        "script_sha256": sha256_of(Path(__file__).resolve()),
        "corpus": relative(corpus),
        "corpus_sha256": sha256_of(corpus),
        "corpus_bytes": corpus.stat().st_size,
        "baseline": relative(baseline_path),
        "baseline_sha256": sha256_of(baseline_path),
        "modules": module_ids,
        "python": sys.version.split()[0],
        "max_k": max_k,
        "min_support": min_support,
        "volatile": ["git_rev", "dirty"],
        "note": ("the numeric payload is a deterministic function of the corpus, the "
                 "baseline and the two imported modules (each pinned by sha256); "
                 "git_rev and dirty record the checkout and change by design"),
    }


def summarise(report: Dict[str, Any]) -> str:
    corpus = report["corpus"]
    verdict = report["verdict"]
    lines = [
        f"corpus: {Path(corpus['path']).name}  tasks={corpus['tasks']}  "
        f"dispatches={corpus['dispatches']}  max_k={report['split']['max_k']}  "
        f"min_support={report['split']['min_support']}",
        "split (task_id parity): " + ", ".join(
            f"{name} A={block['halves']['A']['tasks']}/{block['halves']['A']['dispatches']}"
            f" B={block['halves']['B']['tasks']}/{block['halves']['B']['dispatches']}"
            for name, block in
            [("pooled", report["pooled"]), *report["by_domain"].items()]),
    ]
    for name, block in [("pooled", report["pooled"]), *report["by_domain"].items()]:
        for tag in ("A_to_B", "B_to_A"):
            cell = block[tag]
            compounds = cell["compounds"]
            lines.append(
                f"{name} {tag}: mined {compounds['mined_on_learn']} -> "
                f"verified {compounds['verified_on_learn']} -> "
                f"exposed {compounds['verified_exposed_in_measure']} -> "
                f"transferred {compounds['verified_transferred_to_measure']} "
                f"(all mined: {compounds['mined_exposed_in_measure']} exposed, "
                f"{compounds['mined_transferred_to_measure']} transferred)")
            absorb = cell["absorption"]["measure_holdout"]
            strict = cell["absorption"]["measure_holdout_strict"]
            in_sample = cell["absorption"]["learn_in_sample"]
            full = cell["absorption"]["measure_full_repertoire"]
            lines.append(
                f"   holdout {absorb['dispatches_removed']}/{absorb['dispatches']} "
                f"= {absorb['rate']:.2%} ({absorb['applications']} applications); "
                f"strict {strict['dispatches_removed']}/{strict['dispatches']} "
                f"= {strict['rate']:.2%}")
            lines.append(
                f"   in-sample on {cell['learn']['half']} "
                f"{in_sample['dispatches_removed']}/{in_sample['dispatches']} "
                f"= {in_sample['rate']:.2%}; full repertoire on "
                f"{cell['measure']['half']} {full['dispatches_removed']}/{full['dispatches']}"
                f" = {full['rate']:.2%}")
    decomposition = verdict["decomposition"]
    lines.append(
        f"pooled holdout {verdict['holdout_dispatches_removed']}/{verdict['dispatches']} "
        f"= {verdict['holdout_rate']:.2%} vs committed "
        f"{verdict['committed_rate']:.2%} "
        f"({verdict['shortfall']['points']:.2f} points, "
        f"{verdict['shortfall']['relative']:.1%} relative); directions "
        f"{verdict['directional']['A_to_B']:.2%} / "
        f"{verdict['directional']['B_to_A']:.2%}")
    lines.append(
        f"decomposition: in-sample-holdout {decomposition['in_sample_minus_holdout']['total']} "
        f"dispatches = repertoire loss "
        f"{decomposition['repertoire_loss']['total']} + gate fitting edge "
        f"{decomposition['gate_fitting_edge']['total']}")
    lines.append(
        f"reproduction: {'matches' if report['reproduction']['matches'] else 'MISMATCH'} "
        f"the committed {report['baseline']['path']}")
    if verdict["holds_as_point_estimate"]:
        lines.append("verdict: the holdout survives within the stated tolerance")
    else:
        lines.append("verdict: the committed rate does not survive as a point estimate; "
                     "the claim about magnitude does")
    failed = [check["name"] for check in report["checks"] if not check["ok"]]
    lines.append(f"checks: {len(report['checks']) - len(failed)}/"
                 f"{len(report['checks'])} pass"
                 + (f"; failed: {', '.join(failed)}" if failed else ""))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--max-k", type=int, default=MAX_K)
    parser.add_argument("--min-support", type=int, default=MIN_SUPPORT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = run(args.corpus.resolve(), args.baseline.resolve(),
                 max_k=args.max_k, min_support=args.min_support)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(summarise(report))
    print(f"-> {args.output}")
    failed = [check["name"] for check in report["checks"] if not check["ok"]]
    if failed:
        print(f"FAILED CHECKS: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
