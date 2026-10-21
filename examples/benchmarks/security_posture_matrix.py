"""Functional fail-open vs fail-closed security-posture matrix (E5).

Measures the *functional* security posture of the AgentISA coordination
layer (``agentflow.core.isa.coordination``) as a scenario x enforcement-
mode matrix, in the spirit of Section 6.2 of the AgentISA paper:

    scenarios : unregistered resource / unknown data sensitivity /
                incomplete resource-access policy / normal task
                (+ one supplementary row: type-malformed registry)
    modes     : fail-open  (``strict=False``, the historical default)
                fail-closed (``strict=True``, the I4 hardening flag)

What this measures and what it does not
---------------------------------------
* Every number in the output JSON is computed by this script from raw
  per-task decision records (allow/deny + recorded violation reason).
* The study is qualitative + counting ONLY: no latency, throughput,
  cost or any other performance quantity is measured or claimed.
* The decision function under test is deterministic, so per-seed rate
  stdev is expected to be 0.0; the seeds vary the task *surface*
  (resource names, request shapes, sub-variant mix), not the verdict.
* Each scenario holds the "other" security dimension at a known-good
  value so exactly one dimension varies per scenario (mirroring
  ``tests/core/isa/test_security_failclosed.py``).

Usage
-----
    python security_posture_matrix.py                       # defaults
    python security_posture_matrix.py --seeds 5 --tasks-per-scenario 40
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.capability import CapabilityProfile, DomainScore
from agentflow.core.isa.coordination import (
    AgentCoordinator,
    CoordinationRole,
    SecurityError,
    SecurityLevel,
    SecurityMonitor,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODES: Tuple[str, ...] = ("fail_open", "fail_closed")
STRICT_BY_MODE = {"fail_open": False, "fail_closed": True}

REGISTERED_POOL = [
    "gpu_cluster", "s3_bucket", "vector_store", "redis_cache", "kafka_stream",
]
UNREGISTERED_POOL = [
    "shadow_bucket", "rogue_endpoint", "unregistered_gpu",
    "legacy_tape_archive", "test_cluster",
]

MAX_DISTINCT_MESSAGES_PER_CELL = 100

TaskSpec = Tuple[Dict[str, Any], Dict[str, Any], SecurityLevel]
VariantGen = Callable[[random.Random], TaskSpec]


# ---------------------------------------------------------------------------
# Task / agent construction helpers
# ---------------------------------------------------------------------------

def _base_task(task_id: str, **overrides) -> Dict[str, Any]:
    """A task payload that satisfies the always-on auth checks."""
    task = {
        "id": task_id,
        "authentication": "token-bench",
        "authorization": "role:executor",
        "requirements": {"capabilities": {}},
    }
    task.update(overrides)
    return task


def make_profile(reasoning: float = 0.5, optimization: float = 0.5,
                 memory: float = 0.5, specialization: float = 0.5,
                 specializations: Optional[List[str]] = None) -> CapabilityProfile:
    """Build a CapabilityProfile with explicit per-domain scores."""
    profile = CapabilityProfile(
        {"specializations": specializations or ["general"]}
    )
    profile.capabilities = {
        "reasoning": DomainScore(score=reasoning, weight=0.3),
        "optimization": DomainScore(score=optimization, weight=0.3),
        "memory": DomainScore(score=memory, weight=0.2),
        "specialization": DomainScore(score=specialization, weight=0.2),
    }
    return profile


def make_coordinator(config: Dict[str, Any], strict: bool) -> AgentCoordinator:
    """Coordinator in the requested mode with three registered agents."""
    coord = AgentCoordinator(config, security_strict=strict)
    coord.register_agent(
        "coordinator",
        make_profile(reasoning=0.9, optimization=0.9),
        CoordinationRole.COORDINATOR,
    )
    coord.register_agent(
        "executor",
        make_profile(reasoning=0.5, optimization=0.7),
        CoordinationRole.EXECUTOR,
    )
    coord.register_agent(
        "validator",
        make_profile(reasoning=0.95, optimization=0.4),
        CoordinationRole.VALIDATOR,
    )
    return coord


# ---------------------------------------------------------------------------
# Scenario registry: sub-variant generators
# ---------------------------------------------------------------------------
# Each generator returns (monitor/coordinator config, task, required_level).
# The scenario-defining property is fixed; seeds randomise the surface.

def _gen_single_unregistered(rng: random.Random) -> TaskSpec:
    registry = set(rng.sample(REGISTERED_POOL, k=3))
    unreg = rng.choice(UNREGISTERED_POOL)
    config = {"allowed_resources": registry}
    task = _base_task("t", resources=[unreg], data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_mixed_registered_unregistered(rng: random.Random) -> TaskSpec:
    registry = set(rng.sample(REGISTERED_POOL, k=2))
    reg = rng.choice(sorted(registry))
    unreg = rng.choice(UNREGISTERED_POOL)
    config = {"allowed_resources": registry}
    task = _base_task("t", resources=[reg, unreg], data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_string_form_unregistered(rng: random.Random) -> TaskSpec:
    registry = set(rng.sample(REGISTERED_POOL, k=3))
    unreg = rng.choice(UNREGISTERED_POOL)
    config = {"allowed_resources": registry}
    task = _base_task("t", resources=unreg, data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_registry_as_list(rng: random.Random) -> TaskSpec:
    registry = rng.sample(REGISTERED_POOL, k=3)
    unreg = rng.choice(UNREGISTERED_POOL)
    config = {"allowed_resources": registry}
    task = _base_task("t", resources=[unreg], data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_registry_key_absent(rng: random.Random) -> TaskSpec:
    res = rng.choice(REGISTERED_POOL)
    task = _base_task("t", resources=[res], data_sensitivity=0)
    return {}, task, SecurityLevel.LOW


def _gen_registry_explicitly_none(rng: random.Random) -> TaskSpec:
    res = rng.choice(REGISTERED_POOL)
    config = {"allowed_resources": None}
    task = _base_task("t", resources=[res], data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_multi_resource_no_registry(rng: random.Random) -> TaskSpec:
    res = rng.sample(REGISTERED_POOL, k=rng.randint(2, 3))
    task = _base_task("t", resources=res, data_sensitivity=0)
    return {}, task, SecurityLevel.LOW


def _gen_string_resource_no_registry(rng: random.Random) -> TaskSpec:
    res = rng.choice(REGISTERED_POOL)
    task = _base_task("t", resources=res, data_sensitivity=0)
    return {}, task, SecurityLevel.LOW


def _gen_no_sensitivity_declaration(rng: random.Random) -> TaskSpec:
    return {}, _base_task("t"), SecurityLevel.LOW


def _gen_empty_category_list(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_categories=[])
    return {}, task, SecurityLevel.LOW


def _gen_non_numeric_declaration(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_sensitivity="high")
    return {}, task, SecurityLevel.LOW


def _gen_negative_out_of_range(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_sensitivity=-1)
    return {}, task, SecurityLevel.LOW


def _gen_no_resources_zero_sensitivity(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_sensitivity=0)
    return {}, task, SecurityLevel.LOW


def _gen_registered_resources_only(rng: random.Random) -> TaskSpec:
    registry = rng.sample(REGISTERED_POOL, k=3)
    requested = rng.sample(registry, k=rng.randint(1, 3))
    config = {"allowed_resources": set(registry)}
    task = _base_task("t", resources=requested, data_sensitivity=0)
    return config, task, SecurityLevel.LOW


def _gen_declared_medium_within_requirement(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_sensitivity=1)
    return {}, task, SecurityLevel.MEDIUM


def _gen_benign_categories_inferred_medium(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_categories=["public_reports"])
    return {}, task, SecurityLevel.MEDIUM


def _gen_sensitive_category_within_high(rng: random.Random) -> TaskSpec:
    task = _base_task("t", data_categories=["financial"])
    return {}, task, SecurityLevel.HIGH


def _gen_string_registry_charset(rng: random.Random) -> TaskSpec:
    # A type-malformed registry: set("gpu_cluster") degenerates to a
    # character set, so no real resource name can match it.
    config = {"allowed_resources": "gpu_cluster"}
    res = rng.choice(REGISTERED_POOL)
    task = _base_task("t", resources=[res], data_sensitivity=0)
    return config, task, SecurityLevel.LOW


SCENARIOS: List[Dict[str, Any]] = [
    {
        "scenario": "unregistered_resource",
        "label": ("Task requests a resource outside the declared "
                  "allowed_resources registry"),
        "matrix_role": "main",
        "variants": {
            "single_unregistered": _gen_single_unregistered,
            "mixed_registered_and_unregistered":
                _gen_mixed_registered_unregistered,
            "string_form_request": _gen_string_form_unregistered,
            "registry_declared_as_list": _gen_registry_as_list,
        },
        "expected_posture": ("both modes deny; fail-closed adds a "
                             "distinguishing 'Unregistered resources' "
                             "reason with the (strict mode) marker"),
    },
    {
        "scenario": "unknown_sensitivity",
        "label": ("Task's data sensitivity cannot be positively determined "
                  "(undeclared / empty categories / non-numeric / "
                  "out-of-range low)"),
        "matrix_role": "main",
        "variants": {
            "no_declaration": _gen_no_sensitivity_declaration,
            "empty_category_list": _gen_empty_category_list,
            "non_numeric_declaration": _gen_non_numeric_declaration,
            "negative_out_of_range": _gen_negative_out_of_range,
        },
        "expected_posture": ("fail-open silently treats unknown as 0 and "
                             "permits; fail-closed denies with an "
                             "'Unknown data sensitivity' reason"),
    },
    {
        "scenario": "incomplete_policy",
        "label": ("Task requests resources but the coordinator declares no "
                  "allowed_resources registry at all"),
        "matrix_role": "main",
        "variants": {
            "registry_key_absent": _gen_registry_key_absent,
            "registry_explicitly_none": _gen_registry_explicitly_none,
            "multi_resource_request_no_registry":
                _gen_multi_resource_no_registry,
            "string_resource_no_registry": _gen_string_resource_no_registry,
        },
        "expected_posture": ("fail-open permits (the open default); "
                             "fail-closed denies with an 'Incomplete "
                             "resource-access policy' reason"),
    },
    {
        "scenario": "normal_task",
        "label": ("Fully declared, compliant task (registered resources / "
                  "declared or inferable sensitivity within requirement)"),
        "matrix_role": "main",
        "variants": {
            "no_resources_zero_sensitivity":
                _gen_no_resources_zero_sensitivity,
            "registered_resources_only": _gen_registered_resources_only,
            "declared_medium_within_requirement":
                _gen_declared_medium_within_requirement,
            "benign_categories_inferred_medium":
                _gen_benign_categories_inferred_medium,
            "sensitive_category_within_high_requirement":
                _gen_sensitive_category_within_high,
        },
        "expected_posture": ("both modes permit with zero recorded "
                             "violations (no false positives)"),
    },
    {
        "scenario": "malformed_registry_supplementary",
        "label": ("Supplementary row: allowed_resources declared with a "
                  "malformed type (a plain string degenerates to a "
                  "character set)"),
        "matrix_role": "supplementary",
        "variants": {
            "string_registry_degenerates_to_charset":
                _gen_string_registry_charset,
        },
        "expected_posture": ("both modes deny, but for different stated "
                             "reasons; only fail-closed names the "
                             "incomplete policy"),
    },
]

REQUIRED_SCENARIO_IDS = [
    "unregistered_resource", "unknown_sensitivity",
    "incomplete_policy", "normal_task",
]


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def classify_reason(allowed: bool, message: Optional[str]) -> str:
    """Map a decision to a coarse reason class (order matters)."""
    if allowed:
        return "allowed"
    text = (message or "").lower()
    if "unregistered" in text:
        return "unregistered_resources_strict"
    if "incomplete" in text:
        return "incomplete_policy_strict"
    if "unknown data sensitivity" in text:
        return "unknown_sensitivity_strict"
    if "invalid resource access" in text:
        return "generic_invalid_resource_access"
    if "sensitivity mismatch" in text:
        return "sensitivity_mismatch"
    if "missing authentication" in text:
        return "missing_authentication"
    if "missing authorization" in text:
        return "missing_authorization"
    return "other"


def build_tasks(seeds: List[int], n_tasks: int) -> List[Dict[str, Any]]:
    """Generate the deterministic task population.

    One task per (seed, scenario, index); the SAME task dict is then
    evaluated in both enforcement modes, so per-task outcomes can be
    paired for the mode-divergence analysis.
    """
    population = []
    for scenario in SCENARIOS:
        sid = scenario["scenario"]
        variant_names = sorted(scenario["variants"])
        for seed in seeds:
            for index in range(n_tasks):
                rng = random.Random(f"{seed}|{sid}|{index}")
                variant = rng.choice(variant_names)
                config, task, required = scenario["variants"][variant](rng)
                task = dict(task)
                task["id"] = f"{sid}-s{seed}-i{index}"
                population.append({
                    "seed": seed,
                    "scenario": sid,
                    "variant": variant,
                    "index": index,
                    "config": config,
                    "task": task,
                    "required_level": required.name,
                })
    return population


def run_monitor_matrix(population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Evaluate every task through SecurityMonitor in both modes.

    A fresh monitor per (task, mode) so no state carries over.
    """
    raw = []
    for item in population:
        required = SecurityLevel[item["required_level"]]
        for mode in MODES:
            monitor = SecurityMonitor(item["config"], strict=STRICT_BY_MODE[mode])
            allowed = monitor.validate_task(item["task"], required)
            history = monitor.violation_history
            message = history[-1]["violation"] if history else None
            raw.append({
                "scenario": item["scenario"],
                "variant": item["variant"],
                "mode": mode,
                "seed": item["seed"],
                "index": item["index"],
                "allowed": bool(allowed),
                "reason_class": classify_reason(bool(allowed), message),
                "violation_message": message,
                "strict_marker": bool(allowed is False and message
                                      and "(strict mode)" in message),
                "violation_recorded": bool((not allowed) and len(history) == 1),
                "required_level": item["required_level"],
            })
    return raw


def run_coordinator_matrix(population: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Push the same tasks through AgentCoordinator.coordinate_task.

    A fresh coordinator (with three registered agents) per (task, mode).
    Records whether the task executed end-to-end or was stopped at the
    security gate with a recorded violation.
    """
    raw = []
    for item in population:
        required = SecurityLevel[item["required_level"]]
        for mode in MODES:
            coord = make_coordinator(item["config"], STRICT_BY_MODE[mode])
            try:
                coord.coordinate_task(item["task"], required)
                outcome = "executed"
            except SecurityError:
                outcome = "security_error"
            raw.append({
                "scenario": item["scenario"],
                "variant": item["variant"],
                "mode": mode,
                "seed": item["seed"],
                "index": item["index"],
                "outcome": outcome,
                "violation_recorded": bool(
                    len(coord.security_monitor.violation_history) == 1
                ),
            })
    return raw


# ---------------------------------------------------------------------------
# Aggregation (all aggregates are computed from the raw records above)
# ---------------------------------------------------------------------------

def _rate(part: int, whole: int) -> float:
    return (part / whole) if whole else 0.0


def _mean_stdev(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {"mean": 0.0, "stdev": 0.0, "values": []}
    return {
        "mean": statistics.mean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "values": values,
    }


def _cell_note(fail_open_allow: float, strict_allow: float) -> str:
    if fail_open_allow == 1.0 and strict_allow == 0.0:
        return ("mode flip: fail-open permits the whole population, "
                "fail-closed blocks it entirely")
    if fail_open_allow == 0.0 and strict_allow == 0.0:
        return ("both modes deny the whole population; the recorded "
                "reason classes differ between modes")
    if fail_open_allow == 1.0 and strict_allow == 1.0:
        return ("both modes permit the whole population (no false "
                "positives introduced by fail-closed)")
    return ("mixed outcomes across sub-variants; see per_variant and "
            "raw_counts for the split")


def aggregate_matrix(raw: List[Dict[str, Any]],
                     seeds: List[int]) -> List[Dict[str, Any]]:
    """Per scenario x mode cell, computed only from raw records."""
    cells = []
    for scenario in SCENARIOS:
        sid = scenario["scenario"]
        for mode in MODES:
            recs = [r for r in raw if r["scenario"] == sid and r["mode"] == mode]
            n = len(recs)
            n_allowed = sum(1 for r in recs if r["allowed"])
            denied = [r for r in recs if not r["allowed"]]

            per_variant = []
            for variant in sorted(scenario["variants"]):
                vrecs = [r for r in recs if r["variant"] == variant]
                v_allowed = sum(1 for r in vrecs if r["allowed"])
                per_variant.append({
                    "variant": variant,
                    "n": len(vrecs),
                    "allowed": v_allowed,
                    "denied": len(vrecs) - v_allowed,
                })

            per_seed_rates = []
            for seed in seeds:
                srecs = [r for r in recs if r["seed"] == seed]
                per_seed_rates.append(
                    _rate(sum(1 for r in srecs if r["allowed"]), len(srecs))
                    if srecs else 0.0
                )

            message_counts = Counter(r["violation_message"] for r in denied)
            distinct = [
                {"message": msg, "count": cnt}
                for msg, cnt in sorted(message_counts.items(),
                                       key=lambda kv: (-kv[1], kv[0]))
            ]
            if len(distinct) > MAX_DISTINCT_MESSAGES_PER_CELL:
                distinct = distinct[:MAX_DISTINCT_MESSAGES_PER_CELL]
                distinct.append({"message": "... (truncated)", "count": -1})

            cells.append({
                "scenario": sid,
                "scenario_label": scenario["label"],
                "matrix_role": scenario["matrix_role"],
                "expected_posture": scenario["expected_posture"],
                "mode": mode,
                "strict_flag": STRICT_BY_MODE[mode],
                "n_tasks": n,
                "allowed": n_allowed,
                "denied": n - n_allowed,
                "allow_rate": _rate(n_allowed, n),
                "deny_rate": _rate(n - n_allowed, n),
                "allow_rate_per_seed": _mean_stdev(per_seed_rates),
                "per_variant": per_variant,
                "reason_class_counts": dict(Counter(
                    r["reason_class"] for r in recs
                )),
                "distinct_violation_messages": distinct,
                "denied_with_strict_marker": sum(
                    1 for r in denied if r["strict_marker"]),
                "denied_with_recorded_violation": sum(
                    1 for r in denied if r["violation_recorded"]),
                "strict_marker_rate_among_denied": _rate(
                    sum(1 for r in denied if r["strict_marker"]), len(denied)
                ),
                "violation_recorded_rate_among_denied": _rate(
                    sum(1 for r in denied if r["violation_recorded"]),
                    len(denied),
                ),
            })

    # Attach the cross-mode note now that both cells exist.
    by_key = {(c["scenario"], c["mode"]): c for c in cells}
    for scenario in SCENARIOS:
        sid = scenario["scenario"]
        fo = by_key[(sid, "fail_open")]["allow_rate"]
        fc = by_key[(sid, "fail_closed")]["allow_rate"]
        note = _cell_note(fo, fc)
        by_key[(sid, "fail_open")]["observed_note"] = note
        by_key[(sid, "fail_closed")]["observed_note"] = note
    return cells


def aggregate_mode_divergence(raw: List[Dict[str, Any]],
                              seeds: List[int],
                              n_tasks: int) -> List[Dict[str, Any]]:
    """Pair the identical task through both modes (2x2 contingency)."""
    lookup = {(r["scenario"], r["mode"], r["seed"], r["index"]): r
              for r in raw}
    out = []
    for scenario in SCENARIOS:
        sid = scenario["scenario"]
        contingency = {"both_allow": 0, "fail_open_only": 0,
                       "fail_closed_only": 0, "both_deny": 0}
        reason_agree = 0
        both_deny = 0
        for seed in seeds:
            for index in range(n_tasks):
                fo = lookup[(sid, "fail_open", seed, index)]
                fc = lookup[(sid, "fail_closed", seed, index)]
                if fo["allowed"] and fc["allowed"]:
                    contingency["both_allow"] += 1
                elif fo["allowed"] and not fc["allowed"]:
                    contingency["fail_open_only"] += 1
                elif (not fo["allowed"]) and fc["allowed"]:
                    contingency["fail_closed_only"] += 1
                else:
                    contingency["both_deny"] += 1
                    both_deny += 1
                    if fo["reason_class"] == fc["reason_class"]:
                        reason_agree += 1
        n_pairs = len(seeds) * n_tasks
        fo_allow = _rate(contingency["both_allow"] + contingency["fail_open_only"],
                         n_pairs)
        fc_allow = _rate(contingency["both_allow"] + contingency["fail_closed_only"],
                         n_pairs)
        out.append({
            "scenario": sid,
            "n_task_pairs": n_pairs,
            "outcome_contingency": contingency,
            "outcome_agreement_rate": _rate(
                contingency["both_allow"] + contingency["both_deny"], n_pairs
            ),
            "fail_open_allow_rate": fo_allow,
            "fail_closed_allow_rate": fc_allow,
            "reason_class_agreement_rate_among_both_denied": _rate(
                reason_agree, both_deny
            ),
            "observed_note": _cell_note(fo_allow, fc_allow),
        })
    return out


def aggregate_coordinator(raw: List[Dict[str, Any]],
                          seeds: List[int]) -> List[Dict[str, Any]]:
    """Coordinator-level propagation counts per scenario x mode."""
    cells = []
    for scenario in SCENARIOS:
        sid = scenario["scenario"]
        for mode in MODES:
            recs = [r for r in raw if r["scenario"] == sid and r["mode"] == mode]
            n = len(recs)
            executed = sum(1 for r in recs if r["outcome"] == "executed")
            errors = [r for r in recs if r["outcome"] == "security_error"]
            per_seed_rates = []
            for seed in seeds:
                srecs = [r for r in recs if r["seed"] == seed]
                per_seed_rates.append(
                    _rate(sum(1 for r in srecs
                              if r["outcome"] == "executed"), len(srecs))
                    if srecs else 0.0
                )
            cells.append({
                "scenario": sid,
                "mode": mode,
                "n_runs": n,
                "executed": executed,
                "security_error_raised": len(errors),
                "execute_rate": _rate(executed, n),
                "security_error_rate": _rate(len(errors), n),
                "execute_rate_per_seed": _mean_stdev(per_seed_rates),
                "violation_recorded_rate_among_security_errors": _rate(
                    sum(1 for r in errors if r["violation_recorded"]),
                    len(errors),
                ),
            })
    return cells


def aggregate_posture_summary(
        divergence: List[Dict[str, Any]],
        matrix_cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline posture numbers, computed from the aggregated tables."""
    hole_flips = 0
    hole_scenarios = []
    for row in divergence:
        flipped = row["outcome_contingency"]["fail_open_only"]
        if row["scenario"] != "normal_task" and flipped > 0:
            hole_scenarios.append(row["scenario"])
            hole_flips += flipped
    normal = [row for row in divergence if row["scenario"] == "normal_task"][0]
    strict_denied_cells = [
        c for c in matrix_cells
        if c["scenario"] != "normal_task" and c["mode"] == "fail_closed"
    ]
    strict_denials = sum(c["denied"] for c in strict_denied_cells)
    strict_recorded = sum(c["denied_with_recorded_violation"]
                          for c in strict_denied_cells)
    return {
        "hole_scenarios_fail_open_permits_fail_closed_blocks": hole_scenarios,
        "tasks_fail_open_would_permit_but_fail_closed_blocks": hole_flips,
        "normal_task_fail_open_denied": normal["outcome_contingency"][
            "both_deny"] + normal["outcome_contingency"]["fail_closed_only"],
        "normal_task_fail_closed_denied": normal["outcome_contingency"][
            "both_deny"] + normal["outcome_contingency"]["fail_open_only"],
        "strict_mode_denials_total_non_normal": strict_denials,
        "strict_mode_denials_with_recorded_violation_non_normal": strict_recorded,
    }


def raw_count_table(monitor_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Finest-grained table embedded in the JSON: per (scenario, variant,
    mode, seed) allow/deny counts. All headline aggregates above are
    computed from the full per-task records; this table lets a reader
    re-derive them."""
    table = {}
    for r in monitor_raw:
        key = (r["scenario"], r["variant"], r["mode"], r["seed"])
        entry = table.setdefault(key, {"n": 0, "allowed": 0})
        entry["n"] += 1
        if r["allowed"]:
            entry["allowed"] += 1
    rows = []
    for key in sorted(table):
        sid, variant, mode, seed = key
        entry = table[key]
        rows.append({
            "scenario": sid,
            "variant": variant,
            "mode": mode,
            "seed": seed,
            "n": entry["n"],
            "allowed": entry["allowed"],
            "denied": entry["n"] - entry["allowed"],
        })
    return rows


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def capture_git_rev() -> str:
    """Read-only ``git rev-parse --short HEAD`` for provenance."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15, check=True,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_experiment(seeds: List[int], n_tasks: int) -> Dict[str, Any]:
    """Run the full matrix and aggregate; returns the result document."""
    population = build_tasks(seeds, n_tasks)
    monitor_raw = run_monitor_matrix(population)
    coordinator_raw = run_coordinator_matrix(population)

    matrix_cells = aggregate_matrix(monitor_raw, seeds)
    divergence = aggregate_mode_divergence(monitor_raw, seeds, n_tasks)
    coordinator_cells = aggregate_coordinator(coordinator_raw, seeds)
    summary = aggregate_posture_summary(divergence, matrix_cells)

    n_main = sum(1 for s in SCENARIOS if s["matrix_role"] == "main")
    return {
        "experiment": "security_posture_matrix",
        "description": (
            "Functional fail-open vs fail-closed matrix for the AgentISA "
            "coordination security layer: scenarios (unregistered resource "
            "/ unknown data sensitivity / incomplete resource-access policy "
            "/ normal task, plus a supplementary malformed-registry row) x "
            "enforcement modes (strict=False fail-open default vs "
            "strict=True fail-closed). Qualitative + counting only: "
            "decisions and recorded reasons, no performance quantities."
        ),
        "config": {
            "seeds": list(seeds),
            "n_seeds": len(seeds),
            "tasks_per_scenario_per_seed": n_tasks,
            "modes": list(MODES),
            "mode_flags": {"fail_open": "strict=False",
                           "fail_closed": "strict=True"},
            "scenarios_main": [s["scenario"] for s in SCENARIOS
                               if s["matrix_role"] == "main"],
            "scenarios_supplementary": [s["scenario"] for s in SCENARIOS
                                        if s["matrix_role"] != "main"],
        },
        "provenance": {
            "git_rev": capture_git_rev(),
            "git_rev_command": "git rev-parse --short HEAD (read-only)",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(
                timespec="seconds"),
            "seeds": list(seeds),
            "scale": {
                "scenarios_main": n_main,
                "scenarios_supplementary": len(SCENARIOS) - n_main,
                "sub_variants_total": sum(len(s["variants"])
                                          for s in SCENARIOS),
                "task_population": len(population),
                "monitor_evaluations": len(monitor_raw),
                "coordinator_runs": len(coordinator_raw),
                "modes": len(MODES),
            },
            "command": " ".join(sys.argv),
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "module_under_test": "agentflow/core/isa/coordination.py",
            "classes_under_test": ["SecurityMonitor", "AgentCoordinator"],
        },
        "coverage": {
            "dependency": {
                "job": "I4-security-failclosed",
                "status": "shipped",
                "surface": ("SecurityMonitor(config, strict=...) and "
                            "AgentCoordinator(config, security_strict=...)"),
            },
            "coverage_status": "full",
            "covered": [
                "all four required scenarios x both enforcement modes at "
                "the SecurityMonitor.validate_task level (fresh monitor "
                "per task x mode)",
                "the supplementary malformed-registry row in both modes",
                "end-to-end propagation through "
                "AgentCoordinator.coordinate_task for the same task "
                "population (fresh coordinator per task x mode)",
                "violation recording (auditability) for every denial in "
                "both modes, and the '(strict mode)' marker on strict "
                "denials",
            ],
            "not_covered": [
                "the always-on authentication/authorization checks are "
                "held fixed (all tasks carry tokens); they are not a "
                "matrix axis",
                "required_level is LOW everywhere except two normal-task "
                "variants (MEDIUM/HIGH); the sensitivity-vs-requirement "
                "mismatch path is exercised only by the shipped unit "
                "suite, not scaled here",
                "no live network, LLM or ray involvement: posture counting "
                "only, so there are no latency/throughput numbers by "
                "construction",
            ],
        },
        "posture_summary": summary,
        "matrix": matrix_cells,
        "mode_divergence": divergence,
        "coordinator_propagation": coordinator_cells,
        "raw_counts": {"monitor": raw_count_table(monitor_raw)},
        "notes": [
            "Qualitative + counting study only: allow/deny decisions and "
            "recorded violation reasons. No latency, throughput, memory "
            "or any other performance quantity is measured or claimed.",
            "The decision function under test is deterministic; a per-seed "
            "stdev of 0.0 reflects that determinism. Seeds vary the task "
            "surface (resource names, request shapes, sub-variant mix), "
            "not the verdict.",
            "Every task holds the non-varied security dimension at a "
            "known-good value (declared data_sensitivity=0 for the "
            "resource scenarios; no resources requested for the "
            "sensitivity scenarios), so exactly one dimension varies per "
            "scenario row.",
            "The same task dict is evaluated in both modes and paired, so "
            "mode_divergence is a true per-task 2x2 contingency, not a "
            "comparison of independent samples.",
            "Denials are recorded in violation_history in BOTH modes "
            "(auditability is not new); what fail-closed adds is the "
            "distinguishing reason classes and the '(strict mode)' "
            "marker.",
            "The malformed-registry row is supplementary: I4 groups a "
            "type-malformed allowed_resources under 'incomplete policy', "
            "but its fail-open outcome is a generic deny (a string "
            "registry degenerates to a character set), unlike the "
            "canonical missing-registry hole which fail-open permits.",
        ],
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail-open vs fail-closed security posture matrix.")
    parser.add_argument("--seeds", type=int, default=5,
                        help="number of seeds (default 5)")
    parser.add_argument("--tasks-per-scenario", type=int, default=40,
                        help="tasks per scenario per seed (default 40)")
    parser.add_argument("--seed-base", type=int, default=11,
                        help="first seed value; seed k = base + k")
    parser.add_argument("--output", type=str,
                        default=str(Path(__file__).resolve().with_name(
                            "security_posture_results.json")),
                        help="output JSON path")
    args = parser.parse_args(argv)

    seeds = [args.seed_base + k for k in range(args.seeds)]
    result = run_experiment(seeds, args.tasks_per_scenario)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    # Human-readable summary (counts only; no performance claims).
    print(f"wrote {out_path}")
    print(f"git_rev={result['provenance']['git_rev']} "
          f"seeds={seeds} tasks/scenario/seed={args.tasks_per_scenario}")
    for row in result["mode_divergence"]:
        c = row["outcome_contingency"]
        print(f"  {row['scenario']:<36} "
              f"both_allow={c['both_allow']:>4} "
              f"fail_open_only={c['fail_open_only']:>4} "
              f"fail_closed_only={c['fail_closed_only']:>4} "
              f"both_deny={c['both_deny']:>4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
