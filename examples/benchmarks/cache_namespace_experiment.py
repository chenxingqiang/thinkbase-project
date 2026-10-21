"""Semantic-configuration-drift experiment for the AgentISA cache-key
contract (E6-cache-drift).

The paper's cache-key contract (Section 5.2) keys a cached instruction
result on the (instruction, params) pair and assumes the model alias,
prompt-template version, and decoding configuration are held fixed;
the production cache historically could not detect a change in any of
those factors (I5 added the opt-in strict ``CacheKeyNamespace``).
This experiment quantifies both halves of the contract under a
controlled deployment-wide semantic drift:

1. strict namespaced keys -- the first lookup after the drift must
   MISS the pre-drift key (the leaf re-executes under the new
   configuration and serves a fresh result), the second lookup must
   HIT the new key, and reverting the namespace must HIT the
   still-resident pre-drift entry (isolation, not eviction);
2. regular legacy keys -- the same drift leaves the params-only key
   untouched, so post-drift lookups keep HITTING; every one of those
   hits serves an entry that was produced under the pre-drift
   semantic configuration, quantified here as the stale-serve rate.

A benign control axis (reordering the decoding dict's keys, which the
canonical decoding hash must fold away) checks that the strict
namespace does not over-invalidate: no key may change, no entry may
be re-executed, and the served entry stays consistent with the
(reordered but equivalent) deployment configuration.

Every number in the emitted JSON is computed by this script from raw
per-cohort counts; nothing is hand-entered.  The leaf instruction is
an in-process mock whose result embeds the deployment's *current*
semantic configuration at execution time, which is what makes
staleness observable without any live LLM: all hit/miss decisions
come from the real ``InstructionCache`` / ``CompositeInstruction``
code paths.

Usage
-----
    python cache_namespace_experiment.py \
        --seeds 5 --items 100 \
        --output cache_namespace_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo package importable regardless of how the script is
# invoked (same convention as the other scripts in this directory).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.instruction import (  # noqa: E402
    CompositeInstruction,
    Instruction,
    InstructionType,
)
from agentflow.core.isa.instruction_cache import CacheKeyNamespace  # noqa: E402


# --------------------------------------------------------------------------
# Experiment definition
# --------------------------------------------------------------------------

AXES = (
    "template_version",
    "decoding",
    "decoding_benign_reorder",
)
ARMS = (
    "strict_namespaced",
    "regular_legacy",
)
#: Axes where the drift is semantically real, so the strict arm also
#: runs the rollback (isolation-not-eviction) check afterwards.
ROLLBACK_AXES = ("template_version", "decoding")

BASELINE_CONFIG = {
    "model_alias": "alias-a",
    "template_version": "v1",
    "decoding": {"temperature": 0.0, "max_tokens": 64},
}

#: The deployment-wide drift events, one per axis.  ``template_version``
#: and ``decoding`` change semantics; ``decoding_benign_reorder`` only
#: changes dict insertion order, which a correct canonical hash folds
#: away (the over-invalidation control).
DRIFT_EVENTS = {
    "template_version": {
        "kind": "semantic",
        "template_version": "v1 -> v2",
    },
    "decoding": {
        "kind": "semantic",
        "decoding": {"temperature": [0.0, 0.7], "max_tokens": "64 -> 64"},
    },
    "decoding_benign_reorder": {
        "kind": "benign_control",
        "decoding": "key insertion order only; values identical",
    },
}

CACHE_CONFIG = {"max_size": 1000, "ttl": 3600}

#: Raw integer count fields (per cohort) that are summed across seeds.
COUNT_KEYS = (
    "n_items",
    "cold_misses",
    "cold_hits",
    "warm_hits",
    "warm_misses",
    "keys_changed_after_drift",
    "keys_unchanged_after_drift",
    "d1_misses",
    "d1_hits",
    "d1_fresh_served",
    "d1_stale_served",
    "d1_inner_reexecutions",
    "d2_hits",
    "d2_misses",
    "rollback_lookups",
    "rollback_hits",
    "rollback_misses",
    "rollback_served_original",
    "rollback_served_drifted",
    "lookups_total",
    "metrics_mismatches",
)


# --------------------------------------------------------------------------
# Mock leaf and small helpers
# --------------------------------------------------------------------------


class _DriftSensitiveInstruction(Instruction):
    """In-process mock of an LLM-backed leaf step.

    Its result embeds the deployment's *current* semantic
    configuration (model alias, template version, decoding) at
    execution time.  A cached entry produced under an older
    configuration carries that older snapshot, so a post-drift hit
    can be classified as stale by comparing snapshots -- no clock
    arithmetic, no live LLM.
    """

    def __init__(self, deployment: Dict[str, Any], id: str):
        super().__init__(
            id=id,
            name=id,
            type=InstructionType.LLM_INTERACTION,
            params={},
            description="mock leaf embedding the deployment config",
            side_effects=False,
        )
        self.deployment = deployment
        self.executions = 0

    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        self.executions += 1
        return {
            "executions": self.executions,
            "served_model_alias": self.deployment["model_alias"],
            "served_template_version": self.deployment["template_version"],
            "served_decoding": dict(self.deployment["decoding"]),
        }


def _snapshot(config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep-enough copy of a semantic configuration."""
    return {
        "model_alias": config["model_alias"],
        "template_version": config["template_version"],
        "decoding": dict(config["decoding"]),
    }


def _canon(obj: Any) -> str:
    """Canonical JSON (sorted keys) -- order-insensitive comparison."""
    return json.dumps(obj, sort_keys=True)


def _served_config(result: Dict[str, Any]) -> Dict[str, Any]:
    """The semantic configuration a served result was produced under."""
    return {
        "model_alias": result["served_model_alias"],
        "template_version": result["served_template_version"],
        "decoding": result["served_decoding"],
    }


def _apply_drift(axis: str,
                 deployment: Dict[str, Any],
                 namespace: Optional[CacheKeyNamespace]) -> None:
    """Deployment-wide drift event.

    The deployment's semantic configuration changes, and a
    namespaced deployment mirrors that change into its namespace
    object (in place -- the documented I5 drift pattern).  A legacy
    deployment has no namespace to update, which is exactly the
    point: its cache key cannot see the drift.
    """
    if axis == "template_version":
        deployment["template_version"] = "v2"
        if namespace is not None:
            namespace.template_version = "v2"
    elif axis == "decoding":
        deployment["decoding"] = {"temperature": 0.7, "max_tokens": 64}
        if namespace is not None:
            namespace.decoding = {"temperature": 0.7, "max_tokens": 64}
    elif axis == "decoding_benign_reorder":
        deployment["decoding"] = {"max_tokens": 64, "temperature": 0.0}
        if namespace is not None:
            namespace.decoding = {"max_tokens": 64, "temperature": 0.0}
    else:
        raise ValueError(f"unknown drift axis: {axis}")


def _revert_drift(axis: str,
                  deployment: Dict[str, Any],
                  namespace: Optional[CacheKeyNamespace]) -> None:
    """Roll the deployment (and its namespace) back to baseline."""
    if axis == "template_version":
        deployment["template_version"] = BASELINE_CONFIG["template_version"]
        if namespace is not None:
            namespace.template_version = BASELINE_CONFIG["template_version"]
    elif axis == "decoding":
        deployment["decoding"] = dict(BASELINE_CONFIG["decoding"])
        if namespace is not None:
            namespace.decoding = dict(BASELINE_CONFIG["decoding"])
    else:
        raise ValueError(f"axis has no rollback: {axis}")


def _empty_raw(axis: str, arm: str, seed: int, n_items: int) -> Dict[str, Any]:
    return {
        "seed": seed,
        "axis": axis,
        "arm": arm,
        "n_items": n_items,
        "cold_misses": 0,
        "cold_hits": 0,
        "warm_hits": 0,
        "warm_misses": 0,
        "keys_changed_after_drift": 0,
        "keys_unchanged_after_drift": 0,
        "d1_misses": 0,
        "d1_hits": 0,
        "d1_fresh_served": 0,
        "d1_stale_served": 0,
        "d1_inner_reexecutions": 0,
        "d2_hits": 0,
        "d2_misses": 0,
        "rollback_lookups": 0,
        "rollback_hits": 0,
        "rollback_misses": 0,
        "rollback_served_original": 0,
        "rollback_served_drifted": 0,
        "lookups_total": 0,
        "metrics_mismatches": 0,
        "anomalies": [],
    }


# --------------------------------------------------------------------------
# One cohort = one (axis, arm, seed): a fleet of n_items cached
# instructions living through one deployment-wide drift event.
# --------------------------------------------------------------------------


async def _run_cohort(axis: str,
                      arm: str,
                      seed: int,
                      n_items: int) -> Dict[str, Any]:
    deployment = _snapshot(BASELINE_CONFIG)
    namespace: Optional[CacheKeyNamespace] = None
    if arm == "strict_namespaced":
        # One namespace object shared by the whole strict fleet and
        # mutated in place at the drift event.
        namespace = CacheKeyNamespace(
            model_alias=BASELINE_CONFIG["model_alias"],
            template_version=BASELINE_CONFIG["template_version"],
            decoding=dict(BASELINE_CONFIG["decoding"]),
        )
    rng = random.Random(f"e6-cache-drift|{axis}|{arm}|{seed}")

    items: List[Dict[str, Any]] = []
    for i in range(n_items):
        leaf = _DriftSensitiveInstruction(
            deployment, id=f"leaf-{arm}-{axis}-{seed}-{i}"
        )
        composite = CompositeInstruction(
            id=f"comp-{arm}-{axis}-{seed}-{i}",
            name=f"comp-{arm}-{axis}-{seed}-{i}",
            components=[leaf],
            cache_namespace=namespace,
            cache_config=dict(CACHE_CONFIG),
        )
        items.append({
            "composite": composite,
            "leaf": leaf,
            "params": {
                "task_id": i,
                "payload": rng.randint(0, 10 ** 9),
                "priority": rng.choice((1, 2, 3)),
            },
            "observed_hits": 0,
            "observed_misses": 0,
        })

    raw = _empty_raw(axis, arm, seed, n_items)

    async def lookup(it: Dict[str, Any]) -> Tuple[Dict[str, Any], int]:
        """One cached lookup; a miss is when the leaf body actually ran."""
        raw["lookups_total"] += 1
        before = it["leaf"].executions
        result = await it["composite"].execute(it["params"])
        ran = it["leaf"].executions - before
        if ran:
            it["observed_misses"] += 1
        else:
            it["observed_hits"] += 1
        return result, ran

    # Phase 1 -- cold populate under the baseline configuration.
    for it in items:
        result, ran = await lookup(it)
        if ran:
            raw["cold_misses"] += 1
        else:
            raw["cold_hits"] += 1
            raw["anomalies"].append(
                f"cold lookup hit with an empty cache "
                f"(task_id={it['params']['task_id']})"
            )
        it["original_result"] = dict(result)
        it["key_before_drift"] = it["composite"]._get_cache_key(it["params"])

    # Phase 2 -- warm: the just-cached entry must be resident.
    for it in items:
        result, ran = await lookup(it)
        if not ran and result == it["original_result"]:
            raw["warm_hits"] += 1
        else:
            raw["warm_misses"] += 1
            raw["anomalies"].append(
                f"warm lookup did not serve the just-cached entry "
                f"(task_id={it['params']['task_id']})"
            )

    # The drift event, then the key-level measurement.
    _apply_drift(axis, deployment, namespace)
    for it in items:
        key_after = it["composite"]._get_cache_key(it["params"])
        if key_after != it["key_before_drift"]:
            raw["keys_changed_after_drift"] += 1
        else:
            raw["keys_unchanged_after_drift"] += 1

    # Phase 3 -- first post-drift lookup: the measured quantity.
    for it in items:
        result, ran = await lookup(it)
        if ran:
            raw["d1_misses"] += 1
            raw["d1_inner_reexecutions"] += ran
        else:
            raw["d1_hits"] += 1
        if _canon(_served_config(result)) == _canon(_snapshot(deployment)):
            raw["d1_fresh_served"] += 1
        else:
            raw["d1_stale_served"] += 1
        it["d1_result"] = dict(result)

    # Phase 4 -- second post-drift lookup: the new key must be resident.
    for it in items:
        result, ran = await lookup(it)
        if not ran and result == it["d1_result"]:
            raw["d2_hits"] += 1
        else:
            raw["d2_misses"] += 1
            raw["anomalies"].append(
                f"second post-drift lookup did not hit the new key "
                f"(task_id={it['params']['task_id']})"
            )

    # Phase 5 -- rollback: isolation, not eviction (strict arm, real
    # drift axes only).  Reverting the namespace must hit the original
    # pre-drift entry, proving the old key was isolated rather than
    # destroyed.
    if arm == "strict_namespaced" and axis in ROLLBACK_AXES:
        _revert_drift(axis, deployment, namespace)
        raw["rollback_lookups"] = n_items
        for it in items:
            result, ran = await lookup(it)
            if not ran:
                raw["rollback_hits"] += 1
            else:
                raw["rollback_misses"] += 1
                raw["anomalies"].append(
                    f"rollback lookup missed the original entry "
                    f"(task_id={it['params']['task_id']})"
                )
            if result == it["original_result"]:
                raw["rollback_served_original"] += 1
            elif result == it["d1_result"]:
                raw["rollback_served_drifted"] += 1

    # Consistency check: the framework's own per-instruction cache
    # counters must agree with the leaf's ground truth.
    for it in items:
        metrics = it["composite"].metrics
        if (metrics.cache_hits, metrics.cache_misses) != (
            it["observed_hits"], it["observed_misses"]
        ):
            raw["metrics_mismatches"] += 1

    return raw


# --------------------------------------------------------------------------
# Rates and aggregation (all computed, never hand-entered)
# --------------------------------------------------------------------------


def _derive_rates(raw: Dict[str, Any]) -> Dict[str, Any]:
    n = raw["n_items"]
    rates: Dict[str, Any] = {
        "cold_miss_rate": raw["cold_misses"] / n,
        "warm_hit_rate": raw["warm_hits"] / n,
        "key_change_rate": raw["keys_changed_after_drift"] / n,
        "first_post_drift_miss_rate": raw["d1_misses"] / n,
        "first_post_drift_hit_rate": raw["d1_hits"] / n,
        "first_post_drift_fresh_serve_rate": raw["d1_fresh_served"] / n,
        "first_post_drift_stale_serve_rate": raw["d1_stale_served"] / n,
        "second_post_drift_hit_rate": raw["d2_hits"] / n,
    }
    if raw["rollback_lookups"]:
        rates["rollback_hit_rate"] = (
            raw["rollback_hits"] / raw["rollback_lookups"]
        )
        rates["rollback_original_serve_rate"] = (
            raw["rollback_served_original"] / raw["rollback_lookups"]
        )
    else:
        rates["rollback_hit_rate"] = None
        rates["rollback_original_serve_rate"] = None
    return rates


def _aggregate(per_seed_raws: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_seed: List[Dict[str, Any]] = []
    for raw in per_seed_raws:
        entry = dict(raw)
        entry["rates"] = _derive_rates(raw)
        per_seed.append(entry)

    rate_names = list(per_seed[0]["rates"].keys())
    rates_agg: Dict[str, Any] = {}
    for name in rate_names:
        vals = [
            ps["rates"][name] for ps in per_seed if ps["rates"][name] is not None
        ]
        if vals:
            rates_agg[name] = {
                "mean": statistics.mean(vals),
                "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                "min": min(vals),
                "max": max(vals),
                "n_seeds": len(vals),
            }
        else:
            rates_agg[name] = None

    count_totals = {
        key: sum(ps[key] for ps in per_seed) for key in COUNT_KEYS
    }
    anomalies_total = sum(len(ps["anomalies"]) for ps in per_seed)
    return {
        "per_seed": per_seed,
        "aggregate": {
            "rates": rates_agg,
            "count_totals": count_totals,
            "anomalies_total": anomalies_total,
        },
    }


def _build_axes_data(seeds: List[int],
                     n_items: int) -> Dict[str, Any]:
    axes_data: Dict[str, Any] = {}
    for axis in AXES:
        arms_data: Dict[str, Any] = {}
        for arm in ARMS:
            raws = [
                asyncio.run(_run_cohort(axis, arm, seed, n_items))
                for seed in seeds
            ]
            arms_data[arm] = _aggregate(raws)
        axes_data[axis] = arms_data
    return axes_data


def _build_summary(axes_data: Dict[str, Any]) -> Dict[str, Any]:
    def rate(axis: str, arm: str, name: str) -> Any:
        return axes_data[axis][arm]["aggregate"]["rates"][name]

    return {
        "strict_namespaced": {
            "template_version_drift_invalidation_rate": rate(
                "template_version", "strict_namespaced",
                "first_post_drift_miss_rate",
            ),
            "decoding_drift_invalidation_rate": rate(
                "decoding", "strict_namespaced",
                "first_post_drift_miss_rate",
            ),
            "template_version_drift_fresh_serve_rate": rate(
                "template_version", "strict_namespaced",
                "first_post_drift_fresh_serve_rate",
            ),
            "decoding_drift_fresh_serve_rate": rate(
                "decoding", "strict_namespaced",
                "first_post_drift_fresh_serve_rate",
            ),
            "benign_reorder_invalidation_rate": rate(
                "decoding_benign_reorder", "strict_namespaced",
                "first_post_drift_miss_rate",
            ),
            "benign_reorder_key_change_rate": rate(
                "decoding_benign_reorder", "strict_namespaced",
                "key_change_rate",
            ),
            "rollback_hit_rate_after_template_version_drift": rate(
                "template_version", "strict_namespaced", "rollback_hit_rate",
            ),
            "rollback_hit_rate_after_decoding_drift": rate(
                "decoding", "strict_namespaced", "rollback_hit_rate",
            ),
            "rollback_original_serve_rate_after_template_version_drift": (
                rate("template_version", "strict_namespaced",
                     "rollback_original_serve_rate")
            ),
            "rollback_original_serve_rate_after_decoding_drift": rate(
                "decoding", "strict_namespaced",
                "rollback_original_serve_rate",
            ),
        },
        "regular_legacy": {
            "template_version_drift_hit_rate": rate(
                "template_version", "regular_legacy",
                "first_post_drift_hit_rate",
            ),
            "decoding_drift_hit_rate": rate(
                "decoding", "regular_legacy", "first_post_drift_hit_rate",
            ),
            "template_version_drift_invalidation_rate": rate(
                "template_version", "regular_legacy",
                "first_post_drift_miss_rate",
            ),
            "decoding_drift_invalidation_rate": rate(
                "decoding", "regular_legacy", "first_post_drift_miss_rate",
            ),
            "template_version_drift_stale_serve_rate": rate(
                "template_version", "regular_legacy",
                "first_post_drift_stale_serve_rate",
            ),
            "decoding_drift_stale_serve_rate": rate(
                "decoding", "regular_legacy",
                "first_post_drift_stale_serve_rate",
            ),
        },
        "metric_definitions": {
            "invalidation_rate": (
                "fraction of items whose FIRST post-drift lookup missed "
                "the pre-drift key (leaf re-executed under the new "
                "configuration)"
            ),
            "hit_rate": (
                "fraction of items whose first post-drift lookup was "
                "served from cache without re-execution"
            ),
            "fresh_serve_rate": (
                "fraction of first post-drift lookups whose served "
                "result embeds the deployment's CURRENT semantic "
                "configuration (canonical comparison)"
            ),
            "stale_serve_rate": (
                "fraction of first post-drift lookups served an entry "
                "produced under the PRE-drift configuration"
            ),
            "rollback_hit_rate": (
                "fraction of strict items that, after the namespace is "
                "reverted to baseline, hit the still-resident original "
                "entry (isolation, not eviction)"
            ),
        },
    }


def _build_scale(seeds: List[int],
                 n_items: int,
                 axes_data: Dict[str, Any]) -> Dict[str, Any]:
    total_items = 0
    total_lookups = 0
    for axis in AXES:
        for arm in ARMS:
            for ps in axes_data[axis][arm]["per_seed"]:
                total_items += ps["n_items"]
                total_lookups += ps["lookups_total"]
    return {
        "n_seeds": len(seeds),
        "n_axes": len(AXES),
        "n_arms": len(ARMS),
        "n_items_per_cohort": n_items,
        "n_cohorts": len(seeds) * len(AXES) * len(ARMS),
        "total_items": total_items,
        "total_cache_lookups": total_lookups,
        "lookups_per_item": "4 (cold, warm, post-drift x2) + 1 "
                            "rollback lookup for strict cohorts on the "
                            "template_version/decoding axes",
    }


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------


def _git_rev() -> Optional[str]:
    """Short HEAD rev via the one read-only git command."""
    repo_root = Path(__file__).resolve().parents[2]
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    rev = proc.stdout.strip()
    return rev or None


def _sha256_of_file(relative_path: str) -> Optional[str]:
    repo_root = Path(__file__).resolve().parents[2]
    path = repo_root / relative_path
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _build_provenance(seeds: List[int],
                      n_items: int,
                      axes_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "git_rev": _git_rev(),
        "git_rev_command": "git rev-parse --short HEAD",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": list(seeds),
        "scale": _build_scale(seeds, n_items, axes_data),
        "script": "examples/benchmarks/cache_namespace_experiment.py",
        "dependency": {
            "job": "I5-cache-namespacing",
            "implementation_report_status": "shipped",
            "modules_used": [
                "agentflow/core/isa/instruction_cache.py",
                "agentflow/core/isa/instruction.py",
            ],
            "module_sha256": {
                "agentflow/core/isa/instruction_cache.py":
                    _sha256_of_file(
                        "agentflow/core/isa/instruction_cache.py"
                    ),
                "agentflow/core/isa/instruction.py":
                    _sha256_of_file(
                        "agentflow/core/isa/instruction.py"
                    ),
            },
            "usage": (
                "strict arm attaches a shared CacheKeyNamespace to each "
                "CompositeInstruction and mutates it in place at the "
                "drift event; keys are produced by the production "
                "Instruction._get_cache_key -> build_cache_key path"
            ),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "determinism": (
            "pure in-process mock; no network, no sleeps, no live LLM; "
            "identical (seeds, items) reproduce identical measurements "
            "(timestamp and wall time excluded)"
        ),
    }


# --------------------------------------------------------------------------
# Experiment entry points
# --------------------------------------------------------------------------


def run_experiment(seeds: List[int], n_items: int) -> Dict[str, Any]:
    """Run the full drift experiment and return the results dict.

    Deterministic in ``seeds`` and ``n_items``: every measurement in
    ``axes``/``summary``/``config``/``coverage`` is identical across
    repeated calls (only ``provenance`` carries a timestamp).
    """
    axes_data = _build_axes_data(seeds, n_items)
    return {
        "experiment": "cache_namespace_drift",
        "description": (
            "Semantic-configuration-drift experiment for the AgentISA "
            "cache-key contract: after a deployment-wide "
            "template_version / decoding change, strictly namespaced "
            "cache keys must miss the pre-drift key (invalidation, "
            "fresh re-execution, and isolation-not-eviction on "
            "rollback) while regular legacy params-only keys keep "
            "hitting -- and thereby serve entries produced under the "
            "pre-drift configuration.  Hit/miss and stale/fresh "
            "serving are quantified per axis and arm."
        ),
        "provenance": _build_provenance(seeds, n_items, axes_data),
        "config": {
            "seeds": list(seeds),
            "n_items_per_cohort": n_items,
            "axes": list(AXES),
            "arms": list(ARMS),
            "baseline_semantic_config": _snapshot(BASELINE_CONFIG),
            "drift_events": DRIFT_EVENTS,
            "shared_namespace": (
                "one CacheKeyNamespace object shared by all strict-arm "
                "composites in a cohort and mutated in place at the "
                "drift event (the deployment's namespace bookkeeping)"
            ),
            "cache_config": dict(CACHE_CONFIG),
            "cohort_phases": [
                "cold_populate",
                "warm_verify",
                "drift_event",
                "first_post_drift_lookup",
                "second_post_drift_lookup",
                "rollback_isolation_check",
            ],
            "measurement_protocol": (
                "each lookup is CompositeInstruction.execute(params); a "
                "lookup is a miss iff the wrapped leaf body actually ran; "
                "a served result is stale iff its embedded semantic "
                "config snapshot differs (canonically) from the "
                "deployment's current config"
            ),
            "aggregation": {
                "per_seed": (
                    "each cohort runs one seed; rates are computed from "
                    "that cohort's raw counts"
                ),
                "across_seeds": (
                    "mean = statistics.mean; stdev = statistics.stdev "
                    "(sample, ddof=1; 0.0 when a single seed is run); "
                    "min/max recorded alongside"
                ),
                "discipline": (
                    "every aggregate number in this file is computed by "
                    "examples/benchmarks/cache_namespace_experiment.py "
                    "from the per-seed raw counts; none are hand-entered"
                ),
            },
        },
        "coverage": {
            "status": "full",
            "dependency_components": [
                {
                    "job": "I5-cache-namespacing",
                    "implementation_report_status": "shipped",
                    "coverage": "full",
                }
            ],
            "drift_axes_measured": ["template_version", "decoding"],
            "controls": [
                "decoding_benign_reorder (over-invalidation control)",
                "rollback isolation check (strict arm)",
            ],
            "arms": list(ARMS),
            "not_covered": [
                "model_alias drift (the third namespace component; not "
                "part of this experiment's mandate)",
                "shared-store cross-instruction key collisions (covered "
                "by unit tests in tests/core/isa/test_cache_namespacing.py)",
                "TTL expiry and LRU eviction dynamics (ttl=3600s and "
                "max_size=1000 far exceed the 2 keys each per-item cache "
                "ever holds; drift dynamics are the subject here)",
            ],
            "notes": [
                "regular-arm post-drift hits are the expected legacy "
                "behavior being quantified (keys not invalidated); their "
                "served entries embed the pre-drift semantic "
                "configuration, reported as stale_serve_rate",
                "no live LLM is contacted: the leaf is an in-process "
                "mock embedding the current deployment config; all "
                "hit/miss decisions come from the real "
                "InstructionCache/CompositeInstruction code paths",
            ],
        },
        "axes": axes_data,
        "summary": _build_summary(axes_data),
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "AgentISA semantic-configuration-drift cache experiment "
            "(strict namespacing vs legacy keys)"
        )
    )
    parser.add_argument(
        "--seeds", type=int, default=5,
        help="number of seeds (default: 5)",
    )
    parser.add_argument(
        "--seed-base", type=int, default=1000,
        help="seeds are seed_base, seed_base+1, ... (default: 1000)",
    )
    parser.add_argument(
        "--items", type=int, default=100,
        help="items per (axis, arm, seed) cohort (default: 100)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "cache_namespace_results.json",
        help="output JSON path",
    )
    args = parser.parse_args(argv)

    seeds = [args.seed_base + k for k in range(args.seeds)]
    start = time.perf_counter()
    results = run_experiment(seeds, args.items)
    results["wall_time_s"] = time.perf_counter() - start

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))

    print(f"wrote {args.output}")
    print(f"git rev {results['provenance']['git_rev']}, "
          f"seeds {seeds}, {args.items} items/cohort, "
          f"{results['provenance']['scale']['total_cache_lookups']} "
          f"cache lookups")
    summary = results["summary"]
    strict = summary["strict_namespaced"]
    regular = summary["regular_legacy"]
    print(
        "strict invalidation rate: "
        f"template_version={strict['template_version_drift_invalidation_rate']['mean']:.4f} "
        f"decoding={strict['decoding_drift_invalidation_rate']['mean']:.4f} "
        f"(benign reorder={strict['benign_reorder_invalidation_rate']['mean']:.4f})"
    )
    print(
        "regular post-drift hit rate: "
        f"template_version={regular['template_version_drift_hit_rate']['mean']:.4f} "
        f"decoding={regular['decoding_drift_hit_rate']['mean']:.4f}, "
        f"stale serve rate={regular['template_version_drift_stale_serve_rate']['mean']:.4f}"
    )
    print(
        "strict rollback hit rate: "
        f"template_version={strict['rollback_hit_rate_after_template_version_drift']['mean']:.4f} "
        f"decoding={strict['rollback_hit_rate_after_decoding_drift']['mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
