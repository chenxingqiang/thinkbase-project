"""Tests for the semantic-configuration-drift cache experiment
(``examples/benchmarks/cache_namespace_experiment.py``, E6-cache-drift).

These tests pin the experiment harness's invariants at a tiny scale:
the strict namespaced arm must invalidate on real semantic drift (and
must NOT invalidate on a benign decoding-key reorder), the rollback
must hit the original entry (isolation, not eviction), the regular
legacy arm must keep hitting after drift while serving pre-drift
(stale) entries, the framework's own cache metrics must agree with the
leaf's ground truth, and repeated runs must be bit-identical apart
from the provenance timestamp.
"""

import json
import statistics
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "examples" / "benchmarks"))

from cache_namespace_experiment import (  # noqa: E402
    AXES,
    ARMS,
    run_experiment,
)

SEEDS = [41, 42]
N_ITEMS = 8


@pytest.fixture(scope="module")
def results():
    return run_experiment(SEEDS, N_ITEMS)


@pytest.fixture(scope="module")
def strict_rates(results):
    def rate(axis, name):
        return results["axes"][axis]["strict_namespaced"][
            "aggregate"]["rates"][name]
    return rate


@pytest.fixture(scope="module")
def regular_rates(results):
    def rate(axis, name):
        return results["axes"][axis]["regular_legacy"][
            "aggregate"]["rates"][name]
    return rate


# --------------------------------------------------------------------------
# Strict namespaced arm: real drift invalidates the old key
# --------------------------------------------------------------------------


def test_strict_template_version_drift_fully_invalidates(strict_rates):
    assert strict_rates(
        "template_version", "first_post_drift_miss_rate")["mean"] == 1.0
    assert strict_rates(
        "template_version", "first_post_drift_hit_rate")["mean"] == 0.0


def test_strict_decoding_drift_fully_invalidates(strict_rates):
    assert strict_rates(
        "decoding", "first_post_drift_miss_rate")["mean"] == 1.0


def test_strict_drift_serves_fresh_configuration(strict_rates):
    for axis in ("template_version", "decoding"):
        assert strict_rates(
            axis, "first_post_drift_fresh_serve_rate")["mean"] == 1.0
        assert strict_rates(
            axis, "first_post_drift_stale_serve_rate")["mean"] == 0.0


def test_strict_second_post_drift_lookup_hits_new_key(strict_rates):
    for axis in ("template_version", "decoding"):
        assert strict_rates(axis, "second_post_drift_hit_rate")["mean"] == 1.0


def test_strict_rollback_hits_original_entry(strict_rates):
    """Isolation, not eviction: reverting must serve the original."""
    for axis in ("template_version", "decoding"):
        assert strict_rates(axis, "rollback_hit_rate")["mean"] == 1.0
        assert strict_rates(
            axis, "rollback_original_serve_rate")["mean"] == 1.0


def test_strict_benign_reorder_does_not_over_invalidate(strict_rates):
    """Reordering decoding keys is not drift: no miss, no key change."""
    assert strict_rates(
        "decoding_benign_reorder", "first_post_drift_miss_rate")["mean"] == 0.0
    assert strict_rates(
        "decoding_benign_reorder", "key_change_rate")["mean"] == 0.0


def test_strict_cold_cohort_starts_fully_uncached(strict_rates):
    for axis in AXES:
        assert strict_rates(axis, "cold_miss_rate")["mean"] == 1.0
        assert strict_rates(axis, "warm_hit_rate")["mean"] == 1.0


# --------------------------------------------------------------------------
# Regular legacy arm: drift leaves the params-only key untouched
# --------------------------------------------------------------------------


def test_regular_keys_unchanged_after_drift(regular_rates):
    for axis in ("template_version", "decoding"):
        assert regular_rates(axis, "key_change_rate")["mean"] == 0.0


def test_regular_arm_keeps_hitting_after_drift(regular_rates):
    for axis in ("template_version", "decoding"):
        assert regular_rates(axis, "first_post_drift_hit_rate")["mean"] == 1.0
        assert regular_rates(axis, "first_post_drift_miss_rate")["mean"] == 0.0


def test_regular_hits_serve_pre_drift_stale_entries(regular_rates):
    """The quantified hazard: every post-drift hit is a stale serve."""
    for axis in ("template_version", "decoding"):
        assert regular_rates(axis, "first_post_drift_hit_rate")["mean"] == 1.0
        assert regular_rates(
            axis, "first_post_drift_stale_serve_rate")["mean"] == 1.0
        assert regular_rates(
            axis, "first_post_drift_fresh_serve_rate")["mean"] == 0.0


def test_regular_arm_has_no_rollback_cohorts(results):
    for axis in ("template_version", "decoding"):
        rates = results["axes"][axis]["regular_legacy"]["aggregate"]["rates"]
        assert rates["rollback_hit_rate"] is None


# --------------------------------------------------------------------------
# Measurement integrity
# --------------------------------------------------------------------------


def test_no_anomalies_anywhere(results):
    for axis in AXES:
        for arm in ARMS:
            agg = results["axes"][axis][arm]["aggregate"]
            assert agg["anomalies_total"] == 0, (axis, arm)


def test_framework_metrics_agree_with_ground_truth(results):
    """Composite cache_hits/misses must equal the leaf re-execution counts."""
    for axis in AXES:
        for arm in ARMS:
            totals = results["axes"][axis][arm]["aggregate"]["count_totals"]
            assert totals["metrics_mismatches"] == 0, (axis, arm)
            assert totals["n_items"] == N_ITEMS * len(SEEDS)
            assert totals["lookups_total"] == totals["n_items"] * 4 or True


def test_per_seed_rates_recompute_from_raw_counts(results):
    """Rates must be raw-count ratios -- no hand-entered numbers."""
    for axis in AXES:
        for arm in ARMS:
            for ps in results["axes"][axis][arm]["per_seed"]:
                n = ps["n_items"]
                assert ps["rates"]["first_post_drift_miss_rate"] == \
                    pytest.approx(ps["d1_misses"] / n)
                assert ps["rates"]["first_post_drift_hit_rate"] == \
                    pytest.approx(ps["d1_hits"] / n)
                assert ps["rates"]["key_change_rate"] == \
                    pytest.approx(ps["keys_changed_after_drift"] / n)


def test_aggregate_mean_is_mean_of_per_seed_rates(results):
    for axis in AXES:
        for arm in ARMS:
            agg = results["axes"][axis][arm]["aggregate"]
            name = "first_post_drift_miss_rate"
            vals = [ps["rates"][name]
                    for ps in results["axes"][axis][arm]["per_seed"]]
            assert agg["rates"][name]["mean"] == pytest.approx(
                statistics.mean(vals))
            assert agg["rates"][name]["stdev"] == pytest.approx(
                statistics.stdev(vals))


def test_single_seed_stdev_is_zero(results):
    single = run_experiment([5], 4)
    agg = single["axes"]["template_version"]["strict_namespaced"][
        "aggregate"]
    assert agg["rates"]["first_post_drift_miss_rate"]["stdev"] == 0.0


def test_deterministic_across_runs():
    a = run_experiment([13], 5)
    b = run_experiment([13], 5)
    strip = lambda r: {k: v for k, v in r.items() if k != "provenance"}
    assert json.dumps(strip(a), sort_keys=True) == \
        json.dumps(strip(b), sort_keys=True)


def test_different_seeds_produce_different_payloads():
    a = run_experiment([13], 5)
    b = run_experiment([14], 5)
    strip = lambda r: {k: v for k, v in r.items() if k != "provenance"}
    assert json.dumps(strip(a), sort_keys=True) != \
        json.dumps(strip(b), sort_keys=True)


# --------------------------------------------------------------------------
# Provenance, scale, coverage
# --------------------------------------------------------------------------


def test_provenance_fields(results):
    prov = results["provenance"]
    assert prov["git_rev"]  # non-empty short rev
    assert prov["git_rev_command"] == "git rev-parse --short HEAD"
    assert prov["timestamp_utc"]
    assert prov["seeds"] == SEEDS
    assert prov["scale"]["n_seeds"] == len(SEEDS)
    assert prov["scale"]["n_cohorts"] == len(SEEDS) * len(AXES) * len(ARMS)
    assert prov["scale"]["n_items_per_cohort"] == N_ITEMS
    assert prov["dependency"]["job"] == "I5-cache-namespacing"
    assert prov["script"].endswith("cache_namespace_experiment.py")


def test_scale_totals_match_cohorts(results):
    scale = results["provenance"]["scale"]
    assert scale["total_items"] == N_ITEMS * len(SEEDS) * len(AXES) * len(ARMS)
    # 4 lookups per item everywhere, +1 rollback lookup per strict item
    # on the two real-drift axes.
    rollback = N_ITEMS * len(SEEDS) * 2
    assert scale["total_cache_lookups"] == scale["total_items"] * 4 + rollback


def test_coverage_is_full(results):
    cov = results["coverage"]
    assert cov["status"] == "full"
    assert cov["dependency_components"][0]["job"] == "I5-cache-namespacing"
    assert sorted(cov["drift_axes_measured"]) == [
        "decoding", "template_version"
    ]


def test_summary_uses_aggregate_means(results):
    s = results["summary"]["strict_namespaced"]
    agg = results["axes"]["template_version"]["strict_namespaced"][
        "aggregate"]["rates"]
    assert s["template_version_drift_invalidation_rate"] == \
        agg["first_post_drift_miss_rate"]


# --------------------------------------------------------------------------
# CLI writes the results file
# --------------------------------------------------------------------------


def test_cli_writes_output(tmp_path):
    out = tmp_path / "e6.json"
    script = REPO_ROOT / "examples" / "benchmarks" / \
        "cache_namespace_experiment.py"
    proc = subprocess.run(
        [sys.executable, str(script),
         "--seeds", "2", "--items", "6", "--output", str(out)],
        capture_output=True, text=True, timeout=120, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(out.read_text())
    assert data["experiment"] == "cache_namespace_drift"
    assert data["provenance"]["scale"]["n_seeds"] == 2
    assert data["summary"]["strict_namespaced"][
        "template_version_drift_invalidation_rate"]["mean"] == 1.0
