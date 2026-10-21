"""Tests for the E2 capability-calibration harness.

``examples/benchmarks/capability_calibration.py`` calibrates the I1
static capability extraction against measured behavior of the
microbench instruction pools (mock LLM backend, InstructionCompiler,
executor gates).  These tests pin the harness's honesty contract at a
tiny deterministic scale (``--quick`` configuration):

* the extraction-nonzero proof is real (not the stub-era constant 0),
* designated correlations carry the expected direction,
* domains/metrics without a measured counterpart are reported N/A
  with evidence instead of a fabricated number,
* aggregates are recomputable from the stored raw records,
* the whole pipeline is deterministic apart from the provenance
  timestamp / wall time,
* the committed full-scale results file is internally consistent.

Everything runs offline against the deterministic mock backend; no
network, no live gateway, no ray import on the harness's dependency
path.
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pytest

# Make the benchmark scripts importable (they live in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import capability_calibration as cc  # noqa: E402


RESULTS_PATH = (BENCH_DIR / "capability_calibration_results.json")


def _strip_nondeterminism(results: dict) -> dict:
    out = json.loads(json.dumps(results))
    out["provenance"].pop("timestamp", None)
    out["provenance"].pop("wall_time_s", None)
    return out


@pytest.fixture(scope="module")
def quick_results() -> dict:
    return cc.build_results(cc.quick_config())


# ---------------------------------------------------------------------------
# Determinism and provenance
# ---------------------------------------------------------------------------

def test_quick_run_is_deterministic():
    first = _strip_nondeterminism(cc.build_results(cc.quick_config()))
    second = _strip_nondeterminism(cc.build_results(cc.quick_config()))
    assert first == second


def test_provenance_fields(quick_results):
    provenance = quick_results["provenance"]
    assert re.fullmatch(r"[0-9a-f]{7,}", provenance["git_rev"]) or \
        provenance["git_rev"].startswith("unavailable")
    # ISO-8601 timestamp with timezone offset.
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}",
        provenance["timestamp"])
    seeds = quick_results["provenance"]["seed_protocol"]
    assert seeds["seeds_per_condition"] == cc.quick_config()["seeds"]
    scale = provenance["scale"]
    assert scale["n_replicas_total"] == sum(
        scale["n_replicas_per_protocol"].values())
    assert provenance["wall_time_s"] >= 0.0


def test_all_extracted_values_in_unit_interval(quick_results):
    for protocol in quick_results["protocols"].values():
        for rec in protocol["per_replica"]:
            for key, value in rec["flat"].items():
                if key.startswith(("metric.", "domain.")):
                    assert 0.0 <= value <= 1.0, (key, value)


# ---------------------------------------------------------------------------
# Extraction-nonzero proof (the point of E2)
# ---------------------------------------------------------------------------

def test_extraction_is_not_constant_zero(quick_results):
    proof = quick_results["extraction_nonzero_proof"]
    counts = proof["nonzero_metrics_per_seed"]
    assert counts["of_total"] == 17
    assert counts["mean"] >= 5  # stub era: 0/17 by construction
    domain_counts = proof["nonzero_domains_per_seed"]
    assert domain_counts["of_total"] == 8
    assert domain_counts["mean"] >= 4
    anywhere = proof["across_all_replicas_and_probe_arms"]
    assert anywhere["metrics_nonzero_somewhere_count"] == 17
    assert any(v > 0 for v in anywhere["metric_max_across_all_replicas"]
               .values())


def test_base_pool_values_are_real_not_stubbed(quick_results):
    """Values the stub era could never produce (stub: everything 0.0)."""
    base = quick_results["extraction_nonzero_proof"]["base_metric_values"]
    assert base["creativity"]["mean"] > 0.3
    assert base["long_term_memory"]["mean"] > 0.3
    assert base["coherence"]["mean"] > 0.3
    assert base["logical_consistency"]["mean"] == pytest.approx(1.0)
    assert base["error_handling"]["mean"] == pytest.approx(1.0)
    assert base["resource_usage"]["mean"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Correlations: designated pairings behave as designed
# ---------------------------------------------------------------------------

def test_creativity_instrument_agreement_is_exact(quick_results):
    corr = quick_results["correlations"]["per_metric"]["creativity"][
        "correlation"]
    assert corr["status"] == "ok"
    assert corr["pearson_r"] == pytest.approx(1.0, abs=1e-9)


def test_memory_calibration_positive(quick_results):
    corr = quick_results["correlations"]["per_metric"]["long_term_memory"][
        "correlation"]
    assert corr["status"] == "ok"
    assert corr["pearson_r"] > 0.5
    assert corr["spearman_rho"] > 0.5


def test_reasoning_and_coordination_calibrations_negative(quick_results):
    per_metric = quick_results["correlations"]["per_metric"]
    for metric in ("causal_understanding", "collaboration"):
        corr = per_metric[metric]["correlation"]
        assert corr["status"] == "ok"
        assert corr["pearson_r"] < -0.5, metric


def test_declared_and_enforced_calibrations_are_exact(quick_results):
    per_metric = quick_results["correlations"]["per_metric"]
    assert per_metric["resource_usage"]["correlation"]["pearson_r"] == \
        pytest.approx(-1.0, abs=1e-9)
    assert per_metric["policy_compliance"]["correlation"]["pearson_r"] == \
        pytest.approx(1.0, abs=1e-9)


def test_domain_correlations_have_expected_signs(quick_results):
    per_domain = quick_results["correlations"]["per_domain"]
    # Intensity protocols have n = 6 even at quick scale and are
    # strongly coupled; generation blends creativity with coherence
    # over only n = 4 quick pools, so only its sign is pinned here
    # (the strong bound is pinned on the full-scale committed file).
    # (lower, upper) exclusive bounds on pearson r; None = unbounded.
    expectations = {
        "reasoning": (None, -0.5),
        "memory": (+0.5, None),
        "generation": (+0.0, None),
        "optimization": (None, -0.5),
        "coordination": (None, -0.5),
        "security": (+0.5, None),
    }
    for domain, bounds in expectations.items():
        entry = per_domain[domain]
        assert entry["verdict"] == "correlated", domain
        r = entry["correlation"]["pearson_r"]
        lower, upper = bounds
        if lower is not None:
            assert r > lower, (domain, r)
        if upper is not None:
            assert r < upper, (domain, r)


# ---------------------------------------------------------------------------
# N/A honesty
# ---------------------------------------------------------------------------

def test_na_domains_report_no_coefficient(quick_results):
    per_domain = quick_results["correlations"]["per_domain"]
    for domain in ("specialization", "robustness"):
        entry = per_domain[domain]
        assert entry["status"] == "na"
        assert "correlation" not in entry
        assert entry["reason"]
        assert entry["evidence"]
    coverage = quick_results["coverage"]
    assert set(coverage["domains_na"]) == {"specialization", "robustness"}
    assert set(coverage["domains_with_coefficient"]) == {
        "reasoning", "memory", "generation", "optimization",
        "coordination", "security"}


def test_na_metrics_carry_evidence(quick_results):
    per_metric = quick_results["correlations"]["per_metric"]
    assert len(per_metric) == 17
    for metric, entry in per_metric.items():
        if entry.get("status") == "na":
            assert "correlation" not in entry
            assert entry["reason"]
            assert entry["evidence"]
        else:
            assert "correlation" in entry


def test_threat_detection_na_is_backed_by_measured_invariance(
        quick_results):
    """The audit channel is measured, not assumed inert."""
    per_metric = quick_results["correlations"]["per_metric"]
    assert per_metric["threat_detection"]["status"] == "na"
    audit_records = quick_results["protocols"]["P_audit"]["per_replica"]
    assert audit_records, "P_audit protocol must have replicas"
    # Extraction responds across intensities...
    threats = {rec["flat"]["metric.threat_detection"]
               for rec in audit_records}
    assert max(threats) > 0.9
    assert min(threats) == 0.0
    # ...while every measured battery output stays invariant per seed.
    numeric_keys = sorted(
        key for key, value in audit_records[0]["measured"].items()
        if isinstance(value, (int, float)))
    assert numeric_keys
    for key in numeric_keys:
        by_seed = {}
        for rec in audit_records:
            by_seed.setdefault(rec["seed"], []).append(rec["measured"][key])
        for values in by_seed.values():
            assert max(values) == min(values), key


def test_inertness_probes_report_exact_deltas(quick_results):
    probes = {p["probe"]: p for p in quick_results["inertness_probes"]}
    assert len(probes) == 10
    for name, probe in probes.items():
        assert probe["seeds"]
        # Deltas are exactly reproducible on deterministic paths.
        assert probe["max_abs_measured_delta"] >= 0.0
        if probe["verdict"] == "inert_in_microbench_paths":
            assert probe["max_abs_measured_delta"] == 0.0
            assert probe["measured_keys_moved"] == {}
        else:
            assert probe["measured_keys_moved"]
    # The agent-type probe is the one with a detected effect (fusion
    # keys on instruction type).
    agent = probes["agent_type_recategorization"]
    assert agent["verdict"] == "behavioral_effect_detected"
    assert "fusion_only_reduction" in agent["measured_keys_moved"]


# ---------------------------------------------------------------------------
# Aggregate discipline: recomputable from raw records
# ---------------------------------------------------------------------------

def test_aggregates_are_recomputed_from_raw_records(quick_results):
    protocol = quick_results["protocols"]["P_cache"]
    for intensity, entry in protocol["aggregates_by_intensity"].items():
        for flat_key, stats_entry in entry.items():
            raw = [rec["flat"][flat_key] for rec in protocol["per_replica"]
                   if rec["condition"]["intensity"] == float(intensity)
                   and flat_key in rec["flat"]]
            assert raw, (intensity, flat_key)
            assert stats_entry["n"] == len(raw)
            assert stats_entry["mean"] == pytest.approx(
                sum(raw) / len(raw), abs=1e-12)
            if len(raw) > 1:
                mean = sum(raw) / len(raw)
                variance = sum((v - mean) ** 2 for v in raw) / (len(raw) - 1)
                assert stats_entry["stdev"] == pytest.approx(
                    math.sqrt(variance), abs=1e-12)


def test_correlation_n_matches_paired_records(quick_results):
    protocol = quick_results["protocols"]["P_flow"]
    corr = quick_results["correlations"]["per_metric"][
        "causal_understanding"]["correlation"]
    assert corr["n"] == len(protocol["per_replica"])


# ---------------------------------------------------------------------------
# CLI and committed deliverable
# ---------------------------------------------------------------------------

def test_cli_writes_valid_json(tmp_path, monkeypatch, capsys):
    out = tmp_path / "calib.json"
    monkeypatch.setattr(
        sys, "argv",
        ["capability_calibration.py", "--quick", "--output", str(out)])
    cc.main()
    captured = capsys.readouterr()
    assert "results written to" in captured.out
    results = json.loads(out.read_text())
    assert results["experiment"] == "capability_calibration"
    assert results["correlations"]["per_domain"]["security"]["verdict"] == \
        "correlated"


def test_committed_full_scale_results_file():
    """The committed deliverable is the full-scale run, not a quick one."""
    if not RESULTS_PATH.exists():
        pytest.skip("committed results file not present")
    results = json.loads(RESULTS_PATH.read_text())
    assert results["experiment"] == "capability_calibration"
    provenance = results["provenance"]
    assert re.fullmatch(r"[0-9a-f]{7,}", provenance["git_rev"])
    assert provenance["seed_protocol"]["seeds_per_condition"] == 5
    scale = provenance["scale"]
    assert scale["n_replicas_total"] == sum(
        scale["n_replicas_per_protocol"].values())
    assert scale["n_probes"] == 10
    # Full-scale protocol sizes (5 seeds throughout).
    per_protocol = scale["n_replicas_per_protocol"]
    assert per_protocol["P_flow"] == 25  # 5 intensities x 5 seeds
    assert per_protocol["P_mix"] == 20   # 4 mixes x 5 seeds
    assert per_protocol["G_anchor"] == 15  # 3 lengths x 5 seeds
    # The headline claims are present and directional.
    per_domain = results["correlations"]["per_domain"]
    for domain in ("reasoning", "coordination", "optimization"):
        r = per_domain[domain]["correlation"]["pearson_r"]
        assert r < -0.5, domain
    for domain in ("memory", "generation", "security"):
        r = per_domain[domain]["correlation"]["pearson_r"]
        assert r > 0.5, domain
    for domain in ("specialization", "robustness"):
        assert per_domain[domain]["status"] == "na"
    proof = results["extraction_nonzero_proof"]
    assert proof["nonzero_metrics_per_seed"]["mean"] >= 5
    assert proof["across_all_replicas_and_probe_arms"][
        "metrics_nonzero_somewhere_count"] == 17
