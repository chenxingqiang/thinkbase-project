"""Token-mix sweep of the MODELED priced comparison (E9, mock, offline).

``examples/benchmarks/promptcache_sweep.py`` sweeps the §5.7.1 /
Appendix G MODELED pricing overlay (no-optimization vs AgentISA result
caching vs provider prompt cache) across the input/output token-mix
axis (completion-dominated vs input-dominated) crossed with the paper's
three Zipf skew levels, on the deterministic mock workload of the
compiler harness.  These tests pin the mechanics offline: exact
reproduction of the published mock paper-scale anchor, ordering
invariants of the three pricing modes, script-computed aggregation,
crossover interpolation, determinism, and the MODELED labeling /
provenance discipline of the results document.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import pytest

# Make the benchmark script importable (it lives in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import promptcache_sweep as pcs  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISHED_MOCK = (
    REPO_ROOT / "examples" / "paper_real_llm_compiler_results_mock_paper.json")
PUBLISHED_REAL = (
    REPO_ROOT / "examples" / "paper_real_llm_compiler_results_real.json")


def _tiny_config(**overrides) -> dict:
    base = dict(
        n_seeds=3, seed_base=1000, n_draws=60, population=40,
        zipf_levels=[0.5, 1.5],
        mix_ids=["mock_measured", "L4", "L64", "L512"],
    )
    base.update(overrides)
    return base


def _curve_config(**overrides) -> dict:
    """Enough mix points to bracket both crossovers at reduced scale."""
    base = dict(
        n_seeds=2, seed_base=1000, n_draws=100, population=40,
        zipf_levels=[1.0],
        mix_ids=["mock_measured", "L8", "L16", "L32", "L64", "L512"],
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Grid and configuration
# ---------------------------------------------------------------------------

def test_mix_grid_covers_both_regimes_and_published_anchors():
    assert pcs.MOCK_MIX_ID == "mock_measured"
    for level in (4, 8, 16, 24, 32, 48, 64, 128, 256, 384, 512, 1024):
        assert level in pcs.MIX_CONST_L
    # The two published alias ceilings are swept points.
    assert 64 in pcs.MIX_CONST_L      # Section 5.7 max_tokens
    assert 512 in pcs.MIX_CONST_L     # Section 5.7.1 max_tokens
    assert pcs.DEFAULT_ZIPF_LEVELS == (0.5, 1.0, 1.5)
    assert pcs.DEFAULT_N_SEEDS == 5
    assert pcs.DEFAULT_SEED_BASE == 1000
    assert pcs.DEFAULT_DRAWS == 200
    assert pcs.DEFAULT_POPULATION == 40


def test_mix_id_roundtrip():
    for level in pcs.MIX_CONST_L:
        assert pcs.completion_tokens_for_mix(f"L{level}") == level
    assert pcs.completion_tokens_for_mix("mock_measured") is None
    assert pcs.completion_tokens_for_mix("L33") == 33   # off-grid constants OK
    with pytest.raises(ValueError):
        pcs.completion_tokens_for_mix("Lnotanumber")
    with pytest.raises(ValueError):
        pcs.completion_tokens_for_mix("L0")
    with pytest.raises(ValueError):
        pcs.completion_tokens_for_mix("L-5")


# ---------------------------------------------------------------------------
# Anchor: exact reproduction of the published mock paper-scale numbers
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PUBLISHED_MOCK.exists(), reason="published mock results missing")
def test_anchor_exact_reproduction_all_zipf_levels():
    published = json.loads(PUBLISHED_MOCK.read_text())
    rank_table = pcs.precompute_rank_table(40)
    pub_cells = {c["zipf_s"]: c for c in published["data"]}
    for zipf_s in (0.5, 1.0, 1.5):
        cell = pcs.run_cell(
            zipf_s=zipf_s, seed=1000, n_draws=200, population=40,
            rank_table=rank_table, mix_id="mock_measured",
            price_in_per_mt=3.0, price_out_per_mt=15.0, cache_read_rate=0.1)
        pub_modes = pub_cells[zipf_s]["priced_comparison"]["modes"]
        for mode in pcs.COST_MODES:
            got, want = cell["modes"][mode], pub_modes[mode]
            for field in ("calls", "input_tokens_full_price",
                          "input_tokens_cache_read", "output_tokens",
                          "cost_usd"):
                assert got[field] == want[field], (zipf_s, mode, field)


@pytest.mark.skipif(not PUBLISHED_MOCK.exists(), reason="published mock results missing")
def test_anchor_savings_match_published_percentages():
    published = json.loads(PUBLISHED_MOCK.read_text())
    rank_table = pcs.precompute_rank_table(40)
    pub_cells = {c["zipf_s"]: c for c in published["data"]}
    for zipf_s in (0.5, 1.0, 1.5):
        cell = pcs.run_cell(
            zipf_s=zipf_s, seed=1000, n_draws=200, population=40,
            rank_table=rank_table, mix_id="mock_measured",
            price_in_per_mt=3.0, price_out_per_mt=15.0, cache_read_rate=0.1)
        pub = pub_cells[zipf_s]["priced_comparison"]["modes"]
        pub_ag = 1 - pub["agentisa_cache"]["cost_usd"] / pub["no_opt"]["cost_usd"]
        pub_pr = (1 - pub["provider_prompt_cache"]["cost_usd"]
                  / pub["no_opt"]["cost_usd"])
        assert abs(cell["savings"]["agentisa_cache_pct"] - pub_ag * 100) < 0.05
        assert abs(cell["savings"]["provider_prompt_cache_pct"]
                   - pub_pr * 100) < 0.05


# ---------------------------------------------------------------------------
# Ordering invariants of the three pricing modes
# ---------------------------------------------------------------------------

def test_ordering_invariants_hold_in_every_cell():
    results = pcs.run_sweep(_tiny_config())
    assert results["data"]["per_cell"]
    for cell in results["data"]["per_cell"]:
        costs = {m: cell["modes"][m]["cost_usd"] for m in pcs.COST_MODES}
        assert costs["no_opt"] >= costs["provider_prompt_cache"]
        assert costs["provider_prompt_cache"] >= costs["agentisa_cache"]
        for key in ("agentisa_cache_pct", "provider_prompt_cache_pct"):
            assert 0.0 <= cell["savings"][key] <= 100.0
        assert cell["gap_pct_points"] >= 0.0
        assert 0.0 < cell["ratio_prompt_to_agentisa"] <= 1.0


def test_ordering_invariance_block_reports_zero_flips():
    results = pcs.run_sweep(_tiny_config())
    inv = results["analysis"]["ordering_invariance"]
    assert inv["cells_checked"] == len(results["data"]["per_cell"])
    assert inv["flipped_cells"] == 0
    assert inv["min_gap_usd"] > 0.0


# ---------------------------------------------------------------------------
# Aggregation discipline: aggregates recomputed from per-cell raw records
# ---------------------------------------------------------------------------

def test_aggregates_are_recomputed_from_per_cell():
    results = pcs.run_sweep(_tiny_config())
    cells = results["data"]["per_cell"]
    for agg in results["data"]["aggregates"]:
        group = [c for c in cells
                 if c["zipf_s"] == agg["zipf_s"] and c["mix_id"] == agg["mix_id"]]
        assert len(group) == agg["n_seeds"] == 3
        for field, path in [
            ("savings_agentisa_cache_pct",
             ("savings", "agentisa_cache_pct")),
            ("savings_provider_prompt_cache_pct",
             ("savings", "provider_prompt_cache_pct")),
            ("gap_pct_points", ("gap_pct_points",)),
            ("ratio_prompt_to_agentisa", ("ratio_prompt_to_agentisa",)),
            ("cost_usd_no_opt", ("modes", "no_opt", "cost_usd")),
            ("cost_usd_agentisa_cache", ("modes", "agentisa_cache", "cost_usd")),
            ("cost_usd_provider_prompt_cache",
             ("modes", "provider_prompt_cache", "cost_usd")),
        ]:
            vals = []
            for c in group:
                v = c
                for step in path:
                    v = v[step]
                vals.append(v)
            want_mean = statistics.mean(vals)
            want_stdev = statistics.stdev(vals)
            got = agg["metrics"][field]
            assert abs(got["mean"] - want_mean) < 1e-4, field
            assert abs(got["stdev"] - want_stdev) < 1e-4, field


# ---------------------------------------------------------------------------
# Crossover interpolation helper
# ---------------------------------------------------------------------------

def test_interpolate_crossing_decreasing_curve():
    curve = [(1.0, 0.9), (2.0, 0.7), (3.0, 0.4), (4.0, 0.1)]
    hit = pcs.interpolate_crossing(curve, 0.5)
    assert hit is not None
    assert hit["bracket"] == [2.0, 3.0]
    assert abs(hit["x"] - (2.0 + 0.2 / 0.3)) < 1e-9


def test_interpolate_crossing_increasing_curve():
    curve = [(0.0, 10.0), (10.0, 40.0), (20.0, 60.0)]
    hit = pcs.interpolate_crossing(curve, 50.0)
    assert hit is not None
    assert hit["bracket"] == [10.0, 20.0]
    assert abs(hit["x"] - 15.0) < 1e-9


def test_interpolate_crossing_no_crossing_returns_none():
    assert pcs.interpolate_crossing([(1.0, 0.9), (2.0, 0.8)], 0.5) is None
    assert pcs.interpolate_crossing([(1.0, 0.2), (2.0, 0.1)], 0.5) is None
    assert pcs.interpolate_crossing([], 0.5) is None


# ---------------------------------------------------------------------------
# Curve shape: ratio falls with the completion share; mock sits between
# ---------------------------------------------------------------------------

def test_ratio_monotone_decreasing_and_mock_between_neighbors():
    results = pcs.run_sweep(_curve_config())
    curves = {c["zipf_s"]: c for c in results["analysis"]["per_skew"]}
    curve = curves[1.0]["curve"]
    by_mix = {p["mix_id"]: p for p in curve}
    # effective completion tokens per call of the measured mock replies
    mock_x = by_mix["mock_measured"]["x_completion_tokens_per_call"]
    assert 32.0 < mock_x < 48.0
    # the ratio is non-increasing along the mix axis (tiny numeric slack)
    ratios = [p["ratio_prompt_to_agentisa"] for p in curve]
    for a, b in zip(ratios, ratios[1:]):
        assert b <= a + 1e-9


def test_crossovers_bracketed_on_curve_config():
    results = pcs.run_sweep(_curve_config())
    for skew in results["analysis"]["per_skew"]:
        cross = skew["crossovers"]
        assert cross["ratio_0_5"] is not None
        assert cross["ratio_0_5"]["bracket"] == [16.0, 32.0]
        assert cross["provider_saving_5pct"] is not None
        assert cross["provider_saving_5pct"]["bracket"] == [64.0, 512.0]
        assert cross["output_cost_share_50pct"] is not None


def test_regime_membership_consistent_with_crossovers():
    results = pcs.run_sweep(_curve_config())
    for skew in results["analysis"]["per_skew"]:
        cross_x = skew["crossovers"]["ratio_0_5"]["x"]
        for point in skew["curve"]:
            if point["x_completion_tokens_per_call"] < cross_x:
                assert point["mix_id"] in skew["regimes"]["prompt_cache_at_least_half_as_effective"]
            else:
                assert point["mix_id"] in skew["regimes"]["prompt_cache_less_than_half"]


# ---------------------------------------------------------------------------
# Results document: schema, MODELED labels, provenance, determinism
# ---------------------------------------------------------------------------

def test_results_document_schema_and_modeled_labels(tmp_path):
    results = pcs.run_sweep(_tiny_config())
    for key in ("experiment", "harness", "method", "config", "provenance",
                "coverage", "data", "analysis", "notes"):
        assert key in results
    # every cost record is labeled MODELED-on-measured
    for cell in results["data"]["per_cell"]:
        for mode in pcs.COST_MODES:
            assert cell["modes"][mode]["label"] == "MODELED-on-measured"
    assert "MODELED" in results["method"]["labels"]
    prov = results["provenance"]
    for key in ("git_rev", "git_rev_command", "timestamp_utc", "seeds",
                "scale", "script", "dependency", "python"):
        assert key in prov
    assert prov["seeds"] == [1000, 1001, 1002]
    cov = results["coverage"]
    assert cov["sweep_cells_completed"] == cov["sweep_cells_planned"]
    assert cov["sweep_cells_completed"] == len(results["data"]["per_cell"])
    assert cov["blocked"] == []


def test_determinism_same_config_same_data():
    a = pcs.run_sweep(_tiny_config())
    b = pcs.run_sweep(_tiny_config())
    assert a["data"] == b["data"]
    assert a["analysis"] == b["analysis"]


def test_cli_writes_results_file(tmp_path, monkeypatch):
    out = tmp_path / "sweep.json"
    argv = ["--output", str(out), "--seeds", "2", "--draws", "50",
            "--zipf", "1.0", "--mix", "mock_measured", "--mix", "L512"]
    ret = pcs.main(argv)
    assert ret == 0
    doc = json.loads(out.read_text())
    assert doc["coverage"]["sweep_cells_completed"] == 4  # 1 zipf x 2 mix x 2 seeds
    assert doc["config"]["n_seeds"] == 2
    for cell in doc["data"]["per_cell"]:
        assert cell["mix_id"] in ("mock_measured", "L512")


def test_published_real_anchor_quoted_when_present():
    if not PUBLISHED_REAL.exists():
        pytest.skip("published real-gateway results missing")
    results = pcs.run_sweep(_tiny_config())
    real = results["analysis"]["published_anchor_checks"]["real_gateway_run"]
    assert real["source"] == str(
        PUBLISHED_REAL.relative_to(REPO_ROOT))
    # the published numbers are quoted, not recomputed
    assert abs(real["savings_agentisa_cache_pct"] - 56.0) < 0.15
    assert abs(real["savings_provider_prompt_cache_pct"] - 3.3) < 0.15
    assert real["label"] == "MODELED-on-measured (published)"
