"""Tests for the selector DP benchmark.

The benchmark under test is ``examples/benchmarks/selector_dp_bench.py``.

E3-selector-benchmark compares the shipped heuristic tree cover
(``_greedy_tree_cover``) against the MATCH-TREE dynamic program
(supplementary Appendix D.1) on random IR trees.  These tests pin:

* the random generators: exact node counts, determinism per seed, the
  catalogue structure (single-node pattern per op, premium variants,
  fused patterns declaring ``spanned_child_ops``) and the fused-cost
  discount model;
* per-instance records: both arms complete, DP cost <= heuristic cost,
  valid tilings, honest timing fields (median of the recorded reps);
* the aggregation discipline: every aggregate equals ``statistics``
  applied to the raw per-instance records -- nothing is hand-computed;
* coverage honesty: failed instances are excluded from aggregates and
  mark the run ``partial`` (and the CLI exits non-zero);
* the generated artifact ``selector_dp_results.json`` (skipif missing)
  still satisfies the aggregation discipline and the DP<=heuristic
  invariant for all 4 sizes x 5 seeds.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Make the benchmark script importable (it lives in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import selector_dp_bench as bench  # noqa: E402  (needs BENCH_DIR on sys.path)
from agentflow.core.isa.instruction_selector import (  # noqa: E402
    IRTreeNode,
    TreePattern,
    TreeSelectionResult,
)

ARTIFACT = BENCH_DIR / "selector_dp_results.json"
TEST_OPS = ("a", "b", "c")


def _r6(x):
    return round(x, 6)


def _catalogue_costs(catalogue):
    return {key: (instr.name, instr.cost) for key, instr in catalogue.items()}


# ---------------------------------------------------------------------------
# Random IR tree generation
# ---------------------------------------------------------------------------

class TestTreeGeneration:
    def test_tree_has_exactly_the_requested_node_count(self):
        for n in (1, 2, 3, 5, 20, 50):
            rng = random.Random(1000 + n)
            tree = bench.generate_ir_tree(rng, TEST_OPS, n)
            stats = bench.tree_stats(tree)
            assert stats["node_count"] == n

    def test_tree_uses_only_the_given_ops_and_depth_is_bounded(self):
        rng = random.Random(7)
        tree = bench.generate_ir_tree(rng, ("x", "y"), 40)
        ops_in_tree = {node.op for node in bench.iter_tree(tree)}
        assert ops_in_tree <= {"x", "y"}
        assert bench.tree_stats(tree)["depth"] <= 40

    def test_tree_generation_is_deterministic_per_seed(self):
        first = bench.generate_ir_tree(random.Random(4242), TEST_OPS, 25)
        second = bench.generate_ir_tree(random.Random(4242), TEST_OPS, 25)
        other_size = bench.generate_ir_tree(random.Random(4242), TEST_OPS, 24)
        assert bench.tree_signature(first) == bench.tree_signature(second)
        assert bench.tree_signature(first) != bench.tree_signature(other_size)

    def test_payload_probability_controls_node_payloads(self):
        full = bench.generate_ir_tree(
            random.Random(1), TEST_OPS, 15, payload_prob=1.0)
        none_ = bench.generate_ir_tree(
            random.Random(1), TEST_OPS, 15, payload_prob=0.0)
        assert all(node.payload == {"urgency": "high"}
                   for node in bench.iter_tree(full))
        assert all(node.payload == {} for node in bench.iter_tree(none_))
        assert bench.tree_stats(full)["payload_nodes"] == 15
        assert bench.tree_stats(none_)["payload_nodes"] == 0

    def test_node_count_below_one_is_rejected(self):
        with pytest.raises(ValueError):
            bench.generate_ir_tree(random.Random(0), TEST_OPS, 0)


# ---------------------------------------------------------------------------
# Random instruction catalogue
# ---------------------------------------------------------------------------

class TestCatalogueGeneration:
    def test_every_op_has_a_base_single_node_instruction(self):
        catalogue = bench.generate_catalogue(random.Random(11), TEST_OPS)
        for op in TEST_OPS:
            assert op in catalogue
            base = catalogue[op]
            assert base.name == op
            assert "spanned_child_ops" not in base.params
        # A single-node pattern per op guarantees both arms can cover any
        # tree over the op alphabet.
        for op in TEST_OPS:
            assert any(instr.name == op
                       and "spanned_child_ops" not in instr.params
                       for instr in catalogue.values())

    def test_premium_variants_cost_more_and_carry_the_urgency_param(self):
        rng = random.Random(23)
        catalogue = bench.generate_catalogue(rng, TEST_OPS)
        premium_keys = [k for k in catalogue if k.endswith("_premium")]
        assert len(premium_keys) == bench.PREMIUM_OPS_COUNT
        for key in premium_keys:
            instr = catalogue[key]
            base = catalogue[instr.name]
            assert instr.params == {"urgency": "high"}
            assert instr.cost > base.cost

    def test_fused_patterns_declare_span_and_discounted_costs(self):
        rng = random.Random(5)
        catalogue = bench.generate_catalogue(rng, TEST_OPS)
        summary = bench.catalogue_summary(catalogue)
        fused = summary["fused"]
        assert (bench.FUSED_COUNT_RANGE[0] <= len(fused)
                <= bench.FUSED_COUNT_RANGE[1])
        base_costs = summary["base_costs"]
        for entry in fused:
            assert entry["root_op"] in TEST_OPS
            assert 1 <= len(entry["spanned_child_ops"]) <= 2
            assert all(op in TEST_OPS for op in entry["spanned_child_ops"])
            parts = base_costs[entry["root_op"]] + sum(
                base_costs[op] for op in entry["spanned_child_ops"])
            lo, hi = bench.FUSED_DISCOUNT_RANGE
            # Costs are rounded to 3 decimals, so allow half a milli-unit
            # of rounding slack on top of the discount interval.
            assert lo * parts - 1e-3 <= entry["cost"] <= hi * parts + 1e-3
        # The catalogue keys are unique instructions.
        assert len(catalogue) == len(set(catalogue))

    def test_catalogue_generation_is_deterministic_per_seed(self):
        first = bench.generate_catalogue(random.Random(99), TEST_OPS)
        second = bench.generate_catalogue(random.Random(99), TEST_OPS)
        assert _catalogue_costs(first) == _catalogue_costs(second)
        assert (bench.catalogue_summary(first)
                == bench.catalogue_summary(second))


# ---------------------------------------------------------------------------
# Per-instance records
# ---------------------------------------------------------------------------

class TestRunInstance:
    def test_count_sites_matches_marker_in_instruction_id(self):
        # Regression guard: the premium marker must match ids like
        # "instr-d_premium" (no trailing underscore after "premium").
        catalogue = bench.generate_catalogue(random.Random(0), TEST_OPS)
        premium_key = next(k for k in catalogue if k.endswith("_premium"))
        fused_key = next(k for k in catalogue
                         if "spanned_child_ops" in catalogue[k].params)
        node = IRTreeNode(op="z")
        result = TreeSelectionResult(covering=[
            (node, TreePattern(instruction=catalogue[premium_key])),
            (node, TreePattern(instruction=catalogue[fused_key])),
            (node, TreePattern(instruction=catalogue["a"])),
        ])
        assert bench.count_sites(result, "_premium") == 1
        assert bench.count_sites(result, "_fused") == 1
        assert bench.count_sites(result, "_premium_") == 0

    def test_heuristic_picks_premium_variants_dp_never_does(self):
        # The score-first heuristic prefers the costlier premium variant
        # on payload nodes; the cost-minimising DP never does.
        record = bench.run_instance(size=20, seed=0, ops=bench.OPS, reps=2)
        assert record["dp"]["premium_sites_used"] == 0
        assert record["heuristic"]["premium_sites_used"] > 0

    def test_record_reports_both_arms_with_dp_not_costlier(self):
        record = bench.run_instance(size=12, seed=0, ops=TEST_OPS, reps=3)
        assert record["size"] == 12
        assert record["seed"] == 0
        assert record["rng_seed"] == bench.rng_seed_for(12, 0)
        assert record["tree"]["node_count"] == 12
        assert record["catalogue"]["n_single"] == len(TEST_OPS)
        for arm in ("dp", "heuristic"):
            entry = record[arm]
            assert entry["complete"] is True
            assert entry["tiling_valid"] is True
            assert entry["cost"] > 0
        assert (record["dp"]["cost"]
                <= record["heuristic"]["cost"] + bench.COST_EPS)
        gap = record["heuristic"]["cost"] - record["dp"]["cost"]
        assert record["cost_gap_heuristic_minus_dp"] == pytest.approx(gap)
        assert record["cost_ratio_dp_over_heuristic"] == pytest.approx(
            record["dp"]["cost"] / record["heuristic"]["cost"])
        assert record["dp_cost_le_heuristic"] is True
        # The cost-minimising DP never picks a premium variant (it is
        # strictly costlier than the base instruction of the same op).
        assert record["dp"]["premium_sites_used"] == 0

    def test_instance_is_deterministic_apart_from_timings(self):
        first = bench.run_instance(size=15, seed=3, ops=TEST_OPS, reps=2)
        second = bench.run_instance(size=15, seed=3, ops=TEST_OPS, reps=2)
        assert first["dp"]["cost"] == second["dp"]["cost"]
        assert first["heuristic"]["cost"] == second["heuristic"]["cost"]
        assert (first["dp"]["covering_size"]
                == second["dp"]["covering_size"])
        assert (first["dp"]["fused_sites_used"]
                == second["dp"]["fused_sites_used"])
        assert first["catalogue"] == second["catalogue"]
        assert first["tree"] == second["tree"]

    def test_timing_fields_report_the_median_of_recorded_reps(self):
        record = bench.run_instance(size=10, seed=1, ops=TEST_OPS, reps=3)
        for arm in ("dp", "heuristic"):
            reps = record[arm]["time_ms_reps"]
            assert len(reps) == 3
            assert all(t > 0 for t in reps)
            assert record[arm]["time_ms_median"] == pytest.approx(
                statistics.median(reps), abs=1e-6)
            assert record[arm]["time_ms_min"] == pytest.approx(
                min(reps), abs=1e-6)
            assert record[arm]["time_ms_max"] == pytest.approx(
                max(reps), abs=1e-6)
        assert record["time_ratio_dp_over_heuristic_median"] == pytest.approx(
            record["dp"]["time_ms_median"]
            / record["heuristic"]["time_ms_median"])


# ---------------------------------------------------------------------------
# Aggregation discipline (aggregates must equal statistics over raw records)
# ---------------------------------------------------------------------------

class TestAggregationDiscipline:
    def _records(self):
        return [bench.run_instance(size=size, seed=seed, ops=TEST_OPS, reps=2)
                for size in (8, 13) for seed in (0, 1, 2)]

    def test_aggregates_equal_statistics_over_raw_records(self):
        records = self._records()
        agg = bench.aggregate_instances(records)
        assert agg["n_instances"] == 6
        dp = [r["dp"]["cost"] for r in records]
        heur = [r["heuristic"]["cost"] for r in records]
        gaps = [r["cost_gap_heuristic_minus_dp"] for r in records]
        ratios = [r["cost_ratio_dp_over_heuristic"] for r in records]
        assert agg["dp_cost_mean"] == _r6(statistics.mean(dp))
        assert agg["dp_cost_stdev"] == _r6(statistics.stdev(dp))
        assert agg["heuristic_cost_mean"] == _r6(statistics.mean(heur))
        assert agg["heuristic_cost_stdev"] == _r6(statistics.stdev(heur))
        assert agg["cost_gap_mean"] == _r6(statistics.mean(gaps))
        assert agg["cost_gap_stdev"] == _r6(statistics.stdev(gaps))
        assert agg["cost_ratio_dp_over_heuristic_mean"] == _r6(
            statistics.mean(ratios))
        assert agg["cost_ratio_dp_over_heuristic_min"] == _r6(min(ratios))
        assert agg["cost_ratio_dp_over_heuristic_max"] == _r6(max(ratios))
        eps = bench.COST_EPS
        wins = sum(1 for r in records
                   if r["dp"]["cost"] < r["heuristic"]["cost"] - eps)
        ties = sum(1 for r in records
                   if abs(r["cost_gap_heuristic_minus_dp"]) <= eps)
        assert agg["dp_wins"] == wins
        assert agg["cost_ties"] == ties
        assert agg["dp_cost_losses"] == 0
        assert agg["dp_cost_le_heuristic_all"] is True
        depths = [r["tree"]["depth"] for r in records]
        assert agg["tree_depth_mean"] == _r6(statistics.mean(depths))
        assert agg["tree_depth_max"] == max(depths)

    def test_timing_aggregates_use_the_per_instance_medians(self):
        records = self._records()
        agg = bench.aggregate_instances(records)
        dp_ms = [r["dp"]["time_ms_median"] for r in records]
        heur_ms = [r["heuristic"]["time_ms_median"] for r in records]
        ratios = [r["time_ratio_dp_over_heuristic_median"] for r in records]
        assert agg["dp_time_ms_mean"] == _r6(statistics.mean(dp_ms))
        assert agg["dp_time_ms_stdev"] == _r6(statistics.stdev(dp_ms))
        assert agg["heuristic_time_ms_mean"] == _r6(statistics.mean(heur_ms))
        assert agg["heuristic_time_ms_stdev"] == _r6(statistics.stdev(heur_ms))
        assert agg["time_ratio_dp_over_heuristic_mean"] == _r6(
            statistics.mean(ratios))

    def test_mean_stdev_of_a_single_value_is_zero_stdev(self):
        assert bench.mean_stdev([3.25]) == (3.25, 0.0)
        assert bench.mean_stdev([1.0, 3.0]) == (2.0, pytest.approx(
            statistics.stdev([1.0, 3.0])))


# ---------------------------------------------------------------------------
# Report structure and coverage honesty
# ---------------------------------------------------------------------------

class TestReportAndCoverage:
    def test_report_carries_provenance_and_complete_coverage(self):
        report = bench.run_benchmark(
            sizes=(6,), seeds=(0, 1), ops=TEST_OPS, reps=2,
            component_check="unit-test check string")
        prov = report["provenance"]
        assert isinstance(prov["git_rev"], str) and prov["git_rev"]
        datetime.fromisoformat(prov["timestamp_utc"])
        assert prov["seeds"] == [0, 1]
        assert prov["sizes"] == [6]
        assert "random.Random" in prov["rng_seed_formula"]
        assert prov["component"]["pre_bench_component_check"] == (
            "unit-test check string")
        assert prov["component"]["dp_arm"].startswith(
            "InstructionSelector(selection_strategy='dp')")
        assert prov["python_version"]
        cov = report["coverage"]
        assert cov["instances_planned"] == 2
        assert cov["instances_completed"] == 2
        assert cov["status"] == "complete"
        assert cov["failed_instances"] == []
        assert len(report["instances"]) == 2
        assert report["by_size"]["6"]["n_instances"] == 2
        assert report["overall"]["n_instances"] == 2

    def test_failed_instances_mark_coverage_partial_and_stay_out_of_aggregates(
            self, monkeypatch):
        original = bench.run_instance

        def flaky(size, seed, **kwargs):
            if seed == 1:
                raise RuntimeError("boom")
            return original(size, seed, **kwargs)

        monkeypatch.setattr(bench, "run_instance", flaky)
        report = bench.run_benchmark(sizes=(6,), seeds=(0, 1), ops=TEST_OPS,
                                     reps=2)
        cov = report["coverage"]
        assert cov["status"] == "partial"
        assert cov["instances_completed"] == 1
        assert len(cov["failed_instances"]) == 1
        failed = cov["failed_instances"][0]
        assert failed["size"] == 6 and failed["seed"] == 1
        assert "boom" in failed["detail"]
        assert len(report["instances"]) == 1
        assert report["by_size"]["6"]["n_instances"] == 1
        assert report["overall"]["n_instances"] == 1

    def test_summarize_mentions_sizes_and_coverage_status(self):
        report = bench.run_benchmark(sizes=(6,), seeds=(0,), ops=TEST_OPS,
                                     reps=2)
        text = bench.summarize(report)
        assert "coverage: complete" in text
        assert "6" in text
        assert "dp_cost_losses" in text or "losses" in text


class TestMainCLI:
    def test_main_writes_json_and_exits_zero(self, tmp_path):
        out = tmp_path / "out.json"
        rc = bench.main(["--sizes", "6", "--seeds", "0", "--reps", "2",
                         "--output", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        for key in ("benchmark", "description", "provenance", "coverage",
                    "instances", "by_size", "overall", "notes"):
            assert key in data
        assert data["provenance"]["seeds"] == [0]
        assert data["coverage"]["status"] == "complete"

    def test_main_exits_non_zero_when_coverage_is_partial(
            self, tmp_path, monkeypatch):
        original = bench.run_instance

        def flaky(size, seed, **kwargs):
            if seed == 1:
                raise RuntimeError("boom")
            return original(size, seed, **kwargs)

        monkeypatch.setattr(bench, "run_instance", flaky)
        out = tmp_path / "partial.json"
        rc = bench.main(["--sizes", "6", "--seeds", "0", "1", "--reps", "2",
                         "--output", str(out)])
        assert rc == 1
        data = json.loads(out.read_text())
        assert data["coverage"]["status"] == "partial"


# ---------------------------------------------------------------------------
# Generated artifact integrity (the actual deliverable)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated yet")
class TestArtifactIntegrity:
    def test_artifact_aggregates_match_its_raw_instances(self):
        data = json.loads(ARTIFACT.read_text())
        assert data["coverage"]["status"] == "complete"
        assert data["coverage"]["instances_completed"] == 20
        sizes = data["provenance"]["sizes"]
        seeds = data["provenance"]["seeds"]
        assert sizes == [20, 50, 100, 200]
        assert len(seeds) == 5
        assert len(data["instances"]) == 20
        all_dp, all_heur = [], []
        for size in sizes:
            recs = [r for r in data["instances"] if r["size"] == size]
            assert len(recs) == len(seeds)
            assert {r["seed"] for r in recs} == set(seeds)
            agg = data["by_size"][str(size)]
            dp = [r["dp"]["cost"] for r in recs]
            heur = [r["heuristic"]["cost"] for r in recs]
            all_dp.extend(dp)
            all_heur.extend(heur)
            assert agg["dp_cost_mean"] == _r6(statistics.mean(dp))
            assert agg["dp_cost_stdev"] == _r6(statistics.stdev(dp))
            assert agg["heuristic_cost_mean"] == _r6(statistics.mean(heur))
            assert agg["heuristic_cost_stdev"] == _r6(statistics.stdev(heur))
            gaps = [r["cost_gap_heuristic_minus_dp"] for r in recs]
            assert agg["cost_gap_mean"] == _r6(statistics.mean(gaps))
            assert agg["cost_ratio_dp_over_heuristic_mean"] == _r6(
                statistics.mean([r["cost_ratio_dp_over_heuristic"]
                                 for r in recs]))
            assert agg["dp_cost_losses"] == 0
            assert agg["dp_cost_le_heuristic_all"] is True
            assert all(r["dp"]["complete"] and r["heuristic"]["complete"]
                       for r in recs)
            assert all(r["dp"]["tiling_valid"]
                       and r["heuristic"]["tiling_valid"]
                       for r in recs)
        overall = data["overall"]
        assert overall["n_instances"] == 20
        assert overall["dp_cost_mean"] == _r6(statistics.mean(all_dp))
        assert overall["heuristic_cost_mean"] == _r6(statistics.mean(all_heur))
        assert overall["dp_cost_losses"] == 0
        assert overall["dp_cost_le_heuristic_all"] is True
        # The benchmark actually exercises the premium mechanic: the
        # score-first heuristic picks costlier premium variants while
        # the cost-minimising DP never does.
        assert overall["heuristic_premium_sites_total"] > 0
        assert overall["dp_premium_sites_total"] == 0
        assert overall["dp_fused_sites_total"] > 0
        assert overall["heuristic_fused_sites_total"] == 0

    def test_artifact_provenance_block_is_complete(self):
        data = json.loads(ARTIFACT.read_text())
        prov = data["provenance"]
        assert isinstance(prov["git_rev"], str) and prov["git_rev"]
        datetime.fromisoformat(prov["timestamp_utc"])
        assert prov["sizes"] == [20, 50, 100, 200]
        assert len(prov["seeds"]) == 5
        assert prov["timing"]["reps"] >= 5
        assert "37 passed" in prov["component"]["pre_bench_component_check"]
