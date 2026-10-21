"""Tests for the fail-open vs fail-closed security-posture matrix (E5).

``examples/benchmarks/security_posture_matrix.py`` runs the scenario x
enforcement-mode matrix against the real ``SecurityMonitor`` /
``AgentCoordinator`` (the I4 strict flag) and writes
``examples/benchmarks/security_posture_results.json``.  These tests pin
the mechanics offline: scenario registry shape, task-generation
determinism, the measured posture per scenario, mode-divergence pairing,
violation recording, coordinator propagation, and the honesty rules of
the output document (aggregates re-derivable from the embedded raw
table; counting only, no performance quantities).
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

import pytest

# Make the benchmark script importable (it lives in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import security_posture_matrix as spm  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario registry
# ---------------------------------------------------------------------------

class TestScenarioRegistry:
    def test_required_scenarios_present_as_main_rows(self):
        by_id = {s["scenario"]: s for s in spm.SCENARIOS}
        for sid in spm.REQUIRED_SCENARIO_IDS:
            assert sid in by_id
            assert by_id[sid]["matrix_role"] == "main"
            assert len(by_id[sid]["variants"]) >= 2

    def test_supplementary_malformed_registry_row_exists(self):
        by_id = {s["scenario"]: s for s in spm.SCENARIOS}
        suppl = [s for s in spm.SCENARIOS if s["matrix_role"] != "main"]
        assert [s["scenario"] for s in suppl] == [
            "malformed_registry_supplementary"]
        assert len(suppl[0]["variants"]) == 1

    def test_held_constant_dimensions(self):
        # Resource scenarios declare known sensitivity; sensitivity
        # scenarios request no resources (exactly one axis varies).
        rng = random.Random("const-check")
        for name in (
            "single_unregistered", "mixed_registered_and_unregistered",
            "string_form_request", "registry_declared_as_list",
            "registry_key_absent", "registry_explicitly_none",
            "multi_resource_request_no_registry",
            "string_resource_no_registry",
        ):
            gen = next(
                g for s in spm.SCENARIOS
                for n, g in s["variants"].items() if n == name
            )
            for _ in range(5):
                _config, task, _level = gen(rng)
                assert task["data_sensitivity"] == 0
        for name in ("no_declaration", "empty_category_list",
                     "non_numeric_declaration", "negative_out_of_range"):
            gen = next(
                g for s in spm.SCENARIOS
                for n, g in s["variants"].items() if n == name
            )
            for _ in range(5):
                _config, task, _level = gen(rng)
                assert not task.get("resources")


# ---------------------------------------------------------------------------
# Task population determinism
# ---------------------------------------------------------------------------

class TestTaskPopulation:
    def test_same_seeds_build_identical_population(self):
        a = spm.build_tasks([11, 12], 5)
        b = spm.build_tasks([11, 12], 5)
        assert a == b

    def test_different_seeds_change_task_surface(self):
        a = spm.build_tasks([11], 5)
        b = spm.build_tasks([12], 5)
        ids_a = {t["task"]["id"] for t in a}
        ids_b = {t["task"]["id"] for t in b}
        assert not (ids_a & ids_b)

    def test_population_size_matches_scale(self):
        seeds, n = [1, 2, 3], 4
        pop = spm.build_tasks(seeds, n)
        assert len(pop) == len(spm.SCENARIOS) * len(seeds) * n


# ---------------------------------------------------------------------------
# Measured posture per scenario (the functional matrix itself)
# ---------------------------------------------------------------------------

def _small_raw():
    pop = spm.build_tasks([0, 1], 6)
    monitor_raw = spm.run_monitor_matrix(pop)
    coord_raw = spm.run_coordinator_matrix(pop)
    return monitor_raw, coord_raw


class TestMeasuredPosture:
    @pytest.fixture(scope="class")
    def raw(self):
        return _small_raw()[0]

    def _cell(self, raw, scenario, mode):
        recs = [r for r in raw
                if r["scenario"] == scenario and r["mode"] == mode]
        assert recs
        return recs

    def test_unregistered_resource_denied_in_both_modes(self, raw):
        for mode in spm.MODES:
            recs = self._cell(raw, "unregistered_resource", mode)
            assert all(not r["allowed"] for r in recs)
        fo = self._cell(raw, "unregistered_resource", "fail_open")
        fc = self._cell(raw, "unregistered_resource", "fail_closed")
        assert all(r["reason_class"] == "generic_invalid_resource_access"
                   for r in fo)
        assert all(r["reason_class"] == "unregistered_resources_strict"
                   for r in fc)

    def test_unknown_sensitivity_flips_allow_to_deny(self, raw):
        fo = self._cell(raw, "unknown_sensitivity", "fail_open")
        fc = self._cell(raw, "unknown_sensitivity", "fail_closed")
        assert all(r["allowed"] for r in fo)
        assert all(r["reason_class"] == "allowed" for r in fo)
        assert all(not r["allowed"] for r in fc)
        assert all(r["reason_class"] == "unknown_sensitivity_strict"
                   for r in fc)

    def test_incomplete_policy_flips_allow_to_deny(self, raw):
        fo = self._cell(raw, "incomplete_policy", "fail_open")
        fc = self._cell(raw, "incomplete_policy", "fail_closed")
        assert all(r["allowed"] for r in fo)
        assert all(not r["allowed"] for r in fc)
        assert all(r["reason_class"] == "incomplete_policy_strict"
                   for r in fc)

    def test_normal_task_allowed_in_both_modes_without_violations(self, raw):
        for mode in spm.MODES:
            recs = self._cell(raw, "normal_task", mode)
            assert all(r["allowed"] for r in recs)
            assert all(r["reason_class"] == "allowed" for r in recs)

    def test_supplementary_malformed_registry_denied_in_both_modes(self, raw):
        fo = self._cell(raw, "malformed_registry_supplementary", "fail_open")
        fc = self._cell(raw, "malformed_registry_supplementary", "fail_closed")
        assert all(not r["allowed"] for r in fo)
        assert all(not r["allowed"] for r in fc)
        assert all(r["reason_class"] == "generic_invalid_resource_access"
                   for r in fo)
        assert all(r["reason_class"] == "incomplete_policy_strict"
                   for r in fc)

    def test_every_denial_recorded_and_strict_denials_marked(self, raw):
        for r in raw:
            if not r["allowed"]:
                assert r["violation_recorded"]
                expected_marker = r["mode"] == "fail_closed"
                assert r["strict_marker"] is expected_marker


# ---------------------------------------------------------------------------
# Mode divergence pairing
# ---------------------------------------------------------------------------

class TestModeDivergence:
    def test_contingency_matches_scenario_semantics(self):
        seeds, n = [0, 1], 6
        pop = spm.build_tasks(seeds, n)
        raw = spm.run_monitor_matrix(pop)
        div = {row["scenario"]: row for row in
               spm.aggregate_mode_divergence(raw, seeds, n)}
        n_pairs = len(seeds) * n

        for sid in ("unknown_sensitivity", "incomplete_policy"):
            c = div[sid]["outcome_contingency"]
            assert c["fail_open_only"] == n_pairs
            assert c["both_allow"] == c["both_deny"] == \
                c["fail_closed_only"] == 0
            assert div[sid]["outcome_agreement_rate"] == 0.0

        for sid in ("unregistered_resource",
                    "malformed_registry_supplementary"):
            c = div[sid]["outcome_contingency"]
            assert c["both_deny"] == n_pairs
            # Same verdict, different stated reason class.
            assert div[sid][
                "reason_class_agreement_rate_among_both_denied"] == 0.0

        c = div["normal_task"]["outcome_contingency"]
        assert c["both_allow"] == n_pairs
        assert div["normal_task"]["outcome_agreement_rate"] == 1.0


# ---------------------------------------------------------------------------
# Coordinator propagation
# ---------------------------------------------------------------------------

class TestCoordinatorPropagation:
    def test_coordinator_gates_exactly_where_monitor_denies(self):
        seeds, n = [0, 1], 6
        pop = spm.build_tasks(seeds, n)
        monitor_raw = spm.run_monitor_matrix(pop)
        coord_raw = spm.run_coordinator_matrix(pop)
        coord_cells = {(c["scenario"], c["mode"]): c for c in
                       spm.aggregate_coordinator(coord_raw, seeds)}
        for scenario in spm.SCENARIOS:
            for mode in spm.MODES:
                sid = scenario["scenario"]
                allowed = sum(
                    1 for r in monitor_raw
                    if r["scenario"] == sid and r["mode"] == mode
                    and r["allowed"]
                )
                cell = coord_cells[(sid, mode)]
                assert cell["executed"] == allowed
                assert cell["security_error_raised"] == cell["n_runs"] - allowed
                if cell["security_error_raised"]:
                    assert cell[
                        "violation_recorded_rate_among_security_errors"] == 1.0
                else:
                    assert cell[
                        "violation_recorded_rate_among_security_errors"] == 0.0


# ---------------------------------------------------------------------------
# Output document honesty rules
# ---------------------------------------------------------------------------

class TestResultDocument:
    @pytest.fixture(scope="class")
    def doc(self, tmp_path_factory):
        seeds, n = [3, 4], 5
        doc = spm.run_experiment(seeds, n)
        # Make the "same document" guarantee explicit: json round-trip.
        path = tmp_path_factory.mktemp("spm") / "results.json"
        path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_provenance_block(self, doc):
        prov = doc["provenance"]
        assert re.fullmatch(r"[0-9a-f]{7,40}|unknown", prov["git_rev"])
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:00",
                            prov["timestamp_utc"])
        assert prov["seeds"] == [3, 4]
        scale = prov["scale"]
        n_pop = scale["task_population"]
        assert n_pop == 5 * 2 * 5  # scenarios x seeds x tasks
        assert scale["monitor_evaluations"] == n_pop * 2
        assert scale["coordinator_runs"] == n_pop * 2

    def test_every_cell_counts_add_up(self, doc):
        n_expected = 2 * 5  # seeds x tasks per scenario per mode
        for cell in doc["matrix"]:
            assert cell["n_tasks"] == n_expected
            assert cell["allowed"] + cell["denied"] == cell["n_tasks"]
            assert cell["allow_rate"] == pytest.approx(
                cell["allowed"] / cell["n_tasks"])
            assert sum(v["n"] for v in cell["per_variant"]) == cell["n_tasks"]
            assert sum(v["allowed"] for v in cell["per_variant"]) == \
                cell["allowed"]
            total_reasons = sum(cell["reason_class_counts"].values())
            assert total_reasons == cell["n_tasks"]

    def test_aggregates_rederivable_from_embedded_raw_table(self, doc):
        table = doc["raw_counts"]["monitor"]
        assert all(row["allowed"] + row["denied"] == row["n"] for row in table)
        for cell in doc["matrix"]:
            rows = [r for r in table if r["scenario"] == cell["scenario"]
                    and r["mode"] == cell["mode"]]
            assert sum(r["n"] for r in rows) == cell["n_tasks"]
            assert sum(r["allowed"] for r in rows) == cell["allowed"]
            assert sum(r["denied"] for r in rows) == cell["denied"]
            # Per-variant split also re-derives.
            for v in cell["per_variant"]:
                vrows = [r for r in rows if r["variant"] == v["variant"]]
                assert sum(r["n"] for r in vrows) == v["n"]
                assert sum(r["allowed"] for r in vrows) == v["allowed"]

    def test_posture_summary_consistent_with_contingency(self, doc):
        div = {row["scenario"]: row for row in doc["mode_divergence"]}
        summary = doc["posture_summary"]
        flips = sum(div[sid]["outcome_contingency"]["fail_open_only"]
                    for sid in summary[
                        "hole_scenarios_fail_open_permits_fail_closed_blocks"])
        assert summary[
            "tasks_fail_open_would_permit_but_fail_closed_blocks"] == flips
        assert summary["normal_task_fail_open_denied"] == 0
        assert summary["normal_task_fail_closed_denied"] == 0

    def test_counting_only_no_performance_quantities(self, doc):
        forbidden = {"latency_ms", "wall_ms", "duration", "throughput",
                     "speedup", "tokens", "memory_overhead", "api_calls",
                     "time_s", "ms_total"}
        blocks = (doc["matrix"] + doc["mode_divergence"]
                  + doc["coordinator_propagation"] + [doc["posture_summary"]])
        for block in blocks:
            overlap = forbidden & set(block)
            assert not overlap, f"performance-like key {overlap}"
        assert any(
            re.search(r"[Nn]o .{0,80}performance quantity is measured "
                      r"or claimed", note)
            for note in doc["notes"]
        )

    def test_coverage_block_states_dependency_and_limits(self, doc):
        cov = doc["coverage"]
        assert cov["dependency"]["job"] == "I4-security-failclosed"
        assert cov["dependency"]["status"] == "shipped"
        assert cov["coverage_status"] == "full"
        assert cov["covered"] and cov["not_covered"]

    def test_per_seed_stdev_is_zero_for_deterministic_decisions(self, doc):
        # Deterministic decision function: per-seed allow rates within a
        # cell are identical, so the script-computed stdev must be 0.0.
        for cell in doc["matrix"]:
            values = cell["allow_rate_per_seed"]["values"]
            assert len(values) == 2
            assert cell["allow_rate_per_seed"]["stdev"] == 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_main_writes_output_file(self, tmp_path, capsys):
        out = tmp_path / "security_posture_results.json"
        rc = spm.main(["--seeds", "2", "--tasks-per-scenario", "3",
                       "--output", str(out)])
        assert rc == 0
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["experiment"] == "security_posture_matrix"
        assert doc["config"]["seeds"] == [11, 12]
        assert doc["provenance"]["git_rev"]
        printed = capsys.readouterr().out
        assert "wrote" in printed
        assert "both_allow" in printed

    def test_default_seed_count_is_five(self, tmp_path):
        out = tmp_path / "results.json"
        spm.main(["--seeds", "5", "--tasks-per-scenario", "2",
                  "--output", str(out)])
        doc = json.loads(out.read_text(encoding="utf-8"))
        assert doc["config"]["n_seeds"] == 5
        assert doc["provenance"]["scale"]["monitor_evaluations"] == \
            5 * 5 * 2 * 2


# ---------------------------------------------------------------------------
# Reason classifier
# ---------------------------------------------------------------------------

class TestClassifyReason:
    @pytest.mark.parametrize("message,expected", [
        (None, "allowed"),
        ("Invalid resource access", "generic_invalid_resource_access"),
        ("Unregistered resources (strict mode): ['shadow_bucket']",
         "unregistered_resources_strict"),
        ("Incomplete resource-access policy: no allowed_resources "
         "registry declared (strict mode)", "incomplete_policy_strict"),
        ("Incomplete resource-access policy: allowed_resources must be "
         "a set or sequence, got str (strict mode)",
         "incomplete_policy_strict"),
        ("Unknown data sensitivity: no declaration (strict mode)",
         "unknown_sensitivity_strict"),
        ("Unknown data sensitivity declaration: 'high' (strict mode)",
         "unknown_sensitivity_strict"),
        ("Unknown data sensitivity level: -1 (strict mode)",
         "unknown_sensitivity_strict"),
        ("Data sensitivity mismatch", "sensitivity_mismatch"),
        ("Missing authentication", "missing_authentication"),
        ("Missing authorization", "missing_authorization"),
        ("something unseen", "other"),
    ])
    def test_mapping(self, message, expected):
        assert spm.classify_reason(message is None, message) == expected
