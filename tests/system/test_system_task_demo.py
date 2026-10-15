"""Determinism and repeatability tests for the system-level task demo.

These tests pin the controllability properties the demonstration
claims: the same seed realizes the same system-level task
byte-for-byte, repeated realizations are served from cache, and the
hot-path-optimized instruction stream preserves the task's observable
output.

Runs fully offline (deterministic rule-based classifier backend, real
sandbox file I/O in temp dirs).
"""

import asyncio
import sys
from pathlib import Path

import pytest

# Make the examples module importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from system_task_demo import run_demo  # noqa: E402


@pytest.fixture(scope="module")
def demo_results():
    return asyncio.run(run_demo(
        seed=7, files=3, lines_per_file=60, repeats=6, hot_threshold=3,
    ))


class TestControllability:
    def test_deterministic_across_fresh_stacks(self, demo_results):
        det = demo_results["determinism"]
        assert det["byte_identical"] is True
        assert det["run1_sha256"] == det["run2_sha256"]
        assert len(det["run1_sha256"]) == 64  # sha256 hex digest

    def test_all_repeats_identical(self, demo_results):
        rep = demo_results["repeats"]
        assert rep["all_reports_identical"] is True
        assert rep["report_sha256"] is not None

    def test_cache_serves_repeats(self, demo_results):
        rep = demo_results["repeats"]
        calls = rep["per_repeat_model_calls"]
        assert len(calls) == 6
        assert calls[0] > 0  # first realization actually classifies
        assert all(c == 0 for c in calls[1:])
        assert rep["repeats_served_from_cache"] == 5
        assert rep["cache_hit_rate"] == pytest.approx(5 / 6)

    def test_verdict_all_pass(self, demo_results):
        assert all(demo_results["verdict"].values())


class TestHotPathOptimization:
    def test_closed_loop_adoption(self, demo_results):
        """The runtime must adopt the optimized form automatically."""
        cl = demo_results["repeats"]["closed_loop"]
        # threshold=3 -> trigger at the 4th recording -> adoption from
        # repeat 5 onward; repeats=6 -> [7,7,7,7,6,6].
        assert cl["instructions_per_repeat"] == [7, 7, 7, 7, 6, 6]
        assert cl["optimized_form_adopted_at_repeat"] == 5
        assert cl["optimized_stream_executed_repeats"] == 2

    def test_dead_guard_eliminated(self, demo_results):
        hot = demo_results["hot_path"]
        # The dead defensive guard is the only eliminable instruction.
        assert hot["eliminated"] == ["check_syntax"]

    def test_optimized_stream_preserves_output(self, demo_results):
        hot = demo_results["hot_path"]
        assert hot["shrink_ratio"] == pytest.approx(1 / 7)
        assert hot["optimized_report_matches"] is True


class TestStaticCompilation:
    def test_aggressive_reduces_real_stream(self, demo_results):
        sc = demo_results["static_compile"]
        assert sc["stream_length"] == 7
        none = sc["NONE"]["effective_instruction_count"]
        aggressive = sc["AGGRESSIVE"]["effective_instruction_count"]
        assert none == 7
        assert aggressive < none
        assert sc["effective_reduction"] == pytest.approx(1 - aggressive / none)

    def test_compiled_stream_preserves_dataflow(self, demo_results):
        """Every step whose output is consumed downstream must survive.

        Guards against the DCE dependency-naming failure mode where a
        mismatch between declared dependencies and output names makes
        the compiler drop live producers of consumed data.
        """
        sc = demo_results["static_compile"]
        for level in ("NONE", "AGGRESSIVE"):
            assert sc[level]["dataflow_preserved"] is True
            assert sc[level]["missing_producers"] == []

    def test_dispatch_count_consistent(self, demo_results):
        sc = demo_results["static_compile"]
        for level in ("NONE", "AGGRESSIVE"):
            assert sc[level]["dispatch_count"] <= sc[level]["effective_instruction_count"]


class TestSeedSensitivity:
    def test_different_seed_changes_corpus(self, demo_results):
        """Different seed -> different report (control is meaningful)."""
        other = asyncio.run(run_demo(
            seed=7 + 1, files=3, lines_per_file=60, repeats=2, hot_threshold=1,
        ))
        assert (other["repeats"]["report_sha256"]
                != demo_results["repeats"]["report_sha256"])

    def test_same_seed_reproduces_across_processes(self):
        """Two independent runs with identical config -> identical output."""
        first = asyncio.run(run_demo(
            seed=11, files=2, lines_per_file=40, repeats=2, hot_threshold=1,
        ))
        second = asyncio.run(run_demo(
            seed=11, files=2, lines_per_file=40, repeats=2, hot_threshold=1,
        ))
        assert (first["repeats"]["report_sha256"]
                == second["repeats"]["report_sha256"])
        assert (first["determinism"]["run1_sha256"]
                == second["determinism"]["run1_sha256"])
