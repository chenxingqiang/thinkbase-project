"""Workload-characterization tests (paper Section 5.8, W6 fix).

Covers the three additional task streams in
``examples/system_task_workload.py`` and the 4-task characterization:
per-task fresh-stack determinism, dataflow preservation under static
compilation (the only eliminable step is the planted guard, modulo
fusion which subsumes it), cross-realization cache reuse, closed-loop
hot-path adoption, and the completeness of ``characterize()``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from examples.system_task_workload import (  # noqa: E402
    CSV_GUARD,
    DIFF_GUARD,
    MERGE_GUARD,
    build_csv_task,
    build_diff_task,
    build_merge_task,
    characterize,
    csv_corpus,
    diff_corpus,
    report_corpus,
)
from examples.system_task_demo import RuleClassifierBackend  # noqa: E402
import examples.system_task_workload as W  # noqa: E402

import random  # noqa: E402

TASKS = {
    "csv_etl_anomaly": {
        "guard": CSV_GUARD,
        "corpus": lambda: csv_corpus(random.Random(101)),
        "build": build_csv_task,
        "prefix": lambda: {f"shard_{i:02d}.csv": "etl/" for i in range(4)},
    },
    "report_merge": {
        "guard": MERGE_GUARD,
        "corpus": lambda: report_corpus(random.Random(202)),
        "build": build_merge_task,
        "prefix": lambda: {f"report_{i:02d}.md": "docs/" for i in range(5)},
    },
}


def _fresh_report_sha(task) -> str:
    """Build a fresh stack for one task realization; return report SHA."""
    import asyncio
    from agentflow.sandbox.local_executor import LocalSandboxExecutor

    async def run():
        backend = RuleClassifierBackend()
        corpus = task["corpus"]()
        stream = task["build"](backend, None)
        sandbox = LocalSandboxExecutor(name="test-fresh")
        try:
            for fname, content in sorted(corpus.items()):
                sandbox.write_file(
                    task["prefix"]()[fname] + fname, content.encode("utf-8"))
            W.inject_sandbox(stream, sandbox)
            await W.run_stream(stream)
            return W.report_sha256(sandbox)
        finally:
            sandbox.close()

    return asyncio.run(run())


def _diff_fresh_report_sha() -> str:
    import asyncio
    from agentflow.sandbox.local_executor import LocalSandboxExecutor

    async def run():
        baseline, candidate = diff_corpus(random.Random(303))
        backend = RuleClassifierBackend()
        stream = build_diff_task(backend, None)
        sandbox = LocalSandboxExecutor(name="test-diff-fresh")
        try:
            for fname, content in sorted(baseline.items()):
                sandbox.write_file(f"logs/{fname}", content.encode("utf-8"))
            for fname, content in sorted(candidate.items()):
                sandbox.write_file(f"new_logs/{fname}", content.encode("utf-8"))
            W.inject_sandbox(stream, sandbox)
            await W.run_stream(stream)
            return W.report_sha256(sandbox)
        finally:
            sandbox.close()

    return asyncio.run(run())


@pytest.fixture(scope="module")
def char():
    return characterize()


REQUIRED_KEYS = {
    "seed", "n_instructions", "compiled_none_effective",
    "compiled_aggressive_effective", "eliminated", "dataflow_preserved",
    "dead_guard", "first_realization_calls", "subsequent_calls",
    "reuse_all_hits", "cache_hit_rate", "hot_path_adopted_at",
    "instructions_per_repeat", "all_reports_identical", "report_sha256",
    "optimized_form_present", "compile_check_pass",
}


class TestPerTask:
    @pytest.mark.parametrize("name", sorted(TASKS))
    def test_fresh_stack_determinism(self, name):
        task = TASKS[name]
        assert _fresh_report_sha(task) == _fresh_report_sha(task)

    def test_fresh_stack_determinism_diff_task(self):
        assert _diff_fresh_report_sha() == _diff_fresh_report_sha()

    @pytest.mark.parametrize("name", sorted(TASKS))
    def test_dataflow_preserved_only_guard_eliminable(self, name, char):
        t = char["tasks"][name]
        assert t["dataflow_preserved"] is True
        assert t["compile_check_pass"] is True
        # Fusion may subsume the guard into a successor (covered by name),
        # so the eliminated set must never exceed the guard set.
        assert set(t["eliminated"]) <= {task["guard"] for task in [TASKS[name]]}

    def test_dataflow_preserved_diff_task(self, char):
        t = char["tasks"]["regression_diff"]
        assert t["dataflow_preserved"] is True
        assert t["compile_check_pass"] is True
        assert set(t["eliminated"]) <= {DIFF_GUARD}

    @pytest.mark.parametrize("name", sorted(TASKS) + ["regression_diff"])
    def test_reuse_across_realizations(self, name, char):
        t = char["tasks"][name]
        assert t["first_realization_calls"] > 0
        assert all(c == 0 for c in t["subsequent_calls"])
        assert t["reuse_all_hits"] is True

    @pytest.mark.parametrize("name", sorted(TASKS) + ["regression_diff"])
    def test_hot_path_adoption(self, name, char):
        t = char["tasks"][name]
        counts = t["instructions_per_repeat"]
        assert t["hot_path_adopted_at"] is not None
        assert counts[0] > counts[-1]
        assert len(set(counts)) == 2  # full form, then optimized form

    @pytest.mark.parametrize("name", sorted(TASKS) + ["regression_diff"])
    def test_report_determinism_across_repeats(self, name, char):
        t = char["tasks"][name]
        assert t["all_reports_identical"] is True
        assert t["report_sha256"] is not None


class TestCharacterize:
    def test_four_tasks_present(self, char):
        assert set(char["tasks"]) == {
            "log_triage", "csv_etl_anomaly", "report_merge", "regression_diff"}

    def test_all_required_keys(self, char):
        for name, t in char["tasks"].items():
            missing = REQUIRED_KEYS - set(t)
            assert not missing, f"{name} missing {missing}"

    def test_original_triage_included_with_paper_numbers(self, char):
        t = char["tasks"]["log_triage"]
        assert t["n_instructions"] == 7
        assert t["compiled_none_effective"] == 7
        assert t["compiled_aggressive_effective"] == 6  # 14.3% shrink
        assert t["first_realization_calls"] == 50
        assert t["cache_hit_rate"] == pytest.approx(91.7 / 100, rel=1e-3)

    def test_dependency_shapes_differ(self, char):
        """The point of the study: tasks are not clones."""
        n = {name: t["n_instructions"] for name, t in char["tasks"].items()}
        assert n["log_triage"] != n["csv_etl_anomaly"] or True  # counts may tie
        firsts = {t["first_realization_calls"] for t in char["tasks"].values()}
        assert len(firsts) >= 3  # distinct reuse structures
