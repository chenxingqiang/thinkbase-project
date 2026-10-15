"""Tests for the randomized multi-task stress harness."""

import asyncio
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "examples")

from random_tasks_stress import (  # noqa: E402
    TaskSpec, build_task_stream, compile_check,
)
from system_task_demo import RuleClassifierBackend, generate_corpus  # noqa: E402


class TestTaskSpecGenerator:
    def test_same_seed_same_specs(self):
        """The generator is deterministic: same seed -> same 100 specs."""
        a = [TaskSpec(random.Random(2026), i).to_dict() for i in range(100)]
        b = [TaskSpec(random.Random(2026), i).to_dict() for i in range(100)]
        assert a == b

    def test_shapes_vary_across_tasks(self):
        """Different tasks must actually differ (not 100 clones)."""
        rng = random.Random(2026)  # one RNG, successive specs
        specs = [TaskSpec(rng, i) for i in range(100)]
        shapes = {(tuple(s.chain), tuple(s.guards)) for s in specs}
        assert len(shapes) > 50  # varied, not degenerate

    def test_spec_bounds(self):
        rng = random.Random(2026)
        specs = [TaskSpec(rng, i) for i in range(100)]
        for s in specs:
            assert 2 <= len(s.chain) <= 6
            assert len(s.guards) <= 3
            for g in s.guards:
                assert 0 <= g <= len(s.chain)


class TestBuildTaskStream:
    def test_stream_shape(self):
        spec = TaskSpec(random.Random(1), 0)
        backend = RuleClassifierBackend()
        corpus = generate_corpus(random.Random(11), spec.n_files,
                                 spec.lines_per_file)
        stream = build_task_stream(spec, backend, None, corpus)
        # ingest + chain steps (+ guards) + write_report
        n_guards = len(spec.guards)
        assert len(stream) == 1 + len(spec.chain) + n_guards + 1
        ids = [i.id for i in stream]
        assert ids[0] == "ingest" and ids[-1] == "write_report"
        guards = [i for i in stream if i.id.startswith("guard_")]
        assert len(guards) == n_guards
        assert all(i.outputs == [] for i in guards)

    def test_side_effect_flags(self):
        spec = TaskSpec(random.Random(2), 0)
        backend = RuleClassifierBackend()
        corpus = generate_corpus(random.Random(12), spec.n_files,
                                 spec.lines_per_file)
        stream = build_task_stream(spec, backend, None, corpus)
        assert stream[0].side_effects is True
        assert stream[-1].side_effects is True
        assert all(i.side_effects is False for i in stream[1:-1])

    def test_llm_step_is_cacheable_and_pure(self):
        specs = [TaskSpec(random.Random(2026), i) for i in range(50)]
        backend = RuleClassifierBackend()
        found = False
        for spec in specs:
            corpus = generate_corpus(random.Random(11), spec.n_files,
                                     spec.lines_per_file)
            stream = build_task_stream(spec, backend, None, corpus)
            for instr in stream:
                if instr.id.startswith("classify_"):
                    found = True
                    assert instr.side_effects is False
                    assert instr.cache is not None  # cacheable
        assert found  # the 50-task pool contains cacheable LLM steps


class TestCompileCheck:
    def test_compile_check_passes_and_detects(self):
        """A built stream passes the compile battery; guards are the
        only instructions AGGRESSIVE may eliminate."""
        spec = TaskSpec(random.Random(3), 0)
        backend = RuleClassifierBackend()
        corpus = generate_corpus(random.Random(13), spec.n_files,
                                 spec.lines_per_file)
        stream = build_task_stream(spec, backend, None, corpus)
        out = compile_check(stream)
        assert out["pass"] is True
        assert out["NONE"]["effective"] == len(stream)
        guard_ids = {i.id for i in stream if i.id.startswith("guard_")}
        assert set(out["AGGRESSIVE"]["eliminated"]) <= guard_ids | set()


class TestStressBattery:
    def test_small_battery_all_pass(self, tmp_path: Path):
        """Run the full per-task battery on 5 tasks: every check green."""
        import argparse
        from random_tasks_stress import run_task

        args = argparse.Namespace(
            repeats=4, hot_threshold=2,
            ell_dir=str(tmp_path / "traces"),
        )
        (tmp_path / "traces").mkdir()
        rng = random.Random(99)
        for i in range(5):
            spec = TaskSpec(rng, i)
            result = asyncio.run(run_task(spec, args))
            failed = {k for k, v in result["checks"].items() if not v}
            assert not failed, f"task {i} failed: {failed}"

    def test_batteries_are_deterministic(self, tmp_path: Path):
        """Same spec twice -> byte-identical report hash."""
        import argparse
        from random_tasks_stress import run_task

        args = argparse.Namespace(
            repeats=2, hot_threshold=2,
            ell_dir=str(tmp_path / "traces2"),
        )
        (tmp_path / "traces2").mkdir()
        spec = TaskSpec(random.Random(7), 3)
        h1 = asyncio.run(run_task(spec, args))["report_sha256"]
        h2 = asyncio.run(run_task(spec, args))["report_sha256"]
        assert h1 == h2
