"""Tests for the five thesis-direction task suites."""

import argparse
import asyncio
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, "examples")

import thesis_direction_suites as T  # noqa: E402
from system_task_demo import RuleClassifierBackend  # noqa: E402


def make(direction, seed=5):
    corpus = direction["corpus"](random.Random(seed))
    stream = direction["builder"](RuleClassifierBackend(), corpus)
    return stream, corpus


class TestSuiteShapes:
    @pytest.mark.parametrize("key", [d["key"] for d in T.DIRECTIONS])
    def test_every_suite_builds(self, key):
        direction = next(d for d in T.DIRECTIONS if d["key"] == key)
        stream, _ = make(direction)
        assert 6 <= len(stream) <= 9
        ids = [i.id for i in stream]
        assert ids[0].startswith("ingest_")
        assert ids[-1].startswith("write_")
        # side effects live exactly on ingest_/write_ steps (the
        # two-branch suite has an ingest in mid-stream)
        for i in stream:
            if i.id.startswith(("ingest_", "write_")):
                assert i.side_effects
            else:
                assert not i.side_effects
        # exactly one dead guard per suite
        guards = [i for i in stream if i.id.endswith("_guard")]
        assert len(guards) == 1 and guards[0].outputs == []

    def test_five_distinct_directions(self):
        keys = [d["key"] for d in T.DIRECTIONS]
        assert len(set(keys)) == 5

    def test_type_coverage_varies(self):
        """The batch exercises different ISA type mixes."""
        streams = [make(d)[0] for d in T.DIRECTIONS]
        llm_counts = {sum(1 for i in s
                          if str(i.type).endswith("LLM_INTERACTION"))
                      for s in streams}
        assert llm_counts == {0, 1, 2}  # none / one / chained LLM steps


class TestSuiteDataflow:
    def test_rag_qa_is_two_branch(self):
        stream, _ = make(T.DIRECTIONS[1])  # rag_qa
        by_id = {i.id: i for i in stream}
        # two independent ingest roots
        assert by_id["ingest_corpus"].dependencies == []
        assert by_id["ingest_query"].dependencies == []
        # retrieval consumes both branches
        assert set(by_id["retrieve_topk"].dependencies) == {
            "embeddings", "query"}

    def test_coordination_fans_out(self):
        stream, _ = make(T.DIRECTIONS[2])
        by_id = {i.id: i for i in stream}
        plans = [by_id[f"plan_agent_{t}"] for t in "abc"]
        # all three plans consume subtasks and are merged
        assert all(p.dependencies == ["subtasks"] for p in plans)
        assert set(by_id["merge_plans"].dependencies) == {
            "plan_a", "plan_b", "plan_c"}

    def test_code_audit_branches_converge(self):
        stream, _ = make(T.DIRECTIONS[3])
        by_id = {i.id: i for i in stream}
        # stats and severity both derive from sources, converge at summary
        assert by_id["parse_stats"].dependencies == ["sources"]
        assert by_id["detect_patterns"].dependencies == ["sources"]
        assert set(by_id["summarize_audit"].dependencies) == {
            "stats", "severity"}

    def test_llm_steps_are_cacheable(self):
        for d in T.DIRECTIONS:
            stream, _ = make(d)
            for i in stream:
                if str(i.type).endswith("LLM_INTERACTION"):
                    assert i.side_effects is False
                    assert i.cache is not None


class TestBattery:
    def test_full_batch_passes(self, tmp_path: Path):
        """All five suites pass the battery end-to-end (3 repeats)."""
        args = argparse.Namespace(
            # threshold 2 + 4 repeats: adoption observed on repeat 4
            repeats=4, hot_threshold=2,
            trace_dir=str(tmp_path / "traces"),
        )
        (tmp_path / "traces").mkdir()
        for index, direction in enumerate(T.DIRECTIONS):
            result = asyncio.run(T.run_suite(direction, index, args))
            failed = {k for k, v in result["checks"].items() if not v}
            assert not failed, f"{direction['key']}: {failed}"
            assert result["adopted"] is True
        # traces were written for each suite
        traces = list((tmp_path / "traces").glob("*.jsonl"))
        assert len(traces) == 5

    def test_battery_is_deterministic(self, tmp_path: Path):
        """Same direction twice -> byte-identical artifact hash."""
        args = argparse.Namespace(
            repeats=2, hot_threshold=2,
            trace_dir=str(tmp_path / "traces"),
        )
        (tmp_path / "traces").mkdir()
        direction = T.DIRECTIONS[0]
        h1 = asyncio.run(T.run_suite(direction, 0, args))["report_sha256"]
        h2 = asyncio.run(T.run_suite(direction, 0, args))["report_sha256"]
        assert h1 == h2

    def test_llm_suites_cache_after_first_realization(self, tmp_path: Path):
        args = argparse.Namespace(
            repeats=3, hot_threshold=2,
            trace_dir=str(tmp_path / "traces"),
        )
        (tmp_path / "traces").mkdir()
        for index, direction in enumerate(T.DIRECTIONS):
            result = asyncio.run(T.run_suite(direction, index, args))
            if result["llm_steps"]:
                assert result["per_repeat_calls"][0] > 0
                assert all(c == 0
                           for c in result["per_repeat_calls"][1:])
                # every LLM step hit the cache (repeats - 1) times
                assert result["cache_hits"] == \
                    result["llm_steps"] * (args.repeats - 1)
