"""Tests for the real-trace transfer pilot harness (examples/benchmarks).

The harness is pilot infrastructure: it converts real agent trajectories into
instruction streams under a conservative mapping and measures what survives the
shipped compiler's own gates. These tests pin the *harness invariants* -- the
mapping discipline that makes any future number readable -- not the numbers
themselves (the self-test's figures are never results).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PILOT = REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("real_trace_pilot", PILOT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pilot():
    return _load_module()


def test_self_test_runs_and_tags_its_output(pilot, tmp_path: Path) -> None:
    out = tmp_path / "selftest.json"
    assert pilot.main(["--self-test", "--output", str(out)]) == 0
    reports = json.loads(out.read_text(encoding="utf-8"))
    assert len(reports) == 2  # both adapters exercised
    for rep in reports:
        assert rep["self_test"] is True
        assert rep["experiment"] == "real_trace_pilot"
        agg = rep["aggregate"]
        assert agg["effective"]["NONE"] == agg["total_steps"]
        assert agg["total_dead_steps"] == 0  # conservative mapping: no free DCE


def test_same_turn_calls_are_parallel_later_turns_chain(pilot) -> None:
    """Dependency discipline: siblings in one assistant turn share no edge;
    a later turn chains to the live handle (doubt resolves to dependent)."""
    record = {"id": "t", "trajectory": [
        {"tool_calls": [{"name": "get_order", "arguments": {"id": "A1"}},
                        {"name": "get_user", "arguments": {"id": "U1"}}]},
        {"tool_calls": [{"name": "send_email",
                         "arguments": {"to": "U1", "body": "hi"}}]},
    ]}
    steps = pilot.to_steps(pilot.actions_tau_bench(record))
    assert steps[0]["dependencies"] == []
    assert steps[1]["dependencies"] == []          # same turn -> independent
    # later turn consumes the previous step's declared output exactly once
    assert steps[2]["dependencies"] == ["record#U1"]


def test_unknown_tools_are_effectful(pilot) -> None:
    steps = pilot.to_steps([[("mystery_tool", {"x": 1})]])
    assert steps[0]["side_effects"] is True
    assert steps[0]["outputs"] == []


def test_dependency_tokens_share_the_output_namespace(pilot) -> None:
    """The compiler's fusion gate compares dependencies against the
    predecessor's outputs, so the two must be the same tokens."""
    steps = pilot.to_steps([[
        ("read_file", {"path": "a.py"}), ("grep", {"pattern": "x"}),
    ], [
        ("summarize", {"text": "later"}),
    ]])
    outputs = {o for st in steps for o in st["outputs"]}
    for st in steps:
        for dep in st["dependencies"]:
            assert dep in outputs


def test_terminal_artifact_is_protected_from_dce(pilot) -> None:
    """A pure final answer step has no consumer in a real trace; the harness
    declares the terminal turn's outputs required so liveness keeps it."""
    record = {"id": "t", "actions": [
        {"tool": "read_file", "args": {"path": "q.md"}},
        {"tool": "answer", "args": {"text": "done"}},
    ]}
    rep = pilot.run([record], "swe-agent", "test", False)
    row = rep["per_stream"][0]
    assert row["steps"] == 2
    assert row["effective_BASIC"] == 2  # answer not eliminated as dead


def test_cache_measure_counts_repeats(pilot) -> None:
    cm = pilot.cache_measure([
        ("get_order_details", {"order_id": "A1"}),
        ("get_order_details", {"order_id": "A1"}),
        ("get_user_details", {"user_id": "U7"}),
    ])
    assert cm["calls"] == 3
    assert cm["instruction_scoped_hits"] == 1
    assert cm["instruction_scoped_hit_rate"] == pytest.approx(1 / 3, abs=1e-4)
    # the params-only legacy key is not tool-scoped: only equal here because
    # the three calls carry distinct parameters
    assert cm["params_only_hits"] == 1


PREPARE_SWE = REPO_ROOT / "examples" / "real_traces" / "prepare_swe_agent.py"


def test_swe_action_parser_reads_the_last_fenced_block() -> None:
    """SWE-agent emits its action as the last fenced block; prose above it is
    not an action, and a shell command keeps only its first line."""
    spec = importlib.util.spec_from_file_location("prepare_swe_agent", PREPARE_SWE)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    text = (
        "I will inspect the file.\n\n```\nstr_replace_editor view /repo/app.py\n```\n"
    )
    assert mod.parse_action(text) == [
        "str_replace_editor", {"command": "view", "path": "/repo/app.py"}]
    shell = "Let me run the tests.\n\n```bash\ncd /repo && pytest -q\nmore output\n```\n"
    assert mod.parse_action(shell) == ["bash", {"cmd": "cd /repo && pytest -q"}]
    assert mod.parse_action("only reasoning, no action") is None


def test_cache_is_restricted_to_side_effect_free_calls(pilot) -> None:
    """A result cache may serve only pure instructions: repeated effectful
    commands are repetition, not cacheable reuse."""
    cm = pilot.cache_measure([
        ("bash", {"cmd": "pytest"}),          # effectful, repeated
        ("bash", {"cmd": "pytest"}),
        ("get_order_details", {"order_id": "A1"}),   # pure, repeated
        ("get_order_details", {"order_id": "A1"}),
    ])
    assert cm["calls"] == 4
    assert cm["cacheable_calls"] == 2
    assert cm["effectful_repeats"] == 1
    assert cm["instruction_scoped_hits"] == 1
    assert cm["instruction_scoped_hit_rate"] == pytest.approx(0.5)
