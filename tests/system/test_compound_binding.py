"""Compound binding: an instruction is only an instruction if it can be instantiated.

The experiment's claim is narrow and mechanical, so each part of it gets an
invariant test: where an argument came from, whether a learned spec reproduces
held-out occurrences, and what happens when the observation shape changes. The
coverage numbers themselves are recomputed from the committed results JSON by
the paper's number gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "examples" / "benchmarks" / "compound_binding.py"


def _module():
    name = "compound_binding"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _call(tool, args, obs=None):
    return (tool, args, obs)


def test_find_paths_and_read_path_round_trip() -> None:
    cb = _module()
    payload = {"user": {"orders": [{"order_id": "#W1"}, {"order_id": "#W2"}]}}
    paths = cb.find_paths(payload, "#W2")
    assert paths == [["user", "orders", 1, "order_id"]]
    assert cb.read_path(payload, paths[0]) == "#W2"


def test_provenance_separates_observation_from_external_values() -> None:
    cb = _module()
    calls = [
        _call("find_user", {"name": "Yusuf"}, "yusuf_9620"),
        _call("get_user", {"user_id": "yusuf_9620"}, '{"orders": ["#W1"]}'),
        _call("get_order", {"order_id": "#W1"}),
        _call("get_order", {"order_id": "#W9"}),          # not produced by anything
    ]
    origins = cb.provenance_of(calls)
    assert origins[(1, "user_id")] == ("obs", 0)
    assert origins[(2, "order_id")] == ("obs", 1)
    assert origins[(3, "order_id")] == ("external", 0)


def test_learned_spec_replays_on_a_second_occurrence() -> None:
    cb = _module()
    first = [
        _call("find_user", {"name": "A"}, "id_a"),
        _call("get_user", {"user_id": "id_a"}, '{"orders": [{"order_id": "#1"}]}'),
        _call("get_order", {"order_id": "#1"}),
    ]
    spec = cb.learn_spec(first, cb.provenance_of(first))
    produced = cb.replay_spec(
        spec, {"p_name": "B"}, ["id_b", '{"orders": [{"order_id": "#7"}]}', None])
    assert produced[1] == {"user_id": "id_b"}
    assert produced[2] == {"order_id": "#7"}


def test_replay_refuses_to_read_a_result_not_yet_produced() -> None:
    cb = _module()
    calls = [
        _call("first", {"a": "x"}, "one"),
        _call("second", {"b": "one"}),
    ]
    spec = cb.learn_spec(calls, cb.provenance_of(calls))
    spec["steps"][1]["args"]["b"] = {"kind": "path", "step": 1, "path": []}
    try:
        cb.replay_spec(spec, {"p_a": "x"}, ["one", "two"])
    except ValueError:
        return
    raise AssertionError("a compound must not read its own or a later result")


def test_path_drift_does_not_silently_yield_a_wrong_binding() -> None:
    cb = _module()
    payload = ["#W1", "#W2"]
    assert cb.find_paths(payload, "#W2") == [[1]]
    # the same path in a re-shaped observation points at something else
    reshaped = {"order_id": ["#W1", "#W2"]}
    try:
        cb.read_path(reshaped, [1])
    except ValueError:
        return
    raise AssertionError("a path that no longer applies must raise, not guess")


def test_run_is_deterministic_on_a_tiny_corpus(tmp_path: Path) -> None:
    cb = _module()
    rows = []
    for task_id in ("t1", "t2"):
        rows.append({
            "source": "toy", "task_id": task_id, "trial": 0, "reward": 1.0,
            "calls": [
                ["find_user", {"name": "n"}, "id_n"],
                ["get_user", {"user_id": "id_n"}, '{"orders": [{"order_id": "#1"}]}'],
                ["get_order", {"order_id": "#1"}, None],
            ],
        })
    corpus = tmp_path / "toy.jsonl"
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    first = cb.run(corpus)
    second = cb.run(corpus)
    assert first["compounds"] == second["compounds"]
    assert first["reduction_if_substituted"] == second["reduction_if_substituted"]
    assert first["compounds"]["recurring"] >= 1
    assert first["compounds"]["replay_verified"] >= 1
