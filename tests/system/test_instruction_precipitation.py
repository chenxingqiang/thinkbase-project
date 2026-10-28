"""Instruction precipitation: the four rules that make its numbers readable.

The experiment's value depends entirely on refusing to score a task off
instructions it just created, on gating promotion at real support, and on
retiring stale instructions rather than letting the library grow without cost.
Each rule gets an invariant test here; the coverage numbers themselves are
recomputed from the committed results JSON by the paper's number gate.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "examples" / "benchmarks" / "instruction_precipitation.py"


def _module():
    # Register before executing: dataclasses resolve their module via sys.modules.
    name = "instruction_precipitation"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_support_gate_promotes_only_at_the_second_occurrence() -> None:
    """One sighting is a candidate; the second promotes; only then can it serve."""
    ip = _module()
    repertoire = ip.Repertoire(min_support=2, max_idle=100)
    tid = ("get_order_details", ("order_id",))

    assert repertoire.observe(tid, 0) == "candidate"
    assert repertoire.active_count == 0
    assert repertoire.observe(tid, 0) == "promoted"
    assert repertoire.active_count == 1
    assert repertoire.observe(tid, 1) == "hit"


def test_a_task_cannot_score_off_the_instructions_it_creates() -> None:
    """Promotion inside a task does not serve that same task."""
    ip = _module()
    tasks = [
        {"source": "s", "task_id": "t0", "turns": [[["get_order_details", {"order_id": "A1"}]],
                                                   [["get_order_details", {"order_id": "A2"}]]]},
        {"source": "s", "task_id": "t1", "turns": [[["get_order_details", {"order_id": "B1"}]]]},
    ]
    report = ip.run_stream(tasks, "schema", min_support=2, max_idle=100, compound_k=3)

    assert report["per_task"][0]["served"] == 0      # two sightings: candidate, promoted
    assert report["per_task"][1]["served"] == 1      # served by an earlier task's instruction
    assert report["coverage"] == round(1 / 3, 4)


def test_retirement_fires_on_idleness_and_remining_restores_service() -> None:
    """A retired instruction stops serving and must be re-mined to return."""
    ip = _module()
    repertoire = ip.Repertoire(min_support=2, max_idle=1)
    tid = ("list_all_airports", ())
    repertoire.observe(tid, 0)
    repertoire.observe(tid, 0)
    assert repertoire.observe(tid, 1) == "hit"

    assert repertoire.retire_stale(3) == [tid]
    assert repertoire.active_count == 0
    assert repertoire.observe(tid, 3) in {"candidate", "promoted"}
    assert repertoire.remined == 1


def test_first_trial_only_keeps_one_record_per_task(tmp_path) -> None:
    """Replay trials must not be counted as separate tasks."""
    ip = _module()
    corpus = tmp_path / "corpus.jsonl"
    rows = [
        {"source": "s", "task_id": "t1", "trial": 0, "turns": [[["get_order_details", {"order_id": "A"}]]]},
        {"source": "s", "task_id": "t1", "trial": 1, "turns": [[["get_order_details", {"order_id": "A"}]]]},
        {"source": "s", "task_id": "t2", "trial": 0, "turns": [[["get_order_details", {"order_id": "B"}]]]},
    ]
    corpus.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    assert len(ip.load_tasks(corpus)) == 2
    assert len(ip.load_tasks(corpus, first_trial_only=False)) == 3


def test_repeated_runs_are_deterministic() -> None:
    """Same corpus, same rules, same numbers -- the report is reproducible."""
    ip = _module()
    corpus = REPO_ROOT / "examples" / "real_traces" / "swe_agent_sample.jsonl"
    first = ip.run(corpus, "command", min_support=2, max_idle=100, compound_k=3)
    second = ip.run(corpus, "command", min_support=2, max_idle=100, compound_k=3)

    assert first["interleaved"]["coverage"] == second["interleaved"]["coverage"]
    assert first["interleaved"]["repertoire"] == second["interleaved"]["repertoire"]
