"""Phi-layer tests: runtime pre/postcondition enforcement in the RSI stream.

The AgentISA formal specification gives instructions pre/postconditions
(Definition 3.5's validity obligation).  These tests pin the stream's
wiring of that layer:

* the declared dataflow is a runtime-checked precondition (a missing
  producer fails loudly, not silently as a KeyError deep in a closure);
* postconditions fire on violation and stay silent when invariants hold;
* the native AgentISA records (context.preconditions/postconditions)
  carry the same predicates, so the formal view and the runtime check
  cannot drift apart;
* the gate invariant machine-checks INV-1 (mixed JS <= eps) for gated
  and pins each baseline's defining admission behavior;
* an end-to-end synthetic gated run passes every invariant.
"""

import asyncio
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[1]))
sys.path.insert(0, str(_HERE.parents[1] / "examples"))

from agentflow.sandbox.local_executor import LocalSandboxExecutor
from system_task_demo import inject_sandbox, run_stream

from rsix import data as data_mod
from rsix.models import GRU4Rec
from rsix.streams import (build_experiment_stream, check_gate_invariant,
                          check_mix_invariant, _instruction)
from agentflow.core.isa.instruction import InstructionType as IsaType


# ---------------------------------------------------------------------------
# Factory-level enforcement
# ---------------------------------------------------------------------------

def _mk_step(deps, post=None):
    def _exec(p, s):
        return {"out": 1}
    return _instruction("t_step", IsaType.CONTROL_FLOW, _exec,
                        deps=deps, outputs=["out"], side_effects=False,
                        post=post)


def test_precondition_missing_dep_raises():
    step = _mk_step(deps=["never_produced"])
    with pytest.raises(RuntimeError, match="precondition violated"):
        asyncio.run(step.execute({"ctx": {}}))


def test_precondition_passes_when_deps_present():
    step = _mk_step(deps=["x"])
    out = asyncio.run(step.execute({"ctx": {"x": 1}}))
    assert out == {"out": 1}


def test_postcondition_violation_raises():
    step = _mk_step(deps=[], post=lambda ctx, res: "boom")
    with pytest.raises(RuntimeError, match="postcondition violated: boom"):
        asyncio.run(step.execute({"ctx": {}}))


def test_native_condition_records_registered():
    step = _mk_step(deps=["x"], post=lambda ctx, res: None)
    assert len(step.context.preconditions) == 1
    assert len(step.context.postconditions) == 1
    # the native precondition is the resolvable-dataflow predicate
    assert step.verify_conditions({"ctx": {"x": 1}})
    assert not step.verify_conditions({"ctx": {}})


# ---------------------------------------------------------------------------
# Gate invariant: INV-1 machine-check + baseline behavior pinning
# ---------------------------------------------------------------------------

def _gate_res(admitted, stats):
    return {"admitted_r0": admitted, "gate_stats_r0": stats}


def test_gate_invariant_allows_within_budget():
    post = check_gate_invariant("gated", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 3,
             "mixed_js_distance": 0.04}
    assert post({}, _gate_res([(1, 2), (3, 4), (5, 6)], stats)) is None


def test_gate_invariant_catches_budget_breach():
    post = check_gate_invariant("gated", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 9,
             "mixed_js_distance": 0.5}
    msg = post({}, _gate_res([(1, 2)] * 9, stats))
    assert msg is not None and "INV-1" in msg and "0.5" in msg


def test_gate_invariant_catches_missing_js_stat():
    post = check_gate_invariant("gated", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 1}
    msg = post({}, _gate_res([(1, 2)], stats))
    assert msg is not None and "INV-1" in msg


def test_gate_invariant_pins_baselines():
    naive_post = check_gate_invariant("naive", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 7}
    msg = naive_post({}, _gate_res([(1, 2)] * 7, stats))
    assert msg is not None and "naive" in msg

    no_rsi_post = check_gate_invariant("no_rsi", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 1}
    msg = no_rsi_post({}, _gate_res([(1, 2)], stats))
    assert msg is not None and "no_rsi" in msg


def test_gate_invariant_catches_admitted_stat_mismatch():
    post = check_gate_invariant("gated", eps=0.08, rkey="r0")
    stats = {"n_candidates": 10, "n_admitted": 5,
             "mixed_js_distance": 0.01}
    msg = post({}, _gate_res([(1, 2)] * 3, stats))  # list has 3, stat says 5
    assert msg is not None and "stat" in msg


# ---------------------------------------------------------------------------
# Mix invariant: bookkeeping (checkable form of INV-2)
# ---------------------------------------------------------------------------

def test_mix_invariant_passes_on_consistent_bookkeeping():
    post = check_mix_invariant("r0")

    class _Gate:
        mix_lambda = 1.0

    ctx = {"admitted_r0": [(1, 2), (3, 4)],
           "gate": _Gate()}
    res = {"train": {0: [1]}, "last_mix": {"real_events": 100,
                                           "synthetic_events": 2}}
    assert post(ctx, res) is None


def test_mix_invariant_catches_bookkeeping_error():
    post = check_mix_invariant("r0")

    class _Gate:
        mix_lambda = 2.0

    ctx = {"admitted_r0": [(1, 2), (3, 4)],
           "gate": _Gate()}
    res = {"train": {0: [1]}, "last_mix": {"real_events": 100,
                                           "synthetic_events": 3}}
    msg = post(ctx, res)  # expected 2 admitted x 2 reps = 4
    assert msg is not None and "expected=4" in msg


# ---------------------------------------------------------------------------
# End-to-end: a synthetic gated run with every invariant active
# ---------------------------------------------------------------------------

def _synthetic_data_file(tmp_path):
    import random
    rng = random.Random(7)
    lines = []
    n_users, n_items = 30, 60
    for u in range(1, n_users + 1):
        items = rng.sample(range(1, n_items + 1),
                           rng.randint(6, 14))
        for t, i in enumerate(items):
            lines.append(f"{u} {i} 5 {t}")
    path = tmp_path / "u.data"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_end_to_end_synthetic_gated_run(tmp_path):
    data_path = _synthetic_data_file(tmp_path)
    config = dict(variant="gated", seed=0, rounds=1,
                  epochs_init=3, epochs_round=1, generator="topk",
                  eps=0.08, mix_lambda=1.0, data_path=str(data_path))
    _, _, _, n_items = data_mod.dataset(data_path)
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=8, seed=seed))
    # INV-1 is wired for the gated variant on both rounds' gate steps
    gate_steps = [i for i in stream if i.id.startswith("gate_")]
    assert len(gate_steps) == 1

    sb = LocalSandboxExecutor(name="test-postcond")
    try:
        inject_sandbox(stream, sb)
        sha = asyncio.run(run_stream(stream))
        assert len(sha) == 64
        body = sb.read_file("result_gated_0.json").decode()
        assert "mixed_js_distance" in body
        # INV-1 held on the single round (checked by the step itself;
        # re-assert from the persisted record)
        import json
        hist = json.loads(body)
        st = hist["rounds"][0]["gate_stats"]
        assert st["mixed_js_distance"] <= config["eps"] + 1e-9
    finally:
        sb.close()


def test_end_to_end_catches_leakage(tmp_path, monkeypatch):
    """A deliberately corrupted split must fail its postcondition."""
    data_path = _synthetic_data_file(tmp_path)

    import rsix.streams as streams_mod
    orig = data_mod.leave_one_out

    def leaking_split(triples, n_users):
        train, test = orig(triples, n_users)
        if train:
            u0 = next(iter(train))
            if u0 in test and train[u0]:
                # plant the test item back into the train sequence
                train[u0].append(test[u0])
        return train, test

    monkeypatch.setattr(streams_mod.data_mod, "leave_one_out",
                        leaking_split)
    config = dict(variant="naive", seed=0, rounds=1,
                  epochs_init=2, epochs_round=1, generator="topk",
                  mix_lambda=1.0, data_path=str(data_path))
    _, _, _, n_items = data_mod.dataset(data_path)
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=8, seed=seed))
    with pytest.raises(RuntimeError, match="leakage"):
        asyncio.run(run_stream(stream))
