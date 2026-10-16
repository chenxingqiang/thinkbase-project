"""T-layer tests: typed dataflow contracts on the RSI instruction stream.

AgentISA Definition 3.2 gives instructions a type system with dependent
types; Definition 3.5's progress theorem needs well-typed states.  The
RSI stream realizes this as edge contracts: every declared dependency
carries a ``TypeConstraint``, composed into ONE dependent type on the
``ctx`` parameter and checked through the AgentISA-native
``validate_types`` before execution.  These tests pin:

* the native record: ``step.params["ctx"]`` is a dependent-typed
  constraint and ``validate_types`` itself accepts/rejects states;
* runtime enforcement: a contract violation fails loudly with the
  offending edge named, before any step code runs;
* each shared contract rejects its representative bad shapes;
* a consumer-side end-to-end violation: a producer planted with a
  wrong-shaped output is caught by the NEXT step's edge contract
  (types protect consumers, not producers);
* a well-formed synthetic gated run passes the full typed chain.
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

import rsix.streams as streams_mod
from rsix import data as data_mod
from rsix.models import GRU4Rec
from rsix.streams import (_instruction, check_mix_invariant,
                          check_gate_invariant, T_ADMITTED,
                          T_CANDIDATES, T_GATE, T_GATE_STATS, T_METRICS,
                          T_MIXBOOK, T_POP, T_SEQUENCES, T_TEST_MAP,
                          build_experiment_stream)
from agentflow.core.isa.instruction import InstructionType as IsaType
from agentflow.core.isa.instruction import TypeConstraint


# ---------------------------------------------------------------------------
# Native record: the dependent type is registered on the instruction
# ---------------------------------------------------------------------------

def _typed_step():
    def _exec(p, s):
        return {"out": 1}
    return _instruction("t_step", IsaType.CONTROL_FLOW, _exec,
                        deps=["cands"], outputs=["out"],
                        side_effects=False,
                        dep_types={"cands": T_CANDIDATES})


def test_native_params_carry_dependent_type():
    step = _typed_step()
    tc = step.params["ctx"]
    assert isinstance(tc, TypeConstraint)
    assert tc.base_type is dict
    assert tc.dependent_on is not None


def test_native_validate_types_accepts_good_state():
    step = _typed_step()
    good = {"ctx": {"cands": [(1, 2, 0.9, 1.0)]}}
    assert step.validate_types(good)
    assert step.verify_conditions(good)


def test_native_validate_types_rejects_bad_state():
    step = _typed_step()
    bad = {"ctx": {"cands": [(1, 2, 0.9)]}}  # 3-tuple, not a quad
    assert not step.validate_types(bad)


# ---------------------------------------------------------------------------
# Runtime enforcement with the offending edge named
# ---------------------------------------------------------------------------

def test_type_violation_raises_before_execution():
    step = _typed_step()
    with pytest.raises(RuntimeError, match=r"type check failed: "
                                           r"ctx\['cands'\]"):
        asyncio.run(step.execute({"ctx": {"cands": [(1, 2, 0.9)]}}))


def test_type_violation_names_missing_edge():
    step = _typed_step()
    with pytest.raises(RuntimeError,
                       match=r"type check failed: missing ctx key 'cands'"):
        asyncio.run(step.execute({"ctx": {"unrelated": 1}}))


def test_good_state_executes():
    step = _typed_step()
    out = asyncio.run(step.execute({"ctx": {"cands": [(1, 2, 0.9, 1.0)]}}))
    assert out == {"out": 1}


# ---------------------------------------------------------------------------
# Shared contracts reject representative bad shapes
# ---------------------------------------------------------------------------

def test_t_candidates_rejects_malformed():
    assert T_CANDIDATES.dependent_on([(1, 2, 0.9, 1.0)])
    assert not T_CANDIDATES.dependent_on([(1, 2, 0.9)])          # 3-tuple
    assert not T_CANDIDATES.dependent_on([("a", 2, 0.9, 1.0)])   # str user
    assert T_CANDIDATES.dependent_on([])                          # empty ok


def test_t_pop_rejects_negative_and_2d():
    import numpy as np
    assert T_POP.dependent_on(np.array([0.0, 1.0, 2.0]))
    assert not T_POP.dependent_on(np.array([[1.0], [2.0]]))     # 2-D
    assert not T_POP.dependent_on(np.array([1.0, -0.5]))         # negative


def test_t_gate_rejects_unfitted():
    from rsix import fidelity as fid
    gate = fid.FidelityGate()  # constructed, not fitted
    assert not T_GATE.dependent_on(gate)
    gate.fit(__import__("numpy").array([5, 3, 2, 1, 1]))
    assert T_GATE.dependent_on(gate)


def test_t_metrics_rejects_out_of_range():
    assert T_METRICS.dependent_on({"ndcg@10": 0.5, "hr@10": 0.2})
    assert not T_METRICS.dependent_on({"ndcg@10": 1.5, "hr@10": 0.2})
    assert not T_METRICS.dependent_on({"ndcg@10": 0.5})         # missing key


def test_t_sequences_rejects_tuple_values():
    assert T_SEQUENCES.dependent_on({0: [1, 2], 1: [3]})
    assert not T_SEQUENCES.dependent_on({0: (1, 2)})            # tuple
    assert not T_TEST_MAP.dependent_on({0: [1]})                 # list val


# ---------------------------------------------------------------------------
# Consumer-side end-to-end: a planted wrong-shaped producer output is
# caught by the NEXT step's edge contract
# ---------------------------------------------------------------------------

def _synthetic_data_file(tmp_path):
    import random
    rng = random.Random(7)
    lines = []
    for u in range(1, 31):
        for t, i in enumerate(rng.sample(range(1, 61),
                                         rng.randint(6, 14))):
            lines.append(f"{u} {i} 5 {t}")
    path = tmp_path / "u.data"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_end_to_end_catches_wrong_shaped_train(tmp_path, monkeypatch):
    data_path = _synthetic_data_file(tmp_path)

    orig = data_mod.leave_one_out

    def tuple_split(triples, n_users):
        train, test = orig(triples, n_users)
        # plant: sequences become tuples, violating T_SEQUENCES on the
        # train edge of every consumer
        return ({u: tuple(seq) for u, seq in train.items()}, test)

    monkeypatch.setattr(streams_mod.data_mod, "leave_one_out",
                        tuple_split)
    config = dict(variant="naive", seed=0, rounds=1,
                  epochs_init=2, epochs_round=1, generator="topk",
                  mix_lambda=1.0, data_path=str(data_path))
    _, _, _, n_items = data_mod.dataset(data_path)
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=8, seed=seed))
    # the violation must be caught by a CONSUMER (train_round_0 reads
    # the train edge), named in the error
    with pytest.raises(RuntimeError, match=r"type check failed"):
        asyncio.run(run_stream(stream))


def test_end_to_end_synthetic_gated_typed_chain(tmp_path):
    data_path = _synthetic_data_file(tmp_path)
    config = dict(variant="gated", seed=0, rounds=1,
                  epochs_init=3, epochs_round=1, generator="topk",
                  eps=0.08, mix_lambda=1.0, data_path=str(data_path))
    _, _, _, n_items = data_mod.dataset(data_path)
    stream = build_experiment_stream(
        config, lambda seed: GRU4Rec(n_items, dim=8, seed=seed))
    sb = LocalSandboxExecutor(name="test-typed-chain")
    try:
        inject_sandbox(stream, sb)
        sha = asyncio.run(run_stream(stream))  # full typed chain
        assert len(sha) == 64
    finally:
        sb.close()
