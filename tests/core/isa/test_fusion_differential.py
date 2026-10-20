"""Differential-execution tests for instruction fusion (review item L2).

The shipped fusion merges adjacent, order-independent pure instructions
into a ``CompositeInstruction`` whose execution body replays its
constituents sequentially.  These tests verify the differential property
the paper now claims: executing the fused form must produce exactly the
same state bindings as executing the constituents in sequence, at both
the single-merge and the whole-trace level, and the fusion gate must
refuse pairs whose merge would break dataflow.
"""

import asyncio
from typing import Any, Dict, List

import pytest

from agentflow.core.isa.instruction import Instruction, InstructionType
from agentflow.core.isa.instruction_optimization import StaticOptimizer


class StatefulInstruction(Instruction):
    """Deterministic instruction with executable state semantics.

    Reads its declared dependencies from the threaded state and returns
    (does not mutate) the new bindings for its declared outputs; the
    values are functions of the observed inputs, so any rewrite that
    preserves dataflow must reproduce identical final bindings.
    """

    def __init__(self, instr_id: str, *, side_effects: bool = False,
                 outputs: List[str] = None, deps: List[str] = None):
        super().__init__(
            id=instr_id,
            name=instr_id,
            type=InstructionType.STATE_MANAGEMENT,
            params={},
            description=f"Differential-test instruction {instr_id}",
            side_effects=side_effects,
            outputs=outputs or [],
        )
        self.dependencies = list(deps or [])

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seen = {d: input_params.get(d, "<missing>") for d in self.dependencies}
        signature = "|".join(f"{k}={v}" for k, v in sorted(seen.items()))
        new: Dict[str, Any] = {}
        for out in (self.outputs or []):
            new[out] = f"{out}<-[{signature}]"
        if self.side_effects:
            new[f"fx::{self.id}"] = f"effect[{signature}]"
        return new


async def _run_trace(trace: List[Instruction]) -> Dict[str, Any]:
    state: Dict[str, Any] = {}
    for instr in trace:
        result = await instr.execute(state)
        if isinstance(result, dict):
            state.update(result)
    return state


def _run(trace: List[Instruction]) -> Dict[str, Any]:
    return asyncio.run(_run_trace(trace))


class TestSingleMergeDifferential:

    def test_merged_execution_equals_sequential_execution(self):
        """Fused pair: composite state == running the two in sequence."""
        opt = StaticOptimizer()
        a = StatefulInstruction("a", outputs=["va"])
        b = StatefulInstruction("b", outputs=["vb"])
        assert opt._can_combine(a, b)

        merged = opt._merge_instructions(a, b)
        sequential_state = _run([a, b])
        merged_state = _run([merged])

        assert merged_state == sequential_state
        # The composite must carry both outputs for downstream analyses.
        assert sorted(merged.outputs) == ["va", "vb"]


class TestFusionGate:

    def test_gate_refuses_producer_consumer_pair(self):
        """A pair whose second reads the first's output must not fuse:
        the composite would execute the consumer without the producer's
        bindings (the data-hazard case the gate exists for)."""
        opt = StaticOptimizer()
        producer = StatefulInstruction("p", outputs=["v1"])
        consumer = StatefulInstruction("c", outputs=["v2"], deps=["v1"])
        assert not opt._can_combine(producer, consumer)

    def test_gate_refuses_side_effectful_pair(self):
        opt = StaticOptimizer()
        e1 = StatefulInstruction("e1", side_effects=True)
        e2 = StatefulInstruction("e2", side_effects=True)
        assert not opt._can_combine(e1, e2)

    def test_gate_refuses_type_mismatch(self):
        from agentflow.core.isa.instruction import InstructionType as IT
        opt = StaticOptimizer()
        a = StatefulInstruction("a", outputs=["va"])
        b = StatefulInstruction("b", outputs=["vb"])
        b.type = IT.LLM_INTERACTION
        assert not opt._can_combine(a, b)


class TestWholeTraceDifferential:
    """Original trace vs StaticOptimizer-optimized trace: identical state.

    The sequence mixes dead (no outputs, no effects), pure (outputs
    consumed by the sink), and side-effectful instructions, mirroring
    the paper's synthetic-workload structure.
    """

    def _build_trace(self) -> List[Instruction]:
        return [
            StatefulInstruction("d0"),                              # dead
            StatefulInstruction("p0", outputs=["v0"]),
            StatefulInstruction("p1", outputs=["v1"]),              # fusable with p0
            StatefulInstruction("e0", side_effects=True),
            StatefulInstruction("d1"),                              # dead
            StatefulInstruction("p2", outputs=["v2"]),
            StatefulInstruction("e1", side_effects=True),
            StatefulInstruction("sink", side_effects=True,
                                deps=["v0", "v1", "v2"]),
        ]

    def test_optimized_trace_reproduces_original_state(self):
        trace = self._build_trace()
        opt = StaticOptimizer()
        optimized = opt.optimize(trace)

        assert len(optimized) < len(trace), "optimizer removed nothing"
        original_state = _run(trace)
        optimized_state = _run(optimized)
        assert optimized_state == original_state

    def test_dead_instructions_leave_no_state_trace(self):
        """The removed instructions must be exactly the no-output,
        no-effect ones: removing anything else would change the state."""
        trace = self._build_trace()
        opt = StaticOptimizer()
        optimized = opt.optimize(trace)
        surviving_ids = {i.id for i in optimized}
        assert "d0" not in surviving_ids and "d1" not in surviving_ids
        # Everything with outputs or effects must survive — either as its
        # own instruction or as a component of a merged composite.
        def _survives(instr_id: str) -> bool:
            if instr_id in surviving_ids:
                return True
            return any(
                instr_id in [c.id for c in getattr(i, "components", [])]
                for i in optimized
            )

        for expected in ("p0", "p1", "p2", "e0", "e1", "sink"):
            assert _survives(expected), f"{expected} was dropped"
