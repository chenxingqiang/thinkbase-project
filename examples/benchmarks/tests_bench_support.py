"""Shared helpers for the AgentISA microbenchmark suite.

``BenchInstruction`` is a lightweight ``Instruction`` subclass used by
the dynamic hot-path experiment.  It executes as a pure no-op (the
experiment measures trace shrinkage, not inference) while carrying the
semantic annotations (side effects, outputs, dependencies) that the
optimization passes reason about.
"""

from typing import Any, Dict

from agentflow.core.isa.instruction import Instruction, InstructionType


class BenchInstruction(Instruction):
    """No-op instruction with configurable semantic annotations."""

    def __init__(self, name: str, side_effects: bool = True,
                 outputs=None, dependencies=None):
        super().__init__(
            id=name,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Bench instruction {name}",
            side_effects=side_effects,
            outputs=outputs or [],
        )
        self.dependencies = list(dependencies or [])
        self.executed = False

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executed = True
        return {"status": "executed", "id": self.id}


class StatefulBenchInstruction(BenchInstruction):
    """Bench instruction with a *executable* state semantics.

    Used by the dynamic hot-path experiment's replay-verification arm.
    ``execute`` reads its declared dependencies from the state dict that
    the harness threads through ``input_params`` and *returns* (rather
    than mutates) the new bindings its declared outputs take—values are
    deterministic functions of the observed inputs, so an
    optimization that preserves dataflow must reproduce an identical
    final state.  Side-effectful instructions additionally leave a
    per-id marker so their presence/ordering is observable.
    """

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executed = True
        seen = {d: input_params.get(d, "<missing>") for d in self.dependencies}
        signature = "|".join(f"{k}={v}" for k, v in sorted(seen.items()))
        new: Dict[str, Any] = {}
        for out in (self.outputs or []):
            new[out] = f"{out}<-[{signature}]"
        if self.side_effects:
            new[f"fx::{self.id}"] = f"effect[{signature}]"
        return new
