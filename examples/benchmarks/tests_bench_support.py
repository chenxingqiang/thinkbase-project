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
