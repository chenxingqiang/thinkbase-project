"""ELL workflow abstraction (vendored)."""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ELLWorkflow"]


@dataclass
class ELLWorkflow:
    """A workflow registered with the ELL integration layer.

    Args:
        name: Human-readable workflow name.
        description: Short description of the workflow's purpose.
        steps: Ordered list of step descriptors.
        metadata: Arbitrary workflow metadata.
        id: Unique identifier (auto-generated UUID when omitted).
    """

    name: str
    description: str = ""
    steps: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def add_step(self, step: Dict[str, Any]) -> None:
        """Append a step to the workflow."""
        self.steps.append(step)

    def get_step(self, index: int) -> Optional[Dict[str, Any]]:
        """Return the step at ``index`` if it exists."""
        if 0 <= index < len(self.steps):
            return self.steps[index]
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the workflow to a plain dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "steps": [dict(s) for s in self.steps],
            "metadata": dict(self.metadata),
        }
