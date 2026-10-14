"""Flat compatibility shim for the legacy ``agentflow.core.distributed_workflow`` path.

The distributed workflow classes live in
:mod:`agentflow.core.workflow.distributed_workflow` (the package).
``ResearchDistributedWorkflow`` and ``DistributedConfig`` are re-exported
here so historical imports keep working.
"""

from agentflow.core.workflow.distributed_workflow import (  # noqa: F401
    DistributedWorkflow,
    DistributedConfig,
    ResearchDistributedWorkflow,
)

__all__ = [
    "DistributedWorkflow",
    "DistributedConfig",
    "ResearchDistributedWorkflow",
]
