"""Thin compatibility shim for the legacy ``agentflow.core.research_workflow`` path.

The real implementation lives in
:mod:`agentflow.core.workflow.research_workflow`; every public symbol is
re-exported here so historical imports keep working.
"""

from agentflow.core.workflow.research_workflow import (
    DistributedConfig,
    ResearchDistributedWorkflow,
    ResearchStep,
)

__all__ = [
    "DistributedConfig",
    "ResearchDistributedWorkflow",
    "ResearchStep",
]
