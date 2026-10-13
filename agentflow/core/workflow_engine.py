"""Thin compatibility shim for the legacy ``agentflow.core.workflow_engine`` path.

The real implementation lives in
:mod:`agentflow.core.workflow.workflow_engine`; every public symbol is
re-exported here so historical imports keep working.
"""

from agentflow.core.workflow.workflow_engine import WorkflowEngine

__all__ = ["WorkflowEngine"]
