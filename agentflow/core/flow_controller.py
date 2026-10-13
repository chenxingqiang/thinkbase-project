"""Thin compatibility shim for the legacy ``agentflow.core.flow_controller`` path.

The real implementation lives in
:mod:`agentflow.core.workflow.flow_controller`; every public symbol is
re-exported here so historical imports keep working.
"""

from agentflow.core.workflow.flow_controller import (
    ErrorConfig,
    ErrorStrategy,
    FlowController,
    RouteCondition,
)

__all__ = [
    "ErrorConfig",
    "ErrorStrategy",
    "FlowController",
    "RouteCondition",
]
