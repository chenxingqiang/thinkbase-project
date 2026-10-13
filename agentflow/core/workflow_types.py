"""Legacy import path for workflow types.

The canonical definitions live in
:mod:`agentflow.core.workflow.workflow_types`.  This module re-exports
the full public surface so that historical imports such as
``from agentflow.core.workflow_types import WorkflowStep`` keep
working.

Re-exports are resolved lazily (PEP 562): the canonical module lives
inside the ``agentflow.core.workflow`` package whose ``__init__``
imports modules that themselves import this shim, so an eager
re-export would create a circular import.
"""

from typing import Any

_CANONICAL = "agentflow.core.workflow.workflow_types"

_RE_EXPORTS = (
    "Agent",
    "AgentState",
    "AgentStatus",
    "CollaborationMode",
    "CommunicationProtocol",
    "ContentBlock",
    "ErrorPolicy",
    "Message",
    "MessageRole",
    "MessageType",
    "RetryPolicy",
    "StepConfig",
    "VALID_PROTOCOLS",
    "VALID_STRATEGIES",
    "WorkflowConfig",
    "WorkflowDefinition",
    "WorkflowInstance",
    "WorkflowStatus",
    "WorkflowStep",
    "WorkflowStepStatus",
    "WorkflowStepType",
    "WorkflowStrategy",
)

__all__ = list(_RE_EXPORTS)


def __getattr__(name: str) -> Any:
    """Lazily resolve re-exported names from the canonical module."""
    if name in _RE_EXPORTS:
        import importlib

        canonical = importlib.import_module(_CANONICAL)
        try:
            return getattr(canonical, name)
        except AttributeError:
            raise AttributeError(
                f"module {__name__!r} has no attribute {name!r}"
            ) from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals().keys()) | set(_RE_EXPORTS))
