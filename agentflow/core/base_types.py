"""Consolidated base types for AgentFlow.

This module is the single source of truth for the shared "primitive" enums
used across ``agentflow.core`` and ``agentflow.agents``.

``AgentStatus`` is a consolidated superset enum defined here; it deliberately
supersedes the tiny int-enum that used to live in this file
(``INITIALIZING``/``RUNNING``/``STOPPED``), because
``agentflow.core.workflow.workflow_state`` resolves its legacy
``from .base_types import AgentStatus`` import to this module and expects
members such as ``AgentStatus.INITIALIZED``/``AgentStatus.INITIALIZING``.

``WorkflowStatus`` is canonically defined in
``agentflow.core.workflow.workflow_types`` and re-exported here *lazily*
(PEP 562): ``agentflow.core.config`` imports this module before the workflow
package can initialize, and several modules (e.g. workflow_state) read
``WorkflowStatus`` from here while the workflow package is still importing.
"""

from enum import Enum

from . import enums as _enums
from .enums import StepStatus, WorkflowStepType


class AgentStatus(str, Enum):
    """Consolidated agent status enum.

    Superset of every legacy ``AgentStatus`` variant in the codebase: the
    tiny int-enum that used to live here (``INITIALIZING``/``RUNNING``/
    ``STOPPED``) and the richer enums in ``agentflow.core.enums`` and
    ``agentflow.agents.state``.  ``agentflow.core.workflow.workflow_state``
    resolves its legacy ``from .base_types import AgentStatus`` import to
    this class, so both ``INITIALIZING`` and ``INITIALIZED`` must exist.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZED = "initialized"
    COMPLETED = "completed"
    STOPPED = "stopped"

    def __str__(self) -> str:
        return self.value
    WAITING = "waiting"

    def __str__(self) -> str:
        return self.value


# ``AgentType``/``AgentMode`` are canonically defined in
# :mod:`agentflow.core.enums` (the str-Enum superset used by the agents
# package and the tests).  The tiny int-Enum stubs that used to live
# here were never imported anywhere, so they are simply re-exported.
AgentType = _enums.AgentType
AgentMode = _enums.AgentMode


DictKeyType = str


class MessageType(Enum):
    TEXT = 0
    COMMAND = 1


class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"


# --- Lazy re-exports (PEP 562) -------------------------------------------
_LAZY_RE_EXPORTS = ("WorkflowStatus", "WorkflowStepStatus")


def __getattr__(name):
    """Lazily re-export ``WorkflowStatus`` from the workflow types module.

    Note: ``import_module`` intentionally sits outside any ``except
    AttributeError`` handler -- swallowing AttributeErrors raised while the
    canonical module chain loads would mask real downstream errors.
    """
    if name in _LAZY_RE_EXPORTS:
        import importlib

        canonical = importlib.import_module(
            "agentflow.core.workflow.workflow_types"
        )
        return getattr(canonical, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include the lazily re-exported names in ``dir()``."""
    return sorted(set(globals().keys()) | set(_LAZY_RE_EXPORTS))


__all__ = [
    "AgentType",
    "AgentMode",
    "AgentStatus",
    "DictKeyType",
    "MessageType",
    "MessageRole",
    "StepStatus",
    "WorkflowStepType",
    "WorkflowStatus",
    "WorkflowStepStatus",
]
