"""AgentFlow core workflow package.

The modules in this subpackage originally lived directly inside
``agentflow/core`` and were moved here without rewriting their
package-relative imports.  As a result they still refer to siblings that now
live one level up (``agentflow.core.metrics``, ``agentflow.core.exceptions``,
``agentflow.core.base_types``, ...) or at the top level (``agentflow.agents``,
``agentflow.ell``).

This ``__init__`` therefore does three things:

1. It re-binds those stale import targets to the modules that actually exist
   today (via ``sys.modules`` aliases), so the relocated modules import
   cleanly without editing them.
2. It re-exports the public workflow API (``WorkflowEngine``,
   ``WorkflowStep``, ``WorkflowConfig``, ``WorkflowStatus``, ``StepConfig``,
   ``ErrorPolicy``, ...).
3. It keeps two legacy flat module names alive for backwards compatibility
   with code that still imports them from ``agentflow.core`` directly
   (``agentflow.core.workflow_executor`` and
   ``agentflow.core.agentic_calling``).
"""

import sys as _sys


def _alias_module(alias: str, real: str) -> None:
    """Make ``import <alias>`` resolve to the existing module ``<real>``.

    The alias is registered in ``sys.modules`` (and exposed as an attribute
    of its parent package) so that stale relative imports such as
    ``from .metrics import MetricsManager`` -- which resolve to
    ``agentflow.core.workflow.metrics`` -- find the real
    ``agentflow.core.metrics`` module instead of failing.
    """
    if alias in _sys.modules:
        return
    __import__(real)
    module = _sys.modules[real]
    _sys.modules[alias] = module
    parent_name, _, child = alias.rpartition(".")
    parent = _sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child, module)


# ---------------------------------------------------------------------------
# Stale single-dot relative imports (``.metrics``, ``.exceptions``, ...) used
# by the modules in this package.  The real modules live one level up, in
# ``agentflow.core``.
# ---------------------------------------------------------------------------
_alias_module("agentflow.core.workflow.metrics", "agentflow.core.metrics")
_alias_module("agentflow.core.workflow.exceptions", "agentflow.core.exceptions")
_alias_module("agentflow.core.workflow.agent_config", "agentflow.core.agent_config")
_alias_module("agentflow.core.workflow.model_config", "agentflow.core.model_config")
_alias_module("agentflow.core.workflow.types", "agentflow.core.types")
_alias_module("agentflow.core.workflow.enums", "agentflow.core.enums")
_alias_module("agentflow.core.workflow.base_types", "agentflow.core.base_types")
_alias_module("agentflow.core.workflow.config", "agentflow.core.config")
_alias_module("agentflow.core.workflow.objective_handler", "agentflow.core.objective_handler")
_alias_module("agentflow.core.workflow.context_manager", "agentflow.core.context.context_manager")

# ---------------------------------------------------------------------------
# Stale double-dot relative imports (``..agents``, ``..ell``) used by the
# modules in this package.  They resolve to ``agentflow.core.<x>``, which does
# not exist -- the real packages live at the ``agentflow`` top level.
# ---------------------------------------------------------------------------
_alias_module("agentflow.core.agents", "agentflow.agents")
_alias_module("agentflow.core.agents.agent", "agentflow.agents.agent")
_alias_module("agentflow.core.agents.agent_types", "agentflow.agents.agent_types")
_alias_module("agentflow.core.ell", "agentflow.ell")
_alias_module("agentflow.core.ell.types", "agentflow.ell.types")
_alias_module("agentflow.core.ell.types.message", "agentflow.ell.types.message")
_alias_module("agentflow.core.ell.integration", "agentflow.ell.integration")

# ---------------------------------------------------------------------------
# Legacy flat module names, kept alive for backwards compatibility with code
# (and tests) that still imports them from ``agentflow.core`` directly.
# ---------------------------------------------------------------------------
_alias_module("agentflow.core.workflow_executor", "agentflow.core.workflow.workflow_executor")
_alias_module("agentflow.core.agentic_calling", "agentflow.agents.agentic_calling")


# ---------------------------------------------------------------------------
# Public API of the workflow package.
#
# Note: ``research_workflow`` and ``objective_workflow`` are exported lazily
# (via ``__getattr__`` below) because they import ``agentflow.core.config``
# attribute-wise, and this package may be initialized while
# ``agentflow.core.config`` is still mid-import.
# ---------------------------------------------------------------------------
from .workflow_types import (  # noqa: E402
    Message,
    MessageRole,
    MessageType,
    ContentBlock,
    RetryPolicy,
    ErrorPolicy,
    AgentStatus,
    AgentState,
    Agent,
    WorkflowStepType,
    WorkflowStatus,
    WorkflowStepStatus,
    WorkflowStrategy,
    StepConfig,
    WorkflowStep,
    WorkflowConfig,
    WorkflowInstance,
    CollaborationMode,
    CommunicationProtocol,
    WorkflowDefinition,
    VALID_PROTOCOLS,
    VALID_STRATEGIES,
)

# ``AgentState`` above comes from ``workflow_types``; the import below
# intentionally rebinds it to the state-management model from
# ``workflow_state`` (its canonical home).
from .workflow_state import (  # noqa: E402
    AgentState,
    StepState,
    WorkflowState,
    WorkflowStateManager,
)

# ``WorkflowEngine`` (the WorkflowInstance-based engine with
# create/execute/register semantics) is the package's canonical engine;
# the ``workflow_engine`` module remains importable for its own
# consumers (the flat ``agentflow.core.workflow_engine`` shim).
from .workflow import WorkflowEngine  # noqa: E402
from .workflow_executor import (  # noqa: E402
    VALID_PROTOCOLS,
    VALID_STRATEGIES,
    WorkflowExecutor,
)
from .workflow import Workflow  # noqa: E402
from .flow_controller import (  # noqa: E402
    ErrorConfig,
    ErrorStrategy,
    FlowController,
    RouteCondition,
)
from .base_workflow import BaseWorkflow  # noqa: E402
from .workflow_base import WorkflowBase  # noqa: E402

_LAZY_EXPORTS = {
    # research_workflow
    "ResearchDistributedWorkflow": ("research_workflow", "ResearchDistributedWorkflow"),
    "ResearchStep": ("research_workflow", "ResearchStep"),
    "DistributedConfig": ("research_workflow", "DistributedConfig"),
    # objective_workflow
    "ObjectiveWorkflow": ("objective_workflow", "ObjectiveWorkflow"),
}


def __getattr__(name):
    """Lazily export symbols whose modules cannot be imported eagerly."""
    try:
        submodule, attr = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    import importlib

    module = importlib.import_module(f".{submodule}", package=__name__)
    return getattr(module, attr)


__all__ = [
    # workflow_types
    "Message",
    "MessageRole",
    "MessageType",
    "ContentBlock",
    "RetryPolicy",
    "ErrorPolicy",
    "AgentStatus",
    "AgentState",
    "Agent",
    "WorkflowStepType",
    "WorkflowStatus",
    "WorkflowStepStatus",
    "WorkflowStrategy",
    "StepConfig",
    "WorkflowStep",
    "WorkflowConfig",
    "WorkflowInstance",
    "CollaborationMode",
    "CommunicationProtocol",
    "WorkflowDefinition",
    "VALID_PROTOCOLS",
    "VALID_STRATEGIES",
    # workflow_state
    "StepState",
    "WorkflowState",
    "WorkflowStateManager",
    # engine / executor
    "WorkflowEngine",
    "WorkflowExecutor",
    # high-level workflow helpers
    "Workflow",
    "BaseWorkflow",
    "WorkflowBase",
    # flow control
    "ErrorConfig",
    "ErrorStrategy",
    "FlowController",
    "RouteCondition",
    # research / objective workflows (lazy)
    "DistributedConfig",
    "ResearchDistributedWorkflow",
    "ResearchStep",
    "ObjectiveWorkflow",
]
