"""Legacy flat path for shared agent-facing types.

Canonical homes:
- ``AgentStatus``/``AgentType``/``AgentMode`` -> ``agentflow.core.base_types``
  (re-exported from ``agentflow.core.enums``)
- ``AgentConfig`` -> ``agentflow.core.agent_config``
- ``ModelConfig`` -> ``agentflow.core.config`` (lazy: importing it eagerly
  would re-enter ``agentflow.core.config`` while it is still initializing)

All names are resolved lazily (PEP 562) to avoid import cycles between
this legacy path and the canonical modules.
"""

from .base_types import AgentStatus, AgentType, AgentMode  # noqa: F401

_LAZY_RE_EXPORTS = ("AgentConfig", "ModelConfig", "MessageRole", "MessageType")


def __getattr__(name):
    if name in _LAZY_RE_EXPORTS:
        import importlib

        if name == "AgentConfig":
            return getattr(
                importlib.import_module("agentflow.core.agent_config"),
                name,
            )
        if name == "ModelConfig":
            return getattr(
                importlib.import_module("agentflow.core.config"), name
            )
        return getattr(
            importlib.import_module("agentflow.core.base_types"), name
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals().keys()) | set(_LAZY_RE_EXPORTS))
