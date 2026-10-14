"""Agents package."""

from .agent import Agent, AgentStatus

__all__ = ['Agent', 'AgentStatus', 'BaseTestAgent']


def __getattr__(name):
    """Lazily resolve heavy exports.

    ``BaseTestAgent`` imports Ray at module level (it is a
    ``@ray.remote`` actor); keep it out of the eager import path so a
    plain ``import agentflow`` does not pay Ray's ~2s import cost.
    """
    if name == "BaseTestAgent":
        from .base_agent import BaseTestAgent

        return BaseTestAgent
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

