"""Legacy flat path for Git operations (now under instructions)."""

from agentflow.core.instructions.git_operations import (  # noqa: F401
    GitConfig,
    GitOperationsHandler,
)

__all__ = ["GitConfig", "GitOperationsHandler"]
