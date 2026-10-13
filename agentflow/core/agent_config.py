from typing import List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

# ``ModelConfig`` (the LLM model configuration used alongside AgentConfig:
# provider/name/temperature/...) is defined in ``agentflow.core.config``,
# which itself imports this module.  Re-exporting it eagerly would create a
# circular import, so it is resolved lazily on first attribute access
# (PEP 562).  ``from agentflow.core.agent_config import ModelConfig`` keeps
# working without an import cycle.
_LAZY_RE_EXPORTS = ("ModelConfig",)

def __getattr__(name: str) -> Any:
    """Lazily re-export ``ModelConfig`` from ``agentflow.core.config``."""
    if name in _LAZY_RE_EXPORTS:
        from .config import ModelConfig
        return ModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    """Include lazily re-exported names in ``dir()`` output."""
    return sorted(set(globals().keys()) | set(_LAZY_RE_EXPORTS))

__all__ = ["AgentRole", "AgentCapability", "AgentConfig", "ModelConfig"]

class AgentRole(Enum):
    PROGRAMMER = "programmer"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"

class AgentCapability(Enum):
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"

class AgentConfig(BaseModel):
    agent_name: str = Field(default="default_agent")
    agent_role: AgentRole = Field(default=AgentRole.PROGRAMMER)
    agent_capabilities: List[AgentCapability] = Field(default_factory=list)
    llm_name: str = Field(default="default")
    llm_provider: str = Field(default="default")
    llm_api_key: Optional[str] = Field(default=None)
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.7)

    def __init__(self, **data):
        super().__init__(**data)
        if not self.agent_capabilities:
            self.agent_capabilities = [
                AgentCapability.CODE_GENERATION,
                AgentCapability.CODE_REVIEW,
                AgentCapability.REFACTORING
            ]
