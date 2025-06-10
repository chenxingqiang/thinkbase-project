from typing import List, Any, Optional
from enum import Enum
from pydantic import BaseModel, Field

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
