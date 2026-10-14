"""Agent configuration models.

Canonical home of the rich :class:`AgentConfig` and the LLM
:class:`ModelConfig` used across the workflow engine, the agent
runtime and the examples.  ``agentflow.core.config`` re-exports both
so historical import paths keep working.
"""

from typing import Any, Dict, List, Optional, Union
from enum import Enum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .base_types import AgentType
from .workflow_types import WorkflowConfig


class AgentRole(Enum):
    """Legacy collaborative-coding role labels (kept for compatibility)."""

    PROGRAMMER = "programmer"
    REVIEWER = "reviewer"
    ARCHITECT = "architect"


class AgentCapability(Enum):
    """Legacy capability labels (kept for compatibility)."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    REFACTORING = "refactoring"


VALID_MODEL_PROVIDERS = {
    "openai",
    "anthropic",
    "deepseek",
    "azure",
    "google",
    "cohere",
    "huggingface",
    "ollama",
    "local",
    "default",
    "mock",
}


class ModelConfig(BaseModel):
    """LLM model configuration.

    The provider is validated against :data:`VALID_MODEL_PROVIDERS` so
    misconfigured agents fail at construction time instead of at the
    first LLM call.  Dict-style access (``cfg["name"]``,
    ``cfg.get("temperature")``, ``"stop" in cfg``) is supported for
    call sites that treat the model configuration like a plain
    mapping.
    """

    model_config = ConfigDict(extra="allow")

    provider: str = Field(default="openai")
    name: str = Field(default="gpt-4")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4096)
    top_p: float = Field(default=1.0)
    frequency_penalty: float = Field(default=0.0)
    presence_penalty: float = Field(default=0.0)
    stop: Optional[Union[str, List[str]]] = Field(default=None)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, value: str) -> str:
        if value not in VALID_MODEL_PROVIDERS:
            raise ValueError(
                f"Unknown model provider {value!r}; expected one of "
                f"{sorted(VALID_MODEL_PROVIDERS)}"
            )
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value, or ``default`` if missing."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


class AgentConfig(BaseModel):
    """Rich agent configuration.

    Field contract (exercised by ``tests/unit/test_agent_initialization.py``
    and ``tests/unit/test_workflow_types.py``):

    - ``name``/``description``/``version``/``mode``/``is_distributed``
      basic identity fields;
    - ``type`` validated against the :class:`AgentType` values
      (``research``, ``generic``, ...) — invalid types raise
      ``ValueError`` at construction;
    - ``model`` nested :class:`ModelConfig` (dict or instance),
      defaulting to GPT-4 on OpenAI;
    - ``workflow`` nested :class:`WorkflowConfig` (dict or instance),
      optional;
    - ``id`` auto-generated when omitted;
    - dict-style access mirroring :class:`ModelConfig`.

    Unknown extra keys are preserved (``extra="allow"``) so legacy
    configuration files with additional fields keep loading.
    """

    # validate_default: the empty-name guard must also fire when ``name``
    # falls back to its default (``AgentConfig(type="generic")``), not
    # only when it is passed explicitly.
    model_config = ConfigDict(extra="allow", validate_default=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str = Field(default="")
    description: str = Field(default="")
    type: str = Field(...)
    version: str = Field(default="1.0.0")
    mode: str = Field(default="sequential")
    is_distributed: bool = Field(default=False)
    system_prompt: Optional[str] = Field(default=None)
    model: ModelConfig = Field(default_factory=ModelConfig)
    # Legacy flat provider key (used by ConfigManager round-trips),
    # constrained to the same allow-list as ``model.provider`` via a
    # pattern so failures surface as pydantic pattern errors.
    model_provider: Optional[str] = Field(
        default=None,
        pattern=r"^(anthropic|azure|cohere|deepseek|default|google|"
                r"huggingface|local|mock|ollama|openai)$",
    )
    workflow: Optional[WorkflowConfig] = Field(default=None)
    # Flat runtime bounds kept from the legacy config contract; the
    # same constraints also apply inside ``workflow``/``WorkflowConfig``.
    timeout: float = Field(default=0.0, ge=0)
    max_iterations: int = Field(default=10, ge=1, le=100)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Name cannot be empty")
        return value

    @model_validator(mode="after")
    def _apply_model_provider(self) -> "AgentConfig":
        """Fold a flat ``model_provider`` into ``model.provider``."""
        if self.model_provider is not None:
            object.__setattr__(
                self, "model",
                self.model.model_copy(update={"provider": self.model_provider}),
            )
        return self

    @field_validator("type", mode="before")
    @classmethod
    def _validate_type(cls, value: Any) -> str:
        if isinstance(value, AgentType):
            return value.value
        if isinstance(value, str) and value in {t.value for t in AgentType}:
            return value
        raise ValueError(
            f"Invalid agent type {value!r}; expected one of "
            f"{sorted(t.value for t in AgentType)}"
        )

    @field_validator("mode", mode="before")
    @classmethod
    def _validate_mode(cls, value: Any) -> str:
        if isinstance(value, Enum):
            return value.value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        """Return a configuration value, or ``default`` if missing."""
        try:
            return getattr(self, key)
        except AttributeError:
            return default

    def __getitem__(self, key: str) -> Any:
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


__all__ = [
    "AgentRole",
    "AgentCapability",
    "AgentConfig",
    "ModelConfig",
    "VALID_MODEL_PROVIDERS",
]
