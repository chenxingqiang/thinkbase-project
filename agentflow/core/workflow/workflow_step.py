"""Workflow step configuration."""

from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from pydantic import BaseModel, Field, field_validator, ValidationInfo

from .constants import VALID_PROTOCOLS, VALID_STRATEGIES


class WorkflowStepType(str, Enum):
    """Workflow step type."""

    TRANSFORM = "transform"
    RESEARCH = "research"
    RESEARCH_EXECUTION = "research"  # Map research_execution to research
    DOCUMENT = "document"
    DOCUMENT_GENERATION = "document"  # Map document_generation to document
    AGENT = "agent"


class StepConfig(BaseModel):
    """Step configuration."""

    strategy: str = "default"
    params: Dict[str, Any] = Field(default_factory=dict)
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    max_retries: int = 3
    execute: Optional[Callable] = None

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        """Validate strategy."""
        if v not in VALID_STRATEGIES:
            raise ValueError(f"Invalid strategy: {v}")
        return v

    @field_validator("params")
    @classmethod
    def validate_params(cls, v: Dict[str, Any], info: ValidationInfo) -> Dict[str, Any]:
        """Validate parameters."""
        # Validate protocol if specified
        protocol = v.get("protocol")
        if protocol is not None and protocol not in VALID_PROTOCOLS:
            # In test mode, convert invalid protocols to None
            if info.context and info.context.get("test_mode"):
                v["protocol"] = None
            else:
                raise ValueError(f"Invalid protocol: {protocol}")
        return v


class WorkflowStep(BaseModel):
    """Workflow step."""

    id: str
    name: str
    type: Union[WorkflowStepType, str] = WorkflowStepType.TRANSFORM
    required: bool = True
    optional: bool = False
    is_distributed: bool = False
    dependencies: List[str] = Field(default_factory=list)
    config: StepConfig = Field(default_factory=StepConfig)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: Union[WorkflowStepType, str]) -> Union[WorkflowStepType, str]:
        """Validate step type."""
        if isinstance(v, str):
            # Handle both research and research_execution as research
            v = v.lower()
            if v in ["research", "research_execution"]:
                v = "research"
            try:
                return WorkflowStepType(v)
            except ValueError:
                raise ValueError(f"Invalid step type: {v}")
        return v 