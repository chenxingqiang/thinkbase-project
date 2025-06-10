"""Workflow configuration module."""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class WorkflowConfig(BaseModel):
    """Workflow configuration class."""
    
    max_iterations: int = Field(default=5, description="Maximum number of iterations")
    timeout: int = Field(default=3600, description="Timeout in seconds")
    logging_level: str = Field(default="INFO", description="Logging level")
    required_fields: List[str] = Field(default_factory=list, description="Required fields")
    error_handling: Dict[str, Any] = Field(default_factory=dict, description="Error handling configuration")
    retry_policy: Optional[Dict[str, Any]] = Field(default=None, description="Retry policy configuration")
    error_policy: Optional[Dict[str, Any]] = Field(default=None, description="Error policy configuration")
    is_distributed: bool = Field(default=False, description="Whether the workflow is distributed")
    distributed: bool = Field(default=False, description="Whether the workflow is distributed (alias)")
    steps: List[Dict[str, Any]] = Field(default_factory=list, description="Workflow steps")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Workflow metadata")
    agents: Dict[str, Any] = Field(default_factory=dict, description="Agent configurations")
    
    class Config:
        """Pydantic configuration."""
        arbitrary_types_allowed = True
        extra = "allow" 