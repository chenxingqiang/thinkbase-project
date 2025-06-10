"""Base workflow module."""

from typing import Dict, Any
from pydantic import BaseModel, Field
import uuid

class WorkflowBase(BaseModel):
    """Base workflow class."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(default="")
    description: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict) 