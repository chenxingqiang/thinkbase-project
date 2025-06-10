"""Workflow state management module."""

from enum import Enum
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field, ConfigDict, computed_field, field_validator, model_validator
from dataclasses import dataclass, field
from .workflow_types import WorkflowStepStatus
from ..agents.agent_types import AgentStatus
from .exceptions import WorkflowStateError
import logging
from .base_types import AgentStatus as ConsolidatedAgentStatus, WorkflowStatus, StepStatus

logger = logging.getLogger(__name__)

class WorkflowState(BaseModel):
    """Workflow state class."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra='allow'
    )
    
    workflow_id: str = Field(description="Unique identifier for the workflow")
    name: str = Field(description="Name of the workflow")
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING, description="Current workflow status")
    start_time: Optional[datetime] = Field(default=None, description="Workflow start time")
    end_time: Optional[datetime] = Field(default=None, description="Workflow end time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Workflow metrics")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Workflow result")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    state_history: List[Dict[str, Any]] = Field(default_factory=list, description="History of state changes")
    
    @field_validator('status')
    @classmethod
    def validate_status(cls, v: Union[str, WorkflowStatus]) -> WorkflowStatus:
        """Validate status field."""
        if isinstance(v, str):
            try:
                return WorkflowStatus(v)
            except ValueError:
                raise ValueError(f"Invalid status: {v}")
        if not isinstance(v, WorkflowStatus):
            raise ValueError(f"Invalid status type: {type(v)}")
        return v
            
    def update_status(self, status: Union[WorkflowStatus, str]) -> None:
        """Update workflow status.
        
        Args:
            status: New workflow status
        """
        if isinstance(status, str):
            try:
                status = WorkflowStatus(status)
            except ValueError:
                raise ValueError(f"Invalid status: {status}")
        
        # Record the state change
        self.state_history.append({
            "status": self.status,
            "timestamp": datetime.now()
        })
        
        self.status = status
        self.updated_at = datetime.now()
        
    def validate(self) -> None:
        """Validate workflow state."""
        if not self.workflow_id:
            raise WorkflowStateError("Workflow ID is required")
        if not self.name:
            raise WorkflowStateError("Workflow name is required")
        if not isinstance(self.status, WorkflowStatus):
            raise WorkflowStateError(f"Invalid status type: {type(self.status)}")
        if self.status not in WorkflowStatus.__members__.values():
            raise WorkflowStateError(f"Invalid status value: {self.status}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow state to dictionary.
        
        Returns:
            Dict containing workflow state
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "result": self.result,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "state_history": self.state_history
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowState':
        """Create instance from dictionary.
        
        Args:
            data: Dictionary data
            
        Returns:
            WorkflowState instance
        """
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if "updated_at" in data and isinstance(data["updated_at"], str):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        return cls(**data)

class StepState(BaseModel):
    """Step state class."""
    
    step_id: str = Field(description="Unique identifier for the step")
    name: str = Field(description="Name of the step")
    status: WorkflowStepStatus = Field(default=WorkflowStepStatus.PENDING, description="Current step status")
    start_time: Optional[datetime] = Field(default=None, description="Step start time")
    end_time: Optional[datetime] = Field(default=None, description="Step end time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Step metrics")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Step result")
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )
    
    def validate(self) -> None:
        """Validate step state."""
        if not self.step_id:
            raise WorkflowStateError("Step ID is required")
        if not self.name:
            raise WorkflowStateError("Step name is required")
    
    def update_status(self, status: WorkflowStepStatus) -> None:
        """Update step status.
        
        Args:
            status: New step status
        """
        if not isinstance(status, WorkflowStepStatus):
            raise TypeError(f"Expected WorkflowStepStatus, got {type(status)}")
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert step state to dictionary.
        
        Returns:
            Dict containing step state
        """
        return {
            "step_id": self.step_id,
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "result": self.result
        }

class AgentState(BaseModel):
    """Agent state class."""
    
    agent_id: str = Field(description="Unique identifier for the agent")
    name: str = Field(description="Name of the agent")
    status: ConsolidatedAgentStatus = Field(default=ConsolidatedAgentStatus.INITIALIZED, description="Current agent status")
    start_time: Optional[datetime] = Field(default=None, description="Agent start time")
    end_time: Optional[datetime] = Field(default=None, description="Agent end time")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Agent metrics")
    result: Optional[Dict[str, Any]] = Field(default=None, description="Agent result")
    
    model_config = ConfigDict(
        validate_assignment=True,
        use_enum_values=True
    )
    
    def validate(self) -> None:
        """Validate agent state."""
        if not self.agent_id:
            raise WorkflowStateError("Agent ID is required")
        if not self.name:
            raise WorkflowStateError("Agent name is required")
    
    def update_status(self, status: ConsolidatedAgentStatus) -> None:
        """Update agent status.
        
        Args:
            status: New agent status
        """
        if not isinstance(status, ConsolidatedAgentStatus):
            raise TypeError(f"Expected ConsolidatedAgentStatus, got {type(status)}")
        self.status = status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary.
        
        Returns:
            Dict containing agent state
        """
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "metrics": self.metrics,
            "result": self.result
        }

class WorkflowStateManager:
    """Workflow state manager class."""

    def __init__(self):
        """Initialize workflow state manager."""
        self.workflow_states: Dict[str, WorkflowState] = {}
        self.step_states: Dict[str, Dict[str, StepState]] = {}
        self.agent_states: Dict[str, Dict[str, AgentState]] = {}
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize workflow state manager."""
        if not self._initialized:
            # Initialize state storage
            self.workflow_states = {}
            self.step_states = {}
            self.agent_states = {}
            self._initialized = True
            
    async def cleanup(self) -> None:
        """Clean up workflow state manager."""
        try:
            # Reset state
            self.workflow_states = {}
            self.step_states = {}
            self.agent_states = {}
            self._initialized = False
        except Exception as e:
            logger.error(f"Error during state cleanup: {str(e)}")
            raise
    
    def initialize_workflow(self, workflow_id: str, name: str) -> None:
        """Initialize workflow state.
        
        Args:
            workflow_id: Workflow ID
            name: Workflow name
        """
        self.workflow_states[workflow_id] = WorkflowState(
            workflow_id=workflow_id,
            name=name
        )
        self.step_states[workflow_id] = {}
        self.agent_states[workflow_id] = {}
    
    def initialize_step(self, workflow_id: str, step_id: str, name: str) -> None:
        """Initialize step state.
        
        Args:
            workflow_id: Workflow ID
            step_id: Step ID
            name: Step name
        """
        if workflow_id not in self.step_states:
            self.step_states[workflow_id] = {}
        self.step_states[workflow_id][step_id] = StepState(
            step_id=step_id,
            name=name
        )
    
    def initialize_agent(self, workflow_id: str, agent_id: str, name: str) -> None:
        """Initialize agent state.
        
        Args:
            workflow_id: Workflow ID
            agent_id: Agent ID
            name: Agent name
        """
        if workflow_id not in self.agent_states:
            self.agent_states[workflow_id] = {}
        self.agent_states[workflow_id][agent_id] = AgentState(
            agent_id=agent_id,
            name=name
        )

    def update_workflow_status(self, workflow_id: str, status: WorkflowStatus) -> None:
        """Update workflow status.
        
        Args:
            workflow_id: Workflow ID
            status: New workflow status
        """
        if workflow_id in self.workflow_states:
            self.workflow_states[workflow_id].update_status(status)
    
    def update_step_status(self, workflow_id: str, step_id: str, status: WorkflowStepStatus) -> None:
        """Update step status.
        
        Args:
            workflow_id: Workflow ID
            step_id: Step ID
            status: New step status
        """
        if workflow_id in self.step_states and step_id in self.step_states[workflow_id]:
            self.step_states[workflow_id][step_id].update_status(status) 