"""Agent state and status management."""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

class AgentStatus(str, Enum):
    """Agent status enum."""
    PENDING = "pending"
    PROCESSING = "processing"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZED = "initialized"
    COMPLETED = "completed"
    STOPPED = "stopped"
    WAITING = "waiting"

    def __str__(self) -> str:
        return self.value

@dataclass
class AgentState:
    """Represents the complete state of an agent."""
    status: AgentStatus = AgentStatus.INITIALIZED
    metadata: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)
    current_task: Optional[str] = None
    error_message: Optional[str] = None
    
    def update(self, 
               status: Optional[AgentStatus] = None, 
               metadata: Optional[Dict[str, Any]] = None, 
               current_task: Optional[str] = None, 
               error_message: Optional[str] = None):
        """Update the agent's state."""
        if status is not None:
            self.status = status
        if metadata is not None:
            self.metadata.update(metadata)
        if current_task is not None:
            self.current_task = current_task
        if error_message is not None:
            self.error_message = error_message
        
        # Record state change in history
        state_record = {
            "status": self.status.value,
            "metadata": self.metadata.copy(),
            "current_task": self.current_task,
            "error_message": self.error_message
        }
        self.history.append(state_record)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to a dictionary representation."""
        return {
            "status": self.status.value,
            "metadata": self.metadata,
            "current_task": self.current_task,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> 'AgentState':
        """Create an AgentState from a dictionary."""
        status_str = state_dict.get('status', 'INITIALIZED')
        try:
            status = AgentStatus(status_str.lower())
        except ValueError:
            status = AgentStatus.INITIALIZED
            
        return cls(
            status=status,
            metadata=state_dict.get('metadata', {}),
            current_task=state_dict.get('current_task'),
            error_message=state_dict.get('error_message')
        )
