from enum import Enum

class StepStatus(str, Enum):
    """Step status enum (superset).

    ``str()`` of a member returns its value (see ``__str__`` below), so
    ``str(StepStatus.COMPLETED) == "completed"`` holds for callers that
    interpolate statuses into messages and serialized payloads.
    """

    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZED = "initialized"
    STOPPED = "stopped"
    WAITING = "waiting"
    SKIPPED = "skipped"

    def __str__(self) -> str:
        return self.value
    
class NodeType(str, Enum):
    """Node type enum."""
    AGENT = "agent"
    PROCESSOR = "processor"
    CONNECTOR = "connector"
    
class NodeState(str, Enum):
    """Node state enum."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    
class WorkflowStatus(str, Enum):
    """Workflow status enum."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    STOPPED = "stopped"
    
class ConfigurationType(str, Enum):
    """Configuration type enum."""
    GENERIC = "generic"
    RESEARCH = "research"
    ASSISTANT = "assistant"
    EXPERT = "expert"
    CUSTOM = "custom"
    
class AgentMode(str, Enum):
    """Agent mode enum."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    INTERACTIVE = "interactive" 


"""Base types module."""

from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union

class AgentType(str, Enum):
    """Agent type enum."""
    GENERIC = "generic"
    CUSTOM = "custom"
    RESEARCH = "research"
    DATA_SCIENCE = "data_science"
    WORKFLOW = "workflow"
    ASSISTANT = "assistant"
    INTERACTIVE = "interactive"

class AgentMode(str, Enum):
    """Agent mode enum."""
    SYNC = "sync"
    ASYNC = "async"
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"
    INTERACTIVE = "interactive"

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

class WorkflowStepType(str, Enum):
    """Workflow step type enum."""
    TRANSFORM = "transform"
    RESEARCH = "research"
    RESEARCH_EXECUTION = "research"
    DOCUMENT = "document"
    DOCUMENT_GENERATION = "document"
    AGENT = "agent"
    CUSTOM = "custom"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    DISTRIBUTED = "distributed"
    CONDITIONAL = "conditional"

class WorkflowStatus(str, Enum):
    """Workflow status enum."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    ERROR = "error"
    INITIALIZED = "initialized"
    COMPLETED = "completed"
    STOPPED = "stopped"

    def __str__(self) -> str:
        return self.value

    def __getattr__(self, name):
        """Allow accessing statuses as class attributes."""
        try:
            return self._member_map_[name]
        except KeyError:
            raise AttributeError(name)

class StepStatus(str, Enum):
    """Step status enum."""
    PENDING = "pending"
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
    SKIPPED = "skipped"

    def __str__(self) -> str:
        return self.value

class MessageRole(str, Enum):
    """Message role enum."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    FUNCTION = "function"

class MessageType(str, Enum):
    """Message type enum."""
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    FILE = "file"
    RESULT = "result"
    ERROR = "error"
    STATUS = "status"
    METRICS = "metrics"
    CUSTOM = "custom"
    CODE = "code"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    COMMAND = "command"

class DictKeyType(str, Enum):
    """Dictionary key type enum."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    UUID = "uuid"
    ANY = "any"