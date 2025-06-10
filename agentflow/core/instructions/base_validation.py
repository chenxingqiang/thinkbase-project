"""Base validation classes for instruction validation."""

from enum import Enum
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pydantic import BaseModel

class ViolationType(str, Enum):
    """Type of validation violation."""
    PRECONDITION = "precondition"
    POSTCONDITION = "postcondition"
    INVARIANT = "invariant"
    TYPE = "type"
    CONSTRAINT = "constraint"

class Severity(str, Enum):
    """Severity level of validation violation."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationContext:
    """Context for validation."""
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent: Optional['ValidationContext'] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class ValidationViolation:
    """Represents a validation violation."""
    type: ViolationType
    message: str
    instruction: Any  # Reference to the instruction that caused the violation
    severity: Severity = Severity.ERROR
    context: Optional[Dict[str, Any]] = None
    location: Optional[str] = None
    
    def __str__(self) -> str:
        return f"{self.severity.value.upper()}: {self.message} ({self.type.value})" 