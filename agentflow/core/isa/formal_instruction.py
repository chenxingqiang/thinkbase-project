"""Formal instruction module."""

from enum import Enum
from typing import Dict, Any, List, Optional, Union, Callable
from pydantic import BaseModel, Field
from uuid import uuid4
from datetime import datetime
from agentflow.core.instructions.base_validation import ValidationContext, ValidationViolation, ViolationType, Severity

class InstructionType(str, Enum):
    """Instruction type enum."""
    CONTROL = "control"
    TRANSFORM = "transform"
    TEST = "test"
    ANALYSIS = "analysis"
    CUSTOM = "custom"

class FormalInstruction(BaseModel):
    """Formal instruction class."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    type: InstructionType = Field(default=InstructionType.CONTROL)
    content: str = Field(default="")
    parameters: Dict[str, Any] = Field(default_factory=dict, alias="params")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
    name: str = Field(default="")
    description: str = Field(default="")
    version: str = Field(default="1.0.0")
    author: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: List[str] = Field(default_factory=list)
    enabled: bool = Field(default=True)
    priority: int = Field(default=0)
    deadline: Optional[datetime] = Field(default=None)
    locality: Optional[str] = Field(default=None)
    preconditions: List[Union[Dict[str, Any], Callable]] = Field(default_factory=list)
    postconditions: List[Union[Dict[str, Any], Callable]] = Field(default_factory=list)
    invariants: List[Union[Dict[str, Any], Callable]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        allow_population_by_field_name = True
        validate_assignment = True
        arbitrary_types_allowed = True
        
    def validate_preconditions(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate preconditions.
        
        Args:
            context: Validation context.
            
        Returns:
            List[ValidationViolation]: List of validation violations.
        """
        violations = []
        
        for precondition in self.preconditions:
            if isinstance(precondition, dict):
                condition = precondition.get("condition")
                if not condition:
                    continue
                    
                try:
                    if not eval(condition, {"context": context}):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.PRECONDITION,
                                message=f"Precondition '{condition}' not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.PRECONDITION,
                            message=f"Error evaluating precondition '{condition}': {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
            elif callable(precondition):
                try:
                    if not precondition(context):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.PRECONDITION,
                                message=f"Precondition not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.PRECONDITION,
                            message=f"Error evaluating precondition: {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
                
        return violations
        
    def validate_postconditions(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate postconditions.
        
        Args:
            context: Validation context.
            
        Returns:
            List[ValidationViolation]: List of validation violations.
        """
        violations = []
        
        for postcondition in self.postconditions:
            if isinstance(postcondition, dict):
                condition = postcondition.get("condition")
                if not condition:
                    continue
                    
                try:
                    if not eval(condition, {"context": context}):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.POSTCONDITION,
                                message=f"Postcondition '{condition}' not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.POSTCONDITION,
                            message=f"Error evaluating postcondition '{condition}': {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
            elif callable(postcondition):
                try:
                    if not postcondition(context):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.POSTCONDITION,
                                message=f"Postcondition not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.POSTCONDITION,
                            message=f"Error evaluating postcondition: {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
                
        return violations
        
    def validate_invariants(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate invariants.
        
        Args:
            context: Validation context.
            
        Returns:
            List[ValidationViolation]: List of validation violations.
        """
        violations = []
        
        for invariant in self.invariants:
            if isinstance(invariant, dict):
                condition = invariant.get("condition")
                if not condition:
                    continue
                    
                try:
                    if not eval(condition, {"context": context}):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.INVARIANT,
                                message=f"Invariant '{condition}' not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.INVARIANT,
                            message=f"Error evaluating invariant '{condition}': {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
            elif callable(invariant):
                try:
                    if not invariant(context):
                        violations.append(
                            ValidationViolation(
                                type=ViolationType.INVARIANT,
                                message=f"Invariant not satisfied",
                                instruction=self,
                                severity=Severity.ERROR
                            )
                        )
                except Exception as e:
                    violations.append(
                        ValidationViolation(
                            type=ViolationType.INVARIANT,
                            message=f"Error evaluating invariant: {str(e)}",
                            instruction=self,
                            severity=Severity.ERROR
                        )
                    )
                
        return violations
