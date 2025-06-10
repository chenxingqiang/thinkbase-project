"""Development instructions for code generation and review."""

from typing import Dict, Any, List, Optional, Set, Union, Type, cast
from dataclasses import dataclass, field
import time
import logging
import os
import hashlib
from pathlib import Path
from uuid import uuid4
from datetime import datetime

from agentflow.core.isa.formal_instruction import (
    FormalInstruction,
    InstructionType,
    ValidationContext,
    ValidationViolation,
    ViolationType,
    Severity
)

logger = logging.getLogger(__name__)

@dataclass
class DevelopmentMetrics:
    """Metrics for development operations."""
    tests_designed: int = 0
    issues_found: int = 0
    suggestions_made: int = 0
    changes_applied: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class TestDesignInstruction(FormalInstruction):
    """Design test cases and strategy."""
    
    def __init__(self):
        super().__init__(
            id=str(uuid4()),
            type=InstructionType.TEST,
            content="Design test cases and testing strategy",
            params={
                "test_type": "",
                "framework": "",
                "input_code": "",
                "output_format": ""
            },
            metadata={
                "capabilities": ["test_design", "code_analysis"],
                "tags": ["testing", "design", "development"],
                "cleanup_required": True,
                "resource_requirements": {"memory": 512, "cpu": 0.5}
            },
            name="test_design",
            description="Design test cases and testing strategy",
            version="1.0.0",
            author="AgentFlow",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enabled=True,
            priority=1
        )
        self._metrics = DevelopmentMetrics()
        
        # Add preconditions
        self.preconditions.append(
            lambda context: bool(context.data.get("input_code"))
        )
        
        # Add postconditions
        self.postconditions.append(
            lambda context: bool(context.data.get("test_cases"))
        )
        
        # Add invariants
        self.invariants.append(
            lambda context: len(context.data.get("test_cases", [])) > 0
        )

    def validate_preconditions(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate preconditions."""
        violations = []
        
        if not context.data.get("input_code"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Input code is required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        if not context.data.get("test_type"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Test type is required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        return violations

class CodeReviewInstruction(FormalInstruction):
    """Review code for quality and standards."""
    
    def __init__(self):
        super().__init__(
            id=str(uuid4()),
            type=InstructionType.ANALYSIS,
            content="Review code for quality and standards",
            params={
                "review_type": "",
                "standards": [],
                "input_code": "",
                "output_format": ""
            },
            metadata={
                "capabilities": ["code_review", "quality_analysis"],
                "tags": ["review", "quality", "development"],
                "cleanup_required": True,
                "resource_requirements": {"memory": 768, "cpu": 0.75}
            },
            name="code_review",
            description="Review code for quality and standards",
            version="1.0.0",
            author="AgentFlow",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enabled=True,
            priority=1
        )
        self._metrics = DevelopmentMetrics()
        
        # Add preconditions
        self.preconditions.append(
            lambda context: bool(context.data.get("input_code"))
        )
        
        # Add postconditions
        self.postconditions.append(
            lambda context: bool(context.data.get("review_results"))
        )
        
        # Add invariants
        self.invariants.append(
            lambda context: isinstance(context.data.get("standards", []), list)
        )

    def validate_preconditions(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate preconditions."""
        violations = []
        
        if not context.data.get("input_code"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Input code is required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        if not context.data.get("standards"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Review standards are required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        return violations

class CodeRefactoringInstruction(FormalInstruction):
    """Refactor code for better quality."""
    
    def __init__(self):
        super().__init__(
            id=str(uuid4()),
            type=InstructionType.TRANSFORM,
            content="Refactor code to improve quality and maintainability",
            params={
                "refactoring_type": "",
                "quality_threshold": 0.0,
                "input_code": "",
                "output_format": ""
            },
            metadata={
                "capabilities": ["code_refactoring", "code_analysis"],
                "tags": ["refactoring", "quality", "development"],
                "cleanup_required": True,
                "resource_requirements": {"memory": 1024, "cpu": 1.0}
            },
            name="code_refactoring",
            description="Refactor code to improve quality and maintainability",
            version="1.0.0",
            author="AgentFlow",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            enabled=True,
            priority=1
        )
        self._metrics = DevelopmentMetrics()
        
        # Add preconditions
        self.preconditions.append(
            lambda context: bool(context.data.get("input_code"))
        )
        
        # Add postconditions
        self.postconditions.append(
            lambda context: bool(context.data.get("refactored_code"))
        )
        
        # Add invariants
        self.invariants.append(
            lambda context: 0.0 <= context.data.get("quality_threshold", 0.0) <= 1.0
        )

    def validate_preconditions(self, context: ValidationContext) -> List[ValidationViolation]:
        """Validate preconditions."""
        violations = []
        
        if not context.data.get("input_code"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Input code is required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        if not context.data.get("refactoring_type"):
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Refactoring type is required",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        threshold = context.data.get("quality_threshold", 0.0)
        if not isinstance(threshold, (int, float)) or not 0.0 <= threshold <= 1.0:
            violations.append(
                ValidationViolation(
                    type=ViolationType.PRECONDITION,
                    message="Quality threshold must be between 0.0 and 1.0",
                    instruction=self,
                    severity=Severity.ERROR
                )
            )
            
        return violations 