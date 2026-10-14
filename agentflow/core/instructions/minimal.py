"""Minimal instruction set for basic agent capabilities."""

from enum import Enum
from typing import Dict, Any, Optional, cast
from dataclasses import dataclass
import uuid
import json
import time
import logging
from .base import (
    BaseInstruction,
    ExecutionContext,
    ExecutionResult,
    InstructionType,
    InstructionMetadata
)
from ..isa.result import InstructionStatus, InstructionResult

logger = logging.getLogger(__name__)

@dataclass
class MinimalExecutionContext:
    """Minimal execution context with only essential fields."""
    params: Dict[str, Any]

class BaseMinimalInstruction(BaseInstruction):
    """Base class for minimal instructions with simplified interface."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize minimal instruction.
        
        Args:
            name: Instruction name
            description: Optional instruction description
            instruction_type: Optional instruction type
            metadata: Optional instruction metadata
        """
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.OBJECTIVE_HANDLING,
            metadata=metadata or InstructionMetadata(
                capabilities={"minimal_execution"},
                tags={name.lower(), "minimal"}
            )
        )
    
    async def execute(self, context: MinimalExecutionContext) -> ExecutionResult:
        """Execute minimal instruction.
        
        Args:
            context: Minimal execution context
            
        Returns:
            ExecutionResult: Execution results
        """
        start_time = time.time()
        
        try:
            # Convert minimal context to full context
            full_context = ExecutionContext(
                params=context.params,
                state={},
                metadata={},
                collaboration_info=None,
                resources=None
            )
            
            # Validate context
            if not await self.validate_context(full_context):
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": "Context validation failed"}]
                )
            
            # Execute core logic
            result = await self._execute_core(full_context.params)
            
            # Convert InstructionResult to ExecutionResult
            success = result.status == InstructionStatus.COMPLETED
            output = (
                {"content": result.content} if isinstance(result.content, str)
                else (result.content if result.content is not None else {})
            )
            
            # Update metrics
            execution_time = time.time() - start_time
            self.update_metrics(success, execution_time)
            
            return ExecutionResult(
                success=success,
                output=cast(Dict[str, Any], output),
                errors=[{"error": result.error}] if result.error else [],
                state_updates={},
                metrics=result.metrics if result.metrics is not None else {}
            )
            
        except Exception as e:
            logger.error(f"Instruction execution failed: {str(e)}")
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                state_updates={},
                metrics={}
            )
            
        finally:
            await self.cleanup()
    
    async def _execute_core(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute core instruction logic.
        
        Args:
            params: Execution parameters
            
        Returns:
            InstructionResult: Core execution result
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement _execute_core")

class MinimalTestDesign(BaseMinimalInstruction):
    """Minimal test design instruction."""
    
    def __init__(self):
        super().__init__(
            name="test_design",
            description="Design minimal test cases",
            instruction_type=InstructionType.TEST_DESIGN
        )
    
    async def _execute_core(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute test design logic."""
        try:
            # Implement test design logic here
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.COMPLETED,
                content=json.dumps({"message": "Test design completed"}),
                error=None,
                metrics={"test_cases": 0}
            )
        except Exception as e:
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.FAILED,
                content=None,
                error=str(e),
                metrics={}
            )

class MinimalCodeReview(BaseMinimalInstruction):
    """Minimal code review instruction."""
    
    def __init__(self):
        super().__init__(
            name="code_review",
            description="Perform minimal code review",
            instruction_type=InstructionType.CODE_REVIEW
        )
    
    async def _execute_core(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute code review logic."""
        try:
            # Implement code review logic here
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.COMPLETED,
                content=json.dumps({"message": "Code review completed"}),
                error=None,
                metrics={"issues_found": 0}
            )
        except Exception as e:
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.FAILED,
                content=None,
                error=str(e),
                metrics={}
            )

class MinimalCodeFile(BaseMinimalInstruction):
    """Minimal code file instruction."""
    
    def __init__(self):
        super().__init__(
            name="code_file",
            description="Handle minimal code file operations",
            instruction_type=InstructionType.CODE_GENERATION
        )
    
    async def _execute_core(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute code file operation logic."""
        try:
            # Implement code file operation logic here
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.COMPLETED,
                content=json.dumps({"message": "Code file operation completed"}),
                error=None,
                metrics={"files_processed": 0}
            )
        except Exception as e:
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.FAILED,
                content=None,
                error=str(e),
                metrics={}
            ) 

class MinimalGitWorkflow(BaseMinimalInstruction):
    """Minimal Git workflow instruction.

    Supports the ``status`` and ``commit`` operations used by the
    DeepSeek programming instructions: repository status inspection
    and atomic commit creation.  Operations are validated and reported
    through :class:`InstructionResult` without requiring a live Git
    installation (useful for tests and dry runs).
    """

    def __init__(self):
        super().__init__(
            name="git_workflow",
            description="Minimal Git workflow: status and commit",
            instruction_type=MinimalInstructionType.GIT_WORKFLOW
        )

    async def _execute_core(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute a Git workflow operation."""
        operation = params.get("operation")
        try:
            if operation == "status":
                return InstructionResult(
                    id=str(uuid.uuid4()),
                    status=InstructionStatus.COMPLETED,
                    content=json.dumps({
                        "operation": "status",
                        "branch": params.get("branch", "main"),
                        "clean": True,
                        "files": params.get("files", []),
                    }),
                    error=None,
                    metrics={"operation": "status"},
                )
            elif operation == "commit":
                return InstructionResult(
                    id=str(uuid.uuid4()),
                    status=InstructionStatus.COMPLETED,
                    content=json.dumps({
                        "operation": "commit",
                        "message": params.get("message", "chore: update"),
                        "files": params.get("files", []),
                    }),
                    error=None,
                    metrics={"operation": "commit"},
                )
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.FAILED,
                content=None,
                error=f"Unsupported git operation: {operation!r}",
                metrics={},
            )
        except Exception as e:
            return InstructionResult(
                id=str(uuid.uuid4()),
                status=InstructionStatus.FAILED,
                content=None,
                error=str(e),
                metrics={},
            )


class MinimalInstructionType(str, Enum):
    """Types available in the minimal instruction set."""

    TEST_DESIGN = "test_design"
    CODE_REVIEW = "code_review"
    CODE_FILE = "code_file"
    GIT_WORKFLOW = "git_workflow"


#: Registry of the minimal instruction set shipped with AgentFlow.
MINIMAL_INSTRUCTION_SET = {
    "minimal_test_design": MinimalTestDesign,
    "minimal_code_review": MinimalCodeReview,
    "minimal_code_file": MinimalCodeFile,
    "minimal_git_workflow": MinimalGitWorkflow,
}
