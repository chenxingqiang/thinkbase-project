"""Task planning system for converting thoughts to instructions."""

from typing import Dict, Any, List, Optional, Union, Type, cast, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import sys
import asyncio
import time
import random

# Add parent directory to path to resolve imports
sys.path.append(str(Path(__file__).parent.parent))
from .instructions.base import BaseInstruction, InstructionResult, InstructionStatus
from .instructions.programmer import CodeFileInstruction as BaseCodeFileInstruction
from .instructions.programmer import GitWorkflowInstruction as BaseGitWorkflowInstruction
from ..ell.integration import ell_integration
from ..ell.types.message import Message, MessageRole

from .instructions import (
    BaseInstruction,
    InstructionType,
    RequirementAnalysisInstruction,
    CodeAnalysisInstruction,
    ProjectInitializationInstruction,
    TestDesignInstruction,
    CodeReviewInstruction,
    DocumentationGenerationInstruction,
    ProgrammerThinkingInstruction,
    CodeFileInstruction,
    GitWorkflowInstruction
)

class TaskType(Enum):
    """Types of programming tasks."""
    ANALYSIS = "analysis"
    IMPLEMENTATION = "implementation"
    TESTING = "testing"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"
    BUGFIX = "bugfix"

@dataclass
class TaskPhase:
    """Task phase class."""
    name: str
    instructions: List[BaseInstruction]
    dependencies: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize dependencies if None."""
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class RetryState:
    """State for retry mechanism."""
    count: int = 0
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    last_error: Optional[Exception] = None
    last_attempt: Optional[float] = None

    def should_retry(self) -> bool:
        """Check if should retry."""
        return self.count < self.max_retries

    def get_delay(self) -> float:
        """Get delay for next retry with exponential backoff."""
        delay = min(self.base_delay * (2 ** self.count), self.max_delay)
        jitter = delay * 0.1 * random.random()
        return delay + jitter

    def record_attempt(self, error: Optional[Exception] = None) -> None:
        """Record retry attempt."""
        self.count += 1
        self.last_error = error
        self.last_attempt = time.time()

    def reset(self) -> None:
        """Reset retry state."""
        self.count = 0
        self.last_error = None
        self.last_attempt = None

class ComplexityLevel(Enum):
    """Complexity levels for ELL operations."""
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"

class BaseELLInstruction(BaseInstruction):
    """Base class for ELL-enabled instructions."""
    
    def __init__(self, name: str, description: str):
        """Initialize instruction."""
        super().__init__(name, description)
        self.ell = ell_integration
        self.retry_state = RetryState()
    
    def determine_complexity(self, context: Dict[str, Any]) -> ComplexityLevel:
        """Determine operation complexity.
        
        Args:
            context: Operation context
            
        Returns:
            ComplexityLevel: Operation complexity
        """
        # Base complexity on operation type
        operation = context.get("operation", "").lower()
        if operation in ["create", "modify", "refactor", "analyze"]:
            base_complexity = ComplexityLevel.COMPLEX
        elif operation in ["review", "test", "document"]:
            base_complexity = ComplexityLevel.MODERATE
        else:
            base_complexity = ComplexityLevel.SIMPLE
        
        # Adjust based on content size
        content = context.get("content", "")
        if isinstance(content, str) and len(content) > 1000:
            base_complexity = ComplexityLevel.COMPLEX
        
        # Adjust based on dependencies
        dependencies = context.get("dependencies", [])
        if len(dependencies) > 3:
            base_complexity = ComplexityLevel.COMPLEX
        
        return base_complexity
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction with ELL integration.
        
        Args:
            context: Execution context
            
        Returns:
            Dict[str, Any]: Execution result
        """
        while self.retry_state.should_retry():
            try:
                # Determine complexity
                complexity = self.determine_complexity(context)
                context["complexity"] = complexity.value
                
                # Create message
                message = Message(
                    role=MessageRole.USER,
                    content=str(context),
                    metadata=self._get_metadata(context)
                )
                
                # Process message
                response = await self.ell.process_message(message)
                
                # Reset retry state on success
                self.retry_state.reset()
                
                return {
                    "status": "success",
                    "result": response.content,
                    "metadata": response.metadata,
                    "complexity": complexity.value
                }
                
            except Exception as e:
                self.retry_state.record_attempt(e)
                if not self.retry_state.should_retry():
                    break
                    
                # Wait before retrying
                await asyncio.sleep(self.retry_state.get_delay())
        
        # If we get here, all retries failed
        return {
            "status": "error",
            "result": str(self.retry_state.last_error),
            "metadata": {
                "retry_count": self.retry_state.count,
                "error": str(self.retry_state.last_error),
                "last_attempt": self.retry_state.last_attempt
            }
        }
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for ELL message.
        
        Args:
            context: Execution context
            
        Returns:
            Dict[str, Any]: Message metadata
        """
        raise NotImplementedError("Subclasses must implement _get_metadata")

class CodeFileInstructionImpl(BaseELLInstruction, BaseCodeFileInstruction):
    """Code file instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        BaseELLInstruction.__init__(
            self,
            name="code_file_instruction",
            description="Manages code file operations"
        )
        BaseCodeFileInstruction.__init__(self)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute code file instruction with ELL integration."""
        context["complexity"] = "complex" if context.get("operation") in ["create", "modify"] else "simple"
        return await super()._execute_impl(context)
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for code file instruction."""
        return {
            "instruction_type": "code_file",
            "operation": context.get("operation"),
            "file_path": context.get("file_path"),
            "content": context.get("content"),
            "complexity": context.get("complexity", "simple")
        }

class GitWorkflowInstructionImpl(BaseELLInstruction, BaseGitWorkflowInstruction):
    """Git workflow instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        BaseELLInstruction.__init__(
            self,
            name="git_workflow_instruction",
            description="Manages Git workflow operations"
        )
        BaseGitWorkflowInstruction.__init__(self)
    
    async def _execute_impl(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Git workflow instruction with ELL integration."""
        context["complexity"] = "complex" if context.get("operation") in ["commit", "merge"] else "simple"
        return await super()._execute_impl(context)
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for Git workflow instruction."""
        return {
            "instruction_type": "git_workflow",
            "operation": context.get("operation"),
            "config": context.get("config", {}),
            "complexity": context.get("complexity", "simple")
        }

class RequirementAnalysisInstructionImpl(BaseELLInstruction):
    """Requirement analysis instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        super().__init__(
            name="requirement_analysis",
            description="Analyze project requirements and constraints"
        )
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for requirement analysis."""
        return {
            "instruction_type": "requirement_analysis",
            "analysis_type": context.get("analysis_type", "general"),
            "scope": context.get("scope", "full"),
            "complexity": context.get("complexity", "complex")
        }

class TestDesignInstructionImpl(BaseELLInstruction):
    """Test design instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        super().__init__(
            name="test_design",
            description="Design test cases and testing strategy"
        )
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for test design."""
        return {
            "instruction_type": "test_design",
            "test_type": context.get("test_type", "unit"),
            "framework": context.get("framework", "pytest"),
            "complexity": context.get("complexity", "complex")
        }

class CodeReviewInstructionImpl(BaseELLInstruction):
    """Code review instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        super().__init__(
            name="code_review",
            description="Review code for quality and standards"
        )
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for code review."""
        return {
            "instruction_type": "code_review",
            "review_type": context.get("review_type", "full"),
            "standards": context.get("standards", []),
            "complexity": context.get("complexity", "complex")
        }

class DocumentationGenerationInstructionImpl(BaseELLInstruction):
    """Documentation generation instruction implementation."""
    
    def __init__(self):
        """Initialize instruction."""
        super().__init__(
            name="documentation_generation",
            description="Generate comprehensive documentation"
        )
    
    def _get_metadata(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Get metadata for documentation generation."""
        return {
            "instruction_type": "documentation_generation",
            "doc_type": context.get("doc_type", "general"),
            "format": context.get("format", "markdown"),
            "complexity": context.get("complexity", "complex")
        }

class TaskPlanner:
    """Task planner class."""
    
    def __init__(self):
        """Initialize task planner."""
        self.ell = ell_integration
        self.code_file_instruction = CodeFileInstructionImpl()
        self.git_workflow_instruction = GitWorkflowInstructionImpl()
        self.requirement_analysis_instruction = RequirementAnalysisInstructionImpl()
        self.test_design_instruction = TestDesignInstructionImpl()
        self.code_review_instruction = CodeReviewInstructionImpl()
        self.documentation_instruction = DocumentationGenerationInstructionImpl()
        self.thinking = ProgrammerThinkingInstruction()
        self.default_phases = {
            TaskType.ANALYSIS: self._create_analysis_phases(),
            TaskType.IMPLEMENTATION: self._create_implementation_phases(),
            TaskType.TESTING: self._create_testing_phases(),
            TaskType.DOCUMENTATION: self._create_documentation_phases(),
            TaskType.REFACTORING: self._create_refactoring_phases(),
            TaskType.BUGFIX: self._create_bugfix_phases()
        }
    
    async def plan_task(self, thinking_result: Dict[str, Any]) -> List[BaseInstruction]:
        """Plan task execution based on thinking results.
        
        Args:
            thinking_result: Results from thinking about task
            
        Returns:
            List of instructions to execute
        """
        # Determine task type
        task_type = self._determine_task_type(thinking_result)
        
        # Get phases for task type
        phases = self.default_phases[task_type]
        
        # Customize phases based on thinking results
        customized_phases = await self._customize_phases(phases, thinking_result)
        
        # Convert phases to flat instruction list
        return self._flatten_phases(customized_phases)
    
    def _determine_task_type(self, thinking_result: Dict[str, Any]) -> TaskType:
        """Determine task type from thinking results.
        
        Args:
            thinking_result: Thinking results
            
        Returns:
            Task type
        """
        insights = thinking_result.get("insights", [])
        code_insights = thinking_result.get("code_insights", [])
        
        # Count type indicators
        type_indicators = {
            TaskType.ANALYSIS: 0,
            TaskType.IMPLEMENTATION: 0,
            TaskType.TESTING: 0,
            TaskType.DOCUMENTATION: 0,
            TaskType.REFACTORING: 0,
            TaskType.BUGFIX: 0
        }
        
        # Analyze insights
        for insight in insights:
            content = insight.get("content", "").lower()
            if "analyz" in content or "understand" in content:
                type_indicators[TaskType.ANALYSIS] += 1
            if "implement" in content or "creat" in content:
                type_indicators[TaskType.IMPLEMENTATION] += 1
            if "test" in content or "verif" in content:
                type_indicators[TaskType.TESTING] += 1
            if "document" in content:
                type_indicators[TaskType.DOCUMENTATION] += 1
            if "refactor" in content or "improv" in content:
                type_indicators[TaskType.REFACTORING] += 1
            if "bug" in content or "fix" in content or "error" in content:
                type_indicators[TaskType.BUGFIX] += 1
        
        # Return type with highest indicator count
        return max(type_indicators.items(), key=lambda x: x[1])[0]
    
    def _create_analysis_phases(self) -> List[TaskPhase]:
        """Create phases for analysis task."""
        return [
            TaskPhase(
                name="requirement_analysis",
                instructions=[self.requirement_analysis_instruction]
            ),
            TaskPhase(
                name="code_analysis",
                instructions=[CodeAnalysisInstruction()],
                dependencies=["requirement_analysis"]
            ),
            TaskPhase(
                name="documentation",
                instructions=[self.documentation_instruction],
                dependencies=["code_analysis"]
            )
        ]
    
    def _create_implementation_phases(self) -> List[TaskPhase]:
        """Create phases for implementation task."""
        return [
            TaskPhase(
                name="project_setup",
                instructions=[ProjectInitializationInstruction()]
            ),
            TaskPhase(
                name="test_design",
                instructions=[self.test_design_instruction],
                dependencies=["project_setup"]
            ),
            TaskPhase(
                name="implementation",
                instructions=[
                    self.code_file_instruction,
                    self.git_workflow_instruction
                ],
                dependencies=["test_design"]
            ),
            TaskPhase(
                name="review",
                instructions=[self.code_review_instruction],
                dependencies=["implementation"]
            )
        ]
    
    def _create_testing_phases(self) -> List[TaskPhase]:
        """Create phases for testing task."""
        return [
            TaskPhase(
                name="test_design",
                instructions=[TestDesignInstruction()]
            ),
            TaskPhase(
                name="implementation",
                instructions=[self.code_file_instruction],
                dependencies=["test_design"]
            ),
            TaskPhase(
                name="review",
                instructions=[CodeReviewInstruction()],
                dependencies=["implementation"]
            )
        ]
    
    def _create_documentation_phases(self) -> List[TaskPhase]:
        """Create phases for documentation task."""
        return [
            TaskPhase(
                name="analysis",
                instructions=[CodeAnalysisInstruction()]
            ),
            TaskPhase(
                name="documentation",
                instructions=[self.documentation_instruction],
                dependencies=["analysis"]
            ),
            TaskPhase(
                name="review",
                instructions=[CodeReviewInstruction()],
                dependencies=["documentation"]
            )
        ]
    
    def _create_refactoring_phases(self) -> List[TaskPhase]:
        """Create phases for refactoring task."""
        return [
            TaskPhase(
                name="analysis",
                instructions=[CodeAnalysisInstruction()]
            ),
            TaskPhase(
                name="implementation",
                instructions=[
                    self.code_file_instruction,
                    self.git_workflow_instruction
                ],
                dependencies=["analysis"]
            ),
            TaskPhase(
                name="review",
                instructions=[CodeReviewInstruction()],
                dependencies=["implementation"]
            )
        ]
    
    def _create_bugfix_phases(self) -> List[TaskPhase]:
        """Create phases for bugfix task."""
        return [
            TaskPhase(
                name="analysis",
                instructions=[CodeAnalysisInstruction()]
            ),
            TaskPhase(
                name="implementation",
                instructions=[
                    self.code_file_instruction,
                    self.git_workflow_instruction
                ],
                dependencies=["analysis"]
            ),
            TaskPhase(
                name="testing",
                instructions=[TestDesignInstruction()],
                dependencies=["implementation"]
            ),
            TaskPhase(
                name="review",
                instructions=[CodeReviewInstruction()],
                dependencies=["testing"]
            )
        ]
    
    async def _customize_phases(
        self,
        phases: List[TaskPhase],
        thinking_result: Dict[str, Any]
    ) -> List[TaskPhase]:
        """Customize phases based on thinking results.
        
        Args:
            phases: Default phases
            thinking_result: Thinking results
            
        Returns:
            Customized phases
        """
        message = Message(
            role=MessageRole.USER,
            content=str(thinking_result),
            metadata={
                "operation": "customize_phases",
                "phases": [phase.name for phase in phases]
            }
        )
        
        response = await self.ell.process_message(message)
        response_metadata = response.metadata or {}
        
        updated_phases = []
        for phase in phases:
            if phase.name in response_metadata.get("modified_phases", []):
                phase_mods = response_metadata.get("phase_modifications", {}).get(phase.name, {})
                if phase_mods:
                    phase.instructions = phase_mods.get("instructions", phase.instructions)
                    phase.dependencies = phase_mods.get("dependencies", phase.dependencies)
            updated_phases.append(phase)
            
        return updated_phases
    
    def _flatten_phases(self, phases: List[TaskPhase]) -> List[BaseInstruction]:
        """Convert phases to flat instruction list.
        
        Args:
            phases: Task phases
            
        Returns:
            Flattened instruction list
        """
        instructions = []
        for phase in phases:
            instructions.extend(phase.instructions)
        return instructions

@dataclass
class ELLWorkflow:
    """ELL workflow class."""
    
    name: str
    description: Optional[str] = None
    steps: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize workflow attributes."""
        self.id = f"workflow_{self.name}"
        self.ell = ell_integration
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize workflow."""
        if not self._initialized:
            self.ell.register_workflow(self.id, self)
            self._initialized = True
    
    async def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Run workflow.
        
        Args:
            context: Workflow context
            
        Returns:
            Dict[str, Any]: Workflow results
        """
        await self.initialize()
        
        results = {}
        for step in self.steps:
            message = Message(
                role=MessageRole.USER,
                content=str(context),
                metadata={"step": step}
            )
            
            response = await self.ell.process_message(message)
            results[step["name"]] = {
                "status": "success",
                "result": response.content,
                "metadata": response.metadata
            }
            
        return results 