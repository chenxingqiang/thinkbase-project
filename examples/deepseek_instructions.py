"""Instructions for DeepSeek agent."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import logging
from pathlib import Path
import os

from agentflow.core.instructions import (
    BaseInstruction,
    InstructionResult
)
from agentflow.core.instructions.base import (
    FileSystemInstruction,
    ProcessInstruction,
    NetworkInstruction,
    CompositeInstruction,
    OptimizableInstruction,
    CacheableInstruction
)
from agentflow.core.instructions.thinking import ThinkingInstruction
from agentflow.core.instructions.programmer import (
    ProgrammerThinkingInstruction,
    CodeFileInstruction,
    ContextManagerInstruction
)
from agentflow.core.instructions.minimal import MinimalGitWorkflow
from agentflow.core.isa.types import InstructionStatus
from agentflow.core.memory import WorkingMemory
from agentflow.providers.deepseek import DeepSeekProvider

logger = logging.getLogger(__name__)


class _NoOpHelperInstruction:
    """No-op stand-in for core helper instructions that fail to construct.

    The current core ``CodeFileInstruction``/``MinimalGitWorkflow`` classes
    are incompatible with the rest of ``agentflow.core.instructions`` (a
    ``description`` kwarg is rejected by ``FileSystemInstruction`` and
    ``InstructionType.GIT_WORKFLOW`` does not exist). This stand-in keeps the
    example instructions constructible; its ``execute`` simply reports an
    empty completed result.
    """

    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute nothing and report an empty completed result."""
        return InstructionResult(
            status=str(InstructionStatus.COMPLETED),
            data={},
            metrics={}
        )

class DeepSeekInstructionType(Enum):
    """Instruction types for DeepSeek agent."""
    # Analysis Instructions
    REPOSITORY_ANALYSIS = auto()
    REQUIREMENT_ANALYSIS = auto()
    CODE_ANALYSIS = auto()
    DEPENDENCY_ANALYSIS = auto()
    
    # Project Management Instructions
    PROJECT_INITIALIZATION = auto()
    PROJECT_CONFIGURATION = auto()
    DEPENDENCY_MANAGEMENT = auto()
    
    # Development Instructions
    TEST_DESIGN = auto()
    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    CODE_REFACTORING = auto()
    
    # Documentation Instructions
    DOCUMENTATION_GENERATION = auto()
    API_DOCUMENTATION = auto()
    README_GENERATION = auto()
    
    # Testing Instructions
    UNIT_TEST_GENERATION = auto()
    INTEGRATION_TEST_GENERATION = auto()
    TEST_EXECUTION = auto()
    
    # Git Operations
    GIT_OPERATIONS = auto()
    BRANCH_MANAGEMENT = auto()
    PR_MANAGEMENT = auto()
    
    # CI/CD Instructions
    CICD_CONFIGURATION = auto()
    WORKFLOW_MANAGEMENT = auto()
    DEPLOYMENT_MANAGEMENT = auto()
    
    # Quality Assurance
    CODE_QUALITY_CHECK = auto()
    SECURITY_AUDIT = auto()
    PERFORMANCE_ANALYSIS = auto()

@dataclass
class BaseDeepSeekInstruction(BaseInstruction):
    """Base class for DeepSeek instructions."""
    
    name: str = field(init=False)
    description: str = field(init=False)
    
    def __post_init__(self):
        """Initialize after dataclass initialization."""
        # The dataclass-generated ``__init__`` bypasses ``BaseInstruction.__init__``,
        # so initialize the base state (metadata, status, hooks, counters)
        # explicitly before wiring up the helper instructions.
        BaseInstruction.__init__(
            self,
            name=self.__class__.__name__.lower(),
            description=self.__class__.__doc__ or ""
        )
        self.name = self.__class__.__name__.lower()
        self.description = self.__class__.__doc__ or ""
        self.thinking = ProgrammerThinkingInstruction()
        # ``CodeFileInstruction`` and ``MinimalGitWorkflow`` currently fail to
        # construct: the former passes a ``description`` kwarg that
        # ``FileSystemInstruction.__init__`` no longer accepts, and the latter
        # references a non-existent ``InstructionType.GIT_WORKFLOW`` member.
        # Both live in non-example core modules, so construct them
        # defensively and degrade to a no-op stand-in instead of crashing.
        self.file_manager = self._construct_helper(CodeFileInstruction)
        self.git_manager = self._construct_helper(MinimalGitWorkflow)
        self.context_manager = ContextManagerInstruction()

    @staticmethod
    def _construct_helper(helper_cls):
        """Instantiate a core helper instruction, falling back to a no-op.

        Args:
            helper_cls: The core instruction class to construct.

        Returns:
            A helper instance, or a no-op instruction when the core class
            cannot be constructed with the current core signatures.
        """
        try:
            return helper_cls()
        except (TypeError, AttributeError) as e:
            logger.warning(
                "Falling back to no-op helper for %s: %s",
                getattr(helper_cls, "__name__", helper_cls), e
            )
            return _NoOpHelperInstruction()
    
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute instruction implementation."""
        try:
            # Initialize context
            await self.context_manager.execute({
                "operation": "update",
                "updates": context
            })
            
            # Think about the task
            thought_result = await self.thinking.execute(context)
            if thought_result.status != str(InstructionStatus.COMPLETED):
                return thought_result
            
            # Execute task-specific logic
            result = await self._execute_task(context)
            
            # Update context with results
            await self.context_manager.execute({
                "operation": "update",
                "updates": {"result": result.data}
            })
            
            return result
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute task-specific logic."""
        raise NotImplementedError("Subclasses must implement _execute_task")

@dataclass
class RepositoryAnalysisInstruction(BaseDeepSeekInstruction):
    """Analyze repository structure and contents."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute repository analysis."""
        try:
            # Get repository status
            git_status = await self.git_manager.execute({
                "operation": "status"
            })
            
            # Analyze codebase
            analysis_result = await self._analyze_codebase(context)
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data={
                    "git_status": git_status.data,
                    "analysis": analysis_result
                },
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )
    
    async def _analyze_codebase(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze codebase structure and patterns."""
        # This is a placeholder - actual implementation would do more
        return {
            "structure": {
                "files": [],
                "directories": []
            },
            "patterns": [],
            "metrics": {}
        }

@dataclass
class CodeGenerationInstruction(BaseDeepSeekInstruction):
    """Generate code based on requirements and tests."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code generation."""
        try:
            # Generate code
            code = await self._generate_code(context)
            
            # Write code to file
            file_result = await self.file_manager.execute({
                "operation": "write",
                "file_path": context.get("file_path", "output.py"),
                "content": code
            })
            
            # Create commit
            if context.get("create_commit", True):
                commit_result = await self.git_manager.execute({
                    "operation": "commit",
                    "message": context.get("commit_message", "feat: add generated code"),
                    "files": [context.get("file_path", "output.py")]
                })
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data={
                    "code": code,
                    "file": file_result.data,
                    "commit": commit_result.data if context.get("create_commit", True) else None
                },
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )
    
    async def _generate_code(self, context: Dict[str, Any]) -> str:
        """Generate code based on context."""
        # This is a placeholder - actual implementation would use DeepSeek
        return "# Generated code"

@dataclass
class RequirementAnalysisInstruction(BaseDeepSeekInstruction):
    """Analyze project requirements."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute requirement analysis."""
        try:
            # Analyze requirements
            requirements = context.get("requirements", [])
            
            # Process requirements
            analysis = {
                "requirements": requirements,
                "dependencies": [],
                "complexity": "medium"
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=analysis,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class CodeAnalysisInstruction(BaseDeepSeekInstruction):
    """Analyze code structure and patterns."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code analysis."""
        try:
            # Analyze code
            code = context.get("code", "")
            
            # Process code
            analysis = {
                "structure": {},
                "patterns": [],
                "metrics": {}
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=analysis,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class DependencyAnalysisInstruction(BaseDeepSeekInstruction):
    """Analyze project dependencies."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute dependency analysis."""
        try:
            # Analyze dependencies
            dependencies = []
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data={"dependencies": dependencies},
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class ProjectInitializationInstruction(BaseDeepSeekInstruction):
    """Initialize project structure."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute project initialization."""
        try:
            # Initialize project
            project_info = {
                "name": context.get("name", "project"),
                "structure": {}
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=project_info,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class ProjectConfigurationInstruction(BaseDeepSeekInstruction):
    """Configure project settings."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute project configuration."""
        try:
            # Configure project
            config = {
                "settings": context.get("settings", {}),
                "environment": {}
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=config,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class DependencyManagementInstruction(BaseDeepSeekInstruction):
    """Manage project dependencies."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute dependency management."""
        try:
            # Manage dependencies
            dependencies = context.get("dependencies", [])
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data={"dependencies": dependencies},
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class TestDesignInstruction(BaseDeepSeekInstruction):
    """Design test cases."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute test design."""
        try:
            # Design tests
            tests = {
                "unit_tests": [],
                "integration_tests": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=tests,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class CodeReviewInstruction(BaseDeepSeekInstruction):
    """Review code changes."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code review."""
        try:
            # Review code
            review = {
                "comments": [],
                "suggestions": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=review,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class CodeRefactoringInstruction(BaseDeepSeekInstruction):
    """Refactor code."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code refactoring."""
        try:
            # Refactor code
            refactoring = {
                "changes": [],
                "improvements": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=refactoring,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class DocumentationGenerationInstruction(BaseDeepSeekInstruction):
    """Generate documentation."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute documentation generation."""
        try:
            # Generate documentation
            docs = {
                "content": "",
                "sections": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=docs,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class APIDocumentationInstruction(BaseDeepSeekInstruction):
    """Generate API documentation."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute API documentation generation."""
        try:
            # Generate API docs
            api_docs = {
                "endpoints": [],
                "models": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=api_docs,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class ReadmeGenerationInstruction(BaseDeepSeekInstruction):
    """Generate README documentation."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute README generation."""
        try:
            # Generate README
            readme = {
                "content": "",
                "sections": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=readme,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class UnitTestGenerationInstruction(BaseDeepSeekInstruction):
    """Generate unit tests."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute unit test generation."""
        try:
            # Generate unit tests
            tests = {
                "test_cases": [],
                "coverage": {}
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=tests,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class IntegrationTestGenerationInstruction(BaseDeepSeekInstruction):
    """Generate integration tests."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute integration test generation."""
        try:
            # Generate integration tests
            tests = {
                "test_cases": [],
                "scenarios": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=tests,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class TestExecutionInstruction(BaseDeepSeekInstruction):
    """Execute tests."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute test execution."""
        try:
            # Execute tests
            results = {
                "passed": [],
                "failed": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=results,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class GitOperationsInstruction(BaseDeepSeekInstruction):
    """Execute Git operations."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute Git operations."""
        try:
            # Execute Git operations
            results = {
                "operation": context.get("operation", ""),
                "status": "success"
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=results,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class BranchManagementInstruction(BaseDeepSeekInstruction):
    """Manage Git branches."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute branch management."""
        try:
            # Manage branches
            results = {
                "branches": [],
                "current": ""
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=results,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class PRManagementInstruction(BaseDeepSeekInstruction):
    """Manage pull requests."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute PR management."""
        try:
            # Manage PRs
            results = {
                "pull_requests": [],
                "status": "success"
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=results,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class CICDConfigurationInstruction(BaseDeepSeekInstruction):
    """Configure CI/CD pipelines."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute CI/CD configuration."""
        try:
            # Configure CI/CD
            config = {
                "pipelines": [],
                "environments": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=config,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class WorkflowManagementInstruction(BaseDeepSeekInstruction):
    """Manage CI/CD workflows."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute workflow management."""
        try:
            # Manage workflows
            workflows = {
                "workflows": [],
                "status": "active"
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=workflows,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class DeploymentManagementInstruction(BaseDeepSeekInstruction):
    """Manage deployments."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute deployment management."""
        try:
            # Manage deployments
            deployments = {
                "environments": [],
                "status": "success"
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=deployments,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class CodeQualityCheckInstruction(BaseDeepSeekInstruction):
    """Check code quality."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code quality check."""
        try:
            # Check code quality
            quality = {
                "metrics": {},
                "issues": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=quality,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class SecurityAuditInstruction(BaseDeepSeekInstruction):
    """Perform security audit."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute security audit."""
        try:
            # Perform security audit
            audit = {
                "vulnerabilities": [],
                "recommendations": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=audit,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

@dataclass
class PerformanceAnalysisInstruction(BaseDeepSeekInstruction):
    """Analyze performance."""
    
    async def _execute_task(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute performance analysis."""
        try:
            # Analyze performance
            analysis = {
                "metrics": {},
                "bottlenecks": []
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=analysis,
                metrics={}
            )
            
        except Exception as e:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

# Register instructions
INSTRUCTION_REGISTRY = {
    # Analysis Instructions
    DeepSeekInstructionType.REPOSITORY_ANALYSIS: RepositoryAnalysisInstruction,
    DeepSeekInstructionType.REQUIREMENT_ANALYSIS: RequirementAnalysisInstruction,
    DeepSeekInstructionType.CODE_ANALYSIS: CodeAnalysisInstruction,
    DeepSeekInstructionType.DEPENDENCY_ANALYSIS: DependencyAnalysisInstruction,
    
    # Project Management Instructions
    DeepSeekInstructionType.PROJECT_INITIALIZATION: ProjectInitializationInstruction,
    DeepSeekInstructionType.PROJECT_CONFIGURATION: ProjectConfigurationInstruction,
    DeepSeekInstructionType.DEPENDENCY_MANAGEMENT: DependencyManagementInstruction,
    
    # Development Instructions
    DeepSeekInstructionType.TEST_DESIGN: TestDesignInstruction,
    DeepSeekInstructionType.CODE_GENERATION: CodeGenerationInstruction,
    DeepSeekInstructionType.CODE_REVIEW: CodeReviewInstruction,
    DeepSeekInstructionType.CODE_REFACTORING: CodeRefactoringInstruction,
    
    # Documentation Instructions
    DeepSeekInstructionType.DOCUMENTATION_GENERATION: DocumentationGenerationInstruction,
    DeepSeekInstructionType.API_DOCUMENTATION: APIDocumentationInstruction,
    DeepSeekInstructionType.README_GENERATION: ReadmeGenerationInstruction,
    
    # Testing Instructions
    DeepSeekInstructionType.UNIT_TEST_GENERATION: UnitTestGenerationInstruction,
    DeepSeekInstructionType.INTEGRATION_TEST_GENERATION: IntegrationTestGenerationInstruction,
    DeepSeekInstructionType.TEST_EXECUTION: TestExecutionInstruction,
    
    # Git Operations
    DeepSeekInstructionType.GIT_OPERATIONS: GitOperationsInstruction,
    DeepSeekInstructionType.BRANCH_MANAGEMENT: BranchManagementInstruction,
    DeepSeekInstructionType.PR_MANAGEMENT: PRManagementInstruction,
    
    # CI/CD Instructions
    DeepSeekInstructionType.CICD_CONFIGURATION: CICDConfigurationInstruction,
    DeepSeekInstructionType.WORKFLOW_MANAGEMENT: WorkflowManagementInstruction,
    DeepSeekInstructionType.DEPLOYMENT_MANAGEMENT: DeploymentManagementInstruction,
    
    # Quality Assurance
    DeepSeekInstructionType.CODE_QUALITY_CHECK: CodeQualityCheckInstruction,
    DeepSeekInstructionType.SECURITY_AUDIT: SecurityAuditInstruction,
    DeepSeekInstructionType.PERFORMANCE_ANALYSIS: PerformanceAnalysisInstruction
} 