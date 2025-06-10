"""Instructions module."""

from enum import Enum
from typing import List, Dict, Any, Optional, Type, cast, Union
from dataclasses import dataclass
import importlib
import inspect
import pkgutil
import sys
from pathlib import Path

from .base import (
    BaseInstruction,
    InstructionResult,
    ExecutionContext,
    InstructionType,
    InstructionMetadata
)
from .isa import AgentISAInstruction
from .advanced import (
    AdvancedInstruction,
    ControlFlowInstruction,
    StateManagerInstruction,
    LLMInteractionInstruction,
    ResourceManagerInstruction,
    CompositeInstruction,
    ConditionalInstruction,
    ParallelInstruction,
    IterativeInstruction,
    AdaptiveInstruction,
    DataProcessingInstruction
)
from .thinking import (
    ThinkingInstruction,
    ThinkingStyle,
    CognitivePhase,
    ThoughtContext,
    ThoughtMetrics
)
from .programmer import (
    ProgrammerThinkingInstruction,
    CodeFileInstruction,
    GitWorkflowInstruction,
    ContextManagerInstruction
)
from .analysis import (
    RequirementAnalysisInstruction,
    CodeAnalysisInstruction,
    DependencyAnalysisInstruction
)
from .project import (
    ProjectInitializationInstruction,
    ProjectConfigurationInstruction,
    DependencyManagementInstruction
)
from .development import (
    TestDesignInstruction,
    CodeReviewInstruction,
    CodeRefactoringInstruction
)
from .documentation import (
    DocumentationGenerationInstruction,
    APIDocumentationInstruction,
    ReadmeGenerationInstruction
)
from .registry import registry
from .minimal import BaseMinimalInstruction, MinimalExecutionContext
from ..isa.instruction import Instruction
from ..isa.result import InstructionStatus

# Define instruction categories
class InstructionCategory(str, Enum):
    """Categories of instructions."""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    TEST_DESIGN = "test_design"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    ANALYSIS = "analysis"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    COMPOSITE = "composite"
    CONTROL = "control"
    DATA = "data"
    RESOURCE = "resource"

@dataclass
class InstructionSpec:
    """Specification for an instruction."""
    category: InstructionCategory
    name: str
    description: str
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None

class Pipeline:
    """Instruction pipeline."""
    
    def __init__(self, instructions: List[BaseInstruction]):
        """Initialize pipeline.
        
        Args:
            instructions: List of instructions to execute.
        """
        self.instructions = instructions
        self._instruction_map = {
            instruction.name: instruction
            for instruction in instructions
        }
        
    async def add_instruction(self, instruction: BaseInstruction) -> None:
        """Add an instruction to the pipeline.
        
        Args:
            instruction: Instruction to add.
        """
        self.instructions.append(instruction)
        self._instruction_map[instruction.name] = instruction
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pipeline.
        
        Args:
            context: Execution context.
            
        Returns:
            Pipeline results.
        """
        results = {}
        for instruction in self.instructions:
            # Create proper ExecutionContext
            exec_context = ExecutionContext(
                params=context,
                state={},
                metadata={}
            )
            results[instruction.name] = await instruction.execute(exec_context)
        return results
        
    async def execute_instruction(
        self,
        instruction_name: str,
        params: Dict[str, Any]
    ) -> InstructionResult:
        """Execute a specific instruction.
        
        Args:
            instruction_name: Name of instruction to execute.
            params: Instruction parameters.
            
        Returns:
            Instruction result.
            
        Raises:
            ValueError: If instruction not found.
        """
        if instruction_name not in self._instruction_map:
            raise ValueError(f"Instruction not found: {instruction_name}")
            
        instruction = self._instruction_map[instruction_name]
        # Create proper ExecutionContext
        exec_context = ExecutionContext(
            params=params,
            state={},
            metadata={}
        )
        return await instruction.execute(exec_context)
        
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        for instruction in self.instructions:
            await instruction.cleanup()

# Define instruction class types
InstructionClass = Union[Type[BaseInstruction], Type[BaseMinimalInstruction]]
RegisteredInstruction = Union[BaseInstruction, Instruction]

def _register_module_instructions(module_name: str) -> None:
    """Register all instructions from a module.
    
    Args:
        module_name: Full module name to register instructions from
    """
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                if issubclass(obj, BaseMinimalInstruction) and obj != BaseMinimalInstruction:
                    try:
                        # Create minimal instruction instance with proper type
                        instance = obj(
                            name=name,
                            description=f"Auto-registered minimal instruction: {name}"
                        )
                        registry.register_minimal_instruction(cast(BaseMinimalInstruction, instance))
                        registry.register_instruction(cast(RegisteredInstruction, instance))
                    except Exception as e:
                        print(f"Failed to register minimal instruction {name}: {str(e)}")
                        
                elif issubclass(obj, BaseInstruction) and obj != BaseInstruction:
                    try:
                        # Determine instruction type from class name
                        instruction_type = _get_instruction_type(obj.__name__)
                        
                        # Create regular instruction instance
                        instance = obj(
                            name=name,
                            description=f"Auto-registered instruction: {name}"
                        )
                        registry.register_instruction(cast(RegisteredInstruction, instance))
                    except Exception as e:
                        print(f"Failed to register instruction {name}: {str(e)}")
                        
    except Exception as e:
        print(f"Failed to process module {module_name}: {str(e)}")

def _get_instruction_type(class_name: str) -> InstructionType:
    """Determine instruction type from class name."""
    if "Review" in class_name:
        return InstructionType.CODE_REVIEW
    elif "Refactor" in class_name:
        return InstructionType.REFACTORING
    elif "Generate" in class_name or "Creation" in class_name:
        return InstructionType.CODE_GENERATION
    elif "Test" in class_name:
        return InstructionType.TEST_DESIGN
    elif "Control" in class_name or "Flow" in class_name:
        return InstructionType.CONTROL_FLOW
    elif "State" in class_name:
        return InstructionType.STATE_MANAGEMENT
    elif "LLM" in class_name:
        return InstructionType.LLM_INTERACTION
    elif "Resource" in class_name:
        return InstructionType.RESOURCE_MANAGEMENT
    elif "Collaboration" in class_name:
        return InstructionType.COLLABORATION
    elif "Data" in class_name:
        return InstructionType.DATA_PROCESSING
    else:
        return InstructionType.OBJECTIVE_HANDLING

def register_all_instructions() -> None:
    """Register all instructions from the instructions directory."""
    package_path = Path(__file__).parent
    package_name = __package__

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if not module_info.name.startswith('_'):
            module_name = f"{package_name}.{module_info.name}"
            _register_module_instructions(module_name)

# Register all instructions when the package is imported
register_all_instructions()

# Export commonly used types and classes
__all__ = [
    'BaseInstruction',
    'BaseMinimalInstruction',
    'Instruction',
    'ExecutionContext',
    'MinimalExecutionContext',
    'InstructionStatus',
    'InstructionResult',
    'InstructionType',
    'InstructionMetadata',
    'registry',
    'register_all_instructions'
] 