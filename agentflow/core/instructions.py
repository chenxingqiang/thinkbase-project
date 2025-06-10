"""Core instruction definitions for AgentFlow."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import time
from typing import Dict, Any, List, Optional

class InstructionType(Enum):
    """Types of instructions supported by the agent."""
    REQUIREMENT_ANALYSIS = "requirement_analysis"
    TEST_DESIGN = "test_design"
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    DOCUMENTATION = "documentation"
    REFACTORING = "refactoring"

@dataclass
class InstructionSpec:
    """Specification for an instruction."""
    type: InstructionType
    name: str
    description: str
    parameters: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class InstructionResult:
    """Result of instruction execution."""
    status: str
    data: Any
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class AgentISAInstruction(ABC):
    """Abstract base class for all instructions."""
    
    def __init__(self, spec: InstructionSpec):
        self.spec = spec
        self.metrics: Dict[str, Any] = {}
    
    @abstractmethod
    async def _execute_impl(self, params: Dict[str, Any]) -> InstructionResult:
        """Implementation of instruction execution."""
        pass
    
    async def execute(self, params: Dict[str, Any]) -> InstructionResult:
        """Execute the instruction with given parameters."""
        try:
            start_time = time.time()
            result = await self._execute_impl(params)
            execution_time = time.time() - start_time
            
            # Update metrics
            self.metrics.update({
                "execution_time": execution_time,
                "status": result.status
            })
            
            if result.metrics:
                self.metrics.update(result.metrics)
            
            return result
            
        except Exception as e:
            return InstructionResult(
                status="error",
                data=None,
                error=str(e)
            )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get instruction execution metrics."""
        return self.metrics.copy()

class Pipeline:
    """Instruction execution pipeline."""
    
    def __init__(self):
        self.instructions: Dict[str, AgentISAInstruction] = {}
    
    async def add_instruction(self, instruction: AgentISAInstruction) -> None:
        """Add an instruction to the pipeline."""
        self.instructions[instruction.spec.name] = instruction
    
    async def execute_instruction(
        self,
        name: str,
        params: Dict[str, Any]
    ) -> InstructionResult:
        """Execute a named instruction with parameters."""
        instruction = self.instructions.get(name)
        if instruction is None:
            return InstructionResult(
                status="error",
                data=None,
                error=f"Instruction not found: {name}"
            )
            
        return await instruction.execute(params)
    
    async def cleanup(self) -> None:
        """Clean up pipeline resources."""
        self.instructions.clear() 