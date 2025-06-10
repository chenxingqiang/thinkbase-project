"""ISA instruction base class."""

from typing import Dict, Any, Optional
from .base import BaseInstruction, InstructionResult, InstructionStatus

class AgentISAInstruction(BaseInstruction):
    """Base class for Agent ISA instructions."""
    
    def __init__(self, name: str, description: str):
        super().__init__(name, description)
        self.instruction_type = "agent_isa"
        self.instruction_version = "1.0.0"
        self.instruction_capabilities = set()
        
    async def _execute_impl(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the instruction implementation.
        
        Args:
            context: The execution context.
            
        Returns:
            Optional dictionary with execution results.
        """
        raise NotImplementedError("Subclasses must implement _execute_impl")
        
    async def validate(self, context: Dict[str, Any]) -> bool:
        """Validate instruction context.
        
        Args:
            context: The execution context to validate.
            
        Returns:
            True if context is valid, False otherwise.
        """
        # Basic validation - subclasses should override for specific validation
        return isinstance(context, dict)
        
    def add_capability(self, capability: str) -> None:
        """Add a capability to this instruction.
        
        Args:
            capability: The capability to add.
        """
        self.instruction_capabilities.add(capability)
        
    def has_capability(self, capability: str) -> bool:
        """Check if instruction has a specific capability.
        
        Args:
            capability: The capability to check.
            
        Returns:
            True if instruction has capability, False otherwise.
        """
        return capability in self.instruction_capabilities
        
    async def execute_with_retry(self, context: Dict[str, Any], max_retries: int = 3) -> InstructionResult:
        """Execute instruction with retry logic.
        
        Args:
            context: The execution context.
            max_retries: Maximum number of retry attempts.
            
        Returns:
            InstructionResult with execution status and data.
        """
        attempts = 0
        last_error = None
        
        while attempts < max_retries:
            try:
                result = await self.execute(context)
                if result.status == InstructionStatus.COMPLETED:
                    return result
                last_error = result.error
            except Exception as e:
                last_error = str(e)
            attempts += 1
            
        return InstructionResult(
            status=InstructionStatus.FAILED,
            error=f"Failed after {max_retries} attempts. Last error: {last_error}"
        ) 