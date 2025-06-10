"""Centralized instruction registry."""

from typing import Dict, Set, Type, Optional, Union
from .base import BaseInstruction
from .minimal import (
    MinimalTestDesign,
    MinimalCodeReview,
    MinimalCodeFile,
    BaseMinimalInstruction
)
from ..isa.instruction import Instruction

InstructionType = Union[BaseInstruction, Instruction]

class InstructionRegistry:
    """Centralized registry for all instructions."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InstructionRegistry, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the registry."""
        if not self._initialized:
            self._instructions: Dict[str, InstructionType] = {}
            self._minimal_instructions: Dict[str, BaseMinimalInstruction] = {}
            self._instruction_types: Dict[str, Type[InstructionType]] = {}
            self._initialized = True
    
    def register_instruction(self, instruction: InstructionType) -> None:
        """Register an instruction.
        
        Args:
            instruction: The instruction to register
        """
        self._instructions[instruction.name] = instruction
    
    def register_minimal_instruction(self, instruction: BaseMinimalInstruction) -> None:
        """Register a minimal instruction.
        
        Args:
            instruction: The minimal instruction to register
        """
        self._minimal_instructions[instruction.name] = instruction
    
    def register_instruction_type(self, name: str, instruction_type: Type[InstructionType]) -> None:
        """Register an instruction type.
        
        Args:
            name: Name of the instruction type
            instruction_type: The instruction type class
        """
        self._instruction_types[name] = instruction_type
    
    def get_instruction(self, name: str) -> Optional[InstructionType]:
        """Get an instruction by name.
        
        Args:
            name: Name of the instruction
            
        Returns:
            The instruction if found, None otherwise
        """
        return self._instructions.get(name)
    
    def get_minimal_instruction(self, name: str) -> Optional[BaseMinimalInstruction]:
        """Get a minimal instruction by name.
        
        Args:
            name: Name of the minimal instruction
            
        Returns:
            The minimal instruction if found, None otherwise
        """
        return self._minimal_instructions.get(name)
    
    def get_instruction_type(self, name: str) -> Optional[Type[InstructionType]]:
        """Get an instruction type by name.
        
        Args:
            name: Name of the instruction type
            
        Returns:
            The instruction type if found, None otherwise
        """
        return self._instruction_types.get(name)
    
    def get_all_instructions(self) -> Set[InstructionType]:
        """Get all registered instructions.
        
        Returns:
            Set of all registered instructions
        """
        return set(self._instructions.values())
    
    def get_all_minimal_instructions(self) -> Set[BaseMinimalInstruction]:
        """Get all registered minimal instructions.
        
        Returns:
            Set of all registered minimal instructions
        """
        return set(self._minimal_instructions.values())
    
    def clear(self) -> None:
        """Clear all registered instructions."""
        self._instructions.clear()
        self._minimal_instructions.clear()
        self._instruction_types.clear()

# Global registry instance
registry = InstructionRegistry() 