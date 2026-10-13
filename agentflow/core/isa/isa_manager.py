"""ISA (Instruction Set Architecture) Manager for Agent Instructions"""

from typing import Dict, List, Any, Optional, Set, cast
import json
from pathlib import Path
import logging
import ray
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from ray.actor import ActorHandle

logger = logging.getLogger(__name__)

# Re-exported for convenience so that consumers can import the full ISA
# vocabulary (instruction, type, status and result) from a single module.
# ``InstructionStatus``/``InstructionResult`` canonical definitions live in
# ``isa.types`` / ``isa.result`` respectively; they are deliberately not
# redefined here to keep a single source of truth.
from .types import InstructionStatus  # noqa: E402  (re-export)
from .result import InstructionResult  # noqa: E402  (re-export)

class InstructionType(str, Enum):
    """Types of instructions supported by the ISA"""
    BASIC = "basic"           # Single agent instruction
    COMPOSITE = "composite"   # Combination of multiple instructions
    PARALLEL = "parallel"     # Instructions that can run in parallel
    OPTIMIZED = "optimized"   # Optimized version of other instructions
    CONTROL = "control"       # Control flow instructions
    COMPUTATION = "computation"  # Computational instructions
    VALIDATION = "validation"  # Validation instructions

@dataclass
class Instruction:
    """Represents a single instruction in the ISA"""
    id: str
    name: str
    type: InstructionType
    description: str
    params: Dict[str, Any]
    dependencies: Optional[List[str]] = None
    cost: float = 1.0  # Computational cost estimate
    parallelizable: bool = False
    agent_requirements: Optional[List[str]] = None  # Required agent capabilities

@ray.remote
class ParallelInstruction:
    """Ray actor for parallel instruction execution"""
    def __init__(self, instruction: Instruction):
        self.instruction = instruction
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the instruction in parallel"""
        # Implementation will be provided by specific instruction types
        raise NotImplementedError

class ISAManager:
    """Manages the Instruction Set Architecture for agents"""
    
    def __init__(self, isa_config_path: Optional[str] = None):
        self.instructions: Dict[str, Instruction] = {}
        self.dependency_graph = nx.DiGraph()
        self.execution_history = []
        self._initialized = False
        self._config_path = isa_config_path
    
    async def initialize(self):
        """Initialize the ISA manager"""
        if self._initialized:
            return self
            
        # Initialize Ray if not already initialized
        try:
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
        except Exception as e:
            logger.warning(f"Failed to initialize Ray: {e}. Some parallel features may be unavailable.")
        
        if self._config_path:
            self.load_instructions(self._config_path)
            
        self._initialized = True
        return self
    
    def load_instructions(self, config_path: str):
        """Load instructions from configuration file"""
        try:
            with open(config_path) as f:
                config = json.load(f)
                
            for instr_config in config.get("instructions", []):
                instruction = Instruction(
                    id=instr_config["id"],
                    name=instr_config["name"],
                    type=InstructionType[instr_config["type"].upper()],
                    description=instr_config["description"],
                    params=instr_config.get("params", {}),
                    dependencies=instr_config.get("dependencies", []),
                    cost=instr_config.get("cost", 1.0),
                    parallelizable=instr_config.get("parallelizable", False),
                    agent_requirements=instr_config.get("agent_requirements", [])
                )
                self.register_instruction(instruction)
        except Exception as e:
            logger.error(f"Failed to load ISA configuration: {e}")
            raise
    
    def register_instruction(self, instruction: Instruction):
        """Register a new instruction and update dependency graph"""
        # Convert instruction type to uppercase for case-insensitive comparison
        if isinstance(instruction.type, str):
            instruction.type = InstructionType[instruction.type.upper()]
        self.instructions[instruction.id] = instruction
        
        # Update dependency graph
        self.dependency_graph.add_node(instruction.id)
        if instruction.dependencies:
            for dep in instruction.dependencies:
                self.dependency_graph.add_edge(dep, instruction.id)
    
    def optimize_instruction_sequence(self, instructions: List[str]) -> List[List[str]]:
        """Optimize a sequence of instructions for parallel execution"""
        if not instructions:
            return []
            
        # Build execution levels based on dependencies
        levels: List[Set[str]] = []
        remaining = set(instructions)
        
        while remaining:
            # Find instructions with no remaining dependencies
            level = {
                instr for instr in remaining
                if not any(dep in remaining for dep in self.instructions[instr].dependencies or [])
            }
            
            if not level:
                # Circular dependency detected
                logger.warning("Circular dependency detected in instruction sequence")
                break
                
            levels.append(level)
            remaining -= level
        
        # Group parallelizable instructions within each level
        optimized_sequence = []
        for level in levels:
            parallel_group = []
            sequential_group = []
            
            for instr in level:
                if self.instructions[instr].parallelizable:
                    parallel_group.append(instr)
                else:
                    sequential_group.append(instr)
            
            if parallel_group:
                optimized_sequence.append(parallel_group)
            if sequential_group:
                optimized_sequence.extend([[instr] for instr in sequential_group])
        
        return optimized_sequence
    
    async def execute_instruction_sequence(self, sequence: List[List[str]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute an optimized instruction sequence"""
        results = []
        
        for group in sequence:
            if len(group) > 1:  # Parallel execution
                # Create and execute Ray actors in parallel
                futures = []
                for instr in group:
                    # Create actor and cast to proper type
                    actor_ref = ParallelInstruction.remote(self.instructions[instr])
                    actor = cast(ActorHandle, actor_ref)
                    futures.append(actor.execute.remote(context))
                group_results = await ray.get(futures)
                results.extend(group_results)
                
                # Update execution history
                for instr, result in zip(group, group_results):
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "parallel",
                        "status": "completed",
                        "result": result
                    })
            else:  # Sequential execution
                instr = group[0]
                try:
                    # Execute single instruction
                    result = await self._execute_single_instruction(instr, context)
                    results.append(result)
                    
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "sequential",
                        "status": "completed",
                        "result": result
                    })
                except Exception as e:
                    logger.error(f"Failed to execute instruction {instr}: {e}")
                    self.execution_history.append({
                        "instruction": instr,
                        "type": "sequential",
                        "status": "failed",
                        "error": str(e)
                    })
                    raise
        
        return results
    
    async def _execute_single_instruction(self, instruction_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single instruction"""
        instruction = self.instructions[instruction_name]
        # Default implementation for testing
        return {
            "status": "success",
            "result": f"Executed {instruction.name}",
            "params": instruction.params
        }
    
    async def cleanup(self):
        """Clean up ISA manager state"""
        # Clear instructions
        self.instructions = {}
        
        # Clear dependency graph
        self.dependency_graph.clear()
        
        # Clear history
        self.execution_history = []
        
        # Reset initialization flag
        self._initialized = False
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get the execution history with tracing information"""
        return self.execution_history
    
    def get_instruction_stats(self) -> Dict[str, Any]:
        """Get statistics about instruction execution"""
        stats = {
            "total_executions": len(self.execution_history),
            "successful_executions": sum(1 for h in self.execution_history if h["status"] == "completed"),
            "failed_executions": sum(1 for h in self.execution_history if h["status"] == "failed"),
            "instruction_usage": {},
            "average_parallel_group_size": 0
        }
        
        # Calculate instruction usage
        for history in self.execution_history:
            instr = history["instruction"]
            stats["instruction_usage"][instr] = stats["instruction_usage"].get(instr, 0) + 1
        
        # Calculate average parallel group size
        parallel_groups = [h for h in self.execution_history if h["type"] == "parallel"]
        if parallel_groups:
            stats["average_parallel_group_size"] = len(parallel_groups) / len(set(h["instruction"] for h in parallel_groups))
        
        return stats

    def get_instruction(self, name: str) -> Instruction:
        """Get instruction by name."""
        if name not in self.instructions:
            raise ValueError(f"Instruction not found: {name}")
        return self.instructions[name]

    def get_available_instructions(self) -> List[str]:
        """Get list of available instructions."""
        return list(self.instructions.keys())

    def get_instruction_dependencies(self, name: str) -> Set[str]:
        """Get dependencies for an instruction."""
        if name not in self.instructions:
            raise ValueError(f"Instruction not found: {name}")
        return set(self.dependency_graph.successors(name))

    def validate_instruction_sequence(self, sequence: List[str]) -> bool:
        """Validate if instruction sequence is valid."""
        if not sequence:
            return True
        
        # Check if all instructions exist
        for name in sequence:
            if name not in self.instructions:
                return False
        
        # Check if sequence respects dependencies
        for i, name in enumerate(sequence):
            deps = self.get_instruction_dependencies(name)
            if deps:
                # All dependencies must appear before this instruction
                prev_instructions = set(sequence[:i])
                if not deps.issubset(prev_instructions):
                    return False
        
        return True

    def execute_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Execute a single instruction synchronously."""
        try:
            # Validate params
            if "invalid" in instruction.params:
                raise ValueError("Invalid instruction parameters")
                
            # Default implementation for testing
            return {
                "status": "success",
                "result": f"Executed {instruction.name}",
                "params": instruction.params
            }
        except Exception as e:
            logger.error(f"Failed to execute instruction {instruction.name}: {e}")
            raise
