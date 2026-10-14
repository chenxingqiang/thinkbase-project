"""ISA (Instruction Set Architecture) Manager for Agent Instructions"""

from typing import Dict, List, Any, Optional, Set, cast
import json
import time
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
    STATE = "state"           # State management instructions
    LLM = "llm"               # LLM interaction instructions
    RESOURCE = "resource"     # Resource management instructions

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
        self._execution_count = 0
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

    async def execute_instruction(self, instruction, context: Optional[Dict[str, Any]] = None) -> InstructionResult:
        """Execute a single instruction and record metrics.

        ``instruction`` may be either a registered instruction id or an
        :class:`Instruction` instance.  Execution semantics:

        - STATE instructions merge their parameters into the result
          context (a ``store`` of ``{"key": "value"}`` yields
          ``context["key"] == "value"``);
        - LLM/CONTROL/RESOURCE instructions echo their parameters;
        - a ``should_fail`` parameter marks a simulated failure: the
          error is swallowed into the metrics (the manager guarantees
          graceful degradation rather than raising to the caller).
        """
        if isinstance(instruction, str):
            instruction = self.get_instruction(instruction)
        context = context or {}

        # A parameter whose value is ``None`` is an invalid instruction
        # specification (no operation payload); surface it instead of
        # silently executing an empty operation.
        invalid_params = [
            key for key, value in (instruction.params or {}).items()
            if value is None
        ]
        if invalid_params:
            raise ValueError(
                f"Instruction {instruction.id!r} has invalid parameter "
                f"value(s): {', '.join(invalid_params)}"
            )

        self._execution_count += 1
        started = time.perf_counter()
        result_context: Dict[str, Any] = dict(context)
        status = InstructionStatus.SUCCESS
        error_note = None

        if instruction.type == InstructionType.STATE:
            result_context.update(instruction.params)
        else:
            result_context.update(
                {f"executed_{k}": v for k, v in instruction.params.items()}
            )

        if instruction.params.get("should_fail"):
            # Simulated failure handled gracefully by the manager.
            error_note = "instruction requested simulated failure; recovered"
            result_context["recovered"] = True

        elapsed = time.perf_counter() - started
        self.execution_history.append({
            "instruction": instruction.id,
            "type": "single",
            "status": "completed",
            "result": {"context": result_context},
        })

        return InstructionResult(
            id=instruction.id,
            status=status,
            content=f"Executed {instruction.name}",
            error=error_note,
            context=result_context,
            metrics={
                "execution_time": elapsed,
                "instruction_type": instruction.type.value,
                "recovered_from_failure": error_note is not None,
            },
        )

    def compose_instructions(self, instruction_ids: List[str],
                             composite_id: str) -> "CompositeISAInstruction":
        """Compose registered instructions into a named composite."""
        steps = [self.get_instruction(i) for i in instruction_ids]
        composite = CompositeISAInstruction(
            id=composite_id,
            name=composite_id,
            steps=steps,
        )
        self.instructions[composite_id] = composite
        self.dependency_graph.add_node(composite_id)
        for step in steps:
            self.dependency_graph.add_edge(step.id, composite_id)
        return composite

    def optimize_sequence(self, sequence: List[Instruction]) -> List[Instruction]:
        """Collapse consecutive duplicate instructions in a sequence."""
        optimized: List[Instruction] = []
        for instruction in sequence:
            if optimized and optimized[-1].id == instruction.id:
                continue
            optimized.append(instruction)
        return optimized

    def verify_instruction(self, instruction: Instruction) -> bool:
        """Verify an instruction is well-formed enough to execute.

        A valid instruction needs a non-empty id, a name, a known type
        and a non-empty parameter payload (an instruction without any
        parameters carries no operation specification).
        """
        if not instruction.id or not instruction.name:
            return False
        if not isinstance(instruction.type, InstructionType):
            try:
                InstructionType(instruction.type)
            except ValueError:
                return False
        return bool(instruction.params)

    def create_pipeline(self, instruction_ids: List[str]) -> "ISAPipeline":
        """Create an executable pipeline over registered instructions."""
        return ISAPipeline(self, instruction_ids)

    def get_metrics(self) -> Dict[str, Any]:
        """Return aggregate manager-level execution metrics."""
        return {
            "total_instructions": self._execution_count,
            "registered_instructions": len(self.instructions),
            "history_entries": len(self.execution_history),
        }

    def learn_patterns(self, interactions: List[Dict[str, Any]]) -> List[Instruction]:
        """Learn reusable CONTROL instructions from interaction traces.

        Each interaction maps an input to an action sequence.  The most
        frequent action subsequences are promoted into named CONTROL
        instructions carrying the triggering input as parameters.
        """
        from collections import Counter

        pair_counts: Counter = Counter()
        first_inputs: Dict[tuple, str] = {}
        for interaction in interactions:
            actions = interaction.get("actions", [])
            for a, b in zip(actions, actions[1:]):
                pair_counts[(a, b)] += 1
                first_inputs.setdefault((a, b), interaction.get("input", ""))

        learned: List[Instruction] = []
        for (a, b), count in pair_counts.most_common():
            if count < 2:
                continue
            learned.append(Instruction(
                id=f"learned_{a}_{b}",
                name=f"learned_{a}_{b}",
                type=InstructionType.CONTROL,
                description=f"Learned pattern {a} -> {b} (observed {count}x)",
                params={
                    "actions": [a, b],
                    "example_input": first_inputs[(a, b)],
                    "frequency": count,
                },
            ))
        return learned


class CompositeISAInstruction:
    """Composite instruction combining an ordered list of steps."""

    def __init__(self, id: str, name: str, steps: List[Instruction]):
        self.id = id
        self.name = name
        self.steps = steps
        self.type = InstructionType.COMPOSITE
        self.params = {"steps": [s.id for s in steps]}
        self.description = f"Composite of {len(steps)} steps"
        self.dependencies = [s.id for s in steps]
        self.cost = sum(getattr(s, "cost", 1.0) for s in steps)
        self.parallelizable = False
        self.agent_requirements = []


class ISAPipeline:
    """Sequential pipeline executor over registered instructions."""

    def __init__(self, manager: "ISAManager", instruction_ids: List[str]):
        self.manager = manager
        self.instruction_ids = instruction_ids

    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute every pipeline stage, threading the context through."""
        current = dict(context)
        started = time.perf_counter()
        for instruction_id in self.instruction_ids:
            result = await self.manager.execute_instruction(instruction_id, current)
            current.update(result.context)
        elapsed = time.perf_counter() - started
        return InstructionResult(
            id="pipeline",
            status=InstructionStatus.SUCCESS,
            content="Pipeline completed",
            context=current,
            metrics={
                "execution_time": elapsed,
                "stages": len(self.instruction_ids),
            },
        )
