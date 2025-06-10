"""Main AgentISA interface implementation."""

from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
import logging
import time
from .instruction import Instruction, InstructionType, TypeConstraint
from .agentic_calling import AgenticTransformer, ModelAdapter
from .instruction_learning import InstructionLearner
from .instruction_optimization import OptimizationManager, OptimizationMetrics

logger = logging.getLogger(__name__)

@dataclass
class AgentISAConfig:
    """Configuration for AgentISA."""
    enable_learning: bool = True
    enable_optimization: bool = True
    model_type: str = "default"
    min_pattern_support: int = 3
    min_success_rate: float = 0.7
    hot_path_threshold: int = 100
    stability_threshold: float = 0.9

class AgentISA:
    """Main interface for AgentISA framework."""
    
    def __init__(self, config: Optional[AgentISAConfig] = None):
        """Initialize AgentISA framework."""
        self.config = config or AgentISAConfig()
        
        # Initialize components
        self.transformer = AgenticTransformer()
        self.model_adapter = ModelAdapter(self.config.model_type)
        self.instruction_learner = InstructionLearner(self.transformer)
        self.optimization_manager = OptimizationManager()
        
        # Instruction registry
        self.instructions: Dict[str, Instruction] = {}
        self.execution_history: List[Dict[str, Any]] = []

    async def register_function(self, 
                              func: Callable,
                              context_requirements: Optional[Dict[str, Any]] = None) -> str:
        """Register a function as an instruction."""
        instruction = self.transformer.transform_to_instruction(
            func, context_requirements
        )
        instruction_id = instruction.id
        self.instructions[instruction_id] = instruction
        return instruction_id

    async def register_instruction(self, instruction: Instruction) -> str:
        """Register a pre-built instruction."""
        self.instructions[instruction.id] = instruction
        return instruction.id

    async def execute_instruction(self, 
                                instruction_id: str,
                                params: Dict[str, Any]) -> Any:
        """Execute an instruction with optimization."""
        if instruction_id not in self.instructions:
            raise ValueError(f"Instruction {instruction_id} not found")
            
        instruction = self.instructions[instruction_id]
        
        # Apply optimization if enabled
        if self.config.enable_optimization:
            optimized_instructions = self.optimization_manager.optimize_instructions(
                [instruction]
            )
            if optimized_instructions:
                instruction = optimized_instructions[0]
        
        # Execute instruction
        start_time = time.perf_counter()
        try:
            result = await instruction.execute(params)
            success = True
        except Exception as e:
            logger.error(f"Instruction execution failed: {str(e)}")
            success = False
            result = None
            raise
        finally:
            end_time = time.perf_counter()
            
            # Record execution metrics
            metrics = OptimizationMetrics(
                execution_time=end_time - start_time,
                memory_usage=0.0,  # Would need proper memory tracking
                api_calls=1,  # Would need proper API call tracking
                success_rate=1.0 if success else 0.0
            )
            
            # Record execution for optimization
            if self.config.enable_optimization:
                self.optimization_manager.record_execution(
                    [instruction],
                    metrics
                )
            
            # Record execution history
            self.execution_history.append({
                "instruction_id": instruction_id,
                "params": params,
                "result": result,
                "metrics": metrics,
                "success": success
            })
        
        return result

    async def learn_from_history(self):
        """Learn new instructions from execution history."""
        if not self.config.enable_learning:
            return
            
        # Convert execution history to dialogue format
        dialogues = [
            {
                "turns": [
                    {
                        "role": "system",
                        "content": f"Execute {entry['instruction_id']}"
                    },
                    {
                        "role": "user",
                        "content": str(entry['params'])
                    },
                    {
                        "role": "assistant",
                        "content": str(entry['result'])
                    }
                ],
                "success": entry["success"],
                "context": entry["params"]
            }
            for entry in self.execution_history
        ]
        
        # Learn new instructions
        self.instruction_learner.learn_from_dialogues(dialogues)
        
        # Register learned instructions
        for template in self.instruction_learner.instruction_templates.values():
            instruction = self.instruction_learner.create_instruction(template)
            await self.register_instruction(instruction)

    def get_instruction_stats(self) -> Dict[str, Any]:
        """Get statistics about instruction usage and performance."""
        return {
            "total_instructions": len(self.instructions),
            "total_executions": len(self.execution_history),
            "success_rate": sum(1 for e in self.execution_history if e["success"]) / 
                          len(self.execution_history) if self.execution_history else 0,
            "optimization_stats": self.optimization_manager.static_optimizer.optimization_stats,
            "learned_instructions": len(self.instruction_learner.instruction_templates)
        }

    async def cleanup(self):
        """Clean up resources."""
        self.instructions.clear()
        self.execution_history.clear() 