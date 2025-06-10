"""Base agent class implementing AgentISA design."""

from typing import Dict, Any, Optional, List, Set, Tuple
import uuid
from dataclasses import asdict, dataclass, field
import numpy as np

from ..core.agent_config import AgentConfig, AgentCapability
from ..core.isa.instruction import Instruction
from ..core.instructions import (
    InstructionType,
    BaseInstruction
)
from ..core.instructions.minimal import (
    MinimalTestDesign,
    MinimalCodeReview,
    MinimalCodeFile,
    BaseMinimalInstruction
)
from ..core.instructions.registry import registry
from ..core.instructions.base import ExecutionContext, ExecutionResult
from ..core.isa.result import InstructionStatus, InstructionResult
from .state import AgentState, AgentStatus

@dataclass
class AgentMetrics:
    """Agent performance metrics."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_reward: float = 0.0
    total_cost: float = 0.0
    instruction_usage: Dict[str, int] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        return self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def average_reward(self) -> float:
        return self.total_reward / self.total_executions if self.total_executions > 0 else 0.0

class MinimalInstructionWrapper(BaseMinimalInstruction):
    """Wrapper for minimal instructions to properly implement execute method."""
    
    def __init__(self, base_instruction: BaseMinimalInstruction):
        super().__init__(base_instruction.name, "Wrapper for " + base_instruction.name)
        self.base_instruction = base_instruction
        # Register the wrapped instruction
        registry.register_minimal_instruction(base_instruction)
        registry.register_instruction(self)
        
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute instruction by delegating to base instruction."""
        try:
            result = await self.base_instruction._execute_impl(context.params)
            if not isinstance(result, InstructionResult):
                raise ValueError("Expected InstructionResult from _execute_impl")
                
            # Convert content to dict if it's a string
            output: Dict[str, Any] = (
                {"content": result.content} if isinstance(result.content, str)
                else (result.content if result.content is not None else {})
            )
                
            return ExecutionResult(
                success=result.status == InstructionStatus.COMPLETED,
                output=output,
                errors=[{"error": result.error}] if result.error else [],
                state_updates={},
                metrics=result.metrics if result.metrics is not None else {}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                state_updates={},
                metrics={}
            )
        
    async def _execute_core(self, context: Dict[str, Any]) -> ExecutionResult:
        """Execute instruction by delegating to base instruction."""
        try:
            result = await self.base_instruction._execute_core(context)
            if not isinstance(result, InstructionResult):
                raise ValueError("Expected InstructionResult from _execute_core")
                
            # Convert content to dict if it's a string
            output: Dict[str, Any] = (
                {"content": result.content} if isinstance(result.content, str)
                else (result.content if result.content is not None else {})
            )
                
            return ExecutionResult(
                success=result.status == InstructionStatus.COMPLETED,
                output=output,
                errors=[{"error": result.error}] if result.error else [],
                state_updates={},
                metrics=result.metrics if result.metrics is not None else {}
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                state_updates={},
                metrics={}
            )

class Agent:
    """Base agent class with reinforcement learning capabilities."""
    
    def __init__(self, config: AgentConfig):
        self.id = str(uuid.uuid4())
        self.name = config.agent_name
        self.type = "base"
        self.config = config
        self.state = AgentState(status=AgentStatus.INITIALIZED)
        self.instructions: Set[Instruction] = set()
        self.metrics = AgentMetrics()
        
        # RL-specific attributes
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1
        
        self._initialize_q_table()
        
    def _get_default_instructions(self) -> Set[Instruction]:
        """Get default instructions based on agent capabilities."""
        default_instructions = set()
        for capability in self.config.agent_capabilities:
            if capability == AgentCapability.CODE_GENERATION:
                instruction = MinimalInstructionWrapper(MinimalTestDesign())
                default_instructions.add(instruction)
            elif capability == AgentCapability.CODE_REVIEW:
                instruction = MinimalInstructionWrapper(MinimalCodeReview())
                default_instructions.add(instruction)
            elif capability == AgentCapability.REFACTORING:
                instruction = MinimalInstructionWrapper(MinimalCodeFile())
                default_instructions.add(instruction)
        return default_instructions
    
    def _initialize_q_table(self) -> None:
        """Initialize Q-table with zero values for all state-instruction pairs."""
        for instruction in self.instructions:
            self.q_table[instruction.name] = {}
            for state in AgentStatus:
                self.q_table[instruction.name][state.value] = 0.0
    
    def add_instruction(self, instruction: Instruction) -> None:
        """Add an instruction to the agent's instruction set."""
        self.instructions.add(instruction)
        registry.register_instruction(instruction)
        self._initialize_q_table()
    
    def remove_instruction(self, instruction: Instruction) -> None:
        """Remove an instruction from the agent's instruction set."""
        self.instructions.remove(instruction)
        self._initialize_q_table()
    
    def select_instruction(self, state: AgentState) -> Optional[Instruction]:
        """Select an instruction using epsilon-greedy policy."""
        if not self.instructions:
            return None
            
        if np.random.random() < self.exploration_rate:
            instructions_list = list(self.instructions)
            return instructions_list[np.random.randint(0, len(instructions_list))]
            
        best_instruction = None
        best_q_value = float('-inf')
        
        for instruction in self.instructions:
            q_value = self.q_table[instruction.name][state.status.value]
            if q_value > best_q_value:
                best_q_value = q_value
                best_instruction = instruction
                
        return best_instruction
    
    def update_q_value(self, 
                      instruction: Instruction, 
                      state: AgentState,
                      next_state: AgentState,
                      reward: float) -> None:
        """Update Q-value for a state-instruction pair using Q-learning."""
        current_q = self.q_table[instruction.name][state.status.value]
        
        next_q = max(
            self.q_table[i.name][next_state.status.value] 
            for i in self.instructions
        ) if self.instructions else 0.0
        
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_q - current_q
        )
        
        self.q_table[instruction.name][state.status.value] = new_q
    
    def calculate_reward(self, 
                        instruction: Instruction,
                        result: Dict[str, Any],
                        cost: float) -> float:
        """Calculate reward for instruction execution."""
        # Base reward for successful execution
        reward = 1.0 if result.get("success", False) else -1.0
        
        # Adjust reward based on cost
        reward -= cost * 0.1  # Cost penalty factor
        
        # Additional rewards based on metrics
        if "metrics" in result:
            metrics = result["metrics"]
            if "execution_time" in metrics:
                # Time efficiency reward
                reward += 0.5 if metrics["execution_time"] < 1.0 else 0.0
            if "accuracy" in metrics:
                # Accuracy reward
                reward += metrics["accuracy"]
        
        return reward
    
    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent with the given input data.
        
        Args:
            input_data: Input data for execution
            
        Returns:
            Dict[str, Any]: Execution results containing success status, output and metrics
            
        Raises:
            ValueError: If no instruction is available
            Exception: If execution fails
        """
        raise NotImplementedError("Execute method must be implemented by subclass")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "config": self.config.dict(),
            "state": self.state.to_dict(),
            "instructions": [instruction.name for instruction in self.instructions],
            "metrics": self.metrics.__dict__,
            "q_table": self.q_table
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Create agent from dictionary."""
        config = AgentConfig(**data.get("config", {}))
        agent = cls(config)
        agent.id = data.get("id")
        agent.name = data.get("name")
        agent.type = data.get("type")
        agent.state = AgentState(**data.get("state", {}))
        agent.instructions = set(data.get("instructions", []))
        agent.metrics = AgentMetrics(**data.get("metrics", {}))
        agent.q_table = data.get("q_table", {})
        return agent

    async def initialize(self) -> None:
        """Initialize agent resources."""
        if self.state.status != AgentStatus.INITIALIZED:
            self.state.update(status=AgentStatus.INITIALIZED)
        
        self._initialize_q_table()
    
    def _initialize_state(self) -> None:
        """Initialize agent state."""
        self.state.update(status=AgentStatus.INITIALIZED)
    
    def _initialize_metrics(self) -> None:
        """Initialize agent metrics."""
        self.metrics.total_executions = 0
        self.metrics.successful_executions = 0
        self.metrics.failed_executions = 0
        self.metrics.total_reward = 0.0
        self.metrics.total_cost = 0.0
        self.metrics.instruction_usage = {}
    
    def _initialize_rl_attributes(self) -> None:
        """Initialize reinforcement learning attributes."""
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.1

    def _initialize_all(self) -> None:
        """Initialize all agent resources."""
        self._initialize_state()
        self._initialize_metrics()
        self._initialize_q_table()
        self._initialize_rl_attributes()
        
    def _update_state(self, status: AgentStatus, error_message: Optional[str] = None) -> None:
        """Update agent state."""
        self.state.update(status=status, error_message=error_message)
        
    def _update_instructions(self, instructions: Set[Instruction]) -> None:
        """Update agent instructions."""
        self.instructions.clear()
        self.instructions.update(instructions)
        
    def _update_metrics(self, total_executions: int, successful_executions: int, failed_executions: int, total_reward: float, total_cost: float, instruction_usage: Dict[str, int]) -> None:
        """Update agent metrics."""
        self.metrics.total_executions = total_executions
        self.metrics.successful_executions = successful_executions
        self.metrics.failed_executions = failed_executions
        self.metrics.total_reward = total_reward
        self.metrics.total_cost = total_cost
        self.metrics.instruction_usage = instruction_usage
        
    def _update_q_table(self, q_table: Dict[str, Dict[str, float]]) -> None:
        """Update agent Q-table."""
        self.q_table.clear()
        self.q_table.update(q_table)
        
    def _update_rl_attributes(self, learning_rate: float, discount_factor: float, exploration_rate: float) -> None:
        """Update reinforcement learning attributes."""
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        
    def _update_all(self, state: AgentState, instructions: Set[Instruction], metrics: AgentMetrics, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update all agent resources."""
        self._update_state_all(state, instructions, metrics, q_table, rl_attributes)
        self._update_instructions_all(instructions, metrics, q_table, rl_attributes)
        self._update_metrics_all(metrics, q_table, rl_attributes)
        self._update_q_table_all(q_table, rl_attributes)
        self._update_rl_attributes_all(rl_attributes)
        
    def _update_state_instructions(self, state: AgentState, instructions: Set[Instruction]) -> None:
        """Update agent state and instructions."""
        self._update_state(state.status, state.error_message)
        self._update_instructions(instructions)
        
    def _update_state_metrics(self, state: AgentState, metrics: AgentMetrics) -> None:
        """Update agent state and metrics."""
        self._update_state(state.status, state.error_message)
        self._update_metrics(metrics.total_executions, metrics.successful_executions, metrics.failed_executions, metrics.total_reward, metrics.total_cost, metrics.instruction_usage)
        
    def _update_state_q_table(self, state: AgentState, q_table: Dict[str, Dict[str, float]]) -> None:
        """Update agent state and Q-table."""
        self._update_state(state.status, state.error_message)
        self._update_q_table(q_table)
        
    def _update_state_rl_attributes(self, state: AgentState, rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent state and reinforcement learning attributes."""
        self._update_state(state.status, state.error_message)
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    def _update_state_all(self, state: AgentState, instructions: Set[Instruction], metrics: AgentMetrics, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent state, instructions, metrics, and Q-table."""
        self._update_state_instructions(state, instructions)
        self._update_state_metrics(state, metrics)
        self._update_state_q_table(state, q_table)
        self._update_state_rl_attributes(state, rl_attributes)
        
    def _update_instructions_metrics(self, instructions: Set[Instruction], metrics: AgentMetrics) -> None:
        """Update agent instructions and metrics."""
        self._update_instructions(instructions)
        self._update_metrics(metrics.total_executions, metrics.successful_executions, metrics.failed_executions, metrics.total_reward, metrics.total_cost, metrics.instruction_usage)
        
    def _update_instructions_q_table(self, instructions: Set[Instruction], q_table: Dict[str, Dict[str, float]]) -> None:
        """Update agent instructions and Q-table."""
        self._update_instructions(instructions)
        self._update_q_table(q_table)
        
    def _update_instructions_rl_attributes(self, instructions: Set[Instruction], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent instructions and reinforcement learning attributes."""
        self._update_instructions(instructions)
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    def _update_instructions_all(self, instructions: Set[Instruction], metrics: AgentMetrics, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent instructions, metrics, and Q-table."""
        self._update_instructions_metrics(instructions, metrics)
        self._update_instructions_q_table(instructions, q_table)
        self._update_instructions_rl_attributes(instructions, rl_attributes)
        
    def _update_metrics_q_table(self, metrics: AgentMetrics, q_table: Dict[str, Dict[str, float]]) -> None:
        """Update agent metrics and Q-table."""
        self._update_metrics(metrics.total_executions, metrics.successful_executions, metrics.failed_executions, metrics.total_reward, metrics.total_cost, metrics.instruction_usage)
        self._update_q_table(q_table)
        
    def _update_metrics_rl_attributes(self, metrics: AgentMetrics, rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent metrics and reinforcement learning attributes."""
        self._update_metrics(metrics.total_executions, metrics.successful_executions, metrics.failed_executions, metrics.total_reward, metrics.total_cost, metrics.instruction_usage)
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    def _update_metrics_all(self, metrics: AgentMetrics, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent metrics, Q-table, and reinforcement learning attributes."""
        self._update_metrics_q_table(metrics, q_table)
        self._update_metrics_rl_attributes(metrics, rl_attributes)
        
    def _update_q_table_rl_attributes(self, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent Q-table and reinforcement learning attributes."""
        self._update_q_table(q_table)
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    def _update_q_table_all(self, q_table: Dict[str, Dict[str, float]], rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent Q-table and reinforcement learning attributes."""
        self._update_q_table(q_table)
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    def _update_rl_attributes_all(self, rl_attributes: Tuple[float, float, float]) -> None:
        """Update agent reinforcement learning attributes."""
        self._update_rl_attributes(rl_attributes[0], rl_attributes[1], rl_attributes[2])
        
    