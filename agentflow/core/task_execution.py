"""Task execution system for managing task flow and context."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import asyncio
import logging
from datetime import datetime

from .task_management import (
    Task,
    TaskContext,
    TaskState,
    TaskPriority
)
from .task_planning import TaskPlanner
from .instructions import (
    BaseInstruction,
    InstructionResult,
    ProgrammerThinkingInstruction
)

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    """Metrics for task execution."""
    start_time: datetime
    end_time: Optional[datetime] = None
    phase_durations: Dict[str, float] = field(default_factory=dict)
    instruction_counts: Dict[str, int] = field(default_factory=dict)
    error_counts: Dict[str, int] = field(default_factory=dict)
    thinking_metrics: Dict[str, Any] = field(default_factory=dict)

class TaskExecutor:
    """Executes tasks and manages execution context."""
    
    def __init__(self):
        """Initialize task executor."""
        self.planner = TaskPlanner()
        self.thinking = ProgrammerThinkingInstruction()
        self._current_context: Dict[str, Any] = {}
        self._execution_history: List[Dict[str, Any]] = []
    
    async def execute_task(self, task: Task) -> InstructionResult:
        """Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        # Initialize metrics
        metrics = ExecutionMetrics(start_time=datetime.now())
        
        try:
            # Set current context
            self._current_context = task.context.requirements.copy()
            
            # Execute pre-task thinking
            pre_task_result = await self._pre_task_thinking(task)
            metrics.thinking_metrics["pre_task"] = pre_task_result.metrics
            
            # Execute instructions
            result = await self._execute_instructions(
                task.instructions,
                task.context,
                metrics
            )
            
            # Execute post-task thinking
            post_task_result = await self._post_task_thinking(task, result)
            metrics.thinking_metrics["post_task"] = post_task_result.metrics
            
            # Update metrics
            metrics.end_time = datetime.now()
            
            # Record execution
            self._record_execution(task, result, metrics)
            
            return result
            
        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            metrics.end_time = datetime.now()
            metrics.error_counts["execution"] = metrics.error_counts.get("execution", 0) + 1
            
            return InstructionResult(
                status="failed",
                data={"error": str(e)},
                metrics=metrics.__dict__
            )
            
        finally:
            self._current_context = {}
    
    async def _pre_task_thinking(self, task: Task) -> InstructionResult:
        """Execute pre-task thinking.
        
        Args:
            task: Task to execute
            
        Returns:
            Thinking result
        """
        context = {
            "phase": "pre_task",
            "task": task.description,
            "requirements": task.context.requirements,
            "instructions": [i.name for i in task.instructions]
        }
        
        return await self.thinking.execute(context)
    
    async def _post_task_thinking(
        self,
        task: Task,
        result: InstructionResult
    ) -> InstructionResult:
        """Execute post-task thinking.
        
        Args:
            task: Executed task
            result: Task result
            
        Returns:
            Thinking result
        """
        context = {
            "phase": "post_task",
            "task": task.description,
            "result": result.data,
            "metrics": result.metrics
        }
        
        return await self.thinking.execute(context)
    
    async def _execute_instructions(
        self,
        instructions: List[BaseInstruction],
        task_context: TaskContext,
        metrics: ExecutionMetrics
    ) -> InstructionResult:
        """Execute a list of instructions.
        
        Args:
            instructions: Instructions to execute
            task_context: Task context
            metrics: Execution metrics
            
        Returns:
            Execution result
        """
        results = {}
        
        for instruction in instructions:
            # Update context state
            task_context.update_state(TaskState.EXECUTING)
            
            # Execute instruction
            start_time = datetime.now()
            result = await instruction.execute(self._current_context)
            end_time = datetime.now()
            
            # Update metrics
            duration = (end_time - start_time).total_seconds()
            metrics.phase_durations[instruction.name] = duration
            metrics.instruction_counts[instruction.name] = metrics.instruction_counts.get(instruction.name, 0) + 1
            
            if result.status == "failed":
                metrics.error_counts[instruction.name] = metrics.error_counts.get(instruction.name, 0) + 1
            
            # Update context
            if isinstance(result, InstructionResult):
                self._current_context.update(result.data)
                task_context.add_artifact(instruction.name, result.data)
                task_context.update_metrics(result.metrics)
            
            results[instruction.name] = result
            
            # Think about intermediate results
            await self._think_about_instruction(instruction, result)
        
        # Combine results
        combined_data = {}
        combined_metrics = {}
        
        for name, result in results.items():
            if isinstance(result, InstructionResult):
                combined_data[name] = result.data
                combined_metrics.update(result.metrics)
        
        return InstructionResult(
            status="completed",
            data=combined_data,
            metrics=combined_metrics
        )
    
    async def _think_about_instruction(
        self,
        instruction: BaseInstruction,
        result: InstructionResult
    ) -> None:
        """Think about instruction execution.
        
        Args:
            instruction: Executed instruction
            result: Instruction result
        """
        context = {
            "phase": "instruction",
            "instruction": instruction.name,
            "result": result.data,
            "metrics": result.metrics
        }
        
        thought_result = await self.thinking.execute(context)
        if isinstance(thought_result, InstructionResult):
            self._current_context.setdefault("thoughts", []).append({
                "instruction": instruction.name,
                "insights": thought_result.data.get("insights", [])
            })
    
    def _record_execution(
        self,
        task: Task,
        result: InstructionResult,
        metrics: ExecutionMetrics
    ) -> None:
        """Record task execution.
        
        Args:
            task: Executed task
            result: Task result
            metrics: Execution metrics
        """
        self._execution_history.append({
            "task_id": task.id,
            "description": task.description,
            "start_time": metrics.start_time.isoformat(),
            "end_time": metrics.end_time.isoformat() if metrics.end_time else None,
            "status": result.status,
            "metrics": metrics.__dict__,
            "context": self._current_context.copy()
        })
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history.
        
        Returns:
            List of execution records
        """
        return self._execution_history.copy()
    
    def get_current_context(self) -> Dict[str, Any]:
        """Get current execution context.
        
        Returns:
            Current context
        """
        return self._current_context.copy() 