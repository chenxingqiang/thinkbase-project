"""Task management system for agent workflow."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import asyncio
import logging
from datetime import datetime

from .instructions import (
    BaseInstruction,
    InstructionResult,
    InstructionType,
    Pipeline,
    ThinkingInstruction,
    ProgrammerThinkingInstruction
)

logger = logging.getLogger(__name__)

class TaskState(Enum):
    """Task execution states."""
    INITIALIZED = auto()
    PLANNING = auto()
    EXECUTING = auto()
    THINKING = auto()
    ADJUSTING = auto()
    COMPLETED = auto()
    FAILED = auto()

class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class TaskContext:
    """Context for task execution."""
    task_id: str
    state: TaskState = TaskState.INITIALIZED
    priority: TaskPriority = TaskPriority.MEDIUM
    requirements: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    thinking_process: List[Dict[str, Any]] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def add_thought(self, phase: str, content: str) -> None:
        """Add a thought to the thinking process.
        
        Args:
            phase: Thinking phase
            content: Thought content
        """
        self.thinking_process.append({
            "phase": phase,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
    
    def update_state(self, state: TaskState) -> None:
        """Update task state.
        
        Args:
            state: New state
        """
        self.state = state
        if state == TaskState.INITIALIZED:
            self.start_time = datetime.now()
        elif state in (TaskState.COMPLETED, TaskState.FAILED):
            self.end_time = datetime.now()
    
    def add_artifact(self, name: str, artifact: Any) -> None:
        """Add a task artifact.
        
        Args:
            name: Artifact name
            artifact: Artifact data
        """
        self.artifacts[name] = artifact
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update task metrics.
        
        Args:
            metrics: New metrics
        """
        self.metrics.update(metrics)

@dataclass
class Task:
    """Represents a task to be executed."""
    id: str
    description: str
    instructions: List[BaseInstruction]
    context: TaskContext
    subtasks: List['Task'] = field(default_factory=list)
    parent: Optional['Task'] = None
    
    async def execute(self) -> InstructionResult:
        """Execute the task.
        
        Returns:
            Task execution result
        """
        self.context.update_state(TaskState.EXECUTING)
        
        try:
            # Execute instructions in sequence
            pipeline = Pipeline(self.instructions)
            result = await pipeline.execute(self.context.requirements)
            
            # Update context with results
            for name, instruction_result in result.items():
                if isinstance(instruction_result, InstructionResult):
                    self.context.add_artifact(name, instruction_result.data)
                    self.context.update_metrics(instruction_result.metrics)
            
            self.context.update_state(TaskState.COMPLETED)
            return InstructionResult(
                status="completed",
                data=self.context.artifacts,
                metrics=self.context.metrics
            )
            
        except Exception as e:
            self.context.update_state(TaskState.FAILED)
            return InstructionResult(
                status="failed",
                data={"error": str(e)},
                metrics=self.context.metrics
            )

class TaskManager:
    """Manages task execution and workflow."""
    
    def __init__(self):
        """Initialize task manager."""
        self.tasks: Dict[str, Task] = {}
        self.thinking = ProgrammerThinkingInstruction()
        self._current_task: Optional[Task] = None
    
    async def create_task(
        self,
        description: str,
        requirements: Dict[str, Any],
        priority: TaskPriority = TaskPriority.MEDIUM
    ) -> Task:
        """Create a new task.
        
        Args:
            description: Task description
            requirements: Task requirements
            priority: Task priority
            
        Returns:
            Created task
        """
        # Think about task decomposition
        thought_result = await self._think_about_task(description, requirements)
        
        # Create task context
        task_id = f"task_{len(self.tasks) + 1}"
        context = TaskContext(
            task_id=task_id,
            priority=priority,
            requirements=requirements
        )
        
        # Create task with instructions
        task = Task(
            id=task_id,
            description=description,
            instructions=self._plan_instructions(thought_result),
            context=context
        )
        
        self.tasks[task_id] = task
        return task
    
    async def execute_task(self, task: Task) -> InstructionResult:
        """Execute a task.
        
        Args:
            task: Task to execute
            
        Returns:
            Task execution result
        """
        self._current_task = task
        
        try:
            # Execute task
            result = await task.execute()
            
            # Think about results
            await self._think_about_results(task, result)
            
            return result
            
        finally:
            self._current_task = None
    
    async def _think_about_task(
        self,
        description: str,
        requirements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Think about how to approach a task.
        
        Args:
            description: Task description
            requirements: Task requirements
            
        Returns:
            Thinking results
        """
        context = {
            "problem": description,
            "requirements": requirements,
            "phase": "planning"
        }
        
        result = await self.thinking.execute(context)
        return result.data if isinstance(result, InstructionResult) else result
    
    async def _think_about_results(
        self,
        task: Task,
        result: InstructionResult
    ) -> None:
        """Think about task execution results.
        
        Args:
            task: Executed task
            result: Task result
        """
        context = {
            "task": task.description,
            "results": result.data,
            "metrics": result.metrics,
            "phase": "evaluation"
        }
        
        thought_result = await self.thinking.execute(context)
        if isinstance(thought_result, InstructionResult):
            task.context.add_thought(
                "evaluation",
                thought_result.data.get("insights", "No insights generated")
            )
    
    def _plan_instructions(self, thinking_result: Dict[str, Any]) -> List[BaseInstruction]:
        """Plan instructions based on thinking results.
        
        Args:
            thinking_result: Results from thinking about task
            
        Returns:
            List of instructions to execute
        """
        # This is a placeholder - actual implementation would create
        # appropriate instructions based on thinking results
        return []
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """Get a task by ID.
        
        Args:
            task_id: Task ID
            
        Returns:
            Task if found, None otherwise
        """
        return self.tasks.get(task_id)
    
    def get_current_task(self) -> Optional[Task]:
        """Get currently executing task.
        
        Returns:
            Current task if any, None otherwise
        """
        return self._current_task 