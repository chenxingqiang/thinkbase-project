"""DeepSeek Programmer Agent Implementation.

This module defines an advanced AI-powered programming agent using DeepSeek's 
large language model, integrated with task planning, management, and execution 
capabilities.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from enum import Enum, auto
from dataclasses import dataclass, field

from agentflow.core.agent import Agent, AgentContext, AgentState
from agentflow.core.git import GitConfig, GitOperationsHandler
from agentflow.core.metrics import AgentMetrics
from agentflow.core.task_planning import TaskPlanner, TaskType
from agentflow.core.task_management import Task, TaskContext, TaskState, TaskPriority
from agentflow.core.task_execution import TaskExecutor
from agentflow.core.instructions.thinking import ProgrammerThinkingInstruction
from agentflow.core.workflow import WorkflowManager, WorkflowContext
from agentflow.core.memory import WorkingMemory
from agentflow.core.pipeline import Pipeline
from agentflow.core.llm import LLMConfig, DeepSeekLLM

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DeepSeekGitHubAgentConfig:
    """Configuration for DeepSeek GitHub agent."""
    repository_url: str
    working_dir: str = field(default="./workspace")
    model: str = field(default="deepseek-coder-33b-instruct")
    temperature: float = field(default=0.7)
    max_tokens: int = field(default=4096)
    git_config: Optional[GitConfig] = None
    username: str = field(default="deepseek-agent")
    email: str = field(default="deepseek-agent@noreply.github.com")
    agent_id: str = field(default="deepseek-github-agent")
    agent_name: str = field(default="DeepSeek GitHub Agent")
    task_priority: TaskPriority = field(default=TaskPriority.MEDIUM)

class DeepSeekGitHubAgent(Agent):
    """GitHub agent powered by DeepSeek model with advanced task management."""
    
    def __init__(self, config: DeepSeekGitHubAgentConfig):
        """Initialize agent with comprehensive task management capabilities."""
        super().__init__()
        
        # Configuration and context setup
        self.config = config
        self.context = AgentContext(
            agent_id=config.agent_id,
            agent_name=config.agent_name
        )
        self.context.workspace = Path(config.working_dir)
        
        # LLM Configuration
        self.llm_config = LLMConfig(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens
        )
        self.llm = DeepSeekLLM(self.llm_config)
        
        # Core Agent Components
        self.memory = WorkingMemory()
        self.task_planner = TaskPlanner()
        self.task_executor = TaskExecutor()
        self.workflow_manager = WorkflowManager()
        self.thinking_instruction = ProgrammerThinkingInstruction()
        
        # Pipeline and Metrics
        self.pipeline = Pipeline([])
        self.metrics = AgentMetrics()
        
        # Git Configuration
        self.git_config = config.git_config or GitConfig(
            username=config.username,
            email=config.email,
            default_branch="main"
        )
        self.git_handler = GitOperationsHandler(self.git_config)
        
        # Initialize instruction registry
        self._instruction_registry = {}
        self._register_instructions()
    
    def _register_instructions(self):
        """Register agent instructions."""
        # Add custom instruction registration logic
        pass
    
    async def execute_task(self, task_description: str) -> Dict[str, Any]:
        """Execute a comprehensive programming task.
        
        Args:
            task_description: Detailed description of the programming task
            
        Returns:
            Comprehensive task execution results
        """
        # Create task context
        task_context = TaskContext(
            task_id=f"task_{len(self.context.execution_history) + 1}",
            state=TaskState.INITIALIZED,
            priority=self.config.task_priority,
            requirements={"description": task_description}
        )
        
        try:
            # Thinking Phase
            self.context.state = AgentState.THINKING
            thinking_result = await self.thinking_instruction.execute({
                "task": task_description,
                "context": task_context.to_dict()
            })
            
            # Planning Phase
            self.context.state = AgentState.PLANNING
            instructions = await self.task_planner.plan_task(thinking_result)
            
            # Task Creation
            task = Task(
                id=task_context.task_id,
                description=task_description,
                instructions=instructions,
                context=task_context
            )
            
            # Execution Phase
            self.context.state = AgentState.EXECUTING
            result = await self.task_executor.execute_task(task)
            
            # Workflow Management
            await self._manage_workflow(task)
            
            # Update Execution History
            self.context.execution_history.append({
                "task_id": task.id,
                "description": task_description,
                "result": result,
                "metrics": task_context.metrics
            })
            
            self.context.state = AgentState.COMPLETED
            return result
            
        except Exception as e:
            self.context.state = AgentState.FAILED
            logger.error(f"Task execution failed: {str(e)}")
            raise
    
    async def _manage_workflow(self, task: Task) -> None:
        """Manage task workflow with Git integration.
        
        Args:
            task: Task to manage in workflow
        """
        workflow_context = WorkflowContext(
            task_id=task.id,
            git_config=self.git_config,
            workspace=self.context.workspace
        )
        
        await self.workflow_manager.execute_workflow(
            task=task,
            context=workflow_context
        )
    
    async def review_changes(self, task: Task) -> Dict[str, Any]:
        """Review code changes systematically.
        
        Args:
            task: Task whose changes need review
            
        Returns:
            Comprehensive review results
        """
        self.context.state = AgentState.REVIEWING
        return await self.workflow_manager.review_changes(task)
    
    async def think(self, thought: str) -> None:
        """Record and process a thinking step.
        
        Args:
            thought: Cognitive insight or reasoning
        """
        thinking_result = await self.thinking_instruction.execute({
            "thought": thought,
            "state": self.context.state.name,
            "history": self.context.thinking_process
        })
        
        self.context.thinking_process.append({
            "content": thought,
            "state": self.context.state.name,
            "timestamp": thinking_result.get("timestamp"),
            "insights": thinking_result.get("insights", [])
        })
    
    async def run(self, task: str) -> Dict[str, Any]:
        """Primary method to run a programming task.
        
        Args:
            task: Task description to execute
            
        Returns:
            Comprehensive task execution results
        """
        try:
            result = await self.execute_task(task)
            return result
        except Exception as e:
            logger.error(f"Agent run failed: {e}")
            raise

async def main():
    """Example usage of DeepSeek GitHub Agent."""
    config = DeepSeekGitHubAgentConfig(
        repository_url="https://github.com/example/repo",
        working_dir="./workspace"
    )
    
    agent = DeepSeekGitHubAgent(config)
    
    try:
        # Example task execution
        task = "Create a Python script to analyze GitHub repository statistics"
        result = await agent.run(task)
        print(f"Task Result: {result}")
    except Exception as e:
        print(f"Error executing task: {e}")

if __name__ == "__main__":
    asyncio.run(main())