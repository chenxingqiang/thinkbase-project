"""
DeepSeek GitHub Agent implementation.
This agent specializes in GitHub repository management and code operations.
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path

from agentflow.core.memory import WorkingMemory
from agentflow.core.instructions import (
    BaseInstruction,
    InstructionResult,
    Pipeline
)
from agentflow.core.instructions.minimal import (
    MINIMAL_INSTRUCTION_SET,
    MinimalInstructionType
)
from agentflow.core.isa.types import InstructionStatus
from agentflow.core.git_operations import GitConfig, GitOperationsHandler
from agentflow.providers.deepseek import DeepSeekProvider, DeepSeekConfig
from examples.deepseek_instructions import (
    INSTRUCTION_REGISTRY,
    DeepSeekInstructionType,
    BaseDeepSeekInstruction,
    RequirementAnalysisInstruction,
    CodeAnalysisInstruction,
    TestDesignInstruction,
    CodeGenerationInstruction,
    CodeReviewInstruction,
    GitOperationsInstruction
)

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent execution states."""
    INITIALIZED = auto()
    THINKING = auto()
    PLANNING = auto()
    EXECUTING = auto()
    REVIEWING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class AgentMetrics:
    """Metrics for agent operations."""
    tokens_used: int = 0
    latency: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0
    total_operations: int = 0
    total_successes: int = 0
    thinking_time: float = 0.0
    execution_time: float = 0.0
    git_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentContext:
    """Context for agent execution."""
    state: AgentState = AgentState.INITIALIZED
    current_task: Optional[Dict[str, Any]] = None
    thinking_process: List[Dict[str, Any]] = field(default_factory=list)
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    metrics: AgentMetrics = field(default_factory=AgentMetrics)
    workspace: Path = field(default_factory=lambda: Path("./workspace"))

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

class DeepSeekGitHubAgent:
    """
    An AI agent powered by DeepSeek model for GitHub operations.
    Implements TDD workflow and Git operations management.
    """
    
    def __init__(self, config: DeepSeekGitHubAgentConfig):
        """Initialize GitHub agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config
        self.context = AgentContext()
        self.context.workspace = Path(config.working_dir)
        self.memory = WorkingMemory()
        self.provider = None
        self.pipeline = Pipeline(instructions=[])
        self.git_handler = None
        
    def _register_instructions(self) -> None:
        """Register available instructions."""
        self.pipeline.instructions = [
            RequirementAnalysisInstruction(),
            CodeAnalysisInstruction(),
            TestDesignInstruction(),
            CodeGenerationInstruction(),
            CodeReviewInstruction(),
            GitOperationsInstruction()
        ]
    
    async def think(self, thought: str) -> None:
        """Record a thinking process.
        
        Args:
            thought: Thought content
        """
        self.context.thinking_process.append({
            "timestamp": asyncio.get_event_loop().time(),
            "content": thought
        })
        
    async def analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a task and create execution plan.
        
        Args:
            task: Task specification
            
        Returns:
            Analysis results
        """
        await self.think(f"Analyzing task: {task.get('description', '')}")
        
        try:
            result = await self.pipeline.execute_instruction(
                "requirement_analysis",
                {
                    "task": task,
                    "context": self.context.__dict__
                }
            )
            
            if isinstance(result, InstructionResult):
                self._update_metrics(result.metrics)
                return result.data
            return result
            
        except Exception as e:
            self.context.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def plan_implementation(self, analysis: Dict[str, Any]) -> List[BaseDeepSeekInstruction]:
        """Plan implementation based on analysis.
        
        Args:
            analysis: Task analysis
            
        Returns:
            List of instructions to execute
        """
        await self.think("Planning implementation steps")
        
        # Create instruction sequence based on analysis
        instructions = []
        
        if analysis.get("requires_tests", True):
            instructions.append(TestDesignInstruction())
            
        if analysis.get("requires_code", True):
            instructions.append(CodeGenerationInstruction())
            
        if analysis.get("requires_review", True):
            instructions.append(CodeReviewInstruction())
            
        if analysis.get("requires_git", True):
            instructions.append(GitOperationsInstruction())
            
        return instructions
    
    async def execute_instructions(self, instructions: List[BaseDeepSeekInstruction]) -> Dict[str, Any]:
        """Execute a sequence of instructions.
        
        Args:
            instructions: Instructions to execute
            
        Returns:
            Execution results
        """
        results = {}
        
        for instruction in instructions:
            await self.think(f"Executing instruction: {instruction.name}")
            
            try:
                result = await instruction.execute({
                    "context": self.context.__dict__,
                    "previous_results": results
                })
                
                if isinstance(result, InstructionResult):
                    self._update_metrics(result.metrics)
                    results[instruction.name] = result.data
                else:
                    results[instruction.name] = result
                    
            except Exception as e:
                self.context.metrics.error_count += 1
                results[instruction.name] = {
                    "status": "error",
                    "message": str(e)
                }
                
        return results
    
    async def review_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Review execution results.
        
        Args:
            results: Execution results
            
        Returns:
            Review summary
        """
        await self.think("Reviewing execution results")
        
        try:
            result = await self.pipeline.execute_instruction(
                "code_review",
                {
                    "results": results,
                    "context": self.context.__dict__
                }
            )
            
            if isinstance(result, InstructionResult):
                self._update_metrics(result.metrics)
                return result.data
            return result
            
        except Exception as e:
            self.context.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a complete task.
        
        Args:
            task: Task specification
            
        Returns:
            Task execution results
        """
        self.context.current_task = task
        self.context.state = AgentState.EXECUTING
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Analyze task
            analysis = await self.analyze_task(task)
            if "error" in analysis:
                return analysis
                
            # Plan implementation
            instructions = await self.plan_implementation(analysis)
            
            # Execute instructions
            results = await self.execute_instructions(instructions)
            
            # Review results
            review = await self.review_results(results)
            
            # Update execution history
            self.context.execution_history.append({
                "task": task,
                "analysis": analysis,
                "results": results,
                "review": review,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            self.context.state = AgentState.COMPLETED
            return {
                "status": "success",
                "analysis": analysis,
                "results": results,
                "review": review
            }
            
        except Exception as e:
            self.context.state = AgentState.FAILED
            self.context.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
            
        finally:
            end_time = asyncio.get_event_loop().time()
            self.context.metrics.execution_time += (end_time - start_time)
    
    async def initialize(self) -> None:
        """Initialize agent resources."""
        try:
            # Create workspace
            self.context.workspace.mkdir(parents=True, exist_ok=True)
            
            # Initialize provider
            self.provider = DeepSeekProvider(
                config=DeepSeekConfig(
                    model_name=self.config.model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens
                )
            )
            
            # Initialize Git handler
            if self.config.git_config:
                self.git_handler = GitOperationsHandler(self.config.git_config)
                await self.memory.store("git_handler", self.git_handler)
            
            # Register instructions
            self._register_instructions()
            
            self.context.state = AgentState.INITIALIZED
            
        except Exception as e:
            self.context.state = AgentState.FAILED
            raise RuntimeError(f"Failed to initialize agent: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.provider:
            await self.provider.close()
        await self.memory.clear()
        await self.pipeline.cleanup()
        
    def _update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update agent metrics.
        
        Args:
            metrics: New metrics to update
        """
        self.context.metrics.latency += metrics.get("latency", 0.0)
        self.context.metrics.tokens_used += metrics.get("tokens", 0)
        self.context.metrics.total_operations += 1
        self.context.metrics.total_successes += metrics.get("success", 1.0)
        self.context.metrics.success_rate = (
            self.context.metrics.total_successes / self.context.metrics.total_operations
            if self.context.metrics.total_operations > 0
            else 0.0
        )
        
        # Update Git metrics if available
        if git_metrics := metrics.get("git", {}):
            self.context.metrics.git_metrics.update(git_metrics)

async def main():
    """Main entry point."""
    # Create configuration
    config = DeepSeekGitHubAgentConfig(
        repository_url="https://github.com/example/repo",
        working_dir="./workspace",
        git_config=GitConfig(
            username="deepseek-agent",
            email="deepseek-agent@noreply.github.com",
            default_branch="main"
        )
    )
    
    # Create agent
    agent = DeepSeekGitHubAgent(config)
    
    try:
        # Initialize agent
        await agent.initialize()
        
        # Execute example task
        task = {
            "type": "code_generation",
            "description": "Implement a simple calculator function",
            "requirements": [
                "Support basic arithmetic operations",
                "Handle division by zero",
                "Return results with proper type"
            ]
        }
        
        result = await agent.execute_task(task)
        print(f"\nTask execution result: {json.dumps(result, indent=2)}")
        
    finally:
        # Cleanup
        await agent.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 