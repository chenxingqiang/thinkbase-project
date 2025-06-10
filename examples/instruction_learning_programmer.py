"""
Instruction Learning Programmer Agent Example

This example demonstrates how to create a programmer agent that uses instruction learning
to improve its capabilities over time.
"""

from typing import Dict, List, Any, Optional, Union, cast
from dataclasses import dataclass, field
import asyncio
import json
import logging
from pathlib import Path
from enum import Enum, auto
import time

from agentflow.agents.base import Agent
from agentflow.core.isa.instruction_learning import (
    InstructionLearner,
    DialoguePattern,
    InstructionTemplate
)
from agentflow.core.isa.instruction import Instruction
from agentflow.core.context.context_manager import ContextManager
from agentflow.core.isa.agentic_calling import AgenticTransformer, AgenticConfig, ModelAdapter

logger = logging.getLogger(__name__)

class InstructionType(Enum):
    """Instruction types."""
    CODE_REVIEW = auto()
    REFACTORING = auto()
    CODE_GENERATION = auto()
    LLM_INTERACTION = auto()

@dataclass
class ProviderConfig:
    """Configuration for the model provider."""
    model: str = "deepseek-coder-33b-instruct"
    temperature: float = 0.7
    max_tokens: int = 4096
    api_key: str = ""
    api_base: str = "https://api.deepseek.com/v1"

@dataclass
class InstructionConfig:
    """Configuration for instruction execution."""
    enabled: bool = True
    max_retries: int = 3
    retry_delay: float = 1.0

@dataclass
class MetricsConfig:
    """Configuration for metrics tracking."""
    enabled: bool = True
    track_tokens: bool = True
    track_latency: bool = True
    track_success_rate: bool = True
    track_errors: bool = True

@dataclass
class AgentConfiguration:
    """Complete agent configuration."""
    id: str
    name: str
    description: str
    version: str = "1.0.0"
    type: str = "programmer"
    author: str = ""
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    instructions: Dict[str, InstructionConfig] = field(default_factory=lambda: {
        "code_review": InstructionConfig(),
        "code_generation": InstructionConfig(),
        "refactoring": InstructionConfig()
    })
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfiguration':
        """Create configuration from dictionary."""
        provider_data = data.get("provider", {})
        provider = ProviderConfig(
            model=provider_data.get("model", ProviderConfig.model),
            temperature=provider_data.get("temperature", ProviderConfig.temperature),
            max_tokens=provider_data.get("max_tokens", ProviderConfig.max_tokens),
            api_key=provider_data.get("api_key", ProviderConfig.api_key),
            api_base=provider_data.get("api_base", ProviderConfig.api_base)
        )
        
        instructions_data = data.get("instructions", {})
        instructions = {}
        for name, instr_data in instructions_data.items():
            if isinstance(instr_data, dict):
                instructions[name] = InstructionConfig(
                    enabled=instr_data.get("enabled", True),
                    max_retries=instr_data.get("max_retries", 3),
                    retry_delay=instr_data.get("retry_delay", 1.0)
                )
        
        metrics_data = data.get("metrics", {})
        metrics = MetricsConfig(
            enabled=metrics_data.get("enabled", True),
            track_tokens=metrics_data.get("track_tokens", True),
            track_latency=metrics_data.get("track_latency", True),
            track_success_rate=metrics_data.get("track_success_rate", True),
            track_errors=metrics_data.get("track_errors", True)
        )
        
        return cls(
            id=data.get("id", "instruction-learning-programmer"),
            name=data.get("name", "Instruction Learning Programmer"),
            description=data.get("description", "AI agent that learns from programming interactions"),
            version=data.get("version", "1.0.0"),
            type=data.get("type", "programmer"),
            author=data.get("author", ""),
            provider=provider,
            instructions=instructions,
            metrics=metrics
        )

@dataclass
class CodeReviewResult:
    """Results from a code review."""
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    metrics: Dict[str, float]

@dataclass
class RefactoringPlan:
    """Plan for code refactoring."""
    changes: List[Dict[str, Any]]
    rationale: str
    impact_analysis: Dict[str, Any]

class ProgrammerAgent(Agent):
    """An agent that can perform programming tasks with instruction learning capabilities."""
    
    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        transformer: Optional[AgenticTransformer] = None,
        context_manager: Optional[ContextManager] = None
    ):
        """Initialize the programmer agent.
        
        Args:
            config_path: Path to configuration file
            transformer: Optional transformer for language processing
            context_manager: Optional context manager
        """
        # Load configuration
        if config_path:
            with open(config_path) as f:
                config_data = json.load(f)
        else:
            config_data = {
                "id": "instruction-learning-programmer",
                "name": "Instruction Learning Programmer",
                "description": "AI agent that learns from programming interactions"
            }
        
        # Create agent configuration
        self.agent_config = AgentConfiguration.from_dict(config_data)
        
        # Initialize base agent
        super().__init__(name=self.agent_config.name)
        
        # Set up components
        self.transformer = transformer or AgenticTransformer(config=AgenticConfig(
            enable_caching=True,
            cache_size=1000,
            cache_ttl=3600,
            enable_parallel=True,
            parallel_batch_size=10,
            max_parallel_tasks=100
        ))
        self.context_manager = context_manager or ContextManager()
        
        # Configure transformer if not provided
        if not transformer and self.agent_config.provider:
            # Create model adapter for the provider's model
            model_adapter = ModelAdapter(self.agent_config.provider.model)
            # Register model-specific transformations
            model_adapter.register_transformation(
                "str", "string",
                lambda x: str(x)
            )
            # Use the adapter to translate instructions as needed
            # The adapter will be used internally by the transformer
            # when executing instructions
        
        # Initialize instruction learner
        self.instruction_learner = InstructionLearner(self.transformer)
        
        # Track learned patterns
        self.learned_patterns: Dict[str, DialoguePattern] = {}
        self.instruction_templates: Dict[str, InstructionTemplate] = {}
        
        # Initialize metrics if enabled
        if self.agent_config.metrics.enabled:
            self.metrics = {
                "code_reviews": 0,
                "refactorings": 0,
                "successful_builds": 0,
                "failed_builds": 0,
                "total_tokens": 0,
                "total_latency": 0.0,
                "success_rate": 0.0,
                "errors": []
            }

    def _update_metrics(self, operation: str, success: bool, tokens: int = 0, latency: float = 0.0, error: Optional[str] = None):
        """Update agent metrics."""
        if not hasattr(self, "metrics") or not self.agent_config.metrics.enabled:
            return
            
        if operation in self.metrics:
            self.metrics[operation] += 1
            
        if self.agent_config.metrics.track_tokens:
            self.metrics["total_tokens"] += tokens
            
        if self.agent_config.metrics.track_latency:
            self.metrics["total_latency"] += latency
            
        if self.agent_config.metrics.track_success_rate:
            total_ops = sum(v for k, v in self.metrics.items() if k in ["code_reviews", "refactorings"])
            successful = sum(v for k, v in self.metrics.items() if k == "successful_builds")
            self.metrics["success_rate"] = successful / total_ops if total_ops > 0 else 0.0
            
        if error and self.agent_config.metrics.track_errors:
            self.metrics["errors"].append({
                "operation": operation,
                "error": str(error),
                "timestamp": time.time()
            })

    async def learn_from_experience(self, dialogues: List[Dict[str, Any]]) -> List[InstructionTemplate]:
        """Learn new programming patterns from past dialogues.
        
        Args:
            dialogues: List of dialogue dictionaries containing programming interactions
            
        Returns:
            List[InstructionTemplate]: Newly learned instruction templates
        """
        try:
            # Extract patterns and create templates
            templates = await self.instruction_learner.learn_from_dialogues(dialogues)
            
            # Store templates
            for template in templates:
                self.instruction_templates[template.name] = template
            
            logger.info(f"Learned {len(templates)} new instruction templates")
            return templates
            
        except Exception as e:
            logger.error(f"Error learning from experience: {str(e)}")
            return []

    async def review_code(self, code: Union[str, Path]) -> CodeReviewResult:
        """Perform a code review.
        
        Args:
            code: Code string or path to code file
            
        Returns:
            CodeReviewResult: Results of the code review
        """
        try:
            # Load code if path provided
            if isinstance(code, Path):
                code = code.read_text()
            
            # Initialize review results
            review = CodeReviewResult(
                issues=[],
                suggestions=[],
                metrics={}
            )
            
            # Apply learned patterns to identify issues
            for template in self.instruction_templates.values():
                if template.type == InstructionType.CODE_REVIEW:
                    instruction = self.instruction_learner.create_instruction(template)
                    result = await instruction.execute({
                        "code": code,
                        "context": self.context_manager.get_context()
                    })
                    
                    if result.get("issues"):
                        review.issues.extend(result["issues"])
                    if result.get("suggestions"):
                        review.suggestions.extend(result["suggestions"])
            
            # Update metrics
            self._update_metrics("code_reviews", True)
            
            return review
            
        except Exception as e:
            logger.error(f"Error during code review: {str(e)}")
            self._update_metrics("code_reviews", False)
            return CodeReviewResult([], [], {})

    async def plan_refactoring(self, code: Union[str, Path]) -> RefactoringPlan:
        """Plan code refactoring based on learned patterns.
        
        Args:
            code: Code string or path to code file
            
        Returns:
            RefactoringPlan: Planned refactoring changes
        """
        try:
            # Load code if path provided
            if isinstance(code, Path):
                code = code.read_text()
            
            # Initialize refactoring plan
            plan = RefactoringPlan(
                changes=[],
                rationale="",
                impact_analysis={}
            )
            
            # Apply learned patterns to identify refactoring opportunities
            for template in self.instruction_templates.values():
                if template.type == InstructionType.REFACTORING:
                    instruction = self.instruction_learner.create_instruction(template)
                    result = await instruction.execute({
                        "code": code,
                        "context": self.context_manager.get_context()
                    })
                    
                    if result.get("changes"):
                        plan.changes.extend(result["changes"])
                    if result.get("rationale"):
                        plan.rationale = result["rationale"]
                    if result.get("impact_analysis"):
                        plan.impact_analysis.update(result["impact_analysis"])
            
            # Update metrics
            self._update_metrics("refactorings", True)
            
            return plan
            
        except Exception as e:
            logger.error(f"Error planning refactoring: {str(e)}")
            self._update_metrics("refactorings", False)
            return RefactoringPlan([], "", {})

    async def implement_changes(self, plan: RefactoringPlan) -> bool:
        """Implement planned refactoring changes.
        
        Args:
            plan: Refactoring plan to implement
            
        Returns:
            bool: True if changes were implemented successfully
        """
        try:
            success = True
            for change in plan.changes:
                # Apply each change
                if change.get("type") == "modify":
                    # Modify existing code
                    file_path = Path(change["file"])
                    if file_path.exists():
                        code = file_path.read_text()
                        modified = self._apply_modification(code, change)
                        file_path.write_text(modified)
                elif change.get("type") == "create":
                    # Create new file
                    file_path = Path(change["file"])
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change["content"])
                elif change.get("type") == "delete":
                    # Delete file
                    file_path = Path(change["file"])
                    if file_path.exists():
                        file_path.unlink()
            
            # Update metrics
            if success:
                self._update_metrics("successful_builds", True)
            else:
                self._update_metrics("failed_builds", True)
            
            return success
            
        except Exception as e:
            logger.error(f"Error implementing changes: {str(e)}")
            self._update_metrics("failed_builds", True)
            return False

    def _apply_modification(self, code: str, change: Dict[str, Any]) -> str:
        """Apply a code modification.
        
        Args:
            code: Original code string
            change: Change specification
            
        Returns:
            str: Modified code
        """
        try:
            if change.get("operation") == "replace":
                # Replace specified range
                start = change["start"]
                end = change["end"]
                new_code = change["code"]
                
                lines = code.splitlines()
                lines[start:end] = new_code.splitlines()
                return "\n".join(lines)
                
            elif change.get("operation") == "insert":
                # Insert at specified position
                position = change["position"]
                new_code = change["code"]
                
                lines = code.splitlines()
                lines.insert(position, new_code)
                return "\n".join(lines)
                
            elif change.get("operation") == "delete":
                # Delete specified range
                start = change["start"]
                end = change["end"]
                
                lines = code.splitlines()
                del lines[start:end]
                return "\n".join(lines)
                
            return code
            
        except Exception as e:
            logger.error(f"Error applying modification: {str(e)}")
            return code

async def main():
    """Example usage of the ProgrammerAgent."""
    # Initialize agent with config
    agent = ProgrammerAgent("examples/deepseek_github_agent_config.json")
    
    # Example code to review
    example_code = """
def calculate_sum(numbers):
    sum = 0
    for n in numbers:
        sum += n
    return sum
    
def calculate_average(numbers):
    if len(numbers) == 0:
        return 0
    return calculate_sum(numbers) / len(numbers)
"""
    
    # Learn from example dialogues
    example_dialogues = [
        {
            "turns": [
                {"role": "user", "content": "Please review this code for best practices"},
                {"role": "assistant", "content": "I suggest adding type hints and docstrings"}
            ],
            "success": True
        }
    ]
    
    # Learn patterns
    templates = await agent.learn_from_experience(example_dialogues)
    print(f"Learned {len(templates)} templates")
    
    # Perform code review
    review = await agent.review_code(example_code)
    print("\nCode Review Results:")
    print(f"Issues found: {len(review.issues)}")
    print(f"Suggestions: {len(review.suggestions)}")
    
    # Plan refactoring
    plan = await agent.plan_refactoring(example_code)
    print("\nRefactoring Plan:")
    print(f"Planned changes: {len(plan.changes)}")
    print(f"Rationale: {plan.rationale}")
    
    # Implement changes
    success = await agent.implement_changes(plan)
    print(f"\nChanges implemented: {success}")
    
    # Print metrics
    if agent.agent_config.metrics.enabled:
        print("\nAgent Metrics:")
        for metric, value in agent.metrics.items():
            if metric != "errors":
                print(f"{metric}: {value}")
        if agent.metrics["errors"]:
            print(f"\nErrors: {len(agent.metrics['errors'])}")

if __name__ == "__main__":
    asyncio.run(main()) 