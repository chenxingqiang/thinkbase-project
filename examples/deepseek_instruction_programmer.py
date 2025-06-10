'''
DeepSeek Instruction Programmer Agent Example

This example demonstrates how to create a programmer agent that uses instruction learning,
and integrates the configuration from deepseek_coding_agent_config.json.
'''

import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto
from pydantic import Field, BaseModel

from agentflow.agents.agent import Agent
from agentflow.agents.config import AgentConfig, AgentType
from agentflow.core.isa.instruction_learning import InstructionLearner, DialoguePattern, InstructionTemplate
from agentflow.core.isa.instruction import Instruction
from agentflow.core.context.context_manager import ContextManager
from agentflow.core.isa.agentic_calling import AgenticTransformer

logger = logging.getLogger(__name__)

class AgentRole(Enum):
    PROGRAMMER = auto()
    REVIEWER = auto()
    ARCHITECT = auto()

class AgentCapability(Enum):
    CODE_GENERATION = auto()
    CODE_REVIEW = auto()
    REFACTORING = auto()

class InstructionType(Enum):
    CODE_REVIEW = auto()
    REFACTORING = auto()
    LLM_INTERACTION = auto()

@dataclass
class CodeReviewResult:
    issues: List[Dict[str, Any]]
    suggestions: List[Dict[str, Any]]
    metrics: Dict[str, float]

@dataclass
class RefactoringPlan:
    changes: List[Dict[str, Any]]
    rationale: str
    impact_analysis: Dict[str, Any]

class DeepSeekInstructionProgrammerAgent(Agent, BaseModel):
    """DeepSeek Instruction Programmer Agent."""

    name: str = Field(default="deepseek_programmer")
    config: Optional[AgentConfig] = None
    transformer: Optional[AgenticTransformer] = Field(default=None)
    context_manager: Optional[ContextManager] = Field(default=None)
    instruction_learner: Optional[InstructionLearner] = Field(default=None)
    learned_patterns: Dict[str, DialoguePattern] = Field(default_factory=dict)
    instruction_templates: Dict[str, InstructionTemplate] = Field(default_factory=dict)
    metrics: Dict[str, Any] = Field(default_factory=lambda: {
        "code_reviews": 0,
        "refactorings": 0,
        "successful_builds": 0,
        "failed_builds": 0
    })
    json_config: Dict[str, Any] = Field(default_factory=dict)

    def __init__(
        self,
        name: str = "deepseek_programmer",
        transformer: Optional[AgenticTransformer] = None,
        context_manager: Optional[ContextManager] = None,
        config: Optional[AgentConfig] = None,
        config_file: str = "./deepseek_coding_agent_config.json"
    ):
        # Load configuration from the JSON config file
        json_config = {}
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with config_path.open("r") as f:
                    json_config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config file: {e}")
        else:
            logger.warning(f"Config file {config_file} does not exist.")

        # If no config is provided, create a default AgentConfig
        if config is None:
            config = AgentConfig(
                name=name,
                type=AgentType.GENERIC,
                description="DeepSeek Instruction Programmer Agent",
                model_provider="deepseek",
                model_name="deepseek-coder",
                temperature=0.7,
                max_tokens=2048
            )

        # Initialize base classes
        super().__init__(name=name, config=config)
        BaseModel.__init__(
            self,
            name=name,
            config=config,
            transformer=transformer or AgenticTransformer(),
            context_manager=context_manager or ContextManager({
                "max_tokens": 4000,
                "memory_window": 10,
                "system_prompt": "You are a DeepSeek Instruction Programmer Agent that helps with code generation, review, and refactoring."
            }),
            instruction_learner=InstructionLearner(transformer or AgenticTransformer()),
            json_config=json_config
        )

    async def learn_from_experience(self, dialogues: List[Dict[str, Any]]) -> List[InstructionTemplate]:
        """Learn from example dialogues.
        
        Args:
            dialogues: List of example dialogues
            
        Returns:
            List of learned instruction templates
        """
        try:
            # Validate input
            if not dialogues:
                logger.warning("No dialogues provided for learning")
                return []
                
            # Process dialogues
            templates = await self.instruction_learner.learn_from_dialogues(dialogues)
            if not templates:
                logger.warning("No templates learned from dialogues")
                return []
                
            # Store templates
            for template in templates:
                self.instruction_templates[template.name] = template
                if template.pattern:
                    self.learned_patterns[template.name] = template.pattern
                    
            logger.info(f"Learned {len(templates)} new instruction templates")
            return templates
            
        except Exception as e:
            logger.error(f"Error learning from experience: {str(e)}")
            raise

    async def review_code(self, code: Union[str, Path]) -> CodeReviewResult:
        try:
            if isinstance(code, Path):
                code = code.read_text()
            review = CodeReviewResult(issues=[], suggestions=[], metrics={})
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
            self.metrics["code_reviews"] += 1
            return review
        except Exception as e:
            logger.error(f"Error during code review: {str(e)}")
            return CodeReviewResult([], [], {})

    async def plan_refactoring(self, code: Union[str, Path]) -> RefactoringPlan:
        try:
            if isinstance(code, Path):
                code = code.read_text()
            plan = RefactoringPlan(changes=[], rationale="", impact_analysis={})
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
            self.metrics["refactorings"] += 1
            return plan
        except Exception as e:
            logger.error(f"Error planning refactoring: {str(e)}")
            return RefactoringPlan([], "", {})

    async def implement_changes(self, plan: RefactoringPlan) -> bool:
        try:
            success = True
            for change in plan.changes:
                if change.get("type") == "modify":
                    file_path = Path(change["file"])
                    if file_path.exists():
                        code = file_path.read_text()
                        modified = self._apply_modification(code, change)
                        file_path.write_text(modified)
                elif change.get("type") == "create":
                    file_path = Path(change["file"])
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_path.write_text(change["content"])
                elif change.get("type") == "delete":
                    file_path = Path(change["file"])
                    if file_path.exists():
                        file_path.unlink()
            if success:
                self.metrics["successful_builds"] += 1
            else:
                self.metrics["failed_builds"] += 1
            return success
        except Exception as e:
            logger.error(f"Error implementing changes: {str(e)}")
            self.metrics["failed_builds"] += 1
            return False

    def _apply_modification(self, code: str, change: Dict[str, Any]) -> str:
        try:
            if change.get("operation") == "replace":
                start = change["start"]
                end = change["end"]
                new_code = change["code"]
                lines = code.splitlines()
                lines[start:end] = new_code.splitlines()
                return "\n".join(lines)
            elif change.get("operation") == "insert":
                position = change["position"]
                new_code = change["code"]
                lines = code.splitlines()
                lines.insert(position, new_code)
                return "\n".join(lines)
            elif change.get("operation") == "delete":
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
    """Main function to run the example."""
    agent = DeepSeekInstructionProgrammerAgent(config_file="./deepseek_coding_agent_config.json")

    # Example dialogues for instruction learning
    example_dialogues = [
        {
            "type": "code_review",
            "input": {
                "code": """
def calculate_factorial(n):
    if n < 0:
        return None
    if n == 0:
        return 1
    return n * calculate_factorial(n-1)
""",
                "context": "Reviewing a recursive factorial function"
            },
            "output": {
                "issues": [
                    {"type": "error_handling", "description": "Missing type hints"},
                    {"type": "validation", "description": "No input validation for large numbers"}
                ],
                "suggestions": [
                    {"type": "improvement", "description": "Add type hints for better code clarity"},
                    {"type": "performance", "description": "Consider using iterative approach for large numbers"}
                ]
            }
        },
        {
            "type": "refactoring",
            "input": {
                "code": """
class DataProcessor:
    def process_data(self, data):
        # Process raw data
        processed = []
        for item in data:
            if item is not None:
                if isinstance(item, str):
                    processed.append(item.strip())
                elif isinstance(item, (int, float)):
                    processed.append(str(item))
        return processed
""",
                "context": "Refactoring a data processing class"
            },
            "output": {
                "changes": [
                    {
                        "type": "extract_method",
                        "description": "Extract data validation into separate method",
                        "code": """
def _validate_item(self, item):
    if item is None:
        return None
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, (int, float)):
        return str(item)
    return None
"""
                    }
                ],
                "rationale": "Improve code readability and maintainability by separating concerns"
            }
        }
    ]

    # Learn from example dialogues
    print("Learning from example dialogues...")
    learned_templates = await agent.learn_from_experience(example_dialogues)
    print(f"Learned {len(learned_templates)} templates")

    # Test code review with learned patterns
    test_code = """
def sort_list(items):
    for i in range(len(items)):
        for j in range(len(items)-1):
            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
    return items
"""
    print("\nTesting code review with learned patterns...")
    review_result = await agent.review_code(test_code)
    print("\nCode Review Results:")
    print(f"Issues found: {len(review_result.issues)}")
    print(f"Suggestions: {len(review_result.suggestions)}")

    # Test refactoring with learned patterns
    print("\nTesting refactoring with learned patterns...")
    refactoring_plan = await agent.plan_refactoring(test_code)
    print("\nRefactoring Plan:")
    print(f"Planned changes: {len(refactoring_plan.changes)}")
    print(f"Rationale: {refactoring_plan.rationale}")

    # Implement the refactoring changes
    print("\nImplementing refactoring changes...")
    success = await agent.implement_changes(refactoring_plan)
    print(f"Changes implemented: {success}")

    # Print final metrics
    print("\nAgent Metrics:")
    for metric, value in agent.metrics.items():
        print(f"{metric}: {value}")

    # Print loaded configuration
    print("\nLoaded Configuration:")
    print(json.dumps(agent.json_config, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
