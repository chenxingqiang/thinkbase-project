"""Tests for the instruction learning module."""
import pytest
from typing import List, Dict, Any
from ..transformer import AgenticTransformer
from .instruction_learner import InstructionLearner, DialoguePattern, InstructionTemplate

pytestmark = pytest.mark.asyncio

@pytest.fixture
def transformer():
    return AgenticTransformer()

@pytest.fixture
def instruction_learner(transformer):
    return InstructionLearner(transformer)

@pytest.fixture
def example_dialogues():
    return [
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
        }
    ]

async def test_learn_from_dialogues(instruction_learner, example_dialogues):
    """Test learning from example dialogues."""
    templates = await instruction_learner.learn_from_dialogues(example_dialogues)
    
    assert len(templates) > 0, "Should learn at least one template"
    assert all(isinstance(t, InstructionTemplate) for t in templates), "All items should be InstructionTemplate"
    
    template = templates[0]
    assert template.type == "code_review"
    assert template.pattern.context["type"] == "code_review"
    assert template.pattern.confidence > 0
    
    # Verify pattern extraction
    assert "calculate_factorial" in template.pattern.input_pattern
    assert "type hints" in template.pattern.output_pattern.lower()

async def test_create_instruction(instruction_learner, example_dialogues):
    """Test creating instruction from template."""
    templates = await instruction_learner.learn_from_dialogues(example_dialogues)
    instruction = instruction_learner.create_instruction(templates[0])
    
    assert instruction is not None
    
    # Test instruction execution
    result = await instruction.execute({
        "code": """
def sort_list(items):
    for i in range(len(items)):
        for j in range(len(items)-1):
            if items[j] > items[j+1]:
                items[j], items[j+1] = items[j+1], items[j]
    return items
""",
        "context": {}
    })
    
    assert isinstance(result, dict)
    assert "issues" in result
    assert "suggestions" in result

async def test_empty_dialogues(instruction_learner):
    """Test handling empty dialogues."""
    templates = await instruction_learner.learn_from_dialogues([])
    assert len(templates) == 0

async def test_invalid_dialogue_format(instruction_learner):
    """Test handling invalid dialogue format."""
    invalid_dialogues = [{"invalid": "format"}]
    templates = await instruction_learner.learn_from_dialogues(invalid_dialogues)
    assert len(templates) == 0

async def test_pattern_confidence(instruction_learner, example_dialogues):
    """Test pattern confidence calculation."""
    templates = await instruction_learner.learn_from_dialogues(example_dialogues)
    assert templates[0].pattern.confidence > 0
    assert templates[0].pattern.confidence <= 1
