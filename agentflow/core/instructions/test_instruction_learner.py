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
    
    # Verify input pattern extraction
    input_pattern = template.pattern.input_pattern.lower()
    assert "calculate_factorial" in input_pattern, "Should preserve function name"
    assert "recursive" in input_pattern, "Should preserve context"
    assert "return" in input_pattern, "Should preserve code structure"
    
    # Verify output pattern extraction
    output_pattern = template.pattern.output_pattern.lower()
    assert "error_handling" in output_pattern, "Should preserve issue type"
    assert "type hints" in output_pattern, "Should preserve issue description"
    assert "improvement" in output_pattern, "Should preserve suggestion type"
    assert "iterative approach" in output_pattern, "Should preserve suggestion description"

async def test_empty_dialogues(instruction_learner):
    """Test handling empty dialogues."""
    templates = await instruction_learner.learn_from_dialogues([])
    assert len(templates) == 0

async def test_invalid_dialogue_format(instruction_learner):
    """Test handling invalid dialogue format."""
    invalid_dialogues = [{"invalid": "format"}]
    templates = await instruction_learner.learn_from_dialogues(invalid_dialogues)
    assert len(templates) == 0

async def test_missing_required_fields(instruction_learner):
    """Test handling dialogues with missing required fields."""
    incomplete_dialogues = [
        {"type": "code_review"},  # Missing input and output
        {"input": {}, "output": {}},  # Missing type
        {"type": "code_review", "input": {}}  # Missing output
    ]
    for dialogue in incomplete_dialogues:
        templates = await instruction_learner.learn_from_dialogues([dialogue])
        assert len(templates) == 0, f"Should reject dialogue missing fields: {dialogue}"

async def test_pattern_confidence(instruction_learner, example_dialogues):
    """Test pattern confidence calculation."""
    templates = await instruction_learner.learn_from_dialogues(example_dialogues)
    assert templates[0].pattern.confidence > 0
    assert templates[0].pattern.confidence <= 1

async def test_multiple_dialogues(instruction_learner):
    """Test learning from multiple dialogues."""
    dialogues = [
        {
            "type": "code_review",
            "input": {"code": "def foo(): pass", "context": "Simple function"},
            "output": {"issues": [], "suggestions": []}
        },
        {
            "type": "refactoring",
            "input": {"code": "x = 1", "context": "Variable assignment"},
            "output": {"changes": [], "rationale": "No changes needed"}
        }
    ]
    templates = await instruction_learner.learn_from_dialogues(dialogues)
    assert len(templates) == 2, "Should learn one template per valid dialogue"
    assert templates[0].type != templates[1].type, "Templates should have different types"

async def test_learn_from_refactoring_dialogue(instruction_learner):
    """Test learning from refactoring dialogue."""
    refactoring_dialogue = {
        "type": "refactoring",
        "input": {
            "code": """
class DataProcessor:
    def process_data(self, data):
        processed = []
        for item in data:
            if item is not None:
                processed.append(str(item))
        return processed
""",
            "context": "Refactoring data processing"
        },
        "output": {
            "changes": [
                {
                    "type": "extract_method",
                    "description": "Extract validation logic",
                    "code": """
def _validate_item(self, item):
    return str(item) if item is not None else None
"""
                }
            ],
            "rationale": "Improve code maintainability"
        }
    }
    
    templates = await instruction_learner.learn_from_dialogues([refactoring_dialogue])
    assert len(templates) == 1, "Should learn one template"
    
    template = templates[0]
    assert template.type == "refactoring"
    
    # Verify input pattern extraction
    input_pattern = template.pattern.input_pattern.lower()
    assert "dataprocessor" in input_pattern, "Should preserve class name"
    assert "process_data" in input_pattern, "Should preserve method name"
    assert "data processing" in input_pattern, "Should preserve context"
    
    # Verify output pattern extraction
    output_pattern = template.pattern.output_pattern.lower()
    assert "extract_method" in output_pattern, "Should preserve change type"
    assert "validation logic" in output_pattern, "Should preserve change description"
    assert "improve code maintainability" in output_pattern, "Should preserve rationale"

async def test_nested_dictionary_handling(instruction_learner):
    """Test handling of deeply nested dictionaries."""
    nested_dialogue = {
        "type": "code_review",
        "input": {
            "code": "def foo(): pass",
            "metadata": {
                "author": {
                    "name": "test",
                    "role": "developer"
                },
                "tags": ["python", "function"]
            }
        },
        "output": {
            "review": {
                "quality": {
                    "score": 8,
                    "details": {
                        "readability": "good",
                        "maintainability": "excellent"
                    }
                }
            }
        }
    }
    
    templates = await instruction_learner.learn_from_dialogues([nested_dialogue])
    assert len(templates) == 1, "Should learn one template"
    
    template = templates[0]
    # Verify nested input extraction
    input_pattern = template.pattern.input_pattern.lower()
    assert "author.name: test" in input_pattern, "Should handle nested author name"
    assert "author.role: developer" in input_pattern, "Should handle nested author role"
    assert "python" in input_pattern, "Should handle array items"
    
    # Verify nested output extraction
    output_pattern = template.pattern.output_pattern.lower()
    assert "quality.score: 8" in output_pattern, "Should handle nested score"
    assert "readability: good" in output_pattern, "Should handle nested quality details"
    assert "maintainability: excellent" in output_pattern, "Should handle nested quality details"
