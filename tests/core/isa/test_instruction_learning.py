"""Unit tests for instruction learning system."""

import pytest
from typing import Dict, Any, List
import numpy as np
from agentflow.core.isa.instruction_learning import (
    PatternRecognizer,
    InstructionLearner,
    DialoguePattern,
    InstructionTemplate
)
from agentflow.core.isa.agentic_calling import AgenticTransformer
from agentflow.core.isa.instruction import InstructionType

@pytest.fixture
def sample_dialogues() -> List[Dict[str, Any]]:
    """Sample dialogues for testing."""
    return [
        {
            "turns": [
                {"role": "system", "content": "Execute test_function"},
                {"role": "user", "content": '{"param1": "test1", "param2": 42}'},
                {"role": "assistant", "content": '{"result": "success1"}'}
            ],
            "success": True,
            "context": {"param1": "test1", "param2": 42}
        },
        {
            "turns": [
                {"role": "system", "content": "Execute test_function"},
                {"role": "user", "content": '{"param1": "test2", "param2": 43}'},
                {"role": "assistant", "content": '{"result": "success2"}'}
            ],
            "success": True,
            "context": {"param1": "test2", "param2": 43}
        },
        {
            "turns": [
                {"role": "system", "content": "Execute other_function"},
                {"role": "user", "content": '{"param3": "test3"}'},
                {"role": "assistant", "content": '{"result": "success3"}'}
            ],
            "success": True,
            "context": {"param3": "test3"}
        }
    ]

@pytest.fixture
def pattern_recognizer():
    return PatternRecognizer()

@pytest.fixture
def instruction_learner():
    transformer = AgenticTransformer()
    return InstructionLearner(transformer)

def test_dialogue_to_text(pattern_recognizer, sample_dialogues):
    """Test dialogue to text conversion."""
    text = pattern_recognizer._dialogue_to_text(sample_dialogues[0])
    assert isinstance(text, str)
    assert "system: Execute test_function" in text
    assert "user:" in text
    assert "assistant:" in text

def test_pattern_extraction(pattern_recognizer, sample_dialogues):
    """Test pattern extraction from dialogues."""
    patterns = pattern_recognizer.extract_patterns(sample_dialogues)
    
    assert isinstance(patterns, list)
    assert len(patterns) > 0
    
    # Verify pattern structure
    pattern = patterns[0]
    assert isinstance(pattern, DialoguePattern)
    assert pattern.pattern_id.startswith("pattern_")
    assert pattern.frequency > 0
    assert pattern.success_rate > 0
    assert isinstance(pattern.context_requirements, set)

def test_common_components_extraction(pattern_recognizer, sample_dialogues):
    """Test extraction of common components."""
    components = pattern_recognizer._find_common_components(sample_dialogues[:2])
    assert isinstance(components, list)
    assert len(components) > 0
    assert any("test_function" in comp for comp in components)

def test_context_requirements_extraction(pattern_recognizer, sample_dialogues):
    """Test extraction of context requirements."""
    context_reqs = pattern_recognizer._extract_context_requirements(sample_dialogues[:2])
    assert isinstance(context_reqs, set)
    assert "param1" in context_reqs
    assert "param2" in context_reqs

@pytest.mark.asyncio
async def test_instruction_learning(instruction_learner, sample_dialogues):
    """Test instruction learning from dialogues."""
    # Learn from dialogues
    instruction_learner.learn_from_dialogues(sample_dialogues)
    
    # Verify templates were created
    assert len(instruction_learner.instruction_templates) > 0
    
    # Get a template
    template = next(iter(instruction_learner.instruction_templates.values()))
    assert isinstance(template, InstructionTemplate)
    
    # Create instruction from template
    instruction = instruction_learner.create_instruction(template)
    assert instruction.type == InstructionType.LLM_INTERACTION
    assert instruction.name.startswith("learned_")

@pytest.mark.asyncio
async def test_pattern_to_template_conversion(instruction_learner):
    """Test conversion of pattern to instruction template."""
    pattern = DialoguePattern(
        pattern_id="test_pattern",
        components=["Execute test_function", '{"param1": "test"}'],
        frequency=2,
        success_rate=1.0,
        context_requirements={"param1"}
    )
    
    template = instruction_learner._pattern_to_template(pattern)
    assert isinstance(template, InstructionTemplate)
    assert template.name == "learned_test_pattern"
    assert template.type == InstructionType.LLM_INTERACTION
    assert "param1" in template.preconditions[0]

@pytest.mark.asyncio
async def test_learned_instruction_execution(instruction_learner, sample_dialogues):
    """Test execution of learned instructions."""
    # Learn from dialogues
    instruction_learner.learn_from_dialogues(sample_dialogues)
    
    # Get learned instruction
    template = next(iter(instruction_learner.instruction_templates.values()))
    instruction = instruction_learner.create_instruction(template)
    
    # Execute instruction
    params = {"param1": "test", "param2": 42}
    result = await instruction.execute(params)
    assert isinstance(result, dict)
    assert "status" in result
    assert "params" in result

def test_pattern_recognition_thresholds(pattern_recognizer):
    """Test pattern recognition thresholds."""
    # Test with insufficient support
    insufficient_dialogues = [
        {
            "turns": [
                {"role": "system", "content": "Execute test_function"},
                {"role": "user", "content": '{"param": "test"}'},
                {"role": "assistant", "content": '{"result": "success"}'}
            ],
            "success": True,
            "context": {"param": "test"}
        }
    ]
    
    patterns = pattern_recognizer.extract_patterns(insufficient_dialogues)
    assert len(patterns) == 0
    
    # Test with low success rate
    low_success_dialogues = [
        {
            "turns": [
                {"role": "system", "content": "Execute test_function"},
                {"role": "user", "content": '{"param": "test"}'},
                {"role": "assistant", "content": '{"result": "success"}'}
            ],
            "success": False,
            "context": {"param": "test"}
        } for _ in range(5)
    ]
    
    patterns = pattern_recognizer.extract_patterns(low_success_dialogues)
    assert len(patterns) == 0

@pytest.mark.asyncio
async def test_instruction_template_validation(instruction_learner):
    """Test instruction template validation."""
    # Test with invalid template
    invalid_template = InstructionTemplate(
        name="",  # Invalid empty name
        type=InstructionType.LLM_INTERACTION,
        params={},
        preconditions=[],
        postconditions=[]
    )
    
    with pytest.raises(ValueError):
        instruction_learner.create_instruction(invalid_template)
    
    # Test with valid template
    valid_template = InstructionTemplate(
        name="valid_template",
        type=InstructionType.LLM_INTERACTION,
        params={"param1": str},
        preconditions=["requires_param1"],
        postconditions=[]
    )
    
    instruction = instruction_learner.create_instruction(valid_template)
    assert instruction.name == "valid_template"
    assert instruction.type == InstructionType.LLM_INTERACTION 