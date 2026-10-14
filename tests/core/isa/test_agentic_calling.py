"""Unit tests for Agentic Calling framework."""

import pytest
from typing import Dict, Any
from inspect import Parameter, Signature
from agentflow.core.isa.agentic_calling import (
    AgenticTransformer,
    ModelAdapter,
    FunctionSignature
)
from agentflow.core.isa.instruction import Instruction, InstructionType

async def sample_typed_function(param1: str, param2: int = 42) -> Dict[str, Any]:
    """Test function for transformation."""
    return {"param1": param1, "param2": param2}

@pytest.fixture
def transformer():
    return AgenticTransformer()

@pytest.fixture
def model_adapter():
    return ModelAdapter("test_model")

@pytest.mark.asyncio
async def test_function_signature_extraction(transformer):
    """Test function signature extraction."""
    signature = transformer.extract_signature(sample_typed_function)
    
    assert isinstance(signature, FunctionSignature)
    assert signature.name == "sample_typed_function"
    assert len(signature.parameters) == 2
    assert "param1" in signature.parameters
    assert "param2" in signature.parameters
    assert signature.return_type == Dict[str, Any]
    assert signature.docstring == "Test function for transformation."

@pytest.mark.asyncio
async def test_function_transformation(transformer):
    """Test function to instruction transformation."""
    # Transform function
    instruction = transformer.transform_to_instruction(sample_typed_function)
    
    assert isinstance(instruction, Instruction)
    assert instruction.id.startswith("func_sample_typed_function")
    assert instruction.type == InstructionType.LLM_INTERACTION
    
    # Test execution
    params = {"param1": "test", "param2": 42}
    result = await instruction.execute(params)
    assert result == params

@pytest.mark.asyncio
async def test_context_preservation(transformer):
    """Test context preservation in transformation."""
    # Add context constraint
    def preserve_param1(before: str, after: str) -> bool:
        return len(after) >= len(before)
        
    transformer.add_context_constraint("param1", preserve_param1)
    
    # Test context preservation
    before_state = {"param1": "test"}
    after_state = {"param1": "test_extended"}
    assert transformer.verify_context_preservation(before_state, after_state)
    
    # Test context violation
    invalid_after_state = {"param1": "t"}
    assert not transformer.verify_context_preservation(before_state, invalid_after_state)

@pytest.mark.asyncio
async def test_model_adapter():
    """Test model adapter functionality."""
    adapter = ModelAdapter("test_model")
    
    # Register type transformation
    def transform_int_to_string(value: int) -> str:
        return str(value)
        
    adapter.register_transformation("int", "string", transform_int_to_string)
    
    # Create test instruction
    async def test_func(value: int) -> int:
        return value * 2
        
    transformer = AgenticTransformer()
    instruction = transformer.transform_to_instruction(test_func)
    
    # Test instruction translation
    translated = adapter.translate_instruction(instruction)
    assert isinstance(translated, dict)
    assert "name" in translated
    assert "type" in translated
    assert "parameters" in translated

@pytest.mark.asyncio
async def test_instruction_execution_with_context(transformer):
    """Test instruction execution with context tracking."""
    # Transform function with context requirements
    context_requirements = {
        "param1": lambda x: isinstance(x, str) and len(x) > 0
    }
    
    instruction = transformer.transform_to_instruction(
        sample_typed_function,
        context_requirements
    )
    
    # Execute with valid context
    params = {"param1": "test", "param2": 42}
    result = await instruction.execute(params)
    
    # Verify context updates
    assert "param1" in transformer.context.state
    assert transformer.context.state["param1"] == "test"
    assert len(transformer.context.history) == 1
    
    # Verify history entry
    history_entry = transformer.context.history[0]
    assert history_entry["function"] == "sample_typed_function"
    assert history_entry["params"] == params
    assert history_entry["result"] == result

@pytest.mark.asyncio
async def test_error_handling(transformer):
    """Test error handling in transformation and execution."""
    # Test with invalid function (missing type hints)
    async def invalid_function(param):
        return param
        
    with pytest.raises(Exception):
        transformer.transform_to_instruction(invalid_function)
    
    # Test with invalid parameters
    instruction = transformer.transform_to_instruction(sample_typed_function)
    with pytest.raises(TypeError):
        await instruction.execute({"param1": 42, "param2": "invalid"})

@pytest.mark.asyncio
async def test_model_adapter_type_handling():
    """Test model adapter type handling."""
    adapter = ModelAdapter("test_model")
    
    # Test basic type transformation
    assert adapter._transform_type(int) == "int"
    assert adapter._transform_type(str) == "str"
    
    # Test with generic types
    assert adapter._transform_type(Dict[str, Any]) == "Dict[str, Any]"
    
    # Test custom type transformation
    class CustomType:
        pass
        
    assert adapter._transform_type(CustomType) == "CustomType" 