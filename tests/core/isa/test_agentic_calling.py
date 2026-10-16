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

class TestValidateContext:
    """Pre-execution context validation (paper Section 3.2.3)."""

    def test_single_argument_constraint_applied_directly(self):
        t = AgenticTransformer()
        t.add_context_constraint("budget", lambda v: v >= 0)
        assert t.validate_context({"budget": 10}) is True
        assert t.validate_context({"budget": -1}) is False

    def test_pairwise_constraint_applied_reflexively(self):
        t = AgenticTransformer()
        # A pairwise predicate is applied reflexively -- (v, v) -- so a
        # constraint that inspects its second argument decides the verdict.
        t.add_context_constraint("level", lambda before, after: after >= 0)
        assert t.validate_context({"level": 5}) is True
        assert t.validate_context({"level": -3}) is False

    def test_reflexive_identity_constraint_always_holds(self):
        t = AgenticTransformer()
        # (v, v) trivially satisfies a symmetric relation like after >= before,
        # so reflexive validation passes for every value -- by definition.
        t.add_context_constraint("counter", lambda before, after: after >= before)
        assert t.validate_context({"counter": -3}) is True
        assert t.validate_context({"counter": 7}) is True

    def test_missing_key_vacuously_satisfied(self):
        t = AgenticTransformer()
        t.add_context_constraint("absent", lambda v: False)
        assert t.validate_context({"other": 1}) is True

    def test_multiple_constraints_all_must_hold(self):
        t = AgenticTransformer()
        t.add_context_constraint("a", lambda v: v > 0)
        t.add_context_constraint("b", lambda v: v < 10)
        assert t.validate_context({"a": 1, "b": 2}) is True
        assert t.validate_context({"a": 1, "b": 20}) is False
        assert t.validate_context({"a": -1, "b": 2}) is False

    def test_non_callable_constraint_fails_validation(self):
        t = AgenticTransformer()
        t.add_context_constraint("x", "not-a-callable")
        assert t.validate_context({"x": 1}) is False

    def test_consistent_with_preservation_semantics(self):
        t = AgenticTransformer()
        t.add_context_constraint("monotone", lambda before, after: after >= before)
        # A state that fails reflexive validation must also fail the
        # preservation check against itself, and vice versa.
        assert t.validate_context({"monotone": 1}) is True
        assert t.verify_context_preservation({"monotone": 1}, {"monotone": 1}) is True


class TestNormalizeResponse:
    """Response normalization (paper Section 3.2.3)."""

    def test_anthropic_protocol_shape(self):
        adapter = ModelAdapter("gateway-model")
        raw = {
            "model": "glm-5.3",
            "content": [
                {"type": "text", "text": "hello"},
                {"type": "tool_use", "id": "t1", "name": "f"},
            ],
            "usage": {"input_tokens": 12, "output_tokens": 34},
        }
        resp = adapter.normalize_response(raw)
        assert resp.content == "hello"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0]["name"] == "f"
        assert resp.usage["input_tokens"] == 12
        assert resp.usage["output_tokens"] == 34
        assert resp.model == "glm-5.3"
        assert resp.raw is raw

    def test_openai_protocol_shape(self):
        adapter = ModelAdapter("gateway-model")
        raw = {
            "choices": [
                {"message": {"content": "hi", "tool_calls": [{"id": "c1"}]}}
            ],
            "usage": {"prompt_tokens": 7, "completion_tokens": 9},
            "model": "gpt-x",
        }
        resp = adapter.normalize_response(raw)
        assert resp.content == "hi"
        assert resp.tool_calls == [{"id": "c1"}]
        assert resp.usage["input_tokens"] == 7
        assert resp.usage["output_tokens"] == 9
        assert resp.model == "gpt-x"

    def test_common_key_passthrough(self):
        adapter = ModelAdapter("gateway-model")
        resp = adapter.normalize_response({"text": "plain", "model": "m"})
        assert resp.content == "plain"
        assert resp.model == "m"
        assert resp.usage == {}

    def test_non_dict_passthrough(self):
        adapter = ModelAdapter("gateway-model")
        resp = adapter.normalize_response("bare string")
        assert resp.content == "bare string"
        assert resp.model == "gateway-model"
        assert resp.raw == "bare string"
