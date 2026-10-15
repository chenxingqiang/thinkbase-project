"""Unit tests for instruction system."""

import pytest
from typing import Dict, Any
from agentflow.core.isa.instruction import (
    Instruction, 
    InstructionType,
    TypeConstraint,
    CompositeInstruction
)

class TestInstruction(Instruction):
    """Test instruction implementation."""
    
    def __init__(self, id: str = "test", name: str = "test"):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={
                "param1": TypeConstraint(base_type=str),
                "param2": TypeConstraint(base_type=int)
            },
            description="Test instruction"
        )
        self.executed = False
        self.last_params = None

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executed = True
        self.last_params = input_params
        return {"status": "executed", "params": input_params}

@pytest.fixture
def test_instruction():
    return TestInstruction()

@pytest.mark.asyncio
async def test_instruction_execution(test_instruction):
    """Test basic instruction execution."""
    params = {"param1": "test", "param2": 42}
    result = await test_instruction.execute(params)
    
    assert test_instruction.executed
    assert test_instruction.last_params == params
    assert result["status"] == "executed"
    assert result["params"] == params

@pytest.mark.asyncio
async def test_instruction_type_validation(test_instruction):
    """Test instruction parameter type validation."""
    # Valid parameters
    valid_params = {"param1": "test", "param2": 42}
    assert test_instruction.validate_types(valid_params)
    
    # Invalid parameters
    invalid_params = {"param1": 123, "param2": "test"}
    assert not test_instruction.validate_types(invalid_params)

def test_instruction_conditions(test_instruction):
    """Test instruction conditions."""
    # Add precondition
    test_instruction.add_precondition(lambda state: "required_key" in state)
    
    # Test with valid state
    valid_state = {"required_key": "value"}
    assert test_instruction.verify_conditions(valid_state)
    
    # Test with invalid state
    invalid_state = {"other_key": "value"}
    assert not test_instruction.verify_conditions(invalid_state)

@pytest.mark.asyncio
async def test_composite_instruction():
    """Test composite instruction."""
    # Create component instructions
    component1 = TestInstruction(id="comp1", name="component1")
    component2 = TestInstruction(id="comp2", name="component2")
    
    # Create composite instruction
    composite = CompositeInstruction(
        id="composite",
        name="composite_test",
        components=[component1, component2]
    )
    
    # Execute composite instruction
    params = {"param1": "test", "param2": 42}
    result = await composite.execute(params)
    
    # Verify both components executed
    assert component1.executed
    assert component2.executed
    assert component1.last_params == params
    assert component2.last_params == params
    assert result["status"] == "executed"

@pytest.mark.asyncio
async def test_composite_cache_serves_falsy_results():
    """A falsy-but-present cache entry must be served as a hit.

    The cache protocol uses ``None`` as its miss sentinel, so a cached
    result of ``0``, ``""``, or ``{}`` is a hit.  Gating on truthiness
    would silently re-execute (and re-charge) those results.
    """
    class FalsyInstruction(Instruction):
        def __init__(self):
            super().__init__(
                id="falsy", name="falsy", type=InstructionType.CONTROL_FLOW,
                params={}, description="returns 0",
            )
            self.executions = 0

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            self.executions += 1
            return 0

    inner = FalsyInstruction()
    composite = CompositeInstruction(
        id="c", name="c_falsy", components=[inner],
        cache_config={"max_size": 10, "ttl": 60},
    )

    first = await composite.execute({"x": 1})
    second = await composite.execute({"x": 1})

    assert first == 0
    assert second == 0
    assert inner.executions == 1  # second call served from cache
    assert composite.metrics.cache_hits == 1

@pytest.mark.asyncio
async def test_instruction_error_handling(test_instruction):
    """Test instruction error handling."""
    # Add a failing precondition
    test_instruction.add_precondition(lambda state: False)
    
    # Execute with invalid state should fail verification
    state = {"param1": "test", "param2": 42}
    assert not test_instruction.verify_conditions(state)
    
    # Execute with invalid types should fail validation
    invalid_params = {"param1": 123, "param2": "test"}
    assert not test_instruction.validate_types(invalid_params)

@pytest.mark.asyncio
async def test_instruction_context():
    """Test instruction context management."""
    instruction = TestInstruction()
    
    # Test context initialization
    assert instruction.context.state == {}
    assert instruction.context.type_env == {}
    assert instruction.context.preconditions == []
    assert instruction.context.postconditions == []
    
    # Test context updates
    instruction.context.state["test_key"] = "test_value"
    assert instruction.context.state["test_key"] == "test_value"
    
    # Test condition management
    def test_condition(state: Dict[str, Any]) -> bool:
        return "test_key" in state
        
    instruction.add_precondition(test_condition)
    assert len(instruction.context.preconditions) == 1
    assert instruction.verify_conditions({"test_key": "value"})
    assert not instruction.verify_conditions({})

@pytest.mark.asyncio
async def test_dependent_type_constraints():
    """Test dependent type constraints."""
    # Create instruction with dependent type constraint
    def validate_positive(value: int) -> bool:
        return value > 0
        
    instruction = TestInstruction()
    instruction.params["param2"] = TypeConstraint(
        base_type=int,
        dependent_on=validate_positive
    )
    
    # Test valid parameters
    valid_params = {"param1": "test", "param2": 42}
    assert instruction.validate_types(valid_params)
    
    # Test invalid parameters
    invalid_params = {"param1": "test", "param2": -1}
    assert not instruction.validate_types(invalid_params) 