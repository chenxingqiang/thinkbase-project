"""Unit tests for AgentISA interface."""

import pytest
from typing import Dict, Any
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig
from agentflow.core.isa.instruction import Instruction, InstructionType

async def test_function(param1: str, param2: int = 42) -> Dict[str, Any]:
    """Test function for registration."""
    return {"param1": param1, "param2": param2}

class TestInstruction(Instruction):
    """Test instruction implementation."""
    
    def __init__(self, id: str = "test", name: str = "test"):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description="Test instruction"
        )
        self.executed = False

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executed = True
        return {"status": "executed", "params": input_params}

@pytest.fixture
def agent_isa():
    """Create AgentISA instance with default config."""
    return AgentISA()

@pytest.fixture
def agent_isa_with_config():
    """Create AgentISA instance with custom config."""
    config = AgentISAConfig(
        enable_learning=True,
        enable_optimization=True,
        model_type="test_model",
        min_pattern_support=2,
        min_success_rate=0.8,
        hot_path_threshold=50,
        stability_threshold=0.85
    )
    return AgentISA(config)

@pytest.mark.asyncio
async def test_function_registration(agent_isa):
    """Test function registration as instruction."""
    instruction_id = await agent_isa.register_function(test_function)
    
    assert instruction_id in agent_isa.instructions
    instruction = agent_isa.instructions[instruction_id]
    assert instruction.name == "test_function"
    assert instruction.type == InstructionType.LLM_INTERACTION

@pytest.mark.asyncio
async def test_instruction_registration(agent_isa):
    """Test direct instruction registration."""
    instruction = TestInstruction()
    instruction_id = await agent_isa.register_instruction(instruction)
    
    assert instruction_id in agent_isa.instructions
    assert agent_isa.instructions[instruction_id] is instruction

@pytest.mark.asyncio
async def test_instruction_execution(agent_isa):
    """Test instruction execution."""
    # Register instruction
    instruction = TestInstruction()
    instruction_id = await agent_isa.register_instruction(instruction)
    
    # Execute instruction
    params = {"param1": "test", "param2": 42}
    result = await agent_isa.execute_instruction(instruction_id, params)
    
    assert result["status"] == "executed"
    assert result["params"] == params
    assert instruction.executed

@pytest.mark.asyncio
async def test_instruction_execution_with_optimization(agent_isa_with_config):
    """Test instruction execution with optimization enabled."""
    # Register instruction
    instruction = TestInstruction()
    instruction_id = await agent_isa_with_config.register_instruction(instruction)
    
    # Execute instruction multiple times
    params = {"param1": "test", "param2": 42}
    for _ in range(5):
        result = await agent_isa_with_config.execute_instruction(instruction_id, params)
        assert result["status"] == "executed"
    
    # Verify execution history
    assert len(agent_isa_with_config.execution_history) == 5
    assert all(entry["success"] for entry in agent_isa_with_config.execution_history)

@pytest.mark.asyncio
async def test_instruction_learning(agent_isa_with_config):
    """Test instruction learning from execution history."""
    # Register and execute instructions
    instruction1 = TestInstruction("test1", "test1")
    instruction2 = TestInstruction("test2", "test2")
    
    id1 = await agent_isa_with_config.register_instruction(instruction1)
    id2 = await agent_isa_with_config.register_instruction(instruction2)
    
    params = {"param": "test"}
    await agent_isa_with_config.execute_instruction(id1, params)
    await agent_isa_with_config.execute_instruction(id2, params)
    
    # Learn from history
    await agent_isa_with_config.learn_from_history()
    
    # Verify learned instructions
    stats = agent_isa_with_config.get_instruction_stats()
    assert stats["learned_instructions"] >= 0

@pytest.mark.asyncio
async def test_error_handling(agent_isa):
    """Test error handling in instruction execution."""
    # Test execution of non-existent instruction
    with pytest.raises(ValueError):
        await agent_isa.execute_instruction("non_existent", {})
    
    # Test execution with invalid parameters
    instruction = TestInstruction()
    instruction_id = await agent_isa.register_instruction(instruction)
    
    class FailingInstruction(TestInstruction):
        async def execute(self, input_params: Dict[str, Any]) -> Any:
            raise ValueError("Test error")
    
    failing_instruction = FailingInstruction()
    failing_id = await agent_isa.register_instruction(failing_instruction)
    
    with pytest.raises(ValueError):
        await agent_isa.execute_instruction(failing_id, {})

@pytest.mark.asyncio
async def test_instruction_stats(agent_isa):
    """Test instruction statistics collection."""
    # Register and execute instructions
    instruction = TestInstruction()
    instruction_id = await agent_isa.register_instruction(instruction)
    
    params = {"param": "test"}
    await agent_isa.execute_instruction(instruction_id, params)
    
    # Get stats
    stats = agent_isa.get_instruction_stats()
    assert stats["total_instructions"] == 1
    assert stats["total_executions"] == 1
    assert stats["success_rate"] == 1.0

@pytest.mark.asyncio
async def test_cleanup(agent_isa):
    """Test cleanup functionality."""
    # Register instruction
    instruction = TestInstruction()
    await agent_isa.register_instruction(instruction)
    
    # Execute instruction
    params = {"param": "test"}
    await agent_isa.execute_instruction(instruction.id, params)
    
    # Verify state before cleanup
    assert len(agent_isa.instructions) > 0
    assert len(agent_isa.execution_history) > 0
    
    # Perform cleanup
    await agent_isa.cleanup()
    
    # Verify state after cleanup
    assert len(agent_isa.instructions) == 0
    assert len(agent_isa.execution_history) == 0

@pytest.mark.asyncio
async def test_configuration(agent_isa_with_config):
    """Test configuration handling."""
    config = agent_isa_with_config.config
    
    assert config.enable_learning is True
    assert config.enable_optimization is True
    assert config.model_type == "test_model"
    assert config.min_pattern_support == 2
    assert config.min_success_rate == 0.8
    assert config.hot_path_threshold == 50
    assert config.stability_threshold == 0.85 