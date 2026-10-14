"""Test agent integration with ISA."""

import pytest
import os
from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig
from agentflow.core.isa.isa_manager import Instruction, InstructionType, InstructionStatus
from agentflow.core.isa.types import AgentType
from datetime import datetime

@pytest.fixture
def sample_config():
    """Create a sample configuration for testing."""
    return {
        "AGENT": {
            "name": "TestAgent",
            "type": AgentType.RESEARCH.value,
            "system_prompt": "You are a test agent designed to help with integration testing."
        },
        "MODEL": {
            "name": "gpt-4",
            "provider": "openai"
        },
        "WORKFLOW": {
            "max_iterations": 5
        },
        "isa_config_path": os.path.join(os.path.dirname(__file__), "test_isa_config.json"),
        "domain_config": {
            "instruction_set": ["research", "analyze", "summarize"],
            "optimization_strategy": "rl",
            "isa_config_path": os.path.join(os.path.dirname(__file__), "test_isa_config.json")
        }
    }

@pytest.fixture
def sample_instructions():
    """Sample instructions for testing."""
    return [
        Instruction(
            id="init",
            name="initialize",
            type=InstructionType.CONTROL,
            params={"init_param": "value"},
            description="Initialize system"
        ),
        Instruction(
            id="process",
            name="process_data",
            type=InstructionType.COMPUTATION,
            params={"data_param": "value"},
            description="Process input data"
        ),
        Instruction(
            id="validate",
            name="validate_result",
            type=InstructionType.VALIDATION,
            params={"threshold": 0.9},
            description="Validate results"
        )
    ]

class TestAgentISAIntegration:
    """Test agent integration with ISA."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, sample_config):
        """Test agent initialization with ISA."""
        # Extract agent config from nested structure
        config_dict = {
            "name": sample_config["AGENT"]["name"],
            "type": sample_config["AGENT"]["type"],
            "system_prompt": sample_config["AGENT"]["system_prompt"],
            "model": sample_config["MODEL"],
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "max_iterations": sample_config["WORKFLOW"]["max_iterations"],
                "steps": [
                    {
                        "id": "test-step-1",
                        "name": "test_step",
                        "type": "agent",
                        "description": "Test step for ISA integration",
                        "config": {
                            "strategy": "standard",
                            "params": {}
                        }
                    }
                ]
            },
            "domain_config": sample_config["domain_config"],
            "isa_config_path": sample_config["isa_config_path"],
            "test_mode": True
        }
        config = AgentConfig(**config_dict)
        agent = Agent(config)
        await agent.initialize()
        
        assert agent.isa_manager is not None
        assert agent.instruction_selector is not None
        
    @pytest.mark.asyncio
    async def test_instruction_selection(self, sample_config):
        """Test instruction selection based on input."""
        # Extract agent config from nested structure
        config_dict = {
            "name": sample_config["AGENT"]["name"],
            "type": sample_config["AGENT"]["type"],
            "system_prompt": sample_config["AGENT"]["system_prompt"],
            "model": sample_config["MODEL"],
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "max_iterations": sample_config["WORKFLOW"]["max_iterations"],
                "steps": [
                    {
                        "id": "test-step-1",
                        "name": "test_step",
                        "type": "agent",
                        "description": "Test step for ISA integration",
                        "config": {
                            "strategy": "standard",
                            "params": {}
                        }
                    }
                ]
            },
            "domain_config": sample_config["domain_config"],
            "isa_config_path": sample_config["isa_config_path"],
            "test_mode": True
        }
        config = AgentConfig(**config_dict)
        agent = Agent(config)
        await agent.initialize()
        
        # Test instruction selection
        input_data = {"task": "process data", "params": {"data": "test"}, "text": "process data"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)
        
        assert isinstance(selected, list)
        assert all(isinstance(instr, str) for instr in selected)
        
    @pytest.mark.asyncio
    async def test_instruction_execution(self, sample_config):
        """Test executing selected instructions."""
        # Extract agent config from nested structure
        config_dict = {
            "name": sample_config["AGENT"]["name"],
            "type": sample_config["AGENT"]["type"],
            "system_prompt": sample_config["AGENT"]["system_prompt"],
            "model": sample_config["MODEL"],
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "max_iterations": sample_config["WORKFLOW"]["max_iterations"],
                "steps": [
                    {
                        "id": "test-step-1",
                        "name": "test_step",
                        "type": "agent",
                        "description": "Test step for ISA integration",
                        "config": {
                            "strategy": "standard",
                            "params": {}
                        }
                    }
                ]
            },
            "domain_config": sample_config["domain_config"],
            "isa_config_path": sample_config["isa_config_path"],
            "test_mode": True
        }
        config = AgentConfig(**config_dict)
        agent = Agent(config)
        await agent.initialize()

        # Select and execute instructions
        input_data = {"task": "init and process", "params": {"x": 1, "data": "test"}, "text": "initialize and process data"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)

        # Prepare context for instruction execution
        context = {
            "input_data": input_data,
            "agent_config": config.model_dump(),  # Use model_dump instead of to_dict
            "timestamp": datetime.now().isoformat()
        }

        results = []
        for instruction_id in selected:
            instruction = agent.isa_manager.get_instruction(instruction_id)
            result = await agent.isa_manager.execute_instruction(instruction, context)
            results.append(result)

        # Verify results
        assert len(results) > 0
        for result in results:
            assert result.status == InstructionStatus.SUCCESS
            
    @pytest.mark.asyncio
    async def test_error_handling(self, sample_config):
        """Test error handling in ISA integration."""
        # Extract agent config from nested structure
        config_dict = {
            "name": sample_config["AGENT"]["name"],
            "type": sample_config["AGENT"]["type"],
            "system_prompt": sample_config["AGENT"]["system_prompt"],
            "model": sample_config["MODEL"],
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "max_iterations": sample_config["WORKFLOW"]["max_iterations"],
                "steps": [
                    {
                        "id": "test-step-1",
                        "name": "test_step",
                        "type": "agent",
                        "description": "Test step for ISA integration",
                        "config": {
                            "strategy": "standard",
                            "params": {}
                        }
                    }
                ]
            },
            "domain_config": sample_config["domain_config"],
            "isa_config_path": sample_config["isa_config_path"],
            "test_mode": True
        }
        config = AgentConfig(**config_dict)
        agent = Agent(config)
        await agent.initialize()
        
        # Test invalid instruction handling
        with pytest.raises(ValueError):
            agent.isa_manager.get_instruction("nonexistent")
            
        # Test instruction execution error handling
        # Create an invalid instruction that should fail validation
        invalid_instruction = Instruction(
            id="test_invalid",
            name="test_invalid",
            type=InstructionType.CONTROL,
            params={"required_param": None},  # Invalid parameter value
            description="Invalid instruction for testing"
        )
        
        # Execute with minimal context to trigger validation error
        with pytest.raises(Exception):
            await agent.isa_manager.execute_instruction(invalid_instruction, {
                "input_data": {},
                "agent_config": config.model_dump(),
                "timestamp": datetime.now().isoformat()
            })
            
    @pytest.mark.asyncio
    async def test_instruction_optimization(self, sample_config):
        """Test RL-based instruction optimization."""
        # Extract agent config from nested structure
        config_dict = {
            "name": sample_config["AGENT"]["name"],
            "type": sample_config["AGENT"]["type"],
            "system_prompt": sample_config["AGENT"]["system_prompt"],
            "model": sample_config["MODEL"],
            "workflow": {
                "id": "test-workflow",
                "name": "Test Workflow",
                "max_iterations": sample_config["WORKFLOW"]["max_iterations"],
                "steps": [
                    {
                        "id": "test-step-1",
                        "name": "test_step",
                        "type": "agent",
                        "description": "Test step for ISA integration",
                        "config": {
                            "strategy": "standard",
                            "params": {}
                        }
                    }
                ]
            },
            "domain_config": sample_config["domain_config"],
            "isa_config_path": sample_config["isa_config_path"],
            "test_mode": True
        }
        config = AgentConfig(**config_dict)
        agent = Agent(config)
        await agent.initialize()

        # Test optimization strategy
        input_data = {"task": "complex task", "params": {"data": "test"}, "text": "complex task requiring multiple steps"}
        selected = agent.instruction_selector.select_instructions(input_data, agent.isa_manager.instructions)

        # Prepare context for instruction execution
        context = {
            "input_data": input_data,
            "agent_config": config.model_dump(),  # Use model_dump instead of to_dict
            "timestamp": datetime.now().isoformat()
        }

        # Execute instructions and collect rewards
        rewards = []
        for instruction_id in selected:
            instruction = agent.isa_manager.get_instruction(instruction_id)
            result = await agent.isa_manager.execute_instruction(instruction, context)
            
            # Calculate reward based on instruction execution result
            reward = 1.0 if result.status == InstructionStatus.SUCCESS else -1.0
            rewards.append(reward)

        # Verify rewards and optimization
        assert len(rewards) > 0
        assert all(reward >= 0 for reward in rewards)
