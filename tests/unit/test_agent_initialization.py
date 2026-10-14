import pytest
import ray
from agentflow.core.config import AgentConfig, ConfigurationType, ModelConfig, WorkflowConfig
from agentflow.agents.agent import Agent, AgentState
from agentflow.core.workflow_state import AgentStatus
from pathlib import Path
import json
import uuid
from agentflow.agents.agent_types import AgentType, AgentMode

@pytest.fixture(scope="module")
def test_data_dir():
    """Create test data directory"""
    test_dir = Path(__file__).parent.parent / 'data'
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing"""
    ray.init(ignore_reinit_error=True)
    yield  # Ray lifecycle owned by conftest session fixture

def test_agent_initialization(ray_context):
    """Test basic agent initialization"""
    config_data = {
        "name": "TestAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.5
        },
        "workflow": {
            "id": "test-workflow-1",
            "name": "Test Workflow",
            "max_iterations": 5,
            "logging_level": "INFO",
            "steps": [
                {
                    "id": "step1",
                    "name": "Test Step",
                    "type": "agent",
                    "description": "Test step for initialization",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "TestAgent"
    assert agent.type == AgentType.RESEARCH
    assert agent.config is not None
    assert agent.config.version == "1.0.0"
    assert not agent.config.is_distributed

def test_agent_distributed_initialization(ray_context):
    """Test distributed agent initialization"""
    config_data = {
        "name": "DistributedAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": True,
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-2",
            "name": "Distributed Workflow",
            "max_iterations": 10,
            "distributed": True,
            "steps": [
                {
                    "id": "step1",
                    "name": "Distributed Step",
                    "type": "agent",
                    "description": "Test step for distributed execution",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "DistributedAgent"
    assert agent.type == AgentType.RESEARCH
    assert agent.config is not None
    assert agent.config.is_distributed is True

def test_agent_model_configuration():
    """Test agent model configuration"""
    config_data = {
        "name": "ModelAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.7,
            "max_tokens": 1000
        },
        "workflow": {
            "id": "test-workflow-3",
            "name": "Model Test Workflow",
            "max_iterations": 5,
            "steps": [
                {
                    "id": "step1",
                    "name": "Model Test Step",
                    "type": "agent",
                    "description": "Test step for model configuration",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "ModelAgent"
    assert agent.config is not None
    assert agent.config.model.provider == "openai"
    assert agent.config.model.name == "gpt-4"
    assert agent.config.model.temperature == 0.7
    assert agent.config.model.max_tokens == 1000

def test_agent_workflow_configuration():
    """Test agent workflow configuration"""
    config_data = {
        "name": "WorkflowAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-4",
            "name": "Test Workflow",
            "max_iterations": 7,
            "timeout": 300,
            "distributed": False,
            "steps": [
                {
                    "id": "step1",
                    "name": "Workflow Test Step",
                    "type": "agent",
                    "description": "Test step for workflow configuration",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "WorkflowAgent"
    assert agent.config is not None
    assert isinstance(agent.config.workflow, WorkflowConfig)
    assert agent.config.workflow.max_iterations == 7
    assert agent.config.workflow.timeout == 300
    assert agent.config.workflow.distributed is False

def test_agent_file_initialization(test_data_dir):
    """Test agent initialization from files"""
    config_data = {
        "name": "FileAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-5",
            "name": "File Test Workflow",
            "max_iterations": 10,
            "distributed": False,
            "steps": [
                {
                    "id": "step1",
                    "name": "File Test Step",
                    "type": "agent",
                    "description": "Test step for file initialization",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    # Write config to file
    config_path = test_data_dir / 'config.json'
    config_path.write_text(json.dumps(config_data))

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "FileAgent"
    assert agent.config is not None
    assert agent.config.workflow is not None

def test_agent_state_initialization():
    """Test agent state initialization"""
    config_data = {
        "name": "StateAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "id": "test-workflow-6",
            "name": "State Test Workflow",
            "max_iterations": 5,
            "steps": [
                {
                    "id": "step1",
                    "name": "State Test Step",
                    "type": "agent",
                    "description": "Test step for state initialization",
                    "config": {
                        "strategy": "standard",
                        "params": {}
                    }
                }
            ]
        }
    }

    config = AgentConfig(**config_data)
    agent = Agent(config=config)
    
    assert agent.name == "StateAgent"
    assert agent.status.value == "initialized"
    assert agent.config is not None

def test_agent_invalid_initialization():
    """Test agent initialization with invalid config"""
    config_data = {
        "name": "InvalidAgent",
        "type": "invalid_type",
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "workflow": {}
    }
    
    with pytest.raises(ValueError):
        config = AgentConfig(**config_data)
        Agent(config=config)

def test_agent_invalid_model_provider():
    """Test agent initialization with invalid model provider"""
    config_data = {
        "name": "InvalidAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "invalid_provider",
            "name": "invalid_model"
        }
    }
    
    with pytest.raises(ValueError):
        config = AgentConfig(**config_data)
        Agent(config=config)

def test_agent_invalid_workflow_config():
    """Test agent initialization with invalid workflow config"""
    config_data = {
        "name": "InvalidWorkflowAgent",
        "type": AgentType.RESEARCH,
        "version": "1.0.0",
        "mode": "sequential",
        "is_distributed": False,
        "model": {
            "provider": "openai",
            "name": "gpt-4"
        },
        "workflow": {
            "max_iterations": -1,  # Invalid value
            "steps": [
                {
                    "id": "invalid_step",
                    "name": "Invalid Step",
                    "type": "invalid_type",  # Invalid type
                    "config": {
                        "strategy": "invalid_strategy"  # Invalid strategy
                    }
                }
            ]
        }
    }
    
    with pytest.raises(ValueError):
        config = AgentConfig(**config_data)
        Agent(config=config)

if __name__ == "__main__":
    pytest.main([__file__])
