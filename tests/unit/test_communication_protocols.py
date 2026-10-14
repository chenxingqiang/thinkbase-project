import pytest
import ray
import uuid
from typing import Dict, Any
from agentflow.core.workflow_types import (
    WorkflowConfig as WorkflowConfigType,
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowStatus,
    WorkflowConfig
)
from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.types import AgentType, AgentMode, AgentConfig, ModelConfig
from agentflow.agents.agent import Agent
from pathlib import Path
import json
import asyncio
from datetime import datetime
from agentflow.core.exceptions import ValidationError

@pytest.fixture(scope="module")
def test_data_dir():
    """Create test data directory"""
    test_dir = Path(__file__).parent.parent / 'data'
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir

@pytest.fixture(scope="module")
def ray_context():
    """Initialize Ray for testing"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    # Ray lifecycle owned by conftest session fixture; do not shutdown here.

@pytest.fixture
def workflow_engine():
    """Create a workflow engine instance"""
    async def _create_engine(workflow_def: Dict[str, Any], workflow_config: Dict[str, Any], test_mode: bool = False) -> WorkflowEngine:
        engine = WorkflowEngine()
        try:
            await engine.initialize(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=test_mode)
            
            # Create and register default agent
            model_config = ModelConfig(name="gpt-4", provider="openai")
            workflow_cfg = WorkflowConfig.model_validate(workflow_config, context={"test_mode": test_mode})  # For workflow engine
            agent_workflow_cfg = WorkflowConfig.model_validate(workflow_config, context={"test_mode": test_mode})  # For agent config
            
            agent_config = AgentConfig(
                name="default_agent",
                type=AgentType.RESEARCH,
                model=model_config,  # Pass ModelConfig instance directly
                workflow=agent_workflow_cfg.model_dump()  # Convert back to dictionary
            )
            agent = Agent(config=agent_config, name="default_agent")
            agent.metadata["test_mode"] = test_mode  # Set test mode in agent metadata
            await agent.initialize()  # Initialize the agent before registering
            await engine.register_workflow(agent, workflow_cfg)
            
            # Store the agent's ID for later use
            engine.default_agent_id = agent.id
            
            return engine
        except Exception as e:
            # Re-raise validation errors
            if "Invalid protocol" in str(e):
                raise ValueError(f"Invalid protocol: {workflow_config['steps'][0]['config']['params']['protocol']}")
            raise
    return _create_engine

@pytest.mark.asyncio
async def test_federated_learning_protocol(workflow_engine):
    """Test federated learning protocol"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": [
                {
                    "name": "local_model_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "local_model_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_federated_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "federated_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Federated learning transformation step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "federated"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine with test mode
    engine = await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=True)
    
    # Execute workflow
    result = await engine.execute_workflow(engine.default_agent_id, {"data": {}})
    
    # Verify results
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["steps"]["step1"]["result"] is not None
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_gossip_protocol(workflow_engine):
    """Test gossip protocol"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "GOSSIP"
            },
            "WORKFLOW": [
                {
                    "name": "node_1",
                    "agent_type": "test",
                    "knowledge": {"topic_a": "info_1", "topic_b": "data_1"}
                },
                {
                    "name": "node_2",
                    "agent_type": "test",
                    "knowledge": {"topic_a": "info_2", "topic_c": "data_2"}
                }
            ]
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_gossip_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "gossip_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Gossip protocol transformation step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "gossip"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine with test mode
    engine = await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=True)
    
    # Execute workflow
    result = await engine.execute_workflow(engine.default_agent_id, {"data": {}})
    
    # Verify results
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["steps"]["step1"]["result"] is not None
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol(workflow_engine):
    """Test hierarchical merge protocol"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "DYNAMIC_ROUTING",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "low_level_agent_1": {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "hierarchy_level": 0,
                    "data": "raw_data_1"
                },
                "low_level_agent_2": {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "hierarchy_level": 0,
                    "data": "raw_data_2"
                },
                "mid_level_agent": {
                    "name": "mid_level_agent",
                    "agent_type": "test",
                    "hierarchy_level": 1,
                    "dependencies": ["low_level_agent_1", "low_level_agent_2"]
                }
            }
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_hierarchical_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "hierarchical_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Hierarchical merge transformation step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "hierarchical"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine with test mode
    engine = await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=True)
    
    # Execute workflow
    result = await engine.execute_workflow(engine.default_agent_id, {"data": {}})
    
    # Verify results
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["steps"]["step1"]["result"] is not None
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_invalid_communication_protocol(workflow_engine):
    """Test invalid communication protocol"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "GOSSIP"
            },
            "WORKFLOW": [
                {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_invalid_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "invalid_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Invalid protocol step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "invalid"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine and expect validation error
    with pytest.raises(ValueError, match="Invalid protocol: invalid"):
        await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=False)

@pytest.mark.asyncio
async def test_empty_workflow(workflow_engine):
    """Test empty workflow"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "PARALLEL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": []
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_empty_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": []  # Empty steps list
    }

    # Create workflow engine and expect validation error
    with pytest.raises(ValueError, match="Workflow steps list cannot be empty"):
        await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=False)

@pytest.mark.asyncio
async def test_hierarchical_merge_protocol_with_test_agent(workflow_engine):
    """Test hierarchical merge protocol with test agent"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL_MERGE"
            },
            "WORKFLOW": [
                {
                    "name": "high_level_agent",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.5, "bias1": 0.3}
                },
                {
                    "name": "low_level_agent_1",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.1, "bias1": 0.2}
                },
                {
                    "name": "low_level_agent_2",
                    "agent_type": "test",
                    "model_params": {"weight1": 0.3, "bias1": 0.4}
                }
            ]
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_hierarchical_merge_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "hierarchical_merge_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Hierarchical merge protocol step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "hierarchical_merge"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine with test mode
    engine = await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=True)
    
    # Execute workflow
    result = await engine.execute_workflow(engine.default_agent_id, {"data": {}})
    
    # Verify results
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["steps"]["step1"]["result"] is not None
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_invalid_communication_protocol_with_test_agent(workflow_engine):
    """Test invalid communication protocol with test agent"""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "UNKNOWN_PROTOCOL"
            },
            "WORKFLOW": [
                {
                    "name": "agent_1",
                    "agent_type": "test"
                },
                {
                    "name": "agent_2",
                    "agent_type": "test"
                }
            ]
        }
    }

    workflow_config = {
        "id": str(uuid.uuid4()),
        "name": "test_invalid_protocol_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "invalid_protocol_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Invalid protocol step",
                "required": True,
                "optional": False,
                "is_distributed": True,
                "dependencies": [],
                "config": {
                    "strategy": "custom",
                    "params": {"protocol": "unknown"},
                    "retry_delay": 1.0,
                    "retry_backoff": 2.0,
                    "max_retries": 3
                }
            }
        ]
    }

    # Create workflow engine - in test mode, invalid protocols are converted to None
    engine = await workflow_engine(workflow_def=workflow_def, workflow_config=workflow_config, test_mode=True)
    
    # Verify that the protocol was converted to None
    assert engine.workflows[engine.default_agent_id].steps[0].config.params["protocol"] is None
    # Execute workflow
    result = await engine.execute_workflow(engine.default_agent_id, {"data": {}})
    
    # Verify results
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["steps"]["step1"]["result"] is not None
    assert result["steps"]["step1"]["status"] == "success"
