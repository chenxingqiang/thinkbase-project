"""Unit tests for WorkflowEngine."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.workflow_types import (
    WorkflowStepType, WorkflowStepStatus, Message, StepConfig,
    WorkflowStep, ErrorPolicy, RetryPolicy, WorkflowConfig, WorkflowStatus
)
from agentflow.core.agent_config import AgentConfig
from agentflow.core.model_config import ModelConfig
from agentflow.core.config import WorkflowConfig as ConfigWorkflowConfig
from agentflow.agents.agent_types import AgentType, AgentMode
from agentflow.agents.agent import Agent
from agentflow.ell.integration import ELLIntegration
from agentflow.core.isa.isa_manager import ISAManager, Instruction, InstructionType
from agentflow.core.instruction_selector import InstructionSelector
from agentflow.core.isa.result import InstructionResult, InstructionStatus
from agentflow.core.exceptions import WorkflowExecutionError
import asyncio
import uuid
import logging
from agentflow.core.workflow_executor import WorkflowExecutor
from typing import Dict, Any, List, Optional
from uuid import UUID
import time

# Configure logging for tests
logging.basicConfig(level=logging.WARNING)

@pytest.fixture
def valid_workflow_def():
    """Create a valid workflow definition."""
    return {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": {
                "step1": {
                    "step": 1,
                    "name": "Test Step",
                    "description": "Test step",
                    "input": ["data"],
                    "type": WorkflowStepType.TRANSFORM,
                    "agent_config": {}
                }
            }
        }
    }

@pytest.fixture
def workflow_config(mock_ell):
    """Create a workflow configuration."""
    async def mock_transform_execute(step: WorkflowStep, context: Dict[str, Any]):
        """Mock transform execution function."""
        return {"data": "transformed_" + str(context.get("data", ""))}
    
    return WorkflowConfig(
        id="test-workflow",
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                max_retries=3,
                retry_delay=1.0,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="Test Step",
                type=WorkflowStepType.TRANSFORM,
                description="Test workflow step",
                config=StepConfig(
                    strategy="standard",
                    params={
                        "protocol": "federated", 
                        "execute": mock_transform_execute
                    }
                )
            )
        ]
    )

@pytest.fixture
def mock_isa_manager():
    """Create a mock ISA manager."""
    manager = MagicMock(spec=ISAManager)
    
    async def execute_instruction(instruction: Instruction, context: Dict[str, Any]) -> InstructionResult:
        return InstructionResult(
            id=str(uuid.uuid4()),
            status=InstructionStatus.SUCCESS,
            context={"data": "test_result"}
        )
    
    manager.execute_instruction = AsyncMock(side_effect=execute_instruction)
    return manager

@pytest.fixture
def mock_instruction_selector():
    """Create a mock instruction selector."""
    selector = MagicMock(spec=InstructionSelector)
    selector.select_instructions.return_value = ["test_instruction"]
    return selector

@pytest.fixture
def mock_ell():
    """Create a mock ELL integration."""
    ell = MagicMock(spec=ELLIntegration)
    
    async def process_message(message: str) -> Dict[str, Any]:
        return {"result": "test_result"}
    
    ell.process_message = AsyncMock(side_effect=process_message)
    return ell

@pytest.fixture(autouse=True)
async def cleanup_mocks(mock_ell, mock_isa_manager, mock_instruction_selector):
    """Clean up mocks after each test."""
    yield
    mock_ell.reset_mock()
    mock_isa_manager.reset_mock()
    mock_instruction_selector.reset_mock()

@pytest.fixture
async def workflow_engine(valid_workflow_def, workflow_config, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Create a workflow engine instance."""
    engine = WorkflowEngine()
    await engine.initialize(workflow_def=valid_workflow_def)
    
    # Create and register default agent
    agent_config = AgentConfig(
        name="default_agent",
        type="research",
        model=ModelConfig(name="gpt-4", provider="openai"),
        workflow=workflow_config
    )
    agent = Agent(config=agent_config)
    await agent.initialize()
    await engine.register_workflow(agent, workflow_config)
    
    # Store the agent's ID for later use
    engine.default_agent_id = agent.id
    
    return engine

def test_workflow_engine_init():
    """Test workflow engine initialization."""
    engine = WorkflowEngine()
    assert engine is not None
    assert not engine._initialized
    assert engine.workflows == {}

@pytest.mark.asyncio
async def test_invalid_workflow_config(workflow_engine):
    """Test invalid workflow configuration."""
    engine = await workflow_engine
    with pytest.raises(ValueError):
        await engine.register_workflow(Agent(config=AgentConfig(name="test", type="research", model=ModelConfig(name="gpt-4", provider="openai"))), None)

@pytest.mark.asyncio
async def test_workflow_execution(workflow_engine, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Test workflow execution."""
    engine = await workflow_engine
    result = await engine.execute_workflow(engine.default_agent_id, {"data": "test_input"})
    
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["status"] == "success"
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine):
    """Test workflow cleanup."""
    engine = await workflow_engine
    await engine.cleanup()
    assert engine.status == "cleaned"

@pytest.mark.asyncio
async def test_workflow_with_retry_mechanism():
    """Test workflow with retry mechanism."""
    workflow_def = {
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "FEDERATED"
            },
            "WORKFLOW": {
                "step1": {
                    "step": 1,
                    "name": "Test Step",
                    "description": "Test step",
                    "input": ["data"],
                    "type": WorkflowStepType.TRANSFORM,
                    "agent_config": {}
                }
            }
        }
    }
    
    workflow_config = WorkflowConfig(
        id="test-workflow",
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=False,
            ignore_warnings=True,
            max_errors=10,
            retry_policy=RetryPolicy(
                max_retries=3,
                retry_delay=0.1,
                backoff=2.0,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="Test Step",
                type=WorkflowStepType.TRANSFORM,
                description="Test workflow step",
                config=StepConfig(
                    strategy="standard",
                    params={"protocol": "federated"}
                )
            )
        ]
    )
    
    engine = WorkflowEngine()
    await engine.initialize(workflow_def=workflow_def)
    
    agent_config = AgentConfig(
        name="test_agent",
        type="research",
        model=ModelConfig(name="gpt-4", provider="openai"),
        workflow=workflow_config
    )
    agent = Agent(config=agent_config)
    await agent.initialize()
    await engine.register_workflow(agent, workflow_config)
    
    result = await engine.execute_workflow(agent.id, {"data": "test_input"})
    assert result is not None
    assert result["status"] == "success"
