"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import logging
import time
from typing import Dict, Any, Optional, cast, Union
from datetime import datetime
import uuid

from agentflow.agents.agent import Agent
from agentflow.core.agent_config import AgentConfig
from agentflow.core.model_config import ModelConfig
from agentflow.core.config import WorkflowStepType
from agentflow.core.workflow_types import (
    Message,
    RetryPolicy,
    WorkflowConfig,
    WorkflowStep,
    StepConfig,
    ErrorPolicy,
    WorkflowStrategy
)
from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.workflow_executor import WorkflowExecutor
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.ell.types.message import Message, MessageRole, MessageType, assistant, user
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.isa.instruction_selector import InstructionSelector
from agentflow.core.isa.instruction import Instruction
from agentflow.core.isa.types import InstructionType, InstructionStatus
from agentflow.core.isa.result import InstructionResult
from agentflow.core.types import AgentStatus
from pydantic import PrivateAttr

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

async def mock_transform_function(step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
    """Mock transform function for testing.
    
    Args:
        step: The workflow step being executed
        context: The execution context
        
    Returns:
        Transformed data
    """
    return {"transformed": context}

@pytest.fixture(scope="function")
def event_loop():
    """Create event loop for tests."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        loop.close()

@pytest.fixture(scope="function")
def mock_ell():
    """Create a mock ELL integration."""
    mock = AsyncMock()
    mock.process_message.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Test response",
        type=MessageType.TOOL_RESULT,
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": "tool_result",
            "status": "completed"
        }
    )
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    mock.configure = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_isa_manager():
    """Create a mock ISA manager."""
    mock = AsyncMock()
    mock.get_isa.return_value = "test_isa"
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_instruction_selector():
    """Create a mock instruction selector."""
    mock = AsyncMock()
    mock.select_instruction.return_value = "test_instruction"
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_state_manager():
    """Create a mock state manager."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
def mock_metrics_manager():
    """Create a mock metrics manager."""
    mock = AsyncMock()
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(scope="function")
async def agent(mock_ell, mock_isa_manager, mock_instruction_selector):
    """Create a test agent."""
    workflow_cfg = WorkflowConfig(
        id=str(uuid.uuid4()),
        name="test_workflow",
        max_iterations=5,
        error_policy=ErrorPolicy(
            fail_fast=True,
            ignore_warnings=False,
            max_errors=10,
            retry_policy=RetryPolicy(
                retry_delay=1.0,
                backoff=2.0,
                max_retries=3,
                max_delay=60.0
            )
        ),
        steps=[
            WorkflowStep(
                id="step1",
                name="test_step",
                type=WorkflowStepType.TRANSFORM,
                description="Test transform step for workflow execution",
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={
                        "substrategy": "standard",
                        "execute": mock_transform_function
                    },
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    agent_config = AgentConfig(
        id="test-agent-1",
        name="test_agent",
        type="generic",
        mode="sequential",
        version="1.0.0",
        model=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow_cfg.model_dump()
    )
    agent = Agent(config=agent_config)
    agent._ell = mock_ell
    agent._isa_manager = mock_isa_manager
    agent._instruction_selector = mock_instruction_selector
    await agent.initialize()
    return agent

class TestAgentFramework:
    """Test agent framework functionality."""

    @pytest.fixture(autouse=True)
    def setup_agent(self):
        """Set up test environment."""
        self.workflow_config = WorkflowConfig(
            id="test-workflow-1",
            name="test_workflow",
            max_iterations=5,
            timeout=30,
            steps=[
                WorkflowStep(
                    id="step-1",
                    name="step_1",
                    type=WorkflowStepType.TRANSFORM,
                    description="Test transform step",
                    config=StepConfig(
                        strategy="custom",  # Using a valid strategy
                        params={
                            "method": "standard",
                            "with_mean": True,
                            "with_std": True,
                            "execute": mock_transform_function
                        }
                    )
                )
            ]
        )
        self.agent_config = AgentConfig(
            id="test-agent-1",
            name="test_agent",
            type="generic",
            mode="sequential",
            version="1.0.0",
            model=ModelConfig(
                provider="openai",
                name="gpt-4",
                temperature=0.7,
                max_tokens=4096
            ),
            workflow=self.workflow_config.model_dump()
        )

    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test agent initialization."""
        print("Workflow type:", type(self.agent_config.workflow))
        print("Workflow:", self.agent_config.workflow)
        assert self.agent_config.id == "test-agent-1"
        assert self.agent_config.name == "test_agent"
        assert self.agent_config.type == "generic"
        assert self.agent_config.mode == "sequential"
        assert self.agent_config.version == "1.0.0"
        assert isinstance(self.agent_config.model, ModelConfig)
        assert self.agent_config.model.name == "gpt-4"
        assert isinstance(self.agent_config.workflow, WorkflowConfig)

    @pytest.mark.asyncio
    async def test_agent_workflow_execution(self):
        """Test agent workflow execution."""
        agent = Agent(config=self.agent_config)
        engine = WorkflowEngine()
        await engine.initialize()
        await engine.register_workflow(agent, self.workflow_config)
        data = {"test": "input"}
        result = await engine.execute_workflow(agent.id, data)
        assert result is not None
        assert "steps" in result
        assert "step-1" in result["steps"]
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_data_science_agent_advanced_transformations(self):
        """Test data science agent with advanced transformations."""
        workflow = WorkflowConfig(
            id="test-workflow-2",
            name="test_workflow",
            max_iterations=5,
            timeout=30,
            steps=[
                WorkflowStep(
                    id="step-1",
                    name="step_1",
                    type=WorkflowStepType.TRANSFORM,
                    description="Test transform step",
                    config=StepConfig(
                        strategy="custom",  # Using a valid strategy
                        params={
                            "method": "standard",
                            "with_mean": True,
                            "with_std": True,
                            "execute": mock_transform_function
                        }
                    )
                )
            ]
        )
        agent_config = AgentConfig(
            id="test-agent-2",
            name="test_agent",
            type="data_science",
            mode="sequential",
            version="1.0.0",
            model=ModelConfig(
                provider="openai",
                name="gpt-4",
                temperature=0.7,
                max_tokens=4096
            ),
            workflow=workflow.model_dump()
        )
        agent = Agent(config=agent_config)
        engine = WorkflowEngine()
        await engine.initialize()
        await engine.register_workflow(agent, workflow)
        data = {"test": "input"}
        result = await engine.execute_workflow(agent.id, data)
        assert result is not None
        assert "steps" in result
        assert "step-1" in result["steps"]
        assert result["status"] == "success"
