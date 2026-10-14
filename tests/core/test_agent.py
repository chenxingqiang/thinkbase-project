"""
Tests for Agent functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import logging
import time
import uuid

from agentflow.agents.agent import Agent, AgentState, RemoteAgent
from agentflow.core.base_types import AgentStatus, AgentType
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.core.workflow_types import WorkflowStep, WorkflowStepType, StepConfig
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import (
    Message, 
    MessageRole, 
    ContentBlock, 
    MessageType,
    MessageValidationError
)
from pydantic import ValidationError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

# Add event loop fixture
@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for tests."""
    try:
        policy = asyncio.get_event_loop_policy()
        loop = policy.new_event_loop()
        asyncio.set_event_loop(loop)
        yield loop
    finally:
        loop.close()

@pytest.fixture
def mock_ell():
    """Mock ELL integration."""
    mock = MagicMock(spec=ELLIntegration)
    mock.enabled = True
    mock.tracking_enabled = True
    mock.config = {}
    
    # Configure the mock to return a Message object matching the test expectations
    async def mock_process_message(message):
        return Message(
            content=message.content, 
            role=MessageRole.ASSISTANT,
            type=MessageType.TEXT
        )
    
    mock.process_message = AsyncMock(side_effect=mock_process_message)
    
    return mock

@pytest.fixture
def model_config():
    """Create test model config."""
    return {
        "provider": "default",
        "name": "gpt-3.5-turbo"
    }

@pytest.fixture
def workflow_config():
    """Create test workflow config."""
    return {
        "id": "test-workflow-id",
        "name": "test-workflow",
        "max_iterations": 10,
        "timeout": 3600,
        "steps": [
            {
                "id": "test-step-1",
                "name": "test_step",
                "type": WorkflowStepType.TRANSFORM,
                "description": "Test workflow step",
                "config": {
                    "strategy": "standard",
                    "params": {}
                }
            }
        ]
    }

@pytest.fixture
async def agent(model_config, workflow_config, mock_ell):
    """Create and initialize test agent."""
    _agent = None
    try:
        _agent = Agent(
            id=str(uuid.uuid4()),  # Generate a new UUID for each test
            name="Test Agent",
            description=None,  # Optional field
            type=AgentType.GENERIC,
            mode="sequential",
            config={
                "name": "Test Agent",
                "type": AgentType.GENERIC,
                "system_prompt": "You are a test agent",
                "model": model_config
            },
            workflow=workflow_config
        )
        _agent._ell = mock_ell
        await _agent.initialize()
        _agent.metadata["test_mode"] = True
        yield _agent
    finally:
        if _agent:
            await _agent.cleanup()

@pytest.mark.asyncio
async def test_agent_initial_status():
    """Test agent initial status."""
    agent = Agent(config=AgentConfig(name="test_agent", type=AgentType.GENERIC))
    assert agent.status == AgentStatus.INITIALIZED
    assert agent.state.status == AgentStatus.INITIALIZED

@pytest.mark.asyncio
async def test_agent_status_transitions():
    """Test agent status transitions during message processing."""
    agent = Agent(config=AgentConfig(name="test_agent", type=AgentType.GENERIC))
    await agent.initialize()
    
    # Mock ELL integration
    class MockELL:
        async def process_message(self, message):
            return Message(content=message.content, role=MessageRole.ASSISTANT, type=MessageType.TEXT)
    agent._ell = MockELL()
    
    # Process a message
    message = "test message"
    assert agent.state.status == AgentStatus.INITIALIZED
    await agent.process_message(message)
    assert agent.state.status == AgentStatus.SUCCESS

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    agent_instance = agent
    assert isinstance(agent_instance, Agent)
    assert agent_instance.state.status == AgentStatus("initialized")
    assert agent_instance._initialized

@pytest.mark.asyncio
async def test_message_processing(agent, mock_ell):
    """Test message processing."""
    agent_instance = agent
    assert isinstance(agent_instance, Agent)

    # Ensure the mock is properly set up
    agent_instance._ell = mock_ell
        
    test_message = "Test message"
    response = await agent_instance.process_message(test_message)

    assert isinstance(response, str)
    assert test_message in response  # Check if the test message is contained in the response
        
    # Verify history is properly maintained
    assert len(agent_instance.history) == 2  # Should have user message and assistant response
    assert agent_instance.history[0]["content"] == test_message
    assert agent_instance.history[0]["role"] == "user"
    assert agent_instance.history[1]["content"] == test_message
    assert agent_instance.history[1]["role"] == MessageRole.ASSISTANT
        
    assert agent_instance.state.status in [AgentStatus.SUCCESS, AgentStatus.PROCESSING]

@pytest.mark.asyncio
async def test_error_handling(agent, mock_ell):
    """Test error handling."""
    agent_instance = agent
    assert isinstance(agent_instance, Agent)

    # Set up mock to raise an exception
    mock_ell.process_message.side_effect = Exception("Test error")
    agent_instance._ell = mock_ell

    with pytest.raises(Exception):
        await agent_instance.process_message("Test message")

    assert agent_instance.state.status == "failed"
    assert len(agent_instance.errors) == 1
    assert agent_instance.state.last_error == "Test error"

@pytest.mark.asyncio
async def test_error_limit(agent, mock_ell):
    """Test error limit handling."""
    agent_instance = agent
    assert isinstance(agent_instance, Agent)

    # Set up mock to raise an exception
    mock_ell.process_message.side_effect = Exception("Test error")
    agent_instance._ell = mock_ell

    # Generate more errors than the limit
    for i in range(agent_instance.max_errors + 5):
        with pytest.raises(Exception):
            await agent_instance.process_message("Test message")

    # Check that errors list is limited
    assert len(agent_instance.errors) == agent_instance.max_errors
    assert agent_instance.state.status == "failed"
