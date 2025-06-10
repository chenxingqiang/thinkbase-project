"""Tests for agent status handling."""

import pytest
import ray
from agentflow.agents.agent import Agent, AgentState, RemoteAgent
from agentflow.core.base_types import AgentStatus
from agentflow.core.agent_config import AgentConfig
from agentflow.core.model_config import ModelConfig
from agentflow.ell.types.message import (
    Message, 
    MessageRole, 
    ContentBlock, 
    MessageType,
    MessageValidationError
)
from pydantic import ValidationError

@pytest.mark.asyncio
async def test_agent_initial_status():
    """Test agent initial status."""
    agent = Agent(config=AgentConfig(name="test_agent", type="generic"))
    assert agent.status == AgentStatus.INITIALIZED
    assert agent.state.status == AgentStatus.INITIALIZED

@pytest.mark.asyncio
async def test_agent_status_transitions():
    """Test agent status transitions during message processing."""
    agent = Agent(config=AgentConfig(name="test_agent", type="generic"))
    await agent.initialize()
    
    # Mock ELL integration
    class MockELL:
        async def process_message(self, message):
            return Message(content="test response", role=MessageRole.ASSISTANT)
    agent._ell = MockELL()
    
    # Process a message
    message = "test message"
    assert agent.state.status == AgentStatus.INITIALIZED
    await agent.process_message(message)
    assert agent.state.status == AgentStatus.SUCCESS

@pytest.mark.asyncio
async def test_agent_error_status():
    """Test agent status when errors occur."""
    agent = Agent(config=AgentConfig(name="test_agent", type="generic"))
    await agent.initialize()
    
    # Mock ELL integration to raise an error
    class MockELL:
        async def process_message(self, message):
            raise ValueError("Test error")
    agent._ell = MockELL()
    
    # Process a message that will fail
    message = "test message"
    with pytest.raises(ValueError):
        await agent.process_message(message)
    assert agent.state.status == AgentStatus.FAILED

@pytest.mark.asyncio
async def test_agent_cleanup_status():
    """Test agent status after cleanup."""
    agent = Agent(config=AgentConfig(name="test_agent", type="generic"))
    await agent.initialize()
    await agent.cleanup()
    assert agent.status == AgentStatus.STOPPED

@pytest.mark.asyncio
async def test_remote_agent_status():
    """Test remote agent status handling."""
    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    try:
        # Create a remote agent reference
        agent_ref = RemoteAgent.remote()  # type: ignore
        
        # Get initial status using get_status_remote
        status_str = ray.get(agent_ref.get_status_remote.remote())  # type: ignore
        assert status_str == AgentStatus.INITIALIZED.value
        
        # Initialize
        ray.get(agent_ref.initialize.remote())  # type: ignore
        
        # Check status after initialization
        status_str = ray.get(agent_ref.get_status_remote.remote())  # type: ignore
        assert status_str == AgentStatus.INITIALIZED.value
        
        # Cleanup
        ray.get(agent_ref.cleanup.remote())  # type: ignore
        
        # Check status after cleanup
        status_str = ray.get(agent_ref.get_status_remote.remote())  # type: ignore
        assert status_str == AgentStatus.STOPPED.value
    finally:
        # Shutdown Ray
        if ray.is_initialized():
            ray.shutdown()

@pytest.mark.asyncio
async def test_agent_state_timing():
    """Test agent state timing during message processing."""
    state = AgentState()
    message = "test message"
    
    # Process message
    await state.process_message(message)
    
    # Verify timing
    assert state.start_time is not None
    assert state.end_time is not None
    assert state.end_time >= state.start_time

class ErrorContentBlock(ContentBlock):
    def __str__(self) -> str:
        raise ValueError("Error during string conversion")

class ErrorMessage(Message):
    def __str__(self) -> str:
        raise ValueError("Error during string conversion")

@pytest.mark.asyncio
async def test_agent_state_error_timing():
    """Test agent state timing during error handling."""
    state = AgentState()
    
    # Create a message that will raise an error during string conversion
    message = ErrorMessage(
        role=MessageRole.USER,
        content="test",
        type=MessageType.TEXT
    )
    
    # Process message should raise ValueError
    with pytest.raises(ValueError, match="Error during string conversion"):
        await state.process_message(message)
    
    # Check that timing was recorded despite the error
    assert state.start_time is not None
    assert state.end_time is not None
    assert state.end_time >= state.start_time
    assert state.status == AgentStatus.FAILED
    assert state.last_error is not None
    assert len(state.errors) == 1
    assert "timestamp" in state.errors[0]