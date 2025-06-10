"""
Tests for AgentFlow system functionality
"""

import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock
import logging
import time
from typing import Dict, Any, Optional

from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, StepConfig, WorkflowStepType
from agentflow.core.workflow import WorkflowEngine
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
from agentflow.core.instruction_selector import InstructionSelector
from agentflow.core.isa.isa_manager import ISAManager
from agentflow.core.types import AgentStatus
from agentflow.core.exceptions import WorkflowExecutionError

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Configure pytest-asyncio
pytestmark = pytest.mark.asyncio

@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for tests."""
    try:
        loop = asyncio.new_event_loop()
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
    mock.metrics = {
        "function_calls": 0,
        "total_execution_time": 0.0,
        "errors": 0
    }
    
    # Create a mock response message
    response = Message(
        role=MessageRole.ASSISTANT,
        content="Test response",
        metadata={
            "model": "test-model",
            "timestamp": time.time()
        }
    )
    
    # Set up the mock to return the response
    mock.process_message = AsyncMock(return_value=response)
    mock.configure = MagicMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture(autouse=True)
async def cleanup_mocks(mock_ell):
    """Clean up mocks after each test."""
    # Store original mock function
    original_mock = mock_ell.process_message
    yield
    # Reset mock and restore original function
    mock_ell.reset_mock()
    mock_ell.process_message = original_mock

@pytest.fixture
def mock_isa_manager():
    """Mock ISA manager."""
    mock = MagicMock(spec=ISAManager)
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def mock_instruction_selector():
    """Mock instruction selector."""
    mock = MagicMock(spec=InstructionSelector)
    mock.initialize = AsyncMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.fixture
def workflow_config():
    """Create test workflow config."""
    return WorkflowConfig(
        id="test-workflow-id",
        name="test-workflow",
        max_iterations=10,
        timeout=3600,
        steps=[
            WorkflowStep(
                id="test-step-1",
                name="test_step",
                type=WorkflowStepType.TRANSFORM,
                description="Test workflow step",
                config=StepConfig(strategy="standard")
            )
        ]
    )

@pytest.fixture
def model_config():
    """Create test model config."""
    return ModelConfig(
        provider="default",
        name="gpt-3.5-turbo"
    )

@pytest.fixture
def agent_config(model_config, workflow_config):
    """Create test agent config."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="generic",
        system_prompt="You are a test agent",
        model=model_config,
        workflow=workflow_config,
        config={
            "algorithm": "PPO"
        }
    )

@pytest.fixture
async def agent(workflow_config, mock_ell):
    """Create and initialize test agent."""
    agent_config = AgentConfig(
        name="test_agent",
        type="generic",
        mode="sequential",
        model=ModelConfig(name="gpt-4", provider="openai"),
        workflow=workflow_config
    )
    _agent = Agent(config=agent_config)
    _agent._ell = mock_ell
    await _agent.initialize()
    _agent.metadata["test_mode"] = True
    try:
        yield _agent
    finally:
        await _agent.cleanup()

@pytest.fixture
async def workflow_engine():
    """Create a workflow engine for testing."""
    default_workflow_config = WorkflowConfig(
        id="test-workflow-1",
        name="test_workflow",
        steps=[
            WorkflowStep(
                id="test-step-1", 
                name="test_step", 
                type=WorkflowStepType.TRANSFORM,
                description="Test workflow step",
                config=StepConfig(strategy="standard")
            )
        ]
    )
    engine = WorkflowEngine(workflow_config=default_workflow_config)
    await engine.initialize()
    try:
        yield engine
    finally:
        await engine.cleanup()

@pytest.mark.asyncio
async def test_workflow_initialization(workflow_engine):
    """Test workflow engine initialization."""
    engine = await anext(workflow_engine)
    assert engine._initialized
    assert engine._ell is not None
    assert engine._isa_manager is not None
    assert engine._instruction_selector is not None

@pytest.mark.asyncio
async def test_agent_registration(workflow_engine, agent):
    """Test agent registration in workflow."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    assert workflow_id in engine.workflows

@pytest.mark.asyncio
async def test_workflow_execution(workflow_engine, agent, mock_ell):
    """Test workflow execution."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    # Set up mock ELL
    agent_instance._ell = mock_ell
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input", "test_mode": True}  # Add test_mode flag
    result = await engine.execute_workflow(workflow_id, input_data)
    
    assert result is not None
    assert result["status"] == "success"
    assert "Test response" in result.get("content", "")
    assert mock_ell.process_message.called

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow_engine, agent, mock_ell):
    """Test workflow error handling."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Set up mock to raise an exception
    mock_ell.process_message = AsyncMock(side_effect=WorkflowExecutionError("Test error"))
    agent_instance._ell = mock_ell
    
    with pytest.raises(WorkflowExecutionError):
        await engine.execute_workflow(workflow_id, {"message": "Test input", "test_mode": True})

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, agent_config, mock_ell):
    """Test parallel workflow execution."""
    engine = await anext(workflow_engine)
    agents = []
    workflow_ids = []
    
    # Create and register multiple agents
    for _ in range(3):
        agent = Agent(config=agent_config)
        agent._ell = mock_ell
        await agent.initialize()
        agent.metadata["test_mode"] = True
        agents.append(agent)
        
        workflow_id = await engine.register_workflow(agent)
        workflow_ids.append(workflow_id)
    
    # Execute workflows in parallel
    input_data = {"message": "Test input", "test_mode": True}  # Add test_mode flag
    tasks = [
        engine.execute_workflow(workflow_id, input_data)
        for workflow_id in workflow_ids
    ]
    
    results = await asyncio.gather(*tasks)
    assert all(result is not None for result in results)
    assert all("Test response" in result.get("content", "") for result in results)
    
    # Cleanup agents
    for agent in agents:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Test workflow cleanup."""
    engine = await anext(workflow_engine)
    agent_instance = await anext(agent)
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Execute workflow
    input_data = {"message": "Test input", "test_mode": True}  # Add test_mode flag
    await engine.execute_workflow(workflow_id, input_data)
    
    # Store mock references
    ell = mock_ell
    isa_manager = mock_isa_manager
    instruction_selector = mock_instruction_selector
    
    # Set engine's components to our mocks
    engine._ell = ell
    engine._isa_manager = isa_manager
    engine._instruction_selector = instruction_selector
    
    # Cleanup
    await engine.cleanup()
    
    # Verify cleanup
    assert not engine._initialized
    assert not engine.workflows
    assert ell.cleanup.called
    assert isa_manager.cleanup.called
    assert instruction_selector.cleanup.called
