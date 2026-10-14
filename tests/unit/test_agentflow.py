"""
Tests for AgentFlow system functionality
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
    WorkflowStrategy,
    WorkflowStatus,
    WorkflowStepStatus
)
from agentflow.core.enums import StepStatus
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
                    params={"substrategy": "standard"},
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

@pytest.fixture(scope="function")
async def workflow_engine(mock_ell, mock_isa_manager, mock_instruction_selector, mock_state_manager, mock_metrics_manager):
    """Create a workflow engine for testing."""
    engine = WorkflowEngine()
    engine._ell = mock_ell
    engine._isa_manager = mock_isa_manager
    engine._instruction_selector = mock_instruction_selector
    engine.state_manager = mock_state_manager
    engine.metrics = mock_metrics_manager
    await engine.initialize()
    return engine

@pytest.mark.asyncio
async def test_workflow_initialization(workflow_engine):
    """Test workflow engine initialization."""
    engine = workflow_engine
    assert engine is not None
    assert engine.state_manager is not None
    assert engine.metrics is not None

@pytest.mark.asyncio
async def test_agent_registration(workflow_engine, agent):
    """Test agent registration in workflow."""
    engine = workflow_engine
    test_agent = agent
    
    # Create workflow config from agent's workflow
    workflow_config = WorkflowConfig(
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
                description="Test transform step for workflow registration",
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    assert test_agent.id in engine.workflows

@pytest.mark.asyncio
async def test_workflow_execution(workflow_engine, agent, mock_ell, mock_isa_manager):
    """Test workflow execution."""
    engine = workflow_engine
    test_agent = agent
    
    # Configure mock_ell
    mock_ell.process_message.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Test response",
        type=MessageType.TOOL_RESULT,
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": MessageType.TOOL_RESULT,
            "status": "success"
        }
    )
    
    # Create workflow config
    async def test_transform(step, context):
        """Mock transform function."""
        data = context.content if context else "test"
        return {"data": f"processed_{data}"}

    workflow_config = WorkflowConfig(
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
                        "execute": test_transform
                    },
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    
    # Create a proper Message object
    message = Message(
        role=MessageRole.USER,
        content="Test input",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Execute workflow
    result = await engine.execute_workflow(test_agent.id, message)
    
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["status"] == "success"
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_workflow_error_handling():
    """Test workflow error handling."""
    # Create agent with test mode enabled
    agent = Agent(id="test-agent", name="Test Agent", type="generic")
    agent.metadata["test_mode"] = True

    # Configure mock to raise error for this test
    mock_ell = AsyncMock()
    mock_ell.process_message.side_effect = WorkflowExecutionError("Step failed due to should_fail flag")
    agent._ell = mock_ell

    # Create workflow engine
    engine = WorkflowEngine()
    # Register agent
    engine.agents[agent.id] = agent

    # Create workflow with single step
    async def fail_step(step, context):
        """Mock step that fails."""
        raise WorkflowExecutionError("Step failed due to should_fail flag")

    workflow = WorkflowConfig(
        id="test-workflow",
        name="Test Workflow",
        steps=[
            WorkflowStep(
                id="step-1",
                name="test_step",
                type=WorkflowStepType.AGENT,
                description="Test step",
                config=StepConfig(
                    strategy=WorkflowStrategy.CUSTOM,
                    params={
                        "should_fail": True,
                        "execute": fail_step
                    }
                )
            )
        ]
    )

    # Register workflow
    await engine.register_workflow(agent, workflow)

    # Execute workflow with test context
    message = Message(
        role=MessageRole.USER,
        content="Test input",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT,
            "test_mode": True
        }
    )

    with pytest.raises(WorkflowExecutionError) as exc_info:
        await engine.execute_workflow(agent.id, message)

    # Verify error message
    expected_error = "Step step-1 failed: Step failed due to should_fail flag"
    assert str(exc_info.value) == expected_error

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Test parallel workflow execution."""
    engine = workflow_engine
    
    # Create multiple test agents
    num_agents = 3
    agents = []
    
    for i in range(num_agents):
        async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
            return {"result": "test value"}

        step = WorkflowStep(
            id=f"step-{i}",
            name=f"step_{i}",
            type=WorkflowStepType.TRANSFORM,
            description=f"Test transform step {i} for parallel execution",
            dependencies=[],
            config=StepConfig(
                strategy="custom",
                params={},
                execute=test_execute,
                retry_delay=1.0,
                retry_backoff=2.0,
                max_retries=3
            )
        )

        workflow_cfg = WorkflowConfig(
            id=str(uuid.uuid4()),
            name=f"test_workflow_{i}",
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
            steps=[step]
        )

        agent_config = AgentConfig(
            name=f"test_agent_{i}",
            type="generic",
            mode="parallel",
            model=ModelConfig(name="gpt-4", provider="openai"),
            workflow=workflow_cfg.model_dump()
        )

        agent = Agent(
            id=str(uuid.uuid4()),
            name=agent_config.name,
            type=agent_config.type,
            mode=agent_config.mode,
            config=agent_config.model_dump(),
            workflow=workflow_cfg.model_dump(),
            isa_manager=mock_isa_manager,
            instruction_selector=mock_instruction_selector
        )
        agent._ell = mock_ell
        agent._isa_manager = mock_isa_manager
        agent._instruction_selector = mock_instruction_selector
        await agent.initialize()
        agents.append(agent)

        # Register workflow
        workflow_config = WorkflowConfig.model_validate(workflow_cfg.model_dump())
        await engine.register_workflow(agent, workflow_config)

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Test workflow cleanup."""
    engine = workflow_engine
    test_agent = agent
    
    # Create workflow config
    async def test_transform(step, context):
        """Mock transform function."""
        data = context.content if context else "test"
        return {"data": f"processed_{data}"}

    workflow_config = WorkflowConfig(
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
                description="Test transform step for workflow cleanup",
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={
                        "substrategy": "standard",
                        "execute": test_transform
                    },
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )
    
    # Register workflow
    await engine.register_workflow(test_agent, workflow_config)
    
    # Create a proper Message object
    message = Message(
        role=MessageRole.USER,
        content="Test input",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Execute workflow
    result = await engine.execute_workflow(test_agent.id, message)
    
    assert result is not None
    assert "steps" in result
    assert "step1" in result["steps"]
    assert result["status"] == "success"
    assert result["steps"]["step1"]["status"] == "success"

@pytest.mark.asyncio
async def test_workflow_registration(workflow_engine, mock_ell, mock_isa_manager, mock_instruction_selector):
    """Test workflow registration."""
    engine = workflow_engine
    # Create agent
    agent_id = str(uuid.uuid4())
    agent_name = "Test Agent"

    async def test_execute(context: Dict[str, Any]) -> Dict[str, Any]:
        return {"result": "test value"}

    # Create workflow configuration first
    workflow_cfg = WorkflowConfig(
        id="test-workflow-reg",
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
                description="Test transform step for workflow registration",
                dependencies=[],
                config=StepConfig(
                    strategy="standard",
                    params={"substrategy": "standard"},
                    execute=test_execute,
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    config = AgentConfig(
        id=agent_id,
        name=agent_name,
        type="generic",
        version="1.0.0",
        mode="async",
        parameters={},
        metadata={"test_mode": True},
        model=ModelConfig(name="gpt-4", provider="openai"),
        system_prompt="You are a test agent for workflow engine testing",
        workflow=workflow_cfg.model_dump()
    )

    # Create agent instance
    agent = Agent(config=config)

    # Set up mocks before initialization
    mock_ell = AsyncMock()
    mock_ell.process_message.return_value = Message(
        role=MessageRole.ASSISTANT,
        content="Test response",
        type=MessageType.TOOL_RESULT,
        metadata={
            "role": MessageRole.ASSISTANT,
            "type": MessageType.TOOL_RESULT,
            "status": "completed"
        }
    )
    agent._ell = mock_ell

    mock_isa_manager = AsyncMock()
    mock_isa_manager.get_isa.return_value = "test_isa"
    agent._isa_manager = mock_isa_manager

    mock_instruction_selector = AsyncMock()
    mock_instruction_selector.select_instruction.return_value = "test_instruction"
    agent._instruction_selector = mock_instruction_selector

    # Initialize agent after setting up mocks
    await agent.initialize()

    # Register workflow
    workflow_config = workflow_cfg.model_copy()
    await engine.register_workflow(agent, workflow_config)

    # Verify registration
    assert agent.id in engine.workflows
    registered_workflow = engine.workflows[agent.id]
    assert registered_workflow.id == workflow_cfg.id
    assert registered_workflow.name == workflow_cfg.name
    assert registered_workflow.max_iterations == workflow_cfg.max_iterations
    assert len(registered_workflow.steps) == len(workflow_cfg.steps)

@pytest.mark.asyncio
async def test_workflow_engine_execution():
    """Test workflow engine execution."""
    engine = WorkflowEngine()
    await engine.initialize()
    
    # Create agent
    agent_id = str(uuid.uuid4())
    agent_name = "Test Agent"

    # Create workflow configuration
    async def fail_step(step, context):
        """Mock step that fails."""
        raise WorkflowExecutionError("Step failed due to should_fail flag")

    workflow_config = WorkflowConfig(
        id="test-workflow-exec",
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
                id="step-1",
                name="Test Step 1",
                type=WorkflowStepType.AGENT,
                description="Test transform step for workflow engine execution",
                dependencies=[],
                config=StepConfig(
                    strategy=WorkflowStrategy.CUSTOM,
                    params={
                        "should_fail": True,
                        "execute": fail_step
                    },
                    retry_delay=1.0,
                    retry_backoff=2.0,
                    max_retries=3
                )
            )
        ]
    )

    # Create agent configuration
    config = AgentConfig(
        id=agent_id,
        name=agent_name,
        type="generic",
        version="1.0.0",
        mode="async",
        parameters={},
        metadata={"test_mode": True},
        model=ModelConfig(name="gpt-4", provider="openai"),
        system_prompt="You are a test agent for workflow engine testing",
        workflow=workflow_config.model_dump()
    )

    # Create agent instance
    agent = Agent(config=config)

    # Set up mocks before initialization
    mock_ell = AsyncMock()
    mock_ell.process_message.side_effect = WorkflowExecutionError("Step failed due to should_fail flag")
    agent._ell = mock_ell

    mock_isa_manager = AsyncMock()
    mock_isa_manager.get_isa.return_value = "test_isa"
    agent._isa_manager = mock_isa_manager

    mock_instruction_selector = AsyncMock()
    mock_instruction_selector.select_instruction.return_value = "test_instruction"
    agent._instruction_selector = mock_instruction_selector

    # Initialize agent after setting up mocks
    await agent.initialize()

    # Register workflow
    await engine.register_workflow(agent, workflow_config)

    # Create test input data
    message = Message(
        role=MessageRole.USER,
        content="Test input",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT,
            "test_mode": True
        }
    )

    # Test error handling
    with pytest.raises(WorkflowExecutionError) as exc_info:
        await engine.execute_workflow(agent.id, message)

    # Verify error message
    expected_error = "Step step-1 failed: Step failed due to should_fail flag"
    assert str(exc_info.value) == expected_error
