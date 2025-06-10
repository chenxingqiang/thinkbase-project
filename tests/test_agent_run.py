"""Test agent run module."""

import pytest
import numpy as np
from unittest.mock import AsyncMock, MagicMock
from agentflow.core.config import AgentConfig, ConfigurationType, ModelConfig
from agentflow.core.workflow_types import WorkflowConfig, WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.agents.agent_types import AgentMode
from agentflow.agents.agent import Agent
from agentflow.ell.types.message import Message, MessageRole, MessageType, ContentBlock, ContentWrapper
import json

@pytest.fixture
def mock_ell():
    """Create a mock ELL integration."""
    mock = MagicMock()
    mock.enabled = True
    mock.tracking_enabled = True
    mock.config = {}
    mock.metrics = {
        "function_calls": 0,
        "total_execution_time": 0.0,
        "errors": 0
    }
    
    async def mock_process(message: Message) -> Message:
        # Convert numpy array to list for JSON serialization
        content = message.content
        if not isinstance(content, dict):
            content = {"data": None}
        data = content.get("data")
        if not isinstance(data, np.ndarray):
            return Message(
                role=MessageRole.ASSISTANT,
                type=MessageType.ERROR,
                text="Invalid data format. Expected numpy array."
            )
        
        # Convert numpy array to string with each row on a new line
        data_rows = []
        for row in data:
            # Preserve brackets and convert to string
            row_str = f"[{', '.join(map(str, row.tolist()))}]"
            data_rows.append(row_str)
        data_str = "\n".join(data_rows)
        
        return Message(
            role=MessageRole.ASSISTANT,
            type=MessageType.RESULT,
            text=data_str
        )
    
    mock.process_message = AsyncMock(side_effect=mock_process)
    mock.configure = MagicMock()
    mock.cleanup = AsyncMock()
    return mock

@pytest.mark.asyncio
async def test_agent_run(mock_ell):
    """Test agent run functionality."""
    workflow = WorkflowConfig(
        id="test-workflow-1",
        name="test_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Feature engineering step for data transformation",
                config=StepConfig(
                    strategy="standard",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True
                    }
                )
            )
        ]
    )
    agent_config = AgentConfig(
        id="test-agent-1",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    agent = Agent(config=agent_config)
    agent._ell = mock_ell  # Set the mock ELL
    await agent.initialize()
    
    try:
        data = np.random.randn(10, 2)
        result = await agent.execute({"data": data, "test_mode": True})
        assert result is not None
        # The result should be a string since that's what the Agent.execute method returns
        assert isinstance(result, str)
        # The result should contain the data array representation
        assert "[" in result and "]" in result
        assert len(result.split("\n")) == 10  # 10 rows of data
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_run_with_dependencies(mock_ell):
    """Test agent run with workflow dependencies."""
    workflow = WorkflowConfig(
        id="test-workflow-2",
        name="dependency_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Initial feature engineering step",
                config=StepConfig(
                    strategy="standard",
                    params={
                        "method": "standard",
                        "with_mean": True,
                        "with_std": True
                    }
                )
            ),
            WorkflowStep(
                id="step-2",
                name="step_2",
                type=WorkflowStepType.TRANSFORM,
                description="Outlier removal step",
                dependencies=["step-1"],
                config=StepConfig(
                    strategy="standard",
                    params={
                        "method": "isolation_forest",
                        "threshold": 0.1
                    }
                )
            )
        ]
    )
    agent_config = AgentConfig(
        id="test-agent-2",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    agent = Agent(config=agent_config)
    agent._ell = mock_ell  # Set the mock ELL
    await agent.initialize()
    
    try:
        data = np.random.randn(100, 2)  # More data points for outlier detection
        result = await agent.execute({"data": data, "test_mode": True})
        assert result is not None
        # The result should be a string since that's what the Agent.execute method returns
        assert isinstance(result, str)
        # The result should contain the data array representation
        assert "[" in result and "]" in result
        assert len(result.split("\n")) == 100  # 100 rows of data
    finally:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_agent_run_error_handling(mock_ell):
    """Test agent run error handling."""
    workflow = WorkflowConfig(
        id="test-workflow-3",
        name="error_workflow",
        max_iterations=5,
        timeout=30,
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="Step that should fail",
                config=StepConfig(
                    strategy="standard",  # Use a valid strategy
                    params={"should_fail": True}  # This will trigger an error during execution
                )
            )
        ]
    )
    agent_config = AgentConfig(
        id="test-agent-3",
        name="test_agent",
        type=ConfigurationType.DATA_SCIENCE,
        mode=AgentMode.SEQUENTIAL,
        version="1.0.0",
        model=ModelConfig(
            provider="openai",
            name="gpt-4",
            temperature=0.7,
            max_tokens=4096
        ),
        workflow=workflow
    )
    agent = Agent(config=agent_config)
    
    # Configure mock to raise an error
    mock_ell.process_message.side_effect = WorkflowExecutionError("Test error during execution")
    agent._ell = mock_ell
    await agent.initialize()
    
    try:
        data = np.random.randn(10, 2)
        with pytest.raises(WorkflowExecutionError, match="Test error during execution"):
            await agent.execute({"data": data, "test_mode": True, "should_fail": True})
    finally:
        await agent.cleanup()
