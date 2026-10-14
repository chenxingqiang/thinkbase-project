"""Unit tests for agentic calling functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch
from typing import Dict, Any, List

from agentflow.core.workflow_engine import WorkflowEngine
from agentflow.core.workflow_types import WorkflowStep, WorkflowStepType, StepConfig
from agentflow.core.types import AgentStatus
from agentflow.agents.agent import Agent
from agentflow.core.config import AgentConfig, ModelConfig, WorkflowConfig
from agentflow.core.exceptions import WorkflowExecutionError

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
    await engine.initialize()  # No need to pass workflow_def here since it's handled in __init__
    try:
        yield engine
    finally:
        await engine.cleanup()

@pytest.fixture
def agent_config():
    """Create test agent config."""
    return AgentConfig(
        id="test-agent",
        name="Test Agent",
        description="Test agent for unit tests",
        type="generic",
        system_prompt="You are a test agent",
        model=ModelConfig(
            provider="default",
            name="gpt-3.5-turbo"
        ),
        workflow=WorkflowConfig(
            id="test-workflow",
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
        ),
        config={
            "algorithm": "PPO"
        }
    )

@pytest.fixture
async def agent(agent_config):
    """Create and initialize test agent."""
    _agent = Agent(config=agent_config)
    await _agent.initialize()
    _agent.metadata["test_mode"] = True
    try:
        yield _agent
    finally:
        await _agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_execution_basic(workflow_engine, agent):
    """Test basic workflow execution."""
    engine = workflow_engine
    agent_instance = agent
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input", "data": "Test data content"}
    
    result = await engine.execute_workflow(workflow_id, input_data)
    assert result is not None
    assert "steps" in result
    assert "test-step-1" in result["steps"]
    assert result["steps"]["test-step-1"]["status"] == "success"

@pytest.mark.asyncio
async def test_empty_workflow(workflow_engine):
    """Test execution of empty workflow."""
    engine = workflow_engine
    with pytest.raises(WorkflowExecutionError):
        await engine.execute_workflow("nonexistent", {})

@pytest.mark.asyncio
async def test_workflow_error_handling(workflow_engine, agent):
    """Test error handling during workflow execution."""
    engine = workflow_engine
    agent_instance = agent
    
    workflow_id = await engine.register_workflow(agent_instance)
    
    # Test with invalid input
    with pytest.raises(Exception):
        await engine.execute_workflow(workflow_id, None)

@pytest.mark.asyncio
async def test_workflow_timeout(workflow_engine, agent):
    """Test workflow timeout functionality."""
    engine = workflow_engine
    agent_instance = agent
    
    # Create workflow config with short timeout
    workflow_config = WorkflowConfig(
        id="test-workflow-2",
        name="test_workflow",
        timeout=0.1,  # Very short timeout
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
    
    # Register workflow with timeout config
    workflow_id = await engine.register_workflow(agent_instance, workflow_config)
    input_data = {"message": "Test input", "data": "Test data content"}
    
    # Mock execute to take longer than timeout
    async def slow_execution(*args, **kwargs):
        await asyncio.sleep(0.2)  # Longer than timeout
        return {"content": "Test response", "status": "success", "steps": []}
    
    with patch('agentflow.core.workflow_executor.WorkflowExecutor._execute_step', side_effect=slow_execution):
        with pytest.raises(TimeoutError):
            await engine.execute_workflow(workflow_id, input_data)

@pytest.mark.asyncio
async def test_parallel_workflow_execution(workflow_engine, agent_config):
    """Test parallel workflow execution."""
    engine = workflow_engine
    agents = []
    workflow_ids = []
    
    # Create and register multiple agents
    for _ in range(3):
        agent = Agent(config=agent_config)
        await agent.initialize()
        agent.metadata["test_mode"] = True
        agents.append(agent)
        
        workflow_id = await engine.register_workflow(agent)
        workflow_ids.append(workflow_id)
    
    # Execute workflows in parallel
    input_data = {"message": "Test input", "data": "Test data content"}
    tasks = [
        engine.execute_workflow(workflow_id, input_data)
        for workflow_id in workflow_ids
    ]
    
    results = await asyncio.gather(*tasks)
    assert all(result is not None for result in results)
    assert all(result["status"] == "success" for result in results)
    
    # Cleanup agents
    for agent in agents:
        await agent.cleanup()

@pytest.mark.asyncio
async def test_workflow_cleanup(workflow_engine, agent):
    """Test workflow cleanup."""
    engine = workflow_engine
    agent_instance = agent
    
    workflow_id = await engine.register_workflow(agent_instance)
    input_data = {"message": "Test input", "data": "Test data content"}
    
    await engine.execute_workflow(workflow_id, input_data)
    await engine.cleanup()
    
    assert not engine._initialized
    assert not engine.workflows 