"""Test agent workflow."""

import pytest
import ray
from typing import Dict, Any

from agentflow.core.research_workflow import ResearchDistributedWorkflow
from agentflow.core.workflow_types import WorkflowStepType

@pytest.fixture(scope="session")
def setup_ray():
    """Setup Ray (session-scoped: one cluster per module, not per test)."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield None
    if ray.is_initialized():
        ray.shutdown()

@pytest.fixture
def test_workflow_def() -> Dict[str, Any]:
    """Test workflow definition."""
    return {
        "id": "test-workflow-1",
        "name": "Test Workflow",
        "max_iterations": 5,
        "timeout": 300,
        "steps": [
            {
                "id": "step-1",
                "name": "Test Step 1",
                "type": WorkflowStepType.RESEARCH_EXECUTION,
                "description": "First test step",
                "config": {
                    "strategy": "standard",
                    "params": {}
                }
            }
        ]
    }

@pytest.fixture
def workflow_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "id": "test-workflow-2",
        "name": "Test Workflow 2",
        "max_retries": 3,
        "retry_delay": 0.1,
        "retry_backoff": 1.5,
        "steps": [
            {
                "id": "step-1",
                "name": "Test Step 1",
                "type": WorkflowStepType.RESEARCH_EXECUTION,
                "description": "First test step",
                "config": {
                    "strategy": "standard",
                    "params": {
                        "max_retries": 3,
                        "retry_delay": 0.1,
                        "retry_backoff": 1.5
                    }
                }
            }
        ]
    }

@pytest.mark.asyncio
async def test_agent_workflow_execution(setup_ray, test_workflow_def, workflow_config):
    """Test agent workflow execution."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result["status"] == "success"
    assert "results" in result

@pytest.mark.asyncio
async def test_agent_workflow_error_handling(setup_ray, test_workflow_def, workflow_config):
    """Test agent workflow error handling."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Test with invalid input
    with pytest.raises(ray.exceptions.RayTaskError):
        await workflow.execute_async.remote("invalid_input")

@pytest.mark.asyncio
async def test_agent_workflow_retry_mechanism(setup_ray, test_workflow_def, workflow_config):
    """Test agent workflow retry mechanism."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Set distributed steps
    steps = ["step_1"]
    await workflow.set_distributed_steps.remote(steps)
    assert await workflow.get_distributed_steps.remote() == steps

@pytest.mark.asyncio
async def test_agent_workflow_step_dependencies(setup_ray, test_workflow_def, workflow_config):
    """Test agent workflow step dependencies."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Set distributed steps
    steps = ["step_1"]
    await workflow.set_distributed_steps.remote(steps)
    assert await workflow.get_distributed_steps.remote() == steps
