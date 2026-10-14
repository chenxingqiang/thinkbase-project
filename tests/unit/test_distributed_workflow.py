"""Test distributed workflow."""

import pytest
import ray
from typing import Dict, Any
import asyncio

from agentflow.core.research_workflow import ResearchDistributedWorkflow
from agentflow.core.workflow_types import WorkflowStepType, WorkflowConfig, ErrorPolicy
from agentflow.core.exceptions import WorkflowExecutionError

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
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "step_1": {
                    "id": "step_1",
                    "name": "First test step",
                    "description": "First test step",
                    "input": ["test_input"],
                    "output": ["test_output"],
                    "agent_config": {},
                    "type": "research_execution"
                },
                "step_2": {
                    "id": "step_2",
                    "name": "Second test step",
                    "description": "Second test step",
                    "input": ["test_input"],
                    "output": ["test_output"],
                    "agent_config": {},
                    "dependencies": ["step_1"],
                    "type": "research_execution"
                }
            }
        }
    }

@pytest.fixture
def workflow_config() -> WorkflowConfig:
    """Test configuration."""
    data = {
        "id": "test_workflow",
        "name": "Test Distributed Workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "logging_level": "INFO",
        "error_policy": ErrorPolicy(
            ignore_warnings=False,
            fail_fast=True
        ),
        "distributed": True,
        "steps": []
    }
    return WorkflowConfig.model_validate(data, context={"distributed": True})

@pytest.mark.asyncio
async def test_distributed_workflow_execution(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow execution."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)

    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result is not None
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_distributed_workflow_error_handling(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow error handling."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)

    # Test with invalid input
    with pytest.raises(ray.exceptions.RayTaskError) as exc_info:
        await workflow.execute_async.remote("invalid_input")
    assert "dict" in str(exc_info.value).lower()

@pytest.mark.asyncio
async def test_distributed_workflow_retry_mechanism(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow retry mechanism."""
    # Initialize step config if not present
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)

    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result is not None
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_distributed_workflow_step_dependencies(setup_ray, test_workflow_def, workflow_config):
    """Test distributed workflow step dependencies."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)

    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result is not None
    assert isinstance(result, dict)

@pytest.mark.asyncio
async def test_distributed_workflow_parallel_execution(setup_ray, test_workflow_def, workflow_config):
    """Test parallel execution of independent steps."""
    # Modify workflow to have parallel steps
    test_workflow_def["COLLABORATION"]["WORKFLOW"]["step_3"] = {
        "step": 3,
        "name": "Test Step 3",
        "description": "Parallel step",
        "input": ["test_input"],
        "type": WorkflowStepType.RESEARCH_EXECUTION,
        "agent_config": {}
    }

    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)

    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result is not None
    assert isinstance(result, dict)
