"""Test research workflow."""

import pytest
import ray
from typing import Dict, Any
import uuid

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
        "COLLABORATION": {
            "MODE": "SEQUENTIAL",
            "COMMUNICATION_PROTOCOL": {
                "TYPE": "HIERARCHICAL"
            },
            "WORKFLOW": {
                "step_1": {
                    "step": 1,
                    "name": "Research Step",
                    "description": "Research test step",
                    "input": ["test_input"],
                    "type": WorkflowStepType.RESEARCH_EXECUTION,
                    "agent_config": {}
                },
                "step_2": {
                    "step": 2,
                    "name": "Document Step",
                    "description": "Document test step",
                    "input": ["test_input"],
                    "dependencies": ["step_1"],
                    "type": WorkflowStepType.DOCUMENT_GENERATION,
                    "agent_config": {}
                }
            }
        },
        "INPUT": ["test_input"],
        "OUTPUT": ["test_output"]
    }

@pytest.fixture
def workflow_config():
    """Create test workflow configuration."""
    return {
        "id": str(uuid.uuid4()),
        "name": "test_research_workflow",
        "max_iterations": 5,
        "timeout": 3600,
        "error_policy": {
            "fail_fast": True,
            "ignore_warnings": False,
            "max_errors": 10,
            "retry_policy": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "backoff": 2.0,
                "max_delay": 60.0
            }
        },
        "steps": [
            {
                "id": "step1",
                "name": "research_step",
                "type": "research_execution",
                "description": "Research execution step",
                "config": {
                    "strategy": "standard",
                    "params": {
                        "research_topic": "Test Research",
                        "methodology": "Test Methodology"
                    }
                }
            }
        ],
        "distributed": True,
        "agents": {},
        "error_handling": {},
        "input_data": {},
        "output_data": {}
    }

@pytest.mark.asyncio
async def test_research_workflow_execution(setup_ray, test_workflow_def, workflow_config):
    """Test research workflow execution."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow
    input_data = {"test_input": "test_value"}
    result = await workflow.execute_async.remote(input_data)
    assert result["status"] == "success"
    assert "results" in result

@pytest.mark.asyncio
async def test_research_workflow_error_handling(setup_ray, test_workflow_def, workflow_config):
    """Test research workflow error handling."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Test with invalid input
    with pytest.raises(ray.exceptions.RayTaskError):
        await workflow.execute_async.remote("invalid_input")

@pytest.mark.asyncio
async def test_research_workflow_retry_mechanism(setup_ray, test_workflow_def, workflow_config):
    """Test research workflow retry mechanism."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Set distributed steps
    steps = ["step_1", "step_2"]
    await workflow.set_distributed_steps.remote(steps)
    assert await workflow.get_distributed_steps.remote() == steps

@pytest.mark.asyncio
async def test_research_workflow_step_dependencies(setup_ray, test_workflow_def, workflow_config):
    """Test research workflow step dependencies."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Set distributed steps
    steps = ["step_1", "step_2"]
    await workflow.set_distributed_steps.remote(steps)
    assert await workflow.get_distributed_steps.remote() == steps
