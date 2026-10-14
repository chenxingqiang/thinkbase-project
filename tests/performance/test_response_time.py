"""Test response time performance."""

import pytest
import ray
import time
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
        "INPUT": ["test_input"],
        "OUTPUT": ["test_output"],
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
    }

@pytest.fixture
def workflow_config() -> Dict[str, Any]:
    """Test configuration."""
    return {
        "max_retries": 3,
        "retry_delay": 0.1,
        "retry_backoff": 1.5,
        "step_1_config": {
            "max_retries": 3,
            "retry_delay": 0.1,
            "retry_backoff": 1.5
        },
        "step_2_config": {
            "max_retries": 3,
            "retry_delay": 0.1,
            "retry_backoff": 1.5
        }
    }

@pytest.mark.asyncio
async def test_workflow_response_time(setup_ray, test_workflow_def, workflow_config):
    """Test workflow response time."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow and measure response time
    input_data = {"test_input": "test_value"}
    
    start_time = time.time()
    result = await workflow.execute_async.remote(input_data)
    end_time = time.time()
    
    response_time = end_time - start_time
    
    # Verify result
    assert result["status"] == "success"
    assert "results" in result
    
    # Check response time
    assert response_time < 10  # Maximum allowed response time in seconds

@pytest.mark.asyncio
async def test_workflow_step_response_time(setup_ray, test_workflow_def, workflow_config):
    """Test individual step response times."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Set distributed steps
    steps = ["step_1", "step_2"]
    await workflow.set_distributed_steps.remote(steps)
    
    # Execute workflow and measure step response times
    input_data = {"test_input": "test_value"}
    
    start_time = time.time()
    result = await workflow.execute_async.remote(input_data)
    end_time = time.time()
    
    total_time = end_time - start_time
    
    # Verify result
    assert result["status"] == "success"
    assert "results" in result
    
    # Check step response times
    assert total_time < 15  # Maximum allowed total time in seconds

@pytest.mark.asyncio
async def test_workflow_retry_response_time(setup_ray, test_workflow_def, workflow_config):
    """Test response time with retries."""
    workflow = await ResearchDistributedWorkflow.create_remote_workflow(test_workflow_def, workflow_config)
    assert isinstance(workflow, ray.actor.ActorHandle)
    
    # Execute workflow multiple times
    num_executions = 3
    response_times = []
    input_data = {"test_input": "test_value"}
    
    for _ in range(num_executions):
        start_time = time.time()
        result = await workflow.execute_async.remote(input_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        response_times.append(response_time)
        
        # Verify result
        assert result["status"] == "success"
        assert "results" in result
    
    # Check response times
    avg_response_time = sum(response_times) / len(response_times)
    assert avg_response_time < 12  # Maximum allowed average response time in seconds
    assert max(response_times) < 15  # Maximum allowed single response time in seconds

if __name__ == "__main__":
    pytest.main([__file__, "-v"])