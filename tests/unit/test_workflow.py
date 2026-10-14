"""Tests for workflow functionality."""
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from agentflow.core.workflow import WorkflowEngine, WorkflowInstance
from agentflow.core.metrics import MetricType
from agentflow.core.workflow_types import (
    WorkflowConfig, WorkflowStep, StepConfig, WorkflowStepType,
    ErrorPolicy, RetryPolicy
)
from agentflow.core.enums import WorkflowStatus
from agentflow.agents.agent_types import AgentType
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.config import AgentConfig, ModelConfig
from agentflow.agents.agent import Agent

class MockAgent:
    """Mock agent for testing."""
    
    def __init__(self, name, async_mode=False):
        self.name = name
        self.async_mode = async_mode
        self.executed = False
    
    async def initialize(self):
        """Initialize mock agent."""
        pass
    
    async def execute(self, context):
        self.executed = True
        return {"result": f"{self.name}_result"}

class TestWorkflowExecution:
    """Test workflow execution."""
    
    @pytest.fixture
    def workflow_config(self):
        """Create workflow configuration."""
        return WorkflowConfig(
            id="test-workflow",
            name="Test Workflow",
            max_iterations=5,
            timeout=3600,
            error_policy=ErrorPolicy(
                fail_fast=True,
                ignore_warnings=False,
                max_errors=10,
                retry_policy=RetryPolicy(
                    max_retries=3,
                    retry_delay=1.0,
                    backoff=2.0,
                    max_delay=60.0
                )
            ),
            steps=[
                WorkflowStep(
                    id="step1",
                    name="Agent Step",
                    type=WorkflowStepType.AGENT,
                    description="Test agent step for workflow execution",
                    config=StepConfig(
                        strategy="standard",
                        params={}
                    )
                )
            ]
        )
    
    @pytest.fixture
    async def test_agent(self, workflow_config):
        """Create a test agent."""
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
            workflow=workflow_config
        )
        agent = Agent(config=agent_config)
        await agent.initialize()
        return agent
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, workflow_config, test_agent):
        """Test workflow execution."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Get the agent instance
        agent = test_agent
        
        # Register the agent
        await engine.register_workflow(agent, workflow_config)
        
        # Create workflow instance
        instance = await engine.create_workflow("test_workflow", workflow_config)
        instance.context["test_mode"] = True
        
        # Execute workflow
        result = await engine.execute_workflow(agent.id, instance.context)
        print("Workflow Execution Result:", result)  # Debug print
        
        assert result is not None
        assert result["status"] == "success"
        assert "steps" in result
        assert len(result["steps"]) > 0
        assert isinstance(result["steps"], list)
        
        # Check first step
        first_step = result["steps"][0]
        assert first_step["id"] == "step1"
        assert first_step["type"] == "WorkflowStepType.AGENT"  # Exact string representation
        assert first_step["status"] == "success"
        assert "result" in first_step
        assert "content" in first_step["result"]
    
    @pytest.mark.asyncio
    async def test_workflow_validation(self, workflow_config, test_agent):
        """Test workflow validation."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Get the agent instance
        agent = test_agent
        
        # Register the agent
        await engine.register_workflow(agent, workflow_config)
        
        # Test with invalid workflow instance
        with pytest.raises(ValueError, match="Workflow instance has no steps"):
            instance = await engine.create_workflow("test_workflow")
            await engine.execute_workflow(agent.id, {})
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_config, test_agent):
        """Test workflow error handling."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Get the agent instance
        agent = test_agent
        
        # Register the agent
        await engine.register_workflow(agent, workflow_config)
        
        # Create workflow instance with failing step
        instance = await engine.create_workflow("test_workflow", workflow_config)
        
        # Modify the step to trigger an error
        instance.steps[0].config.params["should_fail"] = True  # Add parameter to trigger failure
        instance.context = {
            "test_mode": True,
            "should_fail": True,  # Add flag to trigger failure
            "message": ""  # Empty message to trigger validation error
        }
        
        # Execute workflow
        with pytest.raises(WorkflowExecutionError) as exc_info:
            await engine.execute_workflow_instance(instance)
        
        assert "Step execution failed" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_workflow_metrics(self, workflow_config, test_agent):
        """Test workflow metrics collection."""
        engine = WorkflowEngine(workflow_config)
        await engine.initialize()
        
        # Get the agent instance
        agent = test_agent
        
        # Register the agent
        await engine.register_workflow(agent, workflow_config)
        
        # Create workflow instance
        instance = await engine.create_workflow("test_workflow", workflow_config)
        context = {
            "test_mode": True,
            "metrics": {
                MetricType.LATENCY.value: [
                    {"value": 100, "timestamp": 1234567890}
                ]
            }
        }
        
        # Execute workflow
        result = await engine.execute_workflow(agent.id, context)
        
        assert result is not None
        assert result["status"] == "success"
        assert "steps" in result
        assert len(result["steps"]) > 0
        assert isinstance(result["steps"], list)  # Steps should be a list now