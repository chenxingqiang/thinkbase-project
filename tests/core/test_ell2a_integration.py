"""Tests for ELL integration functionality."""

import pytest
import asyncio
from typing import Dict, Any
from agentflow.core.ell_integration import ell_integration
from agentflow.core.workflow_types import WorkflowConfig, ErrorPolicy, WorkflowStep, StepConfig, WorkflowStepType
from agentflow.ell.types.message import Message, MessageRole, MessageType
from agentflow.ell.lmp import LMPType
from agentflow.ell.workflow import ELLWorkflow
import uuid

@pytest.fixture(autouse=True)
def cleanup_ell():
    """Clean up ELL integration state before each test."""
    ell_integration.cleanup()
    yield
    ell_integration.cleanup()

@pytest.fixture
def sample_workflow_config():
    """Create a sample workflow configuration."""
    return WorkflowConfig(
        id=str(uuid.uuid4()),
        name="Test Workflow",
        max_iterations=10,
        timeout=30,
        error_policy=ErrorPolicy(),
        steps=[
            WorkflowStep(
                id="step-1",
                name="step_1",
                type=WorkflowStepType.TRANSFORM,
                description="A test transformation step",
                config=StepConfig(
                    strategy="standard",
                    params={"test": "value"}
                )
            )
        ],
        distributed=False,
        metadata={
            "ell_config": {
                "simple": {
                    "model": "test-model",
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 30.0
                },
                "complex": {
                    "model": "test-model-complex",
                    "track_performance": True,
                    "track_memory": True,
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 60.0,
                    "stream": True
                }
            }
        }
    )

@pytest.mark.asyncio
async def test_ell_message_handling():
    """Test ELL message creation and handling."""
    # Create a message
    message = Message(
        role=MessageRole.USER,
        content="Test message",
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    assert message.role == MessageRole.USER
    assert message.content == "Test message"
    assert message.metadata["type"] == MessageType.TEXT

@pytest.mark.asyncio
async def test_ell_workflow_registration():
    """Test ELL workflow registration and management."""
    workflow = ELLWorkflow(
        name="test-workflow",
        description="Test workflow"
    )
    
    # Register workflow
    ell_integration.register_workflow(workflow.id, workflow)
    
    # Verify registration
    workflows = ell_integration.list_workflows()
    assert workflow.id in workflows
    
    # Unregister workflow
    ell_integration.unregister_workflow(workflow.id)
    workflows = ell_integration.list_workflows()
    assert workflow.id not in workflows

@pytest.mark.asyncio
async def test_ell_function_decoration():
    """Test ELL function decoration."""
    
    @ell_integration.with_ell(mode="simple")
    async def test_function():
        return Message(
            role=MessageRole.ASSISTANT,
            content="success",
            metadata={"type": "result"}
        )
    
    result = await test_function()
    assert result.content == "success"
    
    @ell_integration.with_ell(mode="complex")
    async def test_complex_function():
        return Message(
            role=MessageRole.ASSISTANT,
            content="complex success",
            metadata={"type": "result"}
        )
    
    result = await test_complex_function()
    assert result.content == "complex success"

@pytest.mark.asyncio
async def test_ell_performance_tracking():
    """Test ELL performance tracking."""
    
    @ell_integration.track_function()
    async def tracked_function():
        await asyncio.sleep(0.1)  # Simulate work
        return Message(
            role=MessageRole.ASSISTANT,
            content="tracked",
            metadata={"type": "result"}
        )
    
    result = await tracked_function()
    metrics = ell_integration.get_metrics()
    
    assert "tracked_function" in metrics
    assert metrics["tracked_function"]["calls"] > 0
    assert metrics["tracked_function"]["total_time"] > 0

@pytest.mark.asyncio
async def test_ell_mode_configuration(sample_workflow_config):
    """Test ELL mode configuration."""
    # Configure ELL
    ell_config = {
        "simple": {
            "model": "test-model",
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 30.0
        },
        "complex": {
            "model": "test-model-complex",
            "track_performance": True,
            "track_memory": True,
            "max_retries": 3,
            "retry_delay": 1.0,
            "timeout": 60.0,
            "stream": True
        }
    }
    ell_integration.configure(ell_config)
    
    # Get mode config
    simple_config = ell_integration.get_mode_config("simple")
    complex_config = ell_integration.get_mode_config("complex")
    
    assert simple_config["model"] == "test-model"
    assert complex_config["track_performance"] == True
    assert complex_config["track_memory"] == True

@pytest.mark.asyncio
async def test_ell_error_handling():
    """Test ELL error handling."""
    
    @ell_integration.with_ell(mode="simple")
    async def failing_function():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError):
        await failing_function()
    
    metrics = ell_integration.get_metrics()
    assert "failing_function" in metrics
    assert metrics["failing_function"]["errors"] > 0

@pytest.mark.asyncio
async def test_ell_context_management():
    """Test ELL context management."""
    # Create test messages
    messages = [
        Message(
            role=MessageRole.USER,
            content=f"Message {i}",
            type=MessageType.TEXT,
            metadata={
                "role": MessageRole.USER,
                "type": MessageType.TEXT
            }
        ) for i in range(3)
    ]
    
    # Process messages
    for msg in messages:
        await ell_integration.process_message(msg)
    
    # Get messages
    processed_messages = ell_integration.get_messages()
    assert len(processed_messages) == 3
    assert all(isinstance(msg, Message) for msg in processed_messages)

@pytest.mark.asyncio
async def test_ell_workflow_execution(sample_workflow_config):
    """Test ELL workflow execution."""
    # Create a message
    message = Message(
        role=MessageRole.USER,
        content="Test message",
        type=MessageType.TEXT,
        metadata={
            "role": MessageRole.USER,
            "type": MessageType.TEXT
        }
    )
    
    # Process message
    result = await ell_integration.process_message(message)
    assert result is not None
    assert isinstance(result, Message)
    assert result.type == MessageType.RESULT