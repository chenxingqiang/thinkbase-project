"""Test cases for advanced specialized instructions."""

import pytest
import asyncio
from typing import Dict, Any, List
import time
import numpy as np

from agentflow.core.instructions.specialized import (
    DataValidationInstruction,
    ModelTrainingInstruction,
    ErrorRecoveryInstruction,
    PerformanceMonitoringInstruction,
    PipelineOrchestrationInstruction
)
from agentflow.core.instructions.base import InstructionStatus

@pytest.fixture
def data_validator():
    """Create DataValidationInstruction instance."""
    return DataValidationInstruction.remote()

@pytest.fixture
def model_trainer():
    """Create ModelTrainingInstruction instance."""
    return ModelTrainingInstruction.remote()

@pytest.fixture
def error_recovery():
    """Create ErrorRecoveryInstruction instance."""
    return ErrorRecoveryInstruction.remote()

@pytest.fixture
def performance_monitor():
    """Create PerformanceMonitoringInstruction instance."""
    return PerformanceMonitoringInstruction.remote()

@pytest.fixture
def pipeline_orchestrator():
    """Create PipelineOrchestrationInstruction instance."""
    return PipelineOrchestrationInstruction.remote()

@pytest.mark.asyncio
async def test_data_validation_basic(data_validator):
    """Test basic data validation functionality."""
    # Define test schema
    schema = {
        "type": int,
        "range": (0, 100),
        "custom_validator": lambda x: x % 2 == 0  # Must be even
    }
    
    # Test valid data
    context = {
        "data": 42,
        "schema": "test_schema"
    }
    
    # Register schema
    data_validator._schemas["test_schema"] = schema
    
    result = await data_validator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["is_valid"]
    assert not result.data["errors"]
    
    # Test invalid data
    context["data"] = "invalid"
    result = await data_validator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert not result.data["is_valid"]
    assert len(result.data["errors"]) > 0

@pytest.mark.asyncio
async def test_model_training_basic(model_trainer):
    """Test basic model training functionality."""
    # Create dummy training data
    training_data = {
        "features": np.random.rand(100, 10),
        "labels": np.random.randint(0, 2, 100)
    }
    
    context = {
        "model_name": "test_model",
        "training_data": training_data,
        "hyperparameters": {
            "learning_rate": 0.01,
            "batch_size": 32
        }
    }
    
    result = await model_trainer.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "training_result" in result.data
    assert "metrics" in result.data

@pytest.mark.asyncio
async def test_error_recovery_basic(error_recovery):
    """Test basic error recovery functionality."""
    # Define test recovery strategy
    async def recovery_step(context):
        return {"success": True, "action": "Test recovery action"}
    
    error_recovery._recovery_strategies["test_error"] = [recovery_step]
    
    context = {
        "error_type": "test_error",
        "error_context": {"details": "Test error occurred"}
    }
    
    result = await error_recovery.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["recovery_successful"]
    assert len(result.data["actions_taken"]) > 0

@pytest.mark.asyncio
async def test_performance_monitoring_basic(performance_monitor):
    """Test basic performance monitoring functionality."""
    # Define test metrics
    metrics = {
        "cpu_usage": 45.5,
        "memory_usage": 1024.0,
        "latency": 0.1
    }
    
    # Define thresholds
    thresholds = {
        "cpu_usage": {
            "warning": 70.0,
            "critical": 90.0
        },
        "memory_usage": {
            "warning": 2048.0,
            "critical": 4096.0
        }
    }
    
    context = {
        "metrics": metrics,
        "thresholds": thresholds
    }
    
    result = await performance_monitor.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "current_metrics" in result.data
    assert "alerts" in result.data
    assert "insights" in result.data

@pytest.mark.asyncio
async def test_pipeline_orchestration_basic(pipeline_orchestrator):
    """Test basic pipeline orchestration functionality."""
    # Define test pipeline
    async def test_step(context):
        return {
            "status": InstructionStatus.COMPLETED,
            "data": {"result": "success"}
        }
    
    pipeline_config = {
        "steps": [
            {
                "name": "step1",
                "instruction": test_step,
                "input_mapping": {"input1": "pipeline_input1"}
            }
        ],
        "retry_policy": {
            "max_retries": 3,
            "retry_delay": 1.0
        }
    }
    
    context = {
        "pipeline_name": "test_pipeline",
        "pipeline_config": pipeline_config,
        "pipeline_input": {"pipeline_input1": "test_value"}
    }
    
    result = await pipeline_orchestrator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "execution_id" in result.data
    assert "result" in result.data

@pytest.mark.asyncio
async def test_data_validation_complex(data_validator):
    """Test complex data validation scenarios."""
    # Define complex schema with nested validation
    schema = {
        "type": dict,
        "custom_validator": lambda x: all(
            isinstance(v, (int, float)) for v in x.values()
        )
    }
    
    data_validator._schemas["complex_schema"] = schema
    
    # Test valid nested data
    context = {
        "data": {"a": 1, "b": 2.5, "c": 3},
        "schema": "complex_schema"
    }
    
    result = await data_validator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["is_valid"]
    
    # Test invalid nested data
    context["data"] = {"a": 1, "b": "invalid", "c": 3}
    result = await data_validator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert not result.data["is_valid"]

@pytest.mark.asyncio
async def test_model_training_with_validation(model_trainer):
    """Test model training with validation data."""
    # Create dummy training and validation data
    training_data = {
        "features": np.random.rand(100, 10),
        "labels": np.random.randint(0, 2, 100)
    }
    validation_data = {
        "features": np.random.rand(20, 10),
        "labels": np.random.randint(0, 2, 20)
    }
    
    context = {
        "model_name": "test_model",
        "training_data": training_data,
        "validation_data": validation_data,
        "hyperparameters": {
            "learning_rate": 0.01,
            "batch_size": 32
        }
    }
    
    result = await model_trainer.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "validation_metrics" in result.data["training_result"]

@pytest.mark.asyncio
async def test_error_recovery_multiple_strategies(error_recovery):
    """Test error recovery with multiple strategies."""
    # Define multiple recovery steps
    async def step1(context):
        return {"success": True, "action": "Recovery step 1"}
    
    async def step2(context):
        return {"success": True, "action": "Recovery step 2"}
    
    error_recovery._recovery_strategies["multi_step_error"] = [step1, step2]
    
    context = {
        "error_type": "multi_step_error",
        "error_context": {"details": "Test error occurred"}
    }
    
    result = await error_recovery.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["recovery_successful"]
    assert len(result.data["actions_taken"]) == 2

@pytest.mark.asyncio
async def test_performance_monitoring_trends(performance_monitor):
    """Test performance monitoring trend analysis."""
    # Simulate metrics over time
    for i in range(5):
        metrics = {
            "cpu_usage": 40.0 + i * 10,  # Increasing trend
            "memory_usage": 1024.0,
            "latency": 0.1
        }
        
        context = {"metrics": metrics}
        result = await performance_monitor.execute.remote(context)
        assert result.status == InstructionStatus.COMPLETED
    
    # Get insights
    result = await performance_monitor.execute.remote({"metrics": metrics})
    insights = result.data["insights"]
    
    # Should detect increasing trend in CPU usage
    cpu_trends = [
        insight for insight in insights
        if insight["type"] == "trend" and insight["metric"] == "cpu_usage"
    ]
    assert len(cpu_trends) > 0
    assert cpu_trends[0]["data"]["direction"] == "increasing"

@pytest.mark.asyncio
async def test_pipeline_orchestration_complex(pipeline_orchestrator):
    """Test complex pipeline orchestration scenarios."""
    # Define pipeline with dependencies and conditions
    async def step1(context):
        return {
            "status": InstructionStatus.COMPLETED,
            "data": {"value": 42}
        }
    
    async def step2(context):
        return {
            "status": InstructionStatus.COMPLETED,
            "data": {"result": context["step1_output"]["value"] * 2}
        }
    
    pipeline_config = {
        "steps": [
            {
                "name": "step1",
                "instruction": step1
            },
            {
                "name": "step2",
                "instruction": step2,
                "dependencies": ["step1"],
                "conditions": [
                    {
                        "type": "output",
                        "step": "step1",
                        "condition": {
                            "operator": "greater_than",
                            "value": 0
                        }
                    }
                ]
            }
        ]
    }
    
    context = {
        "pipeline_name": "complex_pipeline",
        "pipeline_config": pipeline_config
    }
    
    result = await pipeline_orchestrator.execute.remote(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["result"]["status"] == "completed"
    assert result.data["result"]["output"]["step2"]["result"] == 84

@pytest.mark.asyncio
async def test_concurrent_pipeline_execution(pipeline_orchestrator):
    """Test concurrent execution of multiple pipelines."""
    # Define simple pipeline
    async def test_step(context):
        await asyncio.sleep(0.1)  # Simulate work
        return {
            "status": InstructionStatus.COMPLETED,
            "data": {"result": "success"}
        }
    
    pipeline_config = {
        "steps": [
            {
                "name": "step1",
                "instruction": test_step
            }
        ]
    }
    
    # Register pipeline
    pipeline_orchestrator._register_pipeline("concurrent_test", pipeline_config)
    
    # Execute multiple pipelines concurrently
    tasks = []
    for i in range(3):
        context = {
            "pipeline_name": "concurrent_test",
            "execution_id": f"exec_{i}"
        }
        tasks.append(pipeline_orchestrator.execute.remote(context))
    
    # Wait for all pipelines to complete
    results = await asyncio.gather(*tasks)
    
    assert all(r.status == InstructionStatus.COMPLETED for r in results)
    assert len(set(r.data["execution_id"] for r in results)) == 3

if __name__ == "__main__":
    pytest.main([__file__]) 