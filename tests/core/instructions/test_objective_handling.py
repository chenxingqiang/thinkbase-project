"""Integration tests for objective handling instructions."""

import pytest
from typing import Dict, Any, List
from agentflow.core.instructions.objective_handling import (
    ObjectiveDecompositionInstruction,
    ObjectiveValidationInstruction,
    ObjectiveProgressTrackingInstruction,
    ObjectiveConfig
)
from agentflow.core.instructions.base import ExecutionContext

@pytest.fixture
def objective_config():
    """Create test objective configuration."""
    return ObjectiveConfig(
        max_depth=3,
        max_children=5,
        timeout_seconds=30.0,
        min_success_rate=0.8,
        validation_required=True
    )

@pytest.fixture
def decomposition_instruction(objective_config):
    """Create test objective decomposition instruction."""
    return ObjectiveDecompositionInstruction(
        name="test_decomposition",
        objective_config=objective_config
    )

@pytest.fixture
def validation_instruction(objective_config):
    """Create test objective validation instruction."""
    def validate_progress(data: Dict[str, Any]) -> Dict[str, Any]:
        progress = data.get("progress", 0.0)
        return {
            "success": progress >= 1.0,
            "details": {"progress": progress}
        }
    
    return ObjectiveValidationInstruction(
        name="test_validation",
        objective_config=objective_config,
        validation_rules={
            "progress_check": {
                "validation_function": validate_progress,
                "required_objective_fields": ["id", "type"],
                "required_completion_fields": ["progress"]
            }
        }
    )

@pytest.fixture
def progress_instruction(objective_config):
    """Create test objective progress tracking instruction."""
    return ObjectiveProgressTrackingInstruction(
        name="test_progress",
        objective_config=objective_config
    )

@pytest.mark.asyncio
async def test_objective_decomposition(decomposition_instruction):
    """Test objective decomposition."""
    # Create test objective
    objective = {
        "id": "obj1",
        "type": "complex_task",
        "description": "Complex task with multiple components",
        "requirements": {
            "skills": ["python", "ml"],
            "resources": ["gpu", "data"]
        },
        "depth": 0
    }
    
    # Execute decomposition
    context = ExecutionContext(
        params={"objective": objective},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await decomposition_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert isinstance(result.output, dict)
    assert "sub_objectives" in result.output
    assert "dependencies" in result.output
    
    # Verify sub-objectives
    sub_objectives = result.output["sub_objectives"]
    assert len(sub_objectives) > 0
    for sub in sub_objectives:
        assert "id" in sub
        assert "type" in sub
        assert "description" in sub
        assert "requirements" in sub
        assert "depth" in sub
        assert sub["depth"] == 1
    
    # Verify metrics
    assert decomposition_instruction.metrics.objectives_created > 0

@pytest.mark.asyncio
async def test_objective_validation(validation_instruction):
    """Test objective validation."""
    # Create test objective and completion data
    objective = {
        "id": "obj1",
        "type": "task",
        "description": "Test task"
    }
    
    completion_data = {
        "progress": 1.0,
        "status": "completed",
        "output": {"result": "success"}
    }
    
    # Execute validation
    context = ExecutionContext(
        params={
            "objective": objective,
            "completion_data": completion_data
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await validation_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert isinstance(result.output, dict)
    assert "validation_results" in result.output
    assert result.output["success"]
    
    # Verify validation results
    validation_results = result.output["validation_results"]
    assert len(validation_results) > 0
    assert all(r.get("success", False) for r in validation_results)
    
    # Verify metrics
    assert validation_instruction.metrics.objectives_completed == 1
    assert validation_instruction.metrics.objectives_failed == 0
    assert validation_instruction.metrics.success_rate == 1.0

@pytest.mark.asyncio
async def test_objective_progress_tracking(progress_instruction):
    """Test objective progress tracking."""
    # Create test objective and progress updates
    objective = {
        "id": "obj1",
        "type": "task",
        "description": "Test task"
    }
    
    progress_updates = [
        {"progress": 0.0, "status": "started"},
        {"progress": 0.5, "status": "in_progress"},
        {"progress": 1.0, "status": "completed"}
    ]
    
    results = []
    for update in progress_updates:
        context = ExecutionContext(
            params={
                "objective": objective,
                "progress_update": update
            },
            state={},
            metadata={"timestamp": float(len(results))}  # Simulate time passing
        )
        result = await progress_instruction.execute(context)
        results.append(result)
    
    # Verify results
    assert all(r.success for r in results)
    
    # Verify progress tracking
    final_result = results[-1]
    assert final_result.output["status"] == "completed"
    assert final_result.output["progress"] == 1.0
    assert len(final_result.output["updates"]) == len(progress_updates)
    
    # Verify metrics
    assert progress_instruction.metrics.objectives_completed == 1
    assert progress_instruction.metrics.average_completion_time > 0.0

@pytest.mark.asyncio
async def test_deep_objective_decomposition(decomposition_instruction):
    """Test deep objective decomposition with depth limit."""
    # Create deep objective
    deep_objective = {
        "id": "deep1",
        "type": "complex_task",
        "description": "Deep complex task",
        "requirements": {"complexity": "high"},
        "depth": 0
    }
    
    # Execute decomposition
    context = ExecutionContext(
        params={"objective": deep_objective},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await decomposition_instruction.execute(context)
    
    # Verify depth limit
    def check_depth(obj):
        assert obj["depth"] <= decomposition_instruction.objective_config.max_depth
        for sub in obj.get("sub_objectives", []):
            check_depth(sub)
    
    check_depth(result.output)

@pytest.mark.asyncio
async def test_objective_validation_failure(validation_instruction):
    """Test objective validation failure cases."""
    # Create test objective with incomplete completion data
    objective = {
        "id": "obj1",
        "type": "task",
        "description": "Test task"
    }
    
    completion_data = {
        "progress": 0.5,  # Not complete
        "status": "in_progress"
    }
    
    # Execute validation
    context = ExecutionContext(
        params={
            "objective": objective,
            "completion_data": completion_data
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await validation_instruction.execute(context)
    
    # Verify failure
    assert result.success  # Execution succeeded but validation failed
    assert not result.output["success"]  # Validation result is failure
    
    # Verify metrics
    assert validation_instruction.metrics.objectives_failed == 1
    assert validation_instruction.metrics.success_rate < 1.0

@pytest.mark.asyncio
async def test_objective_progress_history(progress_instruction):
    """Test objective progress history tracking."""
    # Create test objective
    objective = {
        "id": "obj1",
        "type": "task",
        "description": "Test task"
    }
    
    # Create progress updates with timestamps
    updates = [
        (0.0, "started"),
        (0.25, "planning"),
        (0.5, "implementing"),
        (0.75, "testing"),
        (1.0, "completed")
    ]
    
    # Execute updates
    for progress, status in updates:
        context = ExecutionContext(
            params={
                "objective": objective,
                "progress_update": {
                    "progress": progress,
                    "status": status
                }
            },
            state={},
            metadata={"timestamp": float(updates.index((progress, status)))}
        )
        await progress_instruction.execute(context)
    
    # Get final state
    history = progress_instruction.progress_history[objective["id"]]
    
    # Verify history
    assert len(history) == len(updates)
    assert all("timestamp" in entry for entry in history)
    assert all("update" in entry for entry in history)
    
    # Verify chronological order
    timestamps = [entry["timestamp"] for entry in history]
    assert timestamps == sorted(timestamps)
    
    # Verify completion time calculation
    completion_time = progress_instruction._calculate_completion_time(objective["id"])
    assert completion_time == timestamps[-1] - timestamps[0]

@pytest.mark.asyncio
async def test_invalid_objective_handling(decomposition_instruction,
                                        validation_instruction,
                                        progress_instruction):
    """Test handling of invalid objectives."""
    # Test cases with invalid objectives
    invalid_cases = [
        {},  # Empty objective
        {"id": "invalid1"},  # Missing required fields
        None  # No objective
    ]
    
    for invalid_obj in invalid_cases:
        # Test decomposition
        context = ExecutionContext(
            params={"objective": invalid_obj},
            state={},
            metadata={"timestamp": 0.0}
        )
        decomp_result = await decomposition_instruction.execute(context)
        assert not decomp_result.success
        assert len(decomp_result.errors) > 0
        
        # Test validation
        context = ExecutionContext(
            params={
                "objective": invalid_obj,
                "completion_data": {"progress": 1.0}
            },
            state={},
            metadata={"timestamp": 0.0}
        )
        valid_result = await validation_instruction.execute(context)
        assert not valid_result.success
        assert len(valid_result.errors) > 0
        
        # Test progress tracking
        context = ExecutionContext(
            params={
                "objective": invalid_obj,
                "progress_update": {"progress": 0.5}
            },
            state={},
            metadata={"timestamp": 0.0}
        )
        prog_result = await progress_instruction.execute(context)
        assert not prog_result.success
        assert len(prog_result.errors) > 0 