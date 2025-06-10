"""Unit tests for objective decomposition instruction."""

import pytest
from typing import Dict, Any
import time
import asyncio

from agentflow.core.instructions.objective_decomposition import (
    ObjectiveDecompositionInstruction,
    DecompositionConfig,
    ExecutionContext,
    CircularDependencyError,
    DecompositionCache
)

@pytest.fixture
def decomposition_config():
    """Create test configuration."""
    return DecompositionConfig(
        max_depth=3,
        max_children=5,
        min_complexity=0.3,
        max_complexity=0.8,
        similarity_threshold=0.7
    )

@pytest.fixture
def decomposition_instruction(decomposition_config):
    """Create test instruction."""
    return ObjectiveDecompositionInstruction(
        name="test_decomposition",
        config=decomposition_config
    )

@pytest.fixture
def sample_objective():
    """Create sample objective for testing."""
    return {
        "id": "test_obj_1",
        "type": "sequence",
        "description": "Test objective",
        "steps": [
            {
                "id": "step_1",
                "description": "First step",
                "type": "task"
            },
            {
                "id": "step_2",
                "description": "Second step",
                "type": "task"
            },
            {
                "id": "step_3",
                "description": "Third step",
                "type": "task"
            }
        ]
    }

@pytest.mark.asyncio
async def test_objective_decomposition(decomposition_instruction, sample_objective):
    """Test basic objective decomposition."""
    # Create execution context
    context = ExecutionContext(
        params={"objective": sample_objective},
        state={},
        metadata={}
    )
    
    # Execute decomposition
    result = await decomposition_instruction.execute(context)
    
    # Verify success
    assert result.success
    
    # Verify decomposition structure
    decomposition = result.output["decomposition"]
    assert decomposition["objective"] == sample_objective
    assert len(decomposition["sub_objectives"]) == 3  # One for each step
    assert decomposition["depth"] > 0
    
    # Verify metrics
    metrics = result.output["metrics"]
    assert metrics["total_objectives"] == 4  # Parent + 3 sub-objectives
    assert metrics["depth"] > 0
    assert metrics["branching_factor"] > 0

@pytest.mark.asyncio
async def test_deep_objective_decomposition(decomposition_instruction):
    """Test decomposition with nested objectives."""
    deep_objective = {
        "id": "deep_obj_1",
        "type": "sequence",
        "steps": [
            {
                "id": "step_1",
                "type": "sequence",
                "steps": [
                    {"id": "step_1_1", "type": "task"},
                    {"id": "step_1_2", "type": "task"}
                ]
            },
            {
                "id": "step_2",
                "type": "sequence",
                "steps": [
                    {"id": "step_2_1", "type": "task"},
                    {"id": "step_2_2", "type": "task"}
                ]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": deep_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify depth
    assert decomposition["depth"] <= decomposition_instruction.config.max_depth
    
    # Verify sub-objectives
    assert len(decomposition["sub_objectives"]) == 2
    
    # Verify dependencies
    dependencies = decomposition["dependencies"]
    assert len(dependencies) > 0
    assert all(d["type"] == "sequential" for d in dependencies)

@pytest.mark.asyncio
async def test_parallel_objective_decomposition(decomposition_instruction):
    """Test decomposition of parallel objectives."""
    parallel_objective = {
        "id": "parallel_obj_1",
        "type": "parallel",
        "tasks": [
            {"id": "task_1", "type": "task"},
            {"id": "task_2", "type": "task"},
            {"id": "task_3", "type": "task"}
        ]
    }
    
    context = ExecutionContext(
        params={"objective": parallel_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify parallel sub-objectives
    assert len(decomposition["sub_objectives"]) == 3
    
    # Verify no sequential dependencies
    dependencies = decomposition["dependencies"]
    assert len(dependencies) == 0

@pytest.mark.asyncio
async def test_conditional_objective_decomposition(decomposition_instruction):
    """Test decomposition of conditional objectives."""
    conditional_objective = {
        "id": "cond_obj_1",
        "type": "conditional",
        "branches": [
            {
                "id": "branch_1",
                "condition": "condition_1",
                "type": "task"
            },
            {
                "id": "branch_2",
                "condition": "condition_2",
                "type": "task"
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": conditional_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify branches
    assert len(decomposition["sub_objectives"]) == 2
    
    # Verify branch properties
    for sub_obj in decomposition["sub_objectives"]:
        assert "parent_id" in sub_obj
        assert sub_obj["parent_id"] == conditional_objective["id"]

@pytest.mark.asyncio
async def test_invalid_objective_decomposition(decomposition_instruction):
    """Test decomposition with invalid objectives."""
    # Test with missing objective
    context = ExecutionContext(
        params={},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert not result.success
    assert "error" in result.output
    
    # Test with invalid objective type
    invalid_objective = {
        "id": "invalid_obj_1",
        "type": "invalid_type"
    }
    
    context = ExecutionContext(
        params={"objective": invalid_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert result.success  # Should still succeed but with no decomposition
    decomposition = result.output["decomposition"]
    assert len(decomposition["sub_objectives"]) == 0

@pytest.mark.asyncio
async def test_max_children_limit(decomposition_instruction):
    """Test enforcement of max children limit."""
    # Create objective with more steps than max_children
    many_steps_objective = {
        "id": "many_steps_obj",
        "type": "sequence",
        "steps": [
            {"id": f"step_{i}", "type": "task"}
            for i in range(decomposition_instruction.config.max_children + 5)
        ]
    }
    
    context = ExecutionContext(
        params={"objective": many_steps_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify max children limit
    assert len(decomposition["sub_objectives"]) == decomposition_instruction.config.max_children

@pytest.mark.asyncio
async def test_dependency_extraction(decomposition_instruction):
    """Test extraction of dependencies between sub-objectives."""
    # Create objective with explicit dependencies
    dependent_objective = {
        "id": "dep_obj_1",
        "type": "sequence",
        "steps": [
            {"id": "step_1", "type": "task"},
            {
                "id": "step_2",
                "type": "task",
                "depends_on": ["step_1"]
            },
            {
                "id": "step_3",
                "type": "task",
                "depends_on": ["step_2"]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": dependent_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify dependencies
    dependencies = decomposition["dependencies"]
    assert len(dependencies) == 2  # Two sequential dependencies
    
    # Verify dependency chain
    dependency_pairs = [(d["from"], d["to"]) for d in dependencies]
    assert any(p[0].endswith("step_1") and p[1].endswith("step_2") for p in dependency_pairs)
    assert any(p[0].endswith("step_2") and p[1].endswith("step_3") for p in dependency_pairs)

@pytest.mark.asyncio
async def test_complexity_calculation(decomposition_instruction):
    """Test complexity calculation with different factors."""
    # Test component complexity
    component_objective = {
        "id": "comp_obj_1",
        "type": "sequence",
        "steps": [
            {"id": "task_1", "type": "task"},
            {"id": "seq_1", "type": "sequence", "steps": []},
            {"id": "par_1", "type": "parallel", "tasks": []},
            {"id": "cond_1", "type": "conditional", "branches": []}
        ]
    }
    
    context = ExecutionContext(
        params={"objective": component_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert result.success
    
    # Test dependency complexity
    dependency_objective = {
        "id": "dep_obj_1",
        "type": "parallel",
        "tasks": [
            {"id": "task_1", "type": "task"},
            {
                "id": "task_2",
                "type": "task",
                "depends_on": ["task_1"]
            },
            {
                "id": "task_3",
                "type": "task",
                "depends_on": ["task_1", "task_2"]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": dependency_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert result.success
    
    # Test nesting complexity
    nested_objective = {
        "id": "nested_obj_1",
        "type": "sequence",
        "steps": [
            {
                "id": "step_1",
                "type": "sequence",
                "steps": [
                    {
                        "id": "step_1_1",
                        "type": "conditional",
                        "branches": [
                            {"id": "branch_1", "type": "task"}
                        ]
                    }
                ]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": nested_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert result.success
    
    # Test resource complexity
    resource_objective = {
        "id": "resource_obj_1",
        "type": "task",
        "resources": {
            "cpu": 0.8,
            "memory": 0.6,
            "time": 0.7
        }
    }
    
    context = ExecutionContext(
        params={"objective": resource_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    assert result.success

@pytest.mark.asyncio
async def test_circular_dependency_detection(decomposition_instruction):
    """Test detection of circular dependencies."""
    # Create objective with circular dependencies
    circular_objective = {
        "id": "circular_obj_1",
        "type": "sequence",
        "steps": [
            {"id": "step_1", "type": "task"},
            {
                "id": "step_2",
                "type": "task",
                "depends_on": ["step_3"]  # Circular: 1 -> 2 -> 3 -> 1
            },
            {
                "id": "step_3",
                "type": "task",
                "depends_on": ["step_1"]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": circular_objective},
        state={},
        metadata={}
    )
    
    # Should raise CircularDependencyError
    with pytest.raises(CircularDependencyError):
        await decomposition_instruction.execute(context)
    
    # Test self-referential dependency
    self_dependent_objective = {
        "id": "self_dep_obj_1",
        "type": "sequence",
        "steps": [
            {
                "id": "step_1",
                "type": "task",
                "depends_on": ["step_1"]  # Self-dependency
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": self_dependent_objective},
        state={},
        metadata={}
    )
    
    # Should raise CircularDependencyError
    with pytest.raises(CircularDependencyError):
        await decomposition_instruction.execute(context)
    
    # Test indirect circular dependency
    indirect_circular_objective = {
        "id": "indirect_circular_obj_1",
        "type": "sequence",
        "steps": [
            {"id": "step_1", "type": "task"},
            {
                "id": "step_2",
                "type": "task",
                "depends_on": ["step_1"]
            },
            {
                "id": "step_3",
                "type": "task",
                "depends_on": ["step_2"]
            },
            {
                "id": "step_4",
                "type": "task",
                "depends_on": ["step_3"]
            },
            {
                "id": "step_5",
                "type": "task",
                "depends_on": ["step_4", "step_1"]  # Creates circular dependency
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": indirect_circular_objective},
        state={},
        metadata={}
    )
    
    # Should raise CircularDependencyError
    with pytest.raises(CircularDependencyError):
        await decomposition_instruction.execute(context)

@pytest.mark.asyncio
async def test_complex_objective_decomposition(decomposition_instruction):
    """Test decomposition of complex objective with multiple factors."""
    complex_objective = {
        "id": "complex_obj_1",
        "type": "sequence",
        "steps": [
            {
                "id": "step_1",
                "type": "parallel",
                "tasks": [
                    {"id": "task_1_1", "type": "task"},
                    {"id": "task_1_2", "type": "task"}
                ],
                "resources": {
                    "cpu": 0.5,
                    "memory": 0.3
                }
            },
            {
                "id": "step_2",
                "type": "conditional",
                "branches": [
                    {
                        "id": "branch_2_1",
                        "type": "sequence",
                        "steps": [
                            {"id": "task_2_1_1", "type": "task"},
                            {"id": "task_2_1_2", "type": "task"}
                        ]
                    }
                ],
                "depends_on": ["step_1"]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": complex_objective},
        state={},
        metadata={}
    )
    
    result = await decomposition_instruction.execute(context)
    
    assert result.success
    decomposition = result.output["decomposition"]
    
    # Verify structure
    assert len(decomposition["sub_objectives"]) > 0
    assert decomposition["depth"] > 1
    
    # Verify metrics reflect complexity
    metrics = result.output["metrics"]
    assert metrics["total_objectives"] > 2
    assert metrics["branching_factor"] > 0

@pytest.mark.asyncio
async def test_caching(decomposition_instruction):
    """Test caching of decomposition results."""
    # Create test objective
    objective = {
        "id": "cache_test_1",
        "type": "sequence",
        "steps": [
            {"id": "step_1", "type": "task"},
            {"id": "step_2", "type": "task"}
        ]
    }
    
    context = ExecutionContext(
        params={"objective": objective},
        state={},
        metadata={}
    )
    
    # First execution
    start_time = time.time()
    result1 = await decomposition_instruction.execute(context)
    first_duration = time.time() - start_time
    
    assert result1.success
    
    # Second execution (should use cache)
    start_time = time.time()
    result2 = await decomposition_instruction.execute(context)
    second_duration = time.time() - start_time
    
    assert result2.success
    assert result2.output == result1.output
    assert second_duration < first_duration  # Cache should be faster
    
    # Modify objective
    objective["steps"].append({"id": "step_3", "type": "task"})
    
    # Third execution (should not use cache)
    start_time = time.time()
    result3 = await decomposition_instruction.execute(context)
    third_duration = time.time() - start_time
    
    assert result3.success
    assert result3.output != result1.output
    assert third_duration > second_duration  # No cache hit

@pytest.mark.asyncio
async def test_parallel_processing(decomposition_instruction):
    """Test parallel processing of independent objectives."""
    # Create objective with independent parallel tasks
    parallel_objective = {
        "id": "parallel_test_1",
        "type": "parallel",
        "tasks": [
            {
                "id": "task_1",
                "type": "sequence",
                "steps": [
                    {"id": "step_1_1", "type": "task"},
                    {"id": "step_1_2", "type": "task"}
                ]
            },
            {
                "id": "task_2",
                "type": "sequence",
                "steps": [
                    {"id": "step_2_1", "type": "task"},
                    {"id": "step_2_2", "type": "task"}
                ]
            },
            {
                "id": "task_3",
                "type": "sequence",
                "steps": [
                    {"id": "step_3_1", "type": "task"},
                    {"id": "step_3_2", "type": "task"}
                ]
            }
        ]
    }
    
    context = ExecutionContext(
        params={"objective": parallel_objective},
        state={},
        metadata={}
    )
    
    # Execute with parallel processing
    start_time = time.time()
    result = await decomposition_instruction.execute(context)
    parallel_duration = time.time() - start_time
    
    assert result.success
    
    # Execute with reduced parallelism
    decomposition_instruction.config.parallel_batch_size = 1
    
    start_time = time.time()
    sequential_result = await decomposition_instruction.execute(context)
    sequential_duration = time.time() - start_time
    
    assert sequential_result.success
    assert parallel_duration < sequential_duration  # Parallel should be faster
    
    # Verify results are equivalent
    assert result.output["decomposition"]["sub_objectives"] == \
           sequential_result.output["decomposition"]["sub_objectives"]

@pytest.mark.asyncio
async def test_cache_expiration(decomposition_instruction):
    """Test cache expiration and cleanup."""
    # Override cache TTL for testing
    decomposition_instruction.cache.ttl = 1  # 1 second
    
    objective = {
        "id": "cache_expire_1",
        "type": "task"
    }
    
    context = ExecutionContext(
        params={"objective": objective},
        state={},
        metadata={}
    )
    
    # First execution
    result1 = await decomposition_instruction.execute(context)
    assert result1.success
    
    # Wait for cache to expire
    await asyncio.sleep(1.1)
    
    # Second execution (should not use expired cache)
    result2 = await decomposition_instruction.execute(context)
    assert result2.success
    
    # Verify cache was cleared
    cache_key = decomposition_instruction._get_cache_key(objective)
    assert decomposition_instruction.cache.get(cache_key) is None

@pytest.mark.asyncio
async def test_cache_size_limit(decomposition_instruction):
    """Test cache size limit enforcement."""
    # Set small cache size
    decomposition_instruction.cache = DecompositionCache(max_size=2, ttl=3600)
    
    # Create test objectives
    objectives = [
        {
            "id": f"size_test_{i}",
            "type": "task"
        }
        for i in range(4)
    ]
    
    # Execute objectives
    for obj in objectives:
        context = ExecutionContext(
            params={"objective": obj},
            state={},
            metadata={}
        )
        result = await decomposition_instruction.execute(context)
        assert result.success
    
    # Verify cache size
    assert len(decomposition_instruction.cache.cache) == 2
    
    # Verify only most recent entries are cached
    for obj in objectives[:2]:
        cache_key = decomposition_instruction._get_cache_key(obj)
        assert decomposition_instruction.cache.get(cache_key) is None
        
    for obj in objectives[2:]:
        cache_key = decomposition_instruction._get_cache_key(obj)
        assert decomposition_instruction.cache.get(cache_key) is not None

@pytest.mark.asyncio
async def test_parallel_error_handling(decomposition_instruction):
    """Test error handling in parallel processing."""
    # Create objective with some invalid tasks
    mixed_objective = {
        "id": "mixed_test_1",
        "type": "parallel",
        "tasks": [
            {"id": "valid_1", "type": "task"},
            {"id": "invalid_1", "type": "invalid"},  # Should cause error
            {"id": "valid_2", "type": "task"},
            {"id": "invalid_2", "type": None},  # Should cause error
            {"id": "valid_3", "type": "task"}
        ]
    }
    
    context = ExecutionContext(
        params={"objective": mixed_objective},
        state={},
        metadata={}
    )
    
    # Execute with error handling
    result = await decomposition_instruction.execute(context)
    
    assert result.success  # Overall execution should succeed
    decomposition = result.output["decomposition"]
    
    # Verify valid tasks were processed
    valid_ids = {"valid_1", "valid_2", "valid_3"}
    processed_ids = {
        sub["id"]
        for sub in decomposition["sub_objectives"]
    }
    
    assert valid_ids.issubset(processed_ids) 