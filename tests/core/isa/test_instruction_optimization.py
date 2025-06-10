"""Unit tests for instruction optimization system."""

import pytest
from typing import Dict, Any, List, Optional
from agentflow.core.isa.instruction_optimization import (
    InstructionDAG,
    PipelineScheduler,
    StaticOptimizer,
    DynamicOptimizer,
    OptimizationManager,
    OptimizationMetrics
)
from agentflow.core.isa.instruction import Instruction, InstructionType

class TestInstruction(Instruction):
    """Test instruction for optimization testing."""
    
    def __init__(self, id: str, name: str, dependencies: Optional[List[str]] = None):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Test instruction {name}"
        )
        self.executed = False
        self.dependencies = dependencies if dependencies is not None else []

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executed = True
        return {"status": "executed", "id": self.id}

@pytest.fixture
def test_instructions():
    """Create test instructions."""
    return [
        TestInstruction("instr1", "instruction1"),
        TestInstruction("instr2", "instruction2", ["instr1"]),
        TestInstruction("instr3", "instruction3", ["instr2"]),
        TestInstruction("instr4", "instruction4")
    ]

@pytest.fixture
def instruction_dag():
    return InstructionDAG()

@pytest.fixture
def pipeline_scheduler():
    return PipelineScheduler()

@pytest.fixture
def static_optimizer():
    return StaticOptimizer()

@pytest.fixture
def dynamic_optimizer():
    return DynamicOptimizer()

@pytest.fixture
def optimization_manager():
    return OptimizationManager()

def test_instruction_dag_creation(instruction_dag, test_instructions):
    """Test DAG creation and dependency management."""
    # Add instructions to DAG
    for instruction in test_instructions:
        instruction_dag.add_instruction(
            instruction,
            instruction.dependencies
        )
    
    # Verify graph structure
    assert len(instruction_dag.graph.nodes) == len(test_instructions)
    assert instruction_dag.graph.has_edge("instr1", "instr2")
    assert instruction_dag.graph.has_edge("instr2", "instr3")

def test_execution_order(instruction_dag, test_instructions):
    """Test topological sorting of instructions."""
    # Add instructions to DAG
    for instruction in test_instructions:
        instruction_dag.add_instruction(
            instruction,
            instruction.dependencies
        )
    
    # Get execution order
    order = instruction_dag.get_execution_order()
    
    # Verify order respects dependencies
    assert order.index("instr1") < order.index("instr2")
    assert order.index("instr2") < order.index("instr3")

def test_pipeline_scheduling(pipeline_scheduler, test_instructions):
    """Test instruction pipeline scheduling."""
    # Set resource limits
    pipeline_scheduler.set_resource_limit("memory", 100)
    pipeline_scheduler.set_resource_limit("compute", 50)
    
    # Schedule instructions
    stages = pipeline_scheduler.schedule(test_instructions)
    
    assert isinstance(stages, list)
    assert all(isinstance(stage, list) for stage in stages)
    assert sum(len(stage) for stage in stages) == len(test_instructions)

def test_static_optimization(static_optimizer, test_instructions):
    """Test static optimization patterns."""
    # Add peephole pattern
    static_optimizer.add_peephole_pattern(
        ["instruction1", "instruction2"],
        ["combined_instruction"]
    )
    
    # Apply optimization
    optimized = static_optimizer.optimize(test_instructions)
    
    assert isinstance(optimized, list)
    assert len(optimized) <= len(test_instructions)
    assert static_optimizer.optimization_stats["peephole_applied"] >= 0

def test_dynamic_optimization(dynamic_optimizer, test_instructions):
    """Test dynamic optimization."""
    # Record executions
    path_id = "test_path"
    metrics = OptimizationMetrics(
        execution_time=1.0,
        memory_usage=50.0,
        api_calls=1,
        success_rate=1.0
    )
    
    # Record multiple executions to make path hot
    for _ in range(dynamic_optimizer.hot_path_threshold + 1):
        dynamic_optimizer.record_execution(path_id, test_instructions, metrics)
    
    # Verify hot path detection
    assert path_id in dynamic_optimizer.hot_paths
    assert dynamic_optimizer.execution_counts[path_id] > dynamic_optimizer.hot_path_threshold

@pytest.mark.asyncio
async def test_optimization_manager(optimization_manager, test_instructions):
    """Test complete optimization pipeline."""
    # Optimize instructions
    optimized = optimization_manager.optimize_instructions(test_instructions)
    
    assert isinstance(optimized, list)
    assert len(optimized) > 0
    
    # Record execution
    metrics = OptimizationMetrics(
        execution_time=1.0,
        memory_usage=50.0,
        api_calls=1,
        success_rate=1.0
    )
    
    result = optimization_manager.record_execution(optimized, metrics)
    assert isinstance(result, list)

def test_resource_estimation(pipeline_scheduler, test_instructions):
    """Test resource estimation for instructions."""
    resources = pipeline_scheduler._estimate_resources(test_instructions[0])
    assert isinstance(resources, dict)
    assert all(isinstance(v, float) for v in resources.values())

def test_dependency_analysis(pipeline_scheduler, test_instructions):
    """Test instruction dependency analysis."""
    deps = pipeline_scheduler._analyze_dependencies(test_instructions[0])
    assert isinstance(deps, list)

def test_instruction_combination(static_optimizer):
    """Test instruction combination logic."""
    instr1 = TestInstruction("test1", "test1")
    instr2 = TestInstruction("test2", "test2")
    
    # Test combination compatibility
    assert not static_optimizer._can_combine(instr1, instr2)
    
    # Test instruction merging
    merged = static_optimizer._merge_instructions(instr1, instr2)
    assert isinstance(merged, Instruction)

def test_optimization_metrics():
    """Test optimization metrics handling."""
    metrics = OptimizationMetrics(
        execution_time=1.0,
        memory_usage=50.0,
        api_calls=1,
        success_rate=1.0
    )
    
    assert metrics.execution_time == 1.0
    assert metrics.memory_usage == 50.0
    assert metrics.api_calls == 1
    assert metrics.success_rate == 1.0

def test_circular_dependency_detection(instruction_dag):
    """Test detection of circular dependencies."""
    # Create circular dependency
    instr1 = TestInstruction("circular1", "circular1", ["circular2"])
    instr2 = TestInstruction("circular2", "circular2", ["circular1"])
    
    instruction_dag.add_instruction(instr1, instr1.dependencies)
    instruction_dag.add_instruction(instr2, instr2.dependencies)
    
    # Verify circular dependency detection
    with pytest.raises(ValueError):
        instruction_dag.get_execution_order()

def test_pipeline_stage_latency(pipeline_scheduler):
    """Test pipeline stage latency handling."""
    # Add stage latencies
    pipeline_scheduler.add_stage_latency("fetch", 1.0)
    pipeline_scheduler.add_stage_latency("execute", 2.0)
    
    assert pipeline_scheduler.stage_latencies["fetch"] == 1.0
    assert pipeline_scheduler.stage_latencies["execute"] == 2.0 