"""
Instruction Optimization Benchmark for AgentISA Framework.

This benchmark measures:
1. Static optimization effectiveness
2. Dynamic optimization performance
3. Pipeline scheduling efficiency
4. Resource utilization
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import random
import psutil
import time
import numpy as np

from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OptimizationMetrics:
    """Metrics for instruction optimization."""
    static_optimization_reduction: float  # Reduction in instruction count
    dynamic_optimization_speedup: float   # Execution time improvement
    pipeline_efficiency: float            # Resource utilization
    memory_overhead: float               # Memory cost of optimization
    hot_path_detection_rate: float       # Rate of hot path detection
    rewrite_success_rate: float          # Success rate of instruction rewrites

class ComputeInstruction(Instruction):
    """Instruction for compute-intensive tasks."""
    
    def __init__(self, complexity: int = 1000):
        super().__init__(
            id="compute",
            name="compute",
            type=InstructionType.CONTROL_FLOW,
            params={
                "data": TypeConstraint(base_type=list),
                "operation": TypeConstraint(base_type=str)
            },
            description="Perform compute-intensive operations"
        )
        self.complexity = complexity
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        operation = input_params["operation"]
        
        if operation == "sort":
            result = sorted(data)
        elif operation == "filter":
            result = [x for x in data if x > 0]
        elif operation == "transform":
            result = [x * 2 for x in data]
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        # Simulate computational work
        for _ in range(self.complexity):
            result = [x + 1 for x in result]
            
        return {
            "result": result,
            "operation": operation
        }

class OptimizationBenchmark:
    """Benchmark for instruction optimization capabilities."""
    
    def __init__(self, num_instructions: int = 1000):
        self.num_instructions = num_instructions
        self.agent_isa: Optional[AgentISA] = None
        self.instruction_id: Optional[str] = None
        self.execution_times: List[float] = []
        
    async def setup(self):
        """Setup benchmark environment."""
        config = AgentISAConfig(
            enable_optimization=True,
            enable_learning=False,
            hot_path_threshold=50,
            stability_threshold=0.8
        )
        self.agent_isa = AgentISA(config)
        
        # Register instruction
        instruction = ComputeInstruction()
        self.instruction_id = await self.agent_isa.register_instruction(instruction)
        
    def generate_workload(self) -> List[Dict[str, Any]]:
        """Generate workload with optimization opportunities."""
        operations = ["sort", "filter", "transform"]
        workload = []
        
        # Generate workload with patterns
        for _ in range(self.num_instructions):
            # Create data with varying sizes
            data_size = random.randint(100, 1000)
            data = [random.randint(-100, 100) for _ in range(data_size)]
            
            # Introduce patterns for optimization
            if random.random() < 0.4:  # 40% chance of pattern
                # Create a sequence of operations
                workload.extend([
                    {"data": data, "operation": "sort"},
                    {"data": sorted(data), "operation": "filter"},
                    {"data": [x for x in sorted(data) if x > 0], "operation": "transform"}
                ])
            else:
                workload.append({
                    "data": data,
                    "operation": random.choice(operations)
                })
                
        return workload
        
    async def run_benchmark(self) -> OptimizationMetrics:
        """Run optimization benchmark."""
        await self.setup()
        
        if not self.agent_isa or not self.instruction_id:
            raise RuntimeError("Benchmark not properly initialized")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Generate workload
        workload = self.generate_workload()
        original_count = len(workload)
        
        # Measure baseline performance
        baseline_times = []
        for params in workload[:50]:  # First 50 executions without optimization
            start_time = time.perf_counter()
            await self.agent_isa.execute_instruction(self.instruction_id, params)
            baseline_times.append(time.perf_counter() - start_time)
        
        # Run optimized workload
        optimized_times = []
        rewrite_attempts = 0
        rewrite_successes = 0
        hot_paths_detected = 0
        
        for params in workload:
            start_time = time.perf_counter()
            
            try:
                result = await self.agent_isa.execute_instruction(
                    self.instruction_id,
                    params
                )
                
                execution_time = time.perf_counter() - start_time
                optimized_times.append(execution_time)
                
                # Check optimization stats
                stats = self.agent_isa.get_instruction_stats()
                optimization_stats = stats.get("optimization_stats", {})
                
                if optimization_stats.get("instruction_rewrites", 0) > rewrite_attempts:
                    rewrite_attempts = optimization_stats["instruction_rewrites"]
                    if optimization_stats.get("successful_rewrites", 0) > rewrite_successes:
                        rewrite_successes = optimization_stats["successful_rewrites"]
                        
                if optimization_stats.get("hot_paths_detected", 0) > hot_paths_detected:
                    hot_paths_detected = optimization_stats["hot_paths_detected"]
                    
            except Exception as e:
                logger.error(f"Error executing instruction: {str(e)}")
                
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        optimized_count = len(optimized_times)
        static_reduction = 1 - (optimized_count / original_count)
        
        baseline_avg = float(np.mean(baseline_times))
        optimized_avg = float(np.mean(optimized_times[-50:]))  # Last 50 executions
        dynamic_speedup = (baseline_avg - optimized_avg) / baseline_avg
        
        # Calculate pipeline efficiency
        cpu_percent = psutil.Process().cpu_percent()
        memory_percent = psutil.Process().memory_percent()
        pipeline_efficiency = (cpu_percent + memory_percent) / 200  # Normalize to 0-1
        
        # Calculate hot path detection rate
        potential_hot_paths = sum(1 for i in range(len(workload)-2)
                                if workload[i]["operation"] == "sort" and
                                workload[i+1]["operation"] == "filter" and
                                workload[i+2]["operation"] == "transform")
        hot_path_rate = hot_paths_detected / (potential_hot_paths if potential_hot_paths > 0 else 1)
        
        # Calculate rewrite success rate
        rewrite_rate = rewrite_successes / (rewrite_attempts if rewrite_attempts > 0 else 1)
        
        return OptimizationMetrics(
            static_optimization_reduction=static_reduction,
            dynamic_optimization_speedup=dynamic_speedup,
            pipeline_efficiency=pipeline_efficiency,
            memory_overhead=end_memory - start_memory,
            hot_path_detection_rate=hot_path_rate,
            rewrite_success_rate=rewrite_rate
        )
        
    async def cleanup(self):
        """Clean up resources."""
        if self.agent_isa:
            await self.agent_isa.cleanup()

async def main():
    """Run optimization benchmark."""
    # Run benchmark with different instruction counts
    instruction_counts = [100, 500, 1000]
    results = []
    
    for count in instruction_counts:
        print(f"\nRunning benchmark with {count} instructions...")
        benchmark = OptimizationBenchmark(num_instructions=count)
        
        try:
            metrics = await benchmark.run_benchmark()
            results.append({
                "instruction_count": count,
                "metrics": {
                    "static_optimization_reduction": metrics.static_optimization_reduction,
                    "dynamic_optimization_speedup": metrics.dynamic_optimization_speedup,
                    "pipeline_efficiency": metrics.pipeline_efficiency,
                    "memory_overhead": metrics.memory_overhead,
                    "hot_path_detection_rate": metrics.hot_path_detection_rate,
                    "rewrite_success_rate": metrics.rewrite_success_rate
                }
            })
        finally:
            await benchmark.cleanup()
            
    # Print results
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))
    
    # Save results
    with open("optimization_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 