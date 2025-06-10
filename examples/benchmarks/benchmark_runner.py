"""
Benchmark Runner for AgentISA Framework.

This module provides tools to:
1. Run performance benchmarks
2. Compare optimization strategies
3. Measure learning effectiveness
4. Generate performance reports
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json
import logging
import statistics
from datetime import datetime
import os
import psutil
import cProfile
import pstats
import io

from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Results of a benchmark run."""
    name: str
    execution_time: float
    memory_usage: float
    instruction_count: int
    optimization_stats: Dict[str, Any]
    learning_stats: Dict[str, Any]
    error_count: int
    timestamp: datetime

class BenchmarkInstruction(Instruction):
    """Instruction for benchmarking."""
    
    def __init__(self, workload: int = 1000):
        super().__init__(
            id="benchmark",
            name="benchmark",
            type=InstructionType.CONTROL_FLOW,
            params={
                "iterations": TypeConstraint(base_type=int),
                "data_size": TypeConstraint(base_type=int)
            },
            description="Benchmark instruction with configurable workload"
        )
        self.workload = workload
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        iterations = input_params["iterations"]
        data_size = input_params["data_size"]
        
        # Simulate computational work
        result = 0
        for _ in range(iterations):
            data = [i * 2 for i in range(data_size)]
            result += sum(data)
            
        return {
            "result": result,
            "iterations": iterations,
            "data_size": data_size
        }

class BenchmarkRunner:
    """Runs benchmarks for AgentISA framework."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
        self.profiler = cProfile.Profile()
        
    async def run_benchmark(self, name: str, config: AgentISAConfig, 
                           iterations: int = 100, data_size: int = 1000) -> BenchmarkResult:
        """Run a single benchmark."""
        # Initialize agent
        agent_isa = AgentISA(config)
        instruction = BenchmarkInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Track resources
        start_time = time.perf_counter()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        error_count = 0
        
        try:
            # Enable profiling
            self.profiler.enable()
            
            # Run benchmark iterations
            for _ in range(iterations):
                try:
                    await agent_isa.execute_instruction(
                        instruction_id,
                        {
                            "iterations": iterations,
                            "data_size": data_size
                        }
                    )
                except Exception as e:
                    logger.error(f"Error in iteration: {str(e)}")
                    error_count += 1
                    
            # Disable profiling
            self.profiler.disable()
            
            # Calculate metrics
            end_time = time.perf_counter()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Get stats
            stats = agent_isa.get_instruction_stats()
            
            result = BenchmarkResult(
                name=name,
                execution_time=end_time - start_time,
                memory_usage=end_memory - start_memory,
                instruction_count=iterations,
                optimization_stats=stats.get("optimization_stats", {}),
                learning_stats={"learned_instructions": stats.get("learned_instructions", 0)},
                error_count=error_count,
                timestamp=datetime.now()
            )
            
            self.results.append(result)
            return result
            
        finally:
            # Cleanup
            await agent_isa.cleanup()
            
    async def compare_configurations(self, 
                                   configs: List[AgentISAConfig],
                                   names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """Compare different configurations."""
        results = {}
        
        for config, name in zip(configs, names):
            logger.info(f"Running benchmark for configuration: {name}")
            result = await self.run_benchmark(name, config)
            results[name] = result
            
        return results
        
    def generate_report(self, output_file: str = "benchmark_report.json"):
        """Generate benchmark report."""
        report = {
            "summary": self._generate_summary(),
            "detailed_results": [self._result_to_dict(r) for r in self.results],
            "profiling_stats": self._get_profiling_stats()
        }
        
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2, default=str)
            
        return report
        
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics."""
        if not self.results:
            return {}
            
        execution_times = [r.execution_time for r in self.results]
        memory_usages = [r.memory_usage for r in self.results]
        
        return {
            "total_benchmarks": len(self.results),
            "execution_time": {
                "mean": statistics.mean(execution_times),
                "median": statistics.median(execution_times),
                "std_dev": statistics.stdev(execution_times) if len(execution_times) > 1 else 0
            },
            "memory_usage": {
                "mean": statistics.mean(memory_usages),
                "median": statistics.median(memory_usages),
                "std_dev": statistics.stdev(memory_usages) if len(memory_usages) > 1 else 0
            },
            "total_errors": sum(r.error_count for r in self.results)
        }
        
    def _result_to_dict(self, result: BenchmarkResult) -> Dict[str, Any]:
        """Convert benchmark result to dictionary."""
        return {
            "name": result.name,
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "instruction_count": result.instruction_count,
            "optimization_stats": result.optimization_stats,
            "learning_stats": result.learning_stats,
            "error_count": result.error_count,
            "timestamp": result.timestamp.isoformat()
        }
        
    def _get_profiling_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        output = io.StringIO()
        stats = pstats.Stats(self.profiler, stream=output)
        stats.sort_stats("cumulative")
        stats.print_stats(20)  # Top 20 functions
        
        return {
            "profiling_output": output.getvalue()
        }

async def main():
    """Run example benchmarks."""
    runner = BenchmarkRunner()
    
    # Define configurations to compare
    configs = [
        AgentISAConfig(
            enable_optimization=False,
            enable_learning=False
        ),
        AgentISAConfig(
            enable_optimization=True,
            enable_learning=False
        ),
        AgentISAConfig(
            enable_optimization=True,
            enable_learning=True
        )
    ]
    
    names = [
        "baseline",
        "optimization_only",
        "optimization_and_learning"
    ]
    
    # Run benchmarks
    print("Running benchmarks...")
    results = await runner.compare_configurations(configs, names)
    
    # Generate report
    report = runner.generate_report()
    
    # Print summary
    print("\nBenchmark Summary:")
    print(json.dumps(report["summary"], indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 