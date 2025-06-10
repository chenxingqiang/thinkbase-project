"""
Instruction Learning Benchmark for AgentISA Framework.

This benchmark measures:
1. Pattern recognition accuracy
2. Learning speed
3. Instruction reuse effectiveness
4. Memory usage during learning
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import random
import string
import psutil
import time

from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """Metrics for instruction learning."""
    pattern_recognition_accuracy: float
    learning_time: float
    memory_usage: float
    instructions_learned: int
    instruction_reuse_count: int
    execution_time_improvement: float

def generate_random_text(length: int) -> str:
    """Generate random text for testing."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

class TextProcessingInstruction(Instruction):
    """Instruction for text processing tasks."""
    
    def __init__(self):
        super().__init__(
            id="text_processor",
            name="text_processor",
            type=InstructionType.LLM_INTERACTION,
            params={
                "text": TypeConstraint(base_type=str),
                "operation": TypeConstraint(base_type=str)
            },
            description="Process text with various operations"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        text = input_params["text"]
        operation = input_params["operation"]
        
        if operation == "uppercase":
            result = text.upper()
        elif operation == "lowercase":
            result = text.lower()
        elif operation == "reverse":
            result = text[::-1]
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return {
            "result": result,
            "operation": operation
        }

class LearningBenchmark:
    """Benchmark for instruction learning capabilities."""
    
    def __init__(self, sample_size: int = 1000):
        self.sample_size = sample_size
        self.agent_isa: Optional[AgentISA] = None
        self.instruction_id: Optional[str] = None
        self.execution_times: List[float] = []
        
    async def setup(self):
        """Setup benchmark environment."""
        config = AgentISAConfig(
            enable_learning=True,
            enable_optimization=True,
            min_pattern_support=2,
            min_success_rate=0.8
        )
        self.agent_isa = AgentISA(config)
        
        # Register instruction
        instruction = TextProcessingInstruction()
        self.instruction_id = await self.agent_isa.register_instruction(instruction)
        
    async def generate_training_data(self) -> List[Dict[str, Any]]:
        """Generate training data with patterns."""
        operations = ["uppercase", "lowercase", "reverse"]
        data = []
        
        # Generate data with patterns
        for _ in range(self.sample_size):
            text = generate_random_text(20)
            operation = random.choice(operations)
            
            # Introduce patterns
            if random.random() < 0.3:  # 30% chance of pattern
                # Create a sequence of operations
                data.append({
                    "text": text,
                    "operation": "uppercase"
                })
                data.append({
                    "text": text.upper(),
                    "operation": "reverse"
                })
            else:
                data.append({
                    "text": text,
                    "operation": operation
                })
                
        return data
        
    async def run_benchmark(self) -> LearningMetrics:
        """Run learning benchmark."""
        await self.setup()
        
        if not self.agent_isa or not self.instruction_id:
            raise RuntimeError("Benchmark not properly initialized")
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024
        start_time = time.perf_counter()
        
        # Generate and process training data
        training_data = await self.generate_training_data()
        
        pattern_count = 0
        reuse_count = 0
        
        # Process training data
        for params in training_data:
            execution_start = time.perf_counter()
            
            try:
                result = await self.agent_isa.execute_instruction(
                    self.instruction_id,
                    params
                )
                
                execution_time = time.perf_counter() - execution_start
                self.execution_times.append(execution_time)
                
                # Learn from execution
                await self.agent_isa.learn_from_history()
                
                # Check if pattern was recognized
                stats = self.agent_isa.get_instruction_stats()
                new_pattern_count = stats.get("learned_instructions", 0)
                
                if new_pattern_count > pattern_count:
                    pattern_count = new_pattern_count
                    
                # Check for instruction reuse
                if stats.get("optimization_stats", {}).get("instructions_reused", 0) > reuse_count:
                    reuse_count = stats["optimization_stats"]["instructions_reused"]
                    
            except Exception as e:
                logger.error(f"Error processing sample: {str(e)}")
                
        end_time = time.perf_counter()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        # Calculate metrics
        total_patterns = sum(1 for i in range(len(training_data)-1)
                           if training_data[i]["operation"] == "uppercase" and
                           training_data[i+1]["operation"] == "reverse")
                           
        pattern_accuracy = pattern_count / (total_patterns if total_patterns > 0 else 1)
        
        # Calculate execution time improvement
        if len(self.execution_times) > 1:
            initial_times = self.execution_times[:100]  # First 100 executions
            final_times = self.execution_times[-100:]   # Last 100 executions
            time_improvement = (
                sum(initial_times) / len(initial_times) -
                sum(final_times) / len(final_times)
            ) / (sum(initial_times) / len(initial_times))
        else:
            time_improvement = 0
            
        return LearningMetrics(
            pattern_recognition_accuracy=pattern_accuracy,
            learning_time=end_time - start_time,
            memory_usage=end_memory - start_memory,
            instructions_learned=pattern_count,
            instruction_reuse_count=reuse_count,
            execution_time_improvement=time_improvement
        )
        
    async def cleanup(self):
        """Clean up resources."""
        if self.agent_isa:
            await self.agent_isa.cleanup()

async def main():
    """Run learning benchmark."""
    # Run benchmark with different sample sizes
    sample_sizes = [100, 500, 1000]
    results = []
    
    for size in sample_sizes:
        print(f"\nRunning benchmark with sample size: {size}")
        benchmark = LearningBenchmark(sample_size=size)
        
        try:
            metrics = await benchmark.run_benchmark()
            results.append({
                "sample_size": size,
                "metrics": {
                    "pattern_recognition_accuracy": metrics.pattern_recognition_accuracy,
                    "learning_time": metrics.learning_time,
                    "memory_usage": metrics.memory_usage,
                    "instructions_learned": metrics.instructions_learned,
                    "instruction_reuse_count": metrics.instruction_reuse_count,
                    "execution_time_improvement": metrics.execution_time_improvement
                }
            })
        finally:
            await benchmark.cleanup()
            
    # Print results
    print("\nBenchmark Results:")
    print(json.dumps(results, indent=2))
    
    # Save results
    with open("learning_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    asyncio.run(main()) 