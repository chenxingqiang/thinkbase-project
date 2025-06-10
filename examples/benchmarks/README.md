# AgentISA Framework Benchmarks

This directory contains benchmarking tools for evaluating the performance and effectiveness of the AgentISA framework.

## Available Benchmarks

### 1. General Performance Benchmark (`benchmark_runner.py`)

Measures overall framework performance including:
- Instruction execution speed
- Memory usage
- Resource utilization
- Optimization effectiveness
- Learning capabilities

Usage:
```bash
python benchmark_runner.py
```

### 2. Instruction Learning Benchmark (`learning_benchmark.py`)

Specifically measures the instruction learning system:
- Pattern recognition accuracy
- Learning speed
- Instruction reuse effectiveness
- Memory usage during learning

Usage:
```bash
python learning_benchmark.py
```

### 3. Optimization Benchmark (`optimization_benchmark.py`)

Focuses on instruction optimization capabilities:
- Static optimization effectiveness
- Dynamic optimization performance
- Pipeline scheduling efficiency
- Resource utilization

Usage:
```bash
python optimization_benchmark.py
```

## Running Benchmarks

### Prerequisites

1. Install required packages:
```bash
pip install psutil numpy
```

2. Configure environment variables (optional):
```bash
export AGENTISA_BENCHMARK_ITERATIONS=1000  # Default: 100
export AGENTISA_BENCHMARK_COMPLEXITY=2000  # Default: 1000
```

### Running All Benchmarks

Use the provided script to run all benchmarks:
```bash
./run_all_benchmarks.sh
```

Or run individual benchmarks:
```bash
python benchmark_runner.py
python learning_benchmark.py
python optimization_benchmark.py
```

### Benchmark Results

Results are saved in JSON format:
- `benchmark_results.json`: General performance results
- `learning_benchmark_results.json`: Learning system results
- `optimization_benchmark_results.json`: Optimization results

Example result format:
```json
{
    "summary": {
        "total_benchmarks": 3,
        "execution_time": {
            "mean": 1.234,
            "median": 1.123,
            "std_dev": 0.123
        },
        "memory_usage": {
            "mean": 256.7,
            "median": 255.5,
            "std_dev": 12.3
        }
    },
    "detailed_results": [
        {
            "name": "benchmark_1",
            "metrics": {
                "execution_time": 1.234,
                "memory_usage": 256.7,
                "instruction_count": 1000
            }
        }
    ]
}
```

## Customizing Benchmarks

### 1. Adjusting Parameters

Modify benchmark parameters in the respective files:
- `benchmark_runner.py`: `iterations`, `data_size`
- `learning_benchmark.py`: `sample_size`
- `optimization_benchmark.py`: `num_instructions`

### 2. Adding Custom Metrics

1. Define new metrics in the respective metric classes:
```python
@dataclass
class BenchmarkMetrics:
    your_metric: float
```

2. Implement metric collection in the benchmark class:
```python
def collect_custom_metrics(self) -> float:
    # Your metric collection logic
    return value
```

### 3. Creating Custom Workloads

1. Create a new instruction class:
```python
class CustomInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="custom",
            name="custom",
            type=InstructionType.CONTROL_FLOW,
            params={
                "param1": TypeConstraint(base_type=str)
            }
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Your custom logic
        return result
```

2. Use the custom instruction in benchmarks:
```python
instruction = CustomInstruction()
instruction_id = await agent_isa.register_instruction(instruction)
```

## Best Practices

1. **Resource Management**
   - Always use `try`/`finally` blocks
   - Clean up resources after benchmarks
   - Monitor memory usage

2. **Reproducibility**
   - Set random seeds
   - Document environment setup
   - Save full configuration

3. **Result Analysis**
   - Use statistical measures
   - Consider variance
   - Compare across configurations

4. **Performance Considerations**
   - Run benchmarks multiple times
   - Warm up the system
   - Account for background processes

## Troubleshooting

### Common Issues

1. Memory Issues
```python
# Monitor memory usage
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"Memory usage: {memory_info.rss / 1024 / 1024} MB")
```

2. Performance Issues
```python
# Enable performance logging
import cProfile
profiler = cProfile.Profile()
profiler.enable()
# Your code here
profiler.disable()
profiler.print_stats()
```

3. Resource Cleanup
```python
# Ensure proper cleanup
try:
    # Your benchmark code
finally:
    await benchmark.cleanup()
```

### Getting Help

If you encounter issues:
1. Check the logs in `benchmark_logs/`
2. Review the metrics in result files
3. Consult the AgentISA documentation
4. Open an issue on GitHub 