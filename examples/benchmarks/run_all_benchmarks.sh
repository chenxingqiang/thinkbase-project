#!/bin/bash

# Set default values for environment variables if not set
export AGENTISA_BENCHMARK_ITERATIONS=${AGENTISA_BENCHMARK_ITERATIONS:-100}
export AGENTISA_BENCHMARK_COMPLEXITY=${AGENTISA_BENCHMARK_COMPLEXITY:-1000}

# Create logs directory if it doesn't exist
mkdir -p benchmark_logs

# Function to run a benchmark and check its exit status
run_benchmark() {
    local benchmark_name=$1
    local benchmark_script=$2
    echo "Running ${benchmark_name}..."
    python ${benchmark_script} > benchmark_logs/${benchmark_name}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ ${benchmark_name} completed successfully"
        return 0
    else
        echo "✗ ${benchmark_name} failed. Check benchmark_logs/${benchmark_name}.log for details"
        return 1
    fi
}

# Record start time
start_time=$(date +%s)

# Run all benchmarks
echo "Starting benchmarks..."
echo "Iterations: ${AGENTISA_BENCHMARK_ITERATIONS}"
echo "Complexity: ${AGENTISA_BENCHMARK_COMPLEXITY}"
echo

failed_benchmarks=0

# Run each benchmark
run_benchmark "general_performance" "benchmark_runner.py"
failed_benchmarks=$((failed_benchmarks + $?))

run_benchmark "instruction_learning" "learning_benchmark.py"
failed_benchmarks=$((failed_benchmarks + $?))

run_benchmark "optimization" "optimization_benchmark.py"
failed_benchmarks=$((failed_benchmarks + $?))

# Record end time
end_time=$(date +%s)
duration=$((end_time - start_time))

# Generate combined report
echo
echo "Generating combined report..."
python - << EOF
import json
from datetime import datetime
import os

def load_json_file(filename):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None

# Load individual results
results = {
    'general': load_json_file('benchmark_results.json'),
    'learning': load_json_file('learning_benchmark_results.json'),
    'optimization': load_json_file('optimization_benchmark_results.json')
}

# Create combined report
combined_report = {
    'timestamp': datetime.now().isoformat(),
    'configuration': {
        'iterations': int(os.getenv('AGENTISA_BENCHMARK_ITERATIONS', 100)),
        'complexity': int(os.getenv('AGENTISA_BENCHMARK_COMPLEXITY', 1000))
    },
    'results': results,
    'summary': {
        'total_duration_seconds': ${duration},
        'failed_benchmarks': ${failed_benchmarks}
    }
}

# Save combined report
with open('combined_benchmark_results.json', 'w') as f:
    json.dump(combined_report, f, indent=2)

# Print summary
print("\nBenchmark Summary:")
print(f"Total duration: {${duration}} seconds")
print(f"Failed benchmarks: {${failed_benchmarks}}")
print("\nDetailed results saved to: combined_benchmark_results.json")
EOF

# Exit with number of failed benchmarks
exit ${failed_benchmarks} 