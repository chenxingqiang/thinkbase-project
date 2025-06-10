# DeepSeek Agent Example

This example demonstrates how to implement an agent using the DeepSeek model following the Agent ISA (Instruction Set Architecture) specification.

## Files

1. `deepseek_agent_config.json` - Configuration file that defines the agent's capabilities and behavior following the ISA schema
2. `deepseek_agent_example.py` - Python implementation showing how to use the configuration with DeepSeek model

## Configuration

The agent is configured with the following capabilities:

- **Input Modes**: Direct input, context injection, and stream input
- **Output Modes**: Return results, forward to other agents, and store in memory
- **Transformation Methods**: Filter, map, and reduce operations on outputs
- **Collaboration Mode**: Sequential processing with defined workflow stages
- **Performance Tracking**: Execution time, resource utilization, and task completion metrics

## Prerequisites

1. Install the required dependencies:
```bash
pip install agentflow-core
pip install deepseek-coder
```

2. Set up your DeepSeek API credentials in your environment:
```bash
export DEEPSEEK_API_KEY=your_api_key_here
```

## Usage

1. First, ensure you have the configuration file in place:
```bash
cp deepseek_agent_config.json /path/to/your/project/
```

2. Run the example:
```bash
python deepseek_agent_example.py
```

The example demonstrates:
- Code generation using DeepSeek model
- Context-aware processing
- Output transformation
- Metrics tracking

## Example Output

The example will generate a Python implementation of a binary search algorithm with:
- Type hints
- Documentation
- Modern Python style
- Optional test cases

Example output format:
```json
{
    "status": "success",
    "data": {
        "code": "... generated code ...",
        "explanation": "... explanation of the implementation ..."
    },
    "metrics": {
        "execution_time": "...",
        "resource_utilization": "...",
        "task_completion_rate": "..."
    }
}
```

## Customization

You can customize the agent by:
1. Modifying the configuration file (`deepseek_agent_config.json`)
2. Adjusting the model parameters in `DeepSeekAgentConfig` class
3. Implementing custom transformation methods
4. Adding new output processing capabilities

## Error Handling

The agent includes error handling for:
- Invalid inputs
- Model API failures
- Resource constraints
- Timeout conditions

## Integration

This example can be integrated into larger systems by:
1. Importing the `DeepSeekAgent` class
2. Customizing the configuration
3. Using the async interface for pipeline processing

## Notes

- The DeepSeek model used is "deepseek-coder-33b-instruct"
- Default temperature is set to 0.7 for balanced creativity/precision
- Context window is set to 8192 tokens
- Supports both synchronous and asynchronous operation 