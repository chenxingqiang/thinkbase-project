# Tutorial 2: Instruction Composition and Optimization

This tutorial covers advanced topics in the AgentISA framework, focusing on instruction composition and optimization techniques.

## Prerequisites

- Completed Tutorial 1: Getting Started with AgentISA
- Understanding of Python async/await
- Basic knowledge of optimization concepts

## 1. Instruction Composition

### Creating Component Instructions

First, let's create some simple instructions that we'll compose:

```python
from agentflow.core.isa import Instruction, InstructionType, TypeConstraint
from typing import Dict, Any

class ValidateDataInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="validate_data",
            name="validate_data",
            type=InstructionType.CONTROL_FLOW,
            params={
                "data": TypeConstraint(base_type=dict)
            },
            description="Validate input data"
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        
        # Validate required fields
        if "name" not in data or "age" not in data:
            raise ValueError("Missing required fields")
            
        # Validate types
        if not isinstance(data["name"], str):
            raise TypeError("Name must be a string")
        if not isinstance(data["age"], int):
            raise TypeError("Age must be an integer")
            
        return {
            "valid": True,
            "data": data
        }

class TransformDataInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="transform_data",
            name="transform_data",
            type=InstructionType.STATE_MANAGEMENT,
            params={
                "data": TypeConstraint(base_type=dict)
            },
            description="Transform validated data"
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        
        # Transform data
        transformed = {
            "full_name": data["name"].upper(),
            "age_group": "adult" if data["age"] >= 18 else "minor"
        }
        
        return {
            "transformed": transformed,
            "original": data
        }
```

### Creating a Composite Instruction

Now let's compose these instructions:

```python
from agentflow.core.isa import CompositeInstruction

class DataProcessingFlow(CompositeInstruction):
    def __init__(self):
        # Create component instructions
        validator = ValidateDataInstruction()
        transformer = TransformDataInstruction()
        
        super().__init__(
            id="data_processing_flow",
            name="data_processing_flow",
            components=[validator, transformer],
            composition_type="sequential"
        )
```

### Using the Composite Instruction

```python
from agentflow.core.isa import AgentISA

async def main():
    agent_isa = AgentISA()
    
    try:
        # Register composite instruction
        flow = DataProcessingFlow()
        flow_id = await agent_isa.register_instruction(flow)
        
        # Execute flow
        result = await agent_isa.execute_instruction(
            flow_id,
            {
                "data": {
                    "name": "Alice",
                    "age": 25
                }
            }
        )
        
        print(f"Result: {result}")
        
    finally:
        await agent_isa.cleanup()
```

## 2. Optimization Techniques

### Static Optimization

Enable static optimization in your configuration:

```python
from agentflow.core.isa import AgentISAConfig

config = AgentISAConfig(
    enable_optimization=True,
    enable_learning=False  # Focus on optimization only
)

agent_isa = AgentISA(config)
```

### Adding Optimization Patterns

Create instructions with optimization opportunities:

```python
class OptimizableInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="optimizable",
            name="optimizable",
            type=InstructionType.CONTROL_FLOW,
            params={
                "data": TypeConstraint(base_type=list),
                "operation": TypeConstraint(base_type=str)
            }
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        operation = input_params["operation"]
        
        if operation == "sort_and_filter":
            # This can be optimized
            sorted_data = sorted(data)
            filtered_data = [x for x in sorted_data if x > 0]
            return {"result": filtered_data}
        
        return {"result": data}
```

### Dynamic Optimization

Set up dynamic optimization thresholds:

```python
config = AgentISAConfig(
    enable_optimization=True,
    hot_path_threshold=50,  # Detect hot paths after 50 executions
    stability_threshold=0.9  # Require 90% stability for optimization
)

async def main():
    agent_isa = AgentISA(config)
    
    try:
        instruction = OptimizableInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Execute multiple times to trigger optimization
        for _ in range(100):
            data = list(range(-10, 11))  # Sample data
            result = await agent_isa.execute_instruction(
                instruction_id,
                {
                    "data": data,
                    "operation": "sort_and_filter"
                }
            )
        
        # Check optimization stats
        stats = agent_isa.get_instruction_stats()
        print("\nOptimization Statistics:")
        print(json.dumps(stats["optimization_stats"], indent=2))
        
    finally:
        await agent_isa.cleanup()
```

## 3. Advanced Composition Patterns

### Parallel Composition

```python
class ParallelDataProcessor(CompositeInstruction):
    def __init__(self):
        processors = [
            DataProcessingFlow(),
            DataProcessingFlow(),
            DataProcessingFlow()
        ]
        
        super().__init__(
            id="parallel_processor",
            name="parallel_processor",
            components=processors,
            composition_type="parallel"
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Execute components in parallel
        tasks = [
            component.execute(input_params)
            for component in self.components
        ]
        
        results = await asyncio.gather(*tasks)
        return {
            "results": results,
            "count": len(results)
        }
```

### Conditional Composition

```python
class ConditionalProcessor(CompositeInstruction):
    def __init__(self):
        super().__init__(
            id="conditional_processor",
            name="conditional_processor",
            components=[
                ValidateDataInstruction(),
                TransformDataInstruction()
            ],
            composition_type="conditional"
        )
    
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        
        # Execute first component (validation)
        validation_result = await self.components[0].execute({"data": data})
        
        # Only transform if validation succeeds
        if validation_result["valid"]:
            transform_result = await self.components[1].execute({"data": data})
            return {
                "success": True,
                "result": transform_result
            }
        
        return {
            "success": False,
            "error": "Validation failed"
        }
```

## 4. Monitoring and Tuning

### Performance Monitoring

```python
async def monitor_performance():
    agent_isa = AgentISA()
    
    try:
        instruction = OptimizableInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Execute with monitoring
        execution_times = []
        memory_usage = []
        
        for i in range(100):
            start_time = time.perf_counter()
            result = await agent_isa.execute_instruction(
                instruction_id,
                {
                    "data": list(range(1000)),
                    "operation": "sort_and_filter"
                }
            )
            execution_time = time.perf_counter() - start_time
            execution_times.append(execution_time)
            
            # Get memory usage
            memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_usage.append(memory)
            
            if i % 10 == 0:
                print(f"\nIteration {i}:")
                print(f"Execution Time: {execution_time:.4f}s")
                print(f"Memory Usage: {memory:.2f}MB")
        
        # Print statistics
        print("\nPerformance Statistics:")
        print(f"Average Execution Time: {sum(execution_times)/len(execution_times):.4f}s")
        print(f"Average Memory Usage: {sum(memory_usage)/len(memory_usage):.2f}MB")
        
    finally:
        await agent_isa.cleanup()
```

## Next Steps

1. Experiment with different composition patterns
2. Create custom optimization strategies
3. Implement complex workflows
4. Profile and optimize performance

Check out the next tutorial for instruction learning and pattern recognition. 