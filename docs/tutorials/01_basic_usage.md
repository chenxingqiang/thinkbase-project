# Tutorial 1: Getting Started with AgentISA

This tutorial will guide you through the basic usage of the AgentISA framework, including setting up your first agent, registering instructions, and executing them.

## Prerequisites

- Python 3.8 or higher
- AgentISA framework installed
- Basic understanding of async/await in Python

## Installation

```bash
pip install agentflow
```

## 1. Creating Your First Agent

Let's start by creating a simple agent that can process text:

```python
from agentflow.core.isa import AgentISA, AgentISAConfig
import asyncio

# Create configuration
config = AgentISAConfig(
    enable_learning=True,
    enable_optimization=True
)

# Initialize agent
agent_isa = AgentISA(config)
```

## 2. Creating a Simple Instruction

Let's create a text processing instruction:

```python
from agentflow.core.isa import Instruction, InstructionType, TypeConstraint
from typing import Dict, Any

class TextProcessorInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="text_processor",
            name="text_processor",
            type=InstructionType.CONTROL_FLOW,
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
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
        return {
            "result": result,
            "operation": operation
        }
```

## 3. Registering and Using the Instruction

Now let's register and use our instruction:

```python
async def main():
    # Create agent
    agent_isa = AgentISA()
    
    try:
        # Register instruction
        instruction = TextProcessorInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Execute instruction
        result = await agent_isa.execute_instruction(
            instruction_id,
            {
                "text": "Hello, World!",
                "operation": "uppercase"
            }
        )
        
        print(f"Result: {result}")  # {"result": "HELLO, WORLD!", "operation": "uppercase"}
        
    finally:
        # Clean up
        await agent_isa.cleanup()

# Run the example
asyncio.run(main())
```

## 4. Using Function Registration

AgentISA can also transform regular functions into instructions:

```python
async def calculate_sum(numbers: list[int]) -> Dict[str, Any]:
    """Calculate sum of numbers."""
    return {
        "sum": sum(numbers),
        "count": len(numbers)
    }

async def main():
    agent_isa = AgentISA()
    
    try:
        # Register function as instruction
        instruction_id = await agent_isa.register_function(calculate_sum)
        
        # Execute instruction
        result = await agent_isa.execute_instruction(
            instruction_id,
            {"numbers": [1, 2, 3, 4, 5]}
        )
        
        print(f"Result: {result}")  # {"sum": 15, "count": 5}
        
    finally:
        await agent_isa.cleanup()
```

## 5. Adding Context Requirements

Let's add context requirements to our function:

```python
async def update_user_data(name: str, age: int) -> Dict[str, Any]:
    """Update user data with validation."""
    return {
        "name": name,
        "age": age,
        "updated": True
    }

# Define context requirements
context_requirements = {
    "auth_token": lambda token: isinstance(token, str) and len(token) > 0,
    "user_id": lambda id: isinstance(id, int) and id > 0
}

async def main():
    agent_isa = AgentISA()
    
    try:
        # Register function with context requirements
        instruction_id = await agent_isa.register_function(
            update_user_data,
            context_requirements=context_requirements
        )
        
        # Execute with context
        result = await agent_isa.execute_instruction(
            instruction_id,
            {
                "name": "Alice",
                "age": 30,
                "auth_token": "valid_token_123",
                "user_id": 42
            }
        )
        
        print(f"Result: {result}")
        
    finally:
        await agent_isa.cleanup()
```

## 6. Error Handling

Always handle errors appropriately:

```python
async def main():
    agent_isa = AgentISA()
    
    try:
        instruction = TextProcessorInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        try:
            # Try invalid operation
            result = await agent_isa.execute_instruction(
                instruction_id,
                {
                    "text": "Hello",
                    "operation": "invalid_op"
                }
            )
        except ValueError as e:
            print(f"Error: {e}")
            
        try:
            # Try invalid instruction ID
            result = await agent_isa.execute_instruction(
                "non_existent_id",
                {"param": "value"}
            )
        except ValueError as e:
            print(f"Error: {e}")
            
    finally:
        await agent_isa.cleanup()
```

## 7. Checking Instruction Statistics

Monitor instruction usage and performance:

```python
async def main():
    agent_isa = AgentISA()
    
    try:
        # Register and execute some instructions
        instruction = TextProcessorInstruction()
        instruction_id = await agent_isa.register_instruction(instruction)
        
        # Execute multiple times
        for text in ["Hello", "World", "AgentISA"]:
            await agent_isa.execute_instruction(
                instruction_id,
                {
                    "text": text,
                    "operation": "uppercase"
                }
            )
        
        # Get statistics
        stats = agent_isa.get_instruction_stats()
        print("\nInstruction Statistics:")
        print(f"Total Instructions: {stats['total_instructions']}")
        print(f"Total Executions: {stats['total_executions']}")
        print(f"Success Rate: {stats['success_rate']:.2%}")
        print(f"Optimization Stats: {stats['optimization_stats']}")
        
    finally:
        await agent_isa.cleanup()
```

## Next Steps

Now that you understand the basics, you can:
1. Create more complex instructions
2. Use instruction composition
3. Implement learning and optimization
4. Build real-world applications

Check out the other tutorials in this series for advanced topics. 