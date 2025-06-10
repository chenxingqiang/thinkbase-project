# AgentISA Framework Usage Examples

## Basic Usage

### 1. Simple Function Registration and Execution

```python
from agentflow.core.isa import AgentISA
from typing import Dict, Any

# Create AgentISA instance
agent_isa = AgentISA()

# Define a function to register
async def greet_user(name: str, age: int = 25) -> Dict[str, Any]:
    return {
        "message": f"Hello {name}, you are {age} years old!",
        "status": "success"
    }

# Register the function
instruction_id = await agent_isa.register_function(greet_user)

# Execute the instruction
result = await agent_isa.execute_instruction(
    instruction_id,
    {"name": "Alice", "age": 30}
)
print(result)  # {"message": "Hello Alice, you are 30 years old!", "status": "success"}
```

### 2. Custom Instruction Creation

```python
from agentflow.core.isa import Instruction, InstructionType, TypeConstraint

class CalculatorInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="calculator",
            name="calculator",
            type=InstructionType.CONTROL_FLOW,
            params={
                "operation": TypeConstraint(base_type=str),
                "x": TypeConstraint(base_type=float),
                "y": TypeConstraint(base_type=float)
            },
            description="Performs basic arithmetic operations"
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        op = input_params["operation"]
        x = input_params["x"]
        y = input_params["y"]
        
        if op == "add":
            return x + y
        elif op == "subtract":
            return x - y
        elif op == "multiply":
            return x * y
        elif op == "divide":
            if y == 0:
                raise ValueError("Division by zero")
            return x / y
        else:
            raise ValueError(f"Unknown operation: {op}")

# Register and use the custom instruction
agent_isa = AgentISA()
calc_id = await agent_isa.register_instruction(CalculatorInstruction())

result = await agent_isa.execute_instruction(
    calc_id,
    {"operation": "add", "x": 5.0, "y": 3.0}
)
print(result)  # 8.0
```

### 3. Using Context Requirements

```python
from agentflow.core.isa import AgentISA

# Create AgentISA instance
agent_isa = AgentISA()

# Define a function with context requirements
async def update_user_profile(name: str, age: int) -> Dict[str, Any]:
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

# Register function with context requirements
instruction_id = await agent_isa.register_function(
    update_user_profile,
    context_requirements=context_requirements
)

# Execute with context
result = await agent_isa.execute_instruction(
    instruction_id,
    {
        "name": "Bob",
        "age": 25,
        "auth_token": "valid_token_123",
        "user_id": 42
    }
)
```

## Advanced Usage

### 1. Instruction Learning

```python
from agentflow.core.isa import AgentISA, AgentISAConfig

# Configure AgentISA with learning enabled
config = AgentISAConfig(
    enable_learning=True,
    min_pattern_support=2,
    min_success_rate=0.8
)
agent_isa = AgentISA(config)

# Execute some instructions multiple times
# The system will automatically learn patterns

# Trigger learning from execution history
await agent_isa.learn_from_history()

# Check learned instructions
stats = agent_isa.get_instruction_stats()
print(f"Learned {stats['learned_instructions']} new instructions")
```

### 2. Instruction Optimization

```python
from agentflow.core.isa import AgentISA, AgentISAConfig

# Configure AgentISA with optimization enabled
config = AgentISAConfig(
    enable_optimization=True,
    hot_path_threshold=50,
    stability_threshold=0.9
)
agent_isa = AgentISA(config)

# Execute instructions multiple times
# The system will automatically optimize frequently used paths

# Check optimization stats
stats = agent_isa.get_instruction_stats()
print(f"Optimization stats: {stats['optimization_stats']}")
```

### 3. Composite Instructions

```python
from agentflow.core.isa import (
    Instruction,
    CompositeInstruction,
    InstructionType,
    TypeConstraint
)

# Create component instructions
class ValidateUserInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="validate_user",
            name="validate_user",
            type=InstructionType.CONTROL_FLOW,
            params={
                "user_id": TypeConstraint(base_type=int),
                "auth_token": TypeConstraint(base_type=str)
            }
        )

    async def execute(self, input_params: Dict[str, Any]) -> bool:
        # Validation logic here
        return True

class UpdateProfileInstruction(Instruction):
    def __init__(self):
        super().__init__(
            id="update_profile",
            name="update_profile",
            type=InstructionType.STATE_MANAGEMENT,
            params={
                "user_id": TypeConstraint(base_type=int),
                "name": TypeConstraint(base_type=str),
                "age": TypeConstraint(base_type=int)
            }
        )

    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        # Update logic here
        return {"status": "updated"}

# Create composite instruction
composite = CompositeInstruction(
    id="user_update_flow",
    name="user_update_flow",
    components=[
        ValidateUserInstruction(),
        UpdateProfileInstruction()
    ]
)

# Register and use composite instruction
agent_isa = AgentISA()
flow_id = await agent_isa.register_instruction(composite)

result = await agent_isa.execute_instruction(
    flow_id,
    {
        "user_id": 42,
        "auth_token": "valid_token_123",
        "name": "Charlie",
        "age": 35
    }
)
```

### 4. Cross-Model Compatibility

```python
from agentflow.core.isa import AgentISA, AgentISAConfig, ModelAdapter

# Create custom model adapter
class CustomModelAdapter(ModelAdapter):
    def __init__(self):
        super().__init__("custom_model")
        
        # Register custom type transformations
        self.register_transformation(
            "int",
            "string",
            lambda x: str(x)
        )

# Configure AgentISA with custom model
config = AgentISAConfig(model_type="custom_model")
agent_isa = AgentISA(config)

# The system will now use custom type transformations
# when executing instructions
```

## Error Handling

```python
from agentflow.core.isa import AgentISA

agent_isa = AgentISA()

try:
    # Try to execute non-existent instruction
    await agent_isa.execute_instruction(
        "non_existent_id",
        {"param": "value"}
    )
except ValueError as e:
    print(f"Instruction not found: {e}")

try:
    # Try to execute with invalid parameters
    await agent_isa.execute_instruction(
        instruction_id,
        {"invalid_param": "value"}
    )
except TypeError as e:
    print(f"Invalid parameters: {e}")
```

## Resource Management

```python
from agentflow.core.isa import AgentISA

async def main():
    agent_isa = AgentISA()
    
    try:
        # Register and execute instructions
        instruction_id = await agent_isa.register_function(some_function)
        result = await agent_isa.execute_instruction(instruction_id, params)
        
    finally:
        # Clean up resources
        await agent_isa.cleanup()

# Run with proper resource management
await main()
``` 