# AgentISA Framework API Documentation

## Core Components

### 1. Instruction System (`agentflow.core.isa.instruction`)

#### Classes

##### `Instruction`
Base class for all instructions in the AgentISA framework.

```python
class Instruction:
    def __init__(self, id: str, name: str, type: InstructionType, 
                 params: Dict[str, TypeConstraint], description: Optional[str] = None)
```

**Parameters:**
- `id`: Unique identifier for the instruction
- `name`: Human-readable name
- `type`: Type of instruction (from InstructionType enum)
- `params`: Dictionary of parameter names to type constraints
- `description`: Optional description of the instruction

**Methods:**
- `add_precondition(condition: Callable[..., bool])`: Add a precondition
- `add_postcondition(condition: Callable[..., bool])`: Add a postcondition
- `verify_conditions(state: Dict[str, Any]) -> bool`: Verify pre/post conditions
- `validate_types(input_params: Dict[str, Any]) -> bool`: Validate parameter types
- `execute(input_params: Dict[str, Any]) -> Any`: Execute the instruction

##### `CompositeInstruction`
Supports composition of multiple instructions.

```python
class CompositeInstruction(Instruction):
    def __init__(self, id: str, name: str, components: List[Instruction], 
                 composition_type: str = "sequential")
```

### 2. Agentic Calling Framework (`agentflow.core.isa.agentic_calling`)

#### Classes

##### `AgenticTransformer`
Transforms function calls into agent instructions.

```python
class AgenticTransformer:
    def transform_to_instruction(self, func: Callable, 
                               context_requirements: Optional[Dict[str, Any]] = None) -> Instruction
    def add_context_constraint(self, name: str, constraint: Callable[..., bool])
```

##### `ModelAdapter`
Handles cross-model compatibility.

```python
class ModelAdapter:
    def register_transformation(self, source_type: str, target_type: str,
                              transform_func: Callable[[Any], Any])
    def translate_instruction(self, instruction: Instruction) -> Dict[str, Any]
```

### 3. Instruction Learning System (`agentflow.core.isa.instruction_learning`)

#### Classes

##### `PatternRecognizer`
Recognizes patterns in dialogue sequences.

```python
class PatternRecognizer:
    def extract_patterns(self, dialogues: List[Dict[str, Any]]) -> List[DialoguePattern]
```

##### `InstructionLearner`
Learns new instructions from dialogue patterns.

```python
class InstructionLearner:
    def learn_from_dialogues(self, dialogues: List[Dict[str, Any]])
    def create_instruction(self, template: InstructionTemplate) -> Instruction
```

### 4. Instruction Optimization System (`agentflow.core.isa.instruction_optimization`)

#### Classes

##### `InstructionDAG`
Manages instruction dependencies.

```python
class InstructionDAG:
    def add_instruction(self, instruction: Instruction, 
                       dependencies: Optional[List[str]] = None)
    def get_execution_order(self) -> List[str]
```

##### `PipelineScheduler`
Schedules instructions into pipeline stages.

```python
class PipelineScheduler:
    def schedule(self, instructions: List[Instruction]) -> List[List[Instruction]]
```

##### `StaticOptimizer`
Performs static optimization of instructions.

```python
class StaticOptimizer:
    def optimize(self, instructions: List[Instruction]) -> List[Instruction]
```

##### `DynamicOptimizer`
Performs dynamic optimization based on execution history.

```python
class DynamicOptimizer:
    def record_execution(self, path_id: str, instructions: List[Instruction], 
                        metrics: OptimizationMetrics)
```

### 5. Main Interface (`agentflow.core.isa.agent_isa`)

#### Classes

##### `AgentISA`
Main interface for the AgentISA framework.

```python
class AgentISA:
    async def register_function(self, func: Callable, 
                              context_requirements: Optional[Dict[str, Any]] = None) -> str
    async def register_instruction(self, instruction: Instruction) -> str
    async def execute_instruction(self, instruction_id: str, 
                                params: Dict[str, Any]) -> Any
    async def learn_from_history(self)
    def get_instruction_stats(self) -> Dict[str, Any]
```

## Configuration

### `AgentISAConfig`
Configuration for the AgentISA framework.

```python
@dataclass
class AgentISAConfig:
    enable_learning: bool = True
    enable_optimization: bool = True
    model_type: str = "default"
    min_pattern_support: int = 3
    min_success_rate: float = 0.7
    hot_path_threshold: int = 100
    stability_threshold: float = 0.9
```

## Type Definitions

### `InstructionType`
Enum defining instruction categories.

```python
class InstructionType(str, Enum):
    CONTROL_FLOW = "CONTROL_FLOW"
    STATE_MANAGEMENT = "STATE_MANAGEMENT"
    LLM_INTERACTION = "LLM_INTERACTION"
    RESOURCE_MANAGEMENT = "RESOURCE_MANAGEMENT"
```

### `TypeConstraint`
Type constraint with dependent type support.

```python
@dataclass
class TypeConstraint(Generic[T]):
    base_type: type
    dependent_on: Optional[Callable[[T], bool]] = None
    recursive: bool = False
``` 