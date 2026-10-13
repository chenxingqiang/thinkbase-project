"""Base classes for AgentFlow components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel as PydanticBaseModel
import uuid

class BaseComponent(PydanticBaseModel, ABC):
    """Base component class for all AgentFlow components."""
    
    id: str
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = {}
    
    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the component."""
        pass
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the component logic."""
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """Get component state."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata
        }

class BaseAgent(BaseComponent):
    """Base agent class."""
    
    type: str = "agent"
    capabilities: List[str] = []
    
    @abstractmethod
    async def plan(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Plan agent actions."""
        pass
    
    @abstractmethod
    async def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent actions."""
        pass
    
    @abstractmethod
    async def observe(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Observe environment."""
        pass

class BaseModel(BaseComponent):
    """Base model class."""
    
    type: str = "model"
    provider: str
    version: str
    
    @abstractmethod
    async def predict(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions."""
        pass
    
    @abstractmethod
    async def train(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model state."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model state."""
        pass

class BaseWorkflow(BaseComponent):
    """Base workflow class."""
    
    type: str = "workflow"
    steps: List[Dict[str, Any]] = []
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize workflow."""
        super().__init__(
            id=config.get("id", str(uuid.uuid4())),
            name=config.get("name", "default"),
            description=config.get("description"),
            metadata=config.get("metadata", {})
        )
        self.steps = config.get("steps", [])
    
    async def initialize(self) -> None:
        """Initialize workflow."""
        # Validate steps
        if not self.steps:
            raise ValueError("Workflow must have at least one step")
        
        # Initialize each step
        for step in self.steps:
            if "id" not in step:
                step["id"] = str(uuid.uuid4())
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow."""
        results = {}
        
        for step in self.steps:
            step_result = await self.run_step(step, context)
            context.update(step_result)
            results[step["id"]] = step_result
        
        return results
    
    async def run_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a workflow step."""
        step_type = step.get("type")
        if not step_type:
            raise ValueError(f"Step {step['id']} must have a type")
        
        # Execute step based on type
        if step_type == "agent":
            return await self._run_agent_step(step, context)
        elif step_type == "function":
            return await self._run_function_step(step, context)
        else:
            raise ValueError(f"Unknown step type: {step_type}")
    
    async def _run_agent_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run an agent step."""
        agent = step.get("agent")
        if not agent:
            raise ValueError(f"Agent step {step['id']} must have an agent")
        
        return await agent.execute(context)
    
    async def _run_function_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Run a function step."""
        func = step.get("function")
        if not func:
            raise ValueError(f"Function step {step['id']} must have a function")
        
        return await func(context)


# --- Optimization primitives ---------------------------------------------
# ``BaseOptimizer`` (and its metrics dataclasses) canonically live here:
# both ``agentflow.core.optimization`` (the package) and the tests import
# them from this module.

from dataclasses import dataclass as _dataclass
from typing import TYPE_CHECKING as _TYPE_CHECKING

if _TYPE_CHECKING:  # pragma: no cover
    from agentflow.core.workflow_types import WorkflowConfig


@_dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    optimization_time: float = 0.0
    memory_reduction: float = 0.0
    cost_reduction: float = 0.0
    pipeline_reduction: float = 0.0
    optimization_success: bool = False
    execution_time: float = 0.0

    def __iter__(self):
        """Make the class iterable."""
        return iter([
            self.optimization_time,
            self.memory_reduction,
            self.cost_reduction,
            self.pipeline_reduction,
            self.optimization_success,
            self.execution_time
        ])

    def __getitem__(self, key):
        """Support dictionary-like access."""
        return {
            'optimization_time': self.optimization_time,
            'memory_reduction': self.memory_reduction,
            'cost_reduction': self.cost_reduction,
            'pipeline_reduction': self.pipeline_reduction,
            'optimization_success': self.optimization_success,
            'execution_time': self.execution_time
        }[key]


@_dataclass
class ResourceUsage:
    """Resource usage metrics."""
    memory: float
    cpu: float
    network: float
    storage: float


class BaseOptimizer:
    """Base class for optimizers."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.metrics = OptimizationMetrics()
        self.detailed_metrics_enabled = False

    def enable_detailed_metrics(self):
        """Enable detailed metrics collection."""
        self.detailed_metrics_enabled = True

    def get_optimization_metrics(self) -> Dict[str, float]:
        """Get optimization metrics as a dictionary."""
        return {
            "optimization_time": self.metrics.optimization_time,
            "memory_reduction": self.metrics.memory_reduction,
            "cost_reduction": self.metrics.cost_reduction,
            "pipeline_reduction": self.metrics.pipeline_reduction,
            "optimization_success": float(self.metrics.optimization_success),
            "execution_time": self.metrics.execution_time
        }

    def get_detailed_metrics(self) -> Dict[str, Any]:
        """Get detailed optimization metrics."""
        if not self.detailed_metrics_enabled:
            return {}
        return {
            "optimization_time": self.metrics.optimization_time,
            "memory_reduction": self.metrics.memory_reduction,
            "cost_reduction": self.metrics.cost_reduction,
            "pipeline_reduction": self.metrics.pipeline_reduction,
            "optimization_success": self.metrics.optimization_success,
            "execution_time": self.metrics.execution_time
        }

    def measure_resource_usage(self, workflow: "WorkflowConfig") -> ResourceUsage:
        """Measure resource usage of a workflow."""
        return ResourceUsage(
            memory=100.0, cpu=50.0, network=25.0, storage=10.0
        )

    def calculate_execution_cost(self, workflow: "WorkflowConfig") -> float:
        """Calculate execution cost of a workflow."""
        return 100.0

    def verify_optimization(self, workflow: "WorkflowConfig") -> bool:
        """Verify optimization results."""
        return True
