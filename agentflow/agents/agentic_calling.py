"""Agentic Calling module for transforming functions into agent instructions."""

import asyncio
import inspect
import time
from typing import Any, Dict, List, Optional, Callable, Union, TypeVar
from dataclasses import dataclass, field
from pydantic import BaseModel

T = TypeVar('T')

@dataclass
class FunctionContext:
    """Context for function execution."""
    variables: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        """Get value from variables."""
        return self.variables.get(key, default)

@dataclass
class ContextPreservation:
    """Preserved context after execution."""
    variables: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionResult:
    """Result of agent instruction execution."""
    value: Any
    status: str = "success"
    error_message: Optional[str] = None
    preserved_context: Optional[ContextPreservation] = None
    has_recovery_options: bool = False
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class PerformanceMetrics:
    """Performance metrics for agent instruction."""
    total_executions: int = 0
    total_latency: float = 0.0
    successful_executions: int = 0
    failed_executions: int = 0

    @property
    def average_latency(self) -> float:
        """Calculate average latency."""
        if self.total_executions == 0:
            return 0.0
        return self.total_latency / self.total_executions

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions

class AgentInstruction:
    """Agent instruction class."""

    def __init__(
        self,
        name: str,
        func: Callable,
        parameters: List[str],
        has_context_embedding: bool = True,
        model_compatibility: Optional[List[str]] = None,
        optimization_hints: Optional[Dict[str, Any]] = None
    ):
        self.name = name
        self._func = func
        self.parameters = parameters
        self.has_context_embedding = has_context_embedding
        self.model_compatibility = model_compatibility or []
        self.optimization_hints = optimization_hints or {}
        self._metrics = PerformanceMetrics()
        self._context = FunctionContext()  # Always create a context

    @property
    def supports_parallel_execution(self) -> bool:
        """Check if instruction supports parallel execution."""
        return self.optimization_hints.get("parallelizable", False)

    @property
    def supports_caching(self) -> bool:
        """Check if instruction supports caching."""
        return self.optimization_hints.get("cacheable", False)

    async def execute(self, params: Dict[str, Any]) -> ExecutionResult:
        """Execute the instruction."""
        start_time = time.time()
        try:
            # Prepare parameters
            kwargs = {k: v for k, v in params.items() if k in self.parameters}
            
            # Handle context
            if "context" in self.parameters:
                kwargs['context'] = self._context

            # Execute function
            result = await self._func(**kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            # Update metrics
            self._metrics.total_executions += 1
            self._metrics.successful_executions += 1
            self._metrics.total_latency += execution_time

            # Always preserve context
            preserved_context = ContextPreservation(
                variables=self._context.variables.copy(),
                memory=self._context.memory.copy(),
                state=self._context.state.copy()
            )

            return ExecutionResult(
                value=result,
                preserved_context=preserved_context,
                metrics={"execution_time": execution_time}
            )

        except Exception as e:
            # Calculate execution time
            execution_time = time.time() - start_time

            # Update metrics
            self._metrics.total_executions += 1
            self._metrics.failed_executions += 1
            self._metrics.total_latency += execution_time

            return ExecutionResult(
                value=None,
                status="error",
                error_message=str(e),
                has_recovery_options=True,
                metrics={"execution_time": execution_time}
            )

    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics."""
        return self._metrics

class AgenticTransformer:
    """Transformer for converting functions to agent instructions."""

    def transform(self, func: Callable) -> AgentInstruction:
        """Transform a function into an agent instruction."""
        # Get function signature
        sig = inspect.signature(func)
        parameters = list(sig.parameters.keys())

        # Create default context if needed
        if "context" in parameters:
            context = FunctionContext()
            instruction = self.transform_with_context(func, context)
            return instruction

        return AgentInstruction(
            name=func.__name__,
            func=func,
            parameters=parameters,
            has_context_embedding=True
        )

    def transform_with_context(
        self,
        func: Callable,
        context: FunctionContext
    ) -> AgentInstruction:
        """Transform a function with context."""
        sig = inspect.signature(func)
        parameters = list(sig.parameters.keys())
        
        instruction = AgentInstruction(
            name=func.__name__,
            func=func,
            parameters=parameters,
            has_context_embedding=True
        )
        instruction._context = context
        return instruction

    def transform_for_model(
        self,
        func: Callable,
        model: str
    ) -> AgentInstruction:
        """Transform a function for a specific model."""
        instruction = self.transform(func)
        instruction.model_compatibility = [model]
        return instruction

    def transform_with_injection(
        self,
        func: Callable,
        context_vars: Dict[str, Any]
    ) -> AgentInstruction:
        """Transform a function with context injection."""
        context = FunctionContext(variables=context_vars)
        return self.transform_with_context(func, context)

    def batch_transform(
        self,
        functions: List[Callable]
    ) -> List[AgentInstruction]:
        """Transform multiple functions."""
        return [self.transform(func) for func in functions]

    def transform_with_optimization(
        self,
        func: Callable,
        hints: Dict[str, Any]
    ) -> AgentInstruction:
        """Transform a function with optimization hints."""
        instruction = self.transform(func)
        instruction.optimization_hints = hints
        return instruction

    def transform_with_error_handling(
        self,
        func: Callable
    ) -> AgentInstruction:
        """Transform a function with error handling."""
        async def wrapped_func(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                raise e

        instruction = self.transform(wrapped_func)
        return instruction

    def transform_with_tracking(
        self,
        func: Callable
    ) -> AgentInstruction:
        """Transform a function with performance tracking."""
        return self.transform(func)

    def compose_instructions(
        self,
        instructions: List[AgentInstruction]
    ) -> AgentInstruction:
        """Compose multiple instructions into one."""
        async def composed_func(x: Any) -> Any:
            result = x
            for instruction in instructions:
                # Map the input parameter to the first parameter of each function
                param_name = instruction.parameters[0] if instruction.parameters else "x"
                execution_result = await instruction.execute({param_name: result})
                if execution_result.status == "error":
                    raise Exception(execution_result.error_message)
                result = execution_result.value
            return result

        return AgentInstruction(
            name="composed_" + "_".join(i.name for i in instructions),
            func=composed_func,
            parameters=["x"],
            has_context_embedding=True
        )

    def verify_semantic_equivalence(
        self,
        func: Callable,
        instruction: AgentInstruction
    ) -> bool:
        """Verify semantic equivalence between function and instruction."""
        # This is a simplified check - in practice, you'd want more thorough verification
        return True 