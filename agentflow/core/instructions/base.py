"""Base instruction classes for AgentFlow."""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Type, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
import os
import time

logger = logging.getLogger(__name__)

class InstructionType(str, Enum):
    """Instruction categories."""
    CONTROL_FLOW = "CONTROL_FLOW"  # Flow control, branching, loops
    STATE_MANAGEMENT = "STATE_MANAGEMENT"  # State handling, context management
    LLM_INTERACTION = "LLM_INTERACTION"  # LLM operations
    RESOURCE_MANAGEMENT = "RESOURCE_MANAGEMENT"  # Resource handling
    COLLABORATION = "COLLABORATION"  # Agent collaboration
    DATA_PROCESSING = "DATA_PROCESSING"  # Data operations
    DATA_MANAGEMENT = "DATA_MANAGEMENT"  # Data management operations
    OBJECTIVE_HANDLING = "OBJECTIVE_HANDLING"  # Objective management
    CODE_REVIEW = "CODE_REVIEW"  # Code review operations
    REFACTORING = "REFACTORING"  # Code refactoring operations
    CODE_GENERATION = "CODE_GENERATION"  # Code generation operations
    TEST_DESIGN = "TEST_DESIGN"  # Test design operations
    ANALYSIS = "ANALYSIS"  # Analysis operations

@dataclass
class InstructionMetadata:
    """Metadata for instructions."""
    capabilities: Set[str] = field(default_factory=set)
    dependencies: Set[str] = field(default_factory=set)
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    execution_priority: int = 0
    timeout_seconds: Optional[float] = None
    retry_policy: Dict[str, Any] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    cleanup_required: bool = False

@dataclass
class ExecutionContext:
    """Context for instruction execution."""
    params: Dict[str, Any]
    state: Dict[str, Any]
    metadata: Dict[str, Any]
    collaboration_info: Optional[Dict[str, Any]] = None
    resources: Optional[Dict[str, Any]] = None

    def validate_required_params(self, required: Set[str]) -> Tuple[bool, Optional[str]]:
        """Validate required parameters are present."""
        missing = required - set(self.params.keys())
        if missing:
            return False, f"Missing required parameters: {missing}"
        return True, None

    def validate_resources(self, requirements: Dict[str, float]) -> Tuple[bool, Optional[str]]:
        """Validate required resources are available."""
        if not self.resources:
            return False, "No resources available"
        for resource, amount in requirements.items():
            if resource not in self.resources:
                return False, f"Missing resource: {resource}"
            if self.resources[resource] < amount:
                return False, f"Insufficient {resource}"
        return True, None

@dataclass
class ExecutionResult:
    """Result of instruction execution.

    Supports two equivalent construction styles:

    * Legacy style: ``ExecutionResult(success=..., output=..., errors=[...],
      state_updates=..., metrics=...)``
    * Instruction style: ``InstructionResult(status=..., data=..., error=...,
      metrics=...)``

    ``InstructionResult`` is an alias of this class, and both views are kept
    in sync automatically by ``__post_init__``.
    """
    # Legacy view
    success: Optional[bool] = None
    output: Dict[str, Any] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    metrics: Any = field(default_factory=dict)
    # Instruction-style view
    status: Any = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    def __post_init__(self) -> None:
        """Reconcile legacy and instruction-style views of the result."""
        # --- status <-> success ---
        if self.status is None:
            if self.success is True:
                self.status = InstructionStatus.COMPLETED
            elif self.success is False:
                self.status = InstructionStatus.FAILED
            else:
                self.status = InstructionStatus.PENDING
        elif self.success is None:
            self.success = self.status in (
                InstructionStatus.COMPLETED, InstructionStatus.SUCCESS
            )
        # --- data <-> output ---
        if self.data is None:
            self.data = self.output
        elif not self.output:
            self.output = self.data
        # --- error <-> errors ---
        if self.error is None and self.errors:
            first = self.errors[0]
            self.error = first.get("error") if isinstance(first, dict) else str(first)
        elif self.error is not None and not self.errors:
            self.errors = [{"error": self.error}]

@dataclass
class InstructionMetrics:
    """Metrics for a single instruction execution."""
    start_time: float = 0.0
    end_time: float = 0.0
    tokens_used: int = 0
    memory_used: int = 0
    cache_hit: bool = False
    optimization_applied: bool = False
    parallel_execution: bool = False

    @property
    def execution_time(self) -> float:
        """Calculate total execution time."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0

    @property
    def duration(self) -> float:
        """Alias for execution_time."""
        return self.execution_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a plain dictionary."""
        return {
            "start_time": self.start_time,
            "end_time": self.end_time,
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "memory_used": self.memory_used,
            "cache_hit": self.cache_hit,
            "optimization_applied": self.optimization_applied,
            "parallel_execution": self.parallel_execution,
        }

    # Mapping protocol so metrics objects can be merged with dicts
    def keys(self):
        return self.to_dict().keys()

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self):
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())

    def __contains__(self, key: object) -> bool:
        return key in self.to_dict()

class BaseInstruction(ABC):
    """Base class for all instructions."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize instruction.
        
        Args:
            name: Instruction name
            description: Optional instruction description
            instruction_type: Type of instruction
            metadata: Optional instruction metadata
        """
        self.name = name
        self.description = description
        self.instruction_type = instruction_type or self._get_default_type()
        self.metadata = metadata or InstructionMetadata()
        # Execution lifecycle state
        self.status: InstructionStatus = InstructionStatus.PENDING
        self.metrics: Optional[InstructionMetrics] = None
        # Pre/post execution hooks
        self._pre_hooks: List[Any] = []
        self._post_hooks: List[Any] = []
        # Aggregate execution counters (reported via to_dict/update_metrics)
        self._execution_count: int = 0
        self._success_count: int = 0
        self._failure_count: int = 0
        self._total_execution_time: float = 0.0
        self._metrics = InstructionMetrics()
        self._required_params: Set[str] = set()
        self._param_types: Dict[str, Type] = {}
        self._temp_files: Set[str] = set()
        self._state: Dict[str, Any] = {}
        self._acquired_resources: Set[Any] = set()
    
    def _get_default_type(self) -> InstructionType:
        """Get default instruction type based on class name."""
        class_name = self.__class__.__name__
        if "Review" in class_name:
            return InstructionType.CODE_REVIEW
        elif "Refactor" in class_name:
            return InstructionType.REFACTORING
        elif "Generate" in class_name or "Creation" in class_name:
            return InstructionType.CODE_GENERATION
        elif "Test" in class_name:
            return InstructionType.TEST_DESIGN
        elif "Analysis" in class_name:
            return InstructionType.ANALYSIS
        return InstructionType.CONTROL_FLOW
    
    async def execute(self, context: Any) -> ExecutionResult:
        """Execute instruction.

        Template method that drives the execution lifecycle. Subclasses
        normally override ``_execute_impl`` and get this machinery for free;
        they may also override ``execute`` itself for full control.

        1. Run pre-hooks (instruction status is still ``PENDING``).
        2. Transition status to ``RUNNING`` and invoke ``_execute_impl``.
        3. Run post-hooks (instruction status is still ``RUNNING``).
        4. Transition status to ``COMPLETED`` (or ``FAILED`` on error).

        Args:
            context: Execution context (usually a plain dict)

        Returns:
            ExecutionResult: Execution results
        """
        return await self._run_execution_lifecycle(context)

    async def _run_execution_lifecycle(self, context: Any) -> ExecutionResult:
        """Run the full execution lifecycle with hooks and status tracking."""
        start_time = time.time()
        if self.metrics is None:
            self.metrics = InstructionMetrics()
        self.metrics.start_time = start_time

        def _fail(error: str) -> ExecutionResult:
            self.status = InstructionStatus.FAILED
            self.metrics.end_time = time.time()
            self._execution_count += 1
            self._failure_count += 1
            self._total_execution_time += self.metrics.execution_time
            return ExecutionResult(
                status=InstructionStatus.FAILED,
                data={},
                error=error,
                metrics=self.metrics
            )

        try:
            # 1. Pre-hooks run while the status is still PENDING
            for hook in self._pre_hooks:
                await hook(self, context)
        except Exception as e:
            return _fail(str(e))

        # 2. Run the implementation
        self.status = InstructionStatus.RUNNING
        try:
            impl_result = await self._execute_impl(context)
        except Exception as e:
            return _fail(str(e))

        try:
            # 3. Post-hooks run while the status is still RUNNING
            for hook in self._post_hooks:
                await hook(self, context)
        except Exception as e:
            return _fail(str(e))

        # 4. Build the final result
        if isinstance(impl_result, ExecutionResult):
            result = impl_result
            if result.status is None:
                result.status = InstructionStatus.COMPLETED
            if result.metrics is None:
                result.metrics = self.metrics
        elif isinstance(impl_result, dict):
            result = ExecutionResult(
                status=InstructionStatus.COMPLETED,
                data=impl_result,
                error=None,
                metrics=self.metrics
            )
        elif impl_result is None:
            result = ExecutionResult(
                status=InstructionStatus.COMPLETED,
                data={},
                error=None,
                metrics=self.metrics
            )
        else:
            result = ExecutionResult(
                status=InstructionStatus.COMPLETED,
                data={"output": impl_result},
                error=None,
                metrics=self.metrics
            )

        if result.success:
            self.status = InstructionStatus.COMPLETED
            self._execution_count += 1
            self._success_count += 1
        else:
            self.status = InstructionStatus.FAILED
            self._execution_count += 1
            self._failure_count += 1
        self.metrics.end_time = time.time()
        self._total_execution_time += self.metrics.execution_time
        return result

    async def _execute_impl(self, context: Any) -> Any:
        """Execute the core instruction logic.

        Subclasses override this to do real work. May return a plain dict
        (wrapped into a completed result) or a ready-made ``ExecutionResult``.
        """
        return {}

    def add_pre_hook(self, hook: Any) -> None:
        """Add a hook executed before ``_execute_impl``.

        The hook is awaited with ``(instruction, context)`` and runs while the
        instruction status is still ``PENDING``.
        """
        self._pre_hooks.append(hook)

    def add_post_hook(self, hook: Any) -> None:
        """Add a hook executed after ``_execute_impl``.

        The hook is awaited with ``(instruction, context)`` and runs while the
        instruction status is still ``RUNNING``.
        """
        self._post_hooks.append(hook)

    def _combine_results(self, results: List[Any]) -> Dict[str, Any]:
        """Combine sub-instruction results into a single payload."""
        combined: Dict[str, Any] = {"results": []}
        state_updates: Dict[str, Any] = {}
        errors: List[Dict[str, Any]] = []
        for r in results:
            if isinstance(r, ExecutionResult):
                combined["results"].append(r.data)
                state_updates.update(r.state_updates or {})
                if r.errors:
                    errors.extend(r.errors)
            else:
                combined["results"].append(r)
        if state_updates:
            combined["state_updates"] = state_updates
        if errors:
            combined["errors"] = errors
        return combined
    
    async def validate_context(self, context: ExecutionContext) -> bool:
        """Validate execution context.
        
        Args:
            context: Context to validate
            
        Returns:
            bool: True if context is valid
        """
        try:
            # Check required parameters
            for param in self._required_params:
                if param not in context.params:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Validate parameter types
            for param, value in context.params.items():
                if param in self._param_types:
                    if not isinstance(value, self._param_types[param]):
                        raise TypeError(
                            f"Invalid type for parameter {param}. "
                            f"Expected {self._param_types[param]}, got {type(value)}"
                        )
            
            # Check resource requirements
            if self.metadata.resource_requirements:
                if not context.resources:
                    raise ValueError("Missing resource information in context")
                for resource, amount in self.metadata.resource_requirements.items():
                    if resource not in context.resources:
                        raise ValueError(f"Missing required resource: {resource}")
                    if context.resources[resource] < amount:
                        raise ValueError(f"Insufficient {resource}")
            
            return True
            
        except Exception as e:
            logger.error(f"Context validation failed: {str(e)}")
            return False
    
    def update_metrics(self, success: bool, execution_time: float) -> None:
        """Update aggregate instruction execution metrics.
        
        Args:
            success: Whether execution was successful
            execution_time: Time taken for execution
        """
        self._execution_count += 1
        self._total_execution_time += execution_time
        if success:
            self._success_count += 1
        else:
            self._failure_count += 1
    
    async def cleanup(self) -> None:
        """Clean up instruction resources.
        
        This method should be called after instruction execution to clean up
        any allocated resources. Subclasses should override this method if
        they need custom cleanup logic.
        """
        if not self.metadata.cleanup_required:
            return
            
        try:
            # Release any acquired resources
            for resource in self._acquired_resources:
                await self._release_resource(resource)
            
            # Clean up any temporary files
            if hasattr(self, '_temp_files'):
                for file_path in self._temp_files:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        logger.warning(f"Failed to remove temp file {file_path}: {e}")
            
            # Reset any stateful attributes
            if hasattr(self, '_state'):
                self._state = {}
            
        except Exception as e:
            logger.error(f"Cleanup failed for {self.name}: {e}")
    
    async def _release_resource(self, resource: Any) -> None:
        """Release a specific resource."""
        try:
            if hasattr(resource, 'release'):
                await resource.release()
            elif hasattr(resource, 'close'):
                await resource.close()
        except Exception as e:
            logger.error(f"Failed to release resource: {e}")
    
    def get_required_params(self) -> Set[str]:
        """Get required parameters for this instruction."""
        return self._required_params
    
    def get_param_types(self) -> Dict[str, Type]:
        """Get parameter types for this instruction."""
        return self._param_types
    
    def add_required_param(self, param: str, param_type: Type) -> None:
        """Add a required parameter.
        
        Args:
            param: Parameter name
            param_type: Parameter type
        """
        self._required_params.add(param)
        self._param_types[param] = param_type
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instruction to dictionary representation."""
        return {
            "name": self.name,
            "description": self.description,
            "type": self.instruction_type.value,
            "required_params": list(self._required_params),
            "param_types": {
                name: str(type_) 
                for name, type_ in self._param_types.items()
            },
            "metadata": {
                "capabilities": list(self.metadata.capabilities),
                "dependencies": list(self.metadata.dependencies),
                "resource_requirements": self.metadata.resource_requirements,
                "execution_priority": self.metadata.execution_priority,
                "timeout_seconds": self.metadata.timeout_seconds,
                "retry_policy": self.metadata.retry_policy,
                "tags": list(self.metadata.tags)
            },
            "metrics": {
                "total_executions": self._execution_count,
                "successful_executions": self._success_count,
                "failed_executions": self._failure_count,
                "total_execution_time": self._total_execution_time,
                "success_rate": (
                    self._success_count / self._execution_count
                    if self._execution_count > 0 else 0.0
                ),
                "average_execution_time": (
                    self._total_execution_time / self._execution_count
                    if self._execution_count > 0 else 0.0
                ),
                "cache_hits": 0,
                "cache_misses": 0
            }
        }

class CompositeInstruction(BaseInstruction):
    """Composite instruction that contains multiple sub-instructions."""

    def __init__(
        self,
        name: str,
        description: str = "",
        components: Optional[List[BaseInstruction]] = None
    ):
        """Initialize composite instruction.

        Args:
            name: Instruction name
            description: Optional instruction description
            components: Optional list of component instructions
        """
        super().__init__(name=name, description=description)
        self.instructions: List[BaseInstruction] = list(components) if components else []
        # Legacy alias: older consumers reference ``components``
        self.components = self.instructions

        # Combine capabilities and dependencies
        for component in self.instructions:
            self.metadata.capabilities.update(component.metadata.capabilities)
            self.metadata.dependencies.update(component.metadata.dependencies)

    def add_instruction(self, instruction: BaseInstruction) -> None:
        """Add a sub-instruction."""
        self.instructions.append(instruction)
        self.metadata.capabilities.update(instruction.metadata.capabilities)
        self.metadata.dependencies.update(instruction.metadata.dependencies)

    def add_component(self, component: BaseInstruction) -> None:
        """Legacy alias for :meth:`add_instruction`."""
        self.add_instruction(component)

    async def execute(self, context: Any) -> ExecutionResult:
        """Execute component instructions."""
        return await super().execute(context)

    async def _execute_impl(self, context: Any) -> Any:
        """Execute each component and combine the results."""
        results = []
        errors = []
        state_updates = {}

        # Execute each component
        for component in self.instructions:
            result = await component.execute(context)
            results.append(result)

            if not getattr(result, "success", True):
                errors.extend(getattr(result, "errors", None) or [])
            else:
                state_updates.update(getattr(result, "state_updates", None) or {})

        if errors:
            first = errors[0]
            error_msg = first.get("error") if isinstance(first, dict) else str(first)
            return ExecutionResult(
                status=InstructionStatus.FAILED,
                data=self._combine_results(results),
                error=error_msg,
                metrics={}
            )

        # Return combined results
        return self._combine_results(results)

    async def cleanup(self) -> None:
        """Clean up all components."""
        for component in self.instructions:
            await component.cleanup()

    async def validate_context(self, context: ExecutionContext) -> bool:
        """Validate execution context."""
        try:
            # Check required parameters
            required_params = self.get_required_params()
            for param in required_params:
                if param not in context.params:
                    raise ValueError(f"Missing required parameter: {param}")
            
            # Validate parameter types
            param_types = self.get_param_types()
            for param, value in context.params.items():
                if param in param_types and not isinstance(value, param_types[param]):
                    raise TypeError(f"Invalid type for parameter {param}")
            
            # Check resource availability
            if not await self._check_resources(context):
                raise ValueError("Insufficient resources")
            
            # Validate collaboration requirements
            if self.requires_collaboration():
                if not context.collaboration_info:
                    raise ValueError("Missing collaboration information")
            
            return True
        
        except Exception as e:
            logger.error(f"Context validation failed: {str(e)}")
            return False
    
    async def _check_resources(self, context: ExecutionContext) -> bool:
        """Check if required resources are available."""
        # Implement resource checking logic
        return True
    
    def requires_collaboration(self) -> bool:
        """Check if instruction requires collaboration."""
        return bool(self.metadata.capabilities - self.get_local_capabilities())
    
    def get_local_capabilities(self) -> Set[str]:
        """Get capabilities available locally."""
        # Override in subclasses
        return set()

class FileSystemInstruction(BaseInstruction):
    """Base class for file system operations."""
    
    def __init__(self, name: str, description: str = ""):
        super().__init__(name, description)
        self.metadata.capabilities.add("filesystem_access")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute file system operation."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

class ProcessInstruction(BaseInstruction):
    """Base class for process operations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.metadata.capabilities.add("process_management")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute process operation."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

class NetworkInstruction(BaseInstruction):
    """Base class for network operations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.metadata.capabilities.add("network_access")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute network operation."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

class OptimizableInstruction(BaseInstruction):
    """Base class for optimizable operations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.metadata.capabilities.add("optimization")

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute optimizable operation."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

    async def optimize(self, context: ExecutionContext) -> ExecutionResult:
        """Optimize the instruction execution."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

class CacheableInstruction(BaseInstruction):
    """Base class for cacheable operations."""
    
    def __init__(self, name: str):
        super().__init__(name)
        self.metadata.capabilities.add("caching")
        self._cache = {}

    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute cacheable operation with caching."""
        cache_key = self._get_cache_key(context)
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        result = await self._execute_uncached(context)
        self._cache[cache_key] = result
        return result

    async def _execute_uncached(self, context: ExecutionContext) -> ExecutionResult:
        """Execute operation without caching."""
        return ExecutionResult(success=True, output={}, errors=[], state_updates={}, metrics={})

    def _get_cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key from context."""
        return str(hash(frozenset(context.params.items())))

InstructionResult = ExecutionResult

class InstructionStatus(str, Enum):
    """Status of instruction execution.

    Mirrors ``agentflow.core.isa.types.InstructionStatus`` so that all
    instruction modules can uniformly report execution state.
    """
    SUCCESS = "success"
    ERROR = "error"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class ValidationResult:
    """Result of validating an instruction or its context.

    Attributes:
        is_valid: Whether the validation passed.
        message: Human-readable explanation of the validation outcome.
    """
    is_valid: bool = True
    message: str = ""

# --- Lazy re-exports (PEP 562) -------------------------------------------
# Some instruction classes (e.g. ``ParallelInstruction``) live in sibling
# modules such as ``advanced.py``, which themselves import this module at
# import time.  Re-exporting them eagerly here would create a circular
# import, so the attribute is resolved lazily on first access instead.
# This keeps ``from agentflow.core.instructions.base import
# ParallelInstruction`` working without an import cycle.
_LAZY_RE_EXPORTS = {
    "ParallelInstruction": ".advanced",
}

def __getattr__(name: str) -> Any:
    """Lazily re-export classes defined in sibling instruction modules."""
    module_name = _LAZY_RE_EXPORTS.get(name)
    if module_name is not None:
        import importlib
        module = importlib.import_module(module_name, __package__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__() -> List[str]:
    """Include lazily re-exported names in ``dir()`` output."""
    return sorted(set(globals().keys()) | set(_LAZY_RE_EXPORTS))
