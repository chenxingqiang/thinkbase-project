"""Base instruction classes for AgentFlow."""

from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Type, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import logging
import os

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
    """Result of instruction execution."""
    success: bool
    output: Dict[str, Any]
    errors: List[Dict[str, Any]] = field(default_factory=list)
    state_updates: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class InstructionMetrics:
    """Metrics for instruction execution."""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.total_executions if self.total_executions > 0 else 0.0

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
    
    @abstractmethod
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute instruction.
        
        Args:
            context: Execution context
            
        Returns:
            ExecutionResult: Execution results
            
        Raises:
            ValueError: If required parameters are missing
            TypeError: If parameter types are invalid
        """
        if not await self.validate_context(context):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Context validation failed"}]
            )
        return ExecutionResult(success=True, output={})
    
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
        """Update instruction metrics.
        
        Args:
            success: Whether execution was successful
            execution_time: Time taken for execution
        """
        self._metrics.total_executions += 1
        self._metrics.total_execution_time += execution_time
        if success:
            self._metrics.successful_executions += 1
        else:
            self._metrics.failed_executions += 1
    
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
                "total_executions": self._metrics.total_executions,
                "successful_executions": self._metrics.successful_executions,
                "failed_executions": self._metrics.failed_executions,
                "total_execution_time": self._metrics.total_execution_time,
                "success_rate": self._metrics.success_rate,
                "average_execution_time": self._metrics.average_execution_time,
                "cache_hits": self._metrics.cache_hits,
                "cache_misses": self._metrics.cache_misses
            }
        }

class CompositeInstruction(BaseInstruction):
    """Composite instruction that contains multiple sub-instructions."""
    
    def __init__(self, name: str, description: str, components: List[BaseInstruction]):
        """Initialize composite instruction.
        
        Args:
            name: Instruction name
            description: Instruction description
            components: List of component instructions
        """
        super().__init__(name=name, description=description)
        self.components = components
        
        # Combine capabilities and dependencies
        for component in self.components:
            self.metadata.capabilities.update(component.metadata.capabilities)
            self.metadata.dependencies.update(component.metadata.dependencies)
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute component instructions."""
        try:
            results = []
            errors = []
            state_updates = {}
            
            # Execute each component
            for component in self.components:
                result = await component.execute(context)
                results.append(result)
                
                if not result.success:
                    errors.extend(result.errors)
                else:
                    state_updates.update(result.state_updates)
            
            # Return combined results
            return ExecutionResult(
                success=len(errors) == 0,
                output={"results": [r.output for r in results]},
                errors=errors,
                state_updates=state_updates
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}]
            )
    
    async def cleanup(self) -> None:
        """Clean up all components."""
        for component in self.components:
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
    
    def __init__(self, name: str):
        super().__init__(name)
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

class ValidationResult:
    pass

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
