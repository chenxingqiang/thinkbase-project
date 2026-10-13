"""Instruction Set Architecture implementation following AgentISA paper."""

from enum import Enum
from typing import Any, Dict, Optional, List, Callable, TypeVar, Generic, Set, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import time
import json
import hashlib
from collections import OrderedDict

# Type definitions
T = TypeVar('T')
S = TypeVar('S')

class InstructionType(str, Enum):
    """Instruction categories as defined in AgentISA Section 3.1.1"""
    CONTROL_FLOW = "CONTROL_FLOW"  # seq, branch, loop, call
    STATE_MANAGEMENT = "STATE_MANAGEMENT"  # load, store, update, cache
    LLM_INTERACTION = "LLM_INTERACTION"  # query, response, context
    RESOURCE_MANAGEMENT = "RESOURCE_MANAGEMENT"  # allocate, free, monitor
    PARALLEL_EXECUTION = "PARALLEL_EXECUTION"  # parallel tasks
    CACHE_MANAGEMENT = "CACHE_MANAGEMENT"  # cache operations

@dataclass
class TypeConstraint(Generic[T]):
    """Type constraint with dependent type support"""
    base_type: type
    dependent_on: Optional[Callable[[T], bool]] = None
    recursive: bool = False
    nullable: bool = False
    validation: Optional[Callable[[T], bool]] = None

@dataclass
class InstructionMetrics:
    """Metrics for instruction execution"""
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    total_execution_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    parallel_executions: int = 0
    error_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_executions / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def average_execution_time(self) -> float:
        """Calculate average execution time."""
        return self.total_execution_time / self.total_executions if self.total_executions > 0 else 0.0
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_cache_ops = self.cache_hits + self.cache_misses
        return self.cache_hits / total_cache_ops if total_cache_ops > 0 else 0.0

@dataclass
class InstructionContext:
    """Instruction execution context"""
    state: Dict[str, Any]
    type_env: Dict[str, TypeConstraint]
    preconditions: List[Callable[..., bool]]
    postconditions: List[Callable[..., bool]]
    cache_config: Optional[Dict[str, Any]] = None
    parallel_config: Optional[Dict[str, Any]] = None
    metrics: InstructionMetrics = field(default_factory=InstructionMetrics)

class InstructionCache:
    """Cache for instruction results."""
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if key not in self.cache:
            return None
            
        result, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            del self.cache[key]
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return result
    
    def put(self, key: str, value: Any):
        """Add result to cache with timestamp."""
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
            
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()

class Instruction(ABC):
    """Base instruction class following AgentISA formal specification."""
    
    def __init__(self, 
                 id: str,
                 name: str, 
                 type: InstructionType,
                 params: Dict[str, TypeConstraint],
                 description: Optional[str] = None,
                 cache_config: Optional[Dict[str, Any]] = None,
                 parallel_config: Optional[Dict[str, Any]] = None,
                 side_effects: bool = True,
                 outputs: Optional[List[str]] = None):
        self.id = id
        self.name = name
        self.type = type
        self.params = params
        self.description = description
        # Semantic annotations used by the optimizer:
        # - side_effects: whether execution mutates external state.
        #   Defaults to True (conservative: never optimized away).
        # - outputs: names of values produced by this instruction that
        #   later instructions (or the final result) may consume.
        self.side_effects = side_effects
        self.outputs = list(outputs) if outputs else []
        self.dependencies: List[str] = []
        self.context = InstructionContext(
            state={},
            type_env={},
            preconditions=[],
            postconditions=[],
            cache_config=cache_config,
            parallel_config=parallel_config
        )
        self.cache = InstructionCache(
            max_size=cache_config.get("max_size", 1000) if cache_config else 1000,
            ttl=cache_config.get("ttl", 3600) if cache_config else 3600
        )
        self.metrics = InstructionMetrics()

    def add_precondition(self, condition: Callable[..., bool]):
        """Add a precondition to the instruction."""
        self.context.preconditions.append(condition)

    def add_postcondition(self, condition: Callable[..., bool]):
        """Add a postcondition to the instruction."""
        self.context.postconditions.append(condition)

    def verify_conditions(self, state: Dict[str, Any]) -> bool:
        """Verify pre and post conditions."""
        return all(pre(state) for pre in self.context.preconditions)

    def validate_types(self, input_params: Dict[str, Any]) -> bool:
        """Validate input parameters against type constraints."""
        try:
            for param_name, param_value in input_params.items():
                if param_name in self.params:
                    constraint = self.params[param_name]
                    
                    # Handle null values
                    if param_value is None:
                        if not constraint.nullable:
                            return False
                        continue
                    
                    # Check base type
                    if not isinstance(param_value, constraint.base_type):
                        return False
                        
                    # Check dependent type constraint
                    if constraint.dependent_on and not constraint.dependent_on(param_value):
                        return False
                        
                    # Check additional validation
                    if constraint.validation and not constraint.validation(param_value):
                        return False
                        
                    # Check recursive constraints
                    if constraint.recursive:
                        if isinstance(param_value, (list, tuple)):
                            if not all(isinstance(v, constraint.base_type) for v in param_value):
                                return False
                        elif isinstance(param_value, dict):
                            if not all(isinstance(v, constraint.base_type) for v in param_value.values()):
                                return False
            return True
        except Exception:
            return False

    def _get_cache_key(self, input_params: Dict[str, Any]) -> str:
        """Generate cache key from input parameters."""
        # Sort dictionary to ensure consistent serialization
        serialized = json.dumps(input_params, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()

    async def _execute_parallel(self, tasks: List[Callable]) -> List[Any]:
        """Execute tasks in parallel with configured limits."""
        if not self.context.parallel_config:
            return await asyncio.gather(*[task() for task in tasks])
            
        # Get parallel execution configuration
        batch_size = self.context.parallel_config.get("batch_size", 10)
        
        # Split tasks into batches
        batches = [
            tasks[i:i + batch_size]
            for i in range(0, len(tasks), batch_size)
        ]
        
        # Execute batches
        results = []
        for batch in batches:
            batch_results = await asyncio.gather(*[task() for task in batch])
            results.extend(batch_results)
            
        return results

    def _update_metrics(self, start_time: float, success: bool, cache_hit: bool = False):
        """Update execution metrics."""
        self.metrics.total_executions += 1
        self.metrics.total_execution_time += time.time() - start_time
        
        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
            
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

    @abstractmethod
    async def execute(self, input_params: Dict[str, Any]) -> Any:
        """Execute the instruction."""
        pass

class CompositeInstruction(Instruction):
    """Composite instruction supporting instruction composition."""
    
    def __init__(self,
                 id: str,
                 name: str,
                 components: List[Instruction],
                 composition_type: str = "sequential",
                 cache_config: Optional[Dict[str, Any]] = None,
                 parallel_config: Optional[Dict[str, Any]] = None):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Composite instruction with {composition_type} composition",
            cache_config=cache_config,
            parallel_config=parallel_config
        )
        self.components = components
        self.composition_type = composition_type

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        """Execute composite instruction based on composition type."""
        start_time = time.time()
        
        try:
            # Check cache
            cache_key = self._get_cache_key(input_params)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                self._update_metrics(start_time, True, cache_hit=True)
                return cached_result
            
            if self.composition_type == "sequential":
                result = None
                for component in self.components:
                    result = await component.execute(input_params)
                    
            elif self.composition_type == "parallel":
                tasks = [
                    lambda c=component: c.execute(input_params)
                    for component in self.components
                ]
                results = await self._execute_parallel(tasks)
                result = results[-1] if results else None
                
            else:
                raise ValueError(f"Unsupported composition type: {self.composition_type}")
            
            # Cache result
            self.cache.put(cache_key, result)
            
            self._update_metrics(start_time, True)
            return result
            
        except Exception as e:
            self._update_metrics(start_time, False)
            raise
