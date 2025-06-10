"""Agentic Calling Framework implementation following AgentISA paper Section 3.2."""

from typing import Any, Dict, List, Optional, Callable, Type, Union, Set
from dataclasses import dataclass, field
from inspect import signature, Parameter
import asyncio
import time
import json
import hashlib
from collections import OrderedDict

from .instruction import (
    Instruction,
    TypeConstraint,
    InstructionType,
    InstructionMetrics,
    InstructionCache
)

@dataclass
class FunctionSignature:
    """Function signature representation."""
    name: str
    parameters: Dict[str, Parameter]
    return_type: Type
    docstring: Optional[str]
    async_execution: bool = False

@dataclass
class AgenticContext:
    """Context preservation for agentic calling."""
    state: Dict[str, Any]
    history: List[Dict[str, Any]]
    constraints: Dict[str, Callable[..., bool]]
    cache: Optional[InstructionCache] = None
    metrics: InstructionMetrics = field(default_factory=InstructionMetrics)

@dataclass
class AgenticConfig:
    """Configuration for agentic calling."""
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    enable_parallel: bool = True
    parallel_batch_size: int = 10
    max_parallel_tasks: int = 100

class AgenticTransformer:
    """Transforms function calls into agent instructions."""
    
    def __init__(self, config: Optional[AgenticConfig] = None):
        self.config = config or AgenticConfig()
        self.context = AgenticContext(
            state={},
            history=[],
            constraints={},
            cache=InstructionCache(
                max_size=self.config.cache_size,
                ttl=self.config.cache_ttl
            ) if self.config.enable_caching else None
        )
        self.instruction_cache = {}

    def extract_signature(self, func: Callable) -> FunctionSignature:
        """Extract function signature information."""
        sig = signature(func)
        return FunctionSignature(
            name=func.__name__,
            parameters={name: param for name, param in sig.parameters.items()},
            return_type=sig.return_annotation,
            docstring=func.__doc__,
            async_execution=asyncio.iscoroutinefunction(func)
        )

    def transform_to_instruction(self, 
                               func: Callable,
                               context_requirements: Optional[Dict[str, Any]] = None) -> Instruction:
        """Transform a function into an agent instruction."""
        # Get function signature
        sig = self.extract_signature(func)
        
        # Create parameter type constraints
        param_constraints = {}
        for name, param in sig.parameters.items():
            # Handle Any type correctly
            base_type = (
                param.annotation if param.annotation != Parameter.empty
                else object  # Use object as base type for Any
            )
            param_constraints[name] = TypeConstraint(
                base_type=base_type,
                dependent_on=context_requirements.get(name) if context_requirements else None,
                nullable=param.default is None,
                validation=None  # Add validation if needed
            )

        # Create cache configuration if enabled
        cache_config = {
            "max_size": self.config.cache_size,
            "ttl": self.config.cache_ttl
        } if self.config.enable_caching else None
        
        # Create parallel configuration if enabled
        parallel_config = {
            "batch_size": self.config.parallel_batch_size,
            "max_tasks": self.config.max_parallel_tasks
        } if self.config.enable_parallel else None

        # Create instruction class for this function
        class FunctionInstruction(Instruction):
            def __init__(self, transformer: AgenticTransformer):
                super().__init__(
                    id=f"func_{sig.name}",
                    name=sig.name,
                    type=InstructionType.LLM_INTERACTION,
                    params=param_constraints,
                    description=sig.docstring,
                    cache_config=cache_config,
                    parallel_config=parallel_config
                )
                self.func = func
                self.transformer = transformer
                self.async_execution = sig.async_execution

            async def execute(self, input_params: Dict[str, Any]) -> Any:
                start_time = time.time()
                
                try:
                    # Verify types and conditions
                    if not self.validate_types(input_params):
                        raise TypeError("Invalid parameter types")
                    if not self.verify_conditions(self.transformer.context.state):
                        raise ValueError("Preconditions not met")
                    
                    # Check cache
                    if self.cache:
                        cache_key = self._get_cache_key(input_params)
                        cached_result = self.cache.get(cache_key)
                        if cached_result:
                            self._update_metrics(start_time, True, cache_hit=True)
                            return cached_result
                    
                    # Execute function
                    if self.async_execution:
                        result = await func(**input_params)
                    else:
                        result = func(**input_params)
                    
                    # Cache result
                    if self.cache:
                        self.cache.put(cache_key, result)
                    
                    # Update context
                    self.transformer.context.state.update(input_params)
                    self.transformer.context.history.append({
                        "function": sig.name,
                        "params": input_params,
                        "result": result,
                        "timestamp": time.time()
                    })
                    
                    self._update_metrics(start_time, True)
                    return result
                    
                except Exception as e:
                    self._update_metrics(start_time, False)
                    raise

        return FunctionInstruction(self)

    def add_context_constraint(self, name: str, constraint: Callable[..., bool]):
        """Add a context constraint."""
        self.context.constraints[name] = constraint

    def verify_context_preservation(self, before_state: Dict[str, Any], 
                                  after_state: Dict[str, Any]) -> bool:
        """Verify context preservation between states."""
        for name, constraint in self.context.constraints.items():
            if name in before_state and name in after_state:
                if not constraint(before_state[name], after_state[name]):
                    return False
        return True

class ModelAdapter:
    """Cross-model compatibility adapter."""
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.transformations = {}

    def register_transformation(self, source_type: str, 
                              target_type: str,
                              transform_func: Callable[[Any], Any]):
        """Register a type transformation between models."""
        self.transformations[(source_type, target_type)] = transform_func

    def translate_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Translate instruction to model-specific format."""
        return {
            "name": instruction.name,
            "type": instruction.type.value,
            "parameters": {
                name: self._transform_type(param.base_type)
                for name, param in instruction.params.items()
            },
            "description": instruction.description,
            "metrics": instruction.metrics.__dict__,
            "cache_enabled": instruction.cache is not None,
            "parallel_enabled": instruction.context.parallel_config is not None
        }

    def _transform_type(self, type_: Type) -> str:
        """Transform type to model-specific type."""
        # Add model-specific type transformations
        return str(type_) 