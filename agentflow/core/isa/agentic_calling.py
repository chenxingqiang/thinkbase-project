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
            if param.annotation == Parameter.empty:
                raise TypeError(
                    f"Parameter '{name}' of function '{func.__name__}' is "
                    "missing a type annotation; agentic calling requires "
                    "fully annotated signatures"
                )
            base_type = param.annotation
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

    def validate_context(self, context: Dict[str, Any]) -> bool:
        """Pre-execution validation of a single context state.

        Complements :meth:`verify_context_preservation` (which checks a
        *transition*) by checking one state *before* dispatch. Each
        registered constraint whose key is present in ``context`` is
        evaluated: two-argument pairwise predicates are applied
        reflexively -- ``(value, value)`` -- so a state passes only if
        it satisfies its own transition relation; single-argument
        predicates are applied directly. Constraints whose key is
        absent from ``context`` are vacuously satisfied, mirroring the
        shipped preservation semantics (paper Section 3.2.3).
        """
        for name, constraint in self.context.constraints.items():
            if name not in context:
                continue
            value = context[name]
            try:
                params = list(signature(constraint).parameters.values())
                positional = [
                    p for p in params
                    if p.kind in (Parameter.POSITIONAL_ONLY,
                                  Parameter.POSITIONAL_OR_KEYWORD)
                ]
                has_varargs = any(
                    p.kind == Parameter.VAR_POSITIONAL for p in params
                )
            except (TypeError, ValueError):
                return False
            if has_varargs:
                # Arity unknowable; try the pairwise form, fall back to
                # the single-value form.
                try:
                    ok = constraint(value, value)
                except TypeError:
                    ok = constraint(value)
            elif len(positional) >= 2:
                ok = constraint(value, value)
            else:
                ok = constraint(value)
            if not ok:
                return False
        return True

@dataclass
class AgentResponse:
    """Canonical, provider-neutral response envelope (paper Section 3.2.3).

    ``ModelAdapter.normalize_response`` folds provider-specific response
    payloads into this form so downstream instruction code never branches
    on provider syntax. ``raw`` always retains the unmodified payload.
    """
    content: Any = None
    tool_calls: List[Any] = field(default_factory=list)
    usage: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    raw: Any = None

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

    def normalize_response(self, model_response: Any) -> AgentResponse:
        """Normalize a provider-specific response into an AgentResponse.

        Recognizes the two dominant wire shapes -- the Anthropic protocol
        (``content`` blocks with ``usage.input_tokens``/``output_tokens``)
        and the OpenAI protocol (``choices[0].message`` with
        ``usage.prompt_tokens``/``completion_tokens``) -- and passes
        anything else through with ``content`` taken from the first
        common key present (``text``/``output``/``content``/``result``).
        The canonical ``usage`` dict always exposes ``input_tokens`` and
        ``output_tokens`` keys (copied from whichever protocol shape the
        payload carries; absent otherwise). ``raw`` is never modified.
        """
        usage: Dict[str, Any] = {}
        tool_calls: List[Any] = []
        model = ""
        if isinstance(model_response, dict):
            raw_usage = model_response.get("usage") or {}
            if not isinstance(raw_usage, dict):
                raw_usage = {}
            usage = dict(raw_usage)
            if "input_tokens" not in usage and "prompt_tokens" in raw_usage:
                usage["input_tokens"] = raw_usage["prompt_tokens"]
            if "output_tokens" not in usage and "completion_tokens" in raw_usage:
                usage["output_tokens"] = raw_usage["completion_tokens"]
            model = model_response.get("model") or ""

            if "choices" in model_response and isinstance(
                    model_response["choices"], list) and model_response["choices"]:
                # OpenAI-protocol shape
                message = model_response["choices"][0].get("message", {})
                content = message.get("content")
                tool_calls = list(message.get("tool_calls") or [])
            elif isinstance(model_response.get("content"), list):
                # Anthropic-protocol shape: typed content blocks
                blocks = model_response["content"]
                texts = [b.get("text", "") for b in blocks
                         if isinstance(b, dict) and b.get("type") == "text"]
                content = "\n".join(texts) if texts else blocks
                tool_calls = [b for b in blocks
                              if isinstance(b, dict)
                              and b.get("type") == "tool_use"]
            else:
                for key in ("text", "output", "content", "result"):
                    if key in model_response:
                        content = model_response[key]
                        break
                else:
                    content = model_response
        else:
            content = model_response
        return AgentResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            model=model or self.model_type,
            raw=model_response,
        )


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
        """Transform type to model-specific type name.

        Renders builtin/simple types as bare names (``int``, ``str``),
        typing generics with their arguments (``Dict[str, Any]``) and
        user-defined types by their class name (``CustomType``).
        """
        if getattr(type_, "__origin__", None) is not None:
            # typing generic alias, e.g. Dict[str, Any] / List[int]
            args = ", ".join(self._transform_type(a) for a in type_.__args__)
            name = (
                getattr(type_, "__name__", None)
                or getattr(type_.__origin__, "__name__", str(type_.__origin__))
            )
            return f"{name}[{args}]"
        if isinstance(type_, type):
            return type_.__name__
        name = getattr(type_, "__name__", None)
        if name:
            return name
        return str(type_) 