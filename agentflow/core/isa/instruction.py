"""Instruction Set Architecture implementation following AgentISA paper."""

from enum import Enum
from typing import Any, Dict, Optional, List, Callable, TypeVar, Generic, Set, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import asyncio
import threading
import time
import weakref
from collections import OrderedDict

from .instruction_cache import CacheKeyNamespace, build_cache_key
from .sound_cache import MISS, SoundResultCache, cache_policy_opts_in

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

# ---------------------------------------------------------------------------
# Shared write registry for the opt-in sound cache
# ---------------------------------------------------------------------------
#
# A ``SoundResultCache`` is private to one instruction object, so a write
# executed by instruction A cannot reach an entry cached by instruction B.
# H3(cross-instruction invalidation) therefore needs a process-wide fan-out:
# every live sound cache registers itself weakly here, and a successful
# effectful ``execute()`` calls :func:`record_write` with its footprint.
# Weak references keep garbage-collected caches from leaking into the list.

#: Footprint token carried by entries whose read dependencies are unknown.
#: Every named write also evicts it (coarse invalidation) so an unknown
#: footprint can never outlive a write it might have depended on.
COARSE_FOOTPRINT_TOKEN = "__af_coarse__"

_live_sound_caches: List["weakref.ReferenceType[SoundResultCache]"] = []
_live_sound_caches_lock = threading.Lock()


def register_sound_cache(cache: SoundResultCache) -> None:
    """Register ``cache`` for cross-instruction write invalidation."""
    with _live_sound_caches_lock:
        _live_sound_caches[:] = [r for r in _live_sound_caches if r() is not None]
        _live_sound_caches.append(weakref.ref(cache))


def live_sound_caches() -> List[SoundResultCache]:
    """Return the sound caches still alive, pruning dead weak references."""
    with _live_sound_caches_lock:
        live = [
            cache
            for ref in _live_sound_caches
            if (cache := ref()) is not None
        ]
        if len(live) != len(_live_sound_caches):
            _live_sound_caches[:] = [r for r in _live_sound_caches if r() is not None]
        return live


def record_write(entities: Optional[Any] = None) -> int:
    """Fan a completed write out to every live sound cache; return evictions.

    ``entities`` is the write's entity footprint.  ``None`` or an empty
    footprint means the write could not name what it touched: every live
    cache is flushed instead (coarse invalidation -- over-eviction is
    sound, under-eviction is not).  A named write also evicts entries
    admitted with an unknown *read* footprint (they carry
    :data:`COARSE_FOOTPRINT_TOKEN`).
    """
    if entities is None:
        names: Set[str] = set()
    elif isinstance(entities, str):
        names = {entities}
    else:
        names = set(entities)
    total = 0
    for cache in live_sound_caches():
        if names:
            total += cache.record_write(names | {COARSE_FOOTPRINT_TOKEN})
        else:
            total += cache.record_write(coarse=True)
    return total


class InstructionCache:
    """Legacy, unsound result store (TTL + LRU only).

    .. warning::

       This class enforces none of Theorem 3.7's conditions (H1)-(H3).  It
       is kept as the default purely for backwards compatibility: existing
       call sites, tests and benchmark artifacts pin its historical
       hit/miss behaviour, and changing that default would break them.
       That compatibility is not a claim of soundness.

       - **H1 (identity) is violated.**  The default key built by
         ``build_cache_key`` is the sha256 of ``input_params`` alone: no
         instruction id, no tenant.  Two callers with different identity
         but equal parameters share one entry.
       - **H2 (effects) is violated.**  :meth:`put` stores unconditionally
         and never consults ``side_effects``.  An effectful instruction
         memoises its result, so a hit returns success without performing
         the mutation it was supposed to perform.
       - **H3 (staleness) is violated.**  Entries carry no entity
         footprint, there is no reverse index and there is no invalidation
         hook; a read cached before a write to the same entity is served
         until TTL/LRU eviction.  Each instruction also owns a private
         store (``Instruction.__init__``), so cross-instruction
         invalidation is impossible even if a footprint existed.

       Theorem 3.7's conditions are implemented by
       :class:`agentflow.core.isa.sound_cache.SoundResultCache`, which an
       instruction opts into with ``sound_cache = True`` (and, for reads,
       ``side_effects = False``); effectful instructions then never
       memoise and successful writes fan out through
       :func:`record_write`.
    """

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

    #: Opt into :class:`~agentflow.core.isa.sound_cache.SoundResultCache`
    #: instead of the legacy :class:`InstructionCache`.  Defaults to False
    #: so the shipped behaviour (and every test/artifact pinning it) is
    #: unchanged.  Set it on a subclass or instance; a sound read also
    #: needs ``side_effects = False`` and a computable :meth:`footprint`,
    #: plus a tenant (see ``sound_cache.tenant_scope``) unless
    #: ``identity_dependent`` is disabled.
    #:
    #: ``AGENTFLOW_CACHE_POLICY=typed`` flips this default to True for
    #: instructions that declare ``side_effects = False`` (see
    #: ``sound_cache.cache_policy_opts_in``); unset reproduces the
    #: historical default exactly.  The switch supplies the store, not the
    #: declarations: without a tenant (or an identity-independent read)
    #: the typed store refuses admission and every call misses.
    sound_cache: bool = False

    def __init__(self, 
                 id: str,
                 name: str, 
                 type: InstructionType,
                 params: Dict[str, TypeConstraint],
                 description: Optional[str] = None,
                 cache_config: Optional[Dict[str, Any]] = None,
                 parallel_config: Optional[Dict[str, Any]] = None,
                 side_effects: bool = True,
                 outputs: Optional[List[str]] = None,
                 cache_namespace: Optional[CacheKeyNamespace] = None):
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
        # AGENTFLOW_CACHE_POLICY=typed flips the *default* for declared-pure
        # instructions only.  Unset leaves __dict__ untouched (no instance
        # attribute, no behaviour change).  See sound_cache.cache_policy_opts_in.
        if cache_policy_opts_in(self.__class__.sound_cache, self.side_effects):
            self.sound_cache = True
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
        # Opt-in strict cache-key namespace (AgentISA Section 5.2
        # cache-key contract): when attached, cache keys are extended
        # from (instruction, params) to (instruction, params,
        # model_alias, template_version, decoding_hash), so drifting
        # any semantic configuration misses instead of serving a
        # stale entry.  The default None keeps the legacy key
        # unchanged.
        self.cache_namespace = cache_namespace
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
        """Generate cache key from input parameters.

        Without a ``cache_namespace`` (the default) this is the legacy
        params-only key, byte-identical to the historical format.  With
        a strict namespace attached the key folds in the instruction
        identity and the namespace components (see
        ``instruction_cache.build_cache_key``).
        """
        return build_cache_key(self.id, input_params, self.cache_namespace)

    def footprint(self, params: Dict[str, Any]) -> "frozenset[str]":
        """Entity ids this call reads (or, when effectful, writes).

        Empty means the caller cannot compute it, the cache then falls
        back to coarse invalidation: a read is admitted under
        :data:`COARSE_FOOTPRINT_TOKEN` and evicted by *any* successful
        write, and a write with an empty footprint flushes every live
        sound cache.  Returning a wrong non-empty footprint is the one
        thing this hook cannot make sound; it is the caller's (H0)
        declaration, exactly like ``side_effects``.

        The default is empty: an instruction must override this to get
        meaningful (non-coarse) invalidation.
        """
        return frozenset()

    def _sound_caching_enabled(self) -> bool:
        """Whether reads go through the sound cache instead of the legacy one."""
        return bool(self.sound_cache) and not self.side_effects

    def _get_sound_cache(self) -> SoundResultCache:
        """Lazily build (and register) this instruction's sound cache."""
        cache = self.__dict__.get("_sound_cache_store")
        if cache is None:
            config = self.context.cache_config or {}
            cache = SoundResultCache(
                max_size=config.get("max_size", 1000),
                ttl=config.get("ttl", 3600),
            )
            self._sound_cache_store = cache
            register_sound_cache(cache)
        return cache

    def _sound_footprint(self, params: Dict[str, Any]) -> "frozenset[str]":
        """Read footprint for admission; unknown becomes the coarse token."""
        declared = self.footprint(params)
        if not declared:
            return frozenset({COARSE_FOOTPRINT_TOKEN})
        return frozenset(declared)

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

    #: Explicit (same value as the inherited default) so the opt-in is
    #: visible at the class that actually owns the get/put sites below.
    sound_cache: bool = False

    def __init__(self,
                 id: str,
                 name: str,
                 components: List[Instruction],
                 composition_type: str = "sequential",
                 cache_config: Optional[Dict[str, Any]] = None,
                 parallel_config: Optional[Dict[str, Any]] = None,
                 cache_namespace: Optional[CacheKeyNamespace] = None,
                 side_effects: Optional[bool] = None):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Composite instruction with {composition_type} composition",
            cache_config=cache_config,
            parallel_config=parallel_config,
            cache_namespace=cache_namespace,
            # None preserves the historical default (True).  An explicit
            # False lets a composite of pure reads opt into sound caching;
            # it is the caller's declaration, not derived here.
            side_effects=True if side_effects is None else side_effects,
        )
        self.components = components
        self.composition_type = composition_type

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        """Execute composite instruction based on composition type.

        ``sound_cache = True`` swaps the legacy store for
        :class:`SoundResultCache`: pure composites read/write it, while
        effectful composites are never read from or written to any cache.
        Independently, a successful effectful composite records its
        :meth:`footprint` with :func:`record_write` whenever a sound cache
        is live anywhere in the process, so a write by this instruction
        invalidates other instructions' entries (H3).
        """
        start_time = time.time()
        sound_mode = bool(self.sound_cache)
        sound = self._get_sound_cache() if self._sound_caching_enabled() else None

        try:
            # Check cache
            if sound is not None:
                try:
                    cached_result = sound.get(self.id, input_params)
                except (TypeError, ValueError):
                    # Unkeyable params (e.g. non-JSON values): miss, never
                    # let cache infrastructure fail the execution.
                    cached_result = MISS
                if cached_result is not MISS:
                    self._update_metrics(start_time, True, cache_hit=True)
                    return cached_result
            elif not sound_mode:
                cache_key = self._get_cache_key(input_params)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    self._update_metrics(start_time, True, cache_hit=True)
                    return cached_result
            
            if self.composition_type == "sequential":
                # Thread component results: in a sequential composition
                # each component must see the outputs of its
                # predecessors, and the composite yields the *union* of
                # its components' dict outputs (later components may
                # overwrite keys produced earlier -- that is sequential
                # semantics).  A trailing non-dict result follows the
                # last-component-wins convention.  Keeping only the
                # last component's result, as this loop previously
                # did, silently dropped every intermediate output.
                merged = dict(input_params)
                net_new: Dict[str, Any] = {}
                last_value: Any = None
                for component in self.components:
                    result = await component.execute(merged)
                    last_value = result
                    if isinstance(result, dict):
                        net_new.update(result)
                        merged = {**merged, **result}
                        # Thread into the wrapped-state convention
                        # (``{"ctx": {...}``) used by run_stream-style
                        # execution so nested steps see earlier
                        # outputs.
                        inner = merged.get("ctx")
                        if isinstance(inner, dict):
                            merged["ctx"] = {**inner, **result}
                result = (
                    last_value if not isinstance(last_value, dict)
                    else (net_new or None)
                )

            elif self.composition_type == "parallel":
                tasks = [
                    lambda c=component: c.execute(input_params)
                    for component in self.components
                ]
                results = await self._execute_parallel(tasks)
                # Parallel components act on disjoint state components
                # (Definition 3.6, Property 2): merge every component's
                # outputs; keeping only the last dropped the rest.
                merged_result: Dict[str, Any] = {}
                for r in results:
                    if isinstance(r, dict):
                        merged_result.update(r)
                result = merged_result or (results[-1] if results else None)
                
            else:
                raise ValueError(f"Unsupported composition type: {self.composition_type}")
            
            # Cache result
            if sound is not None:
                try:
                    sound.put(
                        self.id,
                        input_params,
                        result,
                        footprint=self._sound_footprint(input_params),
                        side_effects=False,
                    )
                except (TypeError, ValueError):
                    # Unkeyable params: skip admission, keep the result.
                    pass
            elif not sound_mode:
                self.cache.put(cache_key, result)

            # H3 cross-instruction invalidation: a write performed here must
            # reach entries held by *every* other live sound cache, which a
            # per-instruction store cannot do.  Runs only on the success path
            # (a failed write changed nothing) and only once any sound cache
            # exists, so the legacy-only path pays nothing.  An empty
            # footprint flushes all of them (coarse invalidation).
            if self.side_effects and live_sound_caches():
                record_write(self.footprint(input_params))

            self._update_metrics(start_time, True)
            return result
            
        except Exception as e:
            self._update_metrics(start_time, False)
            raise
