"""Unit tests for strict cache-key namespacing (I5-cache-namespacing).

The AgentISA cache-key contract (paper Section 5.2) keys a cached
entry on the (instruction, params) pair and requires the deployment
to hold model alias, prompt-template version, and decoding
configuration fixed.  When any of those can drift, the deployment
must namespace the cache: attaching a ``CacheKeyNamespace`` opts an
instruction into the strict key
(instruction, params, model_alias, template_version, decoding_hash),
so drifting any component misses the old key and starts hitting
under the new one.  Without a namespace attached, the legacy key --
and therefore every historical hit/miss decision -- is unchanged.
"""

import hashlib
import json
from typing import Any, Dict

import pytest

from agentflow.core.isa.agentic_calling import AgenticTransformer
from agentflow.core.isa.instruction import (
    CompositeInstruction,
    Instruction,
    InstructionType,
)
from agentflow.core.isa.instruction_cache import (
    CacheKeyNamespace,
    build_cache_key,
)

DECODING = {"temperature": 0.0, "max_tokens": 64}


class _Counting(Instruction):
    """Pure instruction that counts its executions."""

    def __init__(self, id: str = "counting"):
        super().__init__(
            id=id,
            name=id,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description="counts executions",
            side_effects=False,
        )
        self.executions = 0

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        self.executions += 1
        return {"executions": self.executions}


def _namespaced_composite(namespace, inner=None):
    """Composite wrapping a counting instruction under ``namespace``."""
    inner = inner if inner is not None else _Counting()
    composite = CompositeInstruction(
        id="comp",
        name="comp",
        components=[inner],
        cache_namespace=namespace,
        cache_config={"max_size": 10, "ttl": 60},
    )
    return composite, inner


# --------------------------------------------------------------------------
# Default (opt-out) behavior must be unchanged
# --------------------------------------------------------------------------


def test_legacy_key_format_unchanged_without_namespace():
    """Without a namespace the key stays the historical params-only sha256."""
    instruction = _Counting()
    params = {"b": 2, "a": {"y": 1, "x": 1}}
    expected = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()
    assert instruction._get_cache_key(params) == expected


@pytest.mark.asyncio
async def test_legacy_hits_still_work_without_namespace():
    """The default cache-hit behavior is untouched by the opt-in."""
    composite, inner = _namespaced_composite(None)

    assert await composite.execute({"x": 1}) == {"executions": 1}
    assert await composite.execute({"x": 1}) == {"executions": 1}

    assert inner.executions == 1  # second call served from cache
    assert composite.metrics.cache_hits == 1


# --------------------------------------------------------------------------
# The strict key folds in every namespace component
# --------------------------------------------------------------------------


def test_attached_namespace_switches_to_strict_key():
    """Attaching a namespace (even an empty one) opts into the strict key."""
    params = {"x": 1}
    plain = _Counting()
    namespaced = _Counting()
    namespaced.cache_namespace = CacheKeyNamespace(
        model_alias="alias-a", template_version="v1", decoding=DECODING
    )
    assert plain._get_cache_key(params) != namespaced._get_cache_key(params)


def test_strict_key_includes_instruction_identity():
    """The ``instruction`` component separates same-params entries."""
    namespace = CacheKeyNamespace(model_alias="m", template_version="v1")
    params = {"x": 1}
    key_a = build_cache_key("instr-a", params, namespace)
    key_b = build_cache_key("instr-b", params, namespace)
    assert key_a != key_b
    assert key_a == build_cache_key("instr-a", params, namespace)


def test_decoding_hash_is_order_insensitive_but_value_sensitive():
    """Canonical serialization: dict order never drifts the key, values do."""
    original = CacheKeyNamespace(decoding=DECODING)
    reordered = CacheKeyNamespace(
        decoding={"max_tokens": 64, "temperature": 0.0}
    )
    assert original.decoding_hash == reordered.decoding_hash

    drifted = CacheKeyNamespace(
        decoding={"temperature": 0.7, "max_tokens": 64}
    )
    assert original.decoding_hash != drifted.decoding_hash


# --------------------------------------------------------------------------
# Drifting any component invalidates the old key; the new key takes effect
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_model_alias_drift_invalidates_cached_entry():
    """Changing the model alias must miss the old key and hit the new one."""
    namespace = CacheKeyNamespace(
        model_alias="alias-a", template_version="v1", decoding=DECODING
    )
    composite, inner = _namespaced_composite(namespace)

    # First execution misses and is cached under the alias-a key.
    assert await composite.execute({"x": 1}) == {"executions": 1}
    assert await composite.execute({"x": 1}) == {"executions": 1}
    assert inner.executions == 1
    assert composite.metrics.cache_misses == 1
    assert composite.metrics.cache_hits == 1

    # Semantic drift: the alias-b lookup must miss the alias-a entry.
    namespace.model_alias = "alias-b"
    assert await composite.execute({"x": 1}) == {"executions": 2}
    assert await composite.execute({"x": 1}) == {"executions": 2}
    assert inner.executions == 2
    assert composite.metrics.cache_misses == 2
    assert composite.metrics.cache_hits == 2

    # Namespacing isolates rather than evicts: back in the alias-a
    # namespace, that namespace's own entry is still served.
    namespace.model_alias = "alias-a"
    assert await composite.execute({"x": 1}) == {"executions": 1}
    assert inner.executions == 2


@pytest.mark.asyncio
async def test_template_version_drift_invalidates_cached_entry():
    """Changing the prompt-template version must invalidate the old key."""
    namespace = CacheKeyNamespace(
        model_alias="alias-a", template_version="v1", decoding=DECODING
    )
    composite, inner = _namespaced_composite(namespace)

    await composite.execute({"x": 1})
    await composite.execute({"x": 1})
    assert inner.executions == 1

    namespace.template_version = "v2"
    await composite.execute({"x": 1})
    assert inner.executions == 2
    await composite.execute({"x": 1})
    assert inner.executions == 2  # new key now hits


@pytest.mark.asyncio
async def test_decoding_drift_invalidates_cached_entry():
    """Changing the decoding configuration must invalidate the old key."""
    namespace = CacheKeyNamespace(
        model_alias="alias-a", template_version="v1", decoding=DECODING
    )
    composite, inner = _namespaced_composite(namespace)

    await composite.execute({"x": 1})
    await composite.execute({"x": 1})
    assert inner.executions == 1

    namespace.decoding = {"temperature": 0.7, "max_tokens": 64}
    await composite.execute({"x": 1})
    assert inner.executions == 2
    await composite.execute({"x": 1})
    assert inner.executions == 2  # new key now hits


@pytest.mark.asyncio
async def test_instruction_identity_isolates_entries_on_a_shared_cache():
    """Two instructions sharing one cache must not serve each other's entries.

    The strict key's ``instruction`` component is what makes a shared
    ``InstructionCache`` safe; a params-only key would collide across
    instructions and serve the wrong entry.
    """
    namespace = CacheKeyNamespace(model_alias="m", template_version="v1")
    inner_a = _Counting("inner-a")
    inner_b = _Counting("inner-b")
    comp_a = CompositeInstruction(
        id="comp-a", name="comp-a", components=[inner_a],
        cache_namespace=namespace,
    )
    comp_b = CompositeInstruction(
        id="comp-b", name="comp-b", components=[inner_b],
        cache_namespace=namespace,
    )
    comp_b.cache = comp_a.cache  # deployments may share one store

    await comp_a.execute({"x": 1})
    await comp_b.execute({"x": 1})  # must miss: different instruction

    assert inner_a.executions == 1
    assert inner_b.executions == 1


# --------------------------------------------------------------------------
# Opt-in is available to existing instruction subclasses
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_agentic_instruction_opts_in_via_attribute():
    """Existing agentic-calling instructions opt in by assigning the namespace."""
    calls = {"n": 0}

    def add(a: int, b: int) -> int:
        calls["n"] += 1
        return a + b

    instruction = AgenticTransformer().transform_to_instruction(add)
    instruction.cache_namespace = CacheKeyNamespace(model_alias="alias-a")

    assert await instruction.execute({"a": 1, "b": 2}) == 3
    assert await instruction.execute({"a": 1, "b": 2}) == 3
    assert calls["n"] == 1  # second call was a cache hit

    instruction.cache_namespace = CacheKeyNamespace(model_alias="alias-b")
    assert await instruction.execute({"a": 1, "b": 2}) == 3
    assert calls["n"] == 2  # alias drifted: miss, function re-executed
