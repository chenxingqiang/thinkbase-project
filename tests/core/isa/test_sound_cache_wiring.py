"""Wiring tests for the opt-in sound cache in ``instruction.py``.

The shipped default stays the legacy, unsound ``InstructionCache`` (existing
tests and benchmark artifacts pin its hit/miss behaviour), so these tests pin
both directions:

- the legacy default still caches and serves, and is structurally untouched;
- ``sound_cache = True`` routes reads through ``SoundResultCache``: a pure
  instruction caches, an effectful one never does, a write by one instruction
  evicts another instruction's entries through the shared registry, and an
  unknown read footprint is coarsely invalidated.

No existing test is modified.
"""

import hashlib
import json
from typing import Any, Dict

import pytest

from agentflow.core.isa.instruction import (
    CompositeInstruction,
    Instruction,
    InstructionCache,
    InstructionType,
    live_sound_caches,
)
from agentflow.core.isa.sound_cache import SoundResultCache, tenant_scope


class _CountingLeaf(Instruction):
    """Pure leaf that counts its own executions."""

    def __init__(self, id: str = "leaf"):
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
        return {"n": self.executions}


class _SoundRead(CompositeInstruction):
    """Composition of reads that opts into the sound cache."""

    sound_cache = True

    def __init__(self, id: str, inner: Instruction, entities=()):
        super().__init__(
            id=id,
            name=id,
            components=[inner],
            cache_config={"max_size": 10, "ttl": 60},
            side_effects=False,
        )
        self._entities = frozenset(entities)

    def footprint(self, params: Dict[str, Any]) -> frozenset:
        return self._entities


class _SoundWrite(CompositeInstruction):
    """Composition with a declared write that opts into the sound cache."""

    sound_cache = True

    def __init__(self, id: str, inner: Instruction, entities=()):
        super().__init__(
            id=id,
            name=id,
            components=[inner],
            cache_config={"max_size": 10, "ttl": 60},
            side_effects=True,
        )
        self._entities = frozenset(entities)

    def footprint(self, params: Dict[str, Any]) -> frozenset:
        return self._entities


@pytest.mark.asyncio
async def test_legacy_default_still_caches_and_serves():
    """The default (``sound_cache = False``) path is unchanged."""
    inner = _CountingLeaf("legacy-leaf")
    composite = CompositeInstruction(
        id="legacy-composite",
        name="legacy-composite",
        components=[inner],
        cache_config={"max_size": 10, "ttl": 60},
    )
    assert composite.sound_cache is False
    assert isinstance(composite.cache, InstructionCache)

    first = await composite.execute({"x": 1})
    second = await composite.execute({"x": 1})

    assert first == second == {"n": 1}
    assert inner.executions == 1
    assert composite.metrics.cache_hits == 1


@pytest.mark.asyncio
async def test_sound_pure_instruction_caches():
    """``sound_cache = True`` + ``side_effects = False`` admits a sound entry."""
    inner = _CountingLeaf("sound-read-leaf")
    composite = _SoundRead("sound-read", inner, {"doc:1"})

    with tenant_scope("tenant-sound-read"):
        first = await composite.execute({"q": 1})
        second = await composite.execute({"q": 1})

    assert first == second == {"n": 1}
    assert inner.executions == 1
    assert composite.metrics.cache_hits == 1
    store = composite._get_sound_cache()
    assert isinstance(store, SoundResultCache)
    assert store.stats()["admissions"] == 1
    assert store.stats()["entries"] == 1


@pytest.mark.asyncio
async def test_sound_effectful_instruction_never_caches():
    """A declared write is never read from or written to any cache."""
    inner = _CountingLeaf("sound-write-leaf")
    composite = _SoundWrite("sound-write", inner, {"doc:1"})

    with tenant_scope("tenant-sound-write"):
        first = await composite.execute({"q": 1})
        second = await composite.execute({"q": 1})

    assert first == {"n": 1}
    assert second == {"n": 2}
    assert inner.executions == 2
    assert composite.metrics.cache_hits == 0
    # Neither store admitted anything.
    assert composite.cache.cache == {}
    assert "_sound_cache_store" not in composite.__dict__


@pytest.mark.asyncio
async def test_write_through_another_instruction_evicts_sound_entry():
    """A write by B evicts A's cached read through the shared registry."""
    reader_inner = _CountingLeaf("evict-read-leaf")
    reader = _SoundRead("evict-read", reader_inner, {"order:1"})
    writer_inner = _CountingLeaf("evict-write-leaf")
    writer = _SoundWrite("evict-write", writer_inner, {"order:1"})

    with tenant_scope("tenant-evict"):
        assert await reader.execute({"id": 1}) == {"n": 1}
        assert await reader.execute({"id": 1}) == {"n": 1}
        assert reader_inner.executions == 1

        # A different instruction performs the write; its cache is never
        # created (effectful), yet the registry reaches the reader's entry.
        assert any(c is reader._get_sound_cache() for c in live_sound_caches())
        await writer.execute({"id": 1})

        assert await reader.execute({"id": 1}) == {"n": 2}

    assert reader_inner.executions == 2
    stats = reader._get_sound_cache().stats()
    assert stats["evictions_by_reason"]["write"] >= 1


@pytest.mark.asyncio
async def test_unknown_read_footprint_is_coarsely_invalidated():
    """An empty footprint is "unknown", evicted by any named write."""
    inner = _CountingLeaf("coarse-read-leaf")
    # entities=() -> footprint() returns frozenset(): unknown, not constant.
    reader = _SoundRead("coarse-read", inner, ())
    writer_inner = _CountingLeaf("coarse-write-leaf")
    writer = _SoundWrite("coarse-write", writer_inner, {"order:9"})

    with tenant_scope("tenant-coarse"):
        assert await reader.execute({"k": 1}) == {"n": 1}
        assert await reader.execute({"k": 1}) == {"n": 1}
        assert inner.executions == 1

        await writer.execute({"k": 1})  # names a *disjoint* entity
        assert await reader.execute({"k": 1}) == {"n": 2}

    assert reader._get_sound_cache().stats()["coarse_invalidations"] == 0
    assert reader._get_sound_cache().stats()["evictions_by_reason"]["write"] >= 1


def test_legacy_path_is_untouched():
    """Default instructions keep the legacy store and key; no sound cache."""
    instruction = _CountingLeaf("legacy-untouched")
    assert instruction.sound_cache is False
    assert isinstance(instruction.cache, InstructionCache)

    params = {"b": 2, "a": {"y": 1, "x": 1}}
    expected = hashlib.sha256(
        json.dumps(params, sort_keys=True).encode()
    ).hexdigest()
    assert instruction._get_cache_key(params) == expected
    # The footprint hook exists but is inert on the legacy path.
    assert instruction.footprint(params) == frozenset()
    assert "_sound_cache_store" not in instruction.__dict__
