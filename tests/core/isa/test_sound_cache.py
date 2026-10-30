"""Tests for the opt-in sound cache (Theorem 3.7, hypotheses H1-H3).

Every case here pins a condition the shipped ``InstructionCache`` does not enforce:
writes are never admitted, identity-dependent keys carry the tenant, writes evict
exactly the intersecting read footprints (or everything when the footprint is
unknown), coarse/TTL/LRU eviction only ever removes entries, and per-thread tenant
scopes cannot cross-serve.
"""

import threading
import time

import pytest

from agentflow.core.isa.sound_cache import (
    MISS,
    SoundResultCache,
    current_tenant,
    tenant_scope,
)


def _put_read(cache, instruction_id, value, footprint=frozenset()):
    """Admit an identity-independent, declared-pure read."""
    return cache.put(instruction_id, {"q": 1}, value, footprint=footprint,
                     side_effects=False, identity_dependent=False)


def _get_read(cache, instruction_id):
    return cache.get(instruction_id, {"q": 1}, identity_dependent=False)


def test_write_is_never_cached():
    cache = SoundResultCache()
    for _ in range(2):
        # The write's body "runs"; its response must never become an entry.
        assert cache.put("transfer", {"amount": 5}, "ok",
                         footprint={"acct:1"}, side_effects=True) is False
        assert cache.get("transfer", {"amount": 5}) is MISS
    stats = cache.stats()
    assert stats["refusals_by_reason"]["effectful"] == 2
    assert stats["entries"] == 0
    assert stats["misses"] == 2


def test_refusals_are_observable_not_silent():
    cache = SoundResultCache()

    def put(**kwargs):
        return cache.put("x", {"n": 1}, "v", **kwargs)

    # Purity must be declared explicitly: the default is fail-closed.
    assert put(identity_dependent=False) is False
    assert put(side_effects=True, identity_dependent=False) is False
    # An identity-dependent read without a tenant cannot be keyed.
    assert put(side_effects=False) is False
    # "Footprint unknown" is not "footprint empty".
    assert put(side_effects=False, identity_dependent=False,
               footprint=None) is False
    # An empty footprint is a legitimate constant read.
    assert put(side_effects=False, identity_dependent=False) is True

    stats = cache.stats()
    assert stats["refusals_by_reason"] == {
        "effectful": 2, "no_tenant": 1, "footprint_unknown": 1,
    }
    assert stats["admissions"] == 1
    assert stats["entries"] == 1
    assert "never soundness" in cache.soundness_note


def test_identity_dependent_keys_isolate_tenants():
    cache = SoundResultCache()
    with tenant_scope("tenant-a"):
        assert cache.put("profile", {"u": 1}, "A", side_effects=False) is True
    with tenant_scope("tenant-b"):
        assert cache.get("profile", {"u": 1}) is MISS
        assert cache.put("profile", {"u": 1}, "B", side_effects=False) is True
        assert cache.get("profile", {"u": 1}) == "B"
        # An explicit tenant wins over the context tenant.
        assert cache.get("profile", {"u": 1}, tenant="tenant-a") == "A"
    with tenant_scope("tenant-a"):
        assert cache.get("profile", {"u": 1}) == "A"
        # The instruction id is part of the key.
        assert cache.get("profile-other", {"u": 1}) is MISS
        # Params are canonicalised: dict key order does not matter.
        cache.put("profile", {"a": 1, "b": 2}, "AB", footprint={"cfg:1"},
                  side_effects=False)
        assert cache.get("profile", {"b": 2, "a": 1}) == "AB"


def test_identity_independent_entries_share_across_tenants():
    cache = SoundResultCache()
    with tenant_scope("tenant-a"):
        assert cache.put("lookup", {"q": 1}, "shared", footprint="cfg:1",
                         side_effects=False,
                         identity_dependent=False) is True
    with tenant_scope("tenant-b"):
        assert cache.get("lookup", {"q": 1},
                         identity_dependent=False) == "shared"
        # An identity-dependent read must not resolve to the shared entry.
        assert cache.get("lookup", {"q": 1}) is MISS
    assert cache.stats()["entries"] == 1


def test_write_evicts_intersecting_footprints_only():
    cache = SoundResultCache()
    _put_read(cache, "order-1", "a", footprint={"order:1"})
    _put_read(cache, "order-2", "b", footprint={"order:2"})
    _put_read(cache, "mixed", "c", footprint={"order:1", "user:9"})

    # A write naming one entity evicts every entry depending on it.
    assert cache.record_write({"user:9"}) == 1
    assert _get_read(cache, "mixed") is MISS
    assert _get_read(cache, "order-1") == "a"

    # A bare string is one entity, not an iterable of characters.
    assert cache.record_write("order:1") == 1
    assert _get_read(cache, "order-1") is MISS
    assert _get_read(cache, "order-2") == "b"

    stats = cache.stats()
    assert stats["evictions_by_reason"]["write"] == 2
    assert stats["entries"] == 1


def test_coarse_invalidation_evicts_everything():
    cache = SoundResultCache()
    for i in range(3):
        _put_read(cache, f"r{i}", i, footprint={f"e:{i}"})
    assert cache.record_write(None) == 3
    assert cache.stats()["entries"] == 0
    assert cache.stats()["coarse_invalidations"] == 1

    _put_read(cache, "r", 1, footprint={"e:1"})
    assert cache.record_write({"e:1"}, coarse=True) == 1
    # A write naming no entity cannot be narrowed: flush everything.
    _put_read(cache, "r", 1, footprint={"e:1"})
    assert cache.record_write(set()) == 1

    stats = cache.stats()
    assert stats["coarse_invalidations"] == 3
    assert stats["evictions_by_reason"]["write"] == 5
    assert stats["entries"] == 0


def test_lru_and_ttl_evict_without_resurrecting():
    cache = SoundResultCache(max_size=2, ttl=60.0)
    _put_read(cache, "a", "old-a", footprint={"e:a"})
    _put_read(cache, "b", "b", footprint={"e:b"})
    assert _get_read(cache, "a") == "old-a"   # a is now most recently used
    _put_read(cache, "c", "c", footprint={"e:c"})
    assert _get_read(cache, "b") is MISS      # LRU evicted, never resurrected
    assert _get_read(cache, "a") == "old-a"
    assert cache.stats()["evictions_by_reason"]["capacity"] == 1
    _put_read(cache, "b", "new-b", footprint={"e:b"})
    assert _get_read(cache, "b") == "new-b"

    ttl_cache = SoundResultCache(ttl=0.05)
    _put_read(ttl_cache, "k", "stale", footprint={"e:k"})
    time.sleep(0.08)
    assert _get_read(ttl_cache, "k") is MISS
    stats = ttl_cache.stats()
    assert stats["evictions_by_reason"]["ttl"] == 1
    assert stats["entries"] == 0
    _put_read(ttl_cache, "k", "fresh", footprint={"e:k"})
    assert _get_read(ttl_cache, "k") == "fresh"


def test_falsy_results_are_hits_not_misses():
    cache = SoundResultCache()
    for i, value in enumerate([0, "", [], {}, False, None]):
        _put_read(cache, f"falsy-{i}", value, footprint={"e:x"})
        assert _get_read(cache, f"falsy-{i}") == value
    assert cache.stats()["hits"] == 6
    assert cache.stats()["misses"] == 0


def test_tenant_scope_restores_previous_tenant():
    assert current_tenant.get() is None
    with tenant_scope("outer"):
        assert current_tenant.get() == "outer"
        with tenant_scope("inner"):
            assert current_tenant.get() == "inner"
        assert current_tenant.get() == "outer"
    assert current_tenant.get() is None

    with pytest.raises(RuntimeError):
        with tenant_scope("boom"):
            raise RuntimeError("boom")
    assert current_tenant.get() is None


def test_concurrent_tenants_never_cross_serve():
    cache = SoundResultCache(max_size=256, ttl=60.0)
    thread_count, rounds = 20, 40
    barrier = threading.Barrier(thread_count)
    failures = []

    def worker(index):
        tenant = f"tenant-{index}"
        try:
            with tenant_scope(tenant):
                barrier.wait(timeout=5)
                for round_no in range(rounds):
                    admitted = cache.put(
                        "profile", {"u": 1}, (tenant, round_no),
                        footprint=f"user:{index}", side_effects=False,
                    )
                    if not admitted:
                        failures.append((tenant, "put refused"))
                    seen = cache.get("profile", {"u": 1})
                    if seen is MISS:
                        failures.append((tenant, "miss"))
                    elif seen[0] != tenant:
                        failures.append((tenant, seen))
        except BaseException as exc:  # pragma: no cover - failure path
            failures.append((tenant, repr(exc)))

    threads = [threading.Thread(target=worker, args=(i,))
               for i in range(thread_count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert failures == []
    stats = cache.stats()
    assert stats["entries"] == thread_count
    assert stats["hits"] == thread_count * rounds
    assert stats["misses"] == 0
