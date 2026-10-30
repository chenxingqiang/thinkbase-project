"""Opt-in result cache enforcing Theorem 3.7's conditions (H1)-(H3).

Additive and self-contained: ``instruction.py``'s ``InstructionCache`` is not
modified, wrapped or re-exported.  That shipped store still keys on parameters
alone, admits effectful instructions and evicts only by TTL/LRU, so it enforces
none of (H1)-(H3) and remains unsound; nothing on the shipped ``execute()`` path
calls this class yet, and importing this module does not make that path sound.

(H1) Identity: the key is a function of (instruction id, canonical params,
tenant).  The tenant comes from ``tenant=`` or, failing that, the module-level
:data:`current_tenant` ContextVar set by :func:`tenant_scope`; a put without one
is refused (``no_tenant``).  ``identity_dependent=False`` opts into cross-tenant
sharing for genuinely identity-independent reads.

(H2) Effect gate: ``put`` requires ``side_effects=False``; anything else is
refused and counted (``effectful``).  Purity is the caller's declaration; the
class cannot verify (H0), read determinism.

(H3) Footprint: ``put(..., footprint={entity ids})`` records the read's
dependencies; ``record_write(entities)`` evicts every intersecting entry.
``footprint=None`` (unknown) is refused (``footprint_unknown``), never treated
as empty, and a write that cannot name its entities (None, empty, or
``coarse=True``) flushes everything -- over-eviction is sound, under is not.

LRU capacity and TTL are backstops: they cost hits, never soundness.  State is
lock-guarded; the tenant ContextVar is per-context, so concurrent threads with
different tenants cannot observe each other's entries.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Optional

REFUSAL_EFFECTFUL = "effectful"
REFUSAL_NO_TENANT = "no_tenant"
REFUSAL_FOOTPRINT_UNKNOWN = "footprint_unknown"
EVICTION_WRITE = "write"
EVICTION_CAPACITY = "capacity"
EVICTION_TTL = "ttl"

current_tenant: ContextVar[Optional[str]] = ContextVar(
    "af_sound_cache_tenant", default=None
)

SOUNDNESS_NOTE = (
    "Sound only under the caller's declarations: (H0) a cached value must be "
    "determined by the entities in its footprint and repeatable for the same key; "
    "put(side_effects=False) must be truthful; record_write must run after every "
    "write that lands, naming every touched entity (coarse=True when it cannot). "
    "LRU capacity and TTL are backstops that cost hits, never soundness. This class "
    "is opt-in: the shipped InstructionCache in instruction.py still enforces none "
    "of H1-H3 and stays unsound."
)


class _Miss:
    def __repr__(self) -> str: return "MISS"


MISS = _Miss()


@contextmanager
def tenant_scope(tenant: Optional[str]):
    """Set :data:`current_tenant` for the block, restoring the previous value."""
    token = current_tenant.set(tenant)
    try:
        yield tenant
    finally:
        current_tenant.reset(token)


def _tokens(entities: Iterable[str]) -> FrozenSet[str]:
    # A bare string is one entity id, not an iterable of characters: treating the
    # two sides differently would under-evict (footprint token vs. char tokens).
    if isinstance(entities, str):
        return frozenset((entities,))
    return frozenset(entities)


def _cache_key(instruction_id: str, params: Any, tenant: Optional[str],
               identity_dependent: bool) -> str:
    scope: Any = ({"kind": "tenant", "tenant": tenant}
                  if identity_dependent else "shared")
    payload = {"instruction": instruction_id, "params": params, "scope": scope}
    # sort_keys canonicalises dict order; unsupported values raise TypeError
    # (fail closed) instead of hashing an unstable repr().
    encoded = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


@dataclass(frozen=True)
class _Entry:
    value: Any
    footprint: FrozenSet[str]
    tenant: Optional[str]
    identity_dependent: bool
    stored_at: float


class SoundResultCache:
    """A result cache whose default behaviour enforces (H1)-(H3).

    Example::

        cache = SoundResultCache(max_size=1024, ttl=3600)
        with tenant_scope("acme"):
            cache.put("profile", {"user": "u1"}, profile,
                      footprint={"user:u1"}, side_effects=False)
            cache.get("profile", {"user": "u1"})            # -> profile
            cache.record_write({"user:u1"})                 # evicts it
            cache.get("profile", {"user": "u1"}) is MISS    # -> True

    Sound only relative to the caller's declarations; see :data:`SOUNDNESS_NOTE`.
    """

    soundness_note = SOUNDNESS_NOTE

    def __init__(self, max_size: Optional[int] = 1000,
                 ttl: Optional[float] = 3600.0):
        if max_size is not None and max_size <= 0:
            raise ValueError("max_size must be positive or None")
        if ttl is not None and ttl < 0:
            raise ValueError("ttl must be non-negative or None")
        self.max_size = max_size
        self.ttl = ttl
        self._entries: "OrderedDict[str, _Entry]" = OrderedDict()
        self._index: Dict[str, set] = {}
        self._lock = threading.RLock()
        self._hits = self._misses = self._admissions = 0
        self._coarse_invalidations = 0
        self._refusals = {r: 0 for r in (REFUSAL_EFFECTFUL, REFUSAL_NO_TENANT,
                                         REFUSAL_FOOTPRINT_UNKNOWN)}
        self._evictions = {e: 0 for e in (EVICTION_WRITE, EVICTION_CAPACITY,
                                          EVICTION_TTL)}

    def get(self, instruction_id: str, params: Any, *,
            identity_dependent: bool = True,
            tenant: Optional[str] = None) -> Any:
        """Return the cached value, or :data:`MISS`; a hit refreshes LRU order."""
        tenant = current_tenant.get() if tenant is None else tenant
        key = _cache_key(instruction_id, params, tenant, identity_dependent)
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self._misses += 1
                return MISS
            if entry.identity_dependent != identity_dependent or (
                entry.identity_dependent and entry.tenant != tenant
            ):
                # Defensive: keys already separate scopes; never cross-serve.
                self._misses += 1
                return MISS
            if self.ttl is not None and now - entry.stored_at > self.ttl:
                self._drop(key, EVICTION_TTL)
                self._misses += 1
                return MISS
            self._entries.move_to_end(key)
            self._hits += 1
            return entry.value

    def put(self, instruction_id: str, params: Any, value: Any,
            footprint: Optional[Iterable[str]] = frozenset(), *,
            side_effects: bool = True, identity_dependent: bool = True,
            tenant: Optional[str] = None) -> bool:
        """Admit an entry for a pure read; return whether it was admitted.

        Refusals are recorded, never silent: ``effectful`` (``side_effects`` not
        explicitly ``False``; the default is fail-closed), ``no_tenant`` (an
        identity-dependent read with no tenant), ``footprint_unknown``
        (``footprint is None`` -- unknown is not empty).  An empty footprint is a
        legitimate constant read.
        """
        tenant = current_tenant.get() if tenant is None else tenant
        key = _cache_key(instruction_id, params, tenant, identity_dependent)
        with self._lock:
            if side_effects:
                self._refusals[REFUSAL_EFFECTFUL] += 1
                return False
            if identity_dependent and tenant is None:
                self._refusals[REFUSAL_NO_TENANT] += 1
                return False
            if footprint is None:
                self._refusals[REFUSAL_FOOTPRINT_UNKNOWN] += 1
                return False
            self._admit(key, _Entry(value, _tokens(footprint), tenant,
                                    identity_dependent, time.monotonic()))
        return True

    def record_write(self, entities: Optional[Iterable[str]] = None, *,
                     coarse: bool = False) -> int:
        """Evict entries whose footprint intersects ``entities``; return the count.

        ``entities=None``, an empty iterable, or ``coarse=True`` means the write
        cannot be named: evict everything (coarse invalidation, counted apart).
        Call this only after the write has returned successfully.
        """
        with self._lock:
            names = frozenset() if entities is None else _tokens(entities)
            if coarse or not names:
                self._coarse_invalidations += 1
                keys = list(self._entries)
            else:
                keys = set()
                for token in names:
                    keys.update(self._index.get(token, ()))
            for key in keys:
                self._drop(key, EVICTION_WRITE)
            return len(keys)

    def clear(self) -> None:
        """Drop every entry (not an eviction; counters are kept)."""
        with self._lock:
            self._entries.clear()
            self._index.clear()

    def stats(self) -> Dict[str, Any]:
        """Snapshot of counters, entry count and the soundness note."""
        with self._lock:
            return dict(hits=self._hits, misses=self._misses,
                        admissions=self._admissions,
                        entries=len(self._entries),
                        refusals_by_reason=dict(self._refusals),
                        evictions_by_reason=dict(self._evictions),
                        coarse_invalidations=self._coarse_invalidations,
                        soundness_note=SOUNDNESS_NOTE)

    def _admit(self, key: str, entry: _Entry) -> None:
        self._drop(key)
        while self.max_size is not None and len(self._entries) >= self.max_size:
            self._drop(next(iter(self._entries)), EVICTION_CAPACITY)
        self._entries[key] = entry
        for token in entry.footprint:
            self._index.setdefault(token, set()).add(key)
        self._admissions += 1

    def _drop(self, key: str, reason: Optional[str] = None) -> bool:
        entry = self._entries.pop(key, None)
        if entry is None:
            return False
        for token in entry.footprint:
            bucket = self._index.get(token)
            if bucket is not None:
                bucket.discard(key)
                if not bucket:
                    self._index.pop(token, None)
        if reason is not None:
            self._evictions[reason] += 1
        return True
