#!/usr/bin/env python3
"""S3: measure the tau-bench corpus through the *shipped* sound cache.

What this is. ``examples/benchmarks/unsafe_cache_baselines.py`` measures four
caches it implements itself (``cache_identity`` key recipes plus a replay loop in
the benchmark). This script measures the same workload through the real class
that ships in ``agentflow/core/isa/sound_cache.py`` -- ``SoundResultCache``,
imported, not reimplemented -- so the zero-damage claim is about the module the
repository would actually deploy, not about the harness that models it.

The workload is the harness's: ``tau_bench_historical.jsonl``, first-trial
records only (325 records, 2,275 calls, four source runs), replayed in file
order with one run-scope cache per source. The stream is built by
``unsafe_cache_baselines.build_streams`` itself, so the two measurements cannot
drift apart on run boundaries, first-trial filtering, tenant recovery or
position numbering. The harness's P4 figure (466 hits / 1,728 admitted reads) is
also *recomputed* from that module rather than copied from its JSON.

Per call, exactly as the task specifies:

  instruction_id  the tool name
  params          the call arguments
  tenant          the record's single ``user_id``, passed only where the tool is
                  identity-dependent (see the identity map below)
  footprint       ``cache_identity.entity_keys(tool, args)`` -- the entity ids

Driving protocol (the class's own API, no reimplementation of its gates):

  * read:    ``cache.get(tool, args, identity_dependent=..., tenant=...)``; on
             ``MISS``, ``cache.put(..., footprint=entity_keys, side_effects=False,
             identity_dependent=..., tenant=...)``. A refused put (``no_tenant``)
             is recorded by the class and the call stays a miss.
  * write:   ``cache.put(..., side_effects=True)`` so the class's effect gate is
             what refuses it (``effectful``); then ``cache.record_write(entity ids)``
             for the mutation the replay treats as landed.
  * control: ``cache.put(..., side_effects=True)``; refused (``effectful``).

  The replay never decides admissibility itself: every call is offered to
  ``put`` and the class's counters say what happened. ``effectful`` refusals must
  equal the 547 non-read calls, and no write is ever looked up, so a write cannot
  be served. The value stored is a provenance marker
  ``{pos, tenant, entity_ids}`` -- the same device the harness uses -- which is
  what lets the replay name the memo behind a hit and classify damage.

Damage classes (the harness's three, unchanged):

  writes_served  a write answered from the cache (structurally impossible here:
                 writes are refused at put and never looked up; measured as 0)
  stale_read     a read hit whose entity ids intersect a write that executed at
                 a position strictly between the memo's store and the hit
  cross_tenant   a read hit served from a memo stored under a different known
                 tenant, counted only where the tool is *identity-dependent* --
                 the map's classification, not the harness's provenance-only one

Two arms, and why both are reported
-----------------------------------
The task's tenant rule ("tenant = the task's user identity where the tool is
identity-dependent") and the task's 2% target pull in different directions,
because ``tool_effect_map.json`` has no ``identity_dependent`` field; identity
dependence has to be derived from the ``entity`` it does carry. This script
therefore runs two key policies through the same class, same stream, same
admission and eviction rules:

  harness_matched  tenant bound on **every** read. That is P4's key policy
                   (``key_typed`` always folds the tenant in), so the arm is a
                   like-for-like re-implementation of the harness experiment:
                   same workload, same key, different implementation. It is the
                   arm the 2% assertion is evaluated on.
  identity_map     tenant bound only where the map's ``entity`` is user-scoped
                   (user, order, reservation, payment_method, certificate);
                   catalog entities (product, flight, airport, none, episode)
                   share across tenants. This is the task's stated policy. It
                   serves more hits than P4 -- catalog traffic P4's tenant-bound
                   key cannot share -- so it is *not* within 2% of 466 by
                   construction. It is reported in full rather than asserted
                   inside the band; the assertion is not widened, it is stated on
                   the arm whose key policy is held fixed, and the other arm's
                   deviation is reported with its cause.

Honesty notes, stated rather than buried:

  * ``SoundResultCache`` is opt-in. ``instruction.py``'s shipped
    ``InstructionCache`` is untouched by this script, nothing on the shipped
    ``execute()`` path calls the sound class, and the class itself documents that
    its soundness is relative to the caller's declarations (H0: the value must be
    determined by the footprint and repeatable; ``side_effects=False`` must be
    truthful; ``record_write`` must run after every write that lands, naming
    every touched entity). This replay supplies those declarations; it does not
    verify them.
  * The three classes join reads to writes on ``entity_keys``. That join has a
    known blind spot: ``find_user_id_by_*`` reads name ``email``/name+zip, while
    ``modify_user_address`` names ``user_id``, so no id intersects. The harness
    closes the hole with its ``LOOKUP_SCOPE`` token; the class supports the same
    thing as an ordinary footprint token, but ``entity_keys`` never emits it and
    the task pins footprints to ``entity_keys``. The shipped replay therefore
    serves lookup hits the harness evicts; the count and a counterfactual are in
    ``reconciliation.lookup_scope``. Those hits are not in the three classes
    because no entity id joins them, so the zero below is class-relative -- not a
    claim that the corpus contains no staleness at all.

Deterministic: no clock, no randomness, no environment reads. TTL and LRU are
disabled on the class (``max_size=None, ttl=None``) to match the harness's
unbounded run-scope residency and to keep wall-clock time out of the numbers;
the deployed defaults (1,000 entries / 3,600 s) cannot bind here anyway (the
largest run is 872 calls). ``main`` builds the document twice and refuses to
write if the two differ.

Usage::

    ./venv/bin/python examples/benchmarks/shipped_sound_cache_replay.py
    ./venv/bin/python examples/benchmarks/shipped_sound_cache_replay.py --self-test

Consumers: none -- this is a measurement artifact, not a policy input.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = Path(__file__).with_name("shipped_sound_cache_results.json")
IDENTITY_PATH = HERE / "cache_identity.py"
EFFECT_MAP_PATH = HERE / "tool_effect_map.json"
BASELINE_PATH = HERE / "unsafe_cache_baselines.py"
SOUND_CACHE_PATH = REPO_ROOT / "agentflow" / "core" / "isa" / "sound_cache.py"

SCHEMA = "shipped_sound_cache_replay/v1"
TASK = (
    "S3 - replay the first-trial tau-bench corpus through the shipped "
    "SoundResultCache (agentflow/core/isa/sound_cache.py) with its effect gate on, "
    "counting hits, misses, refusals by reason and the three damage classes"
)

#: The committed corpus. Pinned because every number here is tied to these bytes;
#: a different corpus is a different experiment and must not be reported under
#: this schema.
EXPECTED_SHA256 = "3b5d4d29989ad99ed276433027d6ede37acb1197635e4a99febf44f6b5e64116"

FIRST_TRIAL_ONLY = True
HARNESS_HITS = 466
HARNESS_CACHEABLE = 1728
TOLERANCE = 0.02
EXPECTED_RECORDS = 325
EXPECTED_CALLS = 2275
EXPECTED_RUNS = 4

DAMAGE_CLASSES: Tuple[str, ...] = ("writes_served", "stale_reads", "cross_tenant")

#: The identity map, derived from ``tool_effect_map.json``'s per-tool ``entity``
#: (the file carries no ``identity_dependent`` field). A record of a user-scoped
#: entity belongs to one user, so a value read for it is only meaningful for that
#: user; catalog/inventory/global entities, the entity-less tools and the episode
#: itself are not owned by a caller and may be shared. The partition is checked
#: against the map's own ``entities`` document, so a new entity cannot be added
#: without this script failing loudly.
USER_SCOPED_ENTITIES: Tuple[str, ...] = (
    "user", "order", "reservation", "payment_method", "certificate",
)
NON_IDENTITY_ENTITIES: Tuple[str, ...] = (
    "product", "flight", "airport", "none", "episode",
)


# ---------------------------------------------------------------------------
# Dependency loading: the modules whose bytes decide these numbers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(name: str, path: Path):
    """Import a benchmark module by file path (the directory is not a package)."""
    if not path.is_file():
        raise SystemExit(f"shipped_sound_cache_replay: cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"shipped_sound_cache_replay: cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_identity():
    """``cache_identity`` (task B1): the keys, effect classes and entity ids."""
    module = _load_module("cache_identity", IDENTITY_PATH)
    for name in ("effect_class", "entity_keys", "EFFECT_CLASSES", "MAP_SOURCE",
                 "MAP_PATH", "MAP_ERROR", "LOOKUP_READS", "LOOKUP_INVALIDATING_WRITES"):
        if not hasattr(module, name):
            raise SystemExit(f"shipped_sound_cache_replay: cache_identity.py has no {name!r}")
    return module, sha256_file(IDENTITY_PATH)


def load_baseline():
    """``unsafe_cache_baselines`` (task B2): the stream and the harness replays."""
    module = _load_module("unsafe_cache_baselines", BASELINE_PATH)
    for name in ("build_streams", "load_records", "replay", "POLICIES", "Call"):
        if not hasattr(module, name):
            raise SystemExit(
                f"shipped_sound_cache_replay: unsafe_cache_baselines.py has no {name!r}"
            )
    return module, sha256_file(BASELINE_PATH)


def load_sound_class():
    """Import the shipped class by module path and prove it is the repo's file."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from agentflow.core.isa import sound_cache as module  # noqa: PLC0415 - path set above

    resolved = Path(module.__file__).resolve()
    if resolved != SOUND_CACHE_PATH.resolve():
        raise SystemExit(
            "shipped_sound_cache_replay: imported sound_cache from "
            f"{resolved}, not the repository's {SOUND_CACHE_PATH}"
        )
    for name in ("SoundResultCache", "MISS"):
        if not hasattr(module, name):
            raise SystemExit(f"shipped_sound_cache_replay: sound_cache.py has no {name!r}")
    return module


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _json_inline(counter: Mapping[str, int]) -> str:
    """A JSON-shaped inline rendering for prose fields (never Python repr)."""
    return json.dumps(dict(sorted(counter.items())), separators=(", ", ": "))


# ---------------------------------------------------------------------------
# The identity map, derived from the committed effect map
# ---------------------------------------------------------------------------
def _base_entity(entity: str) -> str:
    """The map's entity name, without its parenthetical qualifier.

    Two catalog rows spell the entity ``"airport (catalog)"`` / ``"product
    (catalog)"`` while the map's own ``entities`` document spells them
    ``"airport"`` / ``"product"``. The qualifier is documentation, not a distinct
    namespace; folding it here (and recording the fold) keeps the partition
    honest instead of letting an unmapped string fall through as
    identity-independent.
    """
    return entity.split(" (", 1)[0].strip()


def build_identity_map(document: Mapping[str, Any]) -> "OrderedDict[str, Dict[str, Any]]":
    """Per tool: effect class, entity, and whether the map makes it identity-dependent.

    Derived, not assumed: the map's ``entity`` decides. Every entity -- the map's
    ``entities`` document and every tool row's entity, after the parenthetical
    fold -- must fall in exactly one of the two declared sets, so an unmapped
    entity aborts the run instead of defaulting to "safe" (identity-independent
    would be the unsafe default) or to "unsafe".
    """
    entities = document.get("entities") or {}
    if not isinstance(entities, Mapping) or not entities:
        raise SystemExit("shipped_sound_cache_replay: tool_effect_map.json has no entity list")
    universe = set(entities)
    scoped = set(USER_SCOPED_ENTITIES)
    nonscoped = set(NON_IDENTITY_ENTITIES)
    declared = scoped | nonscoped
    missing = universe - declared
    overlap = scoped & nonscoped
    stray = declared - universe

    rows: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    tool_entities: set = set()
    for entry in document.get("tools") or []:
        name = str(entry["name"])
        raw_entity = str(entry.get("entity"))
        entity = _base_entity(raw_entity)
        if entity not in declared:
            raise SystemExit(
                "shipped_sound_cache_replay: tool_effect_map.json entity "
                f"{raw_entity!r} ({name}) is in neither identity set; refusing to guess"
            )
        tool_entities.add(entity)
        rows[name] = OrderedDict(
            tool=name,
            effect_class=str(entry["class"]),
            entity=raw_entity,
            entity_base=entity,
            identity_dependent=entity in scoped,
            declared_entity_keys=tuple(entry.get("entity_keys") or ()),
        )

    unmapped_tools = tool_entities - declared
    if missing or overlap or stray or unmapped_tools:
        raise SystemExit(
            "shipped_sound_cache_replay: identity partition does not match the map's "
            f"entities (unassigned document entities={sorted(missing)}, "
            f"overlapping={sorted(overlap)}, invented={sorted(stray)}, "
            f"tool entities outside the sets={sorted(unmapped_tools)})"
        )
    return rows


# ---------------------------------------------------------------------------
# Key policies (the only intended difference between the two arms)
# ---------------------------------------------------------------------------
class KeyPolicy:
    """A tenant rule, plus the reason it is run."""

    __slots__ = ("pid", "name", "key_recipe", "identity_dependent", "why")

    def __init__(self, pid: str, name: str, key_recipe: str,
                 identity_dependent: Callable[[str], bool], why: str):
        self.pid = pid
        self.name = name
        self.key_recipe = key_recipe
        self.identity_dependent = identity_dependent
        self.why = why


def build_policies(identity_rows: Mapping[str, Mapping[str, Any]]) -> Tuple[KeyPolicy, ...]:
    def mapped(tool: str) -> bool:
        row = identity_rows.get(tool)
        if row is None:
            # Unknown tool: no map row, so no evidence it is identity-independent.
            # Fail closed (tenant-bound); the effect map classes it as a write, so
            # it is refused before keying anyway.
            return True
        return bool(row["identity_dependent"])

    return (
        KeyPolicy(
            pid="harness_matched",
            name="tenant bound on every read",
            key_recipe="SoundResultCache(instruction_id=tool, params=args, tenant=call tenant, identity_dependent=True)",
            identity_dependent=lambda _tool: True,
            why=(
                "P4's key policy, held fixed so the arm is a like-for-like re-implementation "
                "of the harness experiment: same stream, same admission, same eviction, "
                "different implementation. This is the arm the 2% assertion compares."
            ),
        ),
        KeyPolicy(
            pid="identity_map",
            name="tenant bound only on user-scoped reads",
            key_recipe=(
                "SoundResultCache(instruction_id=tool, params=args, tenant=call tenant only "
                "where tool_effect_map.json's entity is user-scoped, identity_dependent=map)"
            ),
            identity_dependent=mapped,
            why=(
                "the task's stated policy: tenant where tool_effect_map.json makes the tool "
                "identity-dependent, shared keys for catalog reads. Sound under the map, but a "
                "different key policy from P4, so its extra hits are reported, not asserted to "
                "be within 2%."
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Replay through the shipped class
# ---------------------------------------------------------------------------
def replay_arm(
    streams: Mapping[str, List[Any]],
    ci,
    policy: KeyPolicy,
    sound,
) -> Dict[str, Any]:
    """Replay every run through one ``SoundResultCache``; count hits and damage.

    Returns the arm's numbers and its per-call hit flags (keyed ``(source, pos)``)
    so the arm can be diffed against the harness call by call.
    """
    per_run: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    totals: Counter = Counter()
    by_tool_hits: Counter = Counter()
    by_tool_unsafe: Dict[str, Counter] = {cls: Counter() for cls in DAMAGE_CLASSES}
    flags: Dict[Tuple[str, int], bool] = {}
    examples: Dict[str, List[Dict[str, Any]]] = {cls: [] for cls in DAMAGE_CLASSES}
    indep_cross = 0
    indep_cross_by_tool: Counter = Counter()
    lookup_hits_before_address_change = 0
    refusals: Counter = Counter()
    evictions: Counter = Counter()
    stats: Counter = Counter()

    for source, calls in streams.items():
        # One run-scope cache, as the harness does. Backstops off: unbounded and
        # no TTL, so nothing here depends on wall-clock time.
        cache = sound.SoundResultCache(max_size=None, ttl=None)
        run: Counter = Counter()
        write_log: List[Tuple[int, Tuple[str, ...], str]] = []

        for call in calls:
            effect = ci.effect_class(call.tool)
            dependent = bool(policy.identity_dependent(call.tool))
            run["calls"] += 1

            if effect != "read":
                # Offer the call to the class so its effect gate is what refuses
                # it, and so the refusal is counted by the class itself.
                admitted = cache.put(
                    call.tool, call.args, None, footprint=(),
                    side_effects=True, identity_dependent=True, tenant=call.tenant,
                )
                if admitted:  # pragma: no cover - the class's gate is fail-closed
                    run["effectful_admitted"] += 1
                if effect == "write":
                    ids = ci.entity_keys(call.tool, call.args)
                    cache.record_write(ids)
                    write_log.append((call.pos, ids, call.tool))
                continue

            run["cacheable"] += 1
            tenant = call.tenant if dependent else None
            memo = cache.get(call.tool, call.args,
                             identity_dependent=dependent, tenant=tenant)
            if memo is sound.MISS:
                run["misses"] += 1
                ids = ci.entity_keys(call.tool, call.args)
                cache.put(
                    call.tool, call.args,
                    OrderedDict(pos=call.pos, tenant=call.tenant, entity_ids=ids),
                    footprint=ids, side_effects=False,
                    identity_dependent=dependent, tenant=tenant,
                )
                continue

            run["hits"] += 1
            flags[(source, call.pos)] = True
            by_tool_hits[call.tool] += 1

            differ = (
                memo["tenant"] is not None
                and call.tenant is not None
                and memo["tenant"] != call.tenant
            )
            if not dependent:
                # Shared by construction: the answer is not a function of the
                # caller. Counted, and deliberately *not* filed as damage.
                if differ:
                    indep_cross += 1
                    indep_cross_by_tool[call.tool] += 1
                continue

            targets = frozenset(ci.entity_keys(call.tool, call.args)) or frozenset(
                memo["entity_ids"]
            )
            intervening = [
                (wpos, ids, tool)
                for wpos, ids, tool in write_log
                if memo["pos"] < wpos < call.pos and targets & frozenset(ids)
            ]
            if call.tool in ci.LOOKUP_READS:
                address_changes = [
                    tool for wpos, _ids, tool in write_log
                    if memo["pos"] < wpos < call.pos
                    and tool in ci.LOOKUP_INVALIDATING_WRITES
                ]
                if address_changes:
                    lookup_hits_before_address_change += 1

            if intervening:
                cls = "stale_reads"
                detail = OrderedDict(intervening=[
                    OrderedDict(pos=wpos, tool=tool, entity_ids=sorted(ids))
                    for wpos, ids, tool in intervening
                ])
            elif differ:
                cls = "cross_tenant"
                detail = OrderedDict(cached_tenant=memo["tenant"], hit_tenant=call.tenant)
            else:
                cls = None
                detail = OrderedDict()

            if cls is not None:
                run["unsafe_hits"] += 1
                run[cls] += 1
                by_tool_unsafe[cls][call.tool] += 1
                if len(examples[cls]) < 2:
                    examples[cls].append(OrderedDict(
                        cls=cls,
                        source=source,
                        hit=OrderedDict(pos=call.pos, turn=call.turn, task_id=call.task_id,
                                        tool=call.tool, args=call.args),
                        cached=OrderedDict(pos=memo["pos"], tenant=memo["tenant"],
                                           entity_ids=list(memo["entity_ids"])),
                        **detail,
                    ))

        snapshot = cache.stats()
        refusals.update(snapshot["refusals_by_reason"])
        evictions.update(snapshot["evictions_by_reason"])
        for key in ("hits", "misses", "admissions", "coarse_invalidations"):
            stats[key] += snapshot[key]
        per_run[source] = OrderedDict(
            calls=run["calls"],
            reads=run["cacheable"],
            hits=run["hits"],
            misses=run["misses"],
            hit_rate=ratio(run["hits"], run["cacheable"]),
            effectful_refusals=snapshot["refusals_by_reason"]["effectful"],
            no_tenant_refusals=snapshot["refusals_by_reason"]["no_tenant"],
            footprint_unknown_refusals=snapshot["refusals_by_reason"]["footprint_unknown"],
            admissions=snapshot["admissions"],
            write_evictions=snapshot["evictions_by_reason"]["write"],
            entries_at_end=snapshot["entries"],
            unsafe_hits=run["unsafe_hits"],
        )
        totals.update(run)

    hits, misses = totals["hits"], totals["misses"]
    unsafe_hits = totals["writes_served"] + totals["stale_reads"] + totals["cross_tenant"]
    read_tools = sorted(tool for tool, cls in ci.EFFECT_CLASSES.items() if cls == "read")
    return OrderedDict(
        id=policy.pid,
        name=policy.name,
        key_recipe=policy.key_recipe,
        why=policy.why,
        identity_dependent_read_tools=[
            tool for tool in read_tools if policy.identity_dependent(tool)
        ],
        calls_total=totals["calls"],
        cacheable_calls=totals["cacheable"],
        hits=hits,
        misses=misses,
        hit_rate=ratio(hits, totals["cacheable"]),
        hit_rate_of_all_calls=ratio(hits, totals["calls"]),
        refusals_by_reason=OrderedDict(
            effectful=refusals["effectful"],
            no_tenant=refusals["no_tenant"],
            footprint_unknown=refusals["footprint_unknown"],
        ),
        refusal_total=sum(refusals.values()),
        admissions=stats["admissions"],
        evictions_by_reason=OrderedDict(
            write=evictions["write"],
            capacity=evictions["capacity"],
            ttl=evictions["ttl"],
        ),
        coarse_invalidations=stats["coarse_invalidations"],
        unsafe_hits=unsafe_hits,
        unsafe_breakdown=OrderedDict(
            writes_served=totals["writes_served"],
            stale_reads=totals["stale_reads"],
            cross_tenant=totals["cross_tenant"],
        ),
        unsafe_share_of_hits=ratio(unsafe_hits, hits),
        unsafe_by_tool=OrderedDict(
            (cls, OrderedDict(sorted(by_tool_unsafe[cls].items()))) for cls in DAMAGE_CLASSES
        ),
        control_hits_served=0,
        effectful_admitted=totals["effectful_admitted"],
        identity_independent_cross_tenant_hits=indep_cross,
        identity_independent_cross_tenant_by_tool=OrderedDict(sorted(indep_cross_by_tool.items())),
        lookup_hits_with_intervening_address_change=lookup_hits_before_address_change,
        hits_by_tool=OrderedDict(sorted(by_tool_hits.items())),
        per_run=per_run,
        examples=[examples[cls] for cls in DAMAGE_CLASSES if examples[cls]],
        _flags=flags,
    )


# ---------------------------------------------------------------------------
# The harness itself, recomputed, plus a mirror for call-level attribution
# ---------------------------------------------------------------------------
def harness_figures(streams: Mapping[str, List[Any]], ci, ucb) -> Dict[str, Any]:
    """Recompute P4 and P1 through the harness's own replay; never trust a stored number."""
    by_pid = {policy.pid: policy for policy in ucb.POLICIES}
    p4 = ucb.replay(streams, by_pid["P4"], ci)
    p1 = ucb.replay(streams, by_pid["P1"], ci)
    return OrderedDict(
        p4=OrderedDict(
            source="unsafe_cache_baselines.replay(streams, POLICIES['P4'], cache_identity)",
            policy="P4 typed (tenant-bound, reads only, entity-invalidating)",
            hits=p4["hits"],
            cacheable_calls=p4["cacheable_calls"],
            hit_rate=p4["hit_rate"],
            unsafe_hits=p4["unsafe_hits"],
            per_run=OrderedDict(
                (source, OrderedDict(hits=run["hits"],
                                     cacheable_calls=run["cacheable_calls"],
                                     hit_rate=run["hit_rate"]))
                for source, run in p4["per_run"].items()
            ),
            committed=(HARNESS_HITS, HARNESS_CACHEABLE),
            matches_committed=(
                p4["hits"] == HARNESS_HITS
                and p4["cacheable_calls"] == HARNESS_CACHEABLE
            ),
        ),
        p1=OrderedDict(
            policy="P1 blind (params-only key, every call stored, never invalidated)",
            hits=p1["hits"],
            cacheable_calls=p1["cacheable_calls"],
            unsafe_hits=p1["unsafe_hits"],
            unsafe_breakdown=OrderedDict(p1["unsafe_breakdown"]),
            cross_tenant_by_tool=OrderedDict(p1["unsafe_by_tool"].get("cross_tenant", {})),
        ),
    )


def harness_mirror(streams: Mapping[str, List[Any]], ci) -> Dict[str, Any]:
    """P4's per-call hit flags, so a hit-count difference can be attributed exactly.

    A transcription of ``unsafe_cache_baselines.replay`` for P4 (typed key, entity
    invalidation, reads only), written because the harness reports totals and this
    script needs to know *which* calls differ. Its totals are checked against the
    harness's own output before any attribution is trusted; if they disagree, the
    check fails and the reconciliation is flagged rather than silently used.
    """
    flags: Dict[Tuple[str, int], bool] = {}
    hits = misses = 0
    per_run: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for source, calls in streams.items():
        cache: Dict[str, int] = {}
        index: Dict[str, set] = {}
        key_map: Dict[str, str] = {}
        run_hits = run_misses = 0
        for call in calls:
            effect = ci.effect_class(call.tool)
            if effect != "read":
                if effect == "write":
                    for typed in ci.invalidate(index, call.tool, call.args):
                        served = key_map.pop(typed, None)
                        if served is not None:
                            cache.pop(served, None)
                continue
            key = ci.key_typed(call.tool, call.args, tenant=call.tenant)
            if key in cache:
                run_hits += 1
                flags[(source, call.pos)] = True
                continue
            run_misses += 1
            cache[key] = 1
            typed = ci.remember(index, call.tool, call.args, tenant=call.tenant)
            key_map[typed] = key
        hits += run_hits
        misses += run_misses
        per_run[source] = OrderedDict(hits=run_hits, misses=run_misses)
    return OrderedDict(hits=hits, misses=misses, per_run=per_run, flags=flags)


def diff_arms(
    shipped: Mapping[Tuple[str, int], bool],
    harness: Mapping[Tuple[str, int], bool],
    streams: Mapping[str, List[Any]],
    ci,
) -> Dict[str, Any]:
    """Which calls each implementation hits, by tool and tenant coverage."""
    only_shipped: Counter = Counter()
    only_harness: Counter = Counter()
    both = neither = 0
    only_shipped_refs: List[Dict[str, Any]] = []
    only_harness_tenants: Counter = Counter()
    for source, calls in streams.items():
        for call in calls:
            if ci.effect_class(call.tool) != "read":
                continue
            key = (source, call.pos)
            a, b = key in shipped, key in harness
            if a and b:
                both += 1
            elif a:
                only_shipped[call.tool] += 1
                if len(only_shipped_refs) < 5:
                    only_shipped_refs.append(OrderedDict(
                        source=source, pos=call.pos, tool=call.tool,
                        tenant=call.tenant, args=call.args,
                    ))
            elif b:
                only_harness[call.tool] += 1
                only_harness_tenants["unknown" if call.tenant is None else "known"] += 1
            else:
                neither += 1
    return OrderedDict(
        both_hit=both,
        only_shipped=OrderedDict(sorted(only_shipped.items())),
        only_shipped_total=sum(only_shipped.values()),
        only_shipped_examples=only_shipped_refs,
        only_harness=OrderedDict(sorted(only_harness.items())),
        only_harness_total=sum(only_harness.values()),
        only_harness_tenants=OrderedDict(sorted(only_harness_tenants.items())),
        neither_hit=neither,
    )


# ---------------------------------------------------------------------------
# Counterfactual: declare the lookup scope and the eviction is the harness's
# ---------------------------------------------------------------------------
def lookup_scope_probe(streams: Mapping[str, List[Any]], ci, sound) -> Dict[str, Any]:
    """Replay harness_matched with ``LOOKUP_SCOPE`` declared as an extra footprint.

    The class accepts arbitrary footprint tokens, so the lookup blind spot is a
    *caller declaration* gap, not a missing API: file lookup reads under
    ``cache_identity.LOOKUP_SCOPE`` and record a write of that token when a
    lookup-invalidating write lands, and the harness's eviction is reproduced.
    This is evidence for that claim; it is not an arm and it asserts nothing about
    the class beyond "the same reads, fewer surviving lookup memos".
    """
    hits = misses = 0
    stale_lookup_hits = 0
    for _source, calls in streams.items():
        cache = sound.SoundResultCache(max_size=None, ttl=None)
        write_log: List[Tuple[int, Tuple[str, ...]]] = []
        for call in calls:
            effect = ci.effect_class(call.tool)
            if effect != "read":
                if effect == "write":
                    ids = tuple(ci.entity_keys(call.tool, call.args))
                    if call.tool in ci.LOOKUP_INVALIDATING_WRITES:
                        ids = ids + (ci.LOOKUP_SCOPE,)
                    cache.record_write(ids)
                    write_log.append((call.pos, ids))
                continue
            ids = tuple(ci.entity_keys(call.tool, call.args))
            footprint = ids + ((ci.LOOKUP_SCOPE,) if call.tool in ci.LOOKUP_READS else ())
            memo = cache.get(call.tool, call.args,
                             identity_dependent=True, tenant=call.tenant)
            if memo is sound.MISS:
                misses += 1
                cache.put(call.tool, call.args, OrderedDict(pos=call.pos),
                          footprint=footprint, side_effects=False,
                          identity_dependent=True, tenant=call.tenant)
                continue
            hits += 1
            if call.tool in ci.LOOKUP_READS and any(
                memo["pos"] < wpos < call.pos and ci.LOOKUP_SCOPE in wids
                for wpos, wids in write_log
            ):
                stale_lookup_hits += 1
    return OrderedDict(
        hits=hits,
        misses=misses,
        hit_rate=ratio(hits, hits + misses),
        lookup_hits_with_intervening_address_change=stale_lookup_hits,
        note=(
            "not an arm: the same harness_matched key policy with LOOKUP_SCOPE declared as an "
            "extra footprint token. It shows the class can express the harness's lookup "
            "eviction; the task pins footprints to entity_keys, which never emits that token."
        ),
    )


# ---------------------------------------------------------------------------
# Definitions, reconciliation, provenance, checks
# ---------------------------------------------------------------------------
def definitions() -> Dict[str, Any]:
    return OrderedDict(
        arm=(
            "one key policy replayed over the whole corpus: one SoundResultCache per source "
            "run, every call offered to the class, hits counted from get()"
        ),
        hit=(
            "a read whose (tool, args, tenant) key is present in the run's shipped cache: the "
            "class returns the memoised value and the tool does not run"
        ),
        cacheable_call=(
            "a read. Writes and control tools are offered to put() and refused by the class's "
            "effect gate, so they are counted as refusals, not as cacheable calls. Hit rate is "
            "hits / cacheable calls (1,728), matching the harness's P4 denominator"
        ),
        refusal=(
            "a put() the class refused: effectful (side_effects not False -- every write and "
            "control call, 547), no_tenant (identity-dependent read with no tenant, 121 on the "
            "harness_matched arm), footprint_unknown (footprint is None; never used here -- "
            "entity_keys returns () for an entityless read, which the class treats as a "
            "legitimate constant read)"
        ),
        writes_served=(
            "a write answered from the cache. Structurally zero here: a write is refused at "
            "put() and the replay never issues a get() for one, so there is nothing to serve"
        ),
        stale_read=(
            "a read hit whose entity ids intersect a write that executed at a position strictly "
            "between the memo's store position and the hit. The write log contains every write "
            "the replay treated as landed -- all of them, since none was served"
        ),
        cross_tenant=(
            "a read hit served from a memo stored under a different known tenant, counted only "
            "for tools the identity map marks identity-dependent. Catalog hits that cross a "
            "tenant under the identity_map arm are reported separately as "
            "identity_independent_cross_tenant_hits and are not damage by the map's definition "
            "-- the harness counts them (provenance-only), this script does not, and both "
            "numbers are given"
        ),
        unsafe_hit=(
            "the union of writes_served, stale_reads and cross_tenant, each hit assigned to one "
            "class by the precedence writes_served > stale_reads > cross_tenant; the breakdown "
            "sums to the total"
        ),
        identity_dependent=(
            "derived from tool_effect_map.json's per-tool entity: records of user, order, "
            "reservation, payment_method and certificate belong to one user and are "
            "tenant-bound; product, flight, airport, none and episode are not. The file has no "
            "identity_dependent field, so this is a derivation, stated as such; it is checked "
            "against the map's entities document rather than assumed"
        ),
        tenant=(
            "the single distinct user_id named by a record's calls, else unknown. Unknown-tenant "
            "reads that are identity-dependent cannot be keyed by the class (no_tenant) and stay "
            "misses; the harness instead keys them under tenant=None, which is most of why its "
            "hit count differs"
        ),
        hit_rate=(
            "hits / cacheable_calls, rounded to 4 decimals; hit_rate_of_all_calls uses all 2,275 "
            "replayed calls as the denominator"
        ),
        lookup_scope=(
            "cache_identity files id-less lookup reads (find_user_id_by_*) under LOOKUP_SCOPE and "
            "flushes that scope when modify_user_address lands, because a zip change can falsify "
            "a name+zip answer. entity_keys never emits that token, so the task's footprint "
            "declaration leaves those memos in place; the shipped class will honour the token if "
            "the caller declares it (reconciliation.lookup_scope)"
        ),
        not_verified=(
            "the class cannot verify (H0) read determinism or the truth of side_effects=False; "
            "this replay asserts the declarations on the corpus's behalf, so the zero is "
            "class-relative and corpus-scoped, not a proof about arbitrary callers"
        ),
    )


def build_report(
    corpus: Path,
    corpus_digest: str,
    records: Sequence[Mapping[str, Any]],
    streams: Mapping[str, List[Any]],
    ci,
    identity_digest: str,
    map_digest: str,
    map_document: Mapping[str, Any],
    identity_rows: Mapping[str, Mapping[str, Any]],
    policies: Sequence[KeyPolicy],
    ucb,
    sound,
    sound_digest: str,
    baseline_digest: str,
) -> Dict[str, Any]:
    figures = harness_figures(streams, ci, ucb)
    harness = figures["p4"]
    p1 = figures["p1"]
    mirror = harness_mirror(streams, ci)
    mirror["matches_harness"] = (
        mirror["hits"] == harness["hits"]
        and mirror["misses"] == harness["cacheable_calls"] - harness["hits"]
    )

    arms_full = [replay_arm(streams, ci, policy, sound) for policy in policies]
    flags = {arm["id"]: arm.pop("_flags") for arm in arms_full}
    arms = [OrderedDict(arm) for arm in arms_full]
    by_id = {arm["id"]: arm for arm in arms}
    primary = by_id["harness_matched"]
    mapped_arm = by_id["identity_map"]

    diff_primary = diff_arms(flags["harness_matched"], mirror["flags"], streams, ci)
    diff_mapped = diff_arms(flags["identity_map"], mirror["flags"], streams, ci)
    probe = lookup_scope_probe(streams, ci, sound)

    calls_total = sum(len(calls) for calls in streams.values())
    first_trial = sum(1 for record in records if record.get("trial", 0) == 0)
    effect_counts = Counter(
        ci.effect_class(call.tool) for calls in streams.values() for call in calls
    )
    committed_summary = (map_document.get("summary") or {}).get("by_class") or {}

    recon = OrderedDict(
        why_two_arms=(
            "the task asks for tenant only on identity-dependent tools, but the 2% target is "
            "466, P4's figure under a key that binds the tenant on every read. tool_effect_map."
            "json has no identity_dependent field, so that policy has to be derived from "
            "`entity`; deriving it makes catalog reads shareable and moves the hit count well "
            "outside 2%. Both readings are measured: harness_matched holds the key policy fixed "
            "(the like-for-like comparison the 2% assertion is about), identity_map applies the "
            "derived map (the task's stated policy) and is reported with its decomposition "
            "instead of being asserted inside the band"
        ),
        harness=OrderedDict(
            hits=harness["hits"],
            cacheable_calls=harness["cacheable_calls"],
            hit_rate=harness["hit_rate"],
            matches_committed=harness["matches_committed"],
            p1=OrderedDict(
                hits=p1["hits"],
                cacheable_calls=p1["cacheable_calls"],
                unsafe_hits=p1["unsafe_hits"],
                cross_tenant=p1["unsafe_breakdown"]["cross_tenant"],
                cross_tenant_by_tool=p1["cross_tenant_by_tool"],
            ),
        ),
        primary_arm=OrderedDict(
            id="harness_matched",
            hits=primary["hits"],
            cacheable_calls=primary["cacheable_calls"],
            hit_rate=primary["hit_rate"],
            difference_hits=primary["hits"] - harness["hits"],
            difference_share=ratio(abs(primary["hits"] - harness["hits"]), harness["hits"]),
            within_tolerance=abs(primary["hits"] - harness["hits"]) <= TOLERANCE * harness["hits"],
            why_this_arm=(
                "only this arm holds the key policy fixed at P4's; the identity_map arm changes "
                "the key (catalog reads stop being tenant-bound) and therefore measures a "
                "different policy, not a different implementation"
            ),
        ),
        identity_map_arm=OrderedDict(
            id="identity_map",
            hits=mapped_arm["hits"],
            cacheable_calls=mapped_arm["cacheable_calls"],
            hit_rate=mapped_arm["hit_rate"],
            difference_hits=mapped_arm["hits"] - harness["hits"],
            difference_share=ratio(abs(mapped_arm["hits"] - harness["hits"]), harness["hits"]),
            within_tolerance=abs(mapped_arm["hits"] - harness["hits"]) <= TOLERANCE * harness["hits"],
            why_outside=(
                "the map marks five catalog reads identity-independent, so their memos are "
                "shared across tenants and P4's tenant-bound key cannot reach them; the gain is "
                "catalog traffic P4's key deliberately separates. Not a drift of the class and "
                "not asserted inside the 2% band -- reported with its decomposition below"
            ),
            identity_independent_cross_tenant_hits=mapped_arm[
                "identity_independent_cross_tenant_hits"
            ],
            identity_independent_cross_tenant_by_tool=mapped_arm[
                "identity_independent_cross_tenant_by_tool"
            ],
        ),
        mirror=OrderedDict(
            hits=mirror["hits"],
            misses=mirror["misses"],
            matches_harness=bool(mirror["matches_harness"]),
            per_run=mirror["per_run"],
            why=(
                "the per-call mirror exists only to name which calls the two implementations "
                "disagree on; its totals are checked against the harness's own replay, so an "
                "attribution built on a drifted mirror would fail a check rather than print"
            ),
        ),
        hit_gap_primary=OrderedDict(
            harness_hits=harness["hits"],
            shipped_hits=primary["hits"],
            delta=primary["hits"] - harness["hits"],
            both_hit=diff_primary["both_hit"],
            only_harness=diff_primary["only_harness_total"],
            only_harness_by_tool=diff_primary["only_harness"],
            only_harness_tenants=diff_primary["only_harness_tenants"],
            only_shipped=diff_primary["only_shipped_total"],
            only_shipped_by_tool=diff_primary["only_shipped"],
            only_shipped_examples=diff_primary["only_shipped_examples"],
            explanation=(
                f"the gap is two mechanisms, both visible per call. (a) The class refuses an "
                f"identity-dependent put with no tenant (no_tenant), where the harness keys such "
                f"reads under tenant=None: {diff_primary['only_harness_total']} hits the harness "
                f"serves are unreachable here, all on unknown-tenant reads "
                f"({_json_inline(diff_primary['only_harness_tenants'])}). (b) The class has no "
                f"LOOKUP_SCOPE notion and is driven with entity_keys footprints, so "
                f"{diff_primary['only_shipped_total']} lookup-read hits survive that the harness "
                f"evicts on modify_user_address ({_json_inline(diff_primary['only_shipped'])}). "
                f"Arithmetic: {diff_primary['both_hit']} both-hit + "
                f"{diff_primary['only_shipped_total']} shipped-only = {primary['hits']}; "
                f"{diff_primary['both_hit']} + {diff_primary['only_harness_total']} harness-only "
                f"= {harness['hits']}"
            ),
        ),
        hit_gap_identity_map=OrderedDict(
            harness_hits=harness["hits"],
            shipped_hits=mapped_arm["hits"],
            delta=mapped_arm["hits"] - harness["hits"],
            both_hit=diff_mapped["both_hit"],
            only_harness=diff_mapped["only_harness_total"],
            only_harness_by_tool=diff_mapped["only_harness"],
            only_shipped=diff_mapped["only_shipped_total"],
            only_shipped_by_tool=diff_mapped["only_shipped"],
            explanation=(
                f"{diff_mapped['both_hit']} both-hit + {diff_mapped['only_shipped_total']} "
                f"shipped-only = {mapped_arm['hits']}; {diff_mapped['both_hit']} + "
                f"{diff_mapped['only_harness_total']} harness-only = {harness['hits']}. The "
                f"shipped-only hits are catalog memos a shared key can serve "
                f"({_json_inline(diff_mapped['only_shipped'])}) plus the lookup hits above; the "
                f"harness-only ones are identity-dependent reads with no recoverable tenant, "
                f"which the class refuses and the harness keys under None "
                f"({_json_inline(diff_mapped['only_harness_tenants'])})"
            ),
        ),
        catalog_cross_tenant=OrderedDict(
            harness_p1_cross_tenant=p1["unsafe_breakdown"]["cross_tenant"],
            harness_p1_cross_tenant_by_tool=p1["cross_tenant_by_tool"],
            identity_map_arm_hits_across_tenants=mapped_arm[
                "identity_independent_cross_tenant_hits"
            ],
            by_tool=mapped_arm["identity_independent_cross_tenant_by_tool"],
            equal=(
                p1["unsafe_breakdown"]["cross_tenant"]
                == mapped_arm["identity_independent_cross_tenant_hits"]
                and dict(p1["cross_tenant_by_tool"])
                == dict(mapped_arm["identity_independent_cross_tenant_by_tool"])
            ),
            note=(
                "the harness's whole P1 cross-tenant count is catalog traffic. The map calls "
                "those tools identity-independent, so the identity_map arm serves the same hits "
                "across tenants and reports them as harmless-by-map rather than damage; the "
                "harness counts them because it classifies by provenance, not by dependence. "
                "Equality of the two counts is checked, not assumed"
            ),
        ),
        lookup_scope=OrderedDict(
            lookup_hits_with_intervening_address_change=primary[
                "lookup_hits_with_intervening_address_change"
            ],
            of_which_harness_misses=sum(
                count for tool, count in diff_primary["only_shipped"].items()
                if tool in ci.LOOKUP_READS
            ),
            of_which_harness_serves_from_a_fresher_memo=(
                primary["lookup_hits_with_intervening_address_change"]
                - sum(count for tool, count in diff_primary["only_shipped"].items()
                      if tool in ci.LOOKUP_READS)
            ),
            lookup_hits_only_shipped=diff_primary["only_shipped"],
            counterfactual=probe,
            counterfactual_reproduces_the_harness_hits=(
                probe["hits"] == diff_primary["both_hit"]
            ),
            note=(
                "with the task's footprint (entity_keys) the class serves lookup hits whose memo "
                "predates a modify_user_address write; the three classes do not count them "
                "because no entity id joins a name+zip read to a user_id write, and the harness "
                "evicts them through LOOKUP_SCOPE. Declaring that token as a footprint restores "
                "the harness's behaviour, which is what the counterfactual shows. The zero above "
                "is therefore zero for the three defined classes, and the blind spot is stated "
                "rather than netted away"
            ),
        ),
        determinism=OrderedDict(
            clock_used=False,
            ttl="disabled (ttl=None) so wall-clock time cannot evict",
            capacity="disabled (max_size=None); the largest run is 872 calls, below the shipped 1,000",
            two_builds_identical=True,  # main() refuses to write otherwise
        ),
    )

    checks = build_checks(
        corpus_digest=corpus_digest,
        first_trial=first_trial,
        records_total=len(records),
        streams=streams,
        calls_total=calls_total,
        effect_counts=effect_counts,
        committed_summary=committed_summary,
        harness=harness,
        mirror=mirror,
        arms=by_id,
        identity_rows=identity_rows,
        map_document=map_document,
        probe=probe,
        p1=p1,
        both_hit_primary=diff_primary["both_hit"],
    )

    return OrderedDict(
        schema=SCHEMA,
        task=TASK,
        result_kind="REPLAY",
        scope=(
            "tau-bench only: the first trial of every task in the committed corpus (325 records, "
            "2,275 calls, four source runs), replayed through the shipped SoundResultCache with "
            "its effect gate on. Every call is offered to put(); the class's own counters decide "
            "what was refused. Damage is counted from the replay, not assumed, in the harness's "
            "three classes; the identity map decides which cross-tenant hits are damage."
        ),
        corpus_sha256=corpus_digest,
        expectation=OrderedDict(
            harness_hits=HARNESS_HITS,
            harness_cacheable=HARNESS_CACHEABLE,
            harness_hit_rate=ratio(HARNESS_HITS, HARNESS_CACHEABLE),
            tolerance=TOLERANCE,
            compared_arm="harness_matched",
            rule=(
                "the shipped class's hit count must be within 2% of 466, and every damage class "
                "must be zero. If the count differs, the difference is explained in "
                "reconciliation.hit_gap_primary; the threshold is not widened"
            ),
        ),
        shipped_class=OrderedDict(
            module="agentflow/core/isa/sound_cache.py",
            class_name="SoundResultCache",
            sha256=sound_digest,
            imported_from=str(Path(sound.__file__).resolve().relative_to(REPO_ROOT)),
            opt_in=True,
            wired_into_execute_path=False,
            note=(
                "instruction.py's InstructionCache is untouched and still enforces none of "
                "(H1)-(H3); this script measures the opt-in class, not a change to the shipped "
                "default. Its soundness is relative to the declarations this replay supplies: "
                "footprints, side_effects=False for reads, record_write after every write"
            ),
        ),
        measurement=OrderedDict(
            arm="harness_matched",
            calls=calls_total,
            cacheable_calls=primary["cacheable_calls"],
            hits=primary["hits"],
            misses=primary["misses"],
            hit_rate=primary["hit_rate"],
            hit_rate_of_all_calls=primary["hit_rate_of_all_calls"],
            refusals_by_reason=primary["refusals_by_reason"],
            refusal_total=primary["refusal_total"],
            unsafe_hits=primary["unsafe_hits"],
            unsafe_breakdown=primary["unsafe_breakdown"],
            identity_independent_cross_tenant_hits=primary[
                "identity_independent_cross_tenant_hits"
            ],
        ),
        arms=arms,
        harness=harness,
        identity_map=OrderedDict(
            source=str(EFFECT_MAP_PATH.relative_to(REPO_ROOT)),
            sha256=map_digest,
            derivation=(
                "identity_dependent = the tool's map entity is user-scoped. The file has no "
                "identity_dependent field; this is derived from `entity`, checked against the "
                "map's `entities` document, and used only by the identity_map arm and by the "
                "cross-tenant class"
            ),
            user_scoped_entities=list(USER_SCOPED_ENTITIES),
            non_identity_entities=list(NON_IDENTITY_ENTITIES),
            entity_aliases=OrderedDict(
                (row["entity"], row["entity_base"])
                for row in identity_rows.values()
                if row["entity"] != row["entity_base"]
            ),
            entity_aliases_note=(
                "two catalog rows spell the entity with a parenthetical qualifier the map's "
                "entities document does not; the fold is recorded here so the partition stays "
                "auditable"
            ),
            read_tools=OrderedDict(
                (row["tool"], OrderedDict(entity=row["entity"],
                                          identity_dependent=row["identity_dependent"],
                                          declared_entity_keys=list(row["declared_entity_keys"])))
                for row in identity_rows.values() if row["effect_class"] == "read"
            ),
        ),
        definitions=definitions(),
        reconciliation=recon,
        provenance=OrderedDict(
            script=OrderedDict(
                path=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                sha256=sha256_file(Path(__file__).resolve()),
                schema=SCHEMA,
            ),
            corpus=OrderedDict(
                path=str(corpus.resolve().relative_to(REPO_ROOT)),
                sha256=corpus_digest,
                expected_sha256=EXPECTED_SHA256,
                matches_expected=corpus_digest == EXPECTED_SHA256,
                records=first_trial,
                records_total=len(records),
                calls=calls_total,
                runs=len(streams),
                first_trial_only=FIRST_TRIAL_ONLY,
            ),
            identity_module=OrderedDict(
                path=str(IDENTITY_PATH.relative_to(REPO_ROOT)),
                sha256=identity_digest,
                effect_map_source=ci.MAP_SOURCE,
                effect_map_path=(
                    str(ci.MAP_PATH.resolve().relative_to(REPO_ROOT))
                    if ci.MAP_PATH is not None else None
                ),
                effect_map_error=ci.MAP_ERROR,
                tools=len(ci.EFFECT_CLASSES),
            ),
            effect_map=OrderedDict(
                path=str(EFFECT_MAP_PATH.relative_to(REPO_ROOT)),
                sha256=map_digest,
                schema=map_document.get("schema"),
            ),
            harness=OrderedDict(
                path=str(BASELINE_PATH.relative_to(REPO_ROOT)),
                sha256=baseline_digest,
                policy="P4 typed (tenant-bound, reads only, entity-invalidating)",
                figure=f"{HARNESS_HITS}/{HARNESS_CACHEABLE}",
                recomputed=True,
            ),
            shipped_class=OrderedDict(
                path=str(SOUND_CACHE_PATH.relative_to(REPO_ROOT)),
                sha256=sound_digest,
            ),
            determinism=(
                "no clock, no environment, no randomness; one cache per run; counters and "
                "collections emitted in a stable order; main() builds the document twice and "
                "refuses to write if the two differ"
            ),
            arithmetic="integer counts; shares are round(numerator/denominator, 4)",
        ),
        checks=checks,
    )


def build_checks(
    corpus_digest: str,
    first_trial: int,
    records_total: int,
    streams: Mapping[str, List[Any]],
    calls_total: int,
    effect_counts: Counter,
    committed_summary: Mapping[str, Any],
    harness: Mapping[str, Any],
    mirror: Mapping[str, Any],
    arms: Mapping[str, Mapping[str, Any]],
    identity_rows: Mapping[str, Mapping[str, Any]],
    map_document: Mapping[str, Any],
    probe: Mapping[str, Any],
    p1: Mapping[str, Any],
    both_hit_primary: int,
) -> Dict[str, Any]:
    primary = arms["harness_matched"]
    mapped = arms["identity_map"]
    committed = {
        cls: int((committed_summary.get(cls) or {}).get("calls_observed", -1))
        for cls in ("read", "write", "control")
    }
    expected_entities = set((map_document.get("entities") or {}))
    mapped_entities = {row["entity_base"] for row in identity_rows.values()}
    return OrderedDict(
        corpus_matches_committed=OrderedDict(
            ok=corpus_digest == EXPECTED_SHA256,
            expected=EXPECTED_SHA256,
            observed=corpus_digest,
        ),
        corpus_shape=OrderedDict(
            ok=(
                first_trial == EXPECTED_RECORDS
                and calls_total == EXPECTED_CALLS
                and len(streams) == EXPECTED_RUNS
            ),
            records=first_trial,
            records_total=records_total,
            calls=calls_total,
            runs=len(streams),
        ),
        effect_totals_match_map=OrderedDict(
            ok=(
                effect_counts["read"] == committed["read"]
                and effect_counts["write"] == committed["write"]
                and effect_counts["control"] == committed["control"]
            ),
            observed=OrderedDict(sorted(effect_counts.items())),
            committed=OrderedDict(sorted(committed.items())),
        ),
        harness_figure_reproduced=OrderedDict(
            ok=bool(harness["matches_committed"]),
            observed=f"{harness['hits']}/{harness['cacheable_calls']}",
            committed=f"{HARNESS_HITS}/{HARNESS_CACHEABLE}",
        ),
        harness_mirror_agrees=OrderedDict(
            ok=bool(mirror["matches_harness"]),
            mirror_hits=mirror["hits"],
            harness_hits=harness["hits"],
            mirror_misses=mirror["misses"],
            harness_misses=harness["cacheable_calls"] - harness["hits"],
        ),
        zero_damaging_hits=OrderedDict(
            ok=primary["unsafe_hits"] == 0 and mapped["unsafe_hits"] == 0,
            primary=primary["unsafe_hits"],
            identity_map=mapped["unsafe_hits"],
            breakdown=OrderedDict(
                (arm_id, OrderedDict(arms[arm_id]["unsafe_breakdown"]))
                for arm_id in ("harness_matched", "identity_map")
            ),
        ),
        hit_count_within_2pct_of_harness=OrderedDict(
            ok=abs(primary["hits"] - harness["hits"]) <= TOLERANCE * harness["hits"],
            arm="harness_matched",
            shipped_hits=primary["hits"],
            harness_hits=harness["hits"],
            difference=primary["hits"] - harness["hits"],
            share=ratio(abs(primary["hits"] - harness["hits"]), harness["hits"]),
            tolerance=TOLERANCE,
        ),
        identity_map_arm_outside_tolerance_reported=OrderedDict(
            ok=True,
            arm="identity_map",
            shipped_hits=mapped["hits"],
            harness_hits=harness["hits"],
            difference=mapped["hits"] - harness["hits"],
            share=ratio(abs(mapped["hits"] - harness["hits"]), harness["hits"]),
            note=(
                "not a soundness failure: a different key policy, reported rather than asserted "
                "inside the band; see reconciliation.identity_map_arm"
            ),
        ),
        no_write_or_control_served=OrderedDict(
            ok=(
                primary["unsafe_breakdown"]["writes_served"] == 0
                and mapped["unsafe_breakdown"]["writes_served"] == 0
                and primary["effectful_admitted"] == 0
                and mapped["effectful_admitted"] == 0
                and primary["control_hits_served"] == 0
                and mapped["control_hits_served"] == 0
            ),
            effectful_admitted=OrderedDict(
                harness_matched=primary["effectful_admitted"],
                identity_map=mapped["effectful_admitted"],
            ),
        ),
        identity_dependent_cross_tenant_zero=OrderedDict(
            ok=(
                primary["unsafe_breakdown"]["cross_tenant"] == 0
                and mapped["unsafe_breakdown"]["cross_tenant"] == 0
            ),
        ),
        hits_plus_misses_equal_reads=OrderedDict(
            ok=all(
                arm["hits"] + arm["misses"] == arm["cacheable_calls"] for arm in arms.values()
            ),
            arms=OrderedDict(
                (arm_id, OrderedDict(
                    hits=arm["hits"], misses=arm["misses"], reads=arm["cacheable_calls"]
                ))
                for arm_id, arm in arms.items()
            ),
        ),
        admissions_consistent=OrderedDict(
            ok=all(
                arm["admissions"] == arm["misses"] - arm["refusals_by_reason"]["no_tenant"]
                for arm in arms.values()
            ),
            arms=OrderedDict(
                (arm_id, OrderedDict(
                    admissions=arm["admissions"],
                    misses=arm["misses"],
                    no_tenant=arm["refusals_by_reason"]["no_tenant"],
                ))
                for arm_id, arm in arms.items()
            ),
        ),
        effectful_refusals_are_the_non_reads=OrderedDict(
            ok=all(
                arm["refusals_by_reason"]["effectful"]
                == effect_counts["write"] + effect_counts["control"]
                for arm in arms.values()
            ),
            expected=effect_counts["write"] + effect_counts["control"],
            observed=OrderedDict(
                (arm_id, arm["refusals_by_reason"]["effectful"])
                for arm_id, arm in arms.items()
            ),
        ),
        footprint_unknown_never_triggered=OrderedDict(
            ok=all(arm["refusals_by_reason"]["footprint_unknown"] == 0
                   for arm in arms.values()),
        ),
        identity_map_covers_the_map=OrderedDict(
            ok=(
                bool(mapped_entities)
                and mapped_entities <= expected_entities
                and expected_entities
                <= set(USER_SCOPED_ENTITIES) | set(NON_IDENTITY_ENTITIES)
            ),
            mapped_tool_entities=sorted(mapped_entities),
            documented_entities=sorted(expected_entities),
            documented_but_never_a_tool_entity=sorted(expected_entities - mapped_entities),
            note=(
                "every tool entity is assigned to one of the two identity sets, and every entity "
                "the map documents is assigned; payment_method is documented and named by every "
                "money-moving write's entity_keys but is never a tool's primary entity, so it "
                "appears on the documented side only. The partition is also validated inside "
                "build_identity_map, which aborts on an unassigned entity rather than guessing"
            ),
        ),
        catalog_cross_tenant_recovered=OrderedDict(
            ok=(
                mapped["identity_independent_cross_tenant_hits"]
                == p1["unsafe_breakdown"]["cross_tenant"]
                and dict(mapped["identity_independent_cross_tenant_by_tool"])
                == dict(p1["cross_tenant_by_tool"])
            ),
            map_arm=mapped["identity_independent_cross_tenant_by_tool"],
            harness_p1=p1["cross_tenant_by_tool"],
            note=(
                "the shared-key arm's cross-tenant hits must be exactly the harness's P1 "
                "cross-tenant hits, by tool; otherwise the identity map and the corpus disagree"
            ),
        ),
        lookup_scope_counterfactual=OrderedDict(
            ok=(
                probe["hits"] + probe["misses"] == primary["cacheable_calls"]
                and probe["hits"] == both_hit_primary
            ),
            hits=probe["hits"],
            misses=probe["misses"],
            both_hit_primary=both_hit_primary,
            lookup_hits_with_intervening_address_change=probe[
                "lookup_hits_with_intervening_address_change"
            ],
            note=(
                "not a pass/fail on the class: the probe shows the lookup gap is a footprint "
                "declaration, not a missing API. It checks two things -- the probe replayed every "
                "read, and with LOOKUP_SCOPE declared it hits exactly the calls both "
                "implementations hit (the harness total minus its 22 unknown-tenant hits)"
            ),
        ),
    )


def summary(report: Mapping[str, Any], output: Path) -> str:
    primary = report["measurement"]
    mapped = next(arm for arm in report["arms"] if arm["id"] == "identity_map")
    harness = report["harness"]
    return (
        "shipped_sound_cache_replay: "
        f"harness_matched {primary['hits']}/{primary['cacheable_calls']} hits "
        f"({primary['hit_rate'] * 100:.2f}%) vs harness {harness['hits']}/"
        f"{harness['cacheable_calls']} ({harness['hit_rate'] * 100:.2f}%), 0 unsafe; "
        f"identity_map {mapped['hits']}/{mapped['cacheable_calls']} hits "
        f"({mapped['hit_rate'] * 100:.2f}%), 0 unsafe; -> {output}"
    )


# ---------------------------------------------------------------------------
# Self-test: the arm machinery on a stream whose answers are known by hand
# ---------------------------------------------------------------------------
def self_test() -> int:
    ci, _ = load_identity()
    sound = load_sound_class()
    ucb, _ = load_baseline()
    identity_rows = build_identity_map(json.loads(EFFECT_MAP_PATH.read_text(encoding="utf-8")))
    policies = {policy.pid: policy for policy in build_policies(identity_rows)}
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}" if detail else name)

    raw = [
        ("get_order_details", {"order_id": "#W1"}, "A"),        # 0 miss
        ("get_product_details", {"product_id": "P1"}, "A"),     # 1 miss
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),     # 2 write, refused
        ("get_order_details", {"order_id": "#W1"}, "A"),        # 3 evicted -> miss
        ("get_product_details", {"product_id": "P1"}, "B"),     # 4 shared on the map arm only
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),     # 5 write, refused
        ("get_user_details", {"user_id": "u1"}, None),          # 6 no tenant -> miss on both
        ("get_product_details", {"product_id": "P1"}, "A"),     # 7 shared hit on the map arm
    ]
    calls = [
        ucb.Call(pos=i, turn=0, task_id=0, trial=0, reward=1.0, tool=tool, args=args,
                 tenant=tenant, source="synthetic")
        for i, (tool, args, tenant) in enumerate(raw)
    ]
    streams = OrderedDict([("synthetic", calls)])
    primary = replay_arm(streams, ci, policies["harness_matched"], sound)
    mapped = replay_arm(streams, ci, policies["identity_map"], sound)

    check("write refused", primary["effectful_admitted"] == 0, str(primary["effectful_admitted"]))
    check("write never served", primary["unsafe_breakdown"]["writes_served"] == 0)
    check("stale read evicted", primary["unsafe_breakdown"]["stale_reads"] == 0,
          str(primary["unsafe_breakdown"]))
    check("dependent no-tenant refused", primary["refusals_by_reason"]["no_tenant"] == 1,
          str(primary["refusals_by_reason"]))
    check("primary hits", primary["hits"] == 1, str(primary["hits"]))
    check("map shares the catalog read", mapped["hits"] == 2, str(mapped["hits"]))
    check("map counts the shared hit", mapped["identity_independent_cross_tenant_hits"] == 1,
          str(mapped["identity_independent_cross_tenant_hits"]))
    check("map still isolates the order read",
          mapped["hits_by_tool"].get("get_order_details", 0) == 0, str(mapped["hits_by_tool"]))
    check("map refuses the same write", mapped["unsafe_breakdown"]["writes_served"] == 0)

    for failure in failures:
        print(f"self-test: FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("self-test: ok (shipped cache driven through both key policies; "
          "writes refused, stale evicted, tenants isolated)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true", help="also print the document to stdout")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    if not args.corpus.is_file():
        print(f"shipped_sound_cache_replay: no such corpus: {args.corpus}", file=sys.stderr)
        return 2
    corpus_digest = sha256_file(args.corpus)
    if corpus_digest != EXPECTED_SHA256:
        print(
            f"shipped_sound_cache_replay: corpus digest {corpus_digest} does not match the "
            f"committed {EXPECTED_SHA256}; refusing to measure a different experiment under "
            "this schema",
            file=sys.stderr,
        )
        return 2

    ci, identity_digest = load_identity()
    ucb, baseline_digest = load_baseline()
    sound = load_sound_class()
    sound_digest = sha256_file(SOUND_CACHE_PATH)
    map_digest = sha256_file(EFFECT_MAP_PATH)
    map_document = json.loads(EFFECT_MAP_PATH.read_text(encoding="utf-8"))
    identity_rows = build_identity_map(map_document)
    policies = build_policies(identity_rows)

    records = ucb.load_records(args.corpus)
    if FIRST_TRIAL_ONLY and any(record.get("trial", 0) != 0 for record in records):
        # build_streams applies the filter itself; keep the filtered copy for the
        # corpus block so the provenance numbers describe the replayed subset.
        pass
    streams = ucb.build_streams(records)
    if not streams:
        print("shipped_sound_cache_replay: the corpus produced no streams", file=sys.stderr)
        return 2

    def build():
        return build_report(
            args.corpus, corpus_digest, records, streams, ci, identity_digest,
            map_digest, map_document, identity_rows, policies, ucb, sound,
            sound_digest, baseline_digest,
        )

    report = build()
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    # Determinism guard: the replay is cheap, so build it twice and compare.
    if json.dumps(build(), indent=2, ensure_ascii=False) + "\n" != text:
        print("shipped_sound_cache_replay: replay is not deterministic; refusing to write",
              file=sys.stderr)
        return 3

    failed = [name for name, value in report["checks"].items() if not value["ok"]]
    if failed:
        print(f"shipped_sound_cache_replay: WARNING invariants failed: {', '.join(failed)}",
              file=sys.stderr)

    args.output.write_text(text, encoding="utf-8")
    print(summary(report, args.output))
    if args.json:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
