#!/usr/bin/env python3
"""Unsafe cache baselines: what four plausible caches earn on recorded agent traces.

Motivation. Section 5.12 reports 41.70% reuse on a real tau-bench run under an
exact-key cache that serves only side-effect-free calls. A systems reader will
ask the obvious follow-up: 41.70% of *what*, and what would a cache that skipped
one of those three gates have earned instead? This script answers that by
replaying the same corpus through four caches, each a separate run-scope cache,
and counting not just the hits but the hits that are *wrong*.

The four policies differ in which of the three gates -- identity, freshness,
effect -- they implement:

  P1 blind            ``cache_identity.key_blind`` -- sha256 over tool and
                      canonical arguments, with no tenant binding, no effect
                      filter and no invalidation. Every call is stored, writes
                      included. This is the baseline the published reuse figure
                      comes from (reproduced, over the pilot's purity set, under
                      ``reconciliation.anchor``).

  P2 blind+invalidate the same key, but a write evicts the cached reads of every
                      entity it names (the module's own index and ``invalidate``
                      are reused verbatim). Staleness is now handled; writes are
                      still memoised, so a replayed mutation still never runs.

  P3 semantic-ish     tool + a *normalised* argument subset: identity arguments
                      (``user_id``, ``tenant``) are dropped and the rest is
                      stripped, de-nulled and order-normalised. This models an
                      embedding/semantic cache that believes two requests asking
                      "the same thing" are the same request. No invalidation, no
                      effect filter, so writes are memoised as well -- it is
                      identity-blind *and* freshness-blind.

  P4 typed            ``cache_identity.key_typed``: tenant-bound, canonical,
                      effect-typed; writes and control tools are refused, and a
                      write evicts the reads of the entities it names. This is
                      the paper's cache, replayed here as the safety ceiling.

Unsafe hits are counted from the replay, never assumed. A hit is unsafe when the
policy returns a memoised result for

  write       a write call (the mutation never ran, but the agent believes it
              did -- the false-success bug);
  stale read  a read whose entity was written between the moment the memo was
              stored and the hit (the memo predates the write);
  cross-tenant
              a read served by a memo that was stored while serving a different
              tenant (``user_id`` derived per record, see below).

The three classes are assigned with a fixed precedence (write > stale >
cross-tenant), so each unsafe hit is counted exactly once and the breakdown sums
to the total; the raw overlapping counts are reported alongside so the collapse
is visible. A hit on a *control* tool (``think``) is outside all three classes
and is reported separately rather than folded into the total.

Tenant model. The corpus records tool calls, not sessions, so a tenant is
recovered per record: the set of ``user_id`` values its calls name. Exactly one
distinct id -> that id is the tenant; zero or several -> tenant unknown, and a
hit involving an unknown tenant is never counted as cross-tenant. Cross-tenant
counts are therefore a lower bound, and the coverage is reported in
``reconciliation.tenant_coverage``.

Staleness model. A read hit is stale when some write that *executed* (i.e. was
not itself served from cache) names an entity id the read names, and sits
strictly between the memo's position and the hit's. Entity ids come from
``cache_identity.entity_keys``. The join is on the id text, not on a typed
token, because ``entity_keys`` is the module's public identity surface; ids from
different namespaces would only collide if they were spelled identically, which
this corpus never does, and a collision would over-evict (P2/P4) or over-count
staleness (P1/P3), never the reverse.

Two biases run in opposite directions, and both are stated rather than netted
out into a single "unsafe" claim:

  down  reads naming no entity id (``list_all_*``, ``search_*``, ``calculate``)
        cannot be joined to a write, and a tenant that cannot be recovered (49
        of 325 records name no ``user_id``, one names two) cannot be compared;
        every cross-tenant and stale count is therefore a lower bound;
  up    the corpus stores inputs only, so a write that failed is
        indistinguishable from one that landed and is treated as landed, which
        can only add staleness; symmetrically, ``invalidate`` is called on every
        write miss, failed or not, which can only remove hits.

Deterministic: no clock, no randomness, no environment reads; JSON-native args;
sorted output everywhere it is not already ordered by the corpus. Re-running at
the same commit over the same bytes produces a byte-identical file, and ``main``
actually replays twice and compares before writing.

Usage::

    ./venv/bin/python examples/benchmarks/unsafe_cache_baselines.py
    ./venv/bin/python examples/benchmarks/unsafe_cache_baselines.py \\
        --output examples/benchmarks/unsafe_cache_results.json

Consumers: ``safety_table.py`` reads ``policies[]`` and requires the fields
``name``, ``hit_rate``, ``unsafe_hits``, ``calls``, ``assumes`` and
``unsafe_breakdown.{writes_served,stale_reads,cross_tenant}``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = Path(__file__).with_name("unsafe_cache_results.json")
IDENTITY_PATH = Path(__file__).with_name("cache_identity.py")

SCHEMA = "unsafe_cache_baselines/v1"
TASK = (
    "B2 - unsafe cache baselines: four run-scope cache policies replayed over the "
    "first-trial tau-bench records, with unsafe hits counted from the replay"
)
# The committed corpus. Pinned because every published number this file
# decomposes is tied to these exact bytes; a different corpus is a different
# experiment and must not be reported under this schema.
EXPECTED_SHA256 = "3b5d4d29989ad99ed276433027d6ede37acb1197635e4a99febf44f6b5e64116"
FIRST_TRIAL_ONLY = True

# P3's notion of identity: the arguments a semantic cache would consider
# tenant-specific. Dropped from the key; still used for the staleness join,
# which reads the *original* arguments.
IDENTITY_ARG_KEYS: Tuple[str, ...] = ("user_id", "tenant")

UNSAFE_CLASSES: Tuple[str, ...] = ("writes_served", "stale_reads", "cross_tenant")


# ---------------------------------------------------------------------------
# The identity module (task B1), loaded by path so the measurement cannot drift
# from the cache it is measuring
# ---------------------------------------------------------------------------
def load_identity(path: Path = IDENTITY_PATH):
    """Import ``cache_identity.py`` and return ``(module, sha256)``.

    Imported by file path rather than ``import cache_identity`` because the
    benchmarks directory is not a package. The digest is recorded in the output:
    the module's map decides every effect class here, so a change to it must move
    the numbers and show up in provenance.
    """
    if not path.is_file():
        raise SystemExit(
            f"unsafe_cache_baselines: cannot find {path}. The key and eviction "
            "policy is imported from cache_identity.py (task B1); without it this "
            "measurement would have to re-derive the policy, which is exactly the "
            "drift the import exists to prevent."
        )
    spec = importlib.util.spec_from_file_location("cache_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("key_blind", "key_typed", "remember", "invalidate", "entity_keys",
                 "effect_class", "is_cacheable", "LOOKUP_READS", "LOOKUP_SCOPE"):
        if not hasattr(module, name):
            raise SystemExit(f"unsafe_cache_baselines: cache_identity.py has no {name!r}")
    return module, sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# ---------------------------------------------------------------------------
# Streams: one per run, calls in file order, first trial of each task
# ---------------------------------------------------------------------------
class Call:
    """One recorded ``(tool, arguments)`` pair, positioned inside its run."""

    __slots__ = ("pos", "turn", "task_id", "trial", "reward", "tool", "args", "tenant", "source")

    def __init__(self, pos, turn, task_id, trial, reward, tool, args, tenant, source):
        self.pos = pos
        self.turn = turn
        self.task_id = task_id
        self.trial = trial
        self.reward = reward
        self.tool = tool
        self.args = args
        self.tenant = tenant
        self.source = source

    def ref(self) -> Dict[str, Any]:
        return OrderedDict(
            pos=self.pos,
            turn=self.turn,
            task_id=self.task_id,
            tool=self.tool,
            args=self.args,
        )


def record_calls(record: Mapping[str, Any]) -> List[Tuple[int, str, Dict[str, Any]]]:
    out: List[Tuple[int, str, Dict[str, Any]]] = []
    for turn_index, turn in enumerate(record.get("turns") or []):
        for pair in turn:
            out.append((turn_index, str(pair[0]), dict(pair[1] or {})))
    return out


def record_tenant(record: Mapping[str, Any]) -> Optional[str]:
    """The single ``user_id`` the record's calls name, or ``None``.

    Deterministic and conservative: a record that names no user, or more than
    one, yields ``None`` rather than a guess, and calls with an unknown tenant
    are excluded from cross-tenant counting.
    """
    ids = sorted(
        {
            str(args["user_id"])
            for _turn, _tool, args in record_calls(record)
            if isinstance(args.get("user_id"), str)
        }
    )
    return ids[0] if len(ids) == 1 else None


def build_streams(records: Sequence[Mapping[str, Any]]) -> "OrderedDict[str, List[Call]]":
    streams: "OrderedDict[str, List[Call]]" = OrderedDict()
    for record in records:
        if FIRST_TRIAL_ONLY and record.get("trial", 0) != 0:
            continue
        source = str(record.get("source"))
        stream = streams.setdefault(source, [])
        tenant = record_tenant(record)
        for turn_index, tool, args in record_calls(record):
            stream.append(
                Call(
                    pos=len(stream),
                    turn=turn_index,
                    task_id=record.get("task_id"),
                    trial=record.get("trial"),
                    reward=record.get("reward"),
                    tool=tool,
                    args=args,
                    tenant=tenant,
                    source=source,
                )
            )
    return streams


# ---------------------------------------------------------------------------
# P3: the identity-blind, freshness-blind "semantic" key
# ---------------------------------------------------------------------------
def _normalise_value(value: Any) -> Any:
    """Strip strings, drop empties, order lists canonically, recurse into maps."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        out = {}
        for key in sorted(value):
            item = _normalise_value(value[key])
            if item is None or item == "" or item == [] or item == {}:
                continue
            out[key] = item
        return out
    if isinstance(value, (list, tuple)):
        items = [_normalise_value(item) for item in value]
        items = [item for item in items if item is not None and item != "" and item != [] and item != {}]
        try:
            return sorted(items, key=lambda item: json.dumps(item, sort_keys=True, default=str))
        except TypeError:  # pragma: no cover - heterogeneous JSON is still serialisable
            return items
    return value


def normalised_args(args: Mapping[str, Any]) -> Dict[str, Any]:
    """The argument subset P3 keys on: identity arguments removed, rest normalised.

    Dropping ``user_id``/``tenant`` is the whole point -- it is what makes the
    policy identity-blind -- and the normalisation models the sloppiness of a
    semantic key: whitespace does not matter, a null does not matter, and list
    order does not matter. Ids are *not* case-folded: this corpus spells each id
    one way, so lowercasing would invent collisions the data does not contain.
    """
    out: Dict[str, Any] = {}
    for key, value in args.items():
        if key in IDENTITY_ARG_KEYS:
            continue
        item = _normalise_value(value)
        if item is None or item == "" or item == [] or item == {}:
            continue
        out[key] = item
    return out


# ---------------------------------------------------------------------------
# The policies
# ---------------------------------------------------------------------------
class Policy:
    __slots__ = ("pid", "name", "key_recipe", "key_mode", "invalidation",
                 "memoises_writes", "assumes", "description")

    def __init__(self, pid, name, key_recipe, key_mode, invalidation,
                 memoises_writes, assumes, description):
        self.pid = pid
        self.name = name
        self.key_recipe = key_recipe
        self.key_mode = key_mode            # "blind" | "semantic" | "typed"
        self.invalidation = invalidation    # "none" | "entity"
        self.memoises_writes = memoises_writes
        self.assumes = assumes
        self.description = description


POLICIES: Tuple[Policy, ...] = (
    Policy(
        pid="P1",
        name="blind",
        key_recipe="cache_identity.key_blind(tool, args)",
        key_mode="blind",
        invalidation="none",
        memoises_writes=True,
        assumes="identical (tool, args) is the same question; no tenant, no effect filter, no freshness",
        description=(
            "The published exact-key baseline: sha256 over the tool and its canonical "
            "arguments, every call stored and served, nothing ever evicted."
        ),
    ),
    Policy(
        pid="P2",
        name="blind+invalidate",
        key_recipe="cache_identity.key_blind(tool, args) + entity invalidation",
        key_mode="blind",
        invalidation="entity",
        memoises_writes=True,
        assumes="identical (tool, args) is the same question, and a write makes every read of that entity obsolete",
        description=(
            "The same blind key, plus the module's own index and invalidate(): a write "
            "evicts the cached reads naming an entity it names. Freshness is handled; "
            "writes are still memoised, so a replayed mutation still never runs. Service "
            "uses the blind key while eviction uses the typed index, so two tenants that "
            "share a blind key can evict each other's entry: over-eviction, never a stale hit."
        ),
    ),
    Policy(
        pid="P3",
        name="semantic-ish",
        key_recipe="cache_identity.key_blind(tool, normalised(args)) with user_id/tenant dropped",
        key_mode="semantic",
        invalidation="none",
        memoises_writes=True,
        assumes="requests that normalise to the same non-identity arguments are the same question",
        description=(
            "An identity-blind semantic cache: the key ignores who is asking and is "
            "insensitive to whitespace, nulls and list order. No invalidation, no "
            "effect filter -- it is blind to freshness as well as identity."
        ),
    ),
    Policy(
        pid="P4",
        name="typed",
        key_recipe="cache_identity.key_typed(tool, args, tenant) + entity invalidation",
        key_mode="typed",
        invalidation="entity",
        memoises_writes=False,
        assumes="a read's identity is (tenant, tool, canonical args); writes are never cacheable",
        description=(
            "The paper's cache: tenant-bound, canonical, effect-typed keys; writes and "
            "control tools are refused outright, and writes evict the reads of the "
            "entities they name. Replayed here as the safety ceiling."
        ),
    ),
)


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def _intervening(
    write_log: Sequence[Tuple[int, Tuple[str, ...], str, Dict[str, Any]]],
    after: int,
    before: int,
    targets: frozenset,
) -> List[Dict[str, Any]]:
    out = []
    for pos, ids, tool, args in write_log:
        if after < pos < before and targets & frozenset(ids):
            out.append(OrderedDict(pos=pos, tool=tool, args=args, entity_ids=sorted(ids)))
    return out


def replay(streams: "OrderedDict[str, List[Call]]", policy: Policy, ci) -> Dict[str, Any]:
    """Replay every run through one policy and return its counts and evidence."""
    per_run: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    totals: Counter = Counter()
    by_tool: Dict[str, Counter] = {name: Counter() for name in UNSAFE_CLASSES}
    examples: Dict[str, List[Dict[str, Any]]] = {name: [] for name in UNSAFE_CLASSES}
    raw: Counter = Counter()

    for source, calls in streams.items():
        cache: Dict[str, Dict[str, Any]] = {}
        index: Dict[str, set] = {}
        # The module files and evicts *typed* keys; the served key may be blind.
        # key_map translates an evicted typed key back to what the cache holds.
        key_map: Dict[str, str] = {}
        write_log: List[Tuple[int, Tuple[str, ...], str, Dict[str, Any]]] = []
        run = Counter()

        for call in calls:
            effect = ci.effect_class(call.tool)
            run["calls"] += 1

            if not policy.memoises_writes and effect != "read":
                run["not_admitted"] += 1
                if effect == "write":
                    _evict(ci, index, key_map, cache, call)
                    write_log.append((call.pos, ci.entity_keys(call.tool, call.args), call.tool, call.args))
                continue

            run["cacheable"] += 1
            key = _served_key(ci, policy, call)
            memo = cache.get(key)
            if memo is not None:
                run["hits"] += 1
                if effect == "write":
                    raw["writes_served"] += 1
                    _record(run, by_tool, examples, "writes_served", call, memo, source, OrderedDict())
                elif effect == "control":
                    run["control_hits"] += 1
                else:
                    targets = frozenset(ci.entity_keys(call.tool, call.args)) or frozenset(memo["entity_ids"])
                    stale = _intervening(write_log, memo["pos"], call.pos, targets)
                    cross = (
                        memo["tenant"] is not None
                        and call.tenant is not None
                        and memo["tenant"] != call.tenant
                    )
                    if stale:
                        raw["stale_reads"] += 1
                    if cross:
                        raw["cross_tenant"] += 1
                    if stale:
                        _record(run, by_tool, examples, "stale_reads", call, memo, source,
                                OrderedDict(intervening=stale))
                    elif cross:
                        _record(run, by_tool, examples, "cross_tenant", call, memo, source,
                                OrderedDict(cached_tenant=memo["tenant"], hit_tenant=call.tenant))
                continue

            cache[key] = OrderedDict(
                pos=call.pos,
                tool=call.tool,
                args=call.args,
                tenant=call.tenant,
                entity_ids=ci.entity_keys(call.tool, call.args),
            )
            if policy.key_mode == "typed":
                typed = ci.remember(index, call.tool, call.args, tenant=call.tenant)
                key_map[typed] = key
            elif policy.invalidation == "entity" and effect == "read":
                # File the read through the module's own remember() so the index
                # is built exactly as the module specifies, then translate the
                # typed key it returns back to the blind key this cache serves.
                # Eviction therefore keeps the module's semantics (entity tokens,
                # LOOKUP_SCOPE, full flush for a write that names nothing) while
                # the hit path stays key_blind.
                typed = ci.remember(index, call.tool, call.args, tenant=call.tenant)
                key_map[typed] = key
            if effect == "write":
                write_log.append((call.pos, ci.entity_keys(call.tool, call.args), call.tool, call.args))
                if policy.invalidation == "entity":
                    _evict(ci, index, key_map, cache, call)

        totals.update(run)
        per_run[source] = OrderedDict(
            calls=run["calls"],
            cacheable_calls=run["cacheable"],
            hits=run["hits"],
            hit_rate=ratio(run["hits"], run["cacheable"]),
            unsafe_hits=run["unsafe_hits"],
            writes_served=run["writes_served"],
            stale_reads=run["stale_reads"],
            cross_tenant=run["cross_tenant"],
            control_hits=run["control_hits"],
        )

    unsafe_hits = totals["writes_served"] + totals["stale_reads"] + totals["cross_tenant"]
    return OrderedDict(
        id=policy.pid,
        name=policy.name,
        key=policy.key_recipe,
        invalidation=policy.invalidation,
        effect_filter="none (memoises writes and control tools)" if policy.memoises_writes else "reads only",
        memoises_writes=policy.memoises_writes,
        assumes=policy.assumes,
        description=policy.description,
        calls_total=totals["calls"],
        cacheable_calls=totals["cacheable"],
        # ``calls`` is the denominator the consumer reads (safety_table.py's
        # per-1,000 figure). It is the cacheable count, not every replayed call,
        # so the two bases are kept explicit rather than silently interchanged.
        calls=totals["cacheable"],
        hits=totals["hits"],
        hit_rate=ratio(totals["hits"], totals["cacheable"]),
        hit_rate_of_all_calls=ratio(totals["hits"], totals["calls"]),
        unsafe_hits=unsafe_hits,
        unsafe_breakdown=OrderedDict(
            writes_served=totals["writes_served"],
            stale_reads=totals["stale_reads"],
            cross_tenant=totals["cross_tenant"],
        ),
        unsafe_share_of_hits=ratio(unsafe_hits, totals["hits"]),
        unsafe_by_class_overlapping=OrderedDict(
            writes_served=raw["writes_served"],
            stale_reads=raw["stale_reads"],
            cross_tenant=raw["cross_tenant"],
        ),
        control_hits_not_counted=totals["control_hits"],
        not_admitted_calls=totals["not_admitted"],
        unsafe_by_tool=OrderedDict(
            (name, OrderedDict(sorted(by_tool[name].items()))) for name in UNSAFE_CLASSES
        ),
        per_run=per_run,
        examples=[examples[name][0] for name in UNSAFE_CLASSES if examples[name]],
    )


def _evict(ci, index, key_map, cache, call: Call) -> None:
    """Drop the cache entries a write obsoletes, via the module's own invalidate()."""
    for typed_key in ci.invalidate(index, call.tool, call.args):
        served = key_map.pop(typed_key, None)
        if served is not None:
            cache.pop(served, None)


def _served_key(ci, policy: Policy, call: Call) -> str:
    if policy.key_mode == "typed":
        return ci.key_typed(call.tool, call.args, tenant=call.tenant)
    if policy.key_mode == "semantic":
        return ci.key_blind(call.tool, normalised_args(call.args))
    return ci.key_blind(call.tool, call.args)


def _record(run, by_tool, examples, cls, call, memo, source, detail) -> None:
    run[cls] += 1
    run["unsafe_hits"] += 1
    by_tool[cls][call.tool] += 1
    if len(examples[cls]) < 1:
        entry = OrderedDict(
            cls=cls,
            source=source,
            hit=call.ref(),
            cached=OrderedDict(pos=memo["pos"], tool=memo["tool"], args=memo["args"]),
        )
        entry.update(detail)
        examples[cls].append(entry)


# ---------------------------------------------------------------------------
# Anchor: does this harness reproduce the published reuse figure?
# ---------------------------------------------------------------------------
def _load_pilot(path: Path):
    spec = importlib.util.spec_from_file_location("_unsafe_pilot", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def anchor(streams: "OrderedDict[str, List[Call]]", ci) -> Dict[str, Any]:
    """Replay P1's key/scope/purity exactly as Section 5.12 measures it.

    The published 41.70% is hits / side-effect-free calls under ``key_blind`` at
    run scope over first trials. Reproducing it here, through the same replay
    code path that the four policies use, is what licenses reading the policy
    comparison as a decomposition of that number rather than a different
    experiment. The purity set is the pilot's (``real_trace_pilot.is_pure``,
    which counts ``think``); this script's policies use ``cache_identity``'s
    effect map, which classes ``think`` as control -- the counts differ only by
    that, and the difference is reported.
    """
    pilot = _load_pilot(REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py")
    hits = cacheable = 0
    think_hits = 0
    for calls in streams.values():
        seen = set()
        for call in calls:
            if not pilot.is_pure(call.tool):
                continue
            cacheable += 1
            key = ci.key_blind(call.tool, call.args)
            if key in seen:
                hits += 1
                if ci.effect_class(call.tool) == "control":
                    think_hits += 1
            else:
                seen.add(key)
    return OrderedDict(
        published_hits=746,
        published_cacheable=1789,
        published_hit_rate=0.4170,
        reproduced_hits=hits,
        reproduced_cacheable=cacheable,
        reproduced_hit_rate=ratio(hits, cacheable),
        matches=hits == 746 and cacheable == 1789,
        control_hits_in_the_purity_set=think_hits,
        why_this_matters=(
            "the four policies below are the same replay with one gate removed at a time; this shows "
            "the harness reproduces the committed figure before it is used to decompose it"
        ),
        pilot_path="examples/benchmarks/real_trace_pilot.py",
        pilot_sha256=sha256_file(REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py"),
    )


# ---------------------------------------------------------------------------
# Provenance, definitions, reconciliation
# ---------------------------------------------------------------------------
def _policy_table() -> List[Dict[str, Any]]:
    return [
        OrderedDict(
            id=policy.pid,
            name=policy.name,
            key=policy.key_recipe,
            invalidation=policy.invalidation,
            effect_filter="none" if policy.memoises_writes else "reads only",
            memoises_writes=policy.memoises_writes,
            assumes=policy.assumes,
        )
        for policy in POLICIES
    ]


def definitions() -> Dict[str, Any]:
    return OrderedDict(
        hit=(
            "a call whose key is already in the policy's run-scope cache; the memo's value is "
            "returned and the tool does not run"
        ),
        cacheable_call=(
            "a call the policy admits to its cache: every call for P1-P3, reads only for P4 "
            "(writes and control tools are refused before the lookup). Hit rate is hits / cacheable calls."
        ),
        writes_served=(
            "an unsafe hit where the call is a write: the mutation is answered from a memo and never "
            "runs, while the agent's belief that it ran is unchanged"
        ),
        stale_read=(
            "an unsafe hit where the call is a read and some write that executed names an entity id the "
            "read names, at a position strictly between the memo's store position and the hit; the memo "
            "predates the write"
        ),
        cross_tenant=(
            "an unsafe hit where the call is a read, the memo was stored under a different known tenant, "
            "and the hit is not already stale; tenant is the single user_id named by the record, and an "
            "unknown tenant on either side is never counted. The class is defined by provenance, not by "
            "demonstrated damage: a cache cannot tell a tenant-agnostic catalog read (a product row, a "
            "flight search) from a tenant-scoped one, so both are counted; unsafe_by_tool.cross_tenant "
            "shows how much of the class is catalog traffic"
        ),
        unsafe_hit=(
            "the union of writes_served, stale_reads and cross_tenant, each hit assigned to exactly one "
            "class by the precedence writes_served > stale_reads > cross_tenant, so the breakdown sums "
            "to the total; unsafe_by_class_overlapping reports the pre-precedence counts where a hit "
            "can be in more than one class"
        ),
        control_hits=(
            "hits on control tools (think, transfer_to_human_agents). They are memoised by P1-P3 and "
            "refused by P4, but they are not one of the three defined unsafe classes and are reported "
            "separately rather than counted in unsafe_hits"
        ),
        tenant=(
            "the single distinct user_id named by a record's calls, else unknown. Cross-tenant counts "
            "are a lower bound: 49 of 325 records name no user_id and one names two"
        ),
        hit_rate=(
            "hits / cacheable_calls, rounded to 4 decimals; hit_rate_of_all_calls uses every replayed "
            "call as the denominator and is given so the four policies can be compared on one base"
        ),
        not_counted=(
            "reads naming no entity id (list_all_*, search_*, calculate) cannot be joined to a write; "
            "unknown tenants cannot be compared; a failed write is indistinguishable from a landed one "
            "in this corpus, so every replayed write is treated as having landed"
        ),
    )


def reconciliation(
    streams: "OrderedDict[str, List[Call]]",
    policies: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    ci,
) -> Dict[str, Any]:
    """Independent cross-checks that do not go through the replay's own code path."""
    calls_by_effect = Counter()
    calls_with_tenant = 0
    calls_total = 0
    anchor_result = anchor(streams, ci)
    for calls in streams.values():
        for call in calls:
            calls_total += 1
            calls_by_effect[ci.effect_class(call.tool)] += 1
            if call.tenant is not None:
                calls_with_tenant += 1

    # A write served from a never-invalidating cache is exactly a repeated write
    # key inside one run. Count that straight from the corpus and compare.
    repeated_write_keys = 0
    for calls in streams.values():
        seen = set()
        for call in calls:
            if ci.effect_class(call.tool) != "write":
                continue
            key = ci.key_blind(call.tool, call.args)
            if key in seen:
                repeated_write_keys += 1
            else:
                seen.add(key)

    tenant_counts = Counter()
    for record in records:
        if record.get("trial", 0) != 0:
            continue
        ids = {
            str(args["user_id"])
            for _turn, _tool, args in record_calls(record)
            if isinstance(args.get("user_id"), str)
        }
        tenant_counts[len(ids)] += 1

    by_id = {policy["id"]: policy for policy in policies}
    blind = by_id["P1"]
    invalidate = by_id["P2"]
    typed = by_id["P4"]
    return OrderedDict(
        anchor=anchor_result,
        calls_total=calls_total,
        calls_by_effect=OrderedDict(sorted(calls_by_effect.items())),
        runs=len(streams),
        records=sum(1 for record in records if record.get("trial", 0) == 0),
        tenant_coverage=OrderedDict(
            calls_with_known_tenant=calls_with_tenant,
            calls_total=calls_total,
            share=ratio(calls_with_tenant, calls_total),
            records_by_distinct_user_id=OrderedDict(
                (str(k), tenant_counts[k]) for k in sorted(tenant_counts)
            ),
        ),
        write_repeats_independent=OrderedDict(
            repeated_write_keys_in_run=repeated_write_keys,
            blind_writes_served=blind["unsafe_breakdown"]["writes_served"],
            invalidate_writes_served=invalidate["unsafe_breakdown"]["writes_served"],
            matches_blind=repeated_write_keys == blind["unsafe_breakdown"]["writes_served"],
            matches_invalidate=repeated_write_keys == invalidate["unsafe_breakdown"]["writes_served"],
        ),
        invalidation_soundness=OrderedDict(
            invalidate_stale_reads=invalidate["unsafe_breakdown"]["stale_reads"],
            invalidate_stale_reads_is_zero=invalidate["unsafe_breakdown"]["stale_reads"] == 0,
            typed_unsafe_hits=typed["unsafe_hits"],
            typed_unsafe_hits_is_zero=typed["unsafe_hits"] == 0,
            note=(
                "P2 and P4 evict on every write they observe, so a surviving stale read would mean the "
                "index missed an entity the write named; both are reported rather than asserted, because "
                "a nonzero value is a finding about the corpus or the map, not a reason to hide the run"
            ),
        ),
        breakdown_sums=OrderedDict(
            (policy["id"], OrderedDict(
                total=policy["unsafe_hits"],
                parts=sum(policy["unsafe_breakdown"].values()),
                consistent=policy["unsafe_hits"] == sum(policy["unsafe_breakdown"].values()),
            ))
            for policy in policies
        ),
    )


def checks(policies: Sequence[Mapping[str, Any]], recon: Mapping[str, Any]) -> Dict[str, Any]:
    by_id = {policy["id"]: policy for policy in policies}
    return OrderedDict(
        four_policies=OrderedDict(count=len(policies), ok=len(policies) == 4),
        anchor_reproduces_published=OrderedDict(ok=bool(recon["anchor"]["matches"])),
        breakdown_sums=OrderedDict(
            ok=all(item["consistent"] for item in recon["breakdown_sums"].values())
        ),
        write_repeat_match=OrderedDict(
            ok=bool(recon["write_repeats_independent"]["matches_blind"])
            and bool(recon["write_repeats_independent"]["matches_invalidate"])
        ),
        typed_never_unsafe=OrderedDict(ok=by_id["P4"]["unsafe_hits"] == 0),
        invalidate_never_stale=OrderedDict(ok=by_id["P2"]["unsafe_breakdown"]["stale_reads"] == 0),
        typed_never_serves_writes=OrderedDict(ok=by_id["P4"]["unsafe_breakdown"]["writes_served"] == 0),
    )


def build_report(corpus: Path, corpus_digest: str, records, streams, ci, identity_digest: str) -> Dict[str, Any]:
    policies = [replay(streams, policy, ci) for policy in POLICIES]
    recon = reconciliation(streams, policies, records, ci)
    return OrderedDict(
        schema=SCHEMA,
        task=TASK,
        result_kind="REPLAY",
        scope=(
            "tau-bench only: the first trial of every task in the committed corpus, four source runs, "
            "replayed through four run-scope caches. Unsafe hits are counted from the replay, not "
            "estimated: a hit is unsafe when it serves a write, a read whose entity was written after "
            "the memo was stored, or a read produced under a different tenant."
        ),
        corpus_sha256=corpus_digest,
        provenance=OrderedDict(
            script=OrderedDict(
                path=str(Path(__file__).resolve().relative_to(REPO_ROOT)),
                sha256=sha256_file(Path(__file__).resolve()),
                schema=SCHEMA,
            ),
            corpus=OrderedDict(
                path=str(corpus.resolve().relative_to(REPO_ROOT)),
                sha256=corpus_digest,
                records=recon["records"],
                calls=recon["calls_total"],
                runs=recon["runs"],
                first_trial_only=FIRST_TRIAL_ONLY,
                expected_sha256=EXPECTED_SHA256,
                matches_expected=corpus_digest == EXPECTED_SHA256,
            ),
            identity_module=OrderedDict(
                path=str(IDENTITY_PATH.resolve().relative_to(REPO_ROOT)),
                sha256=identity_digest,
                effect_map_source=ci.MAP_SOURCE,
                effect_map_path=(
                    str(ci.MAP_PATH.resolve().relative_to(REPO_ROOT)) if ci.MAP_PATH is not None else None
                ),
                effect_map_error=ci.MAP_ERROR,
                tools=len(ci.EFFECT_CLASSES),
                classes=OrderedDict(
                    (cls, sorted(name for name, value in ci.EFFECT_CLASSES.items() if value == cls))
                    for cls in ("read", "write", "control")
                ),
                why_imported=(
                    "the key recipes, the effect classes and the eviction index are the module's, not "
                    "this script's; re-deriving them here is how a baseline drifts from the cache it "
                    "claims to measure"
                ),
            ),
            anchor_module=OrderedDict(
                path="examples/benchmarks/real_trace_pilot.py",
                sha256=sha256_file(REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py"),
                why=(
                    "its is_pure() is the published purity set, imported only to reproduce Section "
                    "5.12's 746/1789 anchor; the four policies use cache_identity's effect map"
                ),
            ),
            policy_table=_policy_table(),
            determinism=(
                "no clock, no environment, no randomness; JSON-native arguments; every collection "
                "emitted in a stable order; main() replays twice and refuses to write if the two "
                "documents differ"
            ),
            arithmetic="integer counts; shares are round(numerator/denominator, 4); no weighted averages",
        ),
        definitions=definitions(),
        policies=policies,
        reconciliation=recon,
        checks=checks(policies, recon),
    )


# ---------------------------------------------------------------------------
# Self-test: the invariants, on a stream whose answers are known by hand
# ---------------------------------------------------------------------------
def self_test() -> int:
    ci, _digest = load_identity()
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}" if detail else name)

    # 9 calls, one run, two tenants. Tenant A: calls 0-3, 5, 6, 8; tenant B: 4, 7.
    raw = [
        ("get_order_details", {"order_id": "#W1"}, "A"),        # 0 miss
        ("get_product_details", {"product_id": "P1"}, "A"),     # 1 miss
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),     # 2 write, miss
        ("get_order_details", {"order_id": "#W1"}, "A"),        # 3 P1/P3 stale, P2/P4 evicted
        ("get_product_details", {"product_id": "P1"}, "B"),     # 4 tenant-B read
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),     # 5 P1/P2/P3 write hit
        ("get_user_details", {"user_id": "u1"}, "A"),           # 6 miss
        ("get_user_details", {"user_id": "u2"}, "B"),           # 7 P3 hit via dropped id
        ("get_product_details", {"product_id": "P1"}, "A"),     # 8 P4 hit, same tenant as 1
    ]
    calls = [
        Call(pos=i, turn=0, task_id=0, trial=0, reward=1.0, tool=tool, args=args, tenant=tenant, source="synthetic")
        for i, (tool, args, tenant) in enumerate(raw)
    ]
    streams: "OrderedDict[str, List[Call]]" = OrderedDict([("synthetic", calls)])
    results = {policy.pid: replay(streams, policy, ci) for policy in POLICIES}

    # P4 refuses writes: key_typed itself must raise for one.
    try:
        ci.key_typed("cancel_pending_order", {"order_id": "#W1"}, tenant="A")
        check("typed refuses writes", False, "key_typed accepted a write")
    except ValueError:
        pass

    check("P1 write hit counted", results["P1"]["unsafe_breakdown"]["writes_served"] == 1,
          str(results["P1"]["unsafe_breakdown"]))
    check("P1 stale read counted", results["P1"]["unsafe_breakdown"]["stale_reads"] == 1,
          str(results["P1"]["unsafe_breakdown"]))
    check("P1 cross-tenant counted", results["P1"]["unsafe_breakdown"]["cross_tenant"] == 1,
          str(results["P1"]["unsafe_breakdown"]))
    check("P1 total is the union", results["P1"]["unsafe_hits"] == 3)
    check("P1 hits", results["P1"]["hits"] == 4, str(results["P1"]["hits"]))

    check("P2 evicts the stale read", results["P2"]["unsafe_breakdown"]["stale_reads"] == 0,
          str(results["P2"]["unsafe_breakdown"]))
    check("P2 still serves the write", results["P2"]["unsafe_breakdown"]["writes_served"] == 1)
    check("P2 hits", results["P2"]["hits"] == 3, str(results["P2"]["hits"]))

    check("P3 is identity-blind", results["P3"]["unsafe_breakdown"]["cross_tenant"] >= 2,
          str(results["P3"]["unsafe_breakdown"]))
    check("P3 keeps staleness", results["P3"]["unsafe_breakdown"]["stale_reads"] == 1)

    check("P4 is safe on the test stream", results["P4"]["unsafe_hits"] == 0,
          str(results["P4"]["unsafe_breakdown"]))
    check("P4 refuses writes", results["P4"]["unsafe_breakdown"]["writes_served"] == 0)
    check("P4 admits only reads", results["P4"]["cacheable_calls"] == 7, str(results["P4"]["cacheable_calls"]))
    check("P4 still hits same-tenant repeats", results["P4"]["hits"] == 1, str(results["P4"]["hits"]))

    # A blind key is argument-bound but tenant-free; a typed key is both.
    a = {"order_id": "#W1"}
    b = {"order_id": "#W2"}
    check("blind key separates arguments",
          ci.key_blind("get_order_details", a) != ci.key_blind("get_order_details", b))
    check("blind key ignores the tool",
          ci.key_blind("get_order_details", a) != ci.key_blind("get_user_details", a))
    check("typed key is tenant-bound",
          ci.key_typed("get_product_details", {"product_id": "P1"}, tenant="A")
          != ci.key_typed("get_product_details", {"product_id": "P1"}, tenant="B"))

    # P3's normaliser really drops identity and whitespace.
    check("P3 drops identity",
          normalised_args({"user_id": "u1", "zip": " 8020 "}) == {"zip": "8020"},
          str(normalised_args({"user_id": "u1", "zip": " 8020 "})))

    for failure in failures:
        print(f"self-test: FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("self-test: ok (4 policies, 9 synthetic calls, unsafe classes counted by replay)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summary(report: Mapping[str, Any], output: Path) -> str:
    parts = []
    for policy in report["policies"]:
        parts.append(
            f"{policy['name']} {policy['hits']}/{policy['cacheable_calls']} hits "
            f"({policy['hit_rate'] * 100:.2f}%), {policy['unsafe_hits']} unsafe"
        )
    return "unsafe_cache_baselines: " + "; ".join(parts) + f" -> {output}"


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
        print(f"unsafe_cache_baselines: no such corpus: {args.corpus}", file=sys.stderr)
        return 2

    corpus_digest = sha256_file(args.corpus)
    if corpus_digest != EXPECTED_SHA256:
        print(
            f"unsafe_cache_baselines: corpus digest {corpus_digest} does not match the committed "
            f"{EXPECTED_SHA256}; refusing to measure a different experiment under this schema",
            file=sys.stderr,
        )
        return 2

    ci, identity_digest = load_identity()
    records = load_records(args.corpus)
    streams = build_streams(records)

    report = build_report(args.corpus, corpus_digest, records, streams, ci, identity_digest)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    # Determinism guard: the whole replay is cheap, so run it twice and compare.
    again = build_report(args.corpus, corpus_digest, records, build_streams(records),
                         ci, load_identity()[1])
    if json.dumps(again, indent=2, ensure_ascii=False) + "\n" != text:
        print("unsafe_cache_baselines: replay is not deterministic; refusing to write",
              file=sys.stderr)
        return 3

    failed = [name for name, value in report["checks"].items() if not value["ok"]]
    if failed:
        print(f"unsafe_cache_baselines: WARNING invariants failed: {', '.join(failed)}", file=sys.stderr)

    args.output.write_text(text, encoding="utf-8")
    line = summary(report, args.output)
    print(line)
    if args.json:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
