#!/usr/bin/env python3
"""Cost-of-soundness ladder: what each soundness ingredient costs, one rung at a time.

Motivation. The four committed policies replay the same corpus through four
caches, but they differ in three things at once -- which calls are admitted, what
the key binds, and whether writes evict what they falsified -- so their hit rates
are not comparable. "The typed cache hits 26.97% and the blind one hits 42.63%"
invites the reading that soundness costs sixteen points, when part of the gap is
only that the typed cache refuses 547 calls the blind one stores. This script
removes that confound by building a *nested* ladder on a single admitted set.

The admitted set. ``cache_identity`` admits exactly the read tools (its effect map
classes 11 tools as read, 13 as write and 2 as control), which is 1,728 of the
corpus's 2,275 first-trial calls. Every rung of the main ladder scores the same
1,728 calls with the same streams, the same run scope and the same hit/unsafe
definitions; only the key and the eviction policy move. A difference between two
adjacent rungs is therefore attributable to the one ingredient that changed, and
"each ingredient costs X points of hit rate and removes Y unsafe hits" is a claim
about a measured pair, not a decomposition by assumption.

The main ladder, on the 1,728 admitted reads:

  L0  blind key              ``cache_identity.key_blind``: sha256 over the tool and
                             its canonical arguments. No tenant, no invalidation.
                             This is the published 41.70% reuse recipe's identity
                             notion, now restricted to reads by the admitted set.
  L1  + tenant in the key    ``cache_identity.key_typed(tool, args, tenant)``: the
                             tenant enters the digest, so tenant A's row cannot
                             answer tenant B's question. The typed key also carries
                             a domain tag and refuses non-reads; on a reads-only
                             admitted set neither can change a hit, and
                             ``checks.tenant_ingredient_isolated`` demonstrates it
                             rather than asserting it (key_typed with
                             ``tenant=None`` reproduces L0's hit count exactly).
  L2  + entity invalidation  the module's own ``remember``/``invalidate``: a stored
                             read is filed under every entity id its arguments
                             name, and a write that lands evicts the keys filed
                             under the ids it names. A surviving key therefore
                             names no entity the write named.
  L3  + read-only admission  *already given.* The 1,728-read denominator is the
                             typed policy's admission decision, made once before
                             the ladder starts, so L3 adds nothing to L0-L2 and
                             must reproduce L2 exactly. That is the point of the
                             rung: the admission gate's cost cannot be read off a
                             ladder whose denominator it defines, which is why the
                             mirror ladder below prices it separately.

The mirror ladder, on all 2,275 calls (446 writes and 101 control calls admitted
too), adds the same ingredients in the same order. Its first three rungs must key
writes as well as reads, so they use the blind recipe with the tenant folded into
the identity prefix (``key_tenant_blind``) -- a cache that admits writes cannot
have a typed key, because ``key_typed`` refuses to produce one. Its last rung adds
read-only admission, which is exactly L2 measured on the 2,275 denominator. The
gap M2 -> M3 is the price of admitting writes: the extra hits they buy, and the
unsafe hits (mutations answered from cache and never run) that come with them and
that no key or eviction policy can remove. A repeated write's key is never filed
in the index -- only reads are -- so eviction cannot evict a write's own memo.

Unsafe hits are counted from the replay, never assumed, under the same three
classes and the same precedence the committed baselines use:

  writes_served  a write call answered from a memo (the mutation never ran);
  stale_read     a read whose entity was written by an executing write at a
                 position strictly between the memo's store position and the hit;
  cross_tenant   a read served by a memo stored under a different known tenant.

Precedence is writes_served > stale_read > cross_tenant, so each unsafe hit is
counted once and the breakdown sums to the total. A hit on a control tool
(``think``, ``transfer_to_human_agents``) is reported separately, not as unsafe:
it is not a data answer and not one of the three defined classes.

What this ladder does not claim. It is a replay of recorded inputs, not of
responses: staleness is judged by the writes the cache *observed*, so it is
relative to the corpus, and a write that failed is treated as landed (which can
only manufacture staleness, never hide it). Entity joins use the module's
identity subset, so a read that names no id (``list_all_*``, ``search_*``,
``calculate``) cannot be joined to a write and a stale hit there would go
uncounted. Tenants are recovered per record (the single ``user_id`` its calls
name, else unknown, 49 of 325 records), so cross-tenant counts are a lower bound.
Every one of these gaps is a reason for the ladder to understate unsoundness, and
the zero at L2/M3 is a statement about this corpus and this policy, not a
theorem.

Deterministic: no clock, no environment, no randomness, no git lookup; JSON-native
arguments; every collection emitted in a stable order; every rung is replayed
twice and the two documents compared before writing. Rates are integer ratios
rounded to 4 decimals, point deltas to 2, and nothing is weighted or averaged.

Usage::

    ./venv/bin/python examples/benchmarks/soundness_ladder.py --self-test
    ./venv/bin/python examples/benchmarks/soundness_ladder.py \\
        --output examples/benchmarks/soundness_ladder_results.json

Consumers: the paper's cost-of-soundness paragraph reads
``ladders.main.marginal_costs[]`` and ``ladders.mirror.marginal_costs[]``; every
number quoted there is a field in this document.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = Path(__file__).with_name("soundness_ladder_results.json")
IDENTITY_PATH = Path(__file__).with_name("cache_identity.py")

SCHEMA = "soundness_ladder/v1"
TASK = (
    "L1 - cost-of-soundness ladder: one admitted set (the 1,728 reads cache_identity "
    "admits), one ingredient per rung, with hits, hit rate and unsafe hits measured "
    "from the replay"
)
# The committed corpus, pinned: every published number this ladder decomposes is
# tied to these bytes, and a different corpus is a different experiment.
EXPECTED_SHA256 = "3b5d4d29989ad99ed276433027d6ede37acb1197635e4a99febf44f6b5e64116"
FIRST_TRIAL_ONLY = True

# The admitted set, asserted rather than assumed: 1,728 reads out of 2,275 calls.
EXPECTED_CALLS = 2275
EXPECTED_ADMITTED_READS = 1728

# The committed P4 replay (unsafe_cache_results.json) hits 466 of 1,728 with zero
# unsafe. L3 is that policy, so the ladder must land on the same number: if this
# drifts, either the ladder or the committed baseline changed and a paper sentence
# is now quoting a different cache.
EXPECTED_TYPED_HITS = 466

UNSAFE_CLASSES: Tuple[str, ...] = ("writes_served", "stale_reads", "cross_tenant")


# ---------------------------------------------------------------------------
# Imports and receipts
# ---------------------------------------------------------------------------
def load_identity(path: Path = IDENTITY_PATH):
    """Import ``cache_identity.py`` and return ``(module, sha256)``.

    Imported by file path rather than ``import cache_identity`` because the
    benchmarks directory is not a package. The digest is recorded: the module's
    effect map decides the admitted set, its key recipes produce every digest on
    the ladder, and its index performs every eviction, so a change to it must move
    these numbers and show up in provenance.
    """
    if not path.is_file():
        raise SystemExit(
            f"soundness_ladder: cannot find {path}. The effect map, the key recipes "
            "and the eviction index are imported from cache_identity.py; without it "
            "the ladder would have to re-derive the policy it is measuring."
        )
    spec = importlib.util.spec_from_file_location("cache_identity", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ("key_blind", "key_typed", "remember", "invalidate", "entity_keys",
                 "effect_class", "is_cacheable"):
        if not hasattr(module, name):
            raise SystemExit(f"soundness_ladder: cache_identity.py has no {name!r}")
    return module, sha256_file(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def unsafe_hits_text(count: int) -> str:
    """``"1 unsafe hit"`` / ``"168 unsafe hits"`` -- the sentences are read aloud."""
    return f"{count} unsafe hit" + ("" if count == 1 else "s")


def points(before: float, after: float) -> float:
    """Percentage points lost, from two rates, rounded to 2 decimals."""
    return round((before - after) * 100.0, 2)


# ---------------------------------------------------------------------------
# Streams: one per run, calls in file order, first trial of each task
#
# Identical to the committed baselines' scope so the rungs are comparable with
# them and with the published 41.70%.
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

    Deterministic and conservative: a record that names no user, or more than one,
    yields ``None``. A call with an unknown tenant is never counted as
    cross-tenant, because "different" needs two known values.
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
# Keys
# ---------------------------------------------------------------------------
def key_tenant_blind(ci_mod, tool: str, args: Mapping[str, Any],
                     tenant: Optional[str]) -> str:
    """The blind recipe with the tenant folded into its identity prefix.

    ``cache_identity.key_blind(tool, args)`` is ``sha256("tool|" + canonical(args))``
    -- the published recipe, and the one measured at L0. A cache that admits writes
    needs a tenant-bound key for *writes* too, and ``key_typed`` refuses to produce
    one (it is defined only for reads: memoising a write is the bug). So the
    mirror's write-admitting rungs reuse the module's blind recipe verbatim with
    the tenant inside the ``tool`` slot, serialised as ``[tenant, tool]``.

    Soundness. ``json.dumps`` of a two-element list is injective in ``(tenant,
    tool)``: no tenant can spell another tenant's prefix, and ``None`` and the
    string ``"None"`` differ. The arguments are passed through untouched, so the
    key still answers "which question" with the same canonical form as ``key_blind``
    and the same digest equality classes, only finer. On a reads-only stream it
    partitions the calls exactly as ``key_typed`` does -- both are injective in
    ``(tenant, tool, args)`` -- which is what
    ``checks.mirror_key_agrees_with_typed_on_reads`` measures. Its defect is
    inherited unchanged: it will memoise a mutation, which is why it exists only on
    rungs that are meant to.
    """
    identity = json.dumps([tenant, tool])
    return ci_mod.key_blind(identity, args)


def served_key(rung: "Rung", call: Call, ci_mod) -> str:
    if rung.key_mode == "typed":
        return ci_mod.key_typed(call.tool, call.args, tenant=call.tenant)
    if rung.key_mode == "typed_none":
        # Reconciliation only: the typed recipe with the tenant held at None, to
        # show that the recipe swap on its own moves no hit.
        return ci_mod.key_typed(call.tool, call.args, tenant=None)
    if rung.key_mode == "blind_tenant":
        return key_tenant_blind(ci_mod, call.tool, call.args, call.tenant)
    return ci_mod.key_blind(call.tool, call.args)


# ---------------------------------------------------------------------------
# The rungs
# ---------------------------------------------------------------------------
class Rung:
    """One point on a ladder: which calls are admitted, what the key binds, what evicts."""

    __slots__ = ("id", "label", "ingredient", "key_recipe", "key_mode", "invalidation",
                 "admission", "key_binds_tenant", "removes", "note")

    def __init__(self, rid, label, ingredient, key_recipe, key_mode, invalidation,
                 admission, key_binds_tenant, removes=None, note=""):
        self.id = rid
        self.label = label
        self.ingredient = ingredient
        self.key_recipe = key_recipe
        self.key_mode = key_mode              # "blind" | "blind_tenant" | "typed"
        self.invalidation = invalidation      # bool: entity invalidation
        self.admission = admission            # "reads" | "all"
        self.key_binds_tenant = key_binds_tenant
        # The unsafe class this rung's ingredient is meant to remove. Any other
        # class it also removes is reported as an off-target effect, not folded in.
        self.removes = removes
        self.note = note

    def admits(self, effect: str) -> bool:
        return self.admission == "all" or effect == "read"


_READS_ONLY = (
    "reads only, the ladder's admitted set: the 1,728 calls cache_identity admits "
    "(given before rung 0, so identical at every rung)"
)

MAIN_RUNGS: Tuple[Rung, ...] = (
    Rung(
        rid="L0",
        label="blind key",
        ingredient="none -- the baseline",
        key_recipe="cache_identity.key_blind(tool, args)",
        key_mode="blind",
        invalidation=False,
        admission="reads",
        key_binds_tenant=False,
        removes=None,
        note=(
            "sha256 over the tool and its canonical arguments: the published reuse "
            "recipe's identity notion. A hit is always the same question, and says "
            "nothing about who asked or whether the answer still holds."
        ),
    ),
    Rung(
        rid="L1",
        label="blind key + tenant",
        ingredient="tenant in the key",
        key_recipe="cache_identity.key_typed(tool, args, tenant)",
        key_mode="typed",
        invalidation=False,
        admission="reads",
        key_binds_tenant=True,
        removes="cross_tenant",
        note=(
            "the tenant enters the digest, so one tenant cannot be answered with "
            "another's row. key_typed also domain-separates and refuses non-reads; "
            "on a reads-only admitted set neither can move a hit, and "
            "tenant_ingredient_isolated checks that rather than assuming it."
        ),
    ),
    Rung(
        rid="L2",
        label="+ entity invalidation",
        ingredient="entity invalidation",
        key_recipe="cache_identity.key_typed(tool, args, tenant) + remember/invalidate",
        key_mode="typed",
        invalidation=True,
        admission="reads",
        key_binds_tenant=True,
        removes="stale_reads",
        note=(
            "a stored read is filed under every entity id it names; a write that "
            "lands evicts the keys filed under the ids it names, so a surviving key "
            "names no entity the write named."
        ),
    ),
    Rung(
        rid="L3",
        label="+ read-only admission (given)",
        ingredient="read-only admission -- already the baseline, adds nothing here",
        key_recipe="cache_identity.key_typed(tool, args, tenant) + remember/invalidate",
        key_mode="typed",
        invalidation=True,
        admission="reads",
        key_binds_tenant=True,
        removes="writes_served",
        note=(
            "the admission gate is what defines this ladder's denominator, so it was "
            "paid before rung 0 and this rung must reproduce L2 exactly. Its price is "
            "measured on the mirror ladder (M2 -> M3), where the same corpus is "
            "admitted in full."
        ),
    ),
)

_ALL_CALLS = ("all 2,275 calls at M0-M2 (writes and control tools admitted); "
              "reads only at M3, the admission rung")

MIRROR_RUNGS: Tuple[Rung, ...] = (
    Rung(
        rid="M0",
        label="blind key",
        ingredient="none -- the baseline",
        key_recipe="cache_identity.key_blind(tool, args)",
        key_mode="blind",
        invalidation=False,
        admission="all",
        key_binds_tenant=False,
        removes=None,
        note="every call stored and served, nothing evicted: the P1 baseline.",
    ),
    Rung(
        rid="M1",
        label="blind key + tenant",
        ingredient="tenant in the key",
        key_recipe="key_tenant_blind(tool, args, tenant) = cache_identity.key_blind([tenant, tool], args)",
        key_mode="blind_tenant",
        invalidation=False,
        admission="all",
        key_binds_tenant=True,
        removes="cross_tenant",
        note=(
            "the tenant in the key, on a cache that admits writes -- so the key "
            "cannot be key_typed, which refuses to key a write at all."
        ),
    ),
    Rung(
        rid="M2",
        label="+ entity invalidation",
        ingredient="entity invalidation",
        key_recipe="key_tenant_blind(tool, args, tenant) + cache_identity.remember/invalidate",
        key_mode="blind_tenant",
        invalidation=True,
        admission="all",
        key_binds_tenant=True,
        removes="stale_reads",
        note=(
            "freshness is handled for reads, but a write's own memo is never filed "
            "in the index (only reads are), so a repeated write is still served."
        ),
    ),
    Rung(
        rid="M3",
        label="+ read-only admission",
        ingredient="read-only admission",
        key_recipe="cache_identity.key_typed(tool, args, tenant) + remember/invalidate",
        key_mode="typed",
        invalidation=True,
        admission="reads",
        key_binds_tenant=True,
        removes="writes_served",
        note=(
            "the same chain measured on the 2,275 denominator: refuses the 446 writes "
            "and 101 control calls, and is therefore L2 restated. The M2 -> M3 gap is "
            "what admitting writes adds."
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


def _evict(index, key_map, cache, call: Call, rung: Rung, ci_mod) -> None:
    """Drop the entries a landed write obsoletes, through the module's invalidate()."""
    for typed_key in ci_mod.invalidate(index, call.tool, call.args):
        if rung.key_mode == "typed":
            cache.pop(typed_key, None)
            continue
        served = key_map.pop(typed_key, None)
        if served is not None:
            cache.pop(served, None)


def replay(streams: "OrderedDict[str, List[Call]]", rung: Rung,
           denominator: int, ci_mod) -> Dict[str, Any]:
    """Replay every run through one rung and return its counts and evidence.

    ``denominator`` is the ladder's fixed base -- 1,728 on the main ladder, 2,275
    on the mirror -- and is used for ``hit_rate_on_denominator``, the rate rung
    deltas are computed on. ``hit_rate`` (hits / admitted) is the cache's own
    efficiency and is not comparable across a rung that changes admission; the
    ladder-base rate is, because it holds the base fixed while the rungs move.
    """
    per_run: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    totals: Counter = Counter()
    hits_by_effect: Counter = Counter()
    by_tool: Dict[str, Counter] = {name: Counter() for name in UNSAFE_CLASSES}
    examples: Dict[str, List[Dict[str, Any]]] = {name: [] for name in UNSAFE_CLASSES}
    raw: Counter = Counter()

    for source, calls in streams.items():
        cache: Dict[str, Dict[str, Any]] = {}
        index: Dict[str, set] = {}
        # The module files and evicts *typed* keys; the served key may be blind.
        key_map: Dict[str, str] = {}
        write_log: List[Tuple[int, Tuple[str, ...], str, Dict[str, Any]]] = []
        run: Counter = Counter()
        run_hits: Counter = Counter()

        for call in calls:
            effect = ci_mod.effect_class(call.tool)
            run["calls"] += 1

            if not rung.admits(effect):
                run["not_admitted"] += 1
                if effect == "write":
                    write_log.append(
                        (call.pos, ci_mod.entity_keys(call.tool, call.args), call.tool, call.args)
                    )
                    if rung.invalidation:
                        _evict(index, key_map, cache, call, rung, ci_mod)
                continue

            run["admitted"] += 1
            key = served_key(rung, call, ci_mod)
            memo = cache.get(key)
            if memo is not None:
                run["hits"] += 1
                run_hits[effect] += 1
                if effect == "write":
                    raw["writes_served"] += 1
                    _record(run, by_tool, examples, "writes_served", call, memo, source, OrderedDict())
                elif effect == "control":
                    run["control_hits"] += 1
                else:
                    targets = frozenset(ci_mod.entity_keys(call.tool, call.args)) or frozenset(
                        memo["entity_ids"]
                    )
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
                entity_ids=ci_mod.entity_keys(call.tool, call.args),
            )
            if rung.invalidation and effect == "read":
                # File through the module's own remember() so the index is built
                # exactly as the module specifies; for typed keys the served key is
                # the typed key, for blind keys it is translated back.
                typed = ci_mod.remember(index, call.tool, call.args, tenant=call.tenant)
                if rung.key_mode == "typed":
                    if typed != key:  # pragma: no cover - would mean remember() drifted
                        raise SystemExit(
                            "soundness_ladder: typed served key disagrees with remember()"
                        )
                else:
                    key_map[typed] = key
            if effect == "write":
                write_log.append(
                    (call.pos, ci_mod.entity_keys(call.tool, call.args), call.tool, call.args)
                )
                if rung.invalidation:
                    _evict(index, key_map, cache, call, rung, ci_mod)

        totals.update(run)
        for effect_name in ("read", "write", "control"):
            hits_by_effect[effect_name] += run_hits[effect_name]
        per_run[source] = OrderedDict(
            calls=run["calls"],
            admitted=run["admitted"],
            hits=run["hits"],
            hit_rate=ratio(run["hits"], run["admitted"]),
            # This run's contribution to the ladder's hit rate: its hits over the
            # ladder base, not a rate of its own (its own rate is `hit_rate`).
            contribution_to_ladder_rate=ratio(run["hits"], denominator),
            unsafe_hits=run["unsafe_hits"],
            writes_served=run["writes_served"],
            stale_reads=run["stale_reads"],
            cross_tenant=run["cross_tenant"],
            control_hits=run["control_hits"],
        )

    unsafe_hits = totals["writes_served"] + totals["stale_reads"] + totals["cross_tenant"]
    return OrderedDict(
        id=rung.id,
        label=rung.label,
        ingredient_added=rung.ingredient,
        key=rung.key_recipe,
        key_mode=rung.key_mode,
        key_binds_tenant=rung.key_binds_tenant,
        removes=rung.removes,
        invalidation="entity (remember/invalidate)" if rung.invalidation else "none",
        admission=rung.admission,
        note=rung.note,
        calls_total=totals["calls"],
        admitted_calls=totals["admitted"],
        not_admitted_calls=totals["not_admitted"],
        hits=totals["hits"],
        hit_rate=ratio(totals["hits"], totals["admitted"]),
        hit_rate_on_denominator=ratio(totals["hits"], denominator),
        hit_rate_of_all_calls=ratio(totals["hits"], totals["calls"]),
        hits_by_effect=OrderedDict(
            (name, hits_by_effect[name]) for name in ("read", "write", "control")
        ),
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
        unsafe_by_tool=OrderedDict(
            (name, OrderedDict(sorted(by_tool[name].items()))) for name in UNSAFE_CLASSES
        ),
        per_run=per_run,
        examples=[examples[name][0] for name in UNSAFE_CLASSES if examples[name]],
    )


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
# Marginal costs: what each ingredient costs, and what it removes
# ---------------------------------------------------------------------------
def marginal_costs(rungs: Sequence[Mapping[str, Any]], denominator: int) -> List[Dict[str, Any]]:
    """Pairwise deltas between adjacent rungs, with the sentence they license.

    Every rate in the delta is ``hit_rate_on_denominator`` -- hits over the
    ladder's fixed base -- so the pair is measured on one denominator even when
    the later rung admits fewer calls (M3 admits reads only, and comparing its
    hits/admitted rate to M2's would mix 2,275 calls with 1,728 and reverse the
    sign of the cost). The cache's own ``hit_rate`` is still carried per rung.
    """
    out: List[Dict[str, Any]] = []
    for before, after in zip(rungs, rungs[1:]):
        removed = OrderedDict(
            (cls, before["unsafe_breakdown"][cls] - after["unsafe_breakdown"][cls])
            for cls in UNSAFE_CLASSES
        )
        unsafe_removed = before["unsafe_hits"] - after["unsafe_hits"]
        hits_lost = before["hits"] - after["hits"]
        before_rate = before["hit_rate_on_denominator"]
        after_rate = after["hit_rate_on_denominator"]
        lost_points = points(before_rate, after_rate)
        parts = [
            f"{cls} {count}" for cls, count in removed.items() if count
        ]
        detail = f" ({', '.join(parts)})" if parts else ""
        # The ingredient is aimed at one class; the replay can also remove others,
        # and that is a finding to report rather than to smooth away.
        target = after["removes"]
        off_target = OrderedDict(
            (cls, count) for cls, count in removed.items() if count and cls != target
        )
        caveat = ""
        if off_target:
            caveat = (
                f"off-target: {after['ingredient_added']} also removes "
                f"{', '.join(f'{count} {cls}' for cls, count in off_target.items())}, because "
                "the ingredient changes which memo a call reaches, not only whether a stale one "
                "is reachable; the class totals are counted by the replay, not attributed by hand"
            )
        out.append(
            OrderedDict(
                from_rung=before["id"],
                to_rung=after["id"],
                ingredient=after["ingredient_added"],
                denominator=denominator,
                hit_rate_before=before_rate,
                hit_rate_after=after_rate,
                hit_rate_points_lost=lost_points,
                hit_rate_before_on_admitted=before["hit_rate"],
                hit_rate_after_on_admitted=after["hit_rate"],
                hits_before=before["hits"],
                hits_after=after["hits"],
                hits_lost=hits_lost,
                unsafe_before=before["unsafe_hits"],
                unsafe_after=after["unsafe_hits"],
                unsafe_hits_removed=unsafe_removed,
                unsafe_removed_by_class=removed,
                target_class=target,
                removed_off_target=off_target,
                caveat=caveat,
                sentence=(
                    f"{after['ingredient_added']}: costs {lost_points:.2f} points of hit rate "
                    f"on {denominator:,} calls ({before_rate * 100:.2f}% -> {after_rate * 100:.2f}%) "
                    f"and removes {unsafe_hits_text(unsafe_removed)}{detail}."
                ),
            )
        )
    return out


def ladder(
    lid: str,
    title: str,
    admitted_set: str,
    denominator: int,
    rungs: Sequence[Rung],
    streams: "OrderedDict[str, List[Call]]",
    ci_mod,
) -> Dict[str, Any]:
    results = [replay(streams, rung, denominator, ci_mod) for rung in rungs]
    costs = marginal_costs(results, denominator)
    first_rate = results[0]["hit_rate_on_denominator"]
    last_rate = results[-1]["hit_rate_on_denominator"]
    return OrderedDict(
        id=lid,
        title=title,
        admitted_set=admitted_set,
        denominator=denominator,
        denominator_note=(
            "hit_rate is hits / admitted_calls of that rung; hit_rate_on_denominator is hits "
            "over this ladder's fixed base and is the rate adjacent rungs are compared on; "
            "hit_rate_of_all_calls is hits / 2,275, so the two ladders can also be read on one base"
        ),
        rungs=results,
        marginal_costs=costs,
        total_cost=OrderedDict(
            hit_rate_first=first_rate,
            hit_rate_last=last_rate,
            hit_rate_points_lost=points(first_rate, last_rate),
            hits_lost=results[0]["hits"] - results[-1]["hits"],
            unsafe_first=results[0]["unsafe_hits"],
            unsafe_last=results[-1]["unsafe_hits"],
            unsafe_hits_removed=results[0]["unsafe_hits"] - results[-1]["unsafe_hits"],
            statement=(
                f"from {results[0]['id']} to {results[-1]['id']}, the rungs give up "
                f"{points(first_rate, last_rate):.2f} points of hit rate on {denominator:,} calls "
                f"and remove {unsafe_hits_text(results[0]['unsafe_hits'] - results[-1]['unsafe_hits'])}"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Reconciliation: cross-checks that do not go through the replay's own path
# ---------------------------------------------------------------------------
def reconciliation(streams: "OrderedDict[str, List[Call]]", records: Sequence[Mapping[str, Any]],
                   main: Mapping[str, Any], mirror: Mapping[str, Any],
                   ci_mod) -> Dict[str, Any]:
    by_effect: Counter = Counter()
    calls_total = 0
    calls_with_tenant = 0
    for calls in streams.values():
        for call in calls:
            calls_total += 1
            by_effect[ci_mod.effect_class(call.tool)] += 1
            if call.tenant is not None:
                calls_with_tenant += 1

    # A write served from a never-invalidating cache is exactly a repeated write
    # key inside one run. Count it straight from the corpus: it must equal M0's and
    # M2's writes_served, because eviction never removes a write's own memo.
    repeated_write_keys = 0
    for calls in streams.values():
        seen = set()
        for call in calls:
            if ci_mod.effect_class(call.tool) != "write":
                continue
            key = ci_mod.key_blind(call.tool, call.args)
            if key in seen:
                repeated_write_keys += 1
            else:
                seen.add(key)

    tenant_counts: Counter = Counter()
    for record in records:
        if record.get("trial", 0) != 0:
            continue
        ids = {
            str(args["user_id"])
            for _turn, _tool, args in record_calls(record)
            if isinstance(args.get("user_id"), str)
        }
        tenant_counts[len(ids)] += 1

    main_by_id = {rung["id"]: rung for rung in main["rungs"]}
    mirror_by_id = {rung["id"]: rung for rung in mirror["rungs"]}

    # L0's hit count must be reproduced by the module's typed key with no tenant:
    # that is the claim that L1's only behavioural change is the tenant.
    typed_none = replay(streams, _rung_with("L0/typed-none", key_mode="typed_none",
                                            invalidation=False, admission="reads"),
                        EXPECTED_ADMITTED_READS, ci_mod)
    # The mirror's write-capable key, on reads only, must partition exactly as the
    # typed key does -- so M2 -> M3 is an admission change, not a key change.
    reads_blind_tenant = replay(streams, _rung_with("L2/blind-tenant", key_mode="blind_tenant",
                                                   invalidation=True, admission="reads"),
                               EXPECTED_ADMITTED_READS, ci_mod)

    return OrderedDict(
        calls_total=calls_total,
        calls_by_effect=OrderedDict(sorted(by_effect.items())),
        admitted_set=OrderedDict(
            denominator=main["denominator"],
            reads=by_effect["read"],
            writes=by_effect["write"],
            controls=by_effect["control"],
            non_reads=by_effect["write"] + by_effect["control"],
            reads_is_the_denominator=by_effect["read"] == main["denominator"],
            why=(
                "cache_identity admits only effect-class read tools; the 446 writes and 101 "
                "control calls are refused before the lookup, which is exactly the typed "
                "policy's admission decision"
            ),
        ),
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
            m0_writes_served=mirror_by_id["M0"]["unsafe_breakdown"]["writes_served"],
            m1_writes_served=mirror_by_id["M1"]["unsafe_breakdown"]["writes_served"],
            m2_writes_served=mirror_by_id["M2"]["unsafe_breakdown"]["writes_served"],
            matches_m0=repeated_write_keys == mirror_by_id["M0"]["unsafe_breakdown"]["writes_served"],
            invalidation_leaves_writes_alone=(
                mirror_by_id["M1"]["unsafe_breakdown"]["writes_served"]
                == mirror_by_id["M2"]["unsafe_breakdown"]["writes_served"]
            ),
            separated_by_the_tenant=(
                mirror_by_id["M0"]["unsafe_breakdown"]["writes_served"]
                - mirror_by_id["M1"]["unsafe_breakdown"]["writes_served"]
            ),
            why=(
                "a write's own memo is never filed in the eviction index (only reads are), so no "
                "eviction can remove it: writes_served must be unchanged from M1 to M2 and, on "
                "the blind rung, must equal the corpus's repeated-write count. M0 can exceed M1 "
                "only by pairing two identical writes the tenant separates -- one of them is a "
                "call whose record names no user"
            ),
        ),
        typed_key_neutrality=OrderedDict(
            l0_hits=main_by_id["L0"]["hits"],
            key_typed_tenant_none_hits=typed_none["hits"],
            equal=main_by_id["L0"]["hits"] == typed_none["hits"],
            unsafe_hits=typed_none["unsafe_hits"],
            unsafe_hits_equal=typed_none["unsafe_hits"] == main_by_id["L0"]["unsafe_hits"],
            why=(
                "key_typed(tool, args, None) is a bijective relabelling of key_blind's digest "
                "classes, so switching recipe without adding the tenant cannot move a hit; "
                "any L0 -> L1 difference is therefore the tenant"
            ),
        ),
        mirror_key_agreement=OrderedDict(
            reads_only_blind_tenant_hits=reads_blind_tenant["hits"],
            l2_hits=main_by_id["L2"]["hits"],
            equal=reads_blind_tenant["hits"] == main_by_id["L2"]["hits"],
            unsafe_hits=reads_blind_tenant["unsafe_hits"],
            why=(
                "the mirror's write-capable tenant key, replayed over reads only, must hit "
                "exactly as key_typed does; that is what isolates M2 -> M3 as an admission "
                "change rather than a key change"
            ),
        ),
        typed_policy_anchor=OrderedDict(
            committed_policy="P4 typed in examples/benchmarks/unsafe_cache_results.json",
            expected_hits=EXPECTED_TYPED_HITS,
            l3_hits=main_by_id["L3"]["hits"],
            l3_admitted=main_by_id["L3"]["admitted_calls"],
            l3_unsafe=main_by_id["L3"]["unsafe_hits"],
            matches=(
                main_by_id["L3"]["hits"] == EXPECTED_TYPED_HITS
                and main_by_id["L3"]["admitted_calls"] == EXPECTED_ADMITTED_READS
                and main_by_id["L3"]["unsafe_hits"] == 0
            ),
            why=(
                "L3 is the typed policy, so the ladder's top rung must reproduce the "
                "committed P4 replay; a mismatch means the ladder stopped measuring the "
                "cache the paper describes"
            ),
        ),
        identifying_ingredients=OrderedDict(
            (rung["id"], OrderedDict(
                key_mode=rung["key_mode"],
                key_binds_tenant=rung["key_binds_tenant"],
                invalidation=rung["invalidation"],
                admission=rung["admission"],
            ))
            for rung in main["rungs"] + mirror["rungs"]
        ),
    )


def _rung_with(rid: str, key_mode: str, invalidation: bool, admission: str) -> Rung:
    """An auxiliary rung used only by reconciliation; never reported as a rung."""
    return Rung(
        rid=rid,
        label="auxiliary",
        ingredient="auxiliary",
        key_recipe="auxiliary",
        key_mode=key_mode,
        invalidation=invalidation,
        admission=admission,
        key_binds_tenant=key_mode != "blind",
        removes=None,
        note="auxiliary replay for reconciliation only",
    )


# ---------------------------------------------------------------------------
# Checks: the invariants the ladder asserts about itself
# ---------------------------------------------------------------------------
def _check(ok: bool, detail: str = "") -> Dict[str, Any]:
    entry = OrderedDict(ok=bool(ok))
    if detail:
        entry["detail"] = detail
    return entry


def checks(main: Mapping[str, Any], mirror: Mapping[str, Any], recon: Mapping[str, Any],
           corpus_digest: str, deterministic: bool) -> Dict[str, Any]:
    main_rungs = main["rungs"]
    mirror_rungs = mirror["rungs"]
    main_unsafe = [rung["unsafe_hits"] for rung in main_rungs]
    mirror_unsafe = [rung["unsafe_hits"] for rung in mirror_rungs]

    monotone_main = all(a >= b for a, b in zip(main_unsafe, main_unsafe[1:]))
    monotone_mirror = all(a >= b for a, b in zip(mirror_unsafe, mirror_unsafe[1:]))
    breakdowns_ok = all(
        rung["unsafe_hits"] == sum(rung["unsafe_breakdown"].values())
        for rung in main_rungs + mirror_rungs
    )
    rates_ok = True
    for rung in main_rungs + mirror_rungs:
        if rung["admitted_calls"] and round(rung["hits"] / rung["admitted_calls"], 4) != rung["hit_rate"]:
            rates_ok = False
        if round(rung["hits"] / rung["calls_total"], 4) != rung["hit_rate_of_all_calls"]:
            rates_ok = False
        if rung["hits"] and round(rung["unsafe_hits"] / rung["hits"], 4) != rung["unsafe_share_of_hits"]:
            rates_ok = False
        if rung["unsafe_hits"] > rung["hits"]:
            rates_ok = False
    for doc in (main, mirror):
        for rung in doc["rungs"]:
            if round(rung["hits"] / doc["denominator"], 4) != rung["hit_rate_on_denominator"]:
                rates_ok = False
    for cost in main["marginal_costs"] + mirror["marginal_costs"]:
        if cost["denominator"] != (main["denominator"] if cost in main["marginal_costs"]
                                   else mirror["denominator"]):
            rates_ok = False

    l2, l3 = main_rungs[2], main_rungs[3]
    return OrderedDict(
        corpus_matches_committed=OrderedDict(
            ok=corpus_digest == EXPECTED_SHA256,
            detail=f"sha256={corpus_digest}",
        ),
        admitted_set_is_1728=OrderedDict(
            ok=(
                all(rung["admitted_calls"] == EXPECTED_ADMITTED_READS for rung in main_rungs)
                and recon["calls_by_effect"]["read"] == EXPECTED_ADMITTED_READS
                and recon["calls_total"] == EXPECTED_CALLS
            ),
            detail=(
                f"main rungs admit {sorted({r['admitted_calls'] for r in main_rungs})} of "
                f"{recon['calls_total']} calls; reads={recon['calls_by_effect']['read']}"
            ),
        ),
        one_admitted_set=OrderedDict(
            ok=(
                len({rung["admitted_calls"] for rung in main_rungs}) == 1
                and all(rung["calls_total"] == EXPECTED_CALLS for rung in main_rungs)
            ),
            detail="every main rung scores the same calls on the same streams",
        ),
        no_non_reads_admitted_on_main=OrderedDict(
            ok=all(
                rung["hits_by_effect"]["write"] == 0 and rung["hits_by_effect"]["control"] == 0
                for rung in main_rungs
            ),
            detail="a write or control hit on the main ladder would mean the admitted set leaked",
        ),
        unsafe_monotone_main=OrderedDict(
            ok=monotone_main,
            detail=f"unsafe along L0..L3 = {main_unsafe}",
        ),
        unsafe_reaches_zero_main=OrderedDict(
            ok=main_unsafe[2] == 0 and main_unsafe[3] == 0,
            detail=f"L2={main_unsafe[2]}, L3={main_unsafe[3]}",
        ),
        unsafe_monotone_mirror=OrderedDict(
            ok=monotone_mirror,
            detail=f"unsafe along M0..M3 = {mirror_unsafe}",
        ),
        unsafe_reaches_zero_mirror=OrderedDict(
            ok=mirror_unsafe[-1] == 0,
            detail=f"M3={mirror_unsafe[-1]}",
        ),
        mirror_not_cleared_before_admission=OrderedDict(
            ok=mirror_unsafe[2] > 0,
            detail=(
                f"M2={mirror_unsafe[2]} unsafe with the tenant and invalidation both in place; "
                "only read-only admission clears it"
            ),
        ),
        rung3_is_the_baseline_not_a_change=OrderedDict(
            ok=(
                l2["hits"] == l3["hits"]
                and l2["unsafe_hits"] == l3["unsafe_hits"]
                and l2["unsafe_breakdown"] == l3["unsafe_breakdown"]
            ),
            detail="L3 restates L2 by construction: admission was given before rung 0",
        ),
        typed_policy_anchor_reproduced=OrderedDict(
            ok=bool(recon["typed_policy_anchor"]["matches"]),
            detail=(
                f"L3 {l3['hits']}/{l3['admitted_calls']} hits, {l3['unsafe_hits']} unsafe; "
                f"committed P4 {EXPECTED_TYPED_HITS}/{EXPECTED_ADMITTED_READS}"
            ),
        ),
        tenant_ingredient_isolated=OrderedDict(
            ok=bool(recon["typed_key_neutrality"]["equal"])
            and bool(recon["typed_key_neutrality"]["unsafe_hits_equal"]),
            detail=(
                f"key_typed(tenant=None) hits {recon['typed_key_neutrality']['key_typed_tenant_none_hits']} "
                f"vs L0 {recon['typed_key_neutrality']['l0_hits']}"
            ),
        ),
        mirror_key_agrees_with_typed_on_reads=OrderedDict(
            ok=bool(recon["mirror_key_agreement"]["equal"]),
            detail=(
                f"reads-only blind+tenant hits {recon['mirror_key_agreement']['reads_only_blind_tenant_hits']} "
                f"vs L2 {recon['mirror_key_agreement']['l2_hits']}"
            ),
        ),
        write_repeats_match=OrderedDict(
            ok=bool(recon["write_repeats_independent"]["matches_m0"])
            and bool(recon["write_repeats_independent"]["invalidation_leaves_writes_alone"]),
            detail=(
                f"repeated write keys {recon['write_repeats_independent']['repeated_write_keys_in_run']} "
                f"= M0 {recon['write_repeats_independent']['m0_writes_served']}; M1 "
                f"{recon['write_repeats_independent']['m1_writes_served']} == M2 "
                f"{recon['write_repeats_independent']['m2_writes_served']}, so eviction removes no "
                "write's own memo"
            ),
        ),
        breakdown_sums=OrderedDict(ok=breakdowns_ok, detail="unsafe_hits == sum of its three classes"),
        rates_are_exact_ratios=OrderedDict(
            ok=rates_ok,
            detail="every rate is round(numerator/denominator, 4); unsafe never exceeds hits",
        ),
        deterministic_replay=OrderedDict(
            ok=bool(deterministic),
            detail="every rung replayed twice and the two documents compared before writing",
        ),
    )


# ---------------------------------------------------------------------------
# Definitions: the words the numbers are counted under
# ---------------------------------------------------------------------------
def definitions() -> Dict[str, Any]:
    return OrderedDict(
        admitted_set=(
            "the calls a rung stores: the 1,728 effect-class reads on the main ladder, all "
            "2,275 calls on mirror rungs M0-M2, reads again at M3. Admission is decided by "
            "cache_identity.effect_class before any key is computed."
        ),
        rung=(
            "one point on a ladder. Adjacent rungs differ by exactly one ingredient -- the "
            "tenant in the key, then entity invalidation, then read-only admission -- so the "
            "delta between them is attributable."
        ),
        hit=(
            "an admitted call whose key is already in the rung's run-scope cache; the memo is "
            "returned and the tool does not run"
        ),
        hit_rate=(
            "hits / admitted_calls of the same rung, rounded to 4 decimals. The main ladder's "
            "denominator is 1,728 for every rung, which is what makes its rungs comparable; "
            "hit_rate_of_all_calls divides by 2,275 and is reported so both ladders can be "
            "read on one base."
        ),
        writes_served=(
            "an unsafe hit where the call is a write: the mutation is answered from a memo and "
            "never runs, while the agent's belief that it ran is unchanged"
        ),
        stale_read=(
            "an unsafe hit where the call is a read and some write that executed names an entity "
            "id the read names, at a position strictly between the memo's store position and the "
            "hit; the memo predates the write"
        ),
        cross_tenant=(
            "an unsafe hit where the call is a read, the memo was stored under a different known "
            "tenant, and the hit is not already stale. An unknown tenant on either side is never "
            "counted, so this class is a lower bound"
        ),
        unsafe_hit=(
            "the union of writes_served, stale_reads and cross_tenant, each hit assigned to "
            "exactly one class by the precedence writes_served > stale_reads > cross_tenant, so "
            "the breakdown sums to the total. unsafe_by_class_overlapping keeps the "
            "pre-precedence counts where a hit falls in more than one class."
        ),
        ingredient=(
            "one of the three soundness ingredients the committed policies bundle: the tenant in "
            "the key (identity), entity invalidation (freshness), and read-only admission "
            "(effect). The ladder orders them tenant -> invalidation -> admission."
        ),
        marginal_cost=(
            "hit_rate_before - hit_rate_after, in percentage points, and unsafe_before - "
            "unsafe_after, in hits, for one adjacent pair. Positive points lost and positive "
            "unsafe removed are the cost and the benefit of that one ingredient."
        ),
        point=(
            "one percentage point of hit rate on the rung's denominator. Every point quoted is "
            "measured on 1,728 main-ladder calls or 2,275 mirror calls; no rate is averaged "
            "across rungs or runs."
        ),
        control_hits=(
            "hits on control tools (think, transfer_to_human_agents). They are admitted and "
            "memoised on mirror rungs M0-M2, refused on M3 and by every main rung, and are not "
            "one of the three unsafe classes"
        ),
        not_counted=(
            "reads naming no entity id cannot be joined to a write; unknown tenants cannot be "
            "compared; a failed write is indistinguishable from a landed one in this corpus and "
            "is treated as landed. Each gap can only understate unsoundness."
        ),
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def _rung_table(rungs: Iterable[Rung]) -> List[Dict[str, Any]]:
    return [
        OrderedDict(
            id=rung.id,
            label=rung.label,
            ingredient_added=rung.ingredient,
            key=rung.key_recipe,
            key_binds_tenant=rung.key_binds_tenant,
            invalidation=rung.invalidation,
            admission=rung.admission,
            note=rung.note,
        )
        for rung in rungs
    ]


def build_report(corpus: Path, corpus_digest: str, records, streams, identity_digest: str,
                 ci_mod) -> Dict[str, Any]:
    main = ladder(
        lid="main",
        title="nested ladder on the 1,728 reads cache_identity admits",
        admitted_set=_READS_ONLY,
        denominator=EXPECTED_ADMITTED_READS,
        rungs=MAIN_RUNGS,
        streams=streams,
        ci_mod=ci_mod,
    )
    mirror = ladder(
        lid="mirror",
        title="the same rungs with all 2,275 calls admitted, to price read-only admission",
        admitted_set=_ALL_CALLS,
        denominator=EXPECTED_CALLS,
        rungs=MIRROR_RUNGS,
        streams=streams,
        ci_mod=ci_mod,
    )
    recon = reconciliation(streams, records, main, mirror, ci_mod)

    # Determinism is measured, not asserted: replay both ladders again and compare
    # the documents. main() repeats the same comparison at the file level (bytes),
    # which is the guard that actually refuses to write.
    deterministic = (
        [replay(streams, rung, EXPECTED_ADMITTED_READS, ci_mod) for rung in MAIN_RUNGS]
        == main["rungs"]
        and [replay(streams, rung, EXPECTED_CALLS, ci_mod) for rung in MIRROR_RUNGS]
        == mirror["rungs"]
    )

    mirror_by_id = {rung["id"]: rung for rung in mirror["rungs"]}
    m2, m3 = mirror_by_id["M2"], mirror_by_id["M3"]
    added = OrderedDict(
        hits_added=m2["hits"] - m3["hits"],
        hit_rate_points_added=points(m2["hit_rate_on_denominator"], m3["hit_rate_on_denominator"]),
        hits_added_by_effect=OrderedDict(
            read=m2["hits_by_effect"]["read"] - m3["hits_by_effect"]["read"],
            write=m2["hits_by_effect"]["write"] - m3["hits_by_effect"]["write"],
            control=m2["hits_by_effect"]["control"] - m3["hits_by_effect"]["control"],
        ),
        read_hits_from_unexecuted_writes=OrderedDict(
            count=m2["hits_by_effect"]["read"] - m3["hits_by_effect"]["read"],
            why=(
                "a write served from cache never ran, so it evicted nothing: some reads that M3 "
                "would have re-read after the mutation are still answered from the memo. These "
                "extra read hits are a second-order effect of the writes_served defect, not a "
                "separate unsoundness class, and they are why the hits admitting writes buys "
                "exceed the write hits themselves"
            ),
        ),
        unsafe_hits_added=m2["unsafe_hits"] - m3["unsafe_hits"],
        unsafe_added_by_class=OrderedDict(
            writes_served=m2["unsafe_breakdown"]["writes_served"] - m3["unsafe_breakdown"]["writes_served"],
            stale_reads=m2["unsafe_breakdown"]["stale_reads"] - m3["unsafe_breakdown"]["stale_reads"],
            cross_tenant=m2["unsafe_breakdown"]["cross_tenant"] - m3["unsafe_breakdown"]["cross_tenant"],
        ),
        statement=(
            f"admitting writes and control tools buys {m2['hits'] - m3['hits']} extra hits "
            f"({points(m2['hit_rate_on_denominator'], m3['hit_rate_on_denominator']):.2f} points "
            f"on 2,275 calls) and adds "
            f"{unsafe_hits_text(m2['unsafe_hits'] - m3['unsafe_hits'])} that no key or eviction "
            "ingredient removes"
        ),
    )

    report = OrderedDict(
        schema=SCHEMA,
        task=TASK,
        result_kind="REPLAY",
        scope=(
            "tau-bench only: the first trial of every task in the committed corpus, four source "
            "runs, replayed through run-scope caches. The main ladder scores one admitted set "
            "(the 1,728 reads cache_identity admits) and adds one soundness ingredient per rung; "
            "the mirror ladder repeats the rungs with all 2,275 calls admitted. Unsafe hits are "
            "counted from the replay, never estimated."
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
                effect_map_source=ci_mod.MAP_SOURCE,
                effect_map_path=(
                    str(ci_mod.MAP_PATH.resolve().relative_to(REPO_ROOT)) if ci_mod.MAP_PATH is not None else None
                ),
                effect_map_error=ci_mod.MAP_ERROR,
                tools=len(ci_mod.EFFECT_CLASSES),
                classes=OrderedDict(
                    (cls, sorted(name for name, value in ci_mod.EFFECT_CLASSES.items() if value == cls))
                    for cls in ("read", "write", "control")
                ),
                why_imported=(
                    "the admitted set is its effect map, every digest on the ladder is one of its "
                    "key recipes, and every eviction is its index; re-deriving any of them here "
                    "is how a ladder drifts from the cache it claims to measure"
                ),
            ),
            ladders=OrderedDict(
                main=OrderedDict(
                    admitted_set=_READS_ONLY,
                    denominator=EXPECTED_ADMITTED_READS,
                    rungs=_rung_table(MAIN_RUNGS),
                ),
                mirror=OrderedDict(
                    admitted_set=_ALL_CALLS,
                    denominator=EXPECTED_CALLS,
                    rungs=_rung_table(MIRROR_RUNGS),
                ),
            ),
            anchors=OrderedDict(
                committed_typed_policy=OrderedDict(
                    source="examples/benchmarks/unsafe_cache_results.json",
                    policy="P4 typed",
                    hits=EXPECTED_TYPED_HITS,
                    admitted=EXPECTED_ADMITTED_READS,
                    why=(
                        "L3 must reproduce the committed typed replay; the anchor is pinned so a "
                        "drift in either artifact fails the checks instead of moving a paper number"
                    ),
                ),
                published_reuse=OrderedDict(
                    hits=746,
                    cacheable=1789,
                    hit_rate=0.4170,
                    why="L0 is that recipe's identity notion restricted to reads; the ladder "
                        "decomposes a stricter cache, it does not re-derive 41.70%",
                ),
            ),
            determinism=(
                "no clock, no environment, no randomness, no git lookup; JSON-native arguments; "
                "every collection emitted in a stable order; every rung replayed twice and the "
                "documents compared before writing"
            ),
            arithmetic="integer counts; rates are round(numerator/denominator, 4); point deltas round(., 2); no weighted averages",
        ),
        definitions=definitions(),
        ladders=OrderedDict(main=main, mirror=mirror),
        what_admitting_writes_adds=added,
        reconciliation=recon,
        checks=checks(main, mirror, recon, corpus_digest, deterministic),
    )
    return report


# ---------------------------------------------------------------------------
# Self-test: the ladder's invariants on a stream whose answers are known by hand
# ---------------------------------------------------------------------------
def self_test() -> int:
    module, _digest = load_identity()
    failures: List[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        if not condition:
            failures.append(f"{name}: {detail}" if detail else name)

    # 8 calls, one run, two tenants.
    #   0 read order #W1 as A      -> miss everywhere
    #   1 read order #W1 as B      -> L0 cross-tenant hit, L1+ miss
    #   2 cancel order #W1 as A    -> write; refused by the main ladder, evicts at L2
    #   3 read order #W1 as A      -> L0/L1 stale hit, L2 miss (evicted at 2)
    #   4 read order #W1 as A      -> L0/L1 stale hit, L2 hit on the memo stored at 3
    #   5 cancel order #W1 as A    -> write; duplicate of 2, so a write hit at M0-M2
    #   6 think                    -> control; refused by the main ladder
    #   7 think                    -> control hit at M0-M2
    raw = [
        ("get_order_details", {"order_id": "#W1"}, "A"),   # 0
        ("get_order_details", {"order_id": "#W1"}, "B"),   # 1
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),  # 2
        ("get_order_details", {"order_id": "#W1"}, "A"),   # 3
        ("get_order_details", {"order_id": "#W1"}, "A"),   # 4
        ("cancel_pending_order", {"order_id": "#W1"}, "A"),  # 5
        ("think", {"thought": "check inventory"}, "A"),     # 6
        ("think", {"thought": "check inventory"}, "A"),     # 7
    ]
    calls = [
        Call(pos=i, turn=0, task_id=0, trial=0, reward=1.0, tool=tool, args=args,
             tenant=tenant, source="synthetic")
        for i, (tool, args, tenant) in enumerate(raw)
    ]
    streams: "OrderedDict[str, List[Call]]" = OrderedDict([("synthetic", calls)])

    # Both synthetic ladders use the stream's own 8 calls as the fixed base.
    main = {rung.id: replay(streams, rung, len(calls), module) for rung in MAIN_RUNGS}
    mirror = {rung.id: replay(streams, rung, len(calls), module) for rung in MIRROR_RUNGS}

    # Main ladder: 6 admitted reads (calls 0,1,3,4 and 0-based? no: writes/controls refused).
    check("main admits only reads", all(r["admitted_calls"] == 4 for r in main.values()),
          str({k: v["admitted_calls"] for k, v in main.items()}))

    check("L0 counts the cross-tenant hit", main["L0"]["unsafe_breakdown"]["cross_tenant"] == 1,
          str(main["L0"]["unsafe_breakdown"]))
    check("L0 counts the stale reads", main["L0"]["unsafe_breakdown"]["stale_reads"] == 2,
          str(main["L0"]["unsafe_breakdown"]))
    check("L0 hits", main["L0"]["hits"] == 3, str(main["L0"]["hits"]))
    check("L0 unsafe is the union", main["L0"]["unsafe_hits"] == 3, str(main["L0"]["unsafe_hits"]))

    check("L1 removes the cross-tenant hit", main["L1"]["unsafe_breakdown"]["cross_tenant"] == 0,
          str(main["L1"]["unsafe_breakdown"]))
    check("L1 keeps staleness", main["L1"]["unsafe_breakdown"]["stale_reads"] == 2,
          str(main["L1"]["unsafe_breakdown"]))
    check("L1 hits", main["L1"]["hits"] == 2, str(main["L1"]["hits"]))

    check("L2 evicts both stale reads", main["L2"]["unsafe_hits"] == 0,
          str(main["L2"]["unsafe_breakdown"]))
    check("L2 still hits the post-write read", main["L2"]["hits"] == 1, str(main["L2"]["hits"]))
    check("L3 restates L2", main["L3"]["hits"] == main["L2"]["hits"]
          and main["L3"]["unsafe_hits"] == main["L2"]["unsafe_hits"])

    main_unsafe = [main[rid]["unsafe_hits"] for rid in ("L0", "L1", "L2", "L3")]
    check("main unsafe is monotone", all(a >= b for a, b in zip(main_unsafe, main_unsafe[1:])),
          str(main_unsafe))
    check("main unsafe reaches zero", main_unsafe[2] == 0 and main_unsafe[3] == 0, str(main_unsafe))

    # Mirror ladder: every call admitted until the admission rung narrows it.
    check("mirror admits everything up to M2",
          all(mirror[rid]["admitted_calls"] == 8 for rid in ("M0", "M1", "M2")),
          str({k: v["admitted_calls"] for k, v in mirror.items()}))
    check("M3 narrows to reads", mirror["M3"]["admitted_calls"] == 4,
          str(mirror["M3"]["admitted_calls"]))
    check("M0 serves the repeated write", mirror["M0"]["unsafe_breakdown"]["writes_served"] == 1,
          str(mirror["M0"]["unsafe_breakdown"]))
    check("M0 hits", mirror["M0"]["hits"] == 5, str(mirror["M0"]["hits"]))
    check("M1 keeps the write hit and the stale reads",
          mirror["M1"]["unsafe_breakdown"] ==
          OrderedDict(writes_served=1, stale_reads=2, cross_tenant=0),
          str(mirror["M1"]["unsafe_breakdown"]))
    check("M2 clears staleness and keeps the write hit",
          mirror["M2"]["unsafe_breakdown"] ==
          OrderedDict(writes_served=1, stale_reads=0, cross_tenant=0),
          str(mirror["M2"]["unsafe_breakdown"]))
    check("M2 still counts the control hit", mirror["M2"]["control_hits_not_counted"] == 1,
          str(mirror["M2"]["control_hits_not_counted"]))
    check("M3 is safe", mirror["M3"]["unsafe_hits"] == 0, str(mirror["M3"]["unsafe_breakdown"]))
    check("M3 refuses writes and controls", mirror["M3"]["not_admitted_calls"] == 4,
          str(mirror["M3"]["not_admitted_calls"]))
    check("mirror unsafe falls to zero only at M3",
          [mirror[rid]["unsafe_hits"] for rid in ("M0", "M1", "M2", "M3")] == [4, 3, 1, 0],
          str([mirror[rid]["unsafe_hits"] for rid in ("M0", "M1", "M2", "M3")]))

    # The key claims the ladder rests on.
    a = {"order_id": "#W1"}
    check("tenant key separates tenants",
          key_tenant_blind(module, "get_order_details", a, "A")
          != key_tenant_blind(module, "get_order_details", a, "B"))
    check("tenant key separates None from the string",
          key_tenant_blind(module, "get_order_details", a, None)
          != key_tenant_blind(module, "get_order_details", a, "None"))
    check("blind tenant key is not the blind key",
          key_tenant_blind(module, "get_order_details", a, "A")
          != module.key_blind("get_order_details", a))

    # Marginal costs must be arithmetic on the rungs, not a second measurement.
    costs = marginal_costs([main[rid] for rid in ("L0", "L1", "L2", "L3")], len(calls))
    check("costs are pairwise", len(costs) == 3)
    check("tenant cost is measured", costs[0]["hits_lost"] == 1 and costs[0]["unsafe_hits_removed"] == 1,
          json.dumps(costs[0]))
    check("invalidation cost is measured",
          costs[1]["hits_lost"] == 1 and costs[1]["unsafe_hits_removed"] == 2, json.dumps(costs[1]))
    check("admission cost is zero on the main ladder",
          costs[2]["hits_lost"] == 0 and costs[2]["unsafe_hits_removed"] == 0, json.dumps(costs[2]))
    check("points arithmetic", costs[0]["hit_rate_points_lost"] == 12.5,
          str(costs[0]["hit_rate_points_lost"]))
    check("costs are measured on the ladder base",
          all(cost["denominator"] == len(calls)
              and cost["hit_rate_before"] == main[cost["from_rung"]]["hit_rate_on_denominator"]
              for cost in costs), json.dumps(costs[0]))

    for failure in failures:
        print(f"self-test: FAIL {failure}", file=sys.stderr)
    if failures:
        return 1
    print("self-test: ok (8 synthetic calls, both ladders, 3 cost deltas, unsafe falls to 0)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summary(report: Mapping[str, Any], output: Path) -> str:
    parts = []
    for lid in ("main", "mirror"):
        ladder_doc = report["ladders"][lid]
        cells = [
            f"{rung['id']} {rung['hits']} hits "
            f"({rung['hit_rate_on_denominator'] * 100:.2f}% of {ladder_doc['denominator']:,}), "
            f"{rung['unsafe_hits']} unsafe"
            for rung in ladder_doc["rungs"]
        ]
        parts.append(f"{lid}: " + "; ".join(cells))
    return "soundness_ladder: " + " | ".join(parts) + f" -> {output}"


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
        print(f"soundness_ladder: no such corpus: {args.corpus}", file=sys.stderr)
        return 2

    corpus_digest = sha256_file(args.corpus)
    if corpus_digest != EXPECTED_SHA256:
        print(
            f"soundness_ladder: corpus digest {corpus_digest} does not match the committed "
            f"{EXPECTED_SHA256}; refusing to measure a different experiment under this schema",
            file=sys.stderr,
        )
        return 2

    module, identity_digest = load_identity()
    records = load_records(args.corpus)
    streams = build_streams(records)

    report = build_report(args.corpus, corpus_digest, records, streams, identity_digest,
                          module)
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"

    # Determinism guard: replay the whole ladder again and compare the documents.
    again = build_report(args.corpus, corpus_digest, load_records(args.corpus),
                         build_streams(records), load_identity()[1], module)
    if json.dumps(again, indent=2, ensure_ascii=False) + "\n" != text:
        print("soundness_ladder: replay is not deterministic; refusing to write", file=sys.stderr)
        return 3

    failed = [name for name, value in report["checks"].items() if not value["ok"]]
    if failed:
        print(f"soundness_ladder: WARNING invariants failed: {', '.join(failed)}", file=sys.stderr)

    args.output.write_text(text, encoding="utf-8")
    print(summary(report, args.output))
    if args.json:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
