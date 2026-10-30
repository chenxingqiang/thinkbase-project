"""Stale-window audit: does a hit's cached value predate a write to the same entity?

Motivation. Section 5.12 reports **41.70%** reuse across the distinct tasks of one
run on ``examples/real_traces/tau_bench_historical.jsonl``. That number is
measured by an exact-key cache that serves only side-effect-free calls, so a hit
is *sound* in the sense that replaying a read changes nothing. It is not
necessarily *correct*: an exact-key cache returns the value it captured when the
key was first inserted, and nothing in the key says whether the record behind it
was rewritten in between. ``get_order_details("#W2378156")`` read at position 12
and again at position 40 is a hit even if ``exchange_delivered_order_items`` on
that same order ran at position 25 -- and the second read then returns
pre-exchange item state. This script measures how often that window is open.

Two symmetric questions, both answered from the corpus alone:

  stale read   a read call whose key is already in the cache, where a write to
               the same entity (order / user / reservation, joined on the id the
               arguments carry) happened between the cached call and the hit.
  duplicate    a *write* call whose key is already in the cache. Under a cache
  write        that stores effectful calls (the ``naive`` regime of
               ``real_trace_external_baselines.py``) the second
               ``cancel_pending_order`` is served from cache and never runs, so
               the agent's belief that it cancelled twice is false. This is the
               symmetric failure and it is reported with the same care.

Conventions, deliberately identical to the published measurement so the audit
decomposes the 41.70% rather than replacing it:

  key        sha256("tool|" + json.dumps(args, sort_keys=True, default=str))
  purity     imported from ``real_trace_pilot.TOOL_CLASSES``; unknown -> effectful
  streams    one stream per source (four runs), records in file order, first
             trial of each task -- the 325 records / 1,789 cacheable calls whose
             pooled rate is 41.70%
  position   0-based index of a call in its run's stream, counting every call
             (reads and writes), so "write at 25, hit at 40" is readable
  cache      first insertion wins; a hit is served with the value captured at
             the inserting call, so the window is (insert_pos, hit_pos)

What the audit cannot see, and therefore does not count (each is a lower bound
on staleness, never an upper bound on safety):

  * reads with no id in their arguments (``find_user_id_by_email``,
    ``find_user_id_by_name_zip``, ``list_all_*``, ``search_*``, ``calculate``,
    ``think``) cannot be joined to a write, even though ``modify_user_address``
    changes a zip a name+zip lookup depends on;
  * ``product_id`` and the item ids carried by exchange/return calls live in
    different namespaces in tau-bench, so product reads are never joined;
  * ``book_reservation`` creates a reservation whose id is not in its arguments
    (it is in the observation), so no later read can be matched to it;
  * a write that failed (error return) changed nothing, but the corpus records
    inputs only, so every write call is treated as having landed. The
    ``reward == 1.0`` re-run below bounds how much that assumption matters.

Deterministic: no clock, no git rev, ordered dicts, sorted everything, counts
and 4-decimal rates only. Usage:

    ./venv/bin/python examples/benchmarks/stale_window_audit.py --self-test
    ./venv/bin/python examples/benchmarks/stale_window_audit.py \
        --output examples/benchmarks/stale_window_audit_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import Counter, OrderedDict, defaultdict
from statistics import median
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "examples" / "benchmarks" / "stale_window_audit_results.json"
PILOT_PATH = REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py"

SCHEMA = "stale_window_audit/v1"
EXPERIMENT = "stale_window_audit"
EXPECTED_SHA256 = "3b5d4d29989ad99ed276433027d6ede37acb1197635e4a99febf44f6b5e64116"
# Section 5.12's published within-run pair (uncapped, pure-only, first trial). The
# audit must decompose this number, not a different one, so it is asserted.
PUBLISHED_HITS = 746
PUBLISHED_CACHEABLE = 1789
N_EXAMPLES = 5

# Entity extraction. A call carries an entity when one of these argument keys is
# present; the value is the id used to join reads to writes. Only the record
# types a read actually returns are here, and a write is listed only when it
# mutates that record in the tau-bench mock backend.
READ_ENTITY_KEYS: Dict[str, str] = {
    "get_order_details": "order",
    "get_user_details": "user",
    "get_reservation_details": "reservation",
}
WRITE_ENTITIES: Dict[str, Tuple[str, ...]] = {
    # retail: the order record itself
    "cancel_pending_order": ("order",),
    "modify_pending_order_items": ("order",),
    "modify_pending_order_address": ("order",),
    "modify_pending_order_payment": ("order",),
    "exchange_delivered_order_items": ("order",),
    "return_delivered_order_items": ("order",),
    # retail: the user record (address book)
    "modify_user_address": ("user",),
    # airline: the reservation record
    "cancel_reservation": ("reservation",),
    "update_reservation_flights": ("reservation",),
    "update_reservation_baggages": ("reservation",),
    "update_reservation_passengers": ("reservation",),
    # Deliberately absent, with the reason:
    #   book_reservation          creates a reservation, id only in the observation
    #   send_certificate          issues a certificate, mutates no user field
    #   transfer_to_human_agents  ends the episode, mutates nothing
}
ENTITY_ID_FIELD: Dict[str, str] = {
    "order": "order_id",
    "user": "user_id",
    "reservation": "reservation_id",
}


# --------------------------------------------------------------------------
# Loading, keying, purity: imported so the audit cannot drift from the paper
# --------------------------------------------------------------------------
def _pilot_module():
    spec = importlib.util.spec_from_file_location("_stale_pilot", PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PILOT = None


def is_pure(tool: str) -> bool:
    global _PILOT
    if _PILOT is None:
        _PILOT = _pilot_module()
    return bool(_PILOT.is_pure(tool))


def args_text(args: Mapping[str, Any]) -> str:
    return json.dumps(dict(args), sort_keys=True, default=str)


def call_key(tool: str, args: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{tool}|{args_text(args)}".encode("utf-8")).hexdigest()


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


# --------------------------------------------------------------------------
# Entities
# --------------------------------------------------------------------------
def read_entity(tool: str, args: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """The (type, id) a read call addresses, or None if it is not id-addressed."""
    kind = READ_ENTITY_KEYS.get(tool)
    if kind is None:
        return None
    value = args.get(ENTITY_ID_FIELD[kind])
    if not isinstance(value, str) or not value:
        return None
    return (kind, value)


def write_entities(tool: str, args: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """Every (type, id) a write call mutates. Empty when the write is not id-joined."""
    out: List[Tuple[str, str]] = []
    for kind in WRITE_ENTITIES.get(tool, ()):
        value = args.get(ENTITY_ID_FIELD[kind])
        if isinstance(value, str) and value:
            out.append((kind, value))
    return out


# --------------------------------------------------------------------------
# Streams and calls
# --------------------------------------------------------------------------
class Call:
    __slots__ = ("pos", "turn", "task_id", "trial", "reward", "tool", "args", "key")

    def __init__(self, pos, turn, task_id, trial, reward, tool, args, key):
        self.pos = pos
        self.turn = turn
        self.task_id = task_id
        self.trial = trial
        self.reward = reward
        self.tool = tool
        self.args = args
        self.key = key

    def ref(self) -> Dict[str, Any]:
        return OrderedDict(
            pos=self.pos,
            turn=self.turn,
            task_id=self.task_id,
            trial=self.trial,
            tool=self.tool,
            reward=self.reward,
        )


def build_streams(records: Sequence[Mapping[str, Any]], first_trial_only: bool):
    """One call stream per source, in file order, with run-global positions."""
    streams: "OrderedDict[str, List[Call]]" = OrderedDict()
    for record in records:
        if first_trial_only and record.get("trial", 0) != 0:
            continue
        source = str(record.get("source"))
        stream = streams.setdefault(source, [])
        task_id = record.get("task_id")
        trial = record.get("trial")
        reward = record.get("reward")
        for turn_index, turn in enumerate(record.get("turns") or []):
            for pair in turn:
                tool = str(pair[0])
                args = dict(pair[1] or {})
                stream.append(
                    Call(
                        pos=len(stream),
                        turn=turn_index,
                        task_id=task_id,
                        trial=trial,
                        reward=reward,
                        tool=tool,
                        args=args,
                        key=call_key(tool, args),
                    )
                )
    return streams


# --------------------------------------------------------------------------
# Audit
# --------------------------------------------------------------------------
def audit_run(calls: Sequence[Call]) -> Dict[str, Any]:
    """Walk one run; return hits, stale reads, duplicate writes and the evidence."""
    cache: Dict[str, Call] = {}
    writes_by_entity: Dict[Tuple[str, str], List[Call]] = defaultdict(list)
    read_hits: List[Dict[str, Any]] = []
    duplicate_writes: List[Dict[str, Any]] = []
    calls_total = 0
    reads_total = 0
    writes_total = 0
    read_hits_total = 0
    stale_total = 0
    duplicate_write_total = 0

    for call in calls:
        calls_total += 1
        cached = cache.get(call.key)

        if is_pure(call.tool):
            reads_total += 1
            entity = read_entity(call.tool, call.args)
            if cached is not None:
                read_hits_total += 1
                hit = OrderedDict(
                    hit=call.ref(),
                    cached=cached.ref(),
                    entity_type=entity[0] if entity else None,
                    entity_id=entity[1] if entity else None,
                    intervening=[],
                )
                if entity is not None:
                    for write in writes_by_entity.get(entity, ()):
                        if cached.pos < write.pos < call.pos:
                            hit["intervening"].append(write.ref())
                    if hit["intervening"]:
                        stale_total += 1
                        hit["stale"] = True
                        hit["window"] = call.pos - cached.pos
                read_hits.append(hit)
            else:
                cache[call.key] = call
        else:
            writes_total += 1
            if cached is not None:
                duplicate_write_total += 1
                duplicate_writes.append(
                    OrderedDict(
                        write=call.ref(),
                        cached=cached.ref(),
                        entities=[
                            OrderedDict(type=kind, id=value)
                            for kind, value in write_entities(call.tool, call.args)
                        ],
                        gap=call.pos - cached.pos,
                    )
                )
            else:
                cache[call.key] = call
            for entity in write_entities(call.tool, call.args):
                writes_by_entity[entity].append(call)

    stale_hits = [hit for hit in read_hits if hit.get("stale")]
    return OrderedDict(
        calls=calls_total,
        reads=reads_total,
        writes=writes_total,
        read_hits=read_hits_total,
        stale_hits=stale_total,
        stale_share_of_hits=ratio(stale_total, read_hits_total),
        duplicate_writes=duplicate_write_total,
        duplicate_share_of_writes=ratio(duplicate_write_total, writes_total),
        _read_hits=read_hits,
        _duplicate_writes=duplicate_writes,
    )


def audit_coherence(calls: Sequence[Call]) -> Dict[str, int]:
    """The same stream under a cache that invalidates a read when its entity is written.

    This is the counterfactual the audit exists to price: what does coherence
    cost the 41.70%? A write to an order / user / reservation drops every cached
    read of that entity, so the next read of it misses, re-reads and is fresh.
    Reads with no joinable id are never invalidated (the key cannot name their
    dependency), so this is a lower bound on the cost of coherence.
    """
    cache: Dict[str, Call] = {}
    by_entity: Dict[Tuple[str, str], List[str]] = defaultdict(list)
    hits = 0
    reads = 0
    for call in calls:
        if is_pure(call.tool):
            reads += 1
            if call.key in cache:
                hits += 1
                continue
            cache[call.key] = call
            entity = read_entity(call.tool, call.args)
            if entity is not None:
                by_entity[entity].append(call.key)
        else:
            for entity in write_entities(call.tool, call.args):
                for key in by_entity.pop(entity, []):
                    cache.pop(key, None)
    return OrderedDict(reads=reads, hits=hits, hit_rate=ratio(hits, reads))


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def entity_type_breakdown(stale_hits: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counter: Counter = Counter()
    for hit in stale_hits:
        if hit["entity_type"]:
            counter[hit["entity_type"]] += 1
    return OrderedDict(sorted(counter.items()))


def pick_examples(stale_by_source: Dict[str, List[Dict[str, Any]]], n: int) -> List[Dict[str, Any]]:
    """One window per source, then an unrepresented entity type, then the shortest.

    Deterministic and corpus-covering. Windows whose intervening write sits in a
    rewarded episode are preferred, because a write in a failed episode may never
    have landed (the corpus records inputs only, so this cannot be checked).
    """
    source_order = list(stale_by_source)

    def quality(hit: Mapping[str, Any]):
        return (
            0 if hit["intervening"] and hit["intervening"][0]["reward"] == 1.0 else 1,
            hit["window"],
            hit["hit"]["pos"],
        )

    chosen: List[Dict[str, Any]] = []
    for source in source_order:
        if len(chosen) >= n:
            break
        pool = sorted(stale_by_source[source], key=quality)
        if pool:
            chosen.append(pool[0])
    seen_types = {hit["entity_type"] for hit in chosen}
    remaining = [hit for source in source_order for hit in stale_by_source[source] if hit not in chosen]
    remaining.sort(
        key=lambda hit: (
            0 if hit["entity_type"] not in seen_types else 1,
            *quality(hit),
            source_order.index(hit["_source"]),
        )
    )
    for hit in remaining:
        if len(chosen) >= n:
            break
        chosen.append(hit)
        seen_types.add(hit["entity_type"])
    return chosen[:n]


def example_view(hit: Mapping[str, Any], source: str) -> Dict[str, Any]:
    intervening = hit["intervening"]
    return OrderedDict(
        source=source,
        hit_task_id=hit["hit"]["task_id"],
        cached_task_id=hit["cached"]["task_id"],
        entity_type=hit["entity_type"],
        entity_id=hit["entity_id"],
        read_tool=hit["hit"]["tool"],
        cached_pos=hit["cached"]["pos"],
        writes=[
            OrderedDict(
                tool=w["tool"],
                pos=w["pos"],
                task_id=w["task_id"],
                episode_reward=w["reward"],
            )
            for w in intervening
        ],
        hit_pos=hit["hit"]["pos"],
        window_calls=hit["window"],
        hit_turn=hit["hit"]["turn"],
        hit_episode_reward=hit["hit"]["reward"],
    )


def duplicate_write_positions(calls: Sequence[Call]) -> set:
    """Positions of write calls whose key an earlier write already inserted."""
    seen: Dict[str, int] = {}
    out = set()
    for call in calls:
        if is_pure(call.tool):
            continue
        if call.key in seen:
            out.add(call.pos)
        else:
            seen[call.key] = call.pos
    return out


def audit(records: Sequence[Mapping[str, Any]], first_trial_only: bool) -> Dict[str, Any]:
    streams = build_streams(records, first_trial_only)
    per_run: Dict[str, Any] = OrderedDict()
    totals = Counter()
    coherence = Counter()
    stale_by_source: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    dupe_by_source: Dict[str, List[Dict[str, Any]]] = OrderedDict()
    entity_counter: Counter = Counter()
    tool_counts: Dict[str, Counter] = {}
    windows: List[int] = []
    episode: Counter = Counter()
    scope: Counter = Counter()
    duplicate_by_tool: Counter = Counter()
    duplicate_entities: Counter = Counter()
    duplicate_same_task = 0
    duplicate_cross_task = 0
    same_task = 0
    cross_task = 0
    stale_with_duplicate_write = 0

    for source, calls in streams.items():
        dup_positions = duplicate_write_positions(calls)
        result = audit_run(calls)
        stale_hits = result.pop("_read_hits")
        dupes = result.pop("_duplicate_writes")
        stale = [hit for hit in stale_hits if hit.get("stale")]
        for hit in stale_hits:
            counts = tool_counts.setdefault(hit["hit"]["tool"], Counter())
            counts["hits"] += 1
            if hit.get("stale"):
                counts["stale"] += 1
        for hit in stale:
            hit["_source"] = source
            entity_counter[hit["entity_type"]] += 1
            windows.append(hit["window"])
            if hit["hit"]["task_id"] == hit["cached"]["task_id"]:
                same_task += 1
            else:
                cross_task += 1
            write_tasks = {w["task_id"] for w in hit["intervening"]}
            hit_task = hit["hit"]["task_id"]
            cached_task = hit["cached"]["task_id"]
            if hit_task in write_tasks:
                # the hit's own task rewrote the record before re-reading it
                scope["hit_task_wrote"] += 1
                if hit["hit"]["reward"] == 1.0:
                    scope["hit_task_wrote_reward_1"] += 1
                if cached_task in write_tasks:
                    scope["hit_task_and_cached"] += 1
                else:
                    scope["hit_task_only"] += 1
            elif write_tasks == {cached_task}:
                scope["cached_task_only"] += 1
            elif cached_task in write_tasks:
                scope["cached_and_intermediate"] += 1
            else:
                scope["intermediate_task_only"] += 1
            for write in hit["intervening"]:
                write["_duplicate"] = write["pos"] in dup_positions
            if any(w["_duplicate"] for w in hit["intervening"]):
                stale_with_duplicate_write += 1
            # Whether the write landed is not recorded in this corpus; the
            # episode reward is the only proxy, so split on it instead of hiding it.
            write_ok = hit["intervening"][0]["reward"] == 1.0
            hit_ok = hit["hit"]["reward"] == 1.0
            if write_ok and hit_ok:
                episode["both_episodes_reward_1"] += 1
            elif write_ok:
                episode["hit_episode_reward_0"] += 1
            else:
                episode["write_episode_reward_0"] += 1
            if write_ok:
                episode["write_episode_reward_1"] += 1
        for dupe in dupes:
            duplicate_by_tool[dupe["write"]["tool"]] += 1
            for entity in dupe["entities"]:
                duplicate_entities[entity["type"]] += 1
            if dupe["write"]["task_id"] == dupe["cached"]["task_id"]:
                duplicate_same_task += 1
            else:
                duplicate_cross_task += 1
        stale_by_source[source] = stale
        dupe_by_source[source] = dupes
        result["entity_types"] = entity_type_breakdown(stale)
        result["distinct_stale_read_keys"] = len({h["cached"]["pos"] for h in stale})
        result["cross_task_stale"] = sum(
            1 for h in stale if h["hit"]["task_id"] != h["cached"]["task_id"]
        )
        result["same_task_stale"] = len(stale) - result["cross_task_stale"]
        result["stale_share_of_cacheable_calls"] = ratio(result["stale_hits"], result["reads"])
        result["stale_share_of_all_calls"] = ratio(result["stale_hits"], result["calls"])
        result["hit_rate"] = ratio(result["read_hits"], result["reads"])
        coherent = audit_coherence(calls)
        result["coherent_hits"] = coherent["hits"]
        result["coherent_hit_rate"] = coherent["hit_rate"]
        result["invalidated_hits"] = result["read_hits"] - coherent["hits"]
        coherence["hits"] += coherent["hits"]
        coherence["reads"] += coherent["reads"]
        per_run[source] = result
        for field in ("calls", "reads", "writes", "read_hits", "stale_hits", "duplicate_writes"):
            totals[field] += result[field]

    examples = [example_view(hit, hit["_source"]) for hit in pick_examples(stale_by_source, N_EXAMPLES)]

    # Duplicate-write examples, one per source, prefer the cancellations the
    # "the agent believes it cancelled twice" reading is about.
    dupe_examples = []
    for source, dupes in dupe_by_source.items():
        ranked = sorted(
            dupes,
            key=lambda d: (0 if d["write"]["tool"].startswith("cancel") else 1, d["gap"], d["write"]["pos"]),
        )
        for dupe in ranked[:1]:
            dupe_examples.append(
                OrderedDict(
                    source=source,
                    task_id=dupe["write"]["task_id"],
                    cached_task_id=dupe["cached"]["task_id"],
                    write_tool=dupe["write"]["tool"],
                    entities=dupe["entities"],
                    cached_pos=dupe["cached"]["pos"],
                    write_pos=dupe["write"]["pos"],
                    gap_calls=dupe["gap"],
                    same_task=dupe["write"]["task_id"] == dupe["cached"]["task_id"],
                )
            )

    per_tool = OrderedDict()
    for tool in sorted(tool_counts):
        counts = tool_counts[tool]
        per_tool[tool] = OrderedDict(
            hits=counts["hits"],
            stale=counts["stale"],
            stale_share=ratio(counts["stale"], counts["hits"]),
            joinable=tool in READ_ENTITY_KEYS,
        )

    return OrderedDict(
        accounting=OrderedDict(
            calls=totals["calls"],
            reads=totals["reads"],
            writes=totals["writes"],
            read_hits=totals["read_hits"],
            stale_hits=totals["stale_hits"],
            stale_share_of_hits=ratio(totals["stale_hits"], totals["read_hits"]),
            stale_share_of_cacheable_calls=ratio(totals["stale_hits"], totals["reads"]),
            stale_share_of_all_calls=ratio(totals["stale_hits"], totals["calls"]),
            stale_same_task=same_task,
            stale_cross_task=cross_task,
            stale_with_duplicate_write=stale_with_duplicate_write,
            duplicate_writes=totals["duplicate_writes"],
            duplicate_share_of_writes=ratio(totals["duplicate_writes"], totals["writes"]),
        ),
        entity_types=OrderedDict(sorted(entity_counter.items())),
        per_tool=per_tool,
        per_run=per_run,
        windows=OrderedDict(
            min_calls=min(windows) if windows else 0,
            median_calls=median(windows) if windows else 0,
            max_calls=max(windows) if windows else 0,
        ),
        write_task_scope=OrderedDict(
            cached_task_only=scope["cached_task_only"],
            intermediate_task_only=scope["intermediate_task_only"],
            cached_and_intermediate=scope["cached_and_intermediate"],
            hit_task_only=scope["hit_task_only"],
            hit_task_and_cached=scope["hit_task_and_cached"],
        ),
        reset_reading=OrderedDict(
            assumption=(
                "tau-bench runs every task attempt in a fresh environment built from "
                "the domain's base data, so a cached value captured before any write "
                "of its own task equals the state the hit's task starts from"
            ),
            hits_whose_own_task_rewrote_the_record=scope["hit_task_wrote"],
            share_of_hits=ratio(scope["hit_task_wrote"], totals["read_hits"]),
            in_reward_1_episodes=scope["hit_task_wrote_reward_1"],
            note=(
                "a window inside a failed episode cannot be shown to have served a "
                "wrong value, because the write may have been rejected; all five sit "
                "in reward-0 episodes, so this corpus exhibits unjustified hits, not "
                "an observable wrong replay"
            ),
        ),
        episode_reward=OrderedDict(
            (key, episode[key])
            for key in (
                "both_episodes_reward_1",
                "hit_episode_reward_0",
                "write_episode_reward_0",
                "write_episode_reward_1",
            )
        ),
        duplicate_writes=OrderedDict(
            by_tool=OrderedDict(sorted(duplicate_by_tool.items())),
            by_entity=OrderedDict(sorted(duplicate_entities.items())),
            same_task=duplicate_same_task,
            cross_task=duplicate_cross_task,
            examples=dupe_examples,
        ),
        counterfactual=OrderedDict(
            description=(
                "the same stream with a cache that drops a read's key when its entity "
                "is written; the gap to the 746 hits is what coherence costs this run"
            ),
            read_hits=totals["read_hits"],
            coherent_read_hits=coherence["hits"],
            invalidated_hits=totals["read_hits"] - coherence["hits"],
            hit_rate_before=ratio(totals["read_hits"], totals["reads"]),
            hit_rate_after=ratio(coherence["hits"], coherence["reads"]),
            points_lost=round(
                100.0 * (totals["read_hits"] - coherence["hits"]) / totals["reads"], 2
            ),
        ),
        examples=examples,
    )


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------
def self_test() -> int:
    failures = []

    def check(name: str, condition: bool) -> None:
        if not condition:
            failures.append(name)

    class R:
        def __init__(self, **kw):
            self.__dict__.update(kw)

        def ref(self):
            return OrderedDict(
                pos=self.pos, turn=self.turn, task_id=self.task_id,
                trial=self.trial, tool=self.tool,
            )

    def mk(pos, tool, args, task=0):
        return R(pos=pos, turn=0, task_id=task, trial=0, reward=1.0, tool=tool,
                 args=args, key=call_key(tool, args), _duplicate=False)

    # A write between the cached read and the hit opens the window.
    calls = [
        mk(0, "get_order_details", {"order_id": "#A"}),
        mk(1, "cancel_pending_order", {"order_id": "#A", "reason": "x"}),
        mk(2, "get_order_details", {"order_id": "#A"}),
    ]
    result = audit_run(calls)
    check("stale_detected", result["stale_hits"] == 1)
    check("stale_entity_order", result["_read_hits"][0]["entity_type"] == "order")

    # A write to a *different* order must not open it.
    calls = [
        mk(0, "get_order_details", {"order_id": "#A"}),
        mk(1, "cancel_pending_order", {"order_id": "#B", "reason": "x"}),
        mk(2, "get_order_details", {"order_id": "#A"}),
    ]
    check("unrelated_write_not_stale", audit_run(calls)["stale_hits"] == 0)

    # A write before the cached call is out of window.
    calls = [
        mk(0, "cancel_pending_order", {"order_id": "#A", "reason": "x"}),
        mk(1, "get_order_details", {"order_id": "#A"}),
        mk(2, "get_order_details", {"order_id": "#A"}),
    ]
    check("write_before_insert_not_stale", audit_run(calls)["stale_hits"] == 0)

    # A read with no id cannot be joined to a write even when it repeats.
    calls = [
        mk(0, "list_all_airports", {}),
        mk(1, "cancel_reservation", {"reservation_id": "R1"}),
        mk(2, "list_all_airports", {}),
    ]
    result = audit_run(calls)
    check("idless_read_hit_not_stale", result["stale_hits"] == 0 and result["read_hits"] == 1)

    # The symmetric case: the second cancel is a cache duplicate.
    calls = [
        mk(0, "cancel_reservation", {"reservation_id": "R1"}),
        mk(1, "cancel_reservation", {"reservation_id": "R1"}),
    ]
    result = audit_run(calls)
    check("duplicate_write_counted", result["duplicate_writes"] == 1)
    check("duplicate_write_no_read_hit", result["read_hits"] == 0)

    # Purity table still classifies the domain's writes as effectful.
    check("write_is_effectful", not is_pure("cancel_pending_order"))
    check("read_is_pure", is_pure("get_order_details"))

    for name in failures:
        print(f"FAIL {name}", file=sys.stderr)
    print("self-test " + ("ok" if not failures else f"{len(failures)} failure(s)"))
    return 0 if not failures else 1


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def build_block(corpus: Path, corpus_digest: str, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The merge-ready ``stale`` block: tau-bench streams, hit-level staleness."""
    primary = audit(records, first_trial_only=True)
    sensitivity = audit(records, first_trial_only=False)
    key_counts = OrderedDict(
        (source, run["distinct_stale_read_keys"]) for source, run in primary["per_run"].items()
    )
    return OrderedDict(
        schema=SCHEMA,
        result_kind="MEASURED",
        scope=(
            "tau-bench only: 4 source runs (gpt-4o-retail, gpt-4o-airline, "
            "sonnet-35-new-retail, sonnet-35-new-airline), first trial of each task, "
            "file order. Counts *hits* answered from a value that predates a write to "
            "the same entity; the corpus-wide key-level companion is first_trial.c / "
            "all_trials.c above, which counts read keys, not hits."
        ),
        corpus_sha256=corpus_digest,
        provenance=OrderedDict(
            corpus=str(corpus.relative_to(REPO_ROOT))
            if corpus.is_relative_to(REPO_ROOT) else str(corpus),
            corpus_sha256=corpus_digest,
            records=len(records),
            streams="4 runs, first trial per task (325 records, 2,275 calls), file order",
            positions="0-based call index within a run, every call counted",
            key="sha256('tool|' + sorted-json(arguments))",
            purity="real_trace_pilot.TOOL_CLASSES (unknown -> effectful)",
            cache="one per run, first insertion wins, unbounded residency",
            deterministic=True,
            no_model_calls=True,
        ),
        anchor=OrderedDict(
            published_hits=PUBLISHED_HITS,
            published_cacheable_calls=PUBLISHED_CACHEABLE,
            reproduced_hits=primary["accounting"]["read_hits"],
            reproduced_cacheable_calls=primary["accounting"]["reads"],
            matches_published=(
                primary["accounting"]["read_hits"] == PUBLISHED_HITS
                and primary["accounting"]["reads"] == PUBLISHED_CACHEABLE
            ),
        ),
        accounting=primary["accounting"],
        entity_types=primary["entity_types"],
        per_tool=primary["per_tool"],
        per_run=primary["per_run"],
        windows=primary["windows"],
        write_task_scope=primary["write_task_scope"],
        reset_reading=primary["reset_reading"],
        episode_reward=primary["episode_reward"],
        counterfactual=primary["counterfactual"],
        examples=primary["examples"],
        duplicate_writes=primary["duplicate_writes"],
        reconciliation=OrderedDict(
            distinct_stale_read_keys_by_run=key_counts,
            total_distinct_stale_read_keys=sum(key_counts.values()),
            note=(
                "one read key can be answered stale several times, which is why 113 "
                "keys carry the 169 hits; these per-run key counts are the "
                "tau_bench_historical rows of first_trial.c.scope_run.by_run"
            ),
        ),
        sensitivity=OrderedDict(
            all_trials=OrderedDict(
                description=(
                    "the same audit over every record: trial replay re-runs one "
                    "conversation against a reset backend, which inflates hits, stale "
                    "windows and duplicate writes, so this is a bound, not a headline"
                ),
                accounting=sensitivity["accounting"],
                entity_types=sensitivity["entity_types"],
                duplicate_writes=OrderedDict(
                    by_tool=sensitivity["duplicate_writes"]["by_tool"],
                    same_task=sensitivity["duplicate_writes"]["same_task"],
                    cross_task=sensitivity["duplicate_writes"]["cross_task"],
                ),
            ),
            unjoined=OrderedDict(
                reads_without_id=[
                    "find_user_id_by_email",
                    "find_user_id_by_name_zip",
                    "list_all_product_types",
                    "list_all_airports",
                    "search_direct_flight",
                    "search_onestop_flight",
                    "calculate",
                    "think",
                ],
                writes_without_joinable_id=[
                    "book_reservation",
                    "send_certificate",
                    "transfer_to_human_agents",
                ],
                note=(
                    "reads and writes that share no id are never counted as stale, and "
                    "product reads cannot be joined to item ids from another namespace, "
                    "so every count here is a lower bound on staleness"
                ),
            ),
            write_success=(
                "the corpus records inputs only, so whether a write call landed is not "
                "observable; episode_reward splits the windows by the reward of the "
                "write's episode and of the hit's episode"
            ),
        ),
    )


def _indent(text: str, spaces: int = 2) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line else line for line in text.splitlines())


def strip_stale(raw: str) -> str:
    """Reverse this script's own insertion, restoring the base document's bytes."""
    marker = '\n  "stale": {'
    at = raw.rindex(marker)
    kept = raw[:at].rstrip().rstrip(",")
    return kept + "\n}\n"


def merge_into(
    target: Path,
    block: Mapping[str, Any],
    check_key_counts: bool = True,
    replace: bool = False,
) -> int:
    """Append the block under ``stale`` without rewriting a single other byte.

    The target is a sibling artifact (``unsafe_reuse_audit.json``). It may be
    regenerated by its owner, so this is a text insertion before the final
    brace, not a re-serialisation: every existing key keeps its exact bytes and
    its exact order. If the key already exists the merge is a no-op, and, when
    asked, the block refuses to merge unless the target's own key-level stale
    counts agree with this harness's.
    """
    raw = target.read_text(encoding="utf-8")
    doc = json.loads(raw)
    if "stale" in doc:
        if not replace:
            print(f"{target} already carries a 'stale' key; nothing to do", file=sys.stderr)
            return 0
        base = strip_stale(raw)
        base_doc = json.loads(base)
        expected = {k: v for k, v in doc.items() if k != "stale"}
        if base_doc != expected:
            print("refusing to replace: stripping 'stale' would change another key",
                  file=sys.stderr)
            return 7
        raw, doc = base, base_doc
    if check_key_counts:
        try:
            rows = doc["first_trial"]["c"]["scope_run"]["by_run"]
        except KeyError:
            rows = {}
        mine = block["reconciliation"]["distinct_stale_read_keys_by_run"]
        theirs = {
            source: rows.get(f"tau_bench_historical::{source}", {}).get("stale")
            for source in mine
        }
        if rows and theirs != dict(mine):
            print(
                "refusing to merge: key-level stale counts disagree with "
                f"first_trial.c.scope_run.by_run ({theirs} vs {dict(mine)})",
                file=sys.stderr,
            )
            return 3
    body = json.dumps(block, indent=2, ensure_ascii=False)
    close = raw.rindex("}")
    head, tail = raw[:close], raw[close:]
    if head.rstrip().endswith("{"):
        separator = "\n"
    else:
        separator = ",\n"
    kept = head.rstrip()
    insertion = separator + '  "stale": ' + _indent(body, 2).lstrip()
    merged = kept + insertion + "\n" + tail
    check = json.loads(merged)
    if list(check) != list(doc) + ["stale"]:
        print("refusing to merge: key order changed", file=sys.stderr)
        return 4
    for key, value in doc.items():
        if check[key] != value:
            print(f"refusing to merge: key {key!r} would change", file=sys.stderr)
            return 5
    if not merged.startswith(kept) or not merged[len(kept):].startswith(insertion):
        print("refusing to merge: bytes before the new key would change", file=sys.stderr)
        return 6
    target.write_text(merged, encoding="utf-8")
    print(f"merged 'stale' into {target} ({len(block['examples'])} examples, "
          f"{block['accounting']['stale_hits']} stale hits)")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--merge-into", type=Path, default=None,
                        help="append the block under 'stale' in an existing audit JSON")
    parser.add_argument("--replace-stale", action="store_true",
                        help="with --merge-into, rewrite an existing 'stale' block")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the block to stdout")
    args = parser.parse_args(argv)

    if args.self_test:
        return self_test()

    corpus_digest = sha256_file(args.corpus)
    if corpus_digest != EXPECTED_SHA256:
        print(
            f"corpus digest {corpus_digest} does not match the committed {EXPECTED_SHA256}; "
            "refusing to measure",
            file=sys.stderr,
        )
        return 2

    records = load_records(args.corpus)
    block = build_block(args.corpus, corpus_digest, records)
    text = json.dumps(block, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    if args.json:
        sys.stdout.write(text)
    else:
        acc = block["accounting"]
        print(
            f"read hits {acc['read_hits']}, stale {acc['stale_hits']} "
            f"({acc['stale_share_of_hits']:.2%} of hits), "
            f"duplicate writes {acc['duplicate_writes']} "
            f"({acc['duplicate_share_of_writes']:.2%} of writes) -> {args.output}"
        )
    if args.merge_into is not None:
        return merge_into(args.merge_into, block, replace=args.replace_stale)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
