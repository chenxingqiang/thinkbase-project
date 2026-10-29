"""Offline external baselines on real agent traces: what a plain exact-key cache earns.

Motivation. Section 5.12 reports **41.70%** reuse "across the distinct tasks of
one run" and **1.73%** within a task on the tau-bench trajectories in
``examples/real_traces/tau_bench_historical.jsonl``. Those figures are produced
by AgentISA's result cache, which (a) serves only side-effect-free calls and
(b) keeps every key for the whole run. A reader is entitled to ask whether the
headline is an artifact of that design: would an *external*, ISA-agnostic cache
do as well, and how much of it survives a bounded capacity?

This script answers only what the corpus can answer, using nothing but the
ordered ``(tool, arguments)`` pairs. It computes four families of baselines:

  * exact-key LRU at capacities 8..4096, under three scopes -- per task attempt
    (``task``), shared across the tasks of one run (``run``), and shared across
    every run in the corpus (``global``);
  * a static hot-set of the top-N keys learned on the *first half* of a stream
    and measured on the *second half* (N in 16, 64, 256) -- no eviction, no
    admission policy, just a fixed list;
  * same-turn duplicate rate (does one assistant turn repeat a call?);
  * the exact-key cross-task hit rate, decomposed against the published
    41.70% within-run figure.

Two accounting regimes, kept apart everywhere:

  ``naive``   the cache stores and serves *every* call, denominator = all
              calls. This is what an off-the-shelf memoizer does, and it is
              UNSOUND on effectful tools: replaying ``exchange_delivered_order_items``
              from cache would skip a real state change. The number of hits
              that would be unsound replays is reported, never hidden.
  ``sound``   serve only side-effect-free calls, denominator = cacheable calls.
              This is the like-for-like comparison with 41.70%.

Effect classification is not re-derived here. It is imported from the real-trace
pilot (``TOOL_CLASSES`` / ``is_pure``) so that this baseline cannot drift from
the published one; it is also a domain fact about tau-bench (``get_order_details``
reads, ``cancel_pending_order`` writes), not an AgentISA artifact. The ``naive``
numbers are reported precisely so that no conclusion rests on that import.

What this script deliberately does NOT model -- see ``CANNOT_EMULATE`` in the
report, which is generated into the JSON rather than left to prose:
semantic/embedding caches and provider prefix caches. The trace retains neither
embeddings nor prompt token spans, so any number for them would be invented.

Dependencies: the standard library plus the pilot's tool table (importing the
pilot pulls in agentflow, as the pilot's own tests already do); no network.
Deterministic: no timestamps, no git rev, ordered dicts, counts plus 4-decimal
rates -- two runs produce byte-identical JSON. Usage:

    ./venv/bin/python examples/benchmarks/real_trace_external_baselines.py --self-test
    ./venv/bin/python examples/benchmarks/real_trace_external_baselines.py --output /tmp/b.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = (
    REPO_ROOT / "examples" / "benchmarks" / "real_trace_external_baselines_results.json"
)
PILOT_PATH = REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot.py"

SCHEMA = "real_trace_external_baselines/v1"
EXPERIMENT = "real_trace_external_baselines"
CAPACITIES: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
HOT_SET_SIZES: Tuple[int, ...] = (16, 64, 256)
SCOPES: Tuple[str, ...] = ("task", "run", "global")
MODES: Tuple[str, ...] = ("naive", "sound")

# Published reference these baselines are compared against (Section 5.12,
# first-trial convention, uncapped cache, warm pure-only accounting). Hard-coded
# as the *comparison target*, then independently reproduced in ``anchor`` below:
# if the reproduction ever fails, the comparison is void and the script says so.
PUBLISHED_WITHIN_RUN = 0.4170
PUBLISHED_WITHIN_TASK = 0.0173

Call = Tuple[int, str, Any]  # (turn index, tool, args)


# --------------------------------------------------------------------------
# Loading and keying
# --------------------------------------------------------------------------
def _pilot_module():
    spec = importlib.util.spec_from_file_location("_rt_pilot", PILOT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PILOT = None


def is_pure(tool: str) -> bool:
    """Side-effect-free per the pilot's disclosed table (unknown -> effectful)."""
    global _PILOT
    if _PILOT is None:
        _PILOT = _pilot_module()
    return bool(_PILOT.is_pure(tool))


def args_text(args: Mapping[str, Any]) -> str:
    """Canonical argument text: sorted keys, so key order never splits a key."""
    try:
        return json.dumps(dict(args), sort_keys=True, default=str)
    except Exception:  # pragma: no cover - unhashable args are not in this corpus
        return str(args)


def call_key(tool: str, args: Mapping[str, Any]) -> str:
    """The published exact-key recipe: sha256("tool|canonical(args)").

    Identical to the pilot's instruction-scoped key, which is what Section
    5.12's 41.70% is measured under -- so the comparison is key-for-key, and
    the only differences left are purity scope and capacity.
    """
    return hashlib.sha256(f"{tool}|{args_text(args)}".encode("utf-8")).hexdigest()


def load_records(path: Path) -> List[Dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def record_calls(record: Mapping[str, Any]) -> List[Call]:
    """Ordered calls of one trajectory; ``turns[t][c] = [tool, args]``."""
    out: List[Call] = []
    for t_idx, turn in enumerate(record.get("turns") or []):
        for call in turn:
            tool, args = call[0], call[1]
            out.append((t_idx, str(tool), args))
    return out


def first_trial(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    """One trial per task -- the headline convention.

    tau-bench records several trials of the same task; replaying an identical
    task is not workload repetition, so trials > 0 are excluded. Ordered by
    original file position, which is the recording order.
    """
    return [r for r in records if r.get("trial") in (None, 0)]


def streams_by_source(records: Sequence[Mapping[str, Any]]) -> "OrderedDict[str, List]":
    streams: "OrderedDict[str, List]" = OrderedDict()
    for rec in records:
        streams.setdefault(str(rec.get("source") or "unknown"), []).append(rec)
    return streams


def task_id_of(record: Mapping[str, Any]) -> str:
    return f"{record.get('source') or 'unknown'}#{record.get('task_id')}"


def corpus_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Bounded exact-key LRU
# --------------------------------------------------------------------------
def lru_sweep(
    records: Sequence[Mapping[str, Any]],
    capacity: int,
    scope: str,
    mode: str,
) -> Dict[str, Any]:
    """One cache configuration over the stream.

    ``scope`` decides when the cache is flushed: per task attempt (``task``),
    per run/source (``run``), or never (``global``). ``mode`` decides what may
    be stored: everything (``naive``) or side-effect-free calls only (``sound``).

    A hit is charged to the call that found the key present *before* insertion,
    and is decomposed by where the key was first inserted: same task or a
    different one. That decomposition is the whole point of the exercise --
    ``run``-scope hits that come from another task are the cross-task reuse the
    paper's 41.70% refers to.
    """
    if capacity < 0:
        raise ValueError("capacity must be >= 0")
    if scope == "run":
        # run scope is honored by the caller, which sweeps each source stream
        # separately and sums the counters; a flat sequence has no run boundary.
        raise ValueError("run scope must be swept per source")
    cache: "OrderedDict[str, Tuple[str, bool]]" = OrderedDict()  # key -> (task, pure)
    counters = {
        "calls": 0,
        "cacheable_calls": 0,
        "effectful_calls": 0,
        "hits": 0,
        "hits_cacheable": 0,
        "hits_cross_task": 0,
        "hits_same_task": 0,
        "hits_unsound_effectful": 0,
        "insertions": 0,
        "evictions": 0,
        "skipped_by_mode": 0,
    }

    def flush() -> None:
        cache.clear()

    for rec in records:
        task = task_id_of(rec)
        if scope == "task":
            flush()
        for _t_idx, tool, args in record_calls(rec):
            pure = is_pure(tool)
            key = call_key(tool, args)
            counters["calls"] += 1
            counters["cacheable_calls"] += 1 if pure else 0
            counters["effectful_calls"] += 0 if pure else 1

            servable = (mode == "naive") or pure
            hit = servable and key in cache
            if hit:
                origin_task, origin_pure = cache[key]
                counters["hits"] += 1
                if pure:
                    counters["hits_cacheable"] += 1
                if origin_task != task:
                    counters["hits_cross_task"] += 1
                else:
                    counters["hits_same_task"] += 1
                # Serving an effectful call from cache skips a real state
                # change. In sound mode those keys are never inserted, so this
                # counts only in naive mode.
                if not pure or not origin_pure:
                    counters["hits_unsound_effectful"] += 1
                cache.move_to_end(key)

            if not servable:
                counters["skipped_by_mode"] += 1
                continue
            if key in cache:
                cache.move_to_end(key)
            else:
                cache[key] = (task, pure)
                counters["insertions"] += 1
                while capacity and len(cache) > capacity:
                    cache.popitem(last=False)
                    counters["evictions"] += 1


    denom = (
        counters["calls"] if mode == "naive" else counters["cacheable_calls"]
    ) or 1
    counters["hit_rate"] = round(counters["hits"] / denom, 4)
    counters["capacity"] = capacity
    counters["scope"] = scope
    counters["mode"] = mode
    return counters


def _sweep_streams(
    streams: "OrderedDict[str, List]",
    scope: str,
    capacities: Iterable[int],
    mode: str,
) -> Dict[str, Dict[str, Any]]:
    """Run the sweep with proper scope flushing.

    ``run`` scope is per source, so each source is swept separately and the
    counters are summed; ``global`` concatenates the whole corpus in stream
    order; ``task`` is flush-per-record and is order-independent.
    """
    out: Dict[str, Dict[str, Any]] = {}
    all_records = [r for recs in streams.values() for r in recs]
    for cap in capacities:
        if scope == "run":
            total = None
            for recs in streams.values():
                part = lru_sweep(recs, cap, "global", mode)  # 'global' never flushes
                total = part if total is None else _merge(total, part)
            counters = total
        else:
            counters = lru_sweep(all_records, cap, scope, mode)
        denom = (counters["calls"] if mode == "naive"
                 else counters["cacheable_calls"]) or 1
        counters["capacity"] = cap
        counters["scope"] = scope
        counters["mode"] = mode
        counters["hit_rate"] = round(counters["hits"] / denom, 4)
        out[str(cap)] = counters
    return out


def _merge(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    merged = dict(a)
    for k, v in b.items():
        if isinstance(v, (int, float)) and isinstance(merged.get(k), (int, float)):
            merged[k] = merged[k] + v
    return merged


# --------------------------------------------------------------------------
# Static hot-set: train on the first half, measure on the second half
# --------------------------------------------------------------------------
def split_half(records: Sequence[Mapping[str, Any]]) -> Tuple[List, List]:
    n = len(records)
    return list(records[: n // 2]), list(records[n // 2:])


def hot_set_block(
    streams: "OrderedDict[str, List]",
    mode: str,
) -> Dict[str, Any]:
    """Top-N exact keys by training-half frequency, scored on the test half.

    Two trainers. ``global`` learns one list from every source's first half --
    the deployable shape, one cache for the whole tenant. ``per_source`` learns
    a list per source and scores it on that source's own second half, the
    optimistic bound for a workload-aware deployment.

    A key counts as a hit if it is in the list; there is no admission or
    eviction, and the list is fixed before the test half is seen, so the
    measurement is causal.
    """
    train_counts: Dict[str, int] = defaultdict(int)
    train_counts_by_source: Dict[str, Dict[str, int]] = defaultdict(
        lambda: defaultdict(int))
    test_calls: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)

    for source, recs in streams.items():
        train, test = split_half(recs)
        for rec in train:
            for _t, tool, args in record_calls(rec):
                pure = is_pure(tool)
                if mode == "sound" and not pure:
                    continue
                key = call_key(tool, args)
                train_counts[key] += 1
                train_counts_by_source[source][key] += 1
        for rec in test:
            for _t, tool, args in record_calls(rec):
                pure = is_pure(tool)
                if mode == "sound" and not pure:
                    continue
                test_calls[source].append((call_key(tool, args), pure))

    def ranked(counts: Mapping[str, int]) -> List[str]:
        # Deterministic: count desc, then key asc.
        return [k for k, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]

    global_rank = ranked(train_counts)
    per_source_rank = {s: ranked(c) for s, c in train_counts_by_source.items()}

    def score(per_source: bool) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for n in HOT_SET_SIZES:
            hits = calls = 0
            novel = 0
            per = {}
            for source, calls_list in sorted(test_calls.items()):
                rank = per_source_rank[source] if per_source else global_rank
                top = set(rank[:n])
                h = c = nv = 0
                for key, _pure in calls_list:
                    c += 1
                    if key in top:
                        h += 1
                    if key not in train_counts_by_source[source]:
                        nv += 1
                hits += h
                calls += c
                novel += nv
                per[source] = {
                    "calls": c,
                    "hits": h,
                    "hit_rate": round(h / c, 4) if c else 0.0,
                }
            out[str(n)] = {
                "n": n,
                "calls": calls,
                "hits": hits,
                "hit_rate": round(hits / calls, 4) if calls else 0.0,
                "test_keys_unseen_in_train": novel,
                "test_unseen_rate": round(novel / calls, 4) if calls else 0.0,
                "per_source": per,
            }
        return out

    return {
        "mode": mode,
        "split": "per source, first floor(n/2) tasks (first trial) train, remainder test",
        "global": score(per_source=False),
        "per_source": score(per_source=True),
        "train_distinct_keys": len(train_counts),
        "train_top_key_share": (
            round(max(train_counts.values()) / sum(train_counts.values()), 4)
            if train_counts else 0.0
        ),
        "train_top_key_count": max(train_counts.values()) if train_counts else 0,
    }


# --------------------------------------------------------------------------
# Same-turn duplicates and vocabulary
# --------------------------------------------------------------------------
def same_turn_stats(records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Does one assistant turn repeat a call?

    ``turns[t][c]`` is a parallel batch, so two identical siblings in one turn
    are a genuine duplicate the model could have issued once. This measures
    that directly -- and reports the batch-size distribution, because a corpus
    with one call per turn cannot answer the question at all.
    """
    turns = 0
    turns_multi = 0
    calls = 0
    dup_calls = 0
    batch_sizes: Dict[int, int] = defaultdict(int)
    dup_examples = 0
    for rec in records:
        for turn in rec.get("turns") or []:
            turns += 1
            batch_sizes[len(turn)] += 1
            calls += len(turn)
            if len(turn) > 1:
                turns_multi += 1
            keys = [call_key(str(c[0]), c[1]) for c in turn]
            dup = len(keys) - len(set(keys))
            dup_calls += dup
            if dup:
                dup_examples += 1
    return {
        "turns": turns,
        "calls": calls,
        "turns_with_multiple_calls": turns_multi,
        "multi_call_turn_rate": round(turns_multi / turns, 4) if turns else 0.0,
        "batch_size_histogram": {str(k): v for k, v in sorted(batch_sizes.items())},
        "same_turn_duplicate_calls": dup_calls,
        "same_turn_duplicate_rate": round(dup_calls / calls, 4) if calls else 0.0,
        "turns_containing_a_duplicate": dup_examples,
        "identifiable": turns_multi > 0,
        "note": (
            "same-turn repetition is measured, not assumed; when every batch has "
            "one call the rate is structurally 0 and the question is unanswerable "
            "from this corpus"
        ),
    }


def vocabulary_block(streams: "OrderedDict[str, List]", mode: str) -> Dict[str, Any]:
    """Distinct keys per run and the concentration needed to explain reuse."""
    out: Dict[str, Any] = {"mode": mode, "per_source": {}}
    total_keys: set = set()
    total_calls = 0
    for source, recs in streams.items():
        counts: Dict[str, int] = defaultdict(int)
        for rec in recs:
            for _t, tool, args in record_calls(rec):
                if mode == "sound" and not is_pure(tool):
                    continue
                counts[call_key(tool, args)] += 1
        total_calls += sum(counts.values())
        total_keys |= set(counts)
        ranked = sorted(counts.values(), reverse=True)
        cumulative = 0
        cover = {}
        for target in (0.5, 0.8, 0.9):
            for i, c in enumerate(ranked, start=1):
                cumulative += c
                if cumulative >= target * sum(ranked):
                    cover[str(target)] = i
                    break
            cumulative = 0
        out["per_source"][source] = {
            "calls": sum(counts.values()),
            "distinct_keys": len(counts),
            "keys_for_50pct": cover.get("0.5"),
            "keys_for_80pct": cover.get("0.8"),
            "keys_for_90pct": cover.get("0.9"),
        }
    out["calls"] = total_calls
    out["distinct_keys"] = len(total_keys)
    out["calls_per_distinct_key"] = (
        round(total_calls / len(total_keys), 3) if total_keys else 0.0
    )
    return out


def domain_of(source: str) -> str:
    """Coarse benchmark domain, read off the run label (retail / airline).

    Only used to keep the cross-run bonus honest: two backbones running the
    *same* tau-bench task are not two tasks.
    """
    low = source.lower()
    for name in ("retail", "airline"):
        if name in low:
            return name
    return source


def global_origin_decomposition(
    streams: "OrderedDict[str, List]",
) -> Dict[str, Any]:
    """Where do cross-run ("global") hits come from?

    A global cache spans every run in the corpus, so its hit rate mixes three
    very different things: a task repeating itself, the *same* task replayed
    under another backbone, and a genuinely different task reusing a key. Only
    the third is cross-task reuse, and it is the only one comparable to the
    paper's claim. ``task_id`` is a per-domain index, so identity requires the
    same domain *and* the same index.
    """
    cache: Dict[str, Tuple[str, int, str]] = {}
    hits_same_task = hits_same_task_other_run = 0
    hits_other_same_run = hits_other_other_run = 0
    calls = 0
    for source, recs in streams.items():
        for rec in recs:
            tid = int(rec.get("task_id") or 0)
            dom = domain_of(source)
            for _t, tool, args in record_calls(rec):
                if not is_pure(tool):
                    continue
                calls += 1
                key = call_key(tool, args)
                if key in cache:
                    o_source, o_tid, o_dom = cache[key]
                    if (o_dom, o_tid) == (dom, tid) and o_source == source:
                        hits_same_task += 1
                    elif (o_dom, o_tid) == (dom, tid):
                        hits_same_task_other_run += 1
                    elif o_source == source:
                        hits_other_same_run += 1
                    else:
                        hits_other_other_run += 1
                else:
                    # Origin is the FIRST insertion only: a later hit must not
                    # re-home the key, or cross-task reuse reclassifies itself.
                    cache[key] = (source, tid, dom)
    hits = (hits_same_task + hits_same_task_other_run
            + hits_other_same_run + hits_other_other_run)
    denom = calls or 1
    return {
        "calls": calls,
        "hits": hits,
        "hit_rate": round(hits / denom, 4),
        "hits_same_task": hits_same_task,
        "hits_same_task_other_backbone": hits_same_task_other_run,
        "hits_other_task_same_run": hits_other_same_run,
        "hits_other_task_other_run": hits_other_other_run,
        "note": (
            "only the two hits_other_task_* buckets are independent-task reuse; "
            "hits_same_task_other_backbone is one task replayed under a second "
            "model (the same inflation the all-trials sensitivity shows), and "
            "hits_other_task_same_run is already counted by the run scope"
        ),
    }


def split_half_diagnostics(
    streams: "OrderedDict[str, List]",
) -> Dict[str, Any]:
    """What a *frozen* prior can possibly earn on the held-out half.

    Two numbers per run, and the gap between them is the point: how many
    test-half calls use a key already in the train half (the ceiling for any
    static hot set), and how many repeat a key seen earlier *within* the test
    half (reuse that only an online cache can collect). Airline runs show the
    extreme case: a frozen list is worth nothing, because every reusable key is
    first seen inside the half being measured.
    """
    per: Dict[str, Any] = {}
    for source, recs in streams.items():
        train, test = split_half(recs)
        train_keys = set()
        for rec in train:
            for _t, tool, args in record_calls(rec):
                if is_pure(tool):
                    train_keys.add(call_key(tool, args))
        calls = in_train = online = 0
        seen: set = set()
        for rec in test:
            for _t, tool, args in record_calls(rec):
                if not is_pure(tool):
                    continue
                key = call_key(tool, args)
                calls += 1
                in_train += 1 if key in train_keys else 0
                online += 1 if key in seen else 0
                seen.add(key)
        per[source] = {
            "train_tasks": len(train),
            "test_tasks": len(test),
            "train_keys": len(train_keys),
            "test_calls": calls,
            "test_calls_with_train_key": in_train,
            "static_ceiling": round(in_train / calls, 4) if calls else 0.0,
            "test_calls_repeating_within_test": online,
            "online_reuse_rate": round(online / calls, 4) if calls else 0.0,
        }
    return {
        "per_source": per,
        "note": (
            "static_ceiling bounds every hot-set row; online_reuse_rate is what "
            "a cache that learns during the run can still collect"
        ),
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------
CANNOT_EMULATE: List[Dict[str, str]] = [
    {
        "baseline": "semantic / embedding cache",
        "why_not": (
            "The corpus stores parsed (tool, arguments) pairs only. There are no "
            "argument embeddings, no similarity scores and no threshold anywhere "
            "in the record, and no model is reachable offline. Reconstructing "
            "equivalence classes would mean choosing an embedder, a distance "
            "metric and a threshold -- three free parameters that are ours, not "
            "the workload's -- so any hit rate would measure our reconstruction, "
            "not the cache. It is also non-deterministic across embedder "
            "versions, which the paper's reproducibility discipline forbids."
        ),
    },
    {
        "baseline": "provider prefix cache (Anthropic/OpenAI prompt caching)",
        "why_not": (
            "A prefix cache keys on the serialized prompt prefix of a model call "
            "(system prompt + message history) and bills input tokens as cache "
            "reads. The trace keeps neither the prompt text, nor token counts, "
            "nor prefix boundaries, nor per-call timestamps, so prefix identity, "
            "the minimum cacheable prefix (1024/2048 tokens) and the 5-minute TTL "
            "are all unobservable. It also saves input-token cost, not calls: it "
            "is a different denominator from the 41.70% call-level hit rate, and "
            "converting between them would require re-tokenizing reconstructed "
            "prompts -- invention. The paper therefore reports provider prefix "
            "caching only under a MODELED pricing overlay on measured tokens, "
            "which this corpus also lacks."
        ),
    },
    {
        "baseline": "capacity in bytes / TTL-based locality",
        "why_not": (
            "Tool *results* are absent from the corpus, so entry sizes are "
            "unknown; capacity is therefore modeled in entries, not bytes. "
            "Timestamps are absent too, so time-to-live and inter-arrival "
            "locality cannot be evaluated; the LRU sweep is a logical-access "
            "ordering, not a wall-clock trace."
        ),
    },
    {
        "baseline": "end-to-end latency or dollar savings",
        "why_not": (
            "No latency, token or price fields exist in the corpus. Hit rates "
            "are the only defensible offline output; any cost claim would need "
            "a pricing overlay plus measured tokens, neither of which is here."
        ),
    },
]


def build_report(corpus: Path, records: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    first = first_trial(records)
    streams_first = streams_by_source(first)
    streams_all = streams_by_source(records)

    lru: Dict[str, Any] = {}
    for mode in MODES:
        per_mode: Dict[str, Any] = {}
        for scope in SCOPES:
            caps = [0] + list(CAPACITIES)  # 0 = unbounded reference
            per_mode[scope] = _sweep_streams(streams_first, scope, caps, mode)
        lru[mode] = per_mode

    # Anchor: reproduce the published 41.70% / 1.73% under the same key.
    anchor_run = lru["sound"]["run"]["0"]
    anchor_task = lru["sound"]["task"]["0"]
    reproduced_run = anchor_run["hit_rate"]
    reproduced_task = anchor_task["hit_rate"]
    anchor_ok = (
        abs(reproduced_run - PUBLISHED_WITHIN_RUN) <= 0.0005
        and abs(reproduced_task - PUBLISHED_WITHIN_TASK) <= 0.0005
    )

    hot = {mode: hot_set_block(streams_first, mode) for mode in MODES}

    # Per-source cross-task decomposition at unbounded capacity (sound mode).
    # Denominator is the cacheable-call count -- the same denominator the
    # published 41.70% uses -- so the split is directly comparable to it.
    per_source_cross = {}
    for source, recs in streams_first.items():
        c = lru_sweep(recs, 0, "global", "sound")
        denom = c["cacheable_calls"] or 1
        per_source_cross[source] = {
            "calls": c["calls"],
            "cacheable_calls": c["cacheable_calls"],
            "hits": c["hits"],
            "hit_rate": round(c["hits"] / denom, 4),
            "hits_cross_task": c["hits_cross_task"],
            "hits_same_task": c["hits_same_task"],
            "cross_task_hit_rate": round(c["hits_cross_task"] / denom, 4),
            "same_task_hit_rate": round(c["hits_same_task"] / denom, 4),
        }

    # Capacity needed to match the uncapped result, per scope (sound mode).
    def capacity_cliff(scope: str) -> Dict[str, Any]:
        uncapped = lru["sound"][scope]["0"]["hit_rate"]
        first_cap_99 = None
        for cap in sorted(CAPACITIES):
            if uncapped and lru["sound"][scope][str(cap)]["hit_rate"] >= 0.99 * uncapped:
                first_cap_99 = cap
                break
        return {
            "scope": scope,
            "uncapped_hit_rate": uncapped,
            "smallest_capacity_within_1pct_of_uncapped": first_cap_99,
        }

    return {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "self_test": False,
        "description": (
            "Offline external baselines for the real-trace transfer result "
            "(Section 5.12): exact-key LRU at capacities 8..4096 under task, "
            "run and global scopes; a static top-N hot set trained on the first "
            "half and measured on the second; same-turn duplicate rate; and the "
            "exact-key cross-task hit rate. naive = cache everything (unsound on "
            "effectful tools, count reported); sound = side-effect-free only, "
            "the like-for-like comparison with the published 41.70%."
        ),
        "provenance": {
            "corpus": str(corpus.relative_to(REPO_ROOT)),
            "corpus_sha256": corpus_sha256(corpus),
            "records": len(records),
            "first_trial_records": len(first),
            "sources": {s: len(r) for s, r in streams_first.items()},
            "sources_all_trials": {s: len(r) for s, r in streams_all.items()},
            "trials": sorted({int(r.get("trial") or 0) for r in records}),
            "key": 'sha256("tool|" + json.dumps(args, sort_keys=True, default=str))',
            "effect_classification": "imported from real_trace_pilot.TOOL_CLASSES",
            "scopes": {
                "task": "cache flushed at every task attempt",
                "run": "cache shared across the tasks of one source run",
                "global": "one cache across every run in the corpus",
            },
            "capacities": [0] + list(CAPACITIES),
            "capacity_zero_means": "unbounded (reference, matches the published setup)",
        },
        "anchor": {
            "published_within_run": PUBLISHED_WITHIN_RUN,
            "published_within_task": PUBLISHED_WITHIN_TASK,
            "reproduced_within_run": reproduced_run,
            "reproduced_within_task": reproduced_task,
            "reproduced_hits": anchor_run["hits"],
            "reproduced_cacheable_calls": anchor_run["cacheable_calls"],
            "reproduced_calls": anchor_run["calls"],
            "matches_published": anchor_ok,
            "note": (
                "same key, same purity filter, same first-trial records; a "
                "mismatch voids every comparison below"
            ),
        },
        "same_turn": same_turn_stats(first),
        "vocabulary": {mode: vocabulary_block(streams_first, mode) for mode in MODES},
        "lru": lru,
        "hot_set": hot,
        "split_half_diagnostics": split_half_diagnostics(streams_first),
        "cross_task": {
            "per_source_unbounded_sound": per_source_cross,
            "capacity_cliff": [capacity_cliff(s) for s in SCOPES],
        },
        "global_scope_decomposition": global_origin_decomposition(streams_first),
        "cannot_emulate": CANNOT_EMULATE,
    }


# --------------------------------------------------------------------------
# Self-test (never a result)
# --------------------------------------------------------------------------
def _synth_records() -> List[Dict[str, Any]]:
    """Two runs, four tasks, deliberately repeated keys and one duplicate turn."""
    def rec(source, task, trial, turns):
        return {
            "source": source, "task_id": task, "trial": trial, "reward": 1.0,
            "turns": [[c] if not isinstance(c, list) else c for c in turns],
        }

    a = ("get_order_details", {"order_id": "#A1"})
    b = ("get_user_details", {"user_id": "u1"})
    w = ("cancel_pending_order", {"order_id": "#A1"})
    rep = [("get_product_details", {"product_id": "p1"})] * 2
    return [
        rec("s1", 0, 0, [a, b, a, w]),
        rec("s1", 1, 0, [a, b, rep]),          # duplicate siblings in one turn
        rec("s1", 2, 0, [a]),
        rec("s2", 0, 0, [b, b]),
        rec("s2", 1, 0, [a, w]),
        rec("s2", 2, 0, [b]),
    ]


def _self_test() -> int:
    records = _synth_records()
    report = build_report(Path(__file__), records)
    report["self_test"] = True
    report["provenance"]["corpus"] = "<synthetic self-test>"
    report["provenance"]["corpus_sha256"] = "n/a"

    def check(cond: bool, msg: str) -> None:
        if not cond:
            raise AssertionError(msg)

    # LRU monotonically non-decreasing in capacity.
    for mode in MODES:
        for scope in SCOPES:
            prev = -1
            for cap in [0] + list(CAPACITIES):
                r = report["lru"][mode][scope][str(cap)]
                check(r["hits"] >= prev, f"LRU not monotone: {mode}/{scope}/{cap}")
                prev = r["hits"]
                check(r["hits"] <= r["calls"], "hits exceed calls")
                check(r["hits_cross_task"] + r["hits_same_task"] == r["hits"],
                      "hit decomposition must be exhaustive")
    # sound hits never exceed naive hits at the same capacity.
    for scope in SCOPES:
        for cap in ["0"] + [str(c) for c in CAPACITIES]:
            check(report["lru"]["sound"][scope][cap]["hits"]
                  <= report["lru"]["naive"][scope][cap]["hits"],
                  "sound cannot beat naive")
    # cross-task is a strict subset of run-scope hits.
    check(report["lru"]["sound"]["run"]["0"]["hits_cross_task"]
          <= report["lru"]["sound"]["run"]["0"]["hits"], "cross-task subset")
    # hot set: monotone in N, bounded by calls.
    for mode in MODES:
        for trainer in ("global", "per_source"):
            prev = -1
            for n in HOT_SET_SIZES:
                r = report["hot_set"][mode][trainer][str(n)]
                check(r["hits"] >= prev, "hot set not monotone in N")
                check(r["hits"] <= r["calls"], "hot set hits exceed calls")
                prev = r["hits"]
    # same-turn detection works on the synthetic duplicate turn.
    st = report["same_turn"]
    check(st["same_turn_duplicate_calls"] >= 1, "duplicate turn not detected")
    check(st["identifiable"] is True, "multi-call turns should be identifiable")
    # determinism: identical bytes on a second build.
    again = build_report(Path(__file__), records)
    check(json.dumps(report["lru"], sort_keys=True)
          == json.dumps(again["lru"], sort_keys=True), "non-deterministic LRU block")
    check(json.dumps(report["hot_set"], sort_keys=True)
          == json.dumps(again["hot_set"], sort_keys=True), "non-deterministic hot set")
    print("self-test OK (numbers below are NOT results)")
    print(json.dumps(report["same_turn"], indent=1))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)

    if args.self_test:
        return _self_test()

    records = load_records(args.corpus)
    report = build_report(args.corpus, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=1, sort_keys=False) + "\n", encoding="utf-8")

    lru = report["lru"]["sound"]
    print(f"records {len(records)}  first-trial {report['provenance']['first_trial_records']}")
    print(f"anchor reproduced: run {report['anchor']['reproduced_within_run']:.2%} "
          f"task {report['anchor']['reproduced_within_task']:.2%} "
          f"({report['anchor']['matches_published']})")
    for scope in SCOPES:
        print(f"  sound/{scope}: uncapped {lru[scope]['0']['hit_rate']:.2%}  "
              f"@256 {lru[scope]['256']['hit_rate']:.2%}  "
              f"@8 {lru[scope]['8']['hit_rate']:.2%}")
    for mode in MODES:
        row = "  ".join(
            f"N={n}:{report['hot_set'][mode]['global'][str(n)]['hit_rate']:.2%}"
            for n in HOT_SET_SIZES)
        print(f"hot-set/global/{mode}: {row}")
    print(f"same-turn duplicates: {report['same_turn']['same_turn_duplicate_calls']} "
          f"(multi-call turns {report['same_turn']['turns_with_multiple_calls']})")
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
