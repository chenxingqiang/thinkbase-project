#!/usr/bin/env python3
"""Task-clustered percentile-bootstrap intervals for the unsafe-cache safety table.

The safety table reports four numbers on the committed tau-bench corpus: the
blind policy's unsafe-hit rate per 1,000 calls and its unsafe share of hits, the
typed policy's hit rate on admitted reads, and the blind-minus-typed difference
in hit rate. Those are point estimates over one sample of 325 first-trial
trajectories. This script puts an interval around each of them, plus the
deployment-shaped view the table leaves out: how many *tasks* contain at least
one unsafe hit, and what share of tasks that is.

The interval resamples tasks, not calls. That choice is the whole method and it
is deliberate:

  * A task is one first-trial trajectory (one record), labelled
    ``source#task_id`` -- 325 clusters over the four runs. The same ``task_id``
    replayed by a different model is a different trajectory and a different
    cluster.
  * Calls inside one task are dependent. They share a trajectory, a user, a
    tenant and an evolving cache: a hit at turn 3 changes what turns 4-9 cost,
    and an unsafe hit is a property of a *sequence* (a write that lands before
    a stale read), not of a call in isolation. Resampling calls would treat
    those as independent draws, shrink the interval and overstate precision.
  * Resampling tasks keeps every task's call sequence intact and makes the unit
    of variation the one a deployment actually experiences: "if I drew another
    batch of tasks like these, how much would this number move?"

This is the standard cluster bootstrap (Efron's percentile method, 10,000
resamples, seed 20240601, 2.5/97.5 percentiles). It is paired: one resample of
tasks is used for every metric, so differences are computed within a resample,
not by subtracting two independently bootstrapped intervals.

What it does *not* model, stated rather than buried: each run keeps its own
run-scope cache, so a task's outcomes also depend on the tasks before it in the
same run. The cluster bootstrap captures within-task dependence; that
cross-task carryover remains, and the intervals should be read as slightly
optimistic for that reason. A block bootstrap over whole runs would need four
blocks, which is not enough to estimate anything. The percentile interval is
also not bias-corrected (no BCa); with 325 clusters and these rates the bias is
small, but it is a choice, not an accident.

Provenance and reuse. The corpus loader is
``real_trace_external_baselines.load_records`` / ``first_trial`` /
``task_id_of``; the streams, the policies, the identity module, the served-key
recipe and the staleness join come from ``unsafe_cache_baselines``. That
module's ``replay()`` returns totals only -- it keeps no per-call label, so it
cannot be clustered -- so this script re-walks the same call loop one call at a
time, reusing the module's own helpers, and then proves equivalence: the walk's
totals must equal ``replay()``'s, and the blind walk must equal the committed
807 hits / 344 unsafe out of 2,275 calls (and typed 466 / 1,728) before any
interval is computed. If a check fails the script exits non-zero and writes
nothing.

Usage:

    python3 examples/benchmarks/safety_ci.py
    python3 examples/benchmarks/safety_ci.py --output /tmp/safety_ci.json

Deterministic: fixed seed and resample count, no timestamps, no host facts,
ordered keys, and counts kept as integers next to their derived rates. Two runs
produce byte-identical JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = HERE / "safety_ci_results.json"
UNSAFE_MODULE_PATH = HERE / "unsafe_cache_baselines.py"
EXTERNAL_MODULE_PATH = HERE / "real_trace_external_baselines.py"
COMMITTED_RESULTS = HERE / "unsafe_cache_results.json"

SCHEMA = "safety_ci/v1"
TASK = (
    "L2 - task-clustered percentile-bootstrap intervals for the unsafe-cache "
    "safety table, resampling the 325 first-trial tau-bench tasks"
)
RESAMPLES = 10_000
SEED = 20240601
CONFIDENCE = 0.95
ALPHA = (1.0 - CONFIDENCE) / 2.0

# The committed counts these intervals are issued against, read out of
# unsafe_cache_results.json (schema unsafe_cache_baselines/v1) and hard-coded
# here so the assertion does not depend on that file being present. Blind:
# 807 hits, of which 344 unsafe (61 writes served + 179 stale reads + 104
# cross-tenant), over 2,275 calls. Typed: 466 hits over 1,728 admitted reads,
# 0 unsafe by construction.
BLIND_CALLS = 2275
BLIND_HITS = 807
BLIND_UNSAFE = 344
BLIND_BREAKDOWN = OrderedDict(writes_served=61, stale_reads=179, cross_tenant=104)
TYPED_ADMITTED = 1728
TYPED_HITS = 466

# Per-call outcome of one policy on one call.
CallOutcome = Tuple[str, str, bool, bool, Any]  # cluster, effect, admitted, hit, unsafe class

SERIES = (
    "blind_unsafe_hits_per_1000_calls",
    "blind_unsafe_share_of_hits",
    "typed_hit_rate_on_admitted_reads",
    "hit_rate_difference_blind_minus_typed",
    "blind_hit_rate_on_admitted_reads",
    "hit_rate_difference_same_base",
    "tasks_with_unsafe_hit",
    "share_of_tasks_with_unsafe_hit",
)


# ---------------------------------------------------------------------------
# Modules, loaded by path: this directory is not a package, and loading by path
# is what the measurement modules themselves do.
# ---------------------------------------------------------------------------
def load_module(name: str, path: Path):
    if not path.is_file():
        raise SystemExit(f"safety_ci: cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"safety_ci: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - artifact checked out under a symlink
        return str(path.resolve())


def require_attrs(module, names: Sequence[str], where: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise SystemExit(
            f"safety_ci: {where} does not expose {missing}. The per-call walk must "
            "reuse that module's key, eviction and staleness helpers; re-deriving "
            "them here is how this table would drift from the cache it measures."
        )


# ---------------------------------------------------------------------------
# Per-call replay
# ---------------------------------------------------------------------------
def walk(
    streams: "OrderedDict[str, List[Any]]",
    policy,
    ci,
    cluster_of: Mapping[Tuple[str, str], str],
    unsafe,
) -> List[CallOutcome]:
    """Replay one policy call by call, keeping the label ``replay()`` discards.

    The control flow is ``unsafe_cache_baselines.replay``'s, line for line,
    including the eviction-before-lookup for refused writes and the
    ``writes_served > stale_reads > cross_tenant`` precedence. The only change
    is that each call's outcome is appended instead of folded into a Counter;
    ``verify`` below asserts the fold of these rows equals ``replay()``'s.
    """
    rows: List[CallOutcome] = []
    served_key = unsafe._served_key
    intervening = unsafe._intervening
    evict = unsafe._evict

    for source, calls in streams.items():
        cache: Dict[str, Dict[str, Any]] = {}
        index: Dict[str, set] = {}
        key_map: Dict[str, str] = {}
        write_log: List[Tuple[int, Tuple[str, ...], str, Dict[str, Any]]] = []

        for call in calls:
            cluster = cluster_of[(source, str(call.task_id))]
            effect = ci.effect_class(call.tool)

            if not policy.memoises_writes and effect != "read":
                if effect == "write":
                    evict(ci, index, key_map, cache, call)
                    write_log.append(
                        (call.pos, ci.entity_keys(call.tool, call.args), call.tool, call.args)
                    )
                rows.append((cluster, effect, False, False, None))
                continue

            key = served_key(ci, policy, call)
            memo = cache.get(key)
            if memo is not None:
                unsafe_class = None
                if effect == "write":
                    unsafe_class = "writes_served"
                elif effect == "read":
                    targets = frozenset(ci.entity_keys(call.tool, call.args)) or frozenset(
                        memo["entity_ids"]
                    )
                    stale = intervening(write_log, memo["pos"], call.pos, targets)
                    cross = (
                        memo["tenant"] is not None
                        and call.tenant is not None
                        and memo["tenant"] != call.tenant
                    )
                    if stale:
                        unsafe_class = "stale_reads"
                    elif cross:
                        unsafe_class = "cross_tenant"
                rows.append((cluster, effect, True, True, unsafe_class))
                continue

            cache[key] = OrderedDict(
                pos=call.pos,
                tool=call.tool,
                args=call.args,
                tenant=call.tenant,
                entity_ids=ci.entity_keys(call.tool, call.args),
            )
            if policy.key_mode == "typed":
                key_map[ci.remember(index, call.tool, call.args, tenant=call.tenant)] = key
            elif policy.invalidation == "entity" and effect == "read":
                key_map[ci.remember(index, call.tool, call.args, tenant=call.tenant)] = key
            if effect == "write":
                write_log.append(
                    (call.pos, ci.entity_keys(call.tool, call.args), call.tool, call.args)
                )
                if policy.invalidation == "entity":
                    evict(ci, index, key_map, cache, call)
            rows.append((cluster, effect, True, False, None))
    return rows


def fold(rows: Sequence[CallOutcome]) -> Dict[str, Any]:
    """Aggregate per-call labels exactly the way replay() does (same precedence)."""
    cacheable = sum(1 for r in rows if r[2])
    hits = sum(1 for r in rows if r[3])
    read_calls = sum(1 for r in rows if r[2] and r[1] == "read")
    read_hits = sum(1 for r in rows if r[2] and r[1] == "read" and r[3])
    classes = Counter(r[4] for r in rows if r[4])
    return {
        "calls_total": len(rows),
        "cacheable_calls": cacheable,
        "hits": hits,
        "read_calls": read_calls,
        "read_hits": read_hits,
        "unsafe_hits": sum(classes.values()),
        "writes_served": classes["writes_served"],
        "stale_reads": classes["stale_reads"],
        "cross_tenant": classes["cross_tenant"],
    }


def verify(rows: Sequence[CallOutcome], replayed: Mapping[str, Any], policy_name: str) -> None:
    """The walk must reproduce the module's own replay, field for field."""
    got = fold(rows)
    expected = {
        "calls_total": replayed["calls_total"],
        "cacheable_calls": replayed["cacheable_calls"],
        "hits": replayed["hits"],
        "unsafe_hits": replayed["unsafe_hits"],
        "writes_served": replayed["unsafe_breakdown"]["writes_served"],
        "stale_reads": replayed["unsafe_breakdown"]["stale_reads"],
        "cross_tenant": replayed["unsafe_breakdown"]["cross_tenant"],
    }
    diff = {k: (got[k], expected[k]) for k in expected if got[k] != expected[k]}
    if diff:
        raise SystemExit(
            f"safety_ci: per-call walk disagrees with replay() for {policy_name!r}: "
            f"{diff} (walk, replay). Refusing to emit intervals built on a diverging "
            "replay."
        )


def committed_numbers(unsafe_module) -> Dict[str, Any]:
    """The committed 807/344 (and 466/1728), checked against the artifact too."""
    blind = next((p for p in unsafe_module.POLICIES if p.name == "blind"), None)
    typed = next((p for p in unsafe_module.POLICIES if p.name == "typed"), None)
    if blind is None or typed is None:
        raise SystemExit("safety_ci: unsafe_cache_baselines no longer defines blind/typed")

    checks: Dict[str, Any] = OrderedDict()
    if COMMITTED_RESULTS.is_file():
        doc = json.loads(COMMITTED_RESULTS.read_text(encoding="utf-8"))
        committed = {p.get("name"): p for p in doc.get("policies", [])}
        for name, calls, hits, unsafe in (
            ("blind", BLIND_CALLS, BLIND_HITS, BLIND_UNSAFE),
            ("typed", TYPED_ADMITTED, TYPED_HITS, 0),
        ):
            entry = committed.get(name)
            if entry is None:
                raise SystemExit(
                    f"safety_ci: {COMMITTED_RESULTS.name} has no policy {name!r}; the "
                    "committed counts this table is issued against are missing."
                )
            if (entry.get("calls"), entry.get("hits"), entry.get("unsafe_hits")) != (
                calls,
                hits,
                unsafe,
            ):
                raise SystemExit(
                    f"safety_ci: committed {name} counts "
                    f"{(entry.get('calls'), entry.get('hits'), entry.get('unsafe_hits'))} "
                    f"do not match the expected {(calls, hits, unsafe)}."
                )
        checks["committed_json"] = OrderedDict(
            path=rel(COMMITTED_RESULTS),
            sha256=sha256_file(COMMITTED_RESULTS),
            matches=True,
        )
    else:  # pragma: no cover - the artifact is committed
        checks["committed_json"] = None
    return checks


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def draw(rng: random.Random, n: int) -> int:
    """One unbiased uniform index in [0, n), from documented 64-bit draws.

    Uses rejection on getrandbits(64) rather than randrange(): the output then
    depends only on the Mersenne Twister bit stream, which is stable across
    CPython versions, not on the internals of randrange's _randbelow.
    """
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        value = rng.getrandbits(64)
        if value < limit:
            return value % n


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile, the rule NumPy calls 'linear'.

    h = (n - 1) * p; the value at floor(h) plus the fraction to the next one.
    Stated here because an interval without a stated percentile rule is not
    reproducible.
    """
    last = len(sorted_values) - 1
    h = last * p
    lo = int(math.floor(h))
    hi = min(lo + 1, last)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (h - lo)


def bootstrap(
    matrix: Sequence[Tuple[int, int, int, int, int, int, int, int]],
    resamples: int,
    seed: int,
) -> Tuple[Dict[str, List[float]], int]:
    """Resample tasks with replacement; derive every metric within a resample.

    ``matrix`` rows are per-task sums: (calls, hits, unsafe, read_calls,
    read_hits, typed_admitted, typed_hits, has_unsafe). One draw of 325 tasks
    feeds every series, so the difference of hit rates is paired -- both
    policies see the same resampled tasks.
    """
    n = len(matrix)
    rng = random.Random(seed)
    series: Dict[str, List[float]] = {name: [] for name in SERIES}
    used = 0

    for _ in range(resamples):
        calls = hits = unsafe = reads = read_hits = admitted = typed_hits = affected = 0
        for _ in range(n):
            c, h, u, rc, rh, ta, th, af = matrix[draw(rng, n)]
            calls += c
            hits += h
            unsafe += u
            reads += rc
            read_hits += rh
            admitted += ta
            typed_hits += th
            affected += af
        if calls == 0 or admitted == 0:  # pragma: no cover - every task has calls
            continue
        used += 1
        series["blind_unsafe_hits_per_1000_calls"].append(1000.0 * unsafe / calls)
        series["blind_unsafe_share_of_hits"].append(unsafe / hits if hits else 0.0)
        series["typed_hit_rate_on_admitted_reads"].append(typed_hits / admitted)
        series["hit_rate_difference_blind_minus_typed"].append(
            hits / calls - typed_hits / admitted
        )
        series["blind_hit_rate_on_admitted_reads"].append(read_hits / reads if reads else 0.0)
        series["hit_rate_difference_same_base"].append(
            (read_hits / reads if reads else 0.0) - typed_hits / admitted
        )
        series["tasks_with_unsafe_hit"].append(float(affected))
        series["share_of_tasks_with_unsafe_hit"].append(affected / n)
    return series, used


# ---------------------------------------------------------------------------
# Estimates and the document
# ---------------------------------------------------------------------------
def task_matrix(
    blind_rows: Sequence[CallOutcome], typed_rows: Sequence[CallOutcome], clusters: Sequence[str]
) -> Tuple[
    List[Tuple[int, int, int, int, int, int, int, int]],
    Dict[str, Dict[str, int]],
    Dict[str, int],
]:
    blind_by = {c: [0, 0, 0, 0, 0] for c in clusters}
    typed_by = {c: [0, 0] for c in clusters}
    class_tasks = {name: set() for name in ("writes_served", "stale_reads", "cross_tenant")}

    for cluster, effect, admitted, hit, unsafe_class in blind_rows:
        agg = blind_by[cluster]
        agg[0] += 1
        agg[1] += int(hit)
        agg[2] += int(unsafe_class is not None)
        if effect == "read":
            agg[3] += 1
            agg[4] += int(hit)
        if unsafe_class is not None:
            class_tasks[unsafe_class].add(cluster)

    for cluster, _effect, admitted, hit, _unsafe_class in typed_rows:
        if admitted:
            agg = typed_by[cluster]
            agg[0] += 1
            agg[1] += int(hit)

    matrix = []
    per_task = {}
    for cluster in clusters:
        b = blind_by[cluster]
        t = typed_by[cluster]
        has_unsafe = int(b[2] > 0)
        matrix.append((b[0], b[1], b[2], b[3], b[4], t[0], t[1], has_unsafe))
        per_task[cluster] = {
            "calls": b[0],
            "hits": b[1],
            "unsafe_hits": b[2],
            "read_calls": b[3],
            "read_hits": b[4],
            "typed_admitted": t[0],
            "typed_hits": t[1],
        }
    class_counts = {name: len(tasks) for name, tasks in class_tasks.items()}
    return matrix, per_task, class_counts


def number(value: float, decimals: int) -> Any:
    """Round for publication; whole numbers when ``decimals`` is 0 (task counts)."""
    return int(round(value)) if decimals == 0 else round(value, decimals)


def interval(
    values: Sequence[float], estimate: float, decimals: int, unit: str
) -> Dict[str, Any]:
    ordered = sorted(values)
    return OrderedDict(
        estimate=number(estimate, decimals),
        ci_low=number(percentile(ordered, ALPHA), decimals),
        ci_high=number(percentile(ordered, 1.0 - ALPHA), decimals),
        unit=unit,
        resamples=len(ordered),
    )


def build_document(
    *,
    records: Sequence[Mapping[str, Any]],
    first: Sequence[Mapping[str, Any]],
    streams,
    clusters: Sequence[str],
    per_task: Mapping[str, Mapping[str, int]],
    blind_fold: Mapping[str, Any],
    typed_fold: Mapping[str, Any],
    series: Mapping[str, Sequence[float]],
    used: int,
    resamples: int,
    seed: int,
    corpus_sha: str,
    class_counts: Mapping[str, int],
    checks: Mapping[str, Any],
    shas: Mapping[str, str],
) -> Dict[str, Any]:
    calls = blind_fold["calls_total"]
    hits = blind_fold["hits"]
    unsafe = blind_fold["unsafe_hits"]
    reads = typed_fold["cacheable_calls"]
    typed_hits = typed_fold["hits"]
    affected = sum(1 for c in clusters if per_task[c]["unsafe_hits"] > 0)
    n_tasks = len(clusters)

    blind_rate_1k = 1000.0 * unsafe / calls
    blind_share = unsafe / hits
    typed_rate = typed_hits / reads
    difference = hits / calls - typed_rate
    read_hits = sum(per_task[c]["read_hits"] for c in clusters)
    blind_read_rate = read_hits / reads
    same_base = blind_read_rate - typed_rate
    affected_share = affected / n_tasks

    points = OrderedDict(
        blind=OrderedDict(
            calls=calls,
            hits=hits,
            unsafe_hits=unsafe,
            unsafe_breakdown=OrderedDict(
                writes_served=blind_fold["writes_served"],
                stale_reads=blind_fold["stale_reads"],
                cross_tenant=blind_fold["cross_tenant"],
            ),
            unsafe_hits_per_1000_calls=round(blind_rate_1k, 6),
            unsafe_share_of_hits=round(blind_share, 6),
            hit_rate=round(hits / calls, 6),
            read_calls=reads,
            read_hits=read_hits,
        ),
        typed=OrderedDict(
            admitted_reads=reads,
            hits=typed_hits,
            hit_rate_on_admitted_reads=round(typed_rate, 6),
            unsafe_hits=typed_fold["unsafe_hits"],
        ),
        difference=OrderedDict(
            blind_minus_typed=round(difference, 6),
            same_base_read_only=round(same_base, 6),
            note=(
                "blind_minus_typed subtracts rates on different denominators (all 2,275 "
                "admissible calls vs the 1,728 admitted reads) because that is the pair the "
                "safety table publishes; same_base_read_only restricts the blind rate to the "
                "same read calls and is the like-for-like comparison"
            ),
        ),
        tasks=OrderedDict(
            total=n_tasks,
            with_any_unsafe_hit=affected,
            share_with_any_unsafe_hit=round(affected_share, 6),
            with_unsafe_hit_by_class=OrderedDict(class_counts),
            typed_with_any_unsafe_hit=0,
            note=(
                "unsafe hits are the blind policy's; typed produces none by construction "
                "(writes refused, entity invalidation), so its affected-task count is 0"
            ),
        ),
    )

    estimates = OrderedDict(
        blind_unsafe_hits_per_1000_calls=(blind_rate_1k, 3, "unsafe hits per 1,000 calls"),
        blind_unsafe_share_of_hits=(blind_share, 6, "share of hits"),
        typed_hit_rate_on_admitted_reads=(typed_rate, 6, "share of admitted reads"),
        hit_rate_difference_blind_minus_typed=(difference, 6, "difference in hit rate"),
        blind_hit_rate_on_admitted_reads=(blind_read_rate, 6, "share of read calls"),
        hit_rate_difference_same_base=(same_base, 6, "difference in hit rate"),
        tasks_with_unsafe_hit=(float(affected), 0, "tasks"),
        share_of_tasks_with_unsafe_hit=(affected_share, 6, "share of tasks"),
    )
    intervals = OrderedDict()
    for name in SERIES:
        estimate, decimals, unit = estimates[name]
        intervals[name] = interval(series[name], estimate, decimals, unit)

    return OrderedDict(
        schema=SCHEMA,
        task=TASK,
        result_kind="BOOTSTRAP",
        scope=(
            "tau-bench only: the first trial of every task in the committed corpus, four "
            "source runs, replayed through the blind and typed run-scope caches of "
            "unsafe_cache_baselines.POLICIES. The interval is over tasks."
        ),
        method=OrderedDict(
            cluster_unit=(
                "task: one first-trial trajectory, labelled source#task_id; the same task_id "
                "replayed by another model is a different trajectory and a different cluster"
            ),
            clusters=n_tasks,
            cluster_sizes_calls=OrderedDict(
                min=min(per_task[c]["calls"] for c in clusters),
                max=max(per_task[c]["calls"] for c in clusters),
                mean=round(calls / n_tasks, 4),
            ),
            resamples=resamples,
            resamples_used=used,
            seed=seed,
            replacement=True,
            confidence=CONFIDENCE,
            percentiles=[round(100 * ALPHA, 3), round(100 * (1 - ALPHA), 3)],
            percentile_rule=(
                "linear interpolation on the sorted bootstrap draws: h = (n - 1) * p, value "
                "at floor(h) plus the fraction to the next"
            ),
            rng=(
                "CPython random.Random (Mersenne Twister); indices drawn with rejection on "
                "getrandbits(64), not randrange(), so the draw depends only on the bit stream"
            ),
            why_tasks_not_calls=(
                "Calls within one task are dependent: they share a trajectory, a user, a "
                "tenant and an evolving cache, and an unsafe hit is a property of a sequence "
                "(a write landing before a stale read), not of a call in isolation. Resampling "
                "calls would treat the calls of one task as independent, shrink the interval "
                "and overstate precision. Resampling whole tasks keeps each task's call "
                "sequence intact and makes the unit of variation the deployment's: another "
                "batch of tasks like these."
            ),
            paired=True,
            pairing_note=(
                "one task resample feeds every metric, so blind-minus-typed is computed "
                "within a resample; the two policies see the same tasks"
            ),
        ),
        corpus=OrderedDict(
            path=rel(CORPUS),
            sha256=corpus_sha,
            records=len(records),
            first_trial_records=len(first),
            runs=len(streams),
            calls=calls,
            tasks=n_tasks,
            first_trial_only=True,
        ),
        provenance=OrderedDict(
            script=OrderedDict(
                path=rel(Path(__file__)),
                sha256=sha256_file(Path(__file__).resolve()),
                schema=SCHEMA,
            ),
            corpus_sha256=corpus_sha,
            identity_module=OrderedDict(
                path="examples/benchmarks/cache_identity.py", sha256=shas["identity"]
            ),
            unsafe_module=OrderedDict(
                path=rel(UNSAFE_MODULE_PATH), sha256=shas["unsafe"]
            ),
            external_module=OrderedDict(
                path=rel(EXTERNAL_MODULE_PATH), sha256=shas["external"]
            ),
        ),
        definitions=OrderedDict(
            unsafe_hit=(
                "a hit that serves a write (writes_served), a read whose entity was written "
                "after the memo was stored (stale_reads), or a read produced under a "
                "different known tenant (cross_tenant); one class per hit, by that precedence"
            ),
            hit_rate=(
                "hits / admitted calls: all 2,275 calls for blind (it memoises writes too), "
                "only the 1,728 reads for typed (it refuses writes and control tools)"
            ),
            task=(
                "one first-trial record, source#task_id; 325 over four runs. Calls within a "
                "task are the resampling cluster."
            ),
            affected_task=(
                "a task with at least one unsafe hit from the blind policy; the typed view is "
                "reported as 0 for completeness, not as evidence of safety on other workloads"
            ),
        ),
        point=points,
        intervals=intervals,
        checks=OrderedDict(
            blind_equals_committed=OrderedDict(
                expected="807 hits, 344 unsafe, 2,275 calls",
                observed=f"{hits:,} hits, {unsafe:,} unsafe, {calls:,} calls",
                passed=(hits, unsafe, calls) == (BLIND_HITS, BLIND_UNSAFE, BLIND_CALLS),
            ),
            typed_equals_committed=OrderedDict(
                expected="466 hits, 1,728 admitted reads, 0 unsafe",
                observed=(
                    f"{typed_hits:,} hits, {reads:,} admitted reads, "
                    f"{typed_fold['unsafe_hits']:,} unsafe"
                ),
                passed=(typed_hits, reads, typed_fold["unsafe_hits"]) == (TYPED_HITS, TYPED_ADMITTED, 0),
            ),
            walk_equals_module_replay=True,
            committed_json=checks["committed_json"],
            corpus_sha256_pinned=(corpus_sha == "3b5d4d29989ad99ed276433027d6ede37acb1197635e4a99febf44f6b5e64116"),
        ),
        notes=[
            "The interval resamples tasks, not calls, because calls within a task are dependent.",
            (
                "Each run keeps its own run-scope cache, so a task's outcomes also depend on the "
                "tasks before it in that run. The cluster bootstrap captures within-task "
                "dependence; that cross-task carryover remains, making the intervals slightly "
                "optimistic. Blocking on runs is not an option: four blocks cannot support an "
                "interval."
            ),
            "Percentile interval, not BCa: no bias correction or acceleration is applied.",
            (
                "The unsafe share of hits is conditioned on the blind policy's 807 hits; it "
                "answers 'how much of what the cache serves would I have to trust', not 'how "
                "often does a call go wrong'."
            ),
            (
                "The per-task view is the deployment-facing one: an unsafe hit on "
                f"{100.0 * unsafe / calls:.1f}% of calls still means "
                f"{100.0 * affected_share:.1f}% of tasks contain one."
            ),
        ],
        reproduce=(
            "python3 examples/benchmarks/safety_ci.py  (seed 20240601, 10,000 resamples, "
            "byte-identical output on repeat)"
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task-clustered bootstrap intervals for the unsafe-cache safety table."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="results JSON (default: safety_ci_results.json beside this script)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if args.resamples <= 0:
        print("safety_ci: error: --resamples must be positive", file=sys.stderr)
        return 2

    try:
        external = load_module("_safety_external", EXTERNAL_MODULE_PATH)
        unsafe = load_module("_safety_unsafe", UNSAFE_MODULE_PATH)
        require_attrs(
            external,
            ("load_records", "first_trial", "task_id_of"),
            "real_trace_external_baselines",
        )
        require_attrs(
            unsafe,
            ("POLICIES", "build_streams", "replay", "load_identity", "_served_key",
             "_intervening", "_evict"),
            "unsafe_cache_baselines",
        )

        ci, identity_sha = unsafe.load_identity()
        records = external.load_records(CORPUS)
        first = external.first_trial(records)
        streams = unsafe.build_streams(first)
        corpus_sha = (
            external.corpus_sha256(CORPUS) if hasattr(external, "corpus_sha256")
            else sha256_file(CORPUS)
        )
        pinned = getattr(unsafe, "EXPECTED_SHA256", None)
        if pinned is not None and corpus_sha != pinned:
            raise SystemExit(
                f"safety_ci: corpus sha256 {corpus_sha} does not match the hash "
                f"unsafe_cache_baselines pins ({pinned}). The committed 807/344 is tied to "
                "those exact bytes; a different corpus is a different experiment."
            )

        cluster_of: Dict[Tuple[str, str], str] = {}
        clusters: List[str] = []
        for record in first:
            key = (str(record.get("source")), str(record.get("task_id")))
            label = external.task_id_of(record)
            if key not in cluster_of:
                cluster_of[key] = label
                clusters.append(label)
        for source, calls in streams.items():
            for call in calls:
                if (source, str(call.task_id)) not in cluster_of:
                    raise SystemExit(
                        f"safety_ci: call in {source!r} names task {call.task_id!r} with no "
                        "first-trial record; the cluster label cannot be assigned."
                    )

        blind = next(p for p in unsafe.POLICIES if p.name == "blind")
        typed = next(p for p in unsafe.POLICIES if p.name == "typed")

        blind_rows = walk(streams, blind, ci, cluster_of, unsafe)
        typed_rows = walk(streams, typed, ci, cluster_of, unsafe)
        verify(blind_rows, unsafe.replay(streams, blind, ci), "blind")
        verify(typed_rows, unsafe.replay(streams, typed, ci), "typed")

        blind_fold = fold(blind_rows)
        typed_fold = fold(typed_rows)
        checks = committed_numbers(unsafe)

        # The hard assertion the task requires: the committed 807/344 before use.
        if (blind_fold["hits"], blind_fold["unsafe_hits"], blind_fold["calls_total"]) != (
            BLIND_HITS,
            BLIND_UNSAFE,
            BLIND_CALLS,
        ):
            raise SystemExit(
                "safety_ci: blind replay is "
                f"{blind_fold['hits']}/{blind_fold['unsafe_hits']}/{blind_fold['calls_total']} "
                f"(hits/unsafe/calls), not the committed {BLIND_HITS}/{BLIND_UNSAFE}/{BLIND_CALLS}. "
                "Refusing to emit intervals."
            )
        if OrderedDict(
            (k, blind_fold[k]) for k in ("writes_served", "stale_reads", "cross_tenant")
        ) != BLIND_BREAKDOWN:
            raise SystemExit(
                "safety_ci: blind unsafe breakdown is "
                f"{dict((k, blind_fold[k]) for k in BLIND_BREAKDOWN)}, not the committed "
                f"{dict(BLIND_BREAKDOWN)}. Refusing to emit intervals."
            )
        if (typed_fold["hits"], typed_fold["cacheable_calls"], typed_fold["unsafe_hits"]) != (
            TYPED_HITS,
            TYPED_ADMITTED,
            0,
        ):
            raise SystemExit(
                "safety_ci: typed replay is "
                f"{typed_fold['hits']}/{typed_fold['cacheable_calls']}/{typed_fold['unsafe_hits']}, "
                f"not the committed {TYPED_HITS}/{TYPED_ADMITTED}/0. Refusing to emit intervals."
            )
        if blind_fold["read_calls"] != typed_fold["cacheable_calls"]:
            raise SystemExit(
                "safety_ci: blind read calls "
                f"({blind_fold['read_calls']}) != typed admitted reads "
                f"({typed_fold['cacheable_calls']}); the two policies disagree on which calls "
                "are reads."
            )

        matrix, per_task, class_counts = task_matrix(blind_rows, typed_rows, clusters)
        series, used = bootstrap(matrix, args.resamples, args.seed)

        document = build_document(
            records=records,
            first=first,
            streams=streams,
            clusters=clusters,
            per_task=per_task,
            blind_fold=blind_fold,
            typed_fold=typed_fold,
            series=series,
            used=used,
            resamples=args.resamples,
            seed=args.seed,
            corpus_sha=corpus_sha,
            class_counts=class_counts,
            checks=checks,
            shas=OrderedDict(
                identity=identity_sha,
                unsafe=sha256_file(UNSAFE_MODULE_PATH),
                external=sha256_file(EXTERNAL_MODULE_PATH),
            ),
        )

        out = args.output.expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(out)

        p = document["point"]
        meta = document["method"]
        print(
            f"safety_ci: {meta['clusters']} tasks, {document['corpus']['calls']:,} calls, "
            f"{meta['resamples']:,} task resamples (seed {meta['seed']})\n"
            f"  blind unsafe hits/1k calls      {p['blind']['unsafe_hits_per_1000_calls']:.2f}  "
            f"[{document['intervals']['blind_unsafe_hits_per_1000_calls']['ci_low']:.2f}, "
            f"{document['intervals']['blind_unsafe_hits_per_1000_calls']['ci_high']:.2f}]\n"
            f"  blind unsafe share of hits      {p['blind']['unsafe_share_of_hits']:.4f}  "
            f"[{document['intervals']['blind_unsafe_share_of_hits']['ci_low']:.4f}, "
            f"{document['intervals']['blind_unsafe_share_of_hits']['ci_high']:.4f}]\n"
            f"  typed hit rate (admitted)       {p['typed']['hit_rate_on_admitted_reads']:.4f}  "
            f"[{document['intervals']['typed_hit_rate_on_admitted_reads']['ci_low']:.4f}, "
            f"{document['intervals']['typed_hit_rate_on_admitted_reads']['ci_high']:.4f}]\n"
            f"  difference (blind - typed)      {p['difference']['blind_minus_typed']:+.4f}  "
            f"[{document['intervals']['hit_rate_difference_blind_minus_typed']['ci_low']:+.4f}, "
            f"{document['intervals']['hit_rate_difference_blind_minus_typed']['ci_high']:+.4f}]\n"
            f"  tasks with an unsafe hit        {p['tasks']['with_any_unsafe_hit']}/{p['tasks']['total']} "
            f"= {p['tasks']['share_with_any_unsafe_hit']:.4f}  "
            f"[{document['intervals']['share_of_tasks_with_unsafe_hit']['ci_low']:.4f}, "
            f"{document['intervals']['share_of_tasks_with_unsafe_hit']['ci_high']:.4f}]\n"
            f"  wrote {out}"
        )
        return 0
    except json.JSONDecodeError as exc:  # pragma: no cover - corrupt input
        print(f"safety_ci: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
