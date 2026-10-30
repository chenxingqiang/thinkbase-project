#!/usr/bin/env python3
"""Damage-based unsafe accounting: reclassify the committed unsafe hits, then bound them.

The objection this script answers. ``unsafe_cache_baselines`` counts a hit as
unsafe when it serves a write, a read whose entity was written after the memo was
stored, or a read produced under a different tenant -- and those are exactly the
three components of ``cache_identity.key_typed``: effect, entity/freshness,
identity. The typed policy's zero is therefore close to analytic: it is the
predicate's own definition applied to the policy the predicate was written for.
That does not make the number wrong, but it does make it uninformative about
*harm*: a cache cannot damage an answer it was never asked for, and some of the
committed "unsafe" hits cannot damage anything at all.

So this script keeps the replay (the hits are the committed ones, reproduced
through the module's own key, eviction and staleness helpers) and replaces the
*classification* with one that never mentions the typed key:

  D1  write served from cache
      The mutation never ran, while the agent's transcript says it did and the
      agent continues as if the world changed. Damage is the missing side
      effect, independent of any key design. This is the committed
      ``writes_served`` class, unchanged.

  D2  read served across a landed write to the same entity
      The memo predates a write that *executed* (was not itself served from
      cache) and names an entity the read names, so the value returned is older
      than the database. Damage is stale state, again independent of key design.
      This is the committed ``stale_reads`` class, unchanged.

  D3  read whose cached value was produced under a different identity
      The committed ``cross_tenant`` class, split by whether the tool's output
      actually depends on identity:

        D3-damaging   the tool is identity-dependent, so the served value is
                      some other identity's answer;
        D3-harmless   the tool is identity-independent -- a catalog row, a flight
                      search, an arithmetic evaluation, a no-op ``think`` -- so the
                      value is exactly what the tool would have returned anyway.

      The split is *empirical*: see "Identity dependence" below. One class per
      hit, with the committed precedence (write > stale > cross-tenant), so the
      four buckets partition the committed unsafe set and their sum is the
      committed count. That partition is the conservation check.

What this is not. It does not argue the typed policy is unsafe: it re-scores the
same replay under a damage criterion, which can only *lower* the baselines'
counts, and then it quantifies what the typed policy's zero could be hiding. A
count of 0 on a finite sample is not a proof, and the last section turns it into
an interval.

Identity dependence (empirical, from observations, not from the key)
--------------------------------------------------------------------
``examples/real_traces/tau_bench_observed.jsonl`` is the observation-bearing twin
of the committed corpus: the same 325 first-trial records, each call carrying the
environment's output (truncated to 400 characters by the compactor). Two facts
about tau-bench make an empirical test possible, and one thing about this file
makes it necessary to validate first:

  * no tool receives an ambient session identity -- identity reaches a tool only
    through an explicit argument (``user_id``, or ``email`` for the two lookup
    tools), so a cross-tenant hit can only be damaging if the tool's payload is
    selected by such an argument;
  * the environment is deterministic given its state, and every record starts
    from the same initial data, so two observations taken before either record
    has written anything are taken in the same state. Divergence there cannot be
    explained by state drift.

The necessity: the compactor's call->observation join is broken for part of the
file -- 166 of 325 records attach one observation to every call (161 of the 165
sonnet runs, so the usable evidence is almost entirely gpt-4o), and further
mispairings are visible where a product read returns an order payload. So each
observation is validated against its own tool before use: the output must echo
the argument that identifies the record (``"product_id": "<arg>"``,
``"order_id": "<arg>"``, ...) or match the tool's documented return shape (a bare
user id or ``Error`` for the lookups, a numeric literal for ``calculate``, an empty
observation for ``think``). Rejected observations are counted, never silently
dropped, and the acceptance counts per tool are reported. The join defect is a
property of the compactor, not of the traces: the call sequences themselves match
the committed corpus exactly, which is checked, so the evidence is about the same
calls whose hits are being classified.

Two tests are run on the validated, zero-prior-write observations:

  E1  identity selection -- group by (tool, non-identity arguments); a group with
      two or more distinct identity argument values and two or more distinct
      observations shows the payload is selected by the identity argument.
  E2  tenant divergence -- group by (tool, all arguments); a group with two or
      more known tenants and two or more distinct observations shows the payload
      is relative to who asked, not merely to what was asked.

A tool is identity-dependent iff E1 or E2 fires. ``get_user_details`` fires E1
(different ``user_id`` -> different profile); ``find_user_id_by_email`` fires E1
(different ``email`` -> different user id or ``Error``); the catalog tools
(``get_product_details``, ``list_all_product_types``, ``calculate``, ``think``),
the flight searches, ``list_all_airports``, ``get_order_details``,
``get_reservation_details`` and ``find_user_id_by_name_zip`` fire neither -- with
identical arguments and no prior write they return identical observations, which
is the empirical form of "identity-independent". The four tools the task names as
catalog tools must come out independent, and ``get_user_details`` and
``find_user_id_by_email`` must come out dependent; both directions are asserted as
checks, so a change in the data or in a validator shows up as a failure rather
than as a quietly different number.

Intervals
---------
The cluster is a *task instance*, not a cell: one task as one or two models
performed it. The corpus holds 325 first-trial cells -- the 115 retail tasks were
run by two models each (230), and the 50 airline tasks by one model with 45 of
them also run by a second (50 + 45 = 95) -- but only 115 + 50 = 165 task
instances, and the 165 is what the interval resamples. Calls
inside an instance are dependent -- shared trajectory, user, evolving cache, and
an unsafe hit is a property of a sequence -- so resampling calls would overstate
precision. Both models' runs of one task stay inside the same cluster: the
instance, not the model, is the unit a deployment draws.

The bootstrap is Efron's percentile method, 10,000 resamples, seed 20240601,
2.5/97.5 percentiles, linear interpolation, resampling instances with replacement
and recomputing every rate inside a resample (paired, so P2/P3 ride on P1's
draws). Two intervals are the ones the task asks for -- P1's warranted-unsafe
hits per 1,000 calls, and the share of task instances with at least one
warranted-unsafe hit -- and P2/P3 are reported as supplementary point estimates
from the same matrix rather than given intervals of their own.

The typed policy's zero is a count of 0 in a finite sample, so it gets an exact
one-sided 95% upper bound: for k = 0 the Clopper-Pearson limit is
``1 - alpha**(1/n)``, with the rule-of-three approximation ``3/n`` reported beside
it (they agree to three digits at this n, which is the point of reporting both).
The bound is given per 1,000 admitted reads (n = 1,728) and per 1,000 replayed
calls (n = 2,275), plus a secondary bound per 1,000 *hits* (n = 466), because the
quantity a deployment cares about -- "of what this cache served, how much could
have been wrong" -- has the hit count as its denominator.

Deterministic: fixed seed and resample count, no clock, no environment, ordered
keys, integer counts next to every derived rate, and ``main`` builds the whole
document twice and refuses to write if the two differ.

Usage:
    python3 examples/benchmarks/unsafe_damage.py
    python3 examples/benchmarks/unsafe_damage.py --output /tmp/unsafe_damage.json

Output: ``unsafe_damage_results.json`` beside this script (schema
``unsafe_damage/v1``).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
OBSERVED = REPO_ROOT / "examples" / "real_traces" / "tau_bench_observed.jsonl"
DEFAULT_OUTPUT = HERE / "unsafe_damage_results.json"
IDENTITY_PATH = HERE / "cache_identity.py"
UNSAFE_PATH = HERE / "unsafe_cache_baselines.py"

SCHEMA = "unsafe_damage/v1"
TASK = (
    "E1 - damage-based unsafe accounting: reclassify every committed unsafe hit of "
    "P1/P2/P3 into write-served, stale-read and identity-damage (with the harmless "
    "catalog residue counted separately), then bound the typed policy's zero"
)

RESAMPLES = 10_000
SEED = 20240601
CONFIDENCE = 0.95
ALPHA = (1.0 - CONFIDENCE) / 2.0
BOUND_ALPHA = 0.05  # one-sided

# The corpus pins: 325 first-trial cells, 165 task instances, 2,275 calls.
EXPECTED_CELLS = 325
EXPECTED_INSTANCES = 165
EXPECTED_CALLS = 2275

# The committed counts this reclassification is a partition of. Hard-coded so a
# missing or edited artifact fails loudly instead of redefining the experiment.
COMMITTED: Dict[str, Dict[str, int]] = {
    "P1": {"calls": 2275, "hits": 807, "unsafe": 344, "writes_served": 61,
           "stale_reads": 179, "cross_tenant": 104},
    "P2": {"calls": 2275, "hits": 655, "unsafe": 165, "writes_served": 61,
           "stale_reads": 0, "cross_tenant": 104},
    "P3": {"calls": 2275, "hits": 963, "unsafe": 595, "writes_served": 62,
           "stale_reads": 179, "cross_tenant": 354},
    "P4": {"calls": 1728, "hits": 466, "unsafe": 0, "writes_served": 0,
           "stale_reads": 0, "cross_tenant": 0},
}
POLICY_NAMES = {"P1": "blind", "P2": "blind+invalidate", "P3": "semantic-ish", "P4": "typed"}

# The task names these as catalog tools: identity-independent, therefore their
# cross-tenant hits are harmless. The script does not assume it -- it measures it
# and asserts the two agree. ``get_user_details`` is the mirror assertion: a tool
# whose payload is selected by ``user_id`` and must come out dependent.
CATALOG_PRIOR: Tuple[str, ...] = (
    "get_product_details", "list_all_product_types", "calculate", "think",
)
DEPENDENT_PRIOR: Tuple[str, ...] = ("get_user_details", "find_user_id_by_email")

# Arguments that name *whose* data an answer is about. ``tenant`` is not an
# argument (it is recovered per record) and ``first_name``/``last_name``/``zip``
# are ordinary arguments: a key that includes them cannot confuse two people.
IDENTITY_ARG_KEYS: Tuple[str, ...] = ("user_id", "email")

DOMAINS: Tuple[str, ...] = ("retail", "airline")

DAMAGE_CLASSES: Tuple[str, ...] = ("D1", "D2", "D3_damaging", "D3_harmless")


class Row(NamedTuple):
    """One replayed call, with the label ``replay()`` would fold into a Counter."""

    cluster: str
    policy: str
    source: str
    tool: str
    effect: str
    admitted: bool
    hit: bool
    cls: Optional[str]          # committed class: writes_served/stale_reads/cross_tenant
    memo_args: Optional[Dict[str, Any]]
    hit_args: Optional[Dict[str, Any]]
    memo_tenant: Optional[str]
    hit_tenant: Optional[str]


# ---------------------------------------------------------------------------
# Module loading, digests, paths
# ---------------------------------------------------------------------------
def load_module(name: str, path: Path):
    if not path.is_file():
        raise SystemExit(f"unsafe_damage: cannot find {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"unsafe_damage: cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:  # pragma: no cover - checkout behind a symlink
        return str(path.resolve())


def require_attrs(module, names: Sequence[str], where: str) -> None:
    missing = [name for name in names if not hasattr(module, name)]
    if missing:
        raise SystemExit(
            f"unsafe_damage: {where} does not expose {missing}. The replay, the "
            "cluster labels and the identity evidence all reuse that module's own "
            "helpers; re-deriving them here is how this table would drift from the "
            "cache it measures."
        )


def canon(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


# ---------------------------------------------------------------------------
# Clusters: one task instance = (domain, task_id)
# ---------------------------------------------------------------------------
def domain_of(source: Any) -> str:
    """The last ``-``-separated token of ``source``, checked against the domains."""
    text = str(source)
    token = text.rsplit("-", 1)[-1]
    if token not in DOMAINS:
        raise SystemExit(
            f"unsafe_damage: cannot derive a domain from source {text!r}; expected one "
            f"of {DOMAINS} as the last '-'-separated token"
        )
    return token


def cluster_records(
    records: Sequence[Mapping[str, Any]]
) -> Tuple["OrderedDict[str, List[Tuple[str, Any]]]", Dict[str, int]]:
    """Map each task instance to the cells ``(source, task_id)`` that performed it."""
    clusters: "OrderedDict[str, List[Tuple[str, Any]]]" = OrderedDict()
    for record in records:
        if record.get("trial", 0) != 0:
            continue
        label = f"{domain_of(record.get('source'))}#{record.get('task_id')}"
        cell = (str(record.get("source")), record.get("task_id"))
        clusters.setdefault(label, []).append(cell)
    composition = Counter(label.split("#", 1)[0] for label in clusters)
    return clusters, dict(composition)


# ---------------------------------------------------------------------------
# Per-call replay (the control flow of unsafe_cache_baselines.replay, keeping the
# memo's provenance so a hit can be diagnosed instead of only counted)
# ---------------------------------------------------------------------------
def walk(
    streams: "OrderedDict[str, List[Any]]",
    policy,
    ci,
    cluster_of: Mapping[str, str],
    unsafe,
) -> List[Row]:
    rows: List[Row] = []
    served_key = unsafe._served_key
    intervening = unsafe._intervening
    evict = unsafe._evict

    for source, calls in streams.items():
        cache: Dict[str, Dict[str, Any]] = {}
        index: Dict[str, set] = {}
        key_map: Dict[str, str] = {}
        write_log: List[Tuple[int, Tuple[str, ...], str, Dict[str, Any]]] = []

        for call in calls:
            cluster = cluster_of[f"{source}#{call.task_id}"]
            effect = ci.effect_class(call.tool)

            if not policy.memoises_writes and effect != "read":
                if effect == "write":
                    evict(ci, index, key_map, cache, call)
                    write_log.append(
                        (call.pos, ci.entity_keys(call.tool, call.args), call.tool, call.args)
                    )
                rows.append(Row(cluster, policy.pid, source, call.tool, effect,
                                False, False, None, None, None, None, None))
                continue

            key = served_key(ci, policy, call)
            memo = cache.get(key)
            if memo is not None:
                cls = None
                if effect == "write":
                    cls = "writes_served"
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
                        cls = "stale_reads"
                    elif cross:
                        cls = "cross_tenant"
                rows.append(Row(cluster, policy.pid, source, call.tool, effect, True, True,
                                cls, memo["args"], call.args, memo["tenant"], call.tenant))
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
            rows.append(Row(cluster, policy.pid, source, call.tool, effect,
                            True, False, None, None, None, None, None))
    return rows


def fold(rows: Sequence[Row]) -> Dict[str, int]:
    classes = Counter(row.cls for row in rows if row.cls)
    return {
        "calls_total": len(rows),
        "cacheable_calls": sum(1 for row in rows if row.admitted),
        "hits": sum(1 for row in rows if row.hit),
        "unsafe_hits": sum(classes.values()),
        "writes_served": classes["writes_served"],
        "stale_reads": classes["stale_reads"],
        "cross_tenant": classes["cross_tenant"],
    }


def verify(rows: Sequence[Row], replayed: Mapping[str, Any], policy_name: str) -> None:
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
    diff = {key: (got[key], expected[key]) for key in expected if got[key] != expected[key]}
    if diff:
        raise SystemExit(
            f"unsafe_damage: per-call walk disagrees with replay() for {policy_name!r}: "
            f"{diff} (walk, replay). Refusing to classify hits from a diverging replay."
        )


# ---------------------------------------------------------------------------
# Identity dependence, measured on the observation-bearing corpus
# ---------------------------------------------------------------------------
def _echo(key: str, value: Any, text: str) -> bool:
    return f'"{key}": "{value}"' in text


def _numeric(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    try:
        float(stripped)
    except ValueError:
        return False
    return True


def _user_id_shape(text: str) -> bool:
    """A bare tau-bench user id: lowercase words, then a numeric suffix."""
    head, _, tail = text.strip().rpartition("_")
    return bool(head) and tail.isdigit() and all(
        part.isalnum() for part in head.split("_") if part
    )


def validate_observation(tool: str, args: Mapping[str, Any], out: Any) -> Optional[bool]:
    """Is this observation consistent with *this* tool and *these* arguments?

    ``None`` means the tool is not used as identity evidence (writes and the
    transfer/control tools). ``False`` means the compactor's join is broken for
    this call -- the output cannot have come from this tool with these arguments
    -- and the observation is excluded, counted in ``rejected_by_tool``.

    Every rule is content-based, so it cannot be tuned towards a verdict: the
    output must echo the argument that identifies the record, or match the shape
    the tool is documented to return. The compactor truncates at 400 characters,
    so the tests are prefix tests; they never need the full payload.
    """
    if not isinstance(out, str):
        return False
    text = out
    if tool == "think":
        return text == ""  # tau-bench's think returns an empty observation
    if tool == "calculate":
        return _numeric(text)
    if tool == "get_product_details":
        return (_echo("product_id", args.get("product_id"), text)
                and '"variants"' in text and '"order_id"' not in text)
    if tool == "get_order_details":
        return _echo("order_id", args.get("order_id"), text) and '"items"' in text
    if tool == "get_reservation_details":
        return _echo("reservation_id", args.get("reservation_id"), text) and '"flights"' in text
    if tool == "get_user_details":
        return ('"email"' in text and '"name"' in text
                and '"order_id"' not in text and '"reservation_id"' not in text)
    if tool in ("find_user_id_by_email", "find_user_id_by_name_zip"):
        return _user_id_shape(text) or text.startswith("Error")
    if tool == "list_all_product_types":
        return (text.startswith('{"') and '"order_id"' not in text
                and bool(re.match(r'^\{"[A-Za-z][^"]*": "\d+', text)))
    if tool == "list_all_airports":
        return bool(re.match(r'^\{"[A-Z]{3}": "', text))
    if tool == "search_direct_flight":
        return ('"flight_number"' in text and "[[" not in text
                and _echo("origin", args.get("origin"), text)
                and _echo("destination", args.get("destination"), text))
    if tool == "search_onestop_flight":
        return ('"flight_number"' in text
                and _echo("origin", args.get("origin"), text)
                and _echo("destination", args.get("destination"), text))
    return None


def is_degenerate(record: Mapping[str, Any]) -> bool:
    """One observation reused for distinct calls: a provably broken join.

    If two calls with different ``(tool, arguments)`` got byte-identical non-empty
    observations, the compactor paired the wrong message at least once. Empty
    observations are exempt because ``think`` returns one for every argument list.
    """
    pairs = {
        (str(tool), canon(dict(args or {})))
        for tool, args, out in record.get("calls") or []
        if isinstance(out, str) and out != ""
    }
    outputs = {
        out for _tool, _args, out in record.get("calls") or []
        if isinstance(out, str) and out != ""
    }
    return len(pairs) > 1 and len(outputs) == 1


def record_tenant(record: Mapping[str, Any]) -> Optional[str]:
    ids = sorted({
        str(args["user_id"])
        for _tool, args, _out in record.get("calls") or []
        if isinstance(args.get("user_id"), str)
    })
    return ids[0] if len(ids) == 1 else None


def identity_args(args: Mapping[str, Any]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((key, str(args[key])) for key in IDENTITY_ARG_KEYS if key in args))


def non_identity_canon(args: Mapping[str, Any]) -> str:
    return canon({key: value for key, value in args.items() if key not in IDENTITY_ARG_KEYS})


class Observation(NamedTuple):
    tool: str
    non_identity: str                 # canonical non-identity arguments
    identity: Tuple[Tuple[str, str], ...]
    output: str
    tenant: Optional[str]
    source: str
    task_id: Any
    call_index: int


def _fmt_keys(keys: Sequence[str]) -> str:
    return "{}" if not keys else "{" + ", ".join(keys) + "}"


def _sample(
    group: Sequence["Observation"],
    entry: Callable[[Any, str], Dict[str, Any]],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Up to ``limit`` entries with distinct *displayed* outputs.

    Deduplication is on the truncated text that is shown, not on the full output:
    two observations can share their first 160 characters while differing later,
    and printing three identical prefixes would hide the very divergence the group
    was selected for.
    """
    seen = set()
    out: List[Dict[str, Any]] = []
    for item in group:
        text = item.output[:160]
        if text in seen:
            continue
        seen.add(text)
        out.append(entry(item, text))
        if len(out) == limit:
            break
    return out


def _identity_entry(item: "Observation", text: str) -> Dict[str, Any]:
    return OrderedDict(identity=dict(item.identity) or None, tenant=item.tenant, output=text)


def _tenant_entry(item: "Observation", text: str) -> Dict[str, Any]:
    return OrderedDict(tenant=item.tenant, output=text)


def _args_of(item: "Observation") -> Dict[str, Any]:
    arguments = json.loads(item.non_identity)
    for key, value in item.identity:
        arguments[key] = value
    return arguments


def _shape_example(item: "Observation") -> Dict[str, Any]:
    """The single-observation fallback when no comparison was possible."""
    return OrderedDict(arguments=_args_of(item), tenant=item.tenant,
                       output=item.output[:160])


def identity_evidence(
    observed: Sequence[Mapping[str, Any]],
    ci,
    read_tools: Sequence[str],
) -> Dict[str, Any]:
    """Per-tool identity dependence, measured on validated zero-prior-write calls.

    A record's calls are visited in order and a call is only used when no write has
    executed earlier in that record: both records are then in the initial
    environment state, so a divergence between them cannot be state drift. That is
    what makes the test empirical rather than an argument from the key's shape.

    ``argument_keys`` comes from every call in the file, not only the validated
    ones: the *arguments* are never in doubt (the call sequence matches the
    committed corpus), only the observations are, so the argument surface of each
    tool is read off the calls directly.
    """
    argument_keys: Dict[str, set] = {}
    for record in observed:
        for tool, args, _out in record.get("calls") or []:
            argument_keys.setdefault(str(tool), set()).update((args or {}).keys())

    accepted: Counter = Counter()
    rejected: Counter = Counter()
    degenerate = 0
    excluded_by_source: Counter = Counter()
    used_by_source: Counter = Counter()
    observations: Dict[str, List[Observation]] = {}

    for record in observed:
        source = str(record.get("source"))
        if is_degenerate(record):
            degenerate += 1
            excluded_by_source[source] += 1
            continue
        used_by_source[source] += 1
        tenant = record_tenant(record)
        prior_writes = 0
        for index, (tool, args, out) in enumerate(record.get("calls") or []):
            args = dict(args or {})
            verdict = validate_observation(tool, args, out)
            is_write = ci.effect_class(tool) == "write"
            if verdict is None:
                if is_write:
                    prior_writes += 1
                continue
            if verdict is False:
                rejected[tool] += 1
            else:
                accepted[tool] += 1
                if prior_writes == 0:
                    observations.setdefault(tool, []).append(Observation(
                        tool=tool,
                        non_identity=non_identity_canon(args),
                        identity=identity_args(args),
                        output=str(out),
                        tenant=tenant,
                        source=source,
                        task_id=record.get("task_id"),
                        call_index=index,
                    ))
            if is_write:
                prior_writes += 1

    tools: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
    for tool in sorted(set(read_tools) | set(accepted) | set(rejected)):
        items = observations.get(tool, [])
        keys = sorted(argument_keys.get(tool, set()))
        takes_identity = bool(set(keys) & set(IDENTITY_ARG_KEYS))
        by_identity: Dict[str, List[Observation]] = {}
        by_args: Dict[str, List[Observation]] = {}
        for item in items:
            by_identity.setdefault(item.non_identity, []).append(item)
            by_args.setdefault((item.non_identity, item.identity), []).append(item)

        identity_groups = 0
        identity_divergent = 0
        identity_example: Optional[Dict[str, Any]] = None
        for key, group in by_identity.items():
            values = {item.identity for item in group}
            if len(values) < 2:
                continue
            identity_groups += 1
            outputs = {item.output for item in group}
            if len(outputs) > 1:
                identity_divergent += 1
                if identity_example is None:
                    identity_example = OrderedDict(
                        non_identity_arguments=key,
                        identity_values=len(values),
                        distinct_outputs=len(outputs),
                        examples=_sample(group, _identity_entry),
                    )

        tenant_groups = 0
        tenant_divergent = 0
        tenant_example: Optional[Dict[str, Any]] = None
        agreement_example: Optional[Dict[str, Any]] = None
        for key, group in by_args.items():
            tenants = {item.tenant for item in group if item.tenant is not None}
            if len(tenants) < 2:
                continue
            tenant_groups += 1
            outputs = {item.output for item in group}
            if len(outputs) > 1:
                tenant_divergent += 1
                if tenant_example is None:
                    tenant_example = OrderedDict(
                        arguments=_args_of(group[0]),
                        tenants=len(tenants),
                        distinct_outputs=len(outputs),
                        examples=_sample(group, _tenant_entry),
                    )
            elif agreement_example is None:
                agreement_example = OrderedDict(
                    arguments=_args_of(group[0]),
                    tenants=len(tenants),
                    distinct_outputs=len(outputs),
                    examples=_sample(group, _tenant_entry),
                )

        dependent = identity_divergent > 0 or tenant_divergent > 0
        if dependent:
            basis = "measured"
            reasons = []
            if identity_divergent:
                reasons.append(
                    f"E1: {identity_divergent} non-identity argument group(s) hold two or more "
                    "identity values and return different observations"
                )
            if tenant_divergent:
                reasons.append(
                    f"E2: {tenant_divergent} argument group(s) are shared by two or more "
                    "tenants and return different observations"
                )
        elif items:
            basis = "measured" if (identity_groups or tenant_groups) else "structural"
            reasons = [
                f"no validated observation differs across identity values "
                f"({identity_groups} group(s) compared) or across tenants for identical "
                f"arguments ({tenant_groups} group(s) compared); the tool's observed "
                f"arguments {_fmt_keys(keys)} carry "
                + ("an identity binding, but the payload is not selected by it"
                   if takes_identity else "no identity binding")
            ]
        else:
            basis = "structural"
            reasons = [
                f"no observation survived validation ({rejected[tool]} rejected, 0 accepted); "
                f"the verdict rests on the tool's observed arguments {_fmt_keys(keys)}, which "
                + ("carry an identity binding" if takes_identity else "carry no identity binding")
            ]

        payload = " ".join(item.output for item in items[:50])
        tools[tool] = OrderedDict(
            observed_argument_keys=keys,
            takes_identity_argument=takes_identity,
            identity_arguments=sorted({key for item in items for key, _v in item.identity}),
            observations_accepted=accepted[tool],
            observations_rejected=rejected[tool],
            zero_prior_writes=len(items),
            evidence_sufficient=bool(items),
            comparisons=OrderedDict(
                identity_value_groups=identity_groups,
                cross_tenant_identical_argument_groups=tenant_groups,
            ),
            identity_divergent_groups=identity_divergent,
            tenant_divergent_groups=tenant_divergent,
            payload_user_specific=(
                any(field in payload for field in
                    ('"email"', '"name"', '"address"', '"payment_methods"', '"orders"'))
                or (bool(items) and _user_id_shape(items[0].output))
            ),
            identity_dependent=dependent,
            verdict="identity-dependent" if dependent else "identity-independent",
            basis=basis,
            reason="; ".join(reasons),
            evidence_example=(
                identity_example or tenant_example or agreement_example
                or (_shape_example(items[0]) if items else None)
            ),
        )


    coverage = OrderedDict(
        records=len(observed),
        records_excluded_degenerate_join=degenerate,
        records_used=len(observed) - degenerate,
        records_by_source=OrderedDict(sorted(
            Counter(str(record.get("source")) for record in observed).items())),
        excluded_by_source=OrderedDict(sorted(excluded_by_source.items())),
        used_by_source=OrderedDict(sorted(used_by_source.items())),
        observations_accepted=sum(accepted.values()),
        observations_rejected=sum(rejected.values()),
        acceptance_share=round(
            sum(accepted.values()) / max(1, sum(accepted.values()) + sum(rejected.values())), 4
        ),
        accepted_by_tool=OrderedDict(sorted(accepted.items())),
        rejected_by_tool=OrderedDict(sorted(rejected.items())),
        caveat=(
            "the compactor's join collapses to one observation per call for 166 of 325 "
            "records, almost all of them sonnet runs, so the validated evidence is drawn "
            "from the gpt-4o runs (155 of 159 usable records). The call sequences are "
            "unaffected -- they match the committed corpus exactly (checked) -- so the "
            "evidence is about the same calls whose hits are classified; only the "
            "observations are thinner than the call count suggests."
        ),
    )
    return OrderedDict(tools=tools, coverage=coverage)


# ---------------------------------------------------------------------------
# Reclassification
# ---------------------------------------------------------------------------
def reclassify(
    rows: Sequence[Row],
    dependent: Mapping[str, bool],
) -> Dict[str, Any]:
    """Turn the per-call labels into the four damage buckets plus auditor evidence."""
    buckets: Dict[str, int] = {name: 0 for name in DAMAGE_CLASSES}
    by_tool: Dict[str, Counter] = {name: Counter() for name in DAMAGE_CLASSES}
    examples: Dict[str, List[Dict[str, Any]]] = {name: [] for name in DAMAGE_CLASSES}
    identity_differs: Counter = Counter()
    per_cluster: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        agg = per_cluster.setdefault(
            row.cluster,
            {"calls": 0, "warranted": 0, "classes": Counter()},
        )
        if row.admitted:
            agg["calls"] += 1
        if row.cls is None:
            continue
        if row.cls == "writes_served":
            bucket = "D1"
        elif row.cls == "stale_reads":
            bucket = "D2"
        else:
            bucket = "D3_damaging" if dependent.get(row.tool, True) else "D3_harmless"
            if identity_args(row.memo_args or {}) != identity_args(row.hit_args or {}):
                identity_differs[bucket] += 1
        buckets[bucket] += 1
        by_tool[bucket][row.tool] += 1
        agg["classes"][bucket] += 1
        if bucket in ("D1", "D2", "D3_damaging"):
            agg["warranted"] += 1
        if len(examples[bucket]) < 1:
            examples[bucket].append(OrderedDict(
                tool=row.tool,
                source=row.source,
                cluster=row.cluster,
                committed_class=row.cls,
                cached_arguments=row.memo_args,
                hitting_arguments=row.hit_args,
                cached_tenant=row.memo_tenant,
                hit_tenant=row.hit_tenant,
                identity_arguments_differ=(
                    identity_args(row.memo_args or {}) != identity_args(row.hit_args or {})
                ),
            ))

    warranted = buckets["D1"] + buckets["D2"] + buckets["D3_damaging"]
    return OrderedDict(
        calls=sum(1 for row in rows if row.admitted),
        hits=sum(1 for row in rows if row.hit),
        buckets=buckets,
        warranted_unsafe=warranted,
        by_tool={name: OrderedDict(sorted(by_tool[name].items())) for name in DAMAGE_CLASSES},
        examples=[examples[name][0] for name in DAMAGE_CLASSES if examples[name]],
        identity_arguments_differ=OrderedDict(sorted(identity_differs.items())),
        per_cluster=OrderedDict(
            (cluster, OrderedDict(calls=agg["calls"], warranted=agg["warranted"],
                                  classes=OrderedDict(sorted(agg["classes"].items()))))
            for cluster, agg in sorted(per_cluster.items())
        ),
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
def draw(rng: random.Random, n: int) -> int:
    """One unbiased index in [0, n) from documented 64-bit draws.

    Rejection on ``getrandbits(64)`` rather than ``randrange``: the sequence then
    depends only on the Mersenne Twister bit stream, which is stable across
    CPython versions, not on the internals of ``_randbelow``.
    """
    limit = (1 << 64) - ((1 << 64) % n)
    while True:
        value = rng.getrandbits(64)
        if value < limit:
            return value % n


def percentile(sorted_values: Sequence[float], p: float) -> float:
    """Linear-interpolation percentile: h = (n-1)*p, floor plus the fraction."""
    last = len(sorted_values) - 1
    h = last * p
    lo = int(math.floor(h))
    hi = min(lo + 1, last)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (h - lo)


SERIES = (
    "p1_warranted_per_1000_calls",
    "p1_share_instances_with_warranted",
    "p2_warranted_per_1000_calls",
    "p2_share_instances_with_warranted",
    "p3_warranted_per_1000_calls",
    "p3_share_instances_with_warranted",
)


def bootstrap(
    matrix: Sequence[Tuple[int, int, int, int, int, int, int, int, int]],
    resamples: int,
    seed: int,
) -> Tuple[Dict[str, List[float]], int]:
    """Resample task instances with replacement; recompute every series inside.

    ``matrix`` rows are per-instance sums in a fixed order:
    ``(p1_calls, p1_warranted, p2_calls, p2_warranted, p3_calls, p3_warranted,
    p1_has, p2_has, p3_has)``. One draw feeds every series, so P2/P3 ride on the
    same resamples as P1 instead of being bootstrapped independently.
    """
    n = len(matrix)
    rng = random.Random(seed)
    series: Dict[str, List[float]] = {name: [] for name in SERIES}
    used = 0

    for _ in range(resamples):
        calls = warranted = p2c = p2w = p3c = p3w = 0
        has1 = has2 = has3 = 0
        for _ in range(n):
            c1, w1, c2, w2, c3, w3, h1, h2, h3 = matrix[draw(rng, n)]
            calls += c1
            warranted += w1
            p2c += c2
            p2w += w2
            p3c += c3
            p3w += w3
            has1 += h1
            has2 += h2
            has3 += h3
        if calls == 0 or p2c == 0 or p3c == 0:  # pragma: no cover - every instance has calls
            continue
        used += 1
        series["p1_warranted_per_1000_calls"].append(1000.0 * warranted / calls)
        series["p1_share_instances_with_warranted"].append(has1 / n)
        series["p2_warranted_per_1000_calls"].append(1000.0 * p2w / p2c)
        series["p2_share_instances_with_warranted"].append(has2 / n)
        series["p3_warranted_per_1000_calls"].append(1000.0 * p3w / p3c)
        series["p3_share_instances_with_warranted"].append(has3 / n)
    return series, used


def interval(values: Sequence[float], estimate: float, decimals: int, unit: str) -> Dict[str, Any]:
    ordered = sorted(values)
    return OrderedDict(
        estimate=round(estimate, decimals),
        ci_low=round(percentile(ordered, ALPHA), decimals),
        ci_high=round(percentile(ordered, 1.0 - ALPHA), decimals),
        unit=unit,
        confidence=CONFIDENCE,
        resamples=len(ordered),
        percentile_rule="linear interpolation: h = (n-1)*p, floor plus the fraction",
    )


# ---------------------------------------------------------------------------
# The typed policy's zero
# ---------------------------------------------------------------------------
def exact_upper_zero(n: int, alpha: float = BOUND_ALPHA) -> float:
    """Exact one-sided upper limit for k = 0 successes in n trials.

    For k = 0 the Clopper-Pearson limit has the closed form ``1 - alpha**(1/n)``;
    it is the p with ``(1-p)**n = alpha``. No beta quantile and no approximation.
    """
    return 1.0 - alpha ** (1.0 / n)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------

def _method_block(n_instances: int, resamples: int, used: int, seed: int) -> Dict[str, Any]:
    """Definitions, identity protocol, bootstrap, bound: the method section."""
    return OrderedDict(
        damage_classes=OrderedDict(
            D1=(
                "a write served from cache: the mutation never ran while the agent's "
                "belief that it did is unchanged (the committed writes_served class)"
            ),
            D2=(
                "a read served across a landed write to the same entity: the memo "
                "predates a write that executed and names an entity the read names, so "
                "the value is older than the database (the committed stale_reads class)"
            ),
            D3_damaging=(
                "a cross-tenant read on a tool whose output depends on identity: the "
                "served value is some other identity's answer"
            ),
            D3_harmless=(
                "a cross-tenant read on a tool whose output does not depend on identity: "
                "the served value is what the tool would have returned anyway"
            ),
            warranted_unsafe="D1 + D2 + D3_damaging",
            precedence=(
                "one class per hit, with the committed precedence writes_served > "
                "stale_reads > cross_tenant, so the four buckets partition the committed "
                "unsafe set and their sum equals the committed count"
            ),
            why_not_analytic=(
                "each bucket is defined by an observable damage mode, not by a key "
                "component: a missing side effect, a value older than a landed write, a "
                "payload selected by another identity. A policy could implement all three "
                "gates and still serve a stale value if its eviction index were wrong; the "
                "zero is measured here, and then bounded"
            ),
        ),
        identity_dependence=OrderedDict(
            source=(
                "examples/real_traces/tau_bench_observed.jsonl, the observation-bearing "
                "twin of the committed corpus (same 325 first-trial records, outputs "
                "truncated to 400 characters by the compactor)"
            ),
            validation=(
                "each observation must echo the argument that identifies its record "
                '("product_id": "<arg>", "order_id": "<arg>", ...) or match the tool\'s '
                "documented return shape (bare user id or Error for the lookups, a numeric "
                "literal for calculate, an empty observation for think); rejected "
                "observations are counted per tool, never dropped silently"
            ),
            degenerate_join_filter=(
                "a record whose distinct (tool, arguments) calls all carry one non-empty "
                "observation is a provably broken join and is excluded from the evidence"
            ),
            state_control=(
                "only calls with no earlier write in their own record are used: both "
                "records are then in the initial environment state, so divergence cannot "
                "be state drift"
            ),
            E1=(
                "identity selection: a (tool, non-identity arguments) group with two or "
                "more distinct identity values (user_id/email) and two or more distinct "
                "observations shows the payload is selected by the identity argument"
            ),
            E2=(
                "tenant divergence: a (tool, all arguments) group shared by two or more "
                "tenants with two or more distinct observations shows the payload is "
                "relative to who asked"
            ),
            verdict="identity-dependent iff E1 or E2 fires",
            identity_arguments=list(IDENTITY_ARG_KEYS),
        ),
        denominators=OrderedDict(
            calls=(
                "admitted (cacheable) calls: every replayed call for P1-P3, reads only "
                "for P4, which refuses writes and control tools"
            ),
            hits="the policy's hits, the base of every share_of_hits figure",
            share_of_hits="class count / hits, so the four classes sum to the committed "
                          "unsafe share of hits",
            warranted_per_1000_calls="1000 * warranted / admitted calls",
        ),
        bootstrap=OrderedDict(
            cluster_unit=(
                "task instance: one task as one or two models performed it, labelled "
                "domain#task_id; 165 instances over 4 runs"
            ),
            clusters=n_instances,
            resamples=resamples,
            resamples_used=used,
            seed=seed,
            replacement=True,
            confidence=CONFIDENCE,
            percentiles=[round(100 * ALPHA, 3), round(100 * (1 - ALPHA), 3)],
            percentile_rule=(
                "linear interpolation on the sorted draws: h = (n-1)*p, value at "
                "floor(h) plus the fraction to the next"
            ),
            rng=(
                "CPython random.Random (Mersenne Twister); indices drawn with rejection "
                "on getrandbits(64), so the draws depend only on the bit stream"
            ),
            why_instances_not_cells=(
                "the two model runs of one task are two attempts at the same task, not two "
                "independent draws from the task population; their calls share an initial "
                "environment state and a user, so cells are dependent. Resampling "
                "instances keeps every attempt inside its cluster and makes the unit of "
                "variation the deployment's: another batch of tasks."
            ),
            paired=True,
        ),
        exact_bound=OrderedDict(
            statistic="one-sided 95% upper limit for a count of zero",
            formula="1 - alpha**(1/n), alpha = 0.05 (Clopper-Pearson at k = 0)",
            rule_of_three="3/n, reported beside the exact limit as the familiar approximation",
            denominators="admitted reads (1,728), all replayed calls (2,275), served hits (466)",
        ),
    )

def _provenance_block(
    *,
    corpus_sha: str,
    first_trial_records: int,
    expected_sha: str,
    observed_count: int,
    observed_sha: str,
    shas: Mapping[str, str],
) -> Dict[str, Any]:
    """Content hashes and the determinism contract; no clock, no host facts."""
    return OrderedDict(
        script=OrderedDict(path=rel(Path(__file__)), sha256=sha256_file(Path(__file__)),
                           schema=SCHEMA),
        corpus=OrderedDict(
            path=rel(CORPUS),
            sha256=corpus_sha,
            records=first_trial_records,
            calls=EXPECTED_CALLS,
            cells=EXPECTED_CELLS,
            instances=EXPECTED_INSTANCES,
            expected_sha256=expected_sha,
            matches_expected=corpus_sha == expected_sha,
        ),
        observed_corpus=OrderedDict(
            path=rel(OBSERVED),
            sha256=observed_sha,
            records=observed_count,
        ),
        identity_module=OrderedDict(path=rel(IDENTITY_PATH), sha256=shas["identity"]),
        unsafe_module=OrderedDict(
            path=rel(UNSAFE_PATH),
            sha256=shas["unsafe"],
            why=(
                "the keys, the effect classes, the eviction index and the staleness join "
                "are imported from the module that produced the committed hits; only the "
                "classification is re-derived here"
            ),
        ),
        determinism=(
            "no clock, no environment, no randomness outside the seeded bootstrap; sorted "
            "collections everywhere; main() builds the document twice and refuses to "
            "write if the two differ"
        ),
        arithmetic=(
            "integer counts; shares are round(n/d, 6); rates round(..., 4); no averages "
            "of averages"
        ),
    )

def build_document(
    *,
    observed: Sequence[Mapping[str, Any]],
    clusters: Mapping[str, Sequence[Tuple[str, Any]]],
    composition: Mapping[str, int],
    classified: Mapping[str, Mapping[str, Any]],
    evidence: Mapping[str, Any],
    series: Mapping[str, Sequence[float]],
    used: int,
    resamples: int,
    seed: int,
    corpus_sha: str,
    observed_sha: str,
    first_trial_records: int,
    expected_sha: str,
    shas: Mapping[str, str],
) -> Dict[str, Any]:
    dependent = {
        tool: bool(entry["identity_dependent"]) for tool, entry in evidence["tools"].items()
    }
    ordered_clusters = sorted(clusters)
    n_instances = len(ordered_clusters)
    composition = dict(composition)

    per_policy: List[Dict[str, Any]] = []
    for policy_id in ("P1", "P2", "P3", "P4"):
        data = classified[policy_id]
        buckets = data["buckets"]
        warranted = data["warranted_unsafe"]
        hits = data["hits"]
        calls = data["calls"]
        committed = COMMITTED[policy_id]
        per_policy.append(OrderedDict(
            id=policy_id,
            name=POLICY_NAMES[policy_id],
            calls=calls,
            hits=hits,
            hits_committed=committed["hits"],
            D1=buckets["D1"],
            D2=buckets["D2"],
            D3_damaging=buckets["D3_damaging"],
            D3_harmless=buckets["D3_harmless"],
            warranted_unsafe=warranted,
            warranted_unsafe_per_1000_calls=(
                round(1000.0 * warranted / calls, 4) if calls else 0.0
            ),
            warranted_share_of_hits=round(warranted / hits, 6) if hits else 0.0,
            share_of_hits=OrderedDict(
                D1=round(buckets["D1"] / hits, 6) if hits else 0.0,
                D2=round(buckets["D2"] / hits, 6) if hits else 0.0,
                D3_damaging=round(buckets["D3_damaging"] / hits, 6) if hits else 0.0,
                D3_harmless=round(buckets["D3_harmless"] / hits, 6) if hits else 0.0,
                warranted=round(warranted / hits, 6) if hits else 0.0,
            ),
            committed_unsafe_hits=committed["unsafe"],
            reclassified_total=warranted + buckets["D3_harmless"],
            reduction_vs_committed=committed["unsafe"] - warranted,
            by_tool=data["by_tool"],
            identity_arguments_differ=data["identity_arguments_differ"],
            examples=data["examples"],
        ))

    by_id = {entry["id"]: entry for entry in per_policy}

    instance_counts = {name: 0 for name in DAMAGE_CLASSES}
    p1_instances = p2_instances = p3_instances = 0
    for cluster in ordered_clusters:
        for name in DAMAGE_CLASSES:
            instance_counts[name] += 1 if classified["P1"]["per_cluster"][cluster]["classes"].get(name) else 0
        p1_instances += 1 if classified["P1"]["per_cluster"][cluster]["warranted"] else 0
        p2_instances += 1 if classified["P2"]["per_cluster"][cluster]["warranted"] else 0
        p3_instances += 1 if classified["P3"]["per_cluster"][cluster]["warranted"] else 0

    p1_warranted = by_id["P1"]["warranted_unsafe"]
    p1_calls = by_id["P1"]["calls"]

    intervals = OrderedDict(
        p1_warranted_per_1000_calls=interval(
            series["p1_warranted_per_1000_calls"], 1000.0 * p1_warranted / p1_calls, 3,
            "warranted-unsafe hits per 1,000 calls",
        ),
        p1_share_instances_with_warranted=interval(
            series["p1_share_instances_with_warranted"], p1_instances / n_instances, 4,
            "share of task instances with at least one warranted-unsafe hit",
        ),
    )
    supplementary = OrderedDict(
        note=(
            "point estimates only, from the same 10,000 draws (paired with P1) but not "
            "required by the task; the intervals reported above are P1's"
        ),
        p2=OrderedDict(
            warranted_unsafe=by_id["P2"]["warranted_unsafe"],
            warranted_per_1000_calls=by_id["P2"]["warranted_unsafe_per_1000_calls"],
            instances_with_warranted=p2_instances,
            share_instances_with_warranted=round(p2_instances / n_instances, 6),
        ),
        p3=OrderedDict(
            warranted_unsafe=by_id["P3"]["warranted_unsafe"],
            warranted_per_1000_calls=by_id["P3"]["warranted_unsafe_per_1000_calls"],
            instances_with_warranted=p3_instances,
            share_instances_with_warranted=round(p3_instances / n_instances, 6),
        ),
    )

    typed_admitted = COMMITTED["P4"]["calls"]
    typed_hits = COMMITTED["P4"]["hits"]
    bound = OrderedDict(
        method=(
            "exact one-sided Clopper-Pearson limit for k = 0, closed form "
            "1 - alpha**(1/n); the rule-of-three approximation 3/n is reported beside it"
        ),
        alpha_one_sided=BOUND_ALPHA,
        observed_unsafe=0,
        admitted_reads=OrderedDict(
            n=typed_admitted,
            exact_upper=round(exact_upper_zero(typed_admitted), 8),
            exact_upper_per_1000=round(1000.0 * exact_upper_zero(typed_admitted), 3),
            rule_of_three_per_1000=round(3000.0 / typed_admitted, 3),
        ),
        all_replayed_calls=OrderedDict(
            n=EXPECTED_CALLS,
            exact_upper=round(exact_upper_zero(EXPECTED_CALLS), 8),
            exact_upper_per_1000=round(1000.0 * exact_upper_zero(EXPECTED_CALLS), 3),
            rule_of_three_per_1000=round(3000.0 / EXPECTED_CALLS, 3),
        ),
        served_hits=OrderedDict(
            n=typed_hits,
            exact_upper=round(exact_upper_zero(typed_hits), 8),
            exact_upper_per_1000=round(1000.0 * exact_upper_zero(typed_hits), 3),
            rule_of_three_per_1000=round(3000.0 / typed_hits, 3),
        ),
        reading=(
            "a zero on 1,728 admitted reads is consistent with a true warranted-unsafe "
            f"rate up to {1000.0 * exact_upper_zero(typed_admitted):.2f} per 1,000 reads at "
            "95% one-sided confidence; the point estimate is 0, the honest claim is "
            "'below that bound', not 'impossible'"
        ),
    )

    # ---- checks -----------------------------------------------------------
    checks: "OrderedDict[str, Any]" = OrderedDict()

    checks["committed_totals_reproduced"] = OrderedDict(
        expected="the committed (calls, hits, unsafe) for every policy",
        observed={
            entry["id"]: OrderedDict(
                rebuilt=[entry["calls"], entry["hits"], entry["reclassified_total"]],
                committed=[COMMITTED[entry["id"]]["calls"], COMMITTED[entry["id"]]["hits"],
                           COMMITTED[entry["id"]]["unsafe"]],
            )
            for entry in per_policy
        },
        passed=all(
            (entry["calls"], entry["hits"], entry["reclassified_total"])
            == (COMMITTED[entry["id"]]["calls"], COMMITTED[entry["id"]]["hits"],
                COMMITTED[entry["id"]]["unsafe"])
            for entry in per_policy
        ),
    )
    checks["partition_of_committed_unsafe"] = OrderedDict(
        expected=(
            "D1 = writes_served, D2 = stale_reads, D3_damaging + D3_harmless = "
            "cross_tenant, and the four buckets sum to the committed total"
        ),
        observed={
            entry["id"]: OrderedDict(
                D1=entry["D1"],
                committed_writes_served=COMMITTED[entry["id"]]["writes_served"],
                D2=entry["D2"],
                committed_stale_reads=COMMITTED[entry["id"]]["stale_reads"],
                D3=entry["D3_damaging"] + entry["D3_harmless"],
                committed_cross_tenant=COMMITTED[entry["id"]]["cross_tenant"],
                class_sum=entry["D1"] + entry["D2"] + entry["D3_damaging"] + entry["D3_harmless"],
                reclassified_total=entry["reclassified_total"],
            )
            for entry in per_policy
        },
        passed=all(
            entry["D1"] == COMMITTED[entry["id"]]["writes_served"]
            and entry["D2"] == COMMITTED[entry["id"]]["stale_reads"]
            and entry["D3_damaging"] + entry["D3_harmless"]
            == COMMITTED[entry["id"]]["cross_tenant"]
            and entry["D1"] + entry["D2"] + entry["D3_damaging"] + entry["D3_harmless"]
            == entry["reclassified_total"]
            for entry in per_policy
        ),
    )
    checks["cluster_count"] = OrderedDict(
        clusters=n_instances,
        cells=sum(len(members) for members in clusters.values()),
        composition=OrderedDict(sorted(composition.items())),
        passed=(n_instances == EXPECTED_INSTANCES
                and sum(len(members) for members in clusters.values()) == EXPECTED_CELLS),
        why=(
            "165 task instances, 325 cells: retail contributes 115 tasks performed by two "
            "models, airline 50 tasks of which 45 were performed by two models "
            "(115*2 + 50 + 45 = 325). Resampling cells would treat two runs of one task as "
            "independent draws of the task population."
        ),
    )
    catalog_present = {tool: tool in dependent for tool in CATALOG_PRIOR}
    dependent_present = {tool: tool in dependent for tool in DEPENDENT_PRIOR}
    catalog_ok = all(catalog_present.values()) and all(
        not dependent[tool] for tool in CATALOG_PRIOR
    )
    dependent_ok = all(dependent_present.values()) and all(
        dependent[tool] for tool in DEPENDENT_PRIOR
    )
    checks["identity_priors"] = OrderedDict(
        catalog_tools=OrderedDict(
            tools=list(CATALOG_PRIOR),
            present=catalog_present,
            measured={tool: dependent.get(tool) for tool in CATALOG_PRIOR},
            passed=catalog_ok,
        ),
        dependent_tools=OrderedDict(
            tools=list(DEPENDENT_PRIOR),
            present=dependent_present,
            measured={tool: dependent.get(tool) for tool in DEPENDENT_PRIOR},
            passed=dependent_ok,
        ),
        passed=catalog_ok and dependent_ok,
        why=(
            "the catalog list is the task's prior; the measurement is independent of it, so "
            "agreement is evidence and disagreement would be a finding"
        ),
    )
    checks["identity_evidence_covers_every_hit_tool"] = OrderedDict(
        tools_with_d3_hits=sorted({
            tool for entry in per_policy for tool in entry["by_tool"]["D3_damaging"]
        } | {
            tool for entry in per_policy for tool in entry["by_tool"]["D3_harmless"]
        }),
        missing=[
            tool for entry in per_policy for tool in list(entry["by_tool"]["D3_damaging"]) +
            list(entry["by_tool"]["D3_harmless"]) if tool not in dependent
        ],
    )
    checks["identity_evidence_covers_every_hit_tool"]["passed"] = not checks[
        "identity_evidence_covers_every_hit_tool"]["missing"]

    checks["typed_zero_under_damage_split"] = OrderedDict(
        warranted_unsafe=by_id["P4"]["warranted_unsafe"],
        passed=by_id["P4"]["warranted_unsafe"] == 0,
        why=(
            "0 is a count on 1,728 admitted reads, not an analytic consequence of the "
            "classification: D1/D2/D3 are defined by missing side effects, stale state and "
            "identity-selected payloads, none of which names the typed key. The bound below "
            "says how much a zero on this sample could hide."
        ),
    )
    checks["p1_p2_cross_tenant_hits_are_all_harmless"] = OrderedDict(
        p1_d3_damaging=by_id["P1"]["D3_damaging"],
        p2_d3_damaging=by_id["P2"]["D3_damaging"],
        passed=(by_id["P1"]["D3_damaging"] == 0 and by_id["P2"]["D3_damaging"] == 0),
        why=(
            "both keys include every argument, so a cross-tenant hit needs arguments that "
            "carry no identity, which is exactly the set of tools measured identity-"
            "independent; P3 drops user_id, which is why its identity class is not all "
            "harmless"
        ),
    )
    checks["identity_arguments_on_d3_hits"] = OrderedDict(
        D3_damaging=sum(entry["D3_damaging"] for entry in per_policy),
        D3_damaging_with_a_different_identity_argument=sum(
            entry["identity_arguments_differ"].get("D3_damaging", 0) for entry in per_policy
        ),
        D3_harmless=sum(entry["D3_harmless"] for entry in per_policy),
        D3_harmless_with_a_different_identity_argument=sum(
            entry["identity_arguments_differ"].get("D3_harmless", 0) for entry in per_policy
        ),
        passed=(
            sum(entry["identity_arguments_differ"].get("D3_damaging", 0) for entry in per_policy)
            == sum(entry["D3_damaging"] for entry in per_policy)
            and sum(entry["identity_arguments_differ"].get("D3_harmless", 0)
                    for entry in per_policy) == 0
        ),
        why=(
            "the tool-class verdict and the per-hit diagnosis must agree: every hit called "
            "identity-damaging must carry a different identity argument than its memo, and no "
            "hit called harmless may. This is the internal consistency test of D3, independent "
            "of the tool table"
        ),
    )
    checks["bootstrap_resamples"] = OrderedDict(
        requested=resamples, used=used, passed=(used == resamples),
    )

    return OrderedDict(
        schema=SCHEMA,
        task=TASK,
        result_kind="RECLASSIFICATION+BOOTSTRAP",
        reviewer_objection=(
            "the committed unsafe predicate is the union of the three components of the "
            "typed key (effect, entity/freshness, identity), so the typed policy's zero is "
            "analytic. This document reclassifies the same hits by damage -- a mutation that "
            "did not run, a value older than a landed write, a value produced under another "
            "identity -- and splits the identity class by an empirical measure of whether the "
            "tool's output depends on identity at all."
        ),
        scope=(
            "tau-bench only: the first trial of every task in the committed corpus, replayed "
            "through the P1/P2/P3 run-scope caches of unsafe_cache_baselines, plus P4 so the "
            "typed policy's zero is re-derived by the same walk rather than trusted. The hits "
            "are the committed ones; only their classification, the intervals and the bound "
            "are new."
        ),
        method=_method_block(n_instances, resamples, used, seed),
        provenance=_provenance_block(
            corpus_sha=corpus_sha,
            first_trial_records=first_trial_records,
            expected_sha=expected_sha,
            observed_count=len(observed),
            observed_sha=observed_sha,
            shas=shas,
        ),
        identity_dependence=evidence,
        policies=per_policy,
        instances=OrderedDict(
            total=n_instances,
            composition=OrderedDict(sorted(composition.items())),
            cells=sum(len(members) for members in clusters.values()),
            p1=OrderedDict(
                with_warranted_unsafe=p1_instances,
                share_with_warranted_unsafe=round(p1_instances / n_instances, 6),
                with_any_hit_by_class=OrderedDict(instance_counts),
            ),
            p2=OrderedDict(
                with_warranted_unsafe=p2_instances,
                share_with_warranted_unsafe=round(p2_instances / n_instances, 6),
            ),
            p3=OrderedDict(
                with_warranted_unsafe=p3_instances,
                share_with_warranted_unsafe=round(p3_instances / n_instances, 6),
            ),
        ),
        intervals=intervals,
        supplementary_point_estimates=supplementary,
        typed_zero_bound=bound,
        checks=checks,
        notes=[
            (
                "D3's harmlessness is a property of the tool, not of the hit: an identical "
                "argument list that names no identity cannot return a different answer to a "
                "different tenant, and the catalog tools are shown to return identical "
                "observations for identical arguments across tenants before any write."
            ),
            (
                "The reduction is one-sided: a hit counted damaging here is damaging, a hit "
                "called harmless is harmless only in the sense that the served value equals "
                "the value the tool would have produced. Both claims are auditable in "
                "policies[].by_tool and identity_dependence.tools."
            ),
            (
                "P3's identity damage is all get_user_details: its key drops user_id, so "
                "tenant B's profile read is answered with tenant A's profile. P1/P2 cannot do "
                "this, because their keys include the identity argument -- which is why their "
                "identity class is entirely harmless and P3's is not."
            ),
            (
                "The exact bound is per denominator, and the denominators differ in meaning: "
                "per admitted read is a call-level error rate, per served hit is the 'how much "
                "of what it served could be wrong' rate. Both are reported rather than one "
                "being chosen."
            ),
            (
                "payload_user_specific is not the verdict. get_order_details and "
                "get_reservation_details return payloads that name a user (the order's "
                "owner) and find_user_id_by_name_zip returns a user id, yet all three are "
                "identity-independent: the answer is selected by non-identity arguments, so a "
                "key that includes those arguments cannot confuse two people. The task's "
                "second clause -- takes user_id/email and returns user-specific data -- is "
                "realised as E1, which asks whether the payload changes when only the "
                "identity argument changes."
            ),
            (
                "The reduction, spelled out: P1's committed 344 = 240 warranted + 104 "
                "harmless; P2's 165 = 61 + 104; P3's 595 = 491 + 104. The harmless 104 is the "
                "same hit set in all three policies -- the catalog and flight reads -- so it "
                "is one measured quantity seen three times, not three independent findings."
            ),
            (
                "Clustered intervals stay slightly optimistic: each run keeps its own "
                "run-scope cache, so an instance's outcomes also depend on the instances "
                "before it in the same run. Four runs cannot support a block bootstrap, so "
                "that carryover is stated, not modelled."
            ),
        ],
        reproduce=(
            "python3 examples/benchmarks/unsafe_damage.py  (seed 20240601, 10,000 instance "
            "resamples; byte-identical output on repeat)"
        ),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def alignment(
    observed: Sequence[Mapping[str, Any]],
    first: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Do the observed records carry the same calls, in order, as the committed ones?"""
    index = {(str(record.get("source")), str(record.get("task_id"))): record
             for record in observed}
    mismatched = 0
    missing = 0
    for record in first:
        key = (str(record.get("source")), str(record.get("task_id")))
        twin = index.get(key)
        if twin is None:
            missing += 1
            continue
        committed_calls = [
            (str(tool), canon(dict(args or {})))
            for turn in record.get("turns") or []
            for tool, args in turn
        ]
        observed_calls = [
            (str(tool), canon(dict(args or {})))
            for tool, args, _out in twin.get("calls") or []
        ]
        if committed_calls != observed_calls:
            mismatched += 1
    return OrderedDict(
        observed_records=len(observed),
        first_trial_records=len(first),
        missing_twin=missing,
        call_sequence_mismatches=mismatched,
        passed=(missing == 0 and mismatched == 0),
        why=(
            "the identity evidence is about *these* calls only if the observation-bearing "
            "corpus carries the same calls in the same order; a mismatch would mean the "
            "evidence comes from a different workload"
        ),
    )


def summary(document: Mapping[str, Any]) -> str:
    lines = []
    for entry in document["policies"]:
        lines.append(
            f"  {entry['id']} {entry['name']:<18} D1 {entry['D1']:3d}  D2 {entry['D2']:3d}  "
            f"D3-dmg {entry['D3_damaging']:3d}  D3-harmless {entry['D3_harmless']:3d}  "
            f"warranted {entry['warranted_unsafe']:3d} of committed "
            f"{entry['committed_unsafe_hits']:3d} ({100.0 * entry['warranted_share_of_hits']:.1f}% "
            f"of {entry['hits']} hits)"
        )
    rates = document["intervals"]["p1_warranted_per_1000_calls"]
    share = document["intervals"]["p1_share_instances_with_warranted"]
    bound = document["typed_zero_bound"]["admitted_reads"]
    return "\n".join(lines) + (
        f"\n  P1 warranted-unsafe per 1,000 calls {rates['estimate']:.2f} "
        f"[{rates['ci_low']:.2f}, {rates['ci_high']:.2f}]"
        f"\n  P1 instances with a warranted hit   "
        f"{document['instances']['p1']['with_warranted_unsafe']}"
        f"/{document['instances']['total']} = {share['estimate']:.4f} "
        f"[{share['ci_low']:.4f}, {share['ci_high']:.4f}]"
        f"\n  typed zero, exact 95% upper bound   {bound['exact_upper_per_1000']:.2f} per 1,000 "
        f"admitted reads (rule of three {bound['rule_of_three_per_1000']:.2f})"
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Damage-based reclassification of the committed unsafe cache hits."
    )
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--observed", type=Path, default=OBSERVED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)

    if args.resamples <= 0:
        print("unsafe_damage: error: --resamples must be positive", file=sys.stderr)
        return 2
    if not args.corpus.is_file() or not args.observed.is_file():
        print("unsafe_damage: error: corpus or observed corpus missing", file=sys.stderr)
        return 2

    unsafe = load_module("_ud_unsafe", UNSAFE_PATH)
    require_attrs(
        unsafe,
        ("POLICIES", "build_streams", "replay", "load_identity", "load_records",
         "_served_key", "_intervening", "_evict", "EXPECTED_SHA256"),
        "unsafe_cache_baselines",
    )
    ci, identity_sha = unsafe.load_identity()

    corpus_sha = sha256_file(args.corpus)
    if corpus_sha != unsafe.EXPECTED_SHA256:
        print(
            f"unsafe_damage: corpus digest {corpus_sha} does not match the committed "
            f"{unsafe.EXPECTED_SHA256}; refusing to measure a different experiment",
            file=sys.stderr,
        )
        return 2

    read_tools = sorted(
        tool for tool, effect in ci.EFFECT_CLASSES.items() if effect == "read"
    )

    def build() -> Dict[str, Any]:
        records = unsafe.load_records(args.corpus)
        first = [record for record in records if record.get("trial", 0) == 0]
        streams = unsafe.build_streams(first)
        clusters, composition = cluster_records(first)

        cluster_of: Dict[str, str] = {}
        for label, cells in clusters.items():
            for source, task_id in cells:
                cluster_of[f"{source}#{task_id}"] = label
        for source, calls in streams.items():
            for call in calls:
                if f"{source}#{call.task_id}" not in cluster_of:
                    raise SystemExit(
                        f"unsafe_damage: call in {source!r} names task {call.task_id!r} with "
                        "no first-trial record; the cluster label cannot be assigned."
                    )

        policies = {policy.pid: policy for policy in unsafe.POLICIES}
        rows_by_policy: Dict[str, List[Row]] = {}
        for policy_id in ("P1", "P2", "P3", "P4"):
            policy = policies[policy_id]
            rows = walk(streams, policy, ci, cluster_of, unsafe)
            verify(rows, unsafe.replay(streams, policy, ci), policy.name)
            rows_by_policy[policy_id] = rows

        observed = load_jsonl(args.observed)
        evidence = identity_evidence(observed, ci, read_tools)
        dependent = {
            tool: bool(entry["identity_dependent"]) for tool, entry in evidence["tools"].items()
        }
        classified = {
            policy_id: reclassify(rows_by_policy[policy_id], dependent)
            for policy_id in ("P1", "P2", "P3", "P4")
        }

        matrix = []
        for cluster in sorted(clusters):
            cells = []
            for policy_id in ("P1", "P2", "P3"):
                agg = classified[policy_id]["per_cluster"][cluster]
                cells.append((agg["calls"], agg["warranted"], 1 if agg["warranted"] else 0))
            matrix.append((cells[0][0], cells[0][1], cells[1][0], cells[1][1],
                           cells[2][0], cells[2][1], cells[0][2], cells[1][2], cells[2][2]))
        series, used = bootstrap(matrix, args.resamples, args.seed)

        document = build_document(
            observed=observed,
            clusters=clusters,
            composition=composition,
            classified=classified,
            evidence=evidence,
            series=series,
            used=used,
            resamples=args.resamples,
            seed=args.seed,
            corpus_sha=corpus_sha,
            observed_sha=sha256_file(args.observed),
            first_trial_records=len(first),
            expected_sha=unsafe.EXPECTED_SHA256,
            shas=OrderedDict(identity=identity_sha, unsafe=sha256_file(UNSAFE_PATH)),
        )
        document["checks"]["walk_equals_module_replay"] = OrderedDict(
            passed=True,
            why=(
                "each policy's per-call walk was folded and compared field for field with "
                "unsafe_cache_baselines.replay() before any hit was classified"
            ),
        )
        document["checks"]["observed_corpus_aligned_with_committed"] = alignment(observed, first)
        return document

    document = build()
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if json.dumps(build(), indent=2, ensure_ascii=False) + "\n" != text:
        print("unsafe_damage: build is not deterministic; refusing to write", file=sys.stderr)
        return 3

    failed = [
        name for name, value in document["checks"].items()
        if isinstance(value, Mapping) and value.get("passed") is False
    ]
    if failed:
        print(f"unsafe_damage: WARNING invariants failed: {', '.join(failed)}", file=sys.stderr)

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(out)

    print(f"unsafe_damage: {document['instances']['total']} task instances, "
          f"{EXPECTED_CALLS:,} calls, {args.resamples:,} resamples (seed {args.seed})")
    print(summary(document))
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
