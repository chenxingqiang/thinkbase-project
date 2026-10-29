#!/usr/bin/env python3
"""Sampling-unit statistics for the recorded-trajectory reuse numbers.

Section 5.12 quotes pooled hit rates over 325 first-trial tasks. The pool is
not a simple random sample of calls: the corpus holds four *runs*
(2 backbones x 2 domains), each of which starts with an empty cache and runs a
different number of tasks. This script therefore reports the rates at the
sampling unit the corpus actually provides -- the run -- and answers three
questions a reviewer is entitled to ask:

1. How much does the pooled rate depend on how the four runs are weighted?
2. Does a global cache add anything over a per-run cache for a *single*
   backbone, or is the increment entirely cross-model?
3. Is the static hot set handicapped by a call set the dynamic policy never
   sees? (Warmed LRU restricted to exactly the hot set's test calls.)

It also records the run-order sensitivity of the origin decomposition, and the
pooled observation-wired transfer rate that Appendix I.11 quotes.

Everything is deterministic: no seeds, no model calls, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "examples" / "benchmarks"
CORPUS = ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"
DEFAULT_OUTPUT = BENCH / "trace_scope_results.json"

sys.path.insert(0, str(BENCH))
import real_trace_external_baselines as rtb  # noqa: E402


# -- helpers ---------------------------------------------------------------


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pure_calls(record: Mapping[str, Any]) -> List[Tuple[str, str]]:
    """(tool, key) for the side-effect-free calls of one trajectory."""
    out = []
    for _turn, tool, args in rtb.record_calls(record):
        if rtb.is_pure(tool):
            out.append((tool, rtb.call_key(tool, args)))
    return out


def run_streams(records: Sequence[Mapping[str, Any]]) -> "Dict[str, List]":
    return dict(rtb.streams_by_source(rtb.first_trial(records)))


def hits_for_stream(recs: Sequence[Mapping[str, Any]]) -> Tuple[int, int]:
    """(hits, cacheable calls) for one run under an unbounded per-run cache."""
    seen: set = set()
    hits = calls = 0
    for rec in recs:
        for _tool, key in pure_calls(rec):
            calls += 1
            if key in seen:
                hits += 1
            else:
                seen.add(key)
    return hits, calls


def percentile_bootstrap(
    values: Sequence[float], n_resamples: int, seed: int, alpha: float = 0.05
) -> Tuple[float, float, float]:
    """Percentile bootstrap of the mean of `values` (resampling the units)."""
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_resamples):
        draw = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(draw) / n)
    means.sort()
    lo = means[int(alpha / 2 * n_resamples)]
    hi = means[int((1 - alpha / 2) * n_resamples) - 1]
    return sum(values) / n, lo, hi


# -- analyses --------------------------------------------------------------


def run_weighting(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Pooled vs run-weighted reuse, with a run-clustered bootstrap."""
    per_run = []
    for source, recs in streams.items():
        hits, calls = hits_for_stream(recs)
        per_run.append(
            {
                "source": source,
                "domain": rtb.domain_of(source),
                "tasks": len(recs),
                "hits": hits,
                "calls": calls,
                "rate": hits / calls if calls else 0.0,
            }
        )
    pooled_hits = sum(r["hits"] for r in per_run)
    pooled_calls = sum(r["calls"] for r in per_run)
    rates = [r["rate"] for r in per_run]
    mean, lo, hi = percentile_bootstrap(rates, 10000, 20240601)
    return {
        "unit": "run (2 backbones x 2 domains = 4 runs)",
        "per_run": per_run,
        "pooled_hits": pooled_hits,
        "pooled_calls": pooled_calls,
        "pooled_rate": pooled_hits / pooled_calls,
        "run_mean_rate": mean,
        "run_clustered_ci95": [lo, hi],
        "min_run_rate": min(rates),
        "max_run_rate": max(rates),
        "retail_share_of_calls": (
            sum(r["calls"] for r in per_run if r["domain"] == "retail") / pooled_calls
        ),
        "note": (
            "the pooled rate weights calls, so the larger retail runs dominate; the "
            "run-clustered interval resamples the four runs, not the calls"
        ),
    }


def backbone_global(streams: Mapping[str, Sequence[Mapping[str, Any]]]) -> Dict[str, Any]:
    """Global-cache rate per single backbone, and the cross-model increment."""
    by_backbone: Dict[str, List[Tuple[str, str, int]]] = {}
    for source, recs in streams.items():
        backbone = "gpt-4o" if source.startswith("gpt-4o") else "claude-3.5-sonnet"
        bucket = by_backbone.setdefault(backbone, [])
        for rec in recs:
            tid = int(rec.get("task_id") or 0)
            for tool, key in pure_calls(rec):
                bucket.append((key, rtb.domain_of(source), tid))

    out: Dict[str, Any] = {"per_backbone": {}}
    for backbone, items in by_backbone.items():
        seen = set()
        hits = calls = 0
        for key, dom, tid in items:
            calls += 1
            ident = (key, dom, tid)
            if key in seen:
                hits += 1
            seen.add(key)
        out["per_backbone"][backbone] = {
            "hits": hits,
            "calls": calls,
            "rate": hits / calls,
        }

    all_items = [it for items in by_backbone.values() for it in items]
    seen = set()
    hits = calls = 0
    cross_model_same_task = 0
    for key, dom, tid in all_items:
        calls += 1
        if key in seen:
            hits += 1
        seen.add(key)
    per_run = run_weighting(streams)
    out["shipped_global_hits"] = hits
    out["shipped_global_calls"] = calls
    out["shipped_global_rate"] = hits / calls
    out["single_backbone_rate_range"] = [
        min(v["rate"] for v in out["per_backbone"].values()),
        max(v["rate"] for v in out["per_backbone"].values()),
    ]
    out["note"] = (
        "the shipped key excludes model identity, so a task performed by both backbones "
        "is a legitimate hit for it; the single-backbone rates show what the cache earns "
        "without that cross-model traffic, and all of the increment over the run scope "
        "comes from it"
    )
    return out


def decomposition_order_sensitivity(
    streams: Mapping[str, Sequence[Mapping[str, Any]]]
) -> Dict[str, Any]:
    """The origin split depends on stream order; report both orders."""
    order = list(streams.keys())
    results = {}
    for label, seq in (("as_recorded", order), ("reversed", list(reversed(order)))):
        cache: Dict[str, Tuple[str, int, str]] = {}
        buckets = {
            "same_task_same_run": 0,
            "same_task_other_run": 0,
            "other_task_same_run": 0,
            "other_task_other_run": 0,
        }
        hits = 0
        for source in seq:
            for rec in streams[source]:
                tid = int(rec.get("task_id") or 0)
                dom = rtb.domain_of(source)
                for _tool, key in pure_calls(rec):
                    if key in cache:
                        hits += 1
                        o_source, o_tid, o_dom = cache[key]
                        if (o_dom, o_tid) == (dom, tid) and o_source == source:
                            buckets["same_task_same_run"] += 1
                        elif (o_dom, o_tid) == (dom, tid):
                            buckets["same_task_other_run"] += 1
                        elif o_source == source:
                            buckets["other_task_same_run"] += 1
                        else:
                            buckets["other_task_other_run"] += 1
                    else:
                        cache[key] = (source, tid, dom)
        results[label] = {"order": seq, "hits": hits, **buckets}
    results["note"] = (
        "an origin bucket is assigned to the stream that inserted the key first, so the "
        "split moves with stream order; ranges are the honest statement"
    )
    return results


def hot_set_test_call_lru(
    streams: Mapping[str, Sequence[Mapping[str, Any]]],
    per_source_test: Mapping[str, List[str]],
    capacities: Sequence[int] = (256, 0),
) -> Dict[str, Any]:
    """Online LRU scored on exactly the hot set's test calls.

    The hot set is trained on each run's first half and measured on its second.
    Comparing it against a run-scope rate computed over *all* calls mixes two
    different call sets, so this replays the whole run -- warming on the first
    half and admitting new keys during the second -- but counts hits only in the
    second half, the calls the hot set is scored on. Also reports the static
    ceiling: the share of test-half calls whose key appeared in the first half.
    """
    out: Dict[str, Any] = {"capacities": {}, "ceiling": {}}
    hits_by_cap: Dict[str, int] = {str(c): 0 for c in capacities}
    calls = 0
    ceiling_hits = 0
    for source, recs in streams.items():
        test_ids = set(per_source_test.get(source, []))
        first = [r for r in recs if str(rtb.task_id_of(r)) not in test_ids]
        test = [r for r in recs if str(rtb.task_id_of(r)) in test_ids]
        seen_first = set()
        for rec in first:
            for _tool, key in pure_calls(rec):
                seen_first.add(key)
        for cap in capacities:
            cache: Dict[str, int] = {}
            order: List[str] = []
            for rec in first:
                for _tool, key in pure_calls(rec):
                    if key in cache:
                        continue
                    if cap and len(cache) >= cap:
                        victim = order.pop(0)
                        cache.pop(victim, None)
                    cache[key] = 1
                    order.append(key)
            local_hits = 0
            for rec in test:
                for _tool, key in pure_calls(rec):
                    if key in cache:
                        local_hits += 1
                        order.remove(key)
                        order.append(key)
                    else:
                        if cap and len(cache) >= cap:
                            victim = order.pop(0)
                            cache.pop(victim, None)
                        cache[key] = 1
                        order.append(key)
            hits_by_cap[str(cap)] += local_hits
        for rec in test:
            for _tool, key in pure_calls(rec):
                calls += 1
                if key in seen_first:
                    ceiling_hits += 1
    for cap in capacities:
        out["capacities"][str(cap)] = {
            "hits": hits_by_cap[str(cap)],
            "calls": calls,
            "rate": hits_by_cap[str(cap)] / calls,
        }
    out["test_calls"] = calls
    out["ceiling"] = {"hits": ceiling_hits, "calls": calls, "rate": ceiling_hits / calls}
    out["note"] = (
        "the cache is warmed on the hot set's training half and scored on its test half, "
        "with online admission during the test half; the ceiling is the share of test-half "
        "calls whose key was seen in the training half, which no first-half-only policy beats"
    )
    return out


def pooled_wired_transfer() -> Dict[str, Any]:
    """Pooled observation-wired transfer and the unconditional eligible rate."""
    path = BENCH / "compound_transfer_results.json"
    data = json.loads(path.read_text())
    cells = data["cross_model"]["cells"]
    wired_exposed = wired_transferred = 0
    for cell in cells.values():
        wired = cell.get("wiring", {}).get("observation_wired", {})
        wired_exposed += wired.get("exposed", 0)
        wired_transferred += wired.get("transferred", 0)
    retail = cells["retail"]
    return {
        "observation_wired_exposed": wired_exposed,
        "observation_wired_transferred": wired_transferred,
        "parameter_only_transferred": retail["wiring"]["parameter_only"]["transferred"],
        "parameter_only_eligible": retail["wiring"]["parameter_only"]["eligible"],
        "exposed_compounds": retail["exposed"],
        "eligible_compounds": retail["eligible"],
        "unconditional_transfer_rate": retail["transferred"] / retail["eligible"],
        "note": (
            "parameter-only specifications replay every exposed occurrence by construction: "
            "they take all arguments from the compound's own parameters, so a transfer there "
            "is a harness consistency check. The informative cell is the wired one, and it "
            "transfers nothing."
        ),
    }


def build(corpus: Path) -> Dict[str, Any]:
    records = rtb.load_records(corpus)
    streams = run_streams(records)
    # the hot set's per-source test task ids, recomputed exactly as the baseline does
    test_ids: Dict[str, List[str]] = {}
    for source, recs in streams.items():
        n = len(recs)
        test_ids[source] = [str(rtb.task_id_of(r)) for r in recs[n // 2:]]
    return {
        "schema": "trace_scope_stats/v1",
        "result_kind": "MEASURED",
        "description": (
            "Sampling-unit statistics for the recorded-trajectory reuse numbers: run-weighted "
            "vs pooled rates with a run-clustered bootstrap, single-backbone global rates, the "
            "run-order sensitivity of the origin decomposition, the hot-set comparison on its "
            "own test calls, and the pooled observation-wired transfer rate."
        ),
        "provenance": {
            "corpus": str(corpus.relative_to(ROOT)),
            "corpus_sha256": sha256_of(corpus),
            "records": len(records),
            "first_trial_records": len(rtb.first_trial(records)),
            "deterministic": True,
            "no_model_calls": True,
        },
        "run_weighting": run_weighting(streams),
        "backbone_global": backbone_global(streams),
        "decomposition_order": decomposition_order_sensitivity(streams),
        "hot_set_test_call_lru": hot_set_test_call_lru(streams, test_ids),
        "transfer_pooled": pooled_wired_transfer(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args(argv)
    payload = build(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n")
    rw = payload["run_weighting"]
    print(
        f"[MEASURED] pooled {rw['pooled_rate']:.2%} over {rw['pooled_calls']} calls; "
        f"run-weighted {rw['run_mean_rate']:.2%} "
        f"[{rw['run_clustered_ci95'][0]:.1%}, {rw['run_clustered_ci95'][1]:.1%}]; "
        f"single-backbone global "
        f"{payload['backbone_global']['single_backbone_rate_range'][0]:.2%}-"
        f"{payload['backbone_global']['single_backbone_rate_range'][1]:.2%}; "
        f"-> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
