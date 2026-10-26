"""When does the instruction-level cache key stop being at parity?

Review-motivated experiment for the Section 5.2 parity finding: on the
Zipf benchmark the instruction-level (instruction, params) key and a
naive exact-response key on the request string issue *identical* call
counts, because the benchmark's params are single-key dicts and the
naive arm canonicalizes by sorting.  Parity is thus a property of that
workload, not of the mechanism.  This experiment constructs the two
families of workload that separate the keys, using the production
``build_cache_key`` (no mock of the key function):

W1 -- parameter reordering.  Every task carries the same semantic
params under a random insertion order (multi-key dicts, keys inserted
in a random permutation per call).  The instruction-level key
canonicalizes (``sort_keys=True``) and hits; an exact-response key on
the serialized request (insertion-order-sensitive) misses on every
reorder.  This measures the gap the Section 5.2 workload cannot show.

W2 -- template-version churn.  The same instruction and params are
rendered under an alternating template version (v1/v2, semantics
changing with the template).  The *default* instruction key (params
only) keeps hitting -- a FALSE hit once the template changes, which is
the Section 2.1 cache-validity condition violated by construction; the
strict namespace key (folds template_version in) misses exactly when
the template changes, i.e. it is correct where the default key is not.
The exact-response key also misses (its payload embeds the template).

Discipline: hit/miss accounting only, over synthetic dicts -- no LLM,
no tokens, no latency; every number in the output is computed here.
The key functions under test are the production ones, imported, not
reimplemented.

Stdlib + agentflow only.  Usage:
    ./venv/bin/python examples/benchmarks/cache_key_differentiation.py \
        [--tasks N] [--seeds S] [--output FILE]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from agentflow.core.isa.instruction_cache import (  # noqa: E402
    CacheKeyNamespace,
    build_cache_key,
)

DEFAULT_OUTPUT = (
    REPO_ROOT / "examples" / "benchmarks"
    / "cache_key_differentiation_results.json"
)
SCHEMA = "cache_key_differentiation/v1"

N_KEYS = 6  # params dicts carry six keys (order has room to vary)
ZIPF_S = 1.0  # same skew family as Section 5.2 so totals are comparable
PARAM_VALUES = 20


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


UNIVERSE = 100  # semantic task universe size (small, so Zipf repeats)


def draw_tasks(rng: random.Random, n_tasks: int) -> List[Dict[str, int]]:
    """Zipf-drawn task mix over a fixed universe of semantic dicts.

    A universe of ``UNIVERSE`` params dicts is drawn once (same value
    distribution as Section 5.2); then ``n_tasks`` draws pick a universe
    entry by truncated Zipf over entry index -- the same repetition
    structure as the Section 5.2 benchmark, so the absolute hit rates
    are comparable.  Each drawn entry is a copy; presentation order is
    the caller's concern.
    """
    weights = [1.0 / (i + 1) ** ZIPF_S for i in range(PARAM_VALUES)]
    universe = [
        {f"k{i}": rng.choices(range(PARAM_VALUES), weights=weights)[0]
         for i in range(N_KEYS)}
        for _ in range(UNIVERSE)
    ]
    zw = [1.0 / (i + 1) ** ZIPF_S for i in range(UNIVERSE)]
    return [dict(universe[rng.choices(range(UNIVERSE), weights=zw)[0]])
            for _ in range(n_tasks)]


def reorder(rng: random.Random, params: Dict[str, int]) -> Dict[str, int]:
    """Same mapping, random insertion order."""
    items = list(params.items())
    rng.shuffle(items)
    return dict(items)


def w1_reordering(rng: random.Random, n_tasks: int) -> Dict[str, Any]:
    """Instruction key vs order-sensitive response key under reorders."""
    tasks = draw_tasks(rng, n_tasks)
    instr_key_hits = resp_key_hits = 0
    seen_instr: Dict[str, str] = {}
    seen_resp: Dict[str, str] = {}
    for params in tasks:
        presented = reorder(rng, params)
        # Instruction-level key: production default (params-only, canonical).
        ik = build_cache_key("instr_w1", presented)
        if ik in seen_instr:
            instr_key_hits += 1
        else:
            seen_instr[ik] = "v"
        # Exact-response key: the serialized request as an HTTP cache
        # would see it (insertion order preserved, no canonicalization).
        rk = json.dumps({"instruction": "instr_w1", "params": presented})
        if rk in seen_resp:
            resp_key_hits += 1
        else:
            seen_resp[rk] = "v"
    return {
        "scenario": "w1_param_reordering",
        "n_tasks": n_tasks,
        "instruction_key_hits": instr_key_hits,
        "response_key_hits": resp_key_hits,
        "instruction_key_hit_rate": round(instr_key_hits / n_tasks, 4),
        "response_key_hit_rate": round(resp_key_hits / n_tasks, 4),
        "call_gap_percentage_points": round(
            (instr_key_hits - resp_key_hits) / n_tasks * 100, 2),
    }


def w2_template_churn(rng: random.Random, n_tasks: int,
                      period: int = 2) -> Dict[str, Any]:
    """Default vs strict-namespace key under template-version churn."""
    tasks = draw_tasks(rng, n_tasks)
    default_hits = strict_hits = strict_false_hits = 0
    default_false_hits = 0
    seen_default: Dict[str, str] = {}
    seen_strict: Dict[str, str] = {}
    for i, params in enumerate(tasks):
        template = "v1" if (i // period) % 2 == 0 else "v2"
        # Default key: params only -- template-blind.
        dk = build_cache_key("instr_w2", params)
        # Strict key: production namespace with template folded in.
        sk = build_cache_key(
            "instr_w2", params,
            CacheKeyNamespace(model_alias="alias-a",
                              template_version=template))
        if dk in seen_default:
            default_hits += 1
            if seen_default[dk] != template:
                default_false_hits += 1  # served under the other template
        else:
            seen_default[dk] = template
        if sk in seen_strict:
            strict_hits += 1
            if seen_strict[sk] != template:
                strict_false_hits += 1
        else:
            seen_strict[sk] = template
    n = n_tasks
    return {
        "scenario": "w2_template_churn",
        "n_tasks": n_tasks,
        "churn_period": period,
        "default_key_hits": default_hits,
        "default_key_hit_rate": round(default_hits / n, 4),
        "default_key_false_hits": default_false_hits,
        "default_key_false_hit_rate": round(default_false_hits / n, 4),
        "strict_key_hits": strict_hits,
        "strict_key_hit_rate": round(strict_hits / n, 4),
        "strict_key_false_hits": strict_false_hits,
    }


def run(n_tasks: int, seeds: int) -> Dict[str, Any]:
    w1, w2 = [], []
    for seed in range(seeds):
        rng = random.Random(4000 + seed)
        w1.append(w1_reordering(rng, n_tasks))
        rng = random.Random(5000 + seed)
        w2.append(w2_template_churn(rng, n_tasks))

    def agg(rows: List[Dict[str, Any]], keys: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k in keys:
            vals = [r[k] for r in rows]
            out[k] = {
                "mean": round(statistics.mean(vals), 4),
                "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0,
            }
        return out

    return {
        "experiment": "cache_key_differentiation",
        "schema": SCHEMA,
        "description": (
            "Workloads separating the instruction-level cache key from an "
            "exact-response key (W1: multi-key params under random "
            "insertion order) and the default params-only key from the "
            "strict namespace key (W2: alternating template version). "
            "Production build_cache_key under test; hit/miss accounting "
            "only, no LLM backend."
        ),
        "provenance": {
            "git_rev": git_rev(),
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "n_tasks": n_tasks,
            "seeds": seeds,
            "seed_bases": {"w1": 4000, "w2": 5000},
            "zipf_s": ZIPF_S,
            "param_keys": N_KEYS,
            "param_values": PARAM_VALUES,
            "task_universe": UNIVERSE,
        },
        "w1_param_reordering": {
            "per_seed": w1,
            "aggregate": agg(w1, ["instruction_key_hit_rate",
                                  "response_key_hit_rate",
                                  "call_gap_percentage_points"]),
        },
        "w2_template_churn": {
            "per_seed": w2,
            "aggregate": agg(w2, ["default_key_hit_rate",
                                  "default_key_false_hit_rate",
                                  "strict_key_hit_rate"]),
        },
    }


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=int, default=2000)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)

    report = run(args.tasks, args.seeds)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    a1 = report["w1_param_reordering"]["aggregate"]
    a2 = report["w2_template_churn"]["aggregate"]
    print(f"W1 reorder: instr-key hit {a1['instruction_key_hit_rate']['mean']:.1%} vs "
          f"response-key {a1['response_key_hit_rate']['mean']:.1%} "
          f"(gap {a1['call_gap_percentage_points']['mean']:.1f} pp)")
    print(f"W2 churn:   default hit {a2['default_key_hit_rate']['mean']:.1%} "
          f"(false {a2['default_key_false_hit_rate']['mean']:.1%}) vs "
          f"strict hit {a2['strict_key_hit_rate']['mean']:.1%}")
    print(f"-> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
