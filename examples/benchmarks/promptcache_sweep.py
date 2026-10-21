#!/usr/bin/env python3
"""Token-mix sweep of the MODELED priced comparison (E9, mock, offline).

The paper's Section 5.7.1 priced comparison (protocol: Appendix G of
the supplementary material) prices one run's measured tokens under
three modes -- no optimization, AgentISA result caching, and a MODELED
provider prompt cache -- and observes that "the ranking of the two
caching strategies is a function of the input/output token mix, not a
constant": on the real-gateway mix (completions dominate) the prompt
cache saves almost nothing, while on the mock paper-scale mix (short
completions) it saves a visible fraction.  Two anchor points, no
traced curve between them.

This harness sweeps the token-mix axis between those anchors and
crosses it with the paper's three Zipf skew levels:

* mix axis -- completion tokens per call.  ``mock_measured`` uses the
  mock backend's template-accounted replies (the published mock
  paper-scale anchor); ``L<n>`` MODELS the alias's observed
  reply-binds-at-max_tokens behavior (Sections 5.7/5.7.1) as a constant
  ``n`` completion tokens per call.  ``L4`` is the input-dominated
  extreme, ``L512`` the Section 5.7.1 ceiling (completion-dominated).
* skew axis -- Zipf s in {0.5, 1.0, 1.5} (the paper's levels).
* 5 seeds per (skew, mix) cell, 200 draws per cell over the fixed
  40-task universe (paper scale); every aggregate in the output is
  computed by this script from the per-cell records.

Everything priced here is labeled MODELED-on-measured, exactly as in
the §5.7.1 protocol: prompt tokens are template-accounted measurements
(chars/4) of the exact prompt texts the classify step builds; the
swept completion lengths are MODELED constants; prices are
parameterized list-price stand-ins.  No dollar-savings claim is made.

The pricing overlay itself is IMPORTED from the §5.7.1 harness
(``compute_cost`` / ``COST_MODES`` in ``examples/
paper_real_llm_compiler_test.py``) together with the workload
primitives, so the sweep can never drift from the published method;
the seed-1000 ``mock_measured`` cells reproduce the published mock
paper-scale artifact exactly (checked in the output's
``published_anchor_checks`` and by the test suite).

Usage:
    env -u ALL_PROXY ... ./venv/bin/python \\
        examples/benchmarks/promptcache_sweep.py
    ./venv/bin/python examples/benchmarks/promptcache_sweep.py \\
        --output examples/benchmarks/promptcache_sweep_results.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Make the sibling example modules importable (same pattern as the
# §5.7.1 harness itself).
HERE = Path(__file__).resolve().parent            # examples/benchmarks
EXAMPLES_DIR = HERE.parent                        # examples/
REPO_ROOT = EXAMPLES_DIR.parent
for _p in (str(EXAMPLES_DIR), str(REPO_ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import paper_real_llm_compiler_test as pct  # noqa: E402  (§5.7.1 harness)

# The MODELED pricing overlay and the workload primitives, imported
# (never copied) from the published §5.7.1 harness.
COST_MODES = tuple(pct.COST_MODES)
compute_cost = pct.compute_cost
zipf_draws = pct.zipf_draws
CLASSIFY_PROMPT = pct.CLASSIFY_PROMPT
MOCK_REPLY = pct.DeterministicMockBackend._mock_reply

MOCK_MIX_ID = "mock_measured"

# The swept constant completion lengths (max_tokens-style ceilings the
# reasoning alias binds at; 64 = Section 5.7's ceiling, 512 =
# Section 5.7.1's).  Dense at the small end, where the curve's
# crossover sits.
MIX_CONST_L = (4, 8, 16, 24, 32, 48, 64, 128, 256, 384, 512, 1024)

DEFAULT_ZIPF_LEVELS = (0.5, 1.0, 1.5)
DEFAULT_N_SEEDS = 5
DEFAULT_SEED_BASE = 1000
DEFAULT_DRAWS = 200
DEFAULT_POPULATION = 40
DEFAULT_PRICE_IN_PER_MT = 3.0
DEFAULT_PRICE_OUT_PER_MT = 15.0
DEFAULT_CACHE_READ_RATE = 0.1

ANCHORS = {
    "input_dominated_extreme": "L4",
    "completion_dominated": "L512",
    "published_mock_paper_scale": MOCK_MIX_ID,
    "published_real_gateway_mix_analog": "L512",
}

PUBLISHED_MOCK_REL = (
    "examples/paper_real_llm_compiler_results_mock_paper.json")
PUBLISHED_REAL_REL = (
    "examples/paper_real_llm_compiler_results_real.json")
SCRIPT_REL = "examples/benchmarks/promptcache_sweep.py"
OUTPUT_REL = "examples/benchmarks/promptcache_sweep_results.json"

# The two 0.5-crossings the analysis locates on the mean curve.
RATIO_CROSSING_TARGET = 0.5
PROVIDER_SAVING_CROSSING_PCT = 5.0
OUTPUT_COST_SHARE_CROSSING = 0.5


# ---------------------------------------------------------------------------
# Mix axis
# ---------------------------------------------------------------------------

def default_mix_ids() -> List[str]:
    return [MOCK_MIX_ID] + [f"L{level}" for level in MIX_CONST_L]


def completion_tokens_for_mix(mix_id: str) -> Optional[int]:
    """Completion tokens per call modeled by ``mix_id``.

    ``mock_measured`` -> ``None`` (the mock's template-accounted reply
    per key is used instead); ``L<n>`` -> the constant ``n`` (the
    alias's observed reply-binds-at-max_tokens behavior).  Malformed
    or non-positive values raise ``ValueError``.
    """
    if mix_id == MOCK_MIX_ID:
        return None
    if not mix_id.startswith("L"):
        raise ValueError(f"unknown mix id: {mix_id!r}")
    digits = mix_id[1:]
    try:
        n = int(digits)
    except ValueError:
        raise ValueError(f"non-integer mix id: {mix_id!r}") from None
    if n <= 0:
        raise ValueError(f"non-positive completion length: {mix_id!r}")
    return n


def mix_kind(mix_id: str) -> str:
    return ("measured_mock_reply" if completion_tokens_for_mix(mix_id)
            is None else "modeled_max_tokens_binding")


# ---------------------------------------------------------------------------
# Measured input side: the classify step's prompt per task rank
# ---------------------------------------------------------------------------

def precompute_rank_table(population: int) -> Dict[int, Dict[str, Any]]:
    """Per-rank measurements of the §5.7.1 workload's LLM step.

    For every task rank: the classify step's cache key, its prompt
    token count (template-size accounting, chars/4 -- the same
    accounting the harness's DeterministicMockBackend charges), and
    the mock reply's token count.  The prompt and reply are pure
    functions of the rank, so these are the exact values the mock
    backend measures on every run.
    """
    table: Dict[int, Dict[str, Any]] = {}
    for rank in range(population):
        corpus = pct.task_corpus(rank)
        signatures = pct.incident_signatures(corpus.splitlines())
        prompt = CLASSIFY_PROMPT.format(signature_lines="\n".join(
            f"{n}: {sig}" for n, sig in enumerate(signatures)))
        table[rank] = {
            "cache_key": pct.cache_key_id(signatures),
            "prompt_tokens": pct._account_tokens(prompt),
            "mock_reply_tokens": pct._account_tokens(MOCK_REPLY(prompt)),
            "signature_count": len(signatures),
        }
    return table


# ---------------------------------------------------------------------------
# One (zipf, seed, mix) cell
# ---------------------------------------------------------------------------

def run_cell(*, zipf_s: float, seed: int, n_draws: int, population: int,
             rank_table: Dict[int, Dict[str, Any]], mix_id: str,
             price_in_per_mt: float, price_out_per_mt: float,
             cache_read_rate: float) -> Dict[str, Any]:
    """One cell: draw the workload, price it under the three modes.

    Token counts come from the rank table (measured prompt tokens;
    measured mock replies or the MODELED constant completion length);
    ``compute_cost`` -- the §5.7.1 overlay itself -- does the pricing,
    so the three modes are byte-for-byte the published method.
    """
    completion = completion_tokens_for_mix(mix_id)
    draws = zipf_draws(seed, n_draws, population, zipf_s)
    draw_keys: List[str] = []
    per_key: Dict[str, Tuple[int, int]] = {}
    for rank in draws:
        entry = rank_table[rank]
        key = entry["cache_key"]
        draw_keys.append(key)
        if key not in per_key:
            tout = (entry["mock_reply_tokens"] if completion is None
                    else completion)
            per_key[key] = (entry["prompt_tokens"], tout)
    tokens = {"draw_keys": draw_keys, "per_key_tokens": per_key}

    modes = {
        name: compute_cost(
            tokens, name, price_in_per_mt=price_in_per_mt,
            price_out_per_mt=price_out_per_mt,
            cache_read_rate=cache_read_rate)
        for name in COST_MODES
    }
    no_opt = modes["no_opt"]
    agentisa = modes["agentisa_cache"]
    provider = modes["provider_prompt_cache"]

    saving_ag = (1.0 - agentisa["cost_usd"] / no_opt["cost_usd"]) * 100.0
    saving_pr = (1.0 - provider["cost_usd"] / no_opt["cost_usd"]) * 100.0
    saving_ag = round(saving_ag, 4)
    saving_pr = round(saving_pr, 4)
    ratio = round(saving_pr / saving_ag, 4) if saving_ag > 0 else None

    out_cost = no_opt["output_tokens"] * price_out_per_mt
    in_cost = no_opt["input_tokens_full_price"] * price_in_per_mt
    calls = no_opt["calls"] or 1

    return {
        "zipf_s": zipf_s,
        "seed": seed,
        "mix_id": mix_id,
        "mix_kind": mix_kind(mix_id),
        "completion_tokens_per_call_modeled": completion,
        "draws": n_draws,
        "distinct_keys": len(per_key),
        "repeat_draws": no_opt["calls"] - agentisa["calls"],
        "repeat_input_token_mass": provider["input_tokens_cache_read"],
        "modes": modes,
        "savings": {
            "agentisa_cache_pct": saving_ag,
            "provider_prompt_cache_pct": saving_pr,
        },
        "gap_pct_points": round(saving_ag - saving_pr, 4),
        "ratio_prompt_to_agentisa": ratio,
        "mix_coordinates": {
            "prompt_tokens_per_call": round(
                no_opt["input_tokens_full_price"] / calls, 2),
            "completion_tokens_per_call": round(
                no_opt["output_tokens"] / calls, 2),
            "output_input_token_ratio": round(
                no_opt["output_tokens"]
                / max(1, no_opt["input_tokens_full_price"]), 4),
            "output_cost_share": round(
                out_cost / (out_cost + in_cost), 4),
        },
        "label": "MODELED-on-measured",
    }


# ---------------------------------------------------------------------------
# Aggregation (script-computed from the per-cell records)
# ---------------------------------------------------------------------------

AGGREGATED_FIELDS: List[Tuple[str, Tuple[str, ...]]] = [
    ("savings_agentisa_cache_pct", ("savings", "agentisa_cache_pct")),
    ("savings_provider_prompt_cache_pct",
     ("savings", "provider_prompt_cache_pct")),
    ("gap_pct_points", ("gap_pct_points",)),
    ("ratio_prompt_to_agentisa", ("ratio_prompt_to_agentisa",)),
    ("cost_usd_no_opt", ("modes", "no_opt", "cost_usd")),
    ("cost_usd_agentisa_cache", ("modes", "agentisa_cache", "cost_usd")),
    ("cost_usd_provider_prompt_cache",
     ("modes", "provider_prompt_cache", "cost_usd")),
    ("distinct_keys", ("distinct_keys",)),
    ("repeat_draws", ("repeat_draws",)),
    ("prompt_tokens_per_call", ("mix_coordinates",
                                "prompt_tokens_per_call")),
    ("completion_tokens_per_call", ("mix_coordinates",
                                    "completion_tokens_per_call")),
    ("output_input_token_ratio", ("mix_coordinates",
                                 "output_input_token_ratio")),
    ("output_cost_share", ("mix_coordinates", "output_cost_share")),
]


def _cell_field(cell: Dict[str, Any], path: Tuple[str, ...]) -> Any:
    value: Any = cell
    for step in path:
        value = value[step]
    return value


def aggregate_cells(cells: List[Dict[str, Any]]
                     ) -> List[Dict[str, Any]]:
    """Mean/stdev per (zipf, mix) over the cell's seeds.

    Every number here is computed from the per-cell records with
    ``statistics``; nothing is hand-filled.
    """
    grouped: Dict[Tuple[float, str], List[Dict[str, Any]]] = {}
    for cell in cells:
        grouped.setdefault((cell["zipf_s"], cell["mix_id"]), []).append(cell)

    aggregates: List[Dict[str, Any]] = []
    for (zipf_s, mix_id), group in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        metrics: Dict[str, Dict[str, float]] = {}
        for field, path in AGGREGATED_FIELDS:
            values = [
                float(_cell_field(cell, path))
                for cell in group if _cell_field(cell, path) is not None
            ]
            if not values:
                continue
            mean = statistics.mean(values)
            stdev = statistics.stdev(values) if len(values) > 1 else 0.0
            metrics[field] = {
                "mean": round(mean, 4),
                "stdev": round(stdev, 4),
            }
        aggregates.append({
            "zipf_s": zipf_s,
            "mix_id": mix_id,
            "mix_kind": group[0]["mix_kind"],
            "n_seeds": len(group),
            "metrics": metrics,
            "label": "MODELED (aggregated over seeds)",
        })
    return aggregates


# ---------------------------------------------------------------------------
# Curve analysis: crossovers of the token-mix function
# ---------------------------------------------------------------------------

def interpolate_crossing(curve: List[Tuple[float, float]],
                         target: float) -> Optional[Dict[str, Any]]:
    """Linear x where ``curve`` crosses ``target`` between grid points.

    ``curve`` is a list of ``(x, y)`` sorted by ascending x.  Returns
    ``None`` when the curve never brackets ``target``.
    """
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if (y0 - target) * (y1 - target) < 0:
            t = (target - y0) / (y1 - y0)
            return {
                "x": x0 + t * (x1 - x0),
                "bracket": [x0, x1],
                "y_before": y0,
                "y_after": y1,
            }
        if y1 == target:
            return {
                "x": x1,
                "bracket": [x0, x1],
                "y_before": y0,
                "y_after": y1,
            }
    return None


def _interp_value_at(curve: List[Tuple[float, float]],
                     x: float) -> Optional[float]:
    """Linear interpolation of y at ``x`` on an ascending-x curve."""
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return y0
            t = (x - x0) / (x1 - x0)
            return y0 + t * (y1 - y0)
    return None


# ---------------------------------------------------------------------------
# Published-anchor checks (read from the repo's own artifacts)
# ---------------------------------------------------------------------------

def _read_published(rel_path: str) -> Optional[Dict[str, Any]]:
    path = REPO_ROOT / rel_path
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_published_mock_anchor(
        cells: List[Dict[str, Any]], cfg: Dict[str, Any]
        ) -> Dict[str, Any]:
    """Compare the seed-``seed_base`` mock cells with the published run."""
    check: Dict[str, Any] = {
        "source": PUBLISHED_MOCK_REL,
        "role": ("the published mock paper-scale anchor (200 tasks x 1 "
                 "seed x 3 Zipf levels); this sweep's seed-"
                 f"{cfg['seed_base']} '{MOCK_MIX_ID}' cells must "
                 "reproduce its three pricing modes exactly"),
    }
    published = _read_published(PUBLISHED_MOCK_REL)
    if published is None:
        check["present"] = False
        check["all_match"] = None
        return check
    check["present"] = True
    pub_cells = {c["zipf_s"]: c for c in published.get("data", [])}
    by_key = {(c["zipf_s"], c["mix_id"], c["seed"]): c for c in cells}
    per_level: Dict[str, Any] = {}
    all_match = True
    for zipf_s in cfg["zipf_levels"]:
        pub = pub_cells.get(zipf_s)
        mine = by_key.get((zipf_s, MOCK_MIX_ID, cfg["seed_base"]))
        if pub is None or mine is None:
            per_level[str(zipf_s)] = {"compared": False}
            all_match = False
            continue
        level_match = True
        for mode in COST_MODES:
            got, want = mine["modes"][mode], (
                pub["priced_comparison"]["modes"][mode])
            for field in ("calls", "input_tokens_full_price",
                          "input_tokens_cache_read", "output_tokens",
                          "cost_usd"):
                if got[field] != want[field]:
                    level_match = False
        per_level[str(zipf_s)] = {"compared": True,
                                  "all_modes_match": level_match}
        all_match = all_match and level_match
    check["per_zipf_level"] = per_level
    check["all_match"] = all_match
    return check


def quote_published_real_anchor() -> Dict[str, Any]:
    """Quote the published real-gateway priced run (not recomputed)."""
    check: Dict[str, Any] = {
        "source": PUBLISHED_REAL_REL,
        "role": ("the published completion-dominated anchor (real "
                 "gateway, s=1.0, 50 realizations, max_tokens=512); its "
                 "numbers are quoted from the artifact, not recomputed"),
    }
    published = _read_published(PUBLISHED_REAL_REL)
    if published is None:
        check["present"] = False
        return check
    data = published.get("data") or []
    cell = next((c for c in data if c.get("zipf_s") == 1.0), None)
    if cell is None:
        check["present"] = False
        return check
    modes = cell["priced_comparison"]["modes"]
    no_opt = modes["no_opt"]
    out_cost = no_opt["output_tokens"] * DEFAULT_PRICE_OUT_PER_MT
    in_cost = no_opt["input_tokens_full_price"] * DEFAULT_PRICE_IN_PER_MT
    check.update({
        "present": True,
        "label": "MODELED-on-measured (published)",
        "savings_agentisa_cache_pct": round(
            (1 - modes["agentisa_cache"]["cost_usd"]
             / no_opt["cost_usd"]) * 100, 4),
        "savings_provider_prompt_cache_pct": round(
            (1 - modes["provider_prompt_cache"]["cost_usd"]
             / no_opt["cost_usd"]) * 100, 4),
        "mix_coordinates": {
            "prompt_tokens_per_call": round(
                no_opt["input_tokens_full_price"] / no_opt["calls"], 2),
            "completion_tokens_per_call": round(
                no_opt["output_tokens"] / no_opt["calls"], 2),
            "output_input_token_ratio": round(
                no_opt["output_tokens"]
                / max(1, no_opt["input_tokens_full_price"]), 4),
            "output_cost_share": round(out_cost / (out_cost + in_cost), 4),
        },
    })
    return check


# ---------------------------------------------------------------------------
# Analysis: the token-mix function and the validity interval
# ---------------------------------------------------------------------------

def build_analysis(cells: List[Dict[str, Any]],
                   aggregates: List[Dict[str, Any]],
                   cfg: Dict[str, Any]) -> Dict[str, Any]:
    analysis: Dict[str, Any] = {}

    # (1) The ordinal ranking is invariant under this overlay; what
    # varies with the mix is the margin.  Both facts are checked.
    flips = [c for c in cells if c["modes"]["agentisa_cache"]["cost_usd"]
             > c["modes"]["provider_prompt_cache"]["cost_usd"]]
    gaps_usd = [c["modes"]["provider_prompt_cache"]["cost_usd"]
                - c["modes"]["agentisa_cache"]["cost_usd"] for c in cells]
    analysis["ordering_invariance"] = {
        "cells_checked": len(cells),
        "flipped_cells": len(flips),
        "agentisa_cheaper_or_equal_cells": len(cells) - len(flips),
        "min_gap_usd": round(min(gaps_usd), 8) if gaps_usd else None,
        "min_gap_pct_points": round(
            min(c["gap_pct_points"] for c in cells), 4) if cells else None,
        "note": (
            "Under the Section 5.7.1 overlay the cost ordering "
            "agentisa_cache <= provider_prompt_cache <= no_opt is "
            "invariant across the entire swept grid: the result cache "
            "removes whole calls (repeat completions included) while "
            "the prompt-cache discount never touches completions.  The "
            "'ranking is a function of the token mix' claim is "
            "therefore about the MARGIN (how close the provider prompt "
            "cache comes to the result cache), quantified by the "
            "crossovers below."
        ),
        "label": "MODELED",
    }

    # (2) Per-skew mean curve over the mix axis + crossovers.
    per_skew: List[Dict[str, Any]] = []
    for zipf_s in cfg["zipf_levels"]:
        aggs = [a for a in aggregates if a["zipf_s"] == zipf_s]
        curve: List[Dict[str, Any]] = []
        for agg in aggs:
            m = agg["metrics"]
            curve.append({
                "mix_id": agg["mix_id"],
                "mix_kind": agg["mix_kind"],
                "x_completion_tokens_per_call":
                    m["completion_tokens_per_call"]["mean"],
                "savings_agentisa_cache_pct":
                    m["savings_agentisa_cache_pct"]["mean"],
                "savings_provider_prompt_cache_pct":
                    m["savings_provider_prompt_cache_pct"]["mean"],
                "gap_pct_points": m["gap_pct_points"]["mean"],
                "ratio_prompt_to_agentisa":
                    m["ratio_prompt_to_agentisa"]["mean"],
                "output_cost_share": m["output_cost_share"]["mean"],
                "output_cost_share_stdev":
                    m["output_cost_share"]["stdev"],
            })
        curve.sort(key=lambda p: p["x_completion_tokens_per_call"])
        ratio_curve = [(p["x_completion_tokens_per_call"],
                        p["ratio_prompt_to_agentisa"]) for p in curve]
        saving_curve = [(p["x_completion_tokens_per_call"],
                         p["savings_provider_prompt_cache_pct"])
                        for p in curve]
        share_curve = [(p["x_completion_tokens_per_call"],
                        p["output_cost_share"]) for p in curve]

        ratio_cross = interpolate_crossing(
            ratio_curve, RATIO_CROSSING_TARGET)
        saving_cross = interpolate_crossing(
            saving_curve, PROVIDER_SAVING_CROSSING_PCT)
        share_cross = interpolate_crossing(
            share_curve, OUTPUT_COST_SHARE_CROSSING)
        if ratio_cross is not None:
            ratio_cross["output_cost_share_at_crossing"] = _interp_value_at(
                share_curve, ratio_cross["x"])
            ratio_cross["definition"] = (
                "completion tokens per call at which the provider "
                "prompt cache's modeled saving equals half the result "
                "cache's (ratio crosses "
                f"{RATIO_CROSSING_TARGET})")
        if saving_cross is not None:
            saving_cross["definition"] = (
                "completion tokens per call beyond which the provider "
                "prompt cache's modeled saving is below "
                f"{PROVIDER_SAVING_CROSSING_PCT}%")
        if share_cross is not None:
            share_cross["definition"] = (
                "completion tokens per call at which completions reach "
                "half the no-optimization modeled cost (the cost-parity "
                "token-mix boundary)")

        per_skew.append({
            "zipf_s": zipf_s,
            "curve": curve,
            "crossovers": {
                "ratio_0_5": ratio_cross,
                "provider_saving_5pct": saving_cross,
                "output_cost_share_50pct": share_cross,
            },
            "regimes": {
                "prompt_cache_at_least_half_as_effective": [
                    p["mix_id"] for p in curve
                    if p["ratio_prompt_to_agentisa"]
                    >= RATIO_CROSSING_TARGET],
                "prompt_cache_less_than_half": [
                    p["mix_id"] for p in curve
                    if p["ratio_prompt_to_agentisa"]
                    < RATIO_CROSSING_TARGET],
                "provider_prompt_cache_saving_below_5pct": [
                    p["mix_id"] for p in curve
                    if p["savings_provider_prompt_cache_pct"]
                    < PROVIDER_SAVING_CROSSING_PCT],
            },
            "label": "MODELED",
        })
    analysis["per_skew"] = per_skew

    # (3) The completion- vs input-dominated contrast at the anchors.
    contrast: List[Dict[str, Any]] = []
    for skew in per_skew:
        by_mix = {p["mix_id"]: p for p in skew["curve"]}
        in_anchor = by_mix.get(ANCHORS["input_dominated_extreme"])
        out_anchor = by_mix.get(ANCHORS["completion_dominated"])
        entry: Dict[str, Any] = {
            "zipf_s": skew["zipf_s"],
            "input_dominated_anchor": in_anchor,
            "completion_dominated_anchor": out_anchor,
            "label": "MODELED",
        }
        if in_anchor and out_anchor:
            entry["ratio_drop_factor"] = round(
                in_anchor["ratio_prompt_to_agentisa"]
                / out_anchor["ratio_prompt_to_agentisa"], 4)
        contrast.append(entry)
    analysis["regime_contrast"] = contrast

    # (4) Published-anchor checks.
    analysis["published_anchor_checks"] = {
        "mock_paper_scale": check_published_mock_anchor(cells, cfg),
        "real_gateway_run": quote_published_real_anchor(),
    }

    # (5) The L512 sweep point next to the published real-gateway mix.
    l512 = [a for a in aggregates if a["mix_id"]
            == ANCHORS["published_real_gateway_mix_analog"]]
    if l512:
        analysis["real_mix_analog"] = {
            "mix_id": ANCHORS["published_real_gateway_mix_analog"],
            "per_zipf": [
                {
                    "zipf_s": a["zipf_s"],
                    "output_input_token_ratio": a["metrics"][
                        "output_input_token_ratio"]["mean"],
                    "output_cost_share": a["metrics"][
                        "output_cost_share"]["mean"],
                }
                for a in l512
            ],
            "note": (
                "the sweep's L512 point sits at the Section 5.7.1 "
                "real-gateway run's max_tokens level; compare its mix "
                "coordinates with published_anchor_checks."
                "real_gateway_run.mix_coordinates"),
        }

    # (6) Headline strings assembled from the computed numbers only.
    lines: List[str] = []
    inv = analysis["ordering_invariance"]
    lines.append(
        f"ordering invariant in {inv['cells_checked']}/"
        f"{inv['cells_checked']} cells (agentisa_cache cheaper than "
        f"provider_prompt_cache; min gap {inv['min_gap_usd']:.6f} USD)")
    for skew in per_skew:
        cross = skew["crossovers"]["ratio_0_5"]
        if cross is not None:
            share = cross.get("output_cost_share_at_crossing")
            share_txt = (f", output cost share {share:.1%}"
                         if share is not None else "")
            lines.append(
                f"s={skew['zipf_s']}: provider prompt-cache effectiveness "
                f"(saving as fraction of the result cache's) crosses "
                f"{RATIO_CROSSING_TARGET} at "
                f"{cross['x']:.1f} completion tokens/call{share_txt}")
        neg = skew["crossovers"]["provider_saving_5pct"]
        if neg is not None:
            lines.append(
                f"s={skew['zipf_s']}: provider prompt-cache saving falls "
                f"below {PROVIDER_SAVING_CROSSING_PCT}% beyond "
                f"{neg['x']:.1f} completion tokens/call")
    analysis["headline"] = lines
    return analysis


# ---------------------------------------------------------------------------
# Provenance / coverage / document assembly
# ---------------------------------------------------------------------------

def _git_rev() -> Optional[str]:
    """Short HEAD rev via the one read-only git command."""
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=str(REPO_ROOT))
    except Exception:  # noqa: BLE001  (provenance must not crash the run)
        return None
    if proc.returncode != 0:
        return None
    rev = proc.stdout.strip()
    return rev or None


def _sha256_of_file(rel_path: str) -> Optional[str]:
    try:
        import hashlib
        return hashlib.sha256(
            (REPO_ROOT / rel_path).read_bytes()).hexdigest()
    except OSError:
        return None


def build_provenance(cfg: Dict[str, Any],
                     n_cells: int) -> Dict[str, Any]:
    seeds = [cfg["seed_base"] + i for i in range(cfg["n_seeds"])]
    return {
        "git_rev": _git_rev(),
        "git_rev_command": "git rev-parse --short HEAD (read-only)",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "scale": {
            "cells": n_cells,
            "draws_per_cell": cfg["n_draws"],
            "task_population": cfg["population"],
            "zipf_levels": list(cfg["zipf_levels"]),
            "mix_points": len(cfg["mix_ids"]),
            "total_draws": n_cells * cfg["n_draws"],
            "seeds_per_cell": cfg["n_seeds"],
        },
        "script": SCRIPT_REL,
        "dependency": {
            "modules": ["examples/paper_real_llm_compiler_test.py"],
            "module_sha256": {
                "examples/paper_real_llm_compiler_test.py":
                    _sha256_of_file(
                        "examples/paper_real_llm_compiler_test.py"),
            },
            "usage": (
                "compute_cost/COST_MODES (the Section 5.7.1 MODELED "
                "pricing overlay) plus the workload primitives "
                "(task_corpus, incident_signatures, cache_key_id, "
                "zipf_draws, CLASSIFY_PROMPT, the mock reply, and the "
                "chars/4 token accounting) -- imported, never copied"),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "determinism": (
            "pure offline arithmetic over the seeded workload; "
            "identical (seeds, draws, population, mix grid, prices) "
            "reproduce identical data (the timestamp and git rev live "
            "in this provenance block only)"),
    }


def build_coverage(cfg: Dict[str, Any], n_cells: int,
                   anchor_check: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "sweep_cells_planned": n_cells,
        "sweep_cells_completed": n_cells,
        "blocked": [],
        "dependency_components": [
            {
                "component": "examples/paper_real_llm_compiler_test.py",
                "job": "pre-existing §5.7.1 harness (not an E-job "
                       "deliverable)",
                "status": "shipped",
                "evidence": (
                    "anchor reproduction "
                    f"all_match={anchor_check.get('all_match')} "
                    "against " + PUBLISHED_MOCK_REL),
            },
        ],
        "limitations": [
            "mock only: no live gateway, no gateway-reported tokens in "
            "this sweep (the published real-gateway anchor is quoted "
            "from its artifact for context)",
            "the swept L* completion lengths are MODELED constants "
            "(the alias's observed reply-binds-at-max_tokens behavior), "
            "not generated text; classification semantics are out of "
            "scope, as in the published protocol",
            "single workload shape: the Section 5.7.1 seven-step triage "
            "stream with one cacheable LLM-type step",
            "the overlay's documented simplifications stand: no 1.25x "
            "cache-write premium and no 1024-token minimum cacheable "
            "prefix (both favor the provider-cache baseline)",
            "GPTCache / approximate caching explicitly out of scope; "
            "no external-cache head-to-head is claimed",
        ],
    }


def build_method() -> Dict[str, Any]:
    return {
        "basis": (
            "Section 5.7.1 priced comparison (docs/AgentISA.md, 'A "
            "priced comparison') and the Appendix G protocol "
            "(docs/AgentISA_supplementary.md): the same three pricing "
            "modes, the same default prices ($3/M input, $15/M output), "
            "the same 0.1x prompt-cache read discount"),
        "modes": {
            "no_opt": "every draw re-queries at full price",
            "agentisa_cache": (
                "only first occurrences are paid (the measured behavior "
                "of the Section 5.7.1 arms)"),
            "provider_prompt_cache": (
                "MODELED: every draw queries; repeat prompts bill input "
                "tokens as prompt-cache reads at cache_read_rate x the "
                "input price; completions always bill at full price"),
        },
        "labels": (
            "ALL cost, saving, gap, and ratio numbers in this document "
            "are MODELED-on-measured: prompt tokens are template-size "
            "accounted measurements (chars/4) of the exact prompt "
            "texts; the L* completion lengths are MODELED constants; "
            "prices are parameterized list-price stand-ins.  No "
            "dollar-savings claim is made."),
        "token_mix_axis": (
            "completion tokens per call: 'mock_measured' uses the "
            "mock's template-accounted replies (the published mock "
            "paper-scale anchor); 'L<n>' models the alias's observed "
            "reply-binds-at-max_tokens behavior as a constant n tokens "
            "per call (L64 = the Section 5.7 ceiling, L512 = the "
            "Section 5.7.1 ceiling).  The grid spans the "
            "input-dominated extreme through the completion-dominated "
            "regime; realized mixes are in each cell's "
            "mix_coordinates"),
        "simplifications": (
            "inherits the overlay's documented simplifications: no "
            "1.25x cache-write premium and no 1024-token minimum "
            "cacheable prefix (both favor the provider-cache baseline); "
            "list prices are stand-ins because the gateway's actual "
            "pricing is unknown"),
        "workload": (
            "the Section 5.7.1 seven-step service-log triage shape: "
            "one cacheable LLM-type step (classify_signature), one "
            "call per distinct incident-signature batch; seeded "
            "truncated-Zipf draws over the fixed 40-task universe"),
    }


def _resolve_config(config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "n_seeds": DEFAULT_N_SEEDS,
        "seed_base": DEFAULT_SEED_BASE,
        "n_draws": DEFAULT_DRAWS,
        "population": DEFAULT_POPULATION,
        "zipf_levels": list(DEFAULT_ZIPF_LEVELS),
        "mix_ids": default_mix_ids(),
        "price_in_per_mt": DEFAULT_PRICE_IN_PER_MT,
        "price_out_per_mt": DEFAULT_PRICE_OUT_PER_MT,
        "cache_read_rate": DEFAULT_CACHE_READ_RATE,
    }
    if config:
        cfg.update(config)
    if cfg["n_seeds"] < 1:
        raise ValueError("n_seeds must be >= 1")
    if cfg["n_draws"] < 2:
        raise ValueError("n_draws must be >= 2")
    if cfg["population"] < 2:
        raise ValueError("population must be >= 2")
    if not cfg["mix_ids"]:
        raise ValueError("mix_ids must be non-empty")
    if cfg["price_in_per_mt"] <= 0 or cfg["price_out_per_mt"] <= 0:
        raise ValueError("prices must be positive")
    if not 0.0 < cfg["cache_read_rate"] < 1.0:
        raise ValueError("cache_read_rate must be in (0, 1)")
    # validate mix ids eagerly
    for mix_id in cfg["mix_ids"]:
        completion_tokens_for_mix(mix_id)
    return cfg


def run_sweep(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Run the full sweep and return the results document."""
    cfg = _resolve_config(config)
    rank_table = precompute_rank_table(cfg["population"])

    cells: List[Dict[str, Any]] = []
    for zipf_s in cfg["zipf_levels"]:
        for seed_index in range(cfg["n_seeds"]):
            seed = cfg["seed_base"] + seed_index
            for mix_id in cfg["mix_ids"]:
                cell = run_cell(
                    zipf_s=zipf_s, seed=seed, n_draws=cfg["n_draws"],
                    population=cfg["population"],
                    rank_table=rank_table, mix_id=mix_id,
                    price_in_per_mt=cfg["price_in_per_mt"],
                    price_out_per_mt=cfg["price_out_per_mt"],
                    cache_read_rate=cfg["cache_read_rate"])
                cell["seed_index"] = seed_index
                cells.append(cell)

    aggregates = aggregate_cells(cells)
    analysis = build_analysis(cells, aggregates, cfg)
    provenance = build_provenance(cfg, len(cells))
    coverage = build_coverage(
        cfg, len(cells),
        analysis["published_anchor_checks"]["mock_paper_scale"])

    mix_axis = []
    for mix_id in cfg["mix_ids"]:
        mix_axis.append({
            "mix_id": mix_id,
            "mix_kind": mix_kind(mix_id),
            "completion_tokens_per_call_modeled":
                completion_tokens_for_mix(mix_id),
        })

    return {
        "experiment": (
            "E9-promptcache-sweep: token-mix sweep of the MODELED "
            "priced comparison (completion-dominated vs "
            "input-dominated x 3 skew, mock, Section 5.7.1 method)"),
        "harness": SCRIPT_REL,
        "method": build_method(),
        "config": {
            **{k: v for k, v in cfg.items()},
            "anchors": dict(ANCHORS),
            "mix_axis": mix_axis,
            "crossing_targets": {
                "ratio_prompt_to_agentisa": RATIO_CROSSING_TARGET,
                "provider_saving_pct": PROVIDER_SAVING_CROSSING_PCT,
                "output_cost_share": OUTPUT_COST_SHARE_CROSSING,
            },
        },
        "provenance": provenance,
        "coverage": coverage,
        "data": {"per_cell": cells, "aggregates": aggregates},
        "analysis": analysis,
        "notes": [
            "all cost and saving figures are MODELED-on-measured per "
            "the Section 5.7.1 / Appendix G protocol; token counts for "
            "prompts and for the 'mock_measured' replies are "
            "template-accounted measurements, L* completions are "
            "modeled constants",
            "the ordinal cost ranking of the two caching strategies is "
            "invariant under this overlay (see "
            "analysis.ordering_invariance); what the token mix changes "
            "is the margin, quantified by the per-skew crossovers",
            "aggregates (mean/stdev over the "
            f"{cfg['n_seeds']} seeds) are computed by "
            "this script from the per-cell records; no number in this "
            "document is hand-entered",
            "the published real-gateway anchor is quoted from its "
            "artifact for context; this sweep itself is mock-only",
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Token-mix sweep of the MODELED priced comparison "
                    "(Section 5.7.1 method, mock workload, offline).")
    parser.add_argument("--output", default=str(REPO_ROOT / OUTPUT_REL),
                        help="results JSON path (default: " + OUTPUT_REL
                             + ")")
    parser.add_argument("--seeds", type=int, default=DEFAULT_N_SEEDS,
                        help="seeds per (skew, mix) cell (default 5)")
    parser.add_argument("--seed-base", type=int, default=DEFAULT_SEED_BASE,
                        help="first workload seed (default 1000)")
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS,
                        help="draws per cell (default 200, paper scale)")
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION,
                        help="task universe size (default 40)")
    parser.add_argument("--zipf", nargs="+", type=float,
                        default=list(DEFAULT_ZIPF_LEVELS),
                        help="Zipf skew levels (default 0.5 1.0 1.5)")
    parser.add_argument("--mix", action="append", default=None,
                        help="mix id to include (repeatable); default: "
                             "mock_measured + the L* grid")
    parser.add_argument("--price-in", type=float,
                        default=DEFAULT_PRICE_IN_PER_MT,
                        help="input list price per M tokens (default 3.0)")
    parser.add_argument("--price-out", type=float,
                        default=DEFAULT_PRICE_OUT_PER_MT,
                        help="output list price per M tokens (default 15.0)")
    parser.add_argument("--cache-read-rate", type=float,
                        default=DEFAULT_CACHE_READ_RATE,
                        help="MODELED prompt-cache read discount on input "
                             "tokens (default 0.1)")
    parser.add_argument("--quiet", action="store_true",
                        help="suppress the human-readable summary")
    args = parser.parse_args(argv)

    config = {
        "n_seeds": args.seeds,
        "seed_base": args.seed_base,
        "n_draws": args.draws,
        "population": args.population,
        "zipf_levels": list(args.zipf),
        "mix_ids": args.mix if args.mix else default_mix_ids(),
        "price_in_per_mt": args.price_in,
        "price_out_per_mt": args.price_out,
        "cache_read_rate": args.cache_read_rate,
    }
    doc = run_sweep(config)

    out_path = Path(args.output)
    out_path.write_text(json.dumps(doc, indent=2) + "\n")
    if not args.quiet:
        prov = doc["provenance"]
        print(f"git rev {prov['git_rev']}, {prov['timestamp_utc']}")
        print(f"cells: {doc['coverage']['sweep_cells_completed']} "
              f"({doc['config']['n_seeds']} seeds x "
              f"{len(doc['config']['zipf_levels'])} skews x "
              f"{len(doc['config']['mix_ids'])} mixes x "
              f"{doc['config']['n_draws']} draws)")
        anchor = doc["analysis"]["published_anchor_checks"][
            "mock_paper_scale"]
        print(f"published mock anchor all_match: {anchor['all_match']}")
        for line in doc["analysis"]["headline"]:
            print(f"  {line}")
        print(f"results written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
