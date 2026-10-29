"""Cost model over committed results: measured unit prices, modeled savings.

Every avoided *unit* in this file is a count read from a committed results file.
Every second, token, and dollar figure is a **model**: the measured per-call unit
price multiplied by that count. Nothing here is an observation of a saving, and
the emitted JSON says so in its ``result_kind``/``statement`` fields.

Inputs (all committed; nothing computed from live services, no clock, no RNG):

  * ``isa_real_llm_results.json`` -- the real-gateway run (Section 5.7). Supplies
    the measured unit prices: per-call wall clock (``wall_ms / api_calls``), the
    p50/p95 latency band, and gateway-reported input/output tokens per call.
  * ``real_trace_pilot_tau_bench_results.json`` -- Section 5.12 cache scopes.
    F1's avoided calls are ``cache_scopes.first_trial.within_run_hits`` (746),
    over the 1,789 cacheable first-trial calls. Using the corpus-wide 14,285 as
    the denominator would inflate F1 by 8x and is rejected by a check below.
  * ``instruction_precipitation_results.json`` -- Appendix I.7. F2's served steps
    are ``interleaved.served`` (2,245 of 2,340 steps).
  * ``compound_binding_results.json`` -- Appendix I.9. F3's absorbed dispatches
    are ``reduction_if_substituted.steps_removed`` (285 of 2,275 calls).

Price assumption (explicit, and *not* measured): $3 per million input tokens and
$15 per million output tokens, the stand-in list prices of supplementary
Appendix G, adopted there "because the gateway's actual pricing is unknown".
They can be 50% wrong in either direction; the sensitivity band therefore
multiplies them by [0.5, 1.5] and the measured token counts by [0.7, 1.3]
(ceiling-censored at ``max_tokens``, plus tokenizer/mix drift).

Three honest limits are encoded, not just documented:

  1. F2 and F3 count *dispatches*, not model calls. Converting them into avoided
     planner turns requires a planner-bypass assumption that the evaluation never
     instruments; those rows are marked ``assumption_conditioned`` and are zero
     without it.
  2. The three corpora overlap (three readings of the same tau-bench
     recordings), so no total is emitted: ``aggregate.sum_allowed`` is false and
     only the largest single mechanism is reported.
  3. Output tokens are censored at the 64-token budget, so per-call dollars are a
     floor on the 64-regime and the 512-regime would be ~8x higher.

Usage:
    ./venv/bin/python examples/benchmarks/cost_model.py \
        --output examples/benchmarks/cost_model_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

SCHEMA_VERSION = "cost_model/v1"
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_UNIT_RESULTS = REPO_ROOT / "examples" / "benchmarks" / "isa_real_llm_results.json"
DEFAULT_CACHE_RESULTS = (
    REPO_ROOT / "examples" / "benchmarks" / "real_trace_pilot_tau_bench_results.json"
)
DEFAULT_TEMPLATE_RESULTS = (
    REPO_ROOT / "examples" / "benchmarks" / "instruction_precipitation_results.json"
)
DEFAULT_COMPOUND_RESULTS = (
    REPO_ROOT / "examples" / "benchmarks" / "compound_binding_results.json"
)

# Explicit price assumption (supplementary Appendix G stand-ins, not a quote).
PRICE_IN_PER_MTOK = 3.0
PRICE_OUT_PER_MTOK = 15.0
PRICE_BAND = (0.5, 1.5)
TOKEN_BAND = (0.7, 1.3)


# -- loading --------------------------------------------------------------


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_unit_prices(path: Path) -> Dict[str, Any]:
    """Measured per-call unit prices from the real-gateway run.

    Each Zipf arm contributes its own ``baseline`` block; the model uses the mean
    of the three arms, which is what an aggregate workload would pay on average.
    """
    payload = load_json(path)
    arms: List[Dict[str, Any]] = []
    for row in payload["data"]:
        baseline = row["per_seed"][0]["baseline"]
        arms.append(
            {
                "zipf_s": row["zipf_s"],
                "baseline_calls": baseline["api_calls"],
                "wall_ms": baseline["wall_ms"],
                "t_mean_s": baseline["wall_ms"] / baseline["api_calls"] / 1000.0,
                "t_p50_s": baseline["latency_ms_p50"] / 1000.0,
                "t_p95_s": baseline["latency_ms_p95"] / 1000.0,
                "input_tokens_per_call": baseline["input_tokens"] / baseline["api_calls"],
                "output_tokens_per_call": baseline["output_tokens"] / baseline["api_calls"],
            }
        )
    if not arms:
        raise ValueError(f"no real-gateway arms in {path}")

    n = len(arms)
    prices = {
        "t_mean_s": sum(a["t_mean_s"] for a in arms) / n,
        "t_p50_s": sum(a["t_p50_s"] for a in arms) / n,
        "t_p95_s": sum(a["t_p95_s"] for a in arms) / n,
        "input_tokens_per_call": sum(a["input_tokens_per_call"] for a in arms) / n,
        "output_tokens_per_call": sum(a["output_tokens_per_call"] for a in arms) / n,
    }
    prices["model"] = payload["config"]["model"]
    prices["max_tokens"] = payload["config"]["max_tokens"]
    prices["arms"] = arms
    prices["measured"] = [
        "t_mean_s",
        "t_p50_s",
        "t_p95_s",
        "input_tokens_per_call",
        "output_tokens_per_call",
    ]
    return prices


def load_effects(
    cache_path: Path, template_path: Path, compound_path: Path
) -> List[Dict[str, Any]]:
    """Measured avoided-unit counts, one row per mechanism.

    The counts are read verbatim; the rates are derived only for the
    denominator checks, so a reader can always recover what the count is a
    fraction of.
    """
    cache = load_json(cache_path)
    first = cache["cache_scopes"]["first_trial"]
    cache_hits = float(first["within_run_hits"])
    cache_calls = float(first["calls"])

    template = load_json(template_path)
    inter = template["interleaved"]
    served = float(inter["served"])
    steps = float(inter["steps"])

    compound = load_json(compound_path)
    absorbed = float(compound["reduction_if_substituted"]["steps_removed"])
    dispatch_calls = float(compound["reduction_if_substituted"]["steps"])

    return [
        {
            "id": "F1",
            "name": "result cache, cross-task hits",
            "mechanism": "cache",
            "unit_kind": "tool_execution",
            "convertible": False,
            "effect_units": cache_hits,
            "effect_units_are": (
                "recorded tool executions replayed from the cache; the model turn that "
                "emitted each call is still in the trace, so a hit is not an avoided model call"
            ),
            "effect_source": {
                "file": _rel(cache_path),
                "json_path": "cache_scopes.first_trial.within_run_hits",
                "denominator_json_path": "cache_scopes.first_trial.calls",
                "denominator": cache_calls,
                "rate": cache_hits / cache_calls,
                "scope": "cacheable first-trial calls, shared across the tasks of one run",
            },
            "assumption_conditioned": False,
            "conversion_assumption": (
                "none: a cache hit replays a recorded tool result and avoids a tool "
                "execution, not the model turn that emitted the call; no per-tool latency "
                "was measured and no model call is avoided, so no second, token or dollar "
                "figure is attributable from this corpus"
            ),
        },
        {
            "id": "F2",
            "name": "template precipitation, served steps",
            "mechanism": "template",
            "unit_kind": "step",
            "convertible": True,
            "effect_units": served,
            "effect_units_are": "measured template-served dispatches (a step count, not a call count)",
            "effect_source": {
                "file": _rel(template_path),
                "json_path": "interleaved.served",
                "denominator_json_path": "interleaved.steps",
                "denominator": steps,
                "rate": served / steps,
                "scope": "first-trial tau-bench tasks, template promoted by an earlier task",
            },
            "assumption_conditioned": True,
            "conversion_assumption": (
                "planner bypass: a template match substitutes for the planner turn "
                "that would have emitted the dispatch; that turn is never instrumented, "
                "so the seconds/tokens/dollars below are a conditional bound, not a saving"
            ),
        },
        {
            "id": "F3",
            "name": "compound absorption, dispatches",
            "mechanism": "compound",
            "unit_kind": "dispatch",
            "convertible": True,
            "effect_units": absorbed,
            "effect_units_are": "measured dispatches absorbed by greedy longest-match substitution",
            "effect_source": {
                "file": _rel(compound_path),
                "json_path": "reduction_if_substituted.steps_removed",
                "denominator_json_path": "reduction_if_substituted.steps",
                "denominator": dispatch_calls,
                "rate": absorbed / dispatch_calls,
                "scope": "observed-observation corpus; binder bound, not an end-to-end cost measurement",
            },
            "assumption_conditioned": True,
            "conversion_assumption": (
                "planner bypass (as F2), plus within-family generalization: the "
                "compounds were learned and verified on the same corpus"
            ),
        },
    ]


# -- arithmetic -----------------------------------------------------------


def unit_cost(price_in: float, price_out: float, input_tokens: float,
              output_tokens: float) -> float:
    """Dollars per avoided unit under a stated price assumption."""
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


def convert_effect(
    effect_units: float,
    prices: Dict[str, Any],
    price_in: float = PRICE_IN_PER_MTOK,
    price_out: float = PRICE_OUT_PER_MTOK,
) -> Dict[str, Any]:
    """Point estimate: avoid ``effect_units`` at the measured per-unit prices.

    Linear in ``effect_units`` by construction (multiply-then-sum, no offsets),
    which is what lets a caller scale a workload without re-measuring.
    """
    per_call = unit_cost(
        price_in,
        price_out,
        prices["input_tokens_per_call"],
        prices["output_tokens_per_call"],
    )
    return {
        "avoided_units": effect_units,
        "seconds": {
            "mean": effect_units * prices["t_mean_s"],
            "p50": effect_units * prices["t_p50_s"],
            "p95": effect_units * prices["t_p95_s"],
        },
        "input_tokens": effect_units * prices["input_tokens_per_call"],
        "output_tokens": effect_units * prices["output_tokens_per_call"],
        "dollars": effect_units * per_call,
        "unit_cost_dollars": per_call,
    }


def sensitivity(
    effect_units: float,
    prices: Dict[str, Any],
    price_in: float = PRICE_IN_PER_MTOK,
    price_out: float = PRICE_OUT_PER_MTOK,
) -> Dict[str, Any]:
    """Declared band: price x[0.5, 1.5], token counts x[0.7, 1.3].

    The four dollar corners are the cross product; the band is their min/max.
    Seconds are price- and token-invariant; their width comes from the latency
    axis (p50..p95). Band edges are declared bounds, not a confidence interval.
    """
    corners = []
    for price_mult in PRICE_BAND:
        for token_mult in TOKEN_BAND:
            per_call = (
                prices["input_tokens_per_call"] * token_mult * price_in * price_mult
                + prices["output_tokens_per_call"] * token_mult * price_out * price_mult
            ) / 1e6
            corners.append(
                {
                    "price_multiplier": price_mult,
                    "token_multiplier": token_mult,
                    "dollars": effect_units * per_call,
                }
            )
    dollar_values = [c["dollars"] for c in corners]
    return {
        "declared": {
            "price_multiplier_range": list(PRICE_BAND),
            "token_multiplier_range": list(TOKEN_BAND),
            "note": "declared input bounds, not a confidence interval",
        },
        "dollars": {
            "low": min(dollar_values),
            "high": max(dollar_values),
            "corners": corners,
        },
        "input_tokens": {
            "low": effect_units * prices["input_tokens_per_call"] * TOKEN_BAND[0],
            "high": effect_units * prices["input_tokens_per_call"] * TOKEN_BAND[1],
        },
        "output_tokens": {
            "low": effect_units * prices["output_tokens_per_call"] * TOKEN_BAND[0],
            "high": effect_units * prices["output_tokens_per_call"] * TOKEN_BAND[1],
        },
        "seconds": {
            "low": effect_units * prices["t_p50_s"],
            "high": effect_units * prices["t_p95_s"],
            "invariant_to_price": True,
            "invariant_to_token_count": True,
            "note": "width comes from the measured latency axis (p50..p95)",
        },
    }


def _rounded(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def _round_tree(node: Any) -> Any:
    if isinstance(node, dict):
        return {key: _round_tree(val) for key, val in node.items()}
    if isinstance(node, list):
        return [_round_tree(val) for val in node]
    if isinstance(node, float):
        return _rounded(node)
    return node


# -- assembly -------------------------------------------------------------


def _reject_inflated_denominator(effects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Record the valid count next to the one a denominator slip would produce."""
    f1 = next(e for e in effects if e["id"] == "F1")
    correct = f1["effect_units"]
    rate = f1["effect_source"]["rate"]
    return {
        "mechanism": "F1",
        "accepted_denominator": f1["effect_source"]["denominator"],
        "accepted_avoided_calls": correct,
        "rejected_denominator": 14285,
        "rejected_avoided_calls": rate * 14285,
        "why": (
            "14,285 is every call in the corpus; only the 1,789 cacheable first-trial "
            "calls can produce a hit, so the corpus-wide denominator inflates F1 by ~8x"
        ),
    }


def build_payload(
    unit_path: Path = DEFAULT_UNIT_RESULTS,
    cache_path: Path = DEFAULT_CACHE_RESULTS,
    template_path: Path = DEFAULT_TEMPLATE_RESULTS,
    compound_path: Path = DEFAULT_COMPOUND_RESULTS,
    price_in: float = PRICE_IN_PER_MTOK,
    price_out: float = PRICE_OUT_PER_MTOK,
) -> Dict[str, Any]:
    prices = load_unit_prices(unit_path)
    effects = load_effects(cache_path, template_path, compound_path)

    rows = []
    for effect in effects:
        if effect.get("convertible") is False:
            rows.append(
                {
                    **effect,
                    "point": None,
                    "not_priceable": (
                        "the avoided unit is a tool execution, not a model call; pricing it "
                        "at a measured per-model-call unit price would be a unit error"
                    ),
                    "sensitivity": {},
                }
            )
            continue
        point = convert_effect(effect["effect_units"], prices, price_in, price_out)
        point["avoided_calls"] = effect["effect_units"]
        point["avoided_calls_basis"] = (
            "measured model calls (cache hits)"
            if effect["unit_kind"] == "call"
            else f"conditional on planner bypass: {effect['unit_kind']}s counted, not measured calls"
        )
        band = sensitivity(effect["effect_units"], prices, price_in, price_out)
        rows.append(
            {
                **effect,
                "point": point,
                "sensitivity": band,
            }
        )

    unit_price = unit_cost(
        price_in, price_out, prices["input_tokens_per_call"], prices["output_tokens_per_call"]
    )
    priceable = [row for row in rows if row.get("point")]
    top = max(priceable, key=lambda row: row["point"]["dollars"])

    checks = [
        {
            "name": "f1_rate_matches_committed_rate",
            "value": rows[0]["effect_source"]["rate"],
            "expected": 0.41699,
            "tolerance": 1e-4,
        },
        {
            "name": "f2_rate_matches_committed_rate",
            "value": rows[1]["effect_source"]["rate"],
            "expected": 0.95940,
            "tolerance": 1e-4,
        },
        {
            "name": "f3_rate_matches_committed_rate",
            "value": rows[2]["effect_source"]["rate"],
            "expected": 0.12527,
            "tolerance": 1e-4,
        },
        {
            "name": "output_tokens_censored_at_budget",
            "value": prices["output_tokens_per_call"],
            "expected": float(prices["max_tokens"]),
            "tolerance": 1e-9,
            "note": "every call bound at max_tokens, so this is a floor on true output length",
        },
        {
            "name": "unit_price_arithmetic",
            "value": unit_price,
            "expected": (prices["input_tokens_per_call"] * price_in
                         + prices["output_tokens_per_call"] * price_out) / 1e6,
            "tolerance": 1e-12,
        },
        {
            "name": "f1_not_inflated_by_corpus_denominator",
            "value": rows[0]["effect_units"],
            "expected": 746.0,
            "tolerance": 1e-9,
            "note": "rejects 0.41699 x 14,285 = 5,956.8 avoided calls",
        },
    ]
    for check in checks:
        check["ok"] = abs(check["value"] - check["expected"]) <= check["tolerance"]

    payload = {
        "schema": SCHEMA_VERSION,
        "result_kind": "MODELED",
        "not_observed": True,
        "statement": (
            "MODELED, not observed: every avoided count below is measured in a "
            "committed results file, but every second, token and dollar figure is a "
            "model (measured per-call unit price x that count) under the stated price "
            "assumption. No saving was instrumented end to end; no production trace "
            "is involved."
        ),
        "price_assumption": {
            "input_dollars_per_mtok": price_in,
            "output_dollars_per_mtok": price_out,
            "provenance": "docs/AgentISA_supplementary.md, Appendix G (priced-comparison protocol)",
            "status": "assumed stand-in list prices, not a provider quote or invoice",
            "why": "the gateway's actual pricing is unknown, so dollars are a model end to end",
            "regime": {
                "model": prices["model"],
                "max_tokens": prices["max_tokens"],
                "censored": True,
                "note": "tokens are measured at this budget; output is a floor, and a 512-token regime would cost ~8x more per call",
            },
        },
        "units": {
            "unit_price_dollars_per_avoided_unit": unit_price,
            "seconds": "mean / p50 / p95 of measured per-call wall clock",
            "tokens": "gateway-reported input/output tokens per call",
        },
        "unit_prices": {
            "source": _rel(unit_path),
            "measured": prices["measured"],
            "t_mean_s": prices["t_mean_s"],
            "t_p50_s": prices["t_p50_s"],
            "t_p95_s": prices["t_p95_s"],
            "input_tokens_per_call": prices["input_tokens_per_call"],
            "output_tokens_per_call": prices["output_tokens_per_call"],
            "arms": prices["arms"],
        },
        "mechanisms": rows,
        "aggregate": {
            "sum_allowed": False,
            "why": (
                "the three rows are three readings of the same tau-bench recordings "
                "(14,285 calls / 2,340 steps / 2,275 calls) and F2/F3 count the same "
                "steps at different granularity; summing double counts"
            ),
            "rule": "report the largest single mechanism, never a sum or a total",
            "max_mechanism": top["id"],
            "max_mechanism_name": top["name"],
            "point": top["point"],
            "sensitivity": top["sensitivity"],
        },
        "denominator_guard": _reject_inflated_denominator(effects),
        "checks": checks,
        "limits": [
            "prices are assumed stand-ins; money is modeled, never measured",
            "F2/F3 require the unmeasured planner-bypass assumption; without it their seconds, tokens and dollars are zero",
            "output tokens are censored at max_tokens, so dollars are a floor at the 64-token budget",
            "no total is emitted because the corpora overlap; max single mechanism only",
            "sequential, single-gateway, mock-environment recordings: no throughput, concurrency or availability claim",
        ],
        "provenance": {
            "script": "examples/benchmarks/cost_model.py",
            "deterministic": True,
            "inputs": {
                _rel(path): {"sha256": file_digest(path)}
                for path in (unit_path, cache_path, template_path, compound_path)
            },
        },
    }
    return _round_tree(payload)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", default=str(REPO_ROOT / "examples" / "benchmarks"
                                                 / "cost_model_results.json"))
    parser.add_argument("--unit-results", default=str(DEFAULT_UNIT_RESULTS))
    parser.add_argument("--cache-results", default=str(DEFAULT_CACHE_RESULTS))
    parser.add_argument("--template-results", default=str(DEFAULT_TEMPLATE_RESULTS))
    parser.add_argument("--compound-results", default=str(DEFAULT_COMPOUND_RESULTS))
    args = parser.parse_args(argv)

    payload = build_payload(
        Path(args.unit_results),
        Path(args.cache_results),
        Path(args.template_results),
        Path(args.compound_results),
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")

    top = payload["aggregate"]
    print(
        f"[MODELED] {len(payload['mechanisms'])} mechanisms; max {top['max_mechanism']} "
        f"= {top['point']['avoided_units']:.1f} units, "
        f"{top['point']['seconds']['mean']:.1f} s, "
        f"${top['point']['dollars']:.3f} "
        f"(band ${top['sensitivity']['dollars']['low']:.3f}-"
        f"{top['sensitivity']['dollars']['high']:.3f}); "
        f"sum allowed: {payload['aggregate']['sum_allowed']}; -> {out}"
    )
    return 0 if all(c["ok"] for c in payload["checks"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
