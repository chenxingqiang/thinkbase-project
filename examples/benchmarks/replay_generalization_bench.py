#!/usr/bin/env python3
"""Unseen-parameter-value replay-generalization benchmark (E7).

The parameterized-replay contract shipped by I6
(``agentflow/core/isa/instruction_learning.py``) says: a composite
instruction mined from traces that recorded only a handful of
parameter values must, when executed with parameter values *never
observed during mining*, behave exactly like re-running its recorded
constituent sequence with those values -- case by case, no tolerance.

This harness measures that contract at scale and writes the raw
per-case evidence plus every aggregate to
``examples/benchmarks/replay_generalization_results.json``:

* five experiment seeds (task discipline); each rotates onto one of
  three multi-operator stream compositions and drives its own
  deterministic draw of unseen cases;
* ``--cases-per-seed`` unseen parameter combinations per seed
  (default 210, task minimum 200): every coordinate value lies in a
  range disjoint from the recorded mining grid, no recorded
  combination is reused, and the drawn combinations are unique;
* per case the learned composite instruction executes under the
  run_stream calling conventions (wrapped / flat / schema-sibling,
  rotated exactly as in the I6 unit tests) and its result is compared
  for exact equality with the oracle: re-running the recorded
  constituent instances via ``record_trace`` and merging their
  outputs;
* every published aggregate (agreement rates, mean/stdev across the
  five seeds, per-convention and empty-result counts) is computed by
  this script from the raw per-case rows it publishes alongside --
  nothing is hand-computed or hand-filled.

Determinism: the unseen draws, mining, replay and oracle are pure
functions of the experiment seeds; only the timestamp and the
read-only ``git rev-parse --short HEAD`` provenance differ between
runs.  Exit status is 0 only when every case agrees; mismatches and
error cases still produce an honest results file and exit 1.
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import json
import random
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agentflow.core.isa.agentic_calling import AgenticTransformer  # noqa: E402
from agentflow.core.isa.instruction import (  # noqa: E402
    Instruction,
    InstructionType,
)
from agentflow.core.isa.instruction_learning import (  # noqa: E402
    InstructionLearner,
    InstructionTemplate,
    record_trace,
)

SCHEMA_NAME = "replay_generalization_results/v1"
DEFAULT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_CASES_PER_SEED = 210
MIN_CASES_PER_SEED = 200
CONVENTIONS: Tuple[str, ...] = ("wrapped", "flat", "sibling")
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "replay_generalization_results.json"
)
MOD = 97  # value domain of the synthetic arithmetic pipelines

# Unseen draws come from ranges disjoint from the recorded mining grid
# below, so every replayed value is genuinely unseen per coordinate.
UNSEEN_RANGES: Dict[str, Tuple[int, int]] = {
    "seed": (100, 499),
    "factor": (4, 13),
    "shift": (3, 15),
    "cutoff": (21, 98),  # reaches past the value domain -> empty kept
}

CONVENTION_DESCRIPTIONS: Dict[str, str] = {
    "wrapped": "{'ctx': {<params>}} -- the run_stream convention the "
    "mining traces were recorded under",
    "flat": "{<params>} -- the bare parameter dict convention",
    "sibling": "{'ctx': {}, <params>} -- schema parameters passed "
    "alongside an empty ctx (the sibling-binding path I6 added)",
}


# ---------------------------------------------------------------------------
# Synthetic stream operators (pure, deterministic, ctx-connected): every
# step reads the shared ctx and merges its dict outputs back into it.
# ---------------------------------------------------------------------------

class GenValues(Instruction):
    """Derive the raw value list from the seed (pure arithmetic)."""

    def __init__(self):
        super().__init__(
            id="gen_values", name="gen_values",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Derive raw values from the seed",
            side_effects=False, outputs=["values"],
        )
        self.dependencies = ["seed"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seed = input_params["ctx"]["seed"]
        return {
            "values": [(seed * (i + 3)) % MOD + 1 for i in range(8)]
        }


class ScaleValues(Instruction):
    """Scale the values by a factor (pure)."""

    def __init__(self):
        super().__init__(
            id="scale_values", name="scale_values",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Scale values by a factor",
            side_effects=False, outputs=["scaled"],
        )
        self.dependencies = ["values", "factor"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        values = input_params["ctx"]["values"]
        factor = input_params["ctx"]["factor"]
        return {"scaled": [(v * factor) % MOD for v in values]}


class ShiftValues(Instruction):
    """Shift the scaled values (pure)."""

    def __init__(self):
        super().__init__(
            id="shift_values", name="shift_values",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Shift scaled values",
            side_effects=False, outputs=["shifted"],
        )
        self.dependencies = ["scaled", "shift"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        scaled = input_params["ctx"]["scaled"]
        shift = input_params["ctx"]["shift"]
        return {"shifted": [(v + shift) % MOD for v in scaled]}


class FilterValues(Instruction):
    """Keep the values at or above the cutoff (pure)."""

    def __init__(self, input_key: str = "values"):
        super().__init__(
            id="filter_values", name="filter_values",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Filter values by cutoff",
            side_effects=False, outputs=["kept"],
        )
        self.input_key = input_key
        self.dependencies = [input_key, "cutoff"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        values = input_params["ctx"][self.input_key]
        cutoff = input_params["ctx"]["cutoff"]
        return {"kept": [v for v in values if v >= cutoff]}


class TotalKept(Instruction):
    """Summarize the kept values (pure)."""

    def __init__(self):
        super().__init__(
            id="total_kept", name="total_kept",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Total the kept values",
            side_effects=False, outputs=["total", "report"],
        )
        self.dependencies = ["kept"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        kept = input_params["ctx"]["kept"]
        total = sum(kept)
        return {"total": total, "report": f"kept={total}"}


def build_stream(composition: str) -> List[Instruction]:
    """Build a fresh instance of the composition's operator stream."""
    if composition == "filter3":
        return [GenValues(), FilterValues("values"), TotalKept()]
    if composition == "scale4":
        return [GenValues(), ScaleValues(), FilterValues("scaled"), TotalKept()]
    if composition == "chain5":
        return [
            GenValues(), ScaleValues(), ShiftValues(),
            FilterValues("shifted"), TotalKept(),
        ]
    raise ValueError(f"unknown composition: {composition!r}")


# The experiment's stream compositions: multi-operator sequences, each
# with its own parameter surface.  ``recorded`` is the small grid of
# parameter values the mining traces are recorded over (the only values
# the learner ever observes); the unseen replay draws come from ranges
# disjoint from it.  ``params`` lists the external inputs in mined
# order (the order mining discovers them, hence the template's
# parameter order).
COMPOSITION_NAMES: Tuple[str, ...] = ("filter3", "scale4", "chain5")
COMPOSITIONS: Dict[str, Dict[str, Any]] = {
    "filter3": {
        "params": ["seed", "cutoff"],
        "steps": ["gen_values", "filter_values", "total_kept"],
        "recorded": {"seed": (1, 2, 3), "cutoff": (10, 20)},
    },
    "scale4": {
        "params": ["seed", "factor", "cutoff"],
        "steps": ["gen_values", "scale_values", "filter_values",
                  "total_kept"],
        "recorded": {"seed": (1, 2, 3), "factor": (2, 3),
                     "cutoff": (10, 20)},
    },
    "chain5": {
        "params": ["seed", "factor", "shift", "cutoff"],
        "steps": ["gen_values", "scale_values", "shift_values",
                  "filter_values", "total_kept"],
        "recorded": {"seed": (1, 2, 3), "factor": (2, 3), "shift": (1, 2),
                     "cutoff": (10, 20)},
    },
}


def composition_for_seed(seed: int) -> str:
    """Deterministic rotation of the seeds over the compositions."""
    return COMPOSITION_NAMES[seed % len(COMPOSITION_NAMES)]


def recorded_values_for_seed(seed: int) -> Dict[str, Tuple[int, ...]]:
    """The recorded (mining) parameter grid for the seed's composition."""
    return dict(COMPOSITIONS[composition_for_seed(seed)]["recorded"])


def unseen_cases_for_seed(seed: int, n_cases: int) -> List[Dict[str, int]]:
    """Draw ``n_cases`` genuinely unseen parameter combinations.

    Deterministic in ``seed``: one ``random.Random`` per experiment
    seed draws each coordinate from a range disjoint from the recorded
    grid, rejecting duplicate combinations so the draw is unique.
    """
    params = list(COMPOSITIONS[composition_for_seed(seed)]["params"])
    rng = random.Random(f"e7-replay-generalization-seed{seed}")
    cases: List[Dict[str, int]] = []
    seen_tuples: set = set()
    attempts = 0
    while len(cases) < n_cases:
        attempts += 1
        if attempts > 100 * max(n_cases, 1) + 1000:
            raise RuntimeError(
                "unseen-case draw exhausted: range too small for "
                f"{n_cases} unique combinations"
            )
        candidate = {
            p: rng.randint(*UNSEEN_RANGES[p]) for p in params
        }
        key = tuple(candidate[p] for p in params)
        if key in seen_tuples:
            continue
        seen_tuples.add(key)
        cases.append(candidate)
    return cases


def validate_unseen(
    recorded: Dict[str, Tuple[int, ...]],
    params: List[str],
    cases: List[Dict[str, int]],
) -> Dict[str, Any]:
    """Honest guards that every replayed value is genuinely unseen."""
    case_tuples = [tuple(c[p] for p in params) for c in cases]
    recorded_tuples = {
        tuple(combo)
        for combo in itertools.product(*(recorded[p] for p in params))
    }
    return {
        "per_coordinate_disjoint": all(
            all(c[p] not in recorded[p] for p in params) for c in cases
        ),
        "tuples_unique": len(set(case_tuples)) == len(case_tuples),
        "no_recorded_tuple_reused": all(
            t not in recorded_tuples for t in case_tuples
        ),
    }


# ---------------------------------------------------------------------------
# Mining: learn one composite template per experiment seed
# ---------------------------------------------------------------------------

async def _mine_learned(seed: int) -> Dict[str, Any]:
    """Mine the seed's composition over its recorded parameter grid."""
    composition = composition_for_seed(seed)
    spec = COMPOSITIONS[composition]
    params = list(spec["params"])
    recorded = dict(spec["recorded"])
    traces = [
        await record_trace(
            build_stream(composition), dict(zip(params, combo))
        )
        for combo in itertools.product(*(recorded[p] for p in params))
    ]
    learner = InstructionLearner(AgenticTransformer())
    templates = learner.learn_from_traces(traces)
    if len(templates) != 1:
        raise RuntimeError(
            f"seed {seed}: expected exactly one promoted composite "
            f"template from {len(traces)} traces, got {len(templates)}"
        )
    template = templates[0]
    learned = learner.create_instruction(template)
    return {
        "composition": composition,
        "stream_ids": [c.id for c in template.components],
        "params": list(template.params.keys()),
        "schema": dict(template.params),
        "recorded": recorded,
        "n_mining_traces": len(traces),
        "n_templates_promoted": len(templates),
        "template": template,
        "learned": learned,
    }


def mine_learned(seed: int) -> Dict[str, Any]:
    """Sync wrapper around :func:`_mine_learned` (for inspection/tests)."""
    return asyncio.run(_mine_learned(seed))


# ---------------------------------------------------------------------------
# Per-case measurement: replay under a convention vs oracle re-run
# ---------------------------------------------------------------------------

def build_call(convention: str, case: Dict[str, int]) -> Dict[str, Any]:
    """Build the learned instruction's input for one calling convention."""
    if convention == "wrapped":
        return {"ctx": dict(case)}
    if convention == "flat":
        return dict(case)
    if convention == "sibling":
        return {"ctx": {}, **case}
    raise ValueError(f"unknown calling convention: {convention!r}")


async def rerun_constituents(
    template: InstructionTemplate, case_params: Dict[str, int]
) -> Dict[str, Any]:
    """Oracle: re-run the recorded constituent sequence with new values."""
    reference_trace = await record_trace(
        list(template.components), dict(case_params)
    )
    merged: Dict[str, Any] = {}
    for entry in reference_trace:
        merged.update(entry["outputs"])
    return merged


async def evaluate_case(
    learned: Instruction,
    template: InstructionTemplate,
    case_params: Dict[str, int],
    convention: str,
) -> Dict[str, Any]:
    """Execute one case under ``convention`` and compare to the oracle.

    Failures are recorded, never swallowed: a replay exception or an
    oracle exception becomes a non-match case with the error text; a
    divergent-but-successful replay becomes a mismatch carrying both
    results for the report.
    """
    call = build_call(convention, case_params)
    replay: Any = None
    replay_error: Optional[str] = None
    try:
        replay = await learned.execute(call)
    except Exception as exc:  # recorded honestly, never silently
        replay_error = f"{type(exc).__name__}: {exc}"
    expected: Any = None
    reference_error: Optional[str] = None
    try:
        expected = await rerun_constituents(template, case_params)
    except Exception as exc:
        reference_error = f"{type(exc).__name__}: {exc}"
    error = replay_error or reference_error
    return {
        "match": error is None and replay == expected,
        "replay": replay,
        "expected": expected,
        "error": error,
        "empty_kept": (
            1
            if isinstance(expected, dict) and expected.get("kept") == []
            else 0
        ),
    }


# ---------------------------------------------------------------------------
# Per-seed and whole-experiment runs
# ---------------------------------------------------------------------------

async def _run_seed(
    seed: int, n_cases: int, min_cases: Optional[int]
) -> Dict[str, Any]:
    """Mine one composite and measure it on ``n_cases`` unseen cases."""
    mined = await _mine_learned(seed)
    template = mined["template"]
    learned = mined["learned"]
    params = mined["params"]
    recorded = mined["recorded"]
    cases = unseen_cases_for_seed(seed, n_cases)

    rows: List[Tuple[Any, ...]] = []
    mismatches: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    convention_counts = {
        name: {"n": 0, "n_match": 0} for name in CONVENTIONS
    }
    n_match = 0
    n_empty_kept = 0

    for i, case in enumerate(cases):
        conv_idx = i % len(CONVENTIONS)
        convention = CONVENTIONS[conv_idx]
        outcome = await evaluate_case(
            learned, template, dict(case), convention
        )
        match_int = 1 if outcome["match"] else 0
        error_int = 1 if outcome["error"] else 0
        rows.append(
            (seed, i, *[case[p] for p in params], conv_idx,
             match_int, outcome["empty_kept"], error_int)
        )
        n_match += match_int
        n_empty_kept += outcome["empty_kept"]
        convention_counts[convention]["n"] += 1
        convention_counts[convention]["n_match"] += match_int
        if outcome["error"]:
            errors.append({
                "seed": seed, "case_idx": i, "case": dict(case),
                "convention": convention, "error": outcome["error"],
            })
        elif not outcome["match"]:
            mismatches.append({
                "seed": seed, "case_idx": i, "case": dict(case),
                "convention": convention,
                "replay": outcome["replay"],
                "expected": outcome["expected"],
                "error": outcome["error"],
            })

    validation = validate_unseen(recorded, params, cases)
    validation["meets_min"] = bool(
        min_cases is None or len(cases) >= min_cases
    )

    return {
        "seed": seed,
        "composition": mined["composition"],
        "stream_ids": mined["stream_ids"],
        "params": params,
        "recorded_values": {p: list(recorded[p]) for p in params},
        "schema_types": {
            p: mined["schema"][p].base_type.__name__ for p in params
        },
        "n_mining_traces": mined["n_mining_traces"],
        "n_templates_promoted": mined["n_templates_promoted"],
        "n_cases": len(cases),
        "n_match": n_match,
        "n_mismatch": len(mismatches),
        "n_error_cases": len(errors),
        "agreement_rate": n_match / len(cases),
        "n_empty_kept": n_empty_kept,
        "convention_counts": convention_counts,
        "unseen_validation": validation,
        "rows": rows,
        "mismatches": mismatches,
        "errors": errors,
    }


def run_seed(
    seed: int,
    n_cases: int = DEFAULT_CASES_PER_SEED,
    min_cases: Optional[int] = None,
) -> Dict[str, Any]:
    """Sync wrapper around :func:`_run_seed`."""
    return asyncio.run(_run_seed(seed, n_cases, min_cases))


def _aggregate(
    results: List[Dict[str, Any]],
    rows: List[Tuple[Any, ...]],
    seeds: List[int],
) -> Dict[str, Any]:
    """Compute every headline number from the raw per-case rows.

    Single source of truth: the rows themselves.  Row layout is
    ``[seed, case_idx, <value per param>, conv_idx, match, empty_kept,
    error]`` so all aggregates below are recomputable from ``cases``.
    """
    params_len = {r["seed"]: len(r["params"]) for r in results}
    convention_counts = {
        name: {"n": 0, "n_match": 0} for name in CONVENTIONS
    }
    per_seed_counts = {
        seed: {"n": 0, "n_match": 0} for seed in seeds
    }
    n_match = n_mismatch = n_error = n_empty = 0
    for row in rows:
        seed = row[0]
        k = params_len[seed]
        conv_idx, match, empty, err = (
            row[2 + k], row[3 + k], row[4 + k], row[5 + k]
        )
        n_match += match
        n_empty += empty
        if err:
            n_error += 1
        elif not match:
            n_mismatch += 1
        convention_counts[CONVENTIONS[conv_idx]]["n"] += 1
        convention_counts[CONVENTIONS[conv_idx]]["n_match"] += match
        per_seed_counts[seed]["n"] += 1
        per_seed_counts[seed]["n_match"] += match
    rates = [
        per_seed_counts[seed]["n_match"] / per_seed_counts[seed]["n"]
        for seed in seeds
        if per_seed_counts[seed]["n"]
    ]
    n_total = len(rows)
    return {
        "n_seeds": len(seeds),
        "n_cases_total": n_total,
        "n_match_total": n_match,
        "n_mismatch_total": n_mismatch,
        "n_error_cases_total": n_error,
        "agreement_rate": n_match / n_total,
        "n_empty_kept_total": n_empty,
        "convention_counts": convention_counts,
        "per_seed_agreement_rates": rates,
        "mean_per_seed_agreement_rate": statistics.mean(rates),
        "stdev_per_seed_agreement_rate": (
            statistics.stdev(rates) if len(rates) > 1 else 0.0
        ),
    }


async def _run_experiment(
    seeds: List[int], cases_per_seed: int, min_cases: Optional[int]
) -> Dict[str, Any]:
    results = [
        await _run_seed(seed, cases_per_seed, min_cases)
        for seed in seeds
    ]
    cases = [row for r in results for row in r["rows"]]
    mismatches = [m for r in results for m in r["mismatches"]]
    errors = [e for r in results for e in r["errors"]]
    per_seed = [
        {k: v for k, v in r.items() if k != "rows"} for r in results
    ]
    return {
        "seeds": seeds,
        "cases_per_seed": cases_per_seed,
        "min_cases_per_seed": min_cases,
        "per_seed": per_seed,
        "cases": cases,
        "mismatches": mismatches,
        "errors": errors,
        "aggregate": _aggregate(results, cases, seeds),
    }


def run_experiment(
    seeds: Optional[List[int]] = None,
    cases_per_seed: int = DEFAULT_CASES_PER_SEED,
    min_cases: Optional[int] = MIN_CASES_PER_SEED,
) -> Dict[str, Any]:
    """Run the whole measurement; deterministic in the seeds."""
    seed_list = list(DEFAULT_SEEDS) if seeds is None else list(seeds)
    return asyncio.run(
        _run_experiment(seed_list, cases_per_seed, min_cases)
    )


# ---------------------------------------------------------------------------
# Results file: provenance, coverage, headline
# ---------------------------------------------------------------------------

def git_rev() -> str:
    """Read-only ``git rev-parse --short HEAD`` for provenance only.

    This is the single git operation the E7 task permits; any failure
    (missing git, detached environment) is recorded honestly as
    ``"unavailable"``.
    """
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parents[2]),
            capture_output=True, text=True, timeout=15, check=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unavailable"


def experiment_design() -> Dict[str, Any]:
    """The fixed design of the measurement, for the results file."""
    return {
        "question": (
            "Does a learned composite instruction, mined from traces "
            "that recorded only a small grid of parameter values, "
            "replay exactly like re-running its recorded constituent "
            "sequence for parameter values never observed during "
            "mining?"
        ),
        "method": [
            "per experiment seed, mine one composite template from "
            "traces of a multi-operator stream recorded over its "
            "recorded parameter grid (the only values mining sees)",
            "draw N >= 200 unseen parameter combinations per seed from "
            "ranges disjoint from the recorded grid: per-coordinate "
            "disjoint, unique, never a recorded tuple",
            "for each unseen case, execute the learned instruction "
            "under a rotating calling convention and compare its "
            "result for exact equality against re-running the "
            "recorded constituent sequence with the same values",
            "aggregate per-seed agreement rates to mean/stdev; publish "
            "the raw per-case rows alongside every aggregate",
        ],
        "compositions": {
            name: {
                "params": list(spec["params"]),
                "steps": list(spec["steps"]),
                "recorded_mining_values": {
                    p: list(spec["recorded"][p])
                    for p in spec["params"]
                },
            }
            for name, spec in COMPOSITIONS.items()
        },
        "composition_rotation": (
            f"composition_for_seed(seed) = {COMPOSITION_NAMES}"
            "[seed % 3]"
        ),
        "unseen_value_ranges": {
            k: {"low": v[0], "high": v[1]}
            for k, v in UNSEEN_RANGES.items()
        },
        "conventions": dict(CONVENTION_DESCRIPTIONS),
        "streams_note": (
            "synthetic deterministic integer pipelines (mod-97 "
            "arithmetic) over the run_stream ctx convention: every "
            "step reads the shared ctx and merges its dict outputs "
            "back; the replayed operators are pure and stateless"
        ),
    }


def build_payload(
    result: Dict[str, Any], command_argv: List[str]
) -> Dict[str, Any]:
    """Assemble the full results-file payload."""
    agg = result["aggregate"]
    seeds = result["seeds"]
    return {
        "schema": SCHEMA_NAME,
        "provenance": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_rev": git_rev(),
            "git_rev_note": (
                "read-only 'git rev-parse --short HEAD' run from the "
                "script for provenance only -- the single git "
                "operation the E7 task permits; 'unavailable' when "
                "git cannot run"
            ),
            "seeds": {
                "experiment_seeds": seeds,
                "n_experiment_seeds": len(seeds),
                "role": (
                    "each experiment seed rotates onto a stream "
                    "composition (seed % 3) and drives its own "
                    "deterministic draw of unseen cases"
                ),
                "unseen_draw_rng": (
                    "random.Random('e7-replay-generalization-seed"
                    "{seed}') -- recorded draw order, stable across "
                    "runs"
                ),
                "multi_seed_note": (
                    "five seeds by default (task discipline); the "
                    "per-seed agreement rates are aggregated to "
                    "mean/stdev by this script from the raw rows"
                ),
            },
            "scale": {
                "n_experiment_seeds": len(seeds),
                "n_unseen_cases_per_seed": result["cases_per_seed"],
                "n_unseen_cases_total": agg["n_cases_total"],
                "n_compositions": len(COMPOSITIONS),
                "n_calling_conventions": len(CONVENTIONS),
                "min_cases_per_seed_required": (
                    result["min_cases_per_seed"]
                ),
            },
            "oracle": (
                "per case, the recorded constituent sequence is "
                "re-run from scratch via record_trace with the unseen "
                "values as the initial shared context and the step "
                "outputs are merged; the learned composite's replay "
                "result must equal that merged dict exactly "
                "(per-case exact equality, no tolerance)"
            ),
            "conventions": dict(CONVENTION_DESCRIPTIONS),
            "command": [
                "examples/benchmarks/replay_generalization_bench.py",
                *command_argv,
            ],
            "determinism": (
                "the unseen draws, mining, replay and oracle are "
                "deterministic functions of the experiment seeds; "
                "only generated_at_utc and git_rev vary between runs; "
                "every aggregate is computed by this script from the "
                "raw per-case rows published in 'cases'"
            ),
            "cases_row_schema": (
                "[seed, case_idx, <one value per composition param, "
                "in template.params order>, convention_idx, "
                "match(0/1), empty_kept(0/1), error(0/1)]; convention "
                f"names are {CONVENTIONS}[convention_idx]"
            ),
        },
        "experiment": experiment_design(),
        "per_seed": result["per_seed"],
        "aggregate": agg,
        "cases": result["cases"],
        "mismatches": result["mismatches"],
        "errors": result["errors"],
        "coverage": {
            "status": "full",
            "component_status": {
                "I6-replay-parameterization": "shipped",
            },
            "components_required": [
                "I6-replay-parameterization: learned composite replay "
                "+ saved parameter schema + sibling binding (all "
                "shipped per its implementation report; this "
                "benchmark only ran because they are)"
            ],
            "measured": [
                "per-case exact-equality agreement of learned "
                "composite replay vs oracle re-run on unseen parameter "
                "values (the headline)",
                "agreement per calling convention (wrapped/flat/"
                "sibling)",
                "agreement per stream composition "
                "(filter3/scale4/chain5)",
                "empty-result edge coverage (kept == [] cases, "
                "counted in n_empty_kept_total)",
                "unseen-ness guards (per-coordinate disjoint, unique "
                "tuples, no recorded tuple reused, per-seed minimum)",
            ],
            "not_covered": [
                "dialogue-mined templates with an attached executor "
                "(I6's other arm): out of scope -- no recorded "
                "composite body to re-run, nothing to compare",
                "constructor-bound parameters (values baked into "
                "instruction state at construction time): replay "
                "re-runs the recorded instances, so only ctx-carried "
                "values can vary (I6 deviation 1); the streams here "
                "read every parameter from ctx",
                "real-workload / LLM streams: the measured streams "
                "are deterministic synthetic arithmetic; the claim "
                "measured is exact per-case equality on this domain, "
                "not a distributional claim about arbitrary "
                "workloads",
                "schema-violating values: recorded honestly as error "
                "cases by the harness (pinned by the test suite) but "
                "deliberately absent from the committed run's "
                "denominator, which replays only type-valid unseen "
                "values",
            ],
            "provenance_gaps": [],
        },
        "headline": {
            "n_unseen_cases_total": agg["n_cases_total"],
            "n_experiment_seeds": len(seeds),
            "n_unseen_cases_per_seed": result["cases_per_seed"],
            "n_match_total": agg["n_match_total"],
            "n_mismatch_total": agg["n_mismatch_total"],
            "n_error_cases_total": agg["n_error_cases_total"],
            "agreement_rate": agg["agreement_rate"],
            "mean_per_seed_agreement_rate": (
                agg["mean_per_seed_agreement_rate"]
            ),
            "stdev_per_seed_agreement_rate": (
                agg["stdev_per_seed_agreement_rate"]
            ),
            "n_empty_kept_total": agg["n_empty_kept_total"],
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "E7 replay-generalization benchmark: learned composite "
            "instructions vs oracle re-run on unseen parameter values"
        )
    )
    parser.add_argument(
        "--experiment-seeds",
        default=",".join(str(s) for s in DEFAULT_SEEDS),
        help="comma-separated experiment seeds (default: 0,1,2,3,4)",
    )
    parser.add_argument(
        "--cases-per-seed", type=int, default=DEFAULT_CASES_PER_SEED,
        help=f"unseen cases per seed (default: {DEFAULT_CASES_PER_SEED})",
    )
    parser.add_argument(
        "--min-cases-per-seed", type=int, default=MIN_CASES_PER_SEED,
        help=f"required minimum unseen cases per seed "
             f"(default: {MIN_CASES_PER_SEED})",
    )
    parser.add_argument(
        "--output", default=str(DEFAULT_OUTPUT_PATH),
        help="results JSON path",
    )
    args = parser.parse_args(argv)

    seeds = [
        int(s) for s in str(args.experiment_seeds).split(",")
        if s.strip() != ""
    ]
    result = run_experiment(
        seeds, args.cases_per_seed, args.min_cases_per_seed
    )
    agg = result["aggregate"]

    deficient = [
        entry["seed"] for entry in result["per_seed"]
        if entry["n_cases"] < args.min_cases_per_seed
    ]
    if deficient:
        print(
            f"ERROR: seeds {deficient} produced fewer unseen cases "
            f"than the required minimum of {args.min_cases_per_seed} "
            "per seed; refusing to write results",
            file=sys.stderr,
        )
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command_argv = list(sys.argv[1:]) if argv is None else list(argv)
    payload = build_payload(result, command_argv)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")

    print(
        f"seeds={seeds} unseen_cases={agg['n_cases_total']} "
        f"match={agg['n_match_total']} "
        f"mismatch={agg['n_mismatch_total']} "
        f"error_cases={agg['n_error_cases_total']} "
        f"agreement_rate={agg['agreement_rate']:.4f} "
        f"mean_per_seed={agg['mean_per_seed_agreement_rate']:.4f} "
        f"stdev_per_seed={agg['stdev_per_seed_agreement_rate']:.4f} "
        f"empty_kept={agg['n_empty_kept_total']}"
    )
    print(f"results written to {output_path}")
    if agg["n_mismatch_total"] or agg["n_error_cases_total"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
