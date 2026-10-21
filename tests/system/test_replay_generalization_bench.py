"""Replay-generalization benchmark tests (E7).

``examples/benchmarks/replay_generalization_bench.py`` measures the
parameterized-replay contract shipped by I6: a composite instruction
mined from traces that recorded only a handful of parameter values
must, when executed with parameter values *never observed during
mining*, reproduce -- case by case, exactly -- what re-running its
recorded constituent sequence with those same values produces.

These tests pin the harness itself at tiny scale (the committed
``replay_generalization_results.json`` comes from the full 5-seed x
210-case run):

* the unseen-case draw is genuinely unseen (per-coordinate disjoint
  from every recorded value, unique tuples, never a recorded tuple)
  and meets the >= 200-per-seed scale requirement;
* a tiny run agrees on every case across all three calling
  conventions, with schema tightened to the observed types;
* every published aggregate is recomputable from the published raw
  rows, and the measurement is deterministic in the seeds;
* a schema-violating value is recorded honestly as a non-match error
  case instead of aborting or being silently counted as agreement;
* the CLI writes a provenance-complete JSON and fails loudly (exit 1)
  on mismatches or sub-minimum scale.
"""

import asyncio
import itertools
import json
import statistics
import sys
from pathlib import Path

import pytest

# Make the benchmark script importable (it lives in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import replay_generalization_bench as bench  # noqa: E402
from replay_generalization_bench import (  # noqa: E402
    COMPOSITIONS,
    CONVENTIONS,
    DEFAULT_CASES_PER_SEED,
    DEFAULT_SEEDS,
    MIN_CASES_PER_SEED,
    build_call,
    composition_for_seed,
    evaluate_case,
    main,
    mine_learned,
    recorded_values_for_seed,
    run_experiment,
    run_seed,
    unseen_cases_for_seed,
)

SCHEMA_NAME = "replay_generalization_results/v1"


# ---------------------------------------------------------------------------
# Experiment design: compositions, rotation, and the unseen-value draw
# ---------------------------------------------------------------------------

def test_three_parameterized_compositions_rotate_over_the_seeds():
    assert set(COMPOSITIONS) == {"filter3", "scale4", "chain5"}
    # Each composition is a multi-operator stream with its own
    # parameter surface; step ids stay unique inside a stream.
    assert COMPOSITIONS["filter3"]["params"] == ["seed", "cutoff"]
    assert COMPOSITIONS["scale4"]["params"] == ["seed", "factor", "cutoff"]
    assert (
        COMPOSITIONS["chain5"]["params"]
        == ["seed", "factor", "shift", "cutoff"]
    )
    for spec in COMPOSITIONS.values():
        assert len(spec["steps"]) >= 3
        assert len(set(spec["steps"])) == len(spec["steps"])
    # Deterministic rotation covers every composition across the
    # default five seeds.
    assert set(DEFAULT_SEEDS) == {0, 1, 2, 3, 4}
    rotated = {composition_for_seed(s) for s in DEFAULT_SEEDS}
    assert rotated == set(COMPOSITIONS)
    assert all(
        composition_for_seed(s) == composition_for_seed(s)
        for s in DEFAULT_SEEDS
    )


@pytest.mark.parametrize("seed", DEFAULT_SEEDS)
def test_unseen_cases_are_genuinely_unseen_and_sufficient(seed):
    recorded = recorded_values_for_seed(seed)
    params = COMPOSITIONS[composition_for_seed(seed)]["params"]
    cases = unseen_cases_for_seed(seed, DEFAULT_CASES_PER_SEED)

    # Scale: the committed run draws >= 200 unseen cases per seed.
    assert len(cases) == DEFAULT_CASES_PER_SEED
    assert len(cases) >= MIN_CASES_PER_SEED

    # Every coordinate value differs from every recorded value ...
    for case in cases:
        for name in params:
            assert isinstance(case[name], int)
            assert case[name] not in recorded[name]
    # ... no recorded (cross-product) tuple is reused ...
    recorded_tuples = {
        tuple(combo)
        for combo in itertools.product(*(recorded[p] for p in params))
    }
    case_tuples = [tuple(case[p] for p in params) for case in cases]
    assert all(t not in recorded_tuples for t in case_tuples)
    # ... and the unseen tuples are pairwise distinct.
    assert len(set(case_tuples)) == len(case_tuples)


# ---------------------------------------------------------------------------
# Tiny-scale end-to-end measurement
# ---------------------------------------------------------------------------

def test_tiny_run_agrees_on_every_case_all_conventions():
    result = run_seed(0, n_cases=40)

    assert result["n_cases"] == 40
    assert result["n_match"] == 40
    assert result["n_mismatch"] == 0
    assert result["n_error_cases"] == 0
    assert result["agreement_rate"] == 1.0

    # The learned instruction replays the full stream with the
    # recorded parameter schema tightened to the observed type.
    spec = COMPOSITIONS[result["composition"]]
    assert result["stream_ids"] == spec["steps"]
    assert result["params"] == spec["params"]
    assert result["schema_types"] == {p: "int" for p in spec["params"]}
    assert result["n_templates_promoted"] == 1

    # All three calling conventions appear and every case matches.
    assert set(result["convention_counts"]) == set(CONVENTIONS)
    for counts in result["convention_counts"].values():
        assert counts["n"] > 0
        assert counts["n_match"] == counts["n"]

    # Unseen-ness guards hold on the drawn cases.
    validation = result["unseen_validation"]
    assert validation["per_coordinate_disjoint"] is True
    assert validation["tuples_unique"] is True
    assert validation["no_recorded_tuple_reused"] is True
    assert validation["meets_min"] is True

    # Raw rows back the aggregates: one row per case.
    assert len(result["rows"]) == 40
    assert result["n_empty_kept"] >= 0  # measured, not asserted


def test_aggregates_are_recomputable_from_raw_rows():
    result = run_experiment([0, 1], cases_per_seed=24)
    rows = result["cases"]
    assert len(rows) == 48

    params_by_seed = {entry["seed"]: entry["params"]
                      for entry in result["per_seed"]}

    # Recompute every headline aggregate from the raw rows alone.
    n_match = 0
    empty = 0
    per_seed_counts = {seed: {"n": 0, "match": 0} for seed in (0, 1)}
    conv_counts = {name: {"n": 0, "n_match": 0} for name in CONVENTIONS}
    for row in rows:
        seed = row[0]
        k = len(params_by_seed[seed])
        conv_idx, match, empty_kept = row[2 + k], row[3 + k], row[4 + k]
        assert match in (0, 1) and empty_kept in (0, 1)
        n_match += match
        empty += empty_kept
        per_seed_counts[seed]["n"] += 1
        per_seed_counts[seed]["match"] += match
        conv = CONVENTIONS[conv_idx]
        conv_counts[conv]["n"] += 1
        conv_counts[conv]["n_match"] += match

    agg = result["aggregate"]
    assert agg["n_cases_total"] == 48
    assert agg["n_match_total"] == n_match
    assert agg["n_error_cases_total"] == 0
    assert agg["n_mismatch_total"] == 0
    assert agg["agreement_rate"] == n_match / 48
    assert agg["n_empty_kept_total"] == empty
    assert agg["convention_counts"] == conv_counts

    rates = [
        per_seed_counts[seed]["match"] / per_seed_counts[seed]["n"]
        for seed in (0, 1)
    ]
    assert agg["per_seed_agreement_rates"] == rates
    assert agg["mean_per_seed_agreement_rate"] == statistics.mean(rates)
    assert agg["stdev_per_seed_agreement_rate"] == pytest.approx(
        statistics.stdev(rates)
    )


def test_run_experiment_is_deterministic():
    first = run_experiment([0, 1, 2], cases_per_seed=16)
    second = run_experiment([0, 1, 2], cases_per_seed=16)
    assert first["cases"] == second["cases"]
    assert first["aggregate"] == second["aggregate"]
    assert first["per_seed"] == second["per_seed"]
    assert first["errors"] == second["errors"] == []
    assert first["mismatches"] == second["mismatches"] == []


# ---------------------------------------------------------------------------
# Honest failure recording
# ---------------------------------------------------------------------------

def test_schema_violation_is_recorded_not_counted():
    mined = mine_learned(0)
    bad = {name: 7 for name in mined["params"]}
    bad[mined["params"][0]] = "not-an-int"  # unseen value, wrong type

    outcome = asyncio.run(
        evaluate_case(
            mined["learned"], mined["template"], bad, "wrapped"
        )
    )
    assert outcome["match"] is False
    assert outcome["error"] is not None
    assert mined["params"][0] in outcome["error"]

    flat = asyncio.run(
        evaluate_case(mined["learned"], mined["template"], bad, "flat")
    )
    assert flat["match"] is False
    assert flat["error"] is not None


def test_run_seed_records_mismatch_details(monkeypatch):
    real = bench.evaluate_case
    seen = []

    async def sabotaging(learned, template, case_params, convention):
        outcome = await real(learned, template, case_params, convention)
        if len(seen) == 5:  # diverge exactly one case
            outcome = dict(
                outcome, match=False, error=None, replay={"sabotaged": True}
            )
        seen.append(convention)
        return outcome

    monkeypatch.setattr(bench, "evaluate_case", sabotaging)
    result = bench.run_seed(0, n_cases=12)

    assert result["n_cases"] == 12
    assert result["n_match"] == 11
    assert result["n_mismatch"] == 1
    assert result["agreement_rate"] == 11 / 12
    assert len(result["mismatches"]) == 1
    assert result["mismatches"][0]["replay"] == {"sabotaged": True}
    assert result["mismatches"][0]["case_idx"] == 5


# ---------------------------------------------------------------------------
# CLI: results file, provenance, and loud failure
# ---------------------------------------------------------------------------

def test_cli_writes_json_with_provenance(tmp_path):
    out = tmp_path / "replay_generalization_results.json"
    rc = main([
        "--experiment-seeds", "0,1",
        "--cases-per-seed", "24",
        "--min-cases-per-seed", "20",
        "--output", str(out),
    ])
    assert rc == 0

    data = json.loads(out.read_text())
    assert data["schema"] == SCHEMA_NAME

    prov = data["provenance"]
    for key in (
        "generated_at_utc", "git_rev", "git_rev_note", "seeds", "scale",
        "oracle", "conventions", "command", "determinism",
    ):
        assert key in prov
    assert prov["scale"]["n_unseen_cases_total"] == 48
    assert prov["scale"]["n_unseen_cases_per_seed"] == 24
    assert prov["scale"]["n_experiment_seeds"] == 2

    assert data["coverage"]["status"] == "full"
    assert (
        data["coverage"]["component_status"]["I6-replay-parameterization"]
        == "shipped"
    )

    head = data["headline"]
    assert head["n_unseen_cases_total"] == 48
    assert head["n_experiment_seeds"] == 2
    assert head["n_mismatch_total"] == 0
    assert head["agreement_rate"] == 1.0


def test_cli_rejects_runs_below_min_cases(tmp_path, capsys):
    out = tmp_path / "never_written.json"
    rc = main([
        "--experiment-seeds", "0",
        "--cases-per-seed", "10",
        "--min-cases-per-seed", "200",
        "--output", str(out),
    ])
    assert rc == 1
    assert not out.exists()
    assert "min" in capsys.readouterr().err.lower()


def test_cli_writes_results_but_fails_loudly_on_mismatch(
    monkeypatch, tmp_path
):
    real = bench.evaluate_case
    seen = []

    async def sabotaging(learned, template, case_params, convention):
        outcome = await real(learned, template, case_params, convention)
        if len(seen) == 3:
            outcome = dict(
                outcome, match=False, error=None, replay={"sabotaged": True}
            )
        seen.append(convention)
        return outcome

    monkeypatch.setattr(bench, "evaluate_case", sabotaging)
    out = tmp_path / "sabotaged_results.json"
    rc = main([
        "--experiment-seeds", "0",
        "--cases-per-seed", "12",
        "--min-cases-per-seed", "10",
        "--output", str(out),
    ])
    assert rc == 1  # loud failure ...
    data = json.loads(out.read_text())  # ... after honest results
    assert data["aggregate"]["n_mismatch_total"] == 1
    assert data["aggregate"]["n_match_total"] == 11
    assert data["headline"]["agreement_rate"] == 11 / 12
    assert len(data["mismatches"]) == 1


def test_build_call_covers_the_three_conventions():
    case = {"seed": 5, "cutoff": 30}
    assert build_call("wrapped", case) == {"ctx": {"seed": 5, "cutoff": 30}}
    assert build_call("flat", case) == {"seed": 5, "cutoff": 30}
    assert build_call("sibling", case) == {
        "ctx": {}, "seed": 5, "cutoff": 30,
    }
    with pytest.raises(ValueError):
        build_call("nonsense", case)
