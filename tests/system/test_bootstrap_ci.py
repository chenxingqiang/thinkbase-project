"""Tests for the percentile-bootstrap CI script (``scripts/bootstrap_ci.py``).

The script turns committed per-seed microbench raw values into 95% percentile
bootstrap CIs.  These tests pin the contract that matters for honesty:

* every emitted interval brackets its own point estimate and carries the
  ``n`` it was computed from (no silent n=1 intervals);
* the interval is a 10,000-resample percentile bootstrap at 95%;
* output is deterministic -- same input, same bytes, including via the CLI;
* paired arms are differenced by seed, so two identical arms necessarily give
  a difference interval containing (in fact equal to) zero;
* the committed artifact ``bootstrap_ci_results.json`` is in sync with its
  source: same sha256, recomputable means, and the published
  ``paired_interaction`` mean reproduced from its per-seed series.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import bootstrap_ci as bci  # noqa: E402

SOURCE = REPO_ROOT / "examples" / "benchmarks" / "isa_microbench_results.json"
ARTIFACT = REPO_ROOT / "examples" / "benchmarks" / "bootstrap_ci_results.json"


@pytest.fixture(scope="module")
def report():
    return bci.build_report(SOURCE)


@pytest.fixture(scope="module")
def payload():
    return json.loads(SOURCE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# Interval shape
# --------------------------------------------------------------------------

def test_every_metric_brackets_its_mean(report):
    assert report["metrics"], "expected at least one bootstrapable metric"
    for name, entry in report["metrics"].items():
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"], name
        assert entry["ci_low"] <= entry["ci_high"], name


def test_metric_schema_and_n(report):
    for name, entry in report["metrics"].items():
        assert set(entry) == {"mean", "ci_low", "ci_high", "n", "method"}, name
        assert entry["n"] >= 2, f"{name} has n={entry['n']}"
        assert entry["method"] == bci.METHOD
        # the reported n is the number of committed per-seed values
        assert isinstance(entry["n"], int)


def test_method_is_ten_thousand_resample_percentile(report):
    method = report["method"]
    assert method["name"] == "percentile_bootstrap"
    assert method["resamples"] == 10_000
    assert method["confidence"] == 0.95
    assert method["interval"] == [0.025, 0.975]


def test_paired_difference_brackets_mean(report):
    assert report["paired_differences"], "expected a paired difference entry"
    for name, entry in report["paired_differences"].items():
        assert entry["ci_low"] <= entry["mean"] <= entry["ci_high"], name
        assert entry["n"] >= 2, name


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------

def test_build_is_deterministic(report):
    again = bci.build_report(SOURCE)
    assert json.dumps(again, sort_keys=True) == json.dumps(report, sort_keys=True)


def test_cli_is_byte_identical(report, tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    assert bci.main(["--input", str(SOURCE), "--output", str(out_a),
                     "--quiet"]) == 0
    assert bci.main(["--input", str(SOURCE), "--output", str(out_b),
                     "--quiet"]) == 0
    assert out_a.read_bytes() == out_b.read_bytes()

    written = json.loads(out_a.read_text(encoding="utf-8"))
    assert json.dumps(written, sort_keys=True) == json.dumps(
        report, sort_keys=True)


# --------------------------------------------------------------------------
# Paired differences
# --------------------------------------------------------------------------

def test_identical_arms_give_difference_ci_containing_zero():
    arm = [0.70, 0.59, 0.70, 0.64, 0.66]
    result = bci.paired_difference_ci(arm, list(arm), metric="identical")
    assert result["mean"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
    assert result["ci_low"] <= 0.0 <= result["ci_high"]
    assert result["n"] == len(arm)
    assert result["method"] == bci.METHOD


def test_paired_difference_is_per_seed_not_pooled():
    """A constant per-seed offset gives a zero-width interval on the offset."""
    arm_a = [0.10, 0.20, 0.30, 0.40]
    arm_b = [x + 0.05 for x in arm_a]
    result = bci.paired_difference_ci(arm_a, arm_b, metric="offset")
    assert result["mean"] == pytest.approx(-0.05)
    assert result["ci_low"] == pytest.approx(-0.05)
    assert result["ci_high"] == pytest.approx(-0.05)
    assert result["n"] == 4


def test_paired_difference_rejects_misaligned_arms():
    with pytest.raises(ValueError):
        bci.paired_difference_ci([1.0, 2.0, 3.0], [1.0, 2.0], metric="bad")


def test_paired_interaction_matches_published_mean(report, payload):
    published = next(e for e in payload
                     if e["experiment"] == "paired_ablation")["paired_interaction"]
    entry = report["paired_differences"]["paired_ablation.paired_interaction"]
    assert entry["n"] == len(published["per_seed"])
    assert entry["mean"] == pytest.approx(published["mean"], abs=1e-9)
    assert entry["definition"] == published["definition"]


# --------------------------------------------------------------------------
# No CI without raw data
# --------------------------------------------------------------------------

def test_single_observation_gets_no_ci():
    synthetic = [{
        "experiment": "tiny",
        "config": {"seeds": 1},
        "data": [{"seed": 0, "value": 1.5}],
    }]
    metrics, paired, skipped = bci.collect_series(synthetic)
    assert metrics == {}
    assert paired == {}
    assert skipped and skipped[0]["reason"].startswith("n=1")

    with pytest.raises(ValueError):
        bci.bootstrap_mean_ci([1.5], metric="tiny.value")


def test_aggregate_only_experiments_are_declared_not_computed(report):
    names = {entry["experiment"] for entry in report["not_computable"]}
    assert {"cache", "compiler", "ablation"} <= names
    # no metric may be fabricated for an experiment without per-seed raws
    assert not any(k.startswith(("cache.", "compiler.", "ablation."))
                   for k in report["metrics"])


def test_per_seed_rows_resample_each_numeric_field():
    synthetic = [{
        "experiment": "demo",
        "config": {"seeds": 3},
        "data": [
            {"seed": 2, "score": 0.3, "flag": True},
            {"seed": 0, "score": 0.1, "flag": False},
            {"seed": 1, "score": 0.2, "flag": True},
        ],
    }]
    metrics, _, _ = bci.collect_series(synthetic)
    assert list(metrics) == ["demo.score"]
    assert metrics["demo.score"] == [0.1, 0.2, 0.3]  # ordered by seed


# --------------------------------------------------------------------------
# Committed artifact
# --------------------------------------------------------------------------

@pytest.mark.skipif(not ARTIFACT.exists(), reason="artifact not generated")
def test_committed_artifact_is_in_sync(report):
    committed = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert json.dumps(committed, sort_keys=True) == json.dumps(
        report, sort_keys=True)
    assert committed["source"]["sha256"] == hashlib.sha256(
        SOURCE.read_bytes()).hexdigest()
    assert committed["source"]["path"] == (
        SOURCE.relative_to(REPO_ROOT).as_posix())
