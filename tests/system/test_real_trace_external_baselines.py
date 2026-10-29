"""Tests for the real-trace external baseline harness (examples/benchmarks).

The harness exists so that Section 5.12's 41.70% can be read against an
ISA-agnostic cache. These tests pin the *properties that make the comparison
readable* -- monotonicity in capacity, exhaustive hit decomposition, the
sound/naive separation, the key recipe shared with the published measurement --
plus reproducibility of the emitted JSON. They do not pin the numbers
themselves beyond the anchor, which must keep matching the published figure.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS = REPO_ROOT / "examples" / "benchmarks" / "real_trace_external_baselines.py"
CORPUS = REPO_ROOT / "examples" / "real_traces" / "tau_bench_historical.jsonl"


def _load_module():
    spec = importlib.util.spec_from_file_location("real_trace_external_baselines", HARNESS)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def harness():
    return _load_module()


@pytest.fixture(scope="module")
def synthetic_report(harness):
    return harness.build_report(HARNESS, harness._synth_records())


# --------------------------------------------------------------------------
# Self-test contract
# --------------------------------------------------------------------------
def test_self_test_runs_and_writes_no_result_file(harness, tmp_path: Path) -> None:
    out = tmp_path / "should_not_exist.json"
    assert harness.main(["--self-test", "--output", str(out)]) == 0
    assert not out.exists(), "self-test must never emit a results artifact"


# --------------------------------------------------------------------------
# Cache invariants
# --------------------------------------------------------------------------
def test_lru_hits_are_monotone_in_capacity(harness, synthetic_report) -> None:
    caps = ["0"] + [str(c) for c in harness.CAPACITIES]
    for mode in harness.MODES:
        for scope in harness.SCOPES:
            prev = -1
            for cap in caps:
                hits = synthetic_report["lru"][mode][scope][cap]["hits"]
                assert hits >= prev, f"{mode}/{scope} not monotone at cap={cap}"
                prev = hits


def test_hit_decomposition_is_exhaustive(harness, synthetic_report) -> None:
    for mode in harness.MODES:
        for scope in harness.SCOPES:
            for cap in ["0"] + [str(c) for c in harness.CAPACITIES]:
                row = synthetic_report["lru"][mode][scope][cap]
                assert row["hits_cross_task"] + row["hits_same_task"] == row["hits"]
                assert row["hits"] <= row["calls"]
                assert row["hits_cross_task"] <= row["hits"]
                assert row["cacheable_calls"] + row["effectful_calls"] == row["calls"]


def test_sound_mode_never_beats_naive_and_never_replays_effectful(
    harness, synthetic_report
) -> None:
    for scope in harness.SCOPES:
        for cap in ["0"] + [str(c) for c in harness.CAPACITIES]:
            sound = synthetic_report["lru"]["sound"][scope][cap]
            naive = synthetic_report["lru"]["naive"][scope][cap]
            assert sound["hits"] <= naive["hits"]
            # sound mode only ever stores side-effect-free keys.
            assert sound["hits_unsound_effectful"] == 0


def test_task_scope_is_order_insensitive_at_full_capacity(harness) -> None:
    """A per-task cache cannot see across records, so reordering tasks must not
    change the uncapped per-task hit count."""
    records = harness._synth_records()
    forward = harness.lru_sweep(records, 0, "task", "sound")["hits"]
    backward = harness.lru_sweep(list(reversed(records)), 0, "task", "sound")["hits"]
    assert forward == backward


def test_run_scope_requires_grouping(harness) -> None:
    with pytest.raises(ValueError):
        harness.lru_sweep(harness._synth_records(), 0, "run", "sound")


# --------------------------------------------------------------------------
# Key recipe
# --------------------------------------------------------------------------
def test_key_matches_the_published_recipe(harness) -> None:
    """The baseline must hash exactly what Section 5.12 hashes: tool + sorted
    canonical args. Otherwise the comparison is key-for-key in name only."""
    import hashlib

    tool, args = "get_order_details", {"order_id": "#W1", "x": [1, 2]}
    expected = hashlib.sha256(
        f"{tool}|{json.dumps(dict(args), sort_keys=True, default=str)}".encode()
    ).hexdigest()
    assert harness.call_key(tool, args) == expected
    # argument insertion order must not split a key
    assert harness.call_key(tool, {"x": [1, 2], "order_id": "#W1"}) == expected
    # distinct tools sharing args must not collide
    assert harness.call_key("other_tool", args) != expected


# --------------------------------------------------------------------------
# Hot set and same-turn
# --------------------------------------------------------------------------
def test_hot_set_is_monotone_and_bounded(harness, synthetic_report) -> None:
    for mode in harness.MODES:
        for trainer in ("global", "per_source"):
            prev = -1
            for n in harness.HOT_SET_SIZES:
                row = synthetic_report["hot_set"][mode][trainer][str(n)]
                assert row["hits"] >= prev
                assert row["hits"] <= row["calls"]
                assert 0.0 <= row["hit_rate"] <= 1.0
                prev = row["hits"]


def test_hot_set_never_hits_a_key_absent_from_train(harness, synthetic_report) -> None:
    """A static list cannot invent keys: hits are bounded by the train vocabulary."""
    for mode in harness.MODES:
        for n in harness.HOT_SET_SIZES:
            row = synthetic_report["hot_set"][mode]["per_source"][str(n)]
            assert row["hits"] <= row["calls"] - row["test_keys_unseen_in_train"]


def test_same_turn_duplicate_detection(synthetic_report) -> None:
    st = synthetic_report["same_turn"]
    assert st["identifiable"] is True
    assert st["same_turn_duplicate_calls"] == 1
    assert st["turns_containing_a_duplicate"] == 1
    assert st["batch_size_histogram"].get("2") == 1


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------
def test_report_is_deterministic(harness) -> None:
    a = harness.build_report(HARNESS, harness._synth_records())
    b = harness.build_report(HARNESS, harness._synth_records())
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    # no wall-clock or VCS provenance may leak into the artifact
    blob = json.dumps(a)
    assert "utc_timestamp" not in blob and "git_rev" not in blob


# --------------------------------------------------------------------------
# Real corpus (skipped when the corpus is absent)
# --------------------------------------------------------------------------
@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus not present")
def test_anchor_reproduces_the_published_figures(harness) -> None:
    report = harness.build_report(CORPUS, harness.load_records(CORPUS))
    anchor = report["anchor"]
    assert anchor["matches_published"] is True
    assert anchor["reproduced_within_run"] == pytest.approx(0.417, abs=5e-4)
    assert anchor["reproduced_within_task"] == pytest.approx(0.0173, abs=5e-4)
    assert (anchor["reproduced_hits"], anchor["reproduced_cacheable_calls"]) == (
        746, 1789,
    )
    # the denominator that yields 41.70% is the cacheable-call count, not all calls
    assert anchor["reproduced_calls"] == 2275


@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus not present")
def test_corpus_has_one_call_per_turn_so_same_turn_is_unidentifiable(harness) -> None:
    report = harness.build_report(CORPUS, harness.load_records(CORPUS))
    st = report["same_turn"]
    assert st["turns_with_multiple_calls"] == 0
    assert st["same_turn_duplicate_calls"] == 0
    assert st["identifiable"] is False
    assert list(st["batch_size_histogram"]) == ["1"]


@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus not present")
def test_bounded_capacity_degrades_gracefully(harness) -> None:
    report = harness.build_report(CORPUS, harness.load_records(CORPUS))
    run = report["lru"]["sound"]["run"]
    assert run["8"]["hits"] < run["128"]["hits"] <= run["0"]["hits"]
    # the uncapped reference is exactly the run-scope anchor
    assert run["0"]["hits"] == report["anchor"]["reproduced_hits"]


@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus not present")
def test_static_ceiling_bounds_every_hot_set_row(harness) -> None:
    """No static list can hit a key that never appeared in the train half."""
    report = harness.build_report(CORPUS, harness.load_records(CORPUS))
    diag = report["split_half_diagnostics"]["per_source"]
    for mode in harness.MODES:
        for trainer in ("global", "per_source"):
            for n in harness.HOT_SET_SIZES:
                for source, row in report["hot_set"][mode][trainer][str(n)][
                    "per_source"
                ].items():
                    assert row["hits"] <= diag[source]["test_calls_with_train_key"]
    assert report["split_half_diagnostics"]["per_source"]["sonnet-35-new-airline"][
        "static_ceiling"
    ] == 0.0


@pytest.mark.skipif(not CORPUS.exists(), reason="tau-bench corpus not present")
def test_cannot_emulate_block_names_the_unemulable_baselines(harness) -> None:
    report = harness.build_report(CORPUS, harness.load_records(CORPUS))
    names = " ".join(item["baseline"].lower() for item in report["cannot_emulate"])
    assert "semantic" in names
    assert "prefix cache" in names
    for item in report["cannot_emulate"]:
        assert len(item["why_not"]) > 80, "a limitation needs an argument, not a label"
