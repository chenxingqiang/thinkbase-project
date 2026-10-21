"""Offline tests for the hot-path EMA stress harness.

``examples/benchmarks/hotpath_ema_stress.py`` compares the shipped
exact-identity detector against the implemented EMA detector (D.4)
across seeded diverse trajectory streams, measures adoption latency
through ``OptimizationManager``, and measures LRU eviction under
``max_tracked_paths`` caps.  These tests pin the harness mechanics
offline (no network, no live backend): determinism, aggregate
computation from raw events, the core exact-vs-EMA contrast, adoption,
cap bounds, LRU eviction order, re-accumulation after eviction, and the
end-to-end CLI writing a provenance-complete results JSON.
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import pytest

# Make the benchmark scripts importable (they live in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import hotpath_ema_stress as stress  # noqa: E402
from hotpath_ema_stress import (  # noqa: E402
    OracleDetector,
    build_instr_for,
    make_detector,
    run_adoption_cell,
    run_cap_cell,
    run_cap_rescue_cell,
    run_detection_cell,
    run_all,
    stable_metrics,
    stream_bursty,
    stream_diverse_rotated,
    stream_pure_diversity,
    stream_warmup_heavy,
)


def _detection(stream):
    return run_detection_cell(stream, build_instr_for(stream))


# ---------------------------------------------------------------------------
# Determinism and aggregate integrity.
# ---------------------------------------------------------------------------

def test_run_all_is_deterministic_per_seed():
    a = run_all(seeds=2, quick=True, cap_sweep=True)
    b = run_all(seeds=2, quick=True, cap_sweep=True)
    # Wall-clock is run metadata, not a result; everything else must
    # reproduce byte-for-byte from the same seeds.
    a["run_meta"].pop("wall_seconds")
    b["run_meta"].pop("wall_seconds")
    assert a == b


def test_run_all_invariants_hold():
    report = run_all(seeds=1, quick=True, cap_sweep=True)
    assert report["invariants_passed"]
    assert report["headline"]["cap_sweep_all_bounds_respected"] is True


def test_aggregates_match_recomputed_statistics():
    report = run_all(seeds=5, quick=False, cap_sweep=False)
    rot = report["scenarios"]["diverse_rotated"]
    # Recompute the promoted-count mean from the raw per-seed events.
    raw_counts = [len(v["ema_events"]) for v in rot["per_seed"].values()]
    assert raw_counts == [3, 3, 3, 3, 3]
    assert rot["ema_promoted"]["mean"] == statistics.fmean(raw_counts)
    assert rot["ema_promoted"]["stdev"] == statistics.stdev(raw_counts)
    # Recall aggregates come from the same raw events.
    recalls = []
    for v in rot["per_seed"].values():
        oracle_set = {e["path"] for e in v["oracle_events"]}
        ema_set = {e["path"] for e in v["ema_events"]}
        recalls.append(len(ema_set & oracle_set) / len(oracle_set))
    assert rot["ema_recall_vs_oracle"]["mean"] == statistics.fmean(recalls)


def test_oracle_reads_only_the_stream():
    oracle = OracleDetector()
    for path in ["a", "b", "a", "a"]:
        oracle.record(path)
    # a: 3/4 cumulative share > 0.3 and repeated -> hot; b never repeats.
    assert oracle.hot_paths == {"a"}


# ---------------------------------------------------------------------------
# Core exact-vs-EMA contrast on diverse streams.
# ---------------------------------------------------------------------------

def test_diverse_rotation_ema_promotes_exact_does_not():
    raw = _detection(stream_diverse_rotated(None))
    ema_paths = {e[0] for e in raw["events"]["ema"]}
    assert ema_paths == {"v1", "v2", "v3"}
    assert raw["events"]["exact"] == []
    # Every variant stays far below the absolute threshold (40 < 100).
    assert all(e[2] <= 40 for e in raw["events"]["ema"])


def test_pure_diversity_never_triggers_either_detector():
    raw = _detection(stream_pure_diversity(None))
    assert raw["events"]["ema"] == []
    assert raw["events"]["exact"] == []
    assert raw["events"]["oracle"] == []


def test_promotion_events_record_latency():
    raw = _detection(stream_diverse_rotated(None))
    for path, at_recording, recordings in raw["events"]["ema"]:
        assert 1 <= at_recording <= raw["stream_len"]
        assert recordings >= 2  # cold-start guard: repeat required


# ---------------------------------------------------------------------------
# Adoption latency through the manager.
# ---------------------------------------------------------------------------

def test_bursty_adoption_ema_only():
    stream = stream_bursty(None)
    instr_for = build_instr_for(stream)
    ema = run_adoption_cell(stream, instr_for, "ema")
    exact = run_adoption_cell(stream, instr_for, "exact")
    assert exact["detection_index"] is None
    assert exact["first_adoption_index"] is None
    # EMA detects the burst and the manager serves the shrunken trace
    # from the promotion recording on: adoption == detection.
    assert ema["detection_index"] == 13
    assert ema["first_adoption_index"] == 13
    assert ema["full_form_executions_before_adoption"] == 12
    assert ema["full_trace_len"] == 4
    assert ema["adopted_trace_len"] == 1
    assert ema["served_len_at_adoption"] == 1


def test_warmup_heavy_burst_stays_cold_at_default_cap():
    raw = _detection(stream_warmup_heavy(None))
    assert raw["events"]["ema"] == []
    # The oracle (cumulative share) does catch the burst: the miss is a
    # property of the tracked-sum denominator, not of the stream.
    assert [e[0] for e in raw["events"]["oracle"]] == ["burst"]


# ---------------------------------------------------------------------------
# Cap eviction: bounds, LRU order, state cleanup, re-accumulation.
# ---------------------------------------------------------------------------

def _flood_stream(n):
    return [f"p{i}" for i in range(n)]


def test_cap_bounds_every_per_path_mapping():
    stream = _flood_stream(50)
    instr = build_instr_for(stream)
    raw = run_cap_cell(stream, cap=10, instr_for=instr)
    for mode in ("exact", "ema"):
        row = raw[mode]
        assert row["bound_respected"] is True
        assert row["final_counts_size"] == 10
        assert row["final_stats_size"] == 10
        assert row["final_recency_size"] == 10
        assert row["evictions"] == 40


def test_cap_evicts_least_recently_recorded_lru():
    # Refresh "a" so "b" becomes the least recently recorded path.
    stream = ["a", "b", "c", "a", "d"]
    instr = build_instr_for(stream)
    raw = run_cap_cell(stream, cap=3, instr_for=instr)
    for mode in ("exact", "ema"):
        assert raw[mode]["bound_respected"] is True
    # Independently replay the reference LRU over the same stream:
    # recency order after the stream is c (pos 3) < a (refreshed, pos 4)
    # < d (pos 5), so "b" is evicted and "a" survives its refresh.
    reference = OrderedDict()
    for path in stream:
        if path in reference:
            reference.move_to_end(path)
        else:
            reference[path] = None
        while len(reference) > 3:
            reference.popitem(last=False)
    assert list(reference) == ["c", "a", "d"]
    # The detector's own recency register must match the reference order
    # (oldest-first), proving LRU rather than FIFO eviction.
    det = make_detector("ema", cap=3)
    for path in stream:
        det.record_execution(path, build_instr_for([path])[path],
                             stable_metrics())
    assert list(det._path_recency) == ["c", "a", "d"]


def test_evicted_path_state_removed_and_reaccumulates():
    det = make_detector("ema", cap=2)
    instr = build_instr_for(["hot"])
    for _ in range(5):
        det.record_execution("hot", instr["hot"], stable_metrics())
    assert "hot" in det.hot_paths
    for i in range(2):
        det.record_execution(f"other{i}", instr["hot"], stable_metrics())
    # Evicted from every per-path mapping, hot status included.
    assert "hot" not in det.execution_counts
    assert "hot" not in det.ema_counters
    assert "hot" not in det.performance_stats
    assert "hot" not in det.hot_paths
    # Re-recording starts a fresh EMA counter (single observation).
    det.record_execution("hot", instr["hot"], stable_metrics())
    assert det.ema_counters["hot"] == pytest.approx(1.0)
    assert "hot" not in det.hot_paths


def test_cap_rescue_small_cap_restores_burst_detection():
    stream = stream_warmup_heavy(None)
    instr = build_instr_for(stream)
    big = run_cap_rescue_cell(stream, cap=1000, instr_for=instr)
    small = run_cap_rescue_cell(stream, cap=10, instr_for=instr)
    assert big["ema_burst_detection_index"] is None
    assert big["ema_final_denominator"] > 80  # stale warm-up keeps it up
    assert small["ema_burst_detection_index"] is not None
    assert small["ema_final_denominator"] < big["ema_final_denominator"]
    assert small["evictions"] > 0


# ---------------------------------------------------------------------------
# End-to-end CLI: provenance-complete results file.
# ---------------------------------------------------------------------------

def test_main_writes_provenance_complete_json(tmp_path, monkeypatch):
    out = tmp_path / "hotpath_results.json"
    code = stress.main(["--quick", "--output", str(out)])
    assert code == 0
    report = json.loads(out.read_text())
    prov = report["provenance"]
    assert prov["git_rev"]
    assert prov["timestamp_utc"]
    assert prov["rng_seeds"] == [1000]
    assert prov["detector_config"]["ema_window"] == 10
    assert prov["scale"]["quick_mode"] is True
    cov = report["coverage"]
    assert cov["i3_hotpath_ema"] == "shipped"
    assert cov["cap_sweep_run"] is True
    assert "diverse_rotated" in cov["scenarios_run"]
    assert report["headline"]["invariants_passed"] >= 10
    assert report["headline"]["diverse_rotated_ema_promoted_mean"] == 3.0
    assert report["headline"]["diverse_rotated_exact_promoted_mean"] == 0.0
