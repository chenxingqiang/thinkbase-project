"""Multi-seed / multi-backend orchestration tests for the cache bench (mock).

``examples/benchmarks/isa_real_llm_bench.py`` selects its LLM backend with
``--backend mock|gateway`` (default ``mock``).  The mock is a deterministic
offline stand-in with the same accounting surface as the gateway backend,
which makes the multi-seed orchestration -- ``--seeds N`` runs N seeded
workloads per Zipf level and aggregates them to mean/stdev -- exercisable
offline and in CI, removing the single-seed coverage limitation that came
from every seed costing a full real-gateway pass.

The gateway arm keeps the pre-selector script's behavior exactly: same
``RealLLMBackend`` construction arguments, same preflight, same API-key
requirement.  These tests pin the MOCK path only (per the harness's
honesty rules: no network, no live gateway, no real API key); the
gateway's HTTP behavior is covered by
``tests/benchmarks/test_real_llm_bench.py`` via a mocked httpx transport
and by actually running the harness against a reachable gateway.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
from pathlib import Path

import pytest

# Make the benchmark scripts importable (they live in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

import isa_real_llm_bench  # noqa: E402  (module object, for monkeypatching)
from isa_real_llm_bench import (  # noqa: E402
    MOCK_CALL_LATENCY_MS,
    BackendStats,
    DeterministicMockBackend,
    RealLLMBackend,
    RealLLMInstruction,
    _account_tokens,
    build_backend,
    main,
    run_real_cache_experiment,
    zipf_workload,  # re-exported via isa_microbench import
)


def _run(coro):
    return asyncio.run(coro)


def _bench_args(**overrides) -> argparse.Namespace:
    """Minimal Namespace accepted by ``run_real_cache_experiment``."""
    base = dict(
        backend="mock", base_url="https://gateway.test", api_key="",
        ca_cert=None, model="claude-glm-5.3-flash", max_tokens=64,
        seeds=1, tasks=24, zipf=[1.0],
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _experiment(**overrides) -> dict:
    return _run(run_real_cache_experiment(_bench_args(**overrides)))


def _without_wall(obj):
    """Drop every wall-clock measurement (mock runs vary only there)."""
    if isinstance(obj, dict):
        return {k: _without_wall(v) for k, v in obj.items() if "wall" not in k}
    if isinstance(obj, list):
        return [_without_wall(v) for v in obj]
    return obj


def _exact_hit_rate(workload) -> float:
    """Backend-independent expected hit rate (same formula as the bench)."""
    seen, hits = set(), 0
    for idx, params in workload:
        key = (idx, tuple(sorted(params.items())))
        if key in seen:
            hits += 1
        else:
            seen.add(key)
    return hits / len(workload)


# --------------------------------------------------------------------------
# DeterministicMockBackend: the offline accounting surface
# --------------------------------------------------------------------------

class TestDeterministicMockBackend:
    def test_reply_is_pure_function_of_payload(self):
        b1, b2 = DeterministicMockBackend(), DeterministicMockBackend()
        payload = "i3:[('value', 7)]"
        assert _run(b1.infer(payload)) == _run(b2.infer(payload))

    def test_accounting_per_call(self):
        backend = DeterministicMockBackend()
        payload = "i0:[('value', 1)]"
        reply = _run(backend.infer(payload))
        assert backend.api_calls == 1
        assert backend.input_tokens == _account_tokens(payload)
        assert backend.output_tokens == _account_tokens(reply)
        assert backend.wall_ms == MOCK_CALL_LATENCY_MS
        assert backend.latencies_ms == [MOCK_CALL_LATENCY_MS]

    def test_token_source_is_labeled_as_mock(self):
        assert "mock" in DeterministicMockBackend().token_source

    def test_simulated_latency_is_charged_never_slept(self):
        backend = DeterministicMockBackend()
        start = time.perf_counter()
        for _ in range(100):
            _run(backend.infer("payload"))
        elapsed = time.perf_counter() - start
        assert backend.wall_ms == 100 * MOCK_CALL_LATENCY_MS  # simulated clock
        assert elapsed < 1.0  # 100 real calls never slept the simulated 8 s

    def test_stats_snapshot_is_independent(self):
        backend = DeterministicMockBackend()
        _run(backend.infer("a"))
        snap = backend.stats()
        _run(backend.infer("b"))
        assert isinstance(snap, BackendStats)
        assert snap.api_calls == 1  # snapshot unchanged by later calls
        assert backend.stats().api_calls == 2

    def test_counters_reset_by_attribute_assignment(self):
        """The experiment resets counters with direct attribute writes."""
        backend = DeterministicMockBackend()
        _run(backend.infer("a"))
        backend.api_calls = backend.wall_ms = 0
        backend.input_tokens = backend.output_tokens = 0
        backend.latencies_ms = []
        assert backend.stats() == BackendStats(0, 0.0, 0, 0, [])


# --------------------------------------------------------------------------
# Backend selection (--backend mock|gateway)
# --------------------------------------------------------------------------

class TestBackendSelection:
    def test_mock_needs_no_api_key(self):
        backend = build_backend(_bench_args(api_key=""))
        assert isinstance(backend, DeterministicMockBackend)

    def test_gateway_without_api_key_aborts(self):
        with pytest.raises(SystemExit, match="API key required"):
            build_backend(_bench_args(backend="gateway", api_key=""))

    def test_gateway_constructs_real_llm_backend(self):
        backend = build_backend(_bench_args(backend="gateway", api_key="k",
                                           base_url="https://gw.test"))
        assert isinstance(backend, RealLLMBackend)
        assert backend.base_url == "https://gw.test"
        assert backend.api_key == "k"
        _run(backend.close())

    def test_gateway_experiment_keeps_existing_construction(self, monkeypatch):
        """The gateway arm must be wired exactly as before the selector."""
        recorded: dict = {}

        class GatewayStub:
            token_source = "gateway usage field (real)"

            def __init__(self, base_url, api_key, ca_cert, model, max_tokens):
                recorded.update(base_url=base_url, api_key=api_key,
                                ca_cert=ca_cert, model=model,
                                max_tokens=max_tokens)
                self.api_calls = self.input_tokens = self.output_tokens = 0
                self.wall_ms = 0.0
                self.latencies_ms = []

            def stats(self) -> BackendStats:
                return BackendStats(self.api_calls, self.wall_ms,
                                    self.input_tokens, self.output_tokens,
                                    list(self.latencies_ms))

            async def infer(self, payload: str) -> str:
                self.api_calls += 1
                self.wall_ms += 1.0
                self.latencies_ms.append(1.0)
                return "stub"

            async def close(self) -> None:
                pass

        monkeypatch.setattr(isa_real_llm_bench, "RealLLMBackend", GatewayStub)
        res = _experiment(backend="gateway", api_key="k", tasks=4)

        assert recorded == {
            "base_url": "https://gateway.test", "api_key": "k",
            "ca_cert": None, "model": "claude-glm-5.3-flash",
            "max_tokens": 64,
        }
        assert res["config"]["backend"] == "gateway"
        assert res["config"]["token_source"].startswith("gateway usage")


class TestCli:
    def test_default_backend_is_mock_and_defaults_unchanged(self, tmp_path):
        """Bare CLI: offline mock run; every previous flag default kept."""
        out = tmp_path / "out.json"
        main(["--output", str(out)])
        res = json.loads(out.read_text())

        assert res["config"]["backend"] == "mock"
        assert "mock" in res["config"]["token_source"]
        assert res["config"]["seeds"] == 2            # --seeds default
        assert res["config"]["n_tasks"] == 60         # --tasks default
        assert res["config"]["zipf_s"] == [0.5, 1.0, 1.5]
        assert res["config"]["max_tokens"] == 256
        assert res["config"]["model"] == "claude-glm-5.3-flash"
        assert [e["zipf_s"] for e in res["data"]] == [0.5, 1.0, 1.5]
        for entry in res["data"]:
            assert [s["seed"] for s in entry["per_seed"]] == [0, 1]
        assert isinstance(res["wall_time_s"], float)

    def test_gateway_cli_requires_api_key(self, monkeypatch, tmp_path):
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        out = tmp_path / "out.json"
        with pytest.raises(SystemExit, match="API key required"):
            main(["--backend", "gateway", "--output", str(out)])
        assert not out.exists()  # aborted before any run, like before

    def test_unknown_backend_is_rejected(self, tmp_path):
        with pytest.raises(SystemExit):
            main(["--backend", "bogus", "--output", str(tmp_path / "o.json")])


# --------------------------------------------------------------------------
# Multi-seed orchestration (--seeds N: seeded loop + mean/stdev aggregation)
# --------------------------------------------------------------------------

class TestMultiSeedOrchestration:
    def test_n_seeds_produce_n_seeded_workloads(self):
        res = _experiment(seeds=3)
        entry = res["data"][0]
        assert [s["seed"] for s in entry["per_seed"]] == [0, 1, 2]

    def test_seeds_cover_different_workloads(self):
        """The point of multi-seed: different seeds must vary the draw
        sequence (here their hit rates differ), not repeat one workload."""
        res = _experiment(seeds=3, tasks=40)
        rates = [s["hit_rate"] for s in res["data"][0]["per_seed"]]
        assert len(set(rates)) > 1

    def test_aggregation_reports_mean_and_stdev(self):
        res = _experiment(seeds=3, tasks=24)
        entry = res["data"][0]

        hits = [s["hit_rate"] for s in entry["per_seed"]]
        assert entry["hit_rate"]["mean"] == pytest.approx(
            statistics.mean(hits), abs=1e-4)
        assert entry["hit_rate"]["stdev"] == pytest.approx(
            statistics.stdev(hits), abs=1e-4)

        cached = [s["cached"]["api_calls"] for s in entry["per_seed"]]
        assert entry["api_calls_cached"]["mean"] == pytest.approx(
            statistics.mean(cached), abs=1e-4)
        assert entry["api_calls_cached"]["stdev"] == pytest.approx(
            statistics.stdev(cached), abs=1e-4)

        base = [s["baseline"]["api_calls"] for s in entry["per_seed"]]
        assert entry["api_calls_no_cache"]["mean"] == pytest.approx(
            statistics.mean(base), abs=1e-4)
        assert entry["api_calls_no_cache"]["stdev"] == pytest.approx(
            statistics.stdev(base), abs=1e-4)

    def test_single_seed_stdev_is_zero(self):
        res = _experiment(seeds=1)
        entry = res["data"][0]
        assert len(entry["per_seed"]) == 1
        for key in ("api_calls_no_cache", "api_calls_cached",
                    "reduction_ratio", "hit_rate", "expected_hit_rate"):
            assert entry[key]["stdev"] == 0.0

    def test_each_zipf_level_gets_its_own_seed_loop(self):
        res = _experiment(seeds=2, zipf=[0.5, 1.5])
        assert [e["zipf_s"] for e in res["data"]] == [0.5, 1.5]
        for entry in res["data"]:
            assert [s["seed"] for s in entry["per_seed"]] == [0, 1]

    def test_hit_rate_matches_backend_independent_expectation(self):
        """Honesty: the mock must reproduce the exactly computable hit
        rate for every seed (the hit/miss sequence is backend-free)."""
        res = _experiment(seeds=2, tasks=40)
        for s in res["data"][0]["per_seed"]:
            workload = zipf_workload(random.Random(1000 + s["seed"]),
                                    40, 50, 20, 1.0)
            expected = _exact_hit_rate(workload)
            assert s["expected_hit_rate"] == pytest.approx(expected, abs=1e-4)
            assert s["hit_rate"] == pytest.approx(expected, abs=1e-4)

    def test_cache_arm_reduces_calls_on_skewed_workload(self):
        res = _experiment(seeds=2, tasks=40)
        for s in res["data"][0]["per_seed"]:
            assert 0.0 < s["call_reduction"] < 1.0
            assert s["cached"]["api_calls"] < s["baseline"]["api_calls"]

    def test_mock_run_is_deterministic_across_runs(self):
        """Apart from real wall-clock fields, two identical runs must
        produce byte-identical results (template-accounted tokens, exact
        call counts, constant simulated latencies)."""
        a = _experiment(seeds=2, tasks=24)
        b = _experiment(seeds=2, tasks=24)
        assert _without_wall(a) == _without_wall(b)

    def test_mock_latencies_are_the_constant_simulated_charge(self):
        res = _experiment(seeds=1, tasks=8)
        s = res["data"][0]["per_seed"][0]
        assert s["cached"]["latency_ms_p50"] == MOCK_CALL_LATENCY_MS
        assert s["cached"]["latency_ms_p95"] == MOCK_CALL_LATENCY_MS

    def test_config_labels_backend_and_token_source(self):
        res = _experiment()
        assert res["config"]["backend"] == "mock"
        assert "mock" in res["config"]["token_source"]
        assert "template" in res["config"]["token_source"]
        assert "simulated" in res["config"]["backend_note"]


# --------------------------------------------------------------------------
# The mock is a drop-in backend for the experiment's instruction protocol
# --------------------------------------------------------------------------

class TestMockFitsInstructionProtocol:
    def test_real_llm_instruction_routes_through_mock(self):
        backend = DeterministicMockBackend()
        instrs = [RealLLMInstruction(backend, f"i{i}", use_cache=True)  # type: ignore[arg-type]
                  for i in range(4)]
        workload = [(1, {"value": 1}), (1, {"value": 1}),
                   (2, {"value": 3}), (1, {"value": 1})]
        for idx, params in workload:
            _run(instrs[idx].execute(params))

        total = sum(i.metrics.total_executions for i in instrs)
        hits = sum(i.metrics.cache_hits for i in instrs)
        assert total == 4 and hits == 2
        assert backend.api_calls == 2  # only the misses touched the backend
