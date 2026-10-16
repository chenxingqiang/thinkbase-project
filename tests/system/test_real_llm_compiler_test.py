"""Mock-backend tests for the compiler/hot-path real-gateway harness.

The harness (``examples/paper_real_llm_compiler_test.py``) validates:

* W1 -- byte-identity of the final artifacts between the unoptimized
  arm and the compiled + hot-path-adopted arm on the same backend, with
  the shared-cache protocol making the property exact (arm (b) must hit
  every key arm (a) warmed -- zero new model calls -- and reproduce
  every artifact byte-for-byte);
* W2 -- a priced analytical baseline (``compute_cost``) over measured
  token counts, labeled MODELED-on-measured.

These tests pin those properties on the deterministic mock backend
ONLY: no network, no live gateway.  The real-backend path is exercised
by the harness itself when the gateway is reachable (its round-trip
budget is reported in its output JSON).
"""

import asyncio
import json
import socket
import sys
from pathlib import Path

import pytest

# Make the harness module importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from paper_real_llm_compiler_test import (  # noqa: E402
    DeterministicMockBackend,
    compute_cost,
    main,
    run_validation,
)


@pytest.fixture(scope="module")
def results():
    """One shared-cache-protocol run: 24 Zipf(1.5) draws, 20-task universe."""
    return asyncio.run(run_validation(
        DeterministicMockBackend(),
        tasks=24, seeds=1, zipf_levels=[1.5], population=20,
        hot_threshold=3, arm_b_cache="shared", seed_base=1000,
        price_in_per_mt=3.0, price_out_per_mt=15.0,
        cache_read_rate=0.1, wall_measured=False,
    ))


@pytest.fixture(scope="module")
def cell(results):
    return results["data"][0]


class TestByteIdentityAcrossArms:
    def test_shared_protocol_all_realizations_identical(self, cell):
        ident = cell["byte_identity"]
        assert ident["all_realizations_identical"] is True
        assert ident["mismatched_realizations"] == []

    def test_compiled_arm_makes_zero_new_calls(self, cell):
        """Zero cache misses: the optimized stream computed identical
        keys (hence identical prompts) as the unoptimized reference."""
        arm_b = cell["arm_b_compiled_hotpath"]
        assert arm_b["model_calls"] == 0
        assert arm_b["calls_match_expected"] is True
        assert arm_b["expected_model_calls"] == 0
        assert arm_b["cache_misses"] == 0
        assert arm_b["cache_hit_rate"] == 1.0

    def test_independent_caches_also_identical_on_mock(self):
        """Both arms cold: on the deterministic mock the byte-identity
        holds even without the shared-cache mediation."""
        res = asyncio.run(run_validation(
            DeterministicMockBackend(),
            tasks=12, seeds=1, zipf_levels=[1.0], population=12,
            hot_threshold=2, arm_b_cache="independent", seed_base=1000,
            price_in_per_mt=3.0, price_out_per_mt=15.0,
            cache_read_rate=0.1, wall_measured=False,
        ))
        c = res["data"][0]
        assert c["byte_identity"]["all_realizations_identical"] is True
        # both arms ran cold and paid for the same distinct keys
        a, b = c["arm_a_unoptimized"], c["arm_b_compiled_hotpath"]
        assert a["model_calls"] > 0
        assert b["model_calls"] == a["model_calls"]
        assert b["model_calls"] == c["distinct_cache_keys_expected"]

    def test_artifact_hashes_match_across_arms(self, cell):
        a = cell["arm_a_unoptimized"]["per_realization"]
        b = cell["arm_b_compiled_hotpath"]["per_realization"]
        assert len(a) == len(b) == 24
        for ra, rb in zip(a, b):
            assert ra["artifact_sha256"] == rb["artifact_sha256"]


class TestCacheAccounting:
    def test_cache_arm_makes_fewer_calls_than_draws(self, cell):
        """Under skew, repeats recur and the cache arm pays only for
        first occurrences."""
        arm_a = cell["arm_a_unoptimized"]
        assert arm_a["model_calls"] < cell["draws"]
        assert arm_a["model_calls"] == cell["distinct_cache_keys_expected"]
        assert arm_a["calls_match_expected"] is True
        assert cell["draws"] > cell["distinct_cache_keys_expected"]

    def test_repeats_make_zero_calls(self, cell):
        per = cell["arm_a_unoptimized"]["per_realization"]
        repeats = [r for r in per if r["expected_calls"] == 0]
        firsts = [r for r in per if r["expected_calls"] == 1]
        assert repeats, "the skewed workload must contain repeats"
        assert all(r["calls"] == 0 for r in repeats)
        assert all(r["calls"] == 1 for r in firsts)
        assert per[0]["calls"] == 1  # the very first draw is a miss
        # the expected pattern is computed without any backend and the
        # real run must reproduce it exactly
        assert ([r["calls"] for r in per]
                == [r["expected_calls"] for r in per])

    def test_priced_mode_call_counts(self, cell):
        pc = cell["priced_comparison"]
        modes = pc["modes"]
        assert modes["no_opt"]["calls"] == cell["draws"]
        assert modes["agentisa_cache"]["calls"] == cell["distinct_cache_keys_expected"]
        assert modes["provider_prompt_cache"]["calls"] == cell["draws"]
        # strict ordering on this workload (repeats exist)
        assert (modes["no_opt"]["cost_usd"]
                > modes["provider_prompt_cache"]["cost_usd"]
                > modes["agentisa_cache"]["cost_usd"])

    def test_priced_comparison_is_labeled_modeled(self, cell):
        pc = cell["priced_comparison"]
        assert pc["label"] == "MODELED-on-measured"
        assert pc["modes"]["provider_prompt_cache"]["label"] == "MODELED-on-measured"
        assert "0.1" in pc["prices"]["note"]
        assert "out of scope" in pc["gptcache"]


class TestHotPathAdoption:
    def test_adoption_occurred_and_execution_counts_drop(self, cell):
        arm_b = cell["arm_b_compiled_hotpath"]
        # threshold=3 -> draws 1-4 on the full 7-step stream, adoption at
        # draw 5 -> [7]*4 + [6]*20
        assert arm_b["optimized_form_adopted_at_realization"] == 5
        assert arm_b["instructions_per_realization"] == [7] * 4 + [6] * 20
        assert arm_b["stream_eliminated_by_hot_path"] == ["check_format"]
        arm_a = cell["arm_a_unoptimized"]
        assert arm_a["instructions_per_realization"] == [7] * 24
        assert arm_b["instruction_executions"] < arm_a["instruction_executions"]

    def test_unoptimized_arm_never_adopts(self, cell):
        arm_a = cell["arm_a_unoptimized"]
        assert arm_a["optimized_form_adopted_at_realization"] is None
        assert arm_a["stream_eliminated_by_hot_path"] == []

    def test_static_compile_preserves_dataflow(self, cell):
        sc = cell["static_compile"]
        assert sc["pass"] is True
        assert sc["n"] == 7
        assert sc["NONE"]["effective"] == 7
        assert sc["AGGRESSIVE"]["effective"] < 7
        assert sc["AGGRESSIVE"]["dataflow_preserved"] is True
        assert sc["NONE"]["dataflow_preserved"] is True

    def test_verdict_all_pass(self, results):
        assert results["all_pass"] is True
        assert all(v is not False for v in results["verdict"].values())
        assert results["verdict"]["hot_path_adopted"] is True
        assert (results["verdict"]
                ["static_compile_dataflow_preserved"] is True)


class TestComputeCostArithmetic:
    """Hand-computed example: draws a, a, b with measured per-key tokens.

    With price_in=$2/M, price_out=$4/M, cache_read_rate=0.1:
      no_opt        : in 100+100+50=250, out 20+20+10=50 ->
                      cost (250*2 + 50*4)/1e6 = 0.0007
      agentisa_cache: first occurrences only: in 150, out 30 ->
                      cost (150*2 + 30*4)/1e6 = 0.00042
      provider_prompt_cache (MODELED): all draws call; input billed as
                      first 150 full + repeat 100 at 0.1 -> 160
                      equivalent, out 50 -> cost (160*2 + 50*4)/1e6
                      = 0.00052
    """

    TOKENS = {
        "draw_keys": ["a", "a", "b"],
        "per_key_tokens": {"a": (100, 20), "b": (50, 10)},
    }

    def test_no_opt(self):
        r = compute_cost(self.TOKENS, "no_opt", price_in_per_mt=2.0,
                         price_out_per_mt=4.0, cache_read_rate=0.1)
        assert r["calls"] == 3
        assert r["input_tokens_full_price"] == 250
        assert r["input_tokens_cache_read"] == 0
        assert r["output_tokens"] == 50
        assert r["cost_usd"] == pytest.approx(0.0007)

    def test_agentisa_cache(self):
        r = compute_cost(self.TOKENS, "agentisa_cache", price_in_per_mt=2.0,
                         price_out_per_mt=4.0, cache_read_rate=0.1)
        assert r["calls"] == 2
        assert r["input_tokens_full_price"] == 150
        assert r["output_tokens"] == 30
        assert r["cost_usd"] == pytest.approx(0.00042)

    def test_provider_prompt_cache_modeled(self):
        r = compute_cost(self.TOKENS, "provider_prompt_cache",
                         price_in_per_mt=2.0, price_out_per_mt=4.0,
                         cache_read_rate=0.1)
        assert r["calls"] == 3
        assert r["input_tokens_full_price"] == 150
        assert r["input_tokens_cache_read"] == 100
        assert r["input_tokens_billed_equivalent"] == pytest.approx(160.0)
        assert r["output_tokens"] == 50
        assert r["cost_usd"] == pytest.approx(0.00052)
        assert r["label"] == "MODELED-on-measured"

    def test_cache_read_rate_is_a_parameter(self):
        r = compute_cost(self.TOKENS, "provider_prompt_cache",
                         price_in_per_mt=2.0, price_out_per_mt=4.0,
                         cache_read_rate=0.5)
        assert r["input_tokens_billed_equivalent"] == pytest.approx(200.0)
        assert r["cost_usd"] == pytest.approx(0.0006)

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            compute_cost(self.TOKENS, "gptcache")


class TestCLIMockEndToEnd:
    def test_backend_mock_runs_end_to_end_tiny_no_network(
            self, tmp_path, monkeypatch):
        """The CLI must run the full tiny-scale protocol on the mock
        backend with outbound networking disabled at the socket layer.

        IPv4/IPv6 socket creation is blocked (any TCP connection, e.g.
        to the gateway, is impossible); AF_UNIX sockets stay available
        because the asyncio event loop uses a Unix-domain socketpair
        for its internal self-pipe.
        """
        real_socket = socket.socket

        def guarded_socket(family=socket.AF_INET, type=socket.SOCK_STREAM,
                           proto=0, fileno=None):
            if family in (socket.AF_INET, socket.AF_INET6):
                raise AssertionError(
                    "network access attempted during a mock-backend test")
            return real_socket(family, type, proto, fileno)

        monkeypatch.setattr(socket, "socket", guarded_socket)
        monkeypatch.setattr(socket, "create_connection",
                            lambda *a, **k: (_ for _ in ()).throw(
                                AssertionError(
                                    "network access attempted during a "
                                    "mock-backend test")))

        out = tmp_path / "compiler_mock.json"
        rc = main(["--backend", "mock", "--scale", "tiny",
                   "--json", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["backend"] == "mock"
        assert data["all_pass"] is True
        assert data["verdict"]["byte_identical_all_realizations"] is True
        assert data["verdict"]["arm_b_zero_new_calls_when_shared"] is True
        assert len(data["data"]) == 1  # tiny = 50 x 1 seed x 1 Zipf level
        assert data["data"][0]["draws"] == 50
        assert "preflight" not in data  # preflight is real-backend only
        # every priced mode is present and the modeled one is labeled
        modes = data["data"][0]["priced_comparison"]["modes"]
        assert set(modes) == {"no_opt", "agentisa_cache",
                              "provider_prompt_cache"}


class TestMockByteReproducibility:
    def test_two_mock_runs_produce_identical_json(self):
        """Mock-mode output is byte-reproducible: no wall-clock value
        enters the result, so two runs serialize identically."""
        runs = []
        for _ in range(2):
            res = asyncio.run(run_validation(
                DeterministicMockBackend(),
                tasks=10, seeds=1, zipf_levels=[1.0], population=10,
                hot_threshold=2, arm_b_cache="shared", seed_base=1000,
                price_in_per_mt=3.0, price_out_per_mt=15.0,
                cache_read_rate=0.1, wall_measured=False,
            ))
            runs.append(json.dumps(res, indent=1, sort_keys=True))
        assert runs[0] == runs[1]
