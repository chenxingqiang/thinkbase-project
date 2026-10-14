"""Tests for the real-LLM benchmark harness (no gateway required).

The gateway itself is exercised by actually running
``examples/benchmarks/isa_real_llm_bench.py`` against a live endpoint; here
we validate the parts that must be correct independent of the network:

* the cache protocol of ``RealLLMInstruction`` (hits never touch the
  backend, misses are stored),
* the claim that the hit/miss sequence is backend-independent (same seed
  => same workload => exactly computable hit rate),
* ``RealLLMBackend`` bookkeeping and error handling via a mocked
  httpx transport (usage accounting, 4xx aborts without retry).
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import httpx
import pytest

# Make the benchmark scripts importable (they live in examples/benchmarks).
BENCH_DIR = Path(__file__).resolve().parents[2] / "examples" / "benchmarks"
sys.path.insert(0, str(BENCH_DIR))

from isa_real_llm_bench import (  # noqa: E402
    RealLLMBackend,
    RealLLMInstruction,
    zipf_workload,  # re-exported via isa_microbench import
)


class FakeBackend:
    """Minimal stand-in matching the backend protocol of the instruction."""

    def __init__(self) -> None:
        self.calls: List[str] = []

    async def infer(self, payload: str) -> str:
        self.calls.append(payload)
        return f"resp::{payload}"


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------
# Cache protocol of RealLLMInstruction
# --------------------------------------------------------------------------

def test_cache_hit_skips_backend():
    backend = FakeBackend()
    instr = RealLLMInstruction(backend, "op", use_cache=True)  # type: ignore[arg-type]

    first = _run(instr.execute({"value": 1}))
    second = _run(instr.execute({"value": 1}))

    assert first == second
    assert len(backend.calls) == 1  # second execution was a cache hit
    assert instr.metrics.cache_hits == 1
    assert instr.metrics.total_executions == 2


def test_different_params_are_distinct_cache_entries():
    backend = FakeBackend()
    instr = RealLLMInstruction(backend, "op", use_cache=True)  # type: ignore[arg-type]

    _run(instr.execute({"value": 1}))
    _run(instr.execute({"value": 2}))

    assert len(backend.calls) == 2
    assert instr.metrics.cache_hits == 0


def test_no_cache_config_hits_backend_every_time():
    backend = FakeBackend()
    instr = RealLLMInstruction(backend, "op", use_cache=False)  # type: ignore[arg-type]

    _run(instr.execute({"value": 1}))
    _run(instr.execute({"value": 1}))

    assert len(backend.calls) == 2


def test_cache_key_includes_instruction_name():
    b1, b2 = FakeBackend(), FakeBackend()
    i1 = RealLLMInstruction(b1, "op_a", use_cache=True)  # type: ignore[arg-type]
    i2 = RealLLMInstruction(b2, "op_b", use_cache=True)  # type: ignore[arg-type]

    _run(i1.execute({"value": 1}))
    _run(i2.execute({"value": 1}))

    assert len(b1.calls) == 1 and len(b2.calls) == 1  # names differ => no sharing


# --------------------------------------------------------------------------
# Hit/miss sequence is backend-independent (exact expectation)
# --------------------------------------------------------------------------

def _expected_hit_rate(workload) -> float:
    seen: set = set()
    hits = 0
    for idx, params in workload:
        key = (idx, tuple(sorted(params.items())))
        if key in seen:
            hits += 1
        else:
            seen.add(key)
    return hits / len(workload)


def test_same_seed_same_workload():
    w1 = zipf_workload(random.Random(1000), 200, 50, 20, 1.0)
    w2 = zipf_workload(random.Random(1000), 200, 50, 20, 1.0)
    assert w1 == w2


def test_real_instruction_hit_rate_matches_expectation():
    """The instruction must reproduce the exactly computed hit rate."""
    workload = zipf_workload(random.Random(1000), 200, 50, 20, 1.0)
    expected = _expected_hit_rate(workload)

    backend = FakeBackend()
    instrs = [RealLLMInstruction(backend, f"i{i}", use_cache=True)  # type: ignore[arg-type]
              for i in range(50)]
    for idx, params in workload:
        _run(instrs[idx].execute(params))

    total = sum(i.metrics.total_executions for i in instrs)
    hits = sum(i.metrics.cache_hits for i in instrs)
    assert hits / total == pytest.approx(expected, abs=1e-9)
    assert len(backend.calls) == len(workload) - hits  # calls == misses


# --------------------------------------------------------------------------
# RealLLMBackend bookkeeping with a mocked transport
# --------------------------------------------------------------------------

def _message_response(output_tokens: int = 7, text: str = "pong") -> httpx.Response:
    return httpx.Response(200, json={
        "id": "msg_test", "role": "assistant",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 12, "output_tokens": output_tokens},
    })


def _backend_with(transport: httpx.MockTransport) -> RealLLMBackend:
    return RealLLMBackend(
        base_url="https://gateway.test", api_key="k", ca_cert=None,
        model="m", max_tokens=64, transport=transport,
    )


def test_backend_counts_tokens_and_latencies():
    backend = _backend_with(httpx.MockTransport(
        lambda request: _message_response()))
    result = asyncio.run(backend.infer("hi"))
    asyncio.run(backend.close())

    assert result == "pong"
    assert backend.api_calls == 1
    assert backend.input_tokens == 12
    assert backend.output_tokens == 7
    assert len(backend.latencies_ms) == 1


def test_backend_4xx_aborts_without_retry():
    attempts: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, json={"error": {"message": "bad"}})

    backend = _backend_with(httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="gateway rejected request"):
        asyncio.run(backend.infer("hi"))
    asyncio.run(backend.close())

    assert len(attempts) == 1  # no retries on client errors
    assert backend.api_calls == 0


def test_backend_retries_then_succeeds_on_5xx():
    attempts: List[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(503, json={"error": "busy"})
        return _message_response()

    backend = _backend_with(httpx.MockTransport(handler))
    result = asyncio.run(backend.infer("hi"))
    asyncio.run(backend.close())

    assert result == "pong"
    assert len(attempts) == 2
    assert backend.api_calls == 1  # only the success is counted


def test_backend_gives_up_after_three_5xx():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "busy"})

    backend = _backend_with(httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="unreachable after 3 attempts"):
        asyncio.run(backend.infer("hi"))
    asyncio.run(backend.close())
    assert backend.api_calls == 0


def test_backend_skips_thinking_blocks():
    body = {
        "content": [
            {"type": "thinking", "thinking": "internal reasoning"},
            {"type": "text", "text": "answer-part-1"},
            {"type": "text", "text": "answer-part-2"},
        ],
        "usage": {"input_tokens": 1, "output_tokens": 2},
    }
    backend = _backend_with(httpx.MockTransport(
        lambda request: httpx.Response(200, json=body)))
    result = asyncio.run(backend.infer("hi"))
    asyncio.run(backend.close())
    assert result == "answer-part-1answer-part-2"


def test_backend_stats_snapshot():
    backend = _backend_with(httpx.MockTransport(lambda r: _message_response()))
    asyncio.run(backend.infer("a"))
    snap = backend.stats()
    assert snap.api_calls == 1 and snap.input_tokens == 12
    # snapshot must be independent of later mutations
    asyncio.run(backend.infer("b"))
    asyncio.run(backend.close())
    assert snap.api_calls == 1
    assert backend.stats().api_calls == 2
