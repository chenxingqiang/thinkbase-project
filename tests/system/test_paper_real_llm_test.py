"""Tests for the real-paper, real-model AgentISA e2e (gateway mocked).

The live gateway is exercised by actually running
``examples/paper_real_llm_test.py``; here the transport is mocked so the
battery's logic -- caching across repeats, adoption, retrieval
determinism, ELL export -- is verified without the network.  A
deterministic mock also lets the harness's determinism observation be
asserted positively (a deterministic backend MUST yield identical
answers across fresh stacks).
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, "examples")
sys.path.insert(0, "examples/benchmarks")

from paper_real_llm_test import (  # noqa: E402
    QUERIES, build_paper_rag_stream, chunk_paper, retrieve, run_battery,
)
from isa_real_llm_bench import RealLLMBackend  # noqa: E402

FAKE_PAPER = """
% comment line that must be stripped
This paper presents an agent instruction set architecture with a
result cache for LLM calls. The cache reduces repeated model calls
under skewed workloads by 59.5 percent at Zipf exponent 0.5.

The static compiler performs dead-code elimination guarded by a
dataflow liveness analysis, so every instruction whose output is
consumed downstream survives compilation. Semantic preservation is
argued in Appendix C via pre and post conditions.

The closed-loop trace optimizer records executed streams and adopts
an optimized trace once a hot-path threshold is crossed. Dataflow
validity of the compiled stream is asserted before adoption, which
safeguards the rewrite. Reductions of 34.2 percent were measured on
authored streams with dead code.
"""


def mock_backend(handler=None) -> RealLLMBackend:
    """RealLLMBackend over a MockTransport with canned responses."""
    call_state = {"n": 0}

    def default_handler(request: httpx.Request) -> httpx.Response:
        call_state["n"] += 1
        body = json.loads(request.content)
        # classify prompts ask for a JSON array; synthesize asks an answer
        prompt = body["messages"][0]["content"]
        if "JSON array" in prompt:
            text = "[0, 2]"
        else:
            text = ("[0] The paper says the cache cuts repeated calls "
                    "by 59.5 percent under Zipf 0.5. [1] Dead-code "
                    "elimination is guarded by liveness analysis.")
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 8},
            "stop_reason": "end_turn",
        })

    return RealLLMBackend(
        base_url="https://gateway.test", api_key="test-key",
        ca_cert=None, model="test-model", max_tokens=100,
        transport=httpx.MockTransport(handler or default_handler),
    )


class TestChunkAndRetrieve:
    # FAKE_PAPER alone fits in one chunk; repeat it to force multi-chunk
    BIG_PAPER = FAKE_PAPER + ("\n\n" + FAKE_PAPER) * 5

    def test_chunk_paper_is_deterministic(self):
        a = chunk_paper(self.BIG_PAPER)
        b = chunk_paper(self.BIG_PAPER)
        assert a == b and len(a) >= 2
        assert all("%" != c.lstrip()[:1] for c in a)  # comments stripped

    def test_retrieve_is_deterministic_and_bounded(self):
        chunks = chunk_paper(self.BIG_PAPER)
        r1 = retrieve(QUERIES[0], chunks)
        r2 = retrieve(QUERIES[0], chunks)
        assert r1 == r2
        assert 1 <= len(r1) <= 3
        assert all(isinstance(i, int) and 0 <= i < len(chunks) for i in r1)

    def test_retrieve_finds_relevant_chunk(self):
        chunks = chunk_paper(FAKE_PAPER)
        hits = retrieve("result cache hit rates Zipf", chunks)
        # the cache paragraph must be among the retrieved
        assert any("cache" in chunks[i] for i in hits)


class TestStreamShape:
    def test_stream_structure(self):
        stream = build_paper_rag_stream(mock_backend(), QUERIES)
        assert len(stream) == 7
        ids = [i.id for i in stream]
        assert ids[0] == "ingest_paper" and ids[-1] == "write_answers"
        llm = [i for i in stream if str(i.type).endswith("LLM_INTERACTION")]
        assert [i.id for i in llm] == ["classify_relevance",
                                       "synthesize_answer"]
        assert all(i.cache is not None for i in llm)
        guards = [i for i in stream if i.id.endswith("_guard")]
        assert len(guards) == 1 and guards[0].outputs == []
        assert ids[0].startswith("ingest_") and ids[-1].startswith("write_")


class TestBatteryWithMockedGateway:
    def test_full_battery_passes_and_caches(self, tmp_path: Path):
        """With a deterministic mocked gateway the whole battery passes:
        real calls on realization 1 only, zero afterwards, identical
        artifacts, adoption, and even identical fresh-stack answers."""
        args = argparse.Namespace(
            repeats=4, hot_threshold=2,
            trace=str(tmp_path / "trace.jsonl"),
        )
        backend = mock_backend()
        result = asyncio.run(run_battery(
            backend, args, paper_text=FAKE_PAPER))
        failed = {k for k, v in result["checks"].items() if not v}
        assert not failed, failed

        # 3 queries x 2 LLM steps on the first realization
        assert result["per_repeat_calls"] == [6, 0, 0, 0]
        assert result["real_usage"]["api_calls"] == 12  # + fresh stack
        assert result["per_repeat_length"] == [7, 7, 7, 6]  # adoption
        assert result["eliminated"] == ["citation_guard"]

        # deterministic backend => fresh-stack answers identical
        obs = result["answer_observation"]
        assert obs["fresh_answers_identical"] is True
        assert obs["fresh_kept_identical"] is True

    def test_answers_carry_citations(self, tmp_path: Path):
        args = argparse.Namespace(
            repeats=2, hot_threshold=2,
            trace=str(tmp_path / "trace.jsonl"),
        )
        backend = mock_backend()
        result = asyncio.run(run_battery(
            backend, args, paper_text=FAKE_PAPER))
        for answer in result["answers"].values():
            assert "[" in answer  # cited excerpt numbers

    def test_backend_records_usage(self):
        backend = mock_backend()

        async def _go():
            await backend.infer("hello")

        asyncio.run(_go())
        assert backend.api_calls == 1
        assert backend.input_tokens == 10
        assert backend.output_tokens == 8
