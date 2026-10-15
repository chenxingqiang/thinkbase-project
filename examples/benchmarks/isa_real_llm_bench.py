"""Real-LLM end-to-end benchmark for AgentISA result caching.

Methodology mirrors the cache experiment of ``isa_microbench.py`` — the
same ``zipf_workload`` generator, the same instruction pool and parameter
space, the same ``InstructionCache`` protocol — but the LLM backend is a
REAL model served through a local Anthropic-compatible gateway instead of
the counting mock.

Differences from the synthetic suite (all reported in the output):

* ``n_tasks`` and ``seeds`` are reduced (defaults: 60 tasks, 2 seeds vs.
  2,000 tasks, 5 seeds) because each real round trip costs seconds; the
  instruction pool (50) and parameter space (20) are unchanged.
* Latency is measured on the wall clock, not simulated.
* Token usage is taken from the gateway's ``usage`` field.

Honesty rules: failures are retried a bounded number of times for
transient errors only; persistent failures abort the run.  Nothing is
estimated or extrapolated — every number is a measurement.

Usage:
    python isa_real_llm_bench.py --tasks 60 --seeds 2 \
        --model claude-glm-5.3-flash --output isa_real_llm_results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

# Make the repo package importable regardless of how the script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from agentflow.core.isa.instruction import Instruction, InstructionType as IsaInstructionType

from isa_microbench import zipf_workload

DEFAULT_BASE_URL = "https://127.0.0.1:8443"
DEFAULT_CA_CERT = "/opt/homebrew/etc/nginx/ssl/witmem-gw.crt"
DEFAULT_MODEL = "claude-glm-5.3-flash"


@dataclass
class BackendStats:
    """Measurements collected from real gateway round trips."""

    api_calls: int = 0
    wall_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    latencies_ms: List[float] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.latencies_ms is None:
            self.latencies_ms = []

    def merged(self, other: "BackendStats") -> "BackendStats":
        return BackendStats(
            api_calls=self.api_calls + other.api_calls,
            wall_ms=self.wall_ms + other.wall_ms,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            latencies_ms=self.latencies_ms + other.latencies_ms,
        )


class RealLLMBackend:
    """One real ``/v1/messages`` round trip per ``infer`` call.

    Implements the same tiny protocol as the synthetic suite's
    ``MockLLMBackend`` so the experiment code above the backend is
    unchanged, while recording real measurements: call count, wall time,
    token usage and per-call latency.
    """

    def __init__(self, base_url: str, api_key: str, ca_cert: Optional[str],
                 model: str, max_tokens: int, timeout: float = 90.0,
                 transport: Optional[httpx.AsyncBaseTransport] = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self._client = httpx.AsyncClient(
            verify=ca_cert or True,
            trust_env=False,  # bypass any system proxy for the local gateway
            timeout=timeout,
            transport=transport,  # injectable for tests (MockTransport)
        )
        self.api_calls = 0
        self.wall_ms = 0.0
        self.input_tokens = 0
        self.output_tokens = 0
        self.latencies_ms: List[float] = []

    def stats(self) -> BackendStats:
        return BackendStats(self.api_calls, self.wall_ms, self.input_tokens,
                            self.output_tokens, list(self.latencies_ms))

    async def infer(self, payload: str) -> str:
        """One real model round trip; transient errors retried, then raised."""
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": payload[:2000]}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(3):
            start = time.perf_counter()
            try:
                resp = await self._client.post(
                    f"{self.base_url}/v1/messages", json=body, headers=headers
                )
            except httpx.HTTPError as error:  # network/transient
                last_error = error
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if resp.status_code >= 500:
                last_error = RuntimeError(f"gateway {resp.status_code}")
                if attempt < 2:  # transient: retry with backoff
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"gateway unreachable after 3 attempts: {last_error}"
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"gateway rejected request: {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()
            self.api_calls += 1
            self.wall_ms += elapsed_ms
            self.latencies_ms.append(elapsed_ms)
            usage = data.get("usage") or {}
            self.input_tokens += int(usage.get("input_tokens", 0))
            self.output_tokens += int(usage.get("output_tokens", 0))
            # Collect text blocks (thinking blocks are skipped; the
            # response content is irrelevant to the cache experiment).
            texts = [
                block.get("text", "")
                for block in data.get("content", [])
                if block.get("type") == "text"
            ]
            return "".join(texts) or f"[stop={data.get('stop_reason')}]"
        raise RuntimeError(f"gateway unreachable after 3 attempts: {last_error}")

    async def close(self) -> None:
        await self._client.aclose()


class RealLLMInstruction(Instruction):
    """LLM-backed instruction routed through the real gateway.

    Identical cache protocol to the synthetic suite's
    ``MockLLMInstruction``: an execution whose (instruction, params) pair
    is cached is served without touching the backend.
    """

    def __init__(self, backend: RealLLMBackend, name: str, use_cache: bool = True):
        super().__init__(
            id=f"llm_{name}",
            name=name,
            type=IsaInstructionType.LLM_INTERACTION,
            params={},
            description=f"Real LLM instruction {name}",
            side_effects=False,
            outputs=[],
            cache_config={"max_size": 100_000, "ttl": 3600} if use_cache else None,
        )
        self.backend = backend
        self.use_cache = use_cache

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        cache_key = None
        if self.use_cache:
            cache_key = self._get_cache_key(input_params)
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._update_metrics(time.perf_counter(), True, cache_hit=True)
                return cached
        result = await self.backend.infer(f"{self.name}:{sorted(input_params.items())}")
        if self.use_cache and cache_key is not None:
            self.cache.put(cache_key, result)
        self._update_metrics(time.perf_counter(), True)
        return result


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(q * len(ordered)), len(ordered) - 1)
    return ordered[idx]


def stats_summary(s: BackendStats) -> Dict[str, Any]:
    return {
        "api_calls": s.api_calls,
        "wall_ms": round(s.wall_ms, 1),
        "input_tokens": s.input_tokens,
        "output_tokens": s.output_tokens,
        "latency_ms_p50": round(_pct(s.latencies_ms, 0.50), 1),
        "latency_ms_p95": round(_pct(s.latencies_ms, 0.95), 1),
    }


async def run_real_cache_experiment(args: argparse.Namespace) -> Dict[str, Any]:
    backend = RealLLMBackend(
        base_url=args.base_url, api_key=args.api_key, ca_cert=args.ca_cert,
        model=args.model, max_tokens=args.max_tokens,
    )
    results: Dict[str, Any] = {
        "experiment": "cache_real_llm",
        "config": {
            "gateway_base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "seeds": args.seeds,
            "n_tasks": args.tasks,
            "n_instructions": 50,
            "n_param_values": 20,
            "zipf_s": args.zipf,
            "synthetic_reference": {
                "n_tasks": 2000, "seeds": 5, "n_instructions": 50,
                "n_param_values": 20,
                "reduction_mean": [0.5948, 0.6790, 0.7960],
            },
            "scale_note": (
                "Same workload generator (zipf_workload), instruction pool and "
                "parameter space as the synthetic suite; n_tasks and seeds are "
                "reduced because real round trips cost seconds. Latency is "
                "measured on the wall clock; tokens come from gateway usage."
            ),
        },
        "data": [],
    }
    try:
        # Preflight: one real call to fail fast with a clear error.
        t0 = time.perf_counter()
        pong = await backend.infer("Reply with exactly: pong")
        print(f"preflight ok: model={args.model} "
              f"{1000 * (time.perf_counter() - t0):.0f}ms -> {pong[:40]!r}")

        for zipf_s in args.zipf:
            per_seed: List[Dict[str, Any]] = []
            for seed in range(args.seeds):
                rng = random.Random(1000 + seed)
                workload = zipf_workload(rng, args.tasks, 50, 20, zipf_s)

                # Backend-independent reference: the sequence of cache
                # hits/misses depends only on the workload (instruction +
                # params pairs), so it is exactly computable without any
                # LLM.  The real run must reproduce this hit rate.
                seen: set = set()
                expected_hits = 0
                for idx, params in workload:
                    k = (idx, tuple(sorted(params.items())))
                    if k in seen:
                        expected_hits += 1
                    else:
                        seen.add(k)
                expected_hit_rate = expected_hits / args.tasks if args.tasks else 0.0

                # Baseline: no cache (fresh backend counters).
                backend.api_calls = backend.wall_ms = 0
                backend.input_tokens = backend.output_tokens = 0
                backend.latencies_ms = []
                instrs = [RealLLMInstruction(backend, f"i{i}", use_cache=False)
                          for i in range(50)]
                wall0 = time.perf_counter()
                for n_done, (idx, params) in enumerate(workload, 1):
                    await instrs[idx].execute(params)
                    if n_done % 25 == 0:
                        print(f"  [zipf={zipf_s} s{seed} baseline] {n_done}/{args.tasks} "
                              f"({time.perf_counter() - wall0:.0f}s)", flush=True)
                base_wall = (time.perf_counter() - wall0) * 1000.0
                base = backend.stats()
                base.wall_ms = base_wall  # true wall clock, incl. overhead

                # Treatment: InstructionCache enabled.
                backend.api_calls = backend.wall_ms = 0
                backend.input_tokens = backend.output_tokens = 0
                backend.latencies_ms = []
                instrs = [RealLLMInstruction(backend, f"i{i}", use_cache=True)
                          for i in range(50)]
                wall0 = time.perf_counter()
                for n_done, (idx, params) in enumerate(workload, 1):
                    await instrs[idx].execute(params)
                    if n_done % 25 == 0:
                        print(f"  [zipf={zipf_s} s{seed} cached]    {n_done}/{args.tasks} "
                              f"({time.perf_counter() - wall0:.0f}s)", flush=True)
                treat_wall = (time.perf_counter() - wall0) * 1000.0
                treat = backend.stats()
                treat.wall_ms = treat_wall

                total = sum(i.metrics.total_executions for i in instrs)
                hits = sum(i.metrics.cache_hits for i in instrs)
                per_seed.append({
                    "seed": seed,
                    "expected_hit_rate": round(expected_hit_rate, 4),
                    "baseline": stats_summary(base),
                    "cached": stats_summary(treat),
                    "hit_rate": round(hits / total, 4) if total else 0.0,
                    "call_reduction": round(1.0 - treat.api_calls / base.api_calls, 4)
                        if base.api_calls else 0.0,
                    "wall_saved_ms": round(base_wall - treat_wall, 1),
                    "tokens_saved": (base.input_tokens + base.output_tokens)
                                    - (treat.input_tokens + treat.output_tokens),
                })
                print(f"zipf={zipf_s} seed={seed}: "
                      f"calls {base.api_calls}->{treat.api_calls} "
                      f"(-{per_seed[-1]['call_reduction']:.1%}), "
                      f"wall {base_wall:.0f}ms->{treat_wall:.0f}ms, "
                      f"hit={per_seed[-1]['hit_rate']:.1%} "
                      f"(expected {expected_hit_rate:.1%})", flush=True)

            # Aggregate across seeds.
            def agg(key: str, sub: str = None) -> Dict[str, float]:
                vals = [r[key][sub] for r in per_seed] if sub else [r[key] for r in per_seed]
                return {"mean": round(statistics.mean(vals), 4),
                        "stdev": round(statistics.stdev(vals), 4)
                                 if len(vals) > 1 else 0.0}

            results["data"].append({
                "zipf_s": zipf_s,
                "per_seed": per_seed,
                "api_calls_no_cache": agg("baseline", "api_calls"),
                "api_calls_cached": agg("cached", "api_calls"),
                "reduction_ratio": agg("call_reduction"),
                "hit_rate": agg("hit_rate"),
                "expected_hit_rate": agg("expected_hit_rate"),
                "wall_ms_no_cache": agg("baseline", "wall_ms"),
                "wall_ms_cached": agg("cached", "wall_ms"),
                "wall_saved_ms": agg("wall_saved_ms"),
                "latency_ms_p50_cached": agg("cached", "latency_ms_p50"),
            })
        return results
    finally:
        await backend.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AgentISA real-LLM cache benchmark (local gateway)")
    parser.add_argument("--tasks", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--zipf", nargs="+", type=float, default=[0.5, 1.0, 1.5])
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--base-url", type=str,
                        default=os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", type=str,
                        default=os.environ.get("ANTHROPIC_AUTH_TOKEN", ""))
    parser.add_argument("--ca-cert", type=str,
                        default=os.environ.get("NODE_EXTRA_CA_CERTS", DEFAULT_CA_CERT))
    parser.add_argument("--output", type=str, default="isa_real_llm_results.json")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("API key required: --api-key or ANTHROPIC_AUTH_TOKEN")

    t0 = time.perf_counter()
    res = asyncio.run(run_real_cache_experiment(args))
    res["wall_time_s"] = round(time.perf_counter() - t0, 1)
    out_path = Path(__file__).parent / args.output
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nResults written to {out_path}")
    print(json.dumps({k: v for k, v in res.items() if k != "data"}, indent=2)[:800])


if __name__ == "__main__":
    main()
