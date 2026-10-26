#!/usr/bin/env python3
"""Real-gateway validation of AgentISA COMPILATION + HOT-PATH semantics,
plus a priced analytical cost baseline from measured tokens.

This harness addresses two gaps the paper is explicit about: only result
caching had ever been executed against a live model (Section 5.7), and
no cost baseline of any kind existed (Section 5.9, items 1 and 6).

W1 -- semantics preservation of compilation + hot-path, measured.
    The SAME task stream (a seven-step service-log triage with ONE
    cacheable LLM-type step) is executed as two arms against a
    pluggable backend:

    * arm (a) -- the unoptimized authored stream;
    * arm (b) -- the same stream driven through the production
      machinery: the closed loop of ``OptimizationManager`` /
      ``optimize_hot_path`` adopts a shrunken trace mid-run, and the
      static ``InstructionCompiler`` view is compiled at NONE vs
      AGGRESSIVE with the dataflow-preservation assertion.

    Backends (``--backend``):

    * ``real``  -- the same local Anthropic-protocol gateway as the
      existing harness (alias ``claude-glm-5.3-flash``); tokens are
      gateway-reported, latency is measured.
    * ``mock``  -- ``DeterministicMockBackend``: same protocol, counts
      calls, accounts tokens from template sizes, and charges the
      paper's constant 80 ms per simulated call.  Fully deterministic;
      no network.

    The W1 headline is BYTE-IDENTICALITY of the final artifacts between
    the two arms on the same backend.  The default protocol
    (``--arm-b-cache shared``) makes that property exact on ANY backend,
    including a nondeterministic live model: arm (a) runs first, cold,
    and performs every live round trip; arm (b) then executes against
    the result cache warmed by arm (a).  The classify step's cache key
    is sha256 over exactly the input dictionary its prompt is built
    from, so if the compiled / hot-path-adopted stream computes
    identical keys -- checked: it must issue ZERO new gateway calls --
    it sent identical prompts and consumed identical live-model values,
    and every artifact must come out byte-for-byte equal.  Any
    perturbation of the LLM step's inputs by the optimization would
    surface as a cache miss (a live call) and/or artifact divergence.
    ``--arm-b-cache independent`` runs both arms on cold caches for
    comparison; on the real backend that byte-identity is an empirical
    observation (temperature 0 is requested but live models are not
    guaranteed byte-deterministic) and is reported as such, not claimed.

W2 -- a priced ANALYTICAL baseline from MEASURED tokens.
    ``compute_cost(tokens, mode)`` prices the measured per-key token
    counts (gateway-reported on real; template-size-accounted on mock)
    under three modes:

    * ``no_opt``               -- every realization re-queries the
      model at full price (no result cache, no prompt-cache discount);
    * ``agentisa_cache``       -- only first occurrences are paid
      (the measured behavior of both arms);
    * ``provider_prompt_cache``-- MODELED: every realization queries
      the model, but repeated prompts are served as provider
      prompt-cache reads, billed at ``--cache-read-rate`` (default 0.1,
      after Anthropic's published prompt-caching economics) times the
      input-token price; completions always bill at full price.

    The cost columns are MODELED-on-measured: token counts are
    measurements, the pricing overlay (list prices via
    ``--price-in-per-mt`` / ``--price-out-per-mt`` and the read
    discount) is a parameterized model.  The serving gateway's actual
    prices are unknown, so no dollar-savings claim is made -- the
    comparison is between pricing MODES on the same measured tokens.
    GPTCache integration is explicitly out of scope this pass; no
    external-cache head-to-head is claimed.

Determinism: the workload is a seeded truncated-Zipf draw sequence over
a fixed universe of task instances (same weight formula as the
microbenchmark suite); every corpus is generated from a per-task seed;
the mock backend is a pure function of its input; no wall-clock value
enters mock-mode output, which is byte-reproducible run to run.

Usage:
    ./venv/bin/python examples/paper_real_llm_compiler_test.py \
        --backend mock --scale tiny
    ./venv/bin/python examples/paper_real_llm_compiler_test.py \
        --backend real --scale tiny \
        --json examples/paper_real_llm_compiler_results_real.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo package and the sibling example modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples" / "benchmarks"))

from agentflow.core.isa.instruction import (
    Instruction, InstructionType as IsaType,
)
from agentflow.core.isa.instruction_optimization import (
    OptimizationManager, OptimizationMetrics,
)
from agentflow.sandbox.local_executor import LocalSandboxExecutor

from random_tasks_stress import compile_check
from system_task_demo import inject_sandbox, run_stream

DEFAULT_BASE_URL = "https://127.0.0.1:8443"
DEFAULT_CA_CERT = "/opt/homebrew/etc/nginx/ssl/witmem-gw.crt"
DEFAULT_MODEL = "claude-glm-5.3-flash"

# The mock charges the same constant per-call latency the paper's
# microbenchmarks simulate, so its wall accounting is comparable.
MOCK_CALL_LATENCY_MS = 80.0

SEVERITIES = ["low", "medium", "high", "critical"]

SCALE_PRESETS = {
    "tiny": {"tasks": 50, "seeds": 1, "zipf_levels": [1.0]},
    "paper": {"tasks": 200, "seeds": 1, "zipf_levels": [0.5, 1.0, 1.5]},
}


# ---------------------------------------------------------------------------
# Seeded task universe (fixed population of service-log triage instances)
# ---------------------------------------------------------------------------

COMPONENTS = ["auth", "db", "api", "cache", "worker"]
LEVELS = ["INFO", "WARN", "ERROR"]
MESSAGES = [
    "request completed",
    "latency above threshold",
    "connection reset by peer",
    "token refresh failed",
    "disk usage high",
    "timeout waiting for lock",
    "upstream 5xx",
    "schema mismatch on write",
    "cache miss storm",
    "worker heartbeat lost",
]

LINES_PER_TASK = 48


def task_corpus(rank: int, lines_per_task: int = LINES_PER_TASK) -> str:
    """Deterministic log corpus for task ``rank``.

    Each task features a unique anchor signature (its Zipf rank maps to
    a distinct (component, message) pair), so distinct tasks yield
    distinct incident-signature batches -- and therefore distinct cache
    keys and prompts -- while overlaps in the long tail stay realistic.
    """
    rng = random.Random(10_000 + rank)
    anchor_component = COMPONENTS[rank % len(COMPONENTS)]
    anchor_message = MESSAGES[(rank // len(COMPONENTS)) % len(MESSAGES)]
    out = []
    for i in range(lines_per_task):
        if i % 3 == 0:
            component, message = anchor_component, anchor_message
        else:
            component, message = rng.choice(COMPONENTS), rng.choice(MESSAGES)
        ts = (f"2025-06-{rng.randint(1, 28):02d}T{rng.randint(0, 23):02d}:"
              f"{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z")
        out.append(
            f"{ts} level={rng.choice(LEVELS)} component={component} "
            f'msg="{message}" req={rng.randint(1000, 9999)}'
        )
    return "\n".join(out) + "\n"


def parse_log_lines(raw_lines: List[str]) -> List[Dict[str, str]]:
    """Parse log lines into structured entries (shared by stream and
    expectation computation, so the mirror can never drift)."""
    entries = []
    for ln in raw_lines:
        parts = dict(p.split("=", 1) for p in ln.split(" ") if "=" in p)
        msg = parts.get("msg", '"?"').strip('"')
        entries.append({
            "level": parts.get("level", "INFO"),
            "component": parts.get("component", "unknown"),
            "signature": f'{parts.get("component", "unknown")}|{msg}',
        })
    return entries


def incident_signatures(raw_lines: List[str]) -> List[str]:
    """Sorted distinct WARN/ERROR signatures -- the classify step's input."""
    entries = parse_log_lines(raw_lines)
    return sorted({e["signature"] for e in entries
                   if e["level"] in ("WARN", "ERROR")})


def cache_key_id(signatures: List[str]) -> str:
    """Cache key of the classify step for a signature batch, truncated.

    Mirrors ``Instruction._get_cache_key({"signatures": ...})``: the
    same json.dumps serialization, the same sha256, so the backend-free
    expectation is computed with the identical keying the production
    instruction uses.
    """
    serialized = json.dumps({"signatures": "|".join(signatures)}, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def zipf_draws(seed: int, n_draws: int, population: int,
               zipf_s: float) -> List[int]:
    """Draw ``n_draws`` task ranks from a truncated Zipf distribution.

    Same weight formula as the microbenchmark suite's ``zipf_workload``
    (``1 / (i + 1) ** s`` over the support), applied to task ranks; the
    seed pins the sequence exactly.
    """
    rng = random.Random(seed)
    weights = [1.0 / (i + 1) ** zipf_s for i in range(population)]
    total = sum(weights)
    return [rng.choices(range(population), weights=[w / total for w in weights],
                        k=1)[0]
            for _ in range(n_draws)]


# ---------------------------------------------------------------------------
# Backends: same protocol (async infer(payload) -> str) + accounting
# ---------------------------------------------------------------------------

def _account_tokens(text: str) -> int:
    """Deterministic token accounting from text size (chars/4, min 1)."""
    return max(1, math.ceil(len(text) / 4))


def fallback_severity(signature: str) -> str:
    """Deterministic severity used when a reply line does not parse.

    A pure function of the signature (keyword bumps over a hash
    bucket), mirroring the rule-based stand-in of the system-level
    demonstration: the cost accounting is what matters, and any
    unparsed reply must still classify deterministically.
    """
    base = int(hashlib.sha256(signature.encode("utf-8")).hexdigest()[:8], 16)
    if "timeout" in signature or "5xx" in signature or "heartbeat" in signature:
        base += 2
    elif "failed" in signature or "mismatch" in signature or "reset" in signature:
        base += 1
    return SEVERITIES[base % len(SEVERITIES)]


def parse_classification(raw: str, signatures: List[str]) -> Dict[str, str]:
    """Deterministically map a model reply to one severity per signature.

    Expected reply format: one ``<index>: <severity>`` line per
    signature.  Signatures the reply does not mention fall back to the
    deterministic rule, so the parse is a pure function of
    (reply, signatures).
    """
    classification: Dict[str, str] = {}
    for n, sig in enumerate(signatures):
        m = re.search(rf"(?<!\d){n}\s*[:=\-]\s*(low|medium|high|critical)\b",
                      raw, re.IGNORECASE)
        classification[sig] = m.group(1).lower() if m else fallback_severity(sig)
    return classification


class DeterministicMockBackend:
    """Deterministic stand-in with the same protocol as the real gateway.

    Counts every call, accounts tokens from the prompt/reply template
    sizes, and charges the paper's constant 80 ms simulated latency per
    call (``wall_ms`` is accounted, never slept, so mock runs are fast
    and byte-reproducible).  The reply is a pure function of the
    payload, mirroring how the microbenchmark mock stands in for the
    model: the cost accounting is what matters, not the classifier's
    intelligence.
    """

    def __init__(self) -> None:
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_ms = 0.0
        self.latencies_ms: List[float] = []
        self.token_source = "template-size accounting (mock)"

    async def infer(self, payload: str) -> str:
        reply = self._mock_reply(payload)
        self.api_calls += 1
        self.input_tokens += _account_tokens(payload)
        self.output_tokens += _account_tokens(reply)
        self.wall_ms += MOCK_CALL_LATENCY_MS
        self.latencies_ms.append(MOCK_CALL_LATENCY_MS)
        return reply

    @staticmethod
    def _mock_reply(payload: str) -> str:
        """Deterministic reply: one ``<index>: <severity>`` line per
        signature line found in the prompt."""
        lines = []
        for m in re.finditer(r"^\s*(\d+): ([a-z]+)\|(.+)$", payload, re.MULTILINE):
            signature = f"{m.group(2)}|{m.group(3)}"
            lines.append(f"{m.group(1)}: {fallback_severity(signature)}")
        return "\n".join(lines) if lines else "0: medium"

    async def close(self) -> None:
        pass


class RealGatewayBackend:
    """Real ``/v1/messages`` round trips, reusing the existing client.

    Wraps ``isa_real_llm_bench.RealLLMBackend`` with the same gateway
    configuration as the Section 5.7 harness (base URL, internal-CA
    TLS, alias, sequential httpx, 90 s timeout, three bounded retries
    for transient failures) and the same accounting surface as the
    mock, so the experiment code above the backend is unchanged.
    """

    def __init__(self, *, base_url: str, api_key: str, ca_cert: Optional[str],
                 model: str, max_tokens: int, timeout: float = 90.0,
                 max_payload_chars: int = 4000):
        # Imported lazily so mock-only runs never touch httpx at all.
        from isa_real_llm_bench import RealLLMBackend
        self._inner = RealLLMBackend(
            base_url=base_url, api_key=api_key, ca_cert=ca_cert,
            model=model, max_tokens=max_tokens, timeout=timeout,
            temperature=0.0, max_payload_chars=max_payload_chars,
        )
        self.token_source = "gateway usage field (real)"

    @property
    def api_calls(self) -> int:
        return self._inner.api_calls

    @property
    def input_tokens(self) -> int:
        return self._inner.input_tokens

    @property
    def output_tokens(self) -> int:
        return self._inner.output_tokens

    @property
    def wall_ms(self) -> float:
        return self._inner.wall_ms

    @property
    def latencies_ms(self) -> List[float]:
        return self._inner.latencies_ms

    async def infer(self, payload: str) -> str:
        return await self._inner.infer(payload)

    async def close(self) -> None:
        await self._inner.close()


def resolve_api_key() -> Optional[str]:
    """API key from ~/.claude/settings.json or the environment."""
    settings_path = Path.home() / ".claude" / "settings.json"
    if settings_path.exists():
        try:
            settings = json.loads(settings_path.read_text())
            key = settings.get("env", {}).get("ANTHROPIC_AUTH_TOKEN")
            if key:
                return key
        except (OSError, json.JSONDecodeError):
            pass
    return os.environ.get("ANTHROPIC_AUTH_TOKEN")


def backend_snapshot(backend: Any) -> Dict[str, Any]:
    return {
        "api_calls": backend.api_calls,
        "input_tokens": backend.input_tokens,
        "output_tokens": backend.output_tokens,
        "wall_ms": backend.wall_ms,
    }


# ---------------------------------------------------------------------------
# The task stream: seven steps, ONE cacheable LLM-type step
# ---------------------------------------------------------------------------

CLASSIFY_PROMPT = (
    "You are triaging service incidents for a production system. "
    "Classify each incident signature below by operational severity. "
    "Reply with one line per signature, formatted exactly as "
    "'<index>: <severity>' where <severity> is one of: low, medium, "
    "high, critical. No other text.\n\n"
    "Signatures:\n{signature_lines}"
)


def build_triage_stream(backend: Any) -> List[Instruction]:
    """The authored (unoptimized) instruction stream.

    Same shape as the system-level demonstration: two side-effectful
    sandbox steps, pure transforms, a planted dead defensive guard, and
    ONE cacheable LLM-type step that makes a real backend round trip
    per distinct incident-signature batch.
    """

    class IngestLogs(Instruction):
        def __init__(self):
            super().__init__(
                id="ingest_logs", name="ingest_logs",
                type=IsaType.STATE_MANAGEMENT,
                params={}, description="Read the task's log file and rank",
                side_effects=True, outputs=["raw_lines", "task_rank"],
            )
            self.dependencies = []
            self.sandbox = None

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            raw = self.sandbox.read_file("logs/task.log").decode("utf-8")
            meta = json.loads(self.sandbox.read_file("logs/task_meta.json")
                              .decode("utf-8"))
            self._update_metrics(start, True)
            return {"raw_lines": raw.splitlines(),
                    "task_rank": int(meta["rank"])}

    class CheckFormat(Instruction):
        """Defensive format guard; result consumed by nothing (dead)."""

        def __init__(self):
            super().__init__(
                id="check_format", name="check_format",
                type=IsaType.CONTROL_FLOW,
                params={}, description="Validate log line syntax (guard)",
                side_effects=False, outputs=[],
            )
            self.dependencies = ["raw_lines"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            bad = sum(1 for ln in input_params["ctx"]["raw_lines"]
                      if "level=" not in ln)
            self._update_metrics(start, True)
            return {"format_violations": bad}

    class NormalizeEntries(Instruction):
        def __init__(self):
            super().__init__(
                id="normalize_entries", name="normalize_entries",
                type=IsaType.CONTROL_FLOW,
                params={}, description="Parse log lines into entries",
                side_effects=False, outputs=["entries"],
            )
            self.dependencies = ["raw_lines"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            entries = parse_log_lines(input_params["ctx"]["raw_lines"])
            self._update_metrics(start, True)
            return {"entries": entries}

    class ExtractIncidents(Instruction):
        def __init__(self):
            super().__init__(
                id="extract_incidents", name="extract_incidents",
                type=IsaType.CONTROL_FLOW,
                params={}, description="Extract WARN/ERROR incidents",
                side_effects=False, outputs=["incidents"],
            )
            self.dependencies = ["entries"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            incidents = [e for e in input_params["ctx"]["entries"]
                         if e["level"] in ("WARN", "ERROR")]
            self._update_metrics(start, True)
            return {"incidents": incidents}

    class ClassifySignature(Instruction):
        """The ONE cacheable LLM-type step: one backend round trip per
        distinct incident-signature batch; repeats are served from the
        result cache without touching the backend."""

        def __init__(self):
            super().__init__(
                id="classify_signature", name="classify_signature",
                type=IsaType.LLM_INTERACTION,
                params={}, description="Classify incident signatures (LLM)",
                side_effects=False, outputs=["classified"],
                cache_config={"max_size": 100_000, "ttl": 3600},
            )
            self.dependencies = ["incidents"]
            self.backend = backend

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            signatures = sorted(
                {i["signature"] for i in input_params["ctx"]["incidents"]})
            # The cache key is taken over exactly the input the prompt is
            # built from, so key identity implies prompt identity.
            cache_key = self._get_cache_key({"signatures": "|".join(signatures)})
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._update_metrics(start, True, cache_hit=True)
                return cached
            prompt = CLASSIFY_PROMPT.format(
                signature_lines="\n".join(
                    f"{n}: {sig}" for n, sig in enumerate(signatures)))
            raw = await self.backend.infer(prompt)
            classification = parse_classification(raw, signatures)
            result = {"classified": classification}
            self.cache.put(cache_key, result)
            self._update_metrics(start, True)
            return result

    class AggregateTriage(Instruction):
        def __init__(self):
            super().__init__(
                id="aggregate_triage", name="aggregate_triage",
                type=IsaType.CONTROL_FLOW,
                params={}, description="Aggregate incident statistics",
                side_effects=False, outputs=["summary"],
            )
            self.dependencies = ["classified", "incidents"]

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            ctx = input_params["ctx"]
            by_severity: Dict[str, int] = {}
            by_component: Dict[str, int] = {}
            by_signature: Dict[str, int] = {}
            for incident in ctx["incidents"]:
                sig = incident["signature"]
                sev = ctx["classified"][sig]
                by_severity[sev] = by_severity.get(sev, 0) + 1
                comp = sig.split("|", 1)[0]
                by_component[comp] = by_component.get(comp, 0) + 1
                by_signature[sig] = by_signature.get(sig, 0) + 1
            top = sorted(by_signature.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
            self._update_metrics(start, True)
            return {"summary": {
                "task_rank": ctx["task_rank"],
                "total_incidents": len(ctx["incidents"]),
                "distinct_signatures": len(ctx["classified"]),
                "by_severity": by_severity,
                "by_component": by_component,
                "top_signatures": top,
            }}

    class WriteReport(Instruction):
        def __init__(self):
            super().__init__(
                id="write_report", name="write_report",
                type=IsaType.STATE_MANAGEMENT,
                params={}, description="Write the triage report artifact",
                side_effects=True,
            )
            self.dependencies = ["summary"]
            self.sandbox = None

        async def execute(self, input_params: Dict[str, Any]) -> Any:
            start = time.time()
            s = input_params["ctx"]["summary"]
            lines = [
                "# Incident Triage Report",
                "",
                f"- task: {s['task_rank']}",
                f"- incidents: {s['total_incidents']}",
                f"- distinct signatures: {s['distinct_signatures']}",
                "- by severity: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(s["by_severity"].items())),
                "- by component: " + ", ".join(
                    f"{k}={v}" for k, v in sorted(s["by_component"].items())),
                "- top signatures:",
            ]
            lines += [f"  - {sig} ({count})" for sig, count in s["top_signatures"]]
            report = "\n".join(lines) + "\n"
            self.sandbox.write_file("report.md", report.encode("utf-8"))
            self._update_metrics(start, True)
            return {"report_written": True}

    return [IngestLogs(), CheckFormat(), NormalizeEntries(),
            ExtractIncidents(), ClassifySignature(), AggregateTriage(),
            WriteReport()]


async def realize_task(stream: List[Instruction], rank: int, corpus: str,
                       *, tag: str) -> bytes:
    """One realization of the task on a fresh sandbox workspace."""
    sandbox = LocalSandboxExecutor(name=tag)
    try:
        sandbox.write_file("logs/task.log", corpus.encode("utf-8"))
        sandbox.write_file(
            "logs/task_meta.json", json.dumps({"rank": rank}).encode("utf-8"))
        inject_sandbox(stream, sandbox)
        await run_stream(stream)
        return sandbox.read_file("report.md")
    finally:
        sandbox.close()


# ---------------------------------------------------------------------------
# Priced analytical baseline (W2): MODELED-on-measured
# ---------------------------------------------------------------------------

COST_MODES = ("no_opt", "agentisa_cache", "provider_prompt_cache")


def compute_cost(tokens: Dict[str, Any], mode: str, *,
                 price_in_per_mt: float = 3.0,
                 price_out_per_mt: float = 15.0,
                 cache_read_rate: float = 0.1) -> Dict[str, Any]:
    """Price a measured token stream under one cost mode.

    ``tokens`` carries the measurements: ``draw_keys`` is the sequence
    of cache keys actually drawn (the workload), ``per_key_tokens``
    maps each distinct key to its MEASURED (prompt, completion) token
    counts -- the counts of the live call that key caused.  Because the
    prompt is a pure function of the key, every re-draw of a key would
    re-send the same prompt, so the measured counts price all modes:

    * ``no_opt``: every draw re-queries at full price (no result cache,
      no prompt-cache discount);
    * ``agentisa_cache``: only first occurrences are paid (measured
      behavior of both arms);
    * ``provider_prompt_cache`` (MODELED): every draw queries, but
      repeat prompts bill their input tokens as provider prompt-cache
      reads at ``cache_read_rate`` x the input price; completions bill
      at full price.  Assumes the provider cache persists across the
      run and that repeat outputs match the measured output length.

    Prices are parameterized list-price stand-ins ($/M tokens); the
    result is labeled MODELED-on-measured and is not a bill.
    """
    if mode not in COST_MODES:
        raise ValueError(f"unknown cost mode: {mode!r}")
    draw_keys: List[str] = list(tokens["draw_keys"])
    per_key: Dict[str, Tuple[int, int]] = tokens["per_key_tokens"]

    calls = 0
    in_full = 0
    in_read = 0
    out = 0
    seen: set = set()
    for key in draw_keys:
        tin, tout = per_key[key]
        if mode == "no_opt":
            calls += 1
            in_full += tin
            out += tout
        elif mode == "agentisa_cache":
            if key not in seen:
                calls += 1
                in_full += tin
                out += tout
        else:  # provider_prompt_cache
            calls += 1
            if key not in seen:
                in_full += tin
            else:
                in_read += tin
            out += tout
        seen.add(key)

    billed_input = in_full + cache_read_rate * in_read
    cost_usd = (billed_input / 1e6 * price_in_per_mt
                + out / 1e6 * price_out_per_mt)
    return {
        "mode": mode,
        "label": "MODELED-on-measured",
        "calls": calls,
        "input_tokens_full_price": in_full,
        "input_tokens_cache_read": in_read,
        "input_tokens_billed_equivalent": round(billed_input, 1),
        "output_tokens": out,
        "cost_usd": round(cost_usd, 6),
    }


# ---------------------------------------------------------------------------
# The two arms
# ---------------------------------------------------------------------------

async def run_arm(backend: Any, *, arm: str, draws: List[int],
                  corpora: Dict[int, str], draw_key_ids: List[str],
                  expected_calls: List[int], hot_threshold: int,
                  shared_cache: Optional[Any], wall_measured: bool,
                  tag: str) -> Dict[str, Any]:
    """Execute one arm over the full draw sequence.

    ``arm`` is ``"unoptimized"`` (the authored stream, no optimization
    manager, never adopted) or ``"compiled_hotpath"`` (the production
    closed loop: after every realization the execution is recorded and
    the hot-path-optimized trace is adopted when the machinery offers
    one).  ``shared_cache`` optionally points the classify step at
    another arm's result cache.
    """
    stream = build_triage_stream(backend)
    if shared_cache is not None:
        classify = next(i for i in stream if i.id == "classify_signature")
        classify.cache = shared_cache

    manager: Optional[OptimizationManager] = None
    if arm == "compiled_hotpath":
        manager = OptimizationManager()
        manager.dynamic_optimizer.hot_path_threshold = hot_threshold

    active = stream
    per_realization: List[Dict[str, Any]] = []
    artifacts: List[bytes] = []

    for d, rank in enumerate(draws):
        before = backend_snapshot(backend)
        t0 = time.perf_counter()
        artifact = await realize_task(
            active, rank, corpora[rank], tag=f"{tag}_{d}")
        draw_wall_ms = (time.perf_counter() - t0) * 1000.0
        after = backend_snapshot(backend)
        per_realization.append({
            "draw": d,
            "task_rank": rank,
            "cache_key": draw_key_ids[d],
            "expected_calls": expected_calls[d],
            "calls": after["api_calls"] - before["api_calls"],
            "prompt_tokens": after["input_tokens"] - before["input_tokens"],
            "completion_tokens": after["output_tokens"] - before["output_tokens"],
            "accounted_wall_ms": round(
                after["wall_ms"] - before["wall_ms"], 1),
            "instructions": len(active),
            "artifact_sha256": hashlib.sha256(artifact).hexdigest(),
        })
        if wall_measured:
            per_realization[-1]["measured_wall_ms"] = round(draw_wall_ms, 1)
        artifacts.append(artifact)

        if manager is not None:
            metrics = OptimizationMetrics(
                execution_time=0.05, memory_usage=1.0,
                api_calls=per_realization[-1]["calls"], success_rate=1.0)
            result = manager.record_execution(active, metrics)
            if (result is not active and result is not None
                    and len(result) < len(active)):
                active = result  # adopt the hot-path-optimized trace

    instructions_per_realization = [r["instructions"] for r in per_realization]
    adopted_at = next(
        (i + 1 for i, n in enumerate(instructions_per_realization)
         if n < instructions_per_realization[0]),
        None,
    )
    classify = next(i for i in stream if i.id == "classify_signature")
    observed = [r["calls"] for r in per_realization]

    summary: Dict[str, Any] = {
        "arm": arm,
        "realizations": len(draws),
        "model_calls": sum(observed),
        "prompt_tokens": sum(r["prompt_tokens"] for r in per_realization),
        "completion_tokens": sum(r["completion_tokens"] for r in per_realization),
        "accounted_wall_ms": round(
            sum(r["accounted_wall_ms"] for r in per_realization), 1),
        "instruction_executions": sum(
            i.metrics.total_executions for i in stream),
        "instructions_per_realization": instructions_per_realization,
        "instructions_first_realization": instructions_per_realization[0],
        "optimized_form_adopted_at_realization": adopted_at,
        "stream_eliminated_by_hot_path": sorted(
            {i.id for i in stream}
            - {getattr(i, "id", None) for i in active}),
        "cache_hits": classify.metrics.cache_hits,
        "cache_misses": classify.metrics.cache_misses,
        "cache_hit_rate": round(classify.metrics.cache_hit_rate, 4),
        "calls_match_expected": observed == expected_calls,
        "expected_model_calls": sum(expected_calls),
        "per_realization": per_realization,
    }
    if wall_measured:
        summary["measured_wall_ms"] = round(
            sum(r.get("measured_wall_ms", 0.0) for r in per_realization), 1)
    # Private keys consumed by the caller (never serialized): the final
    # artifact bytes and the classify step's result cache, so the
    # shared-cache protocol can wire arm (b) to arm (a)'s cache.
    summary["_artifacts"] = artifacts
    summary["_classify_cache"] = classify.cache
    return summary


# ---------------------------------------------------------------------------
# One (zipf, seed) cell
# ---------------------------------------------------------------------------

async def run_zipf_level(backend: Any, *, zipf_s: float, seed: int,
                         n_draws: int, population: int, hot_threshold: int,
                         arm_b_cache: str, seed_base: int,
                         price_in_per_mt: float, price_out_per_mt: float,
                         cache_read_rate: float, wall_measured: bool
                         ) -> Dict[str, Any]:
    draws = zipf_draws(seed_base + seed, n_draws, population, zipf_s)

    # Backend-independent expectation: the classify step's cache key is
    # a pure function of the task, so the full hit/miss pattern of the
    # run is computable without any model.  The measured arms must
    # reproduce it exactly.
    corpora: Dict[int, str] = {}
    draw_key_ids: List[str] = []
    for rank in draws:
        if rank not in corpora:
            corpora[rank] = task_corpus(rank)
        draw_key_ids.append(
            cache_key_id(incident_signatures(corpora[rank].splitlines())))
    seen: set = set()
    expected_first_miss = []
    for key in draw_key_ids:
        expected_first_miss.append(key not in seen)
        seen.add(key)
    distinct_keys = len(seen)
    expected_pattern = [1 if first else 0 for first in expected_first_miss]
    # Arm (b) replays the same draws: under the shared-cache protocol
    # every key is already resident (checked: arm (b) must issue zero
    # new calls); under independent caches it repeats arm (a)'s pattern.
    expected_arm_b = ([0] * n_draws if arm_b_cache == "shared"
                      else list(expected_pattern))

    tag = f"cmp_z{zipf_s}_s{seed}"

    # Arm (a): unoptimized, cold cache, the live reference run.
    arm_a = await run_arm(
        backend, arm="unoptimized", draws=draws, corpora=corpora,
        draw_key_ids=draw_key_ids, expected_calls=expected_pattern,
        hot_threshold=hot_threshold, shared_cache=None,
        wall_measured=wall_measured, tag=f"{tag}_a")

    # Arm (b): same draws through the real compiler / hot-path closed
    # loop.  Shared protocol: it runs against arm (a)'s warmed result
    # cache and must make every call disappear while reproducing the
    # artifacts byte-for-byte.
    shared_cache = (arm_a_classify_cache(arm_a)
                    if arm_b_cache == "shared" else None)
    arm_b = await run_arm(
        backend, arm="compiled_hotpath", draws=draws, corpora=corpora,
        draw_key_ids=draw_key_ids, expected_calls=expected_arm_b,
        hot_threshold=hot_threshold, shared_cache=shared_cache,
        wall_measured=wall_measured, tag=f"{tag}_b")

    # Static-compilation view of the same stream (real InstructionCompiler),
    # with the dataflow-preservation assertion.
    static_compile = compile_check(
        build_triage_stream(backend), guards={"check_format"})

    # Byte-identity of final outputs between the arms (the W1 headline).
    mismatched = [d for d, (x, y) in
                  enumerate(zip(arm_a["_artifacts"], arm_b["_artifacts"]))
                  if x != y]

    # Measured per-key tokens (from the live calls that actually
    # happened) price every cost mode.
    per_key_tokens: Dict[str, Tuple[int, int]] = {}
    for record in arm_a["per_realization"]:
        if record["calls"] == 1:
            per_key_tokens[record["cache_key"]] = (
                record["prompt_tokens"], record["completion_tokens"])
    for record in arm_b["per_realization"]:  # independent protocol only
        if (record["calls"] == 1
                and record["cache_key"] not in per_key_tokens):
            per_key_tokens[record["cache_key"]] = (
                record["prompt_tokens"], record["completion_tokens"])
    tokens = {"draw_keys": draw_key_ids, "per_key_tokens": per_key_tokens}
    modes = {
        name: compute_cost(
            tokens, name, price_in_per_mt=price_in_per_mt,
            price_out_per_mt=price_out_per_mt, cache_read_rate=cache_read_rate)
        for name in COST_MODES
    }

    # Strip the private in-memory keys before serializing.
    for arm_summary in (arm_a, arm_b):
        arm_summary.pop("_artifacts", None)
        arm_summary.pop("_classify_cache", None)

    return {
        "zipf_s": zipf_s,
        "seed": seed,
        "draws": n_draws,
        "task_population": population,
        "protocol": arm_b_cache,
        "run_order": ["arm_a_unoptimized", "arm_b_compiled_hotpath"],
        "distinct_cache_keys_expected": distinct_keys,
        "expected_first_miss_positions": [
            d for d, first in enumerate(expected_first_miss) if first],
        "arm_a_unoptimized": arm_a,
        "arm_b_compiled_hotpath": arm_b,
        "static_compile": static_compile,
        "byte_identity": {
            "all_realizations_identical": not mismatched,
            "mismatched_realizations": mismatched,
            "arm_b_new_model_calls": arm_b["model_calls"],
            "note": (
                "shared-cache protocol: arm (a) made every live call and "
                "warmed the result cache; arm (b) had to hit the same keys "
                "(zero new calls checked) and reproduce every artifact "
                "byte-for-byte.  Cache keys are sha256 over the exact input "
                "the prompt is built from, so zero misses imply identical "
                "prompts; the byte-identity property is therefore exact on "
                "any backend, not a claim about model determinism."
            ) if arm_b_cache == "shared" else (
                "independent-caches protocol: both arms ran cold; "
                "byte-identity is exact on the deterministic mock and "
                "empirical on a live model (reported, not claimed)."
            ),
        },
        "priced_comparison": {
            "label": "MODELED-on-measured",
            "token_source": getattr(backend, "token_source", ""),
            "measured_distinct_keys": len(per_key_tokens),
            "measured_prompt_tokens": arm_a["prompt_tokens"],
            "measured_completion_tokens": arm_a["completion_tokens"],
            "prices": {
                "price_in_per_mt": price_in_per_mt,
                "price_out_per_mt": price_out_per_mt,
                "cache_read_rate": cache_read_rate,
                "note": (
                    "parameterized list-price stand-ins ($/M tokens); the "
                    "serving gateway's actual pricing is unknown.  The "
                    "provider prompt-cache column is MODELED: repeat draws "
                    "are billed as prompt-cache reads at cache_read_rate x "
                    "the input price (default 0.1, after Anthropic's "
                    "published prompt-caching economics); agentisa_cache "
                    "and the token counts are measured."
                ),
            },
            "modes": modes,
            "gptcache": (
                "GPTCache integration is out of scope this pass; no "
                "external-cache head-to-head is claimed."
            ),
        },
    }


def arm_a_classify_cache(arm_a_summary: Dict[str, Any]) -> Any:
    """The result cache of arm (a)'s classify step.

    ``run_arm`` stashes the stream's classify-step cache on the summary
    under a private key so the shared-cache protocol can wire arm (b)
    to it without serializing it.
    """
    return arm_a_summary["_classify_cache"]


# ---------------------------------------------------------------------------
# Top-level validation
# ---------------------------------------------------------------------------

async def run_validation(backend: Any, *, tasks: int, seeds: int,
                         zipf_levels: List[float], population: int,
                         hot_threshold: int, arm_b_cache: str,
                         seed_base: int, price_in_per_mt: float,
                         price_out_per_mt: float, cache_read_rate: float,
                         wall_measured: bool,
                         backend_config: Optional[Dict[str, Any]] = None
                         ) -> Dict[str, Any]:
    is_real = isinstance(backend, RealGatewayBackend)
    data = []
    for zipf_s in zipf_levels:
        for seed in range(seeds):
            data.append(await run_zipf_level(
                backend, zipf_s=zipf_s, seed=seed, n_draws=tasks,
                population=population, hot_threshold=hot_threshold,
                arm_b_cache=arm_b_cache, seed_base=seed_base,
                price_in_per_mt=price_in_per_mt,
                price_out_per_mt=price_out_per_mt,
                cache_read_rate=cache_read_rate,
                wall_measured=wall_measured))

    verdict = {
        "byte_identical_all_realizations": all(
            d["byte_identity"]["all_realizations_identical"] for d in data),
        "arm_a_calls_match_expected": all(
            d["arm_a_unoptimized"]["calls_match_expected"] for d in data),
        "arm_b_calls_match_expected": all(
            d["arm_b_compiled_hotpath"]["calls_match_expected"] for d in data),
        "arm_b_zero_new_calls_when_shared": (
            all(d["arm_b_compiled_hotpath"]["model_calls"] == 0
                for d in data) if arm_b_cache == "shared" else None),
        "hot_path_adopted": all(
            d["arm_b_compiled_hotpath"]["optimized_form_adopted_at_realization"]
            is not None for d in data),
        "static_compile_dataflow_preserved": all(
            d["static_compile"]["pass"] for d in data),
    }
    checked = [v for v in verdict.values() if v is not None]
    return {
        "experiment": "compiler_hotpath_semantics_and_priced_baseline",
        "harness": "examples/paper_real_llm_compiler_test.py",
        "backend": "real" if is_real else "mock",
        "backend_config": backend_config or {},
        "config": {
            "scale_tasks": tasks, "seeds": seeds,
            "zipf_levels": list(zipf_levels), "task_population": population,
            "hot_path_threshold": hot_threshold, "arm_b_cache": arm_b_cache,
            "seed_base": seed_base, "lines_per_task": LINES_PER_TASK,
            "price_in_per_mt": price_in_per_mt,
            "price_out_per_mt": price_out_per_mt,
            "cache_read_rate": cache_read_rate,
        },
        "verdict": verdict,
        "all_pass": all(checked),
        "data": data,
        "notes": [
            "Cost columns are MODELED-on-measured: token counts are "
            "measured (gateway usage on real, template-size accounting "
            "on mock); the pricing overlay is a parameterized model, "
            "not a bill.",
            "GPTCache integration is out of scope this pass; no "
            "external-cache head-to-head is claimed.",
            "The result cache cannot thrash within a run: max_size "
            "100,000 exceeds the distinct-key count and ttl 3,600 s "
            "exceeds the run duration, so hit/miss decisions are a pure "
            "function of the draw sequence (computed without any "
            "backend and checked against both arms).",
            "Mock-mode wall accounting charges the paper's constant "
            "80 ms per simulated call; real-mode wall is measured.  No "
            "wall-clock value enters mock-mode output, which is "
            "byte-reproducible run to run.",
            "Instruction-execution savings come from compilation and "
            "hot-path adoption (measured per arm); model-call savings "
            "come from the result cache (measured on arm (a) and priced "
            "per mode).  The two attributions are reported separately.",
        ],
    }


def _fmt_cost(mode_result: Dict[str, Any]) -> str:
    return (f"calls={mode_result['calls']:<5} "
            f"in(full/read)={mode_result['input_tokens_full_price']}"
            f"/{mode_result['input_tokens_cache_read']:<5} "
            f"out={mode_result['output_tokens']:<5} "
            f"cost=${mode_result['cost_usd']:.6f}")


def print_report(result: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"AgentISA compiler + hot-path validation "
          f"(backend={result['backend']})")
    print("=" * 78)
    for cell in result["data"]:
        print(f"\nzipf={cell['zipf_s']} seed={cell['seed']}: "
              f"draws={cell['draws']} "
              f"distinct_keys={cell['distinct_cache_keys_expected']} "
              f"(expected, computed without any backend)")
        a = cell["arm_a_unoptimized"]
        b = cell["arm_b_compiled_hotpath"]
        print(f"  arm (a) unoptimized  : calls={a['model_calls']} "
              f"prompt_tokens={a['prompt_tokens']} "
              f"completion_tokens={a['completion_tokens']} "
              f"wall={a['accounted_wall_ms']:.0f}ms "
              f"instr_exec={a['instruction_executions']} "
              f"hit_rate={a['cache_hit_rate']:.1%}")
        print(f"  arm (b) compiled+hot : calls={b['model_calls']} "
              f"wall={b['accounted_wall_ms']:.0f}ms "
              f"instr_exec={b['instruction_executions']} "
              f"adopted_at={b['optimized_form_adopted_at_realization']} "
              f"eliminated={b['stream_eliminated_by_hot_path']}")
        sc = cell["static_compile"]
        print(f"  static compile       : NONE {sc['NONE']['effective']}"
              f" -> AGGRESSIVE {sc['AGGRESSIVE']['effective']} "
              f"(dataflow preserved={sc['AGGRESSIVE']['dataflow_preserved']})")
        ident = cell["byte_identity"]
        print(f"  byte-identity        : "
              f"{cell['draws'] - len(ident['mismatched_realizations'])}"
              f"/{cell['draws']} realizations identical, "
              f"arm-b new calls={ident['arm_b_new_model_calls']}")
        pc = cell["priced_comparison"]
        print(f"  priced ({pc['label']}, "
              f"cache_read_rate={result['config']['cache_read_rate']}):")
        for name in COST_MODES:
            marker = "  [MODELED]" if name == "provider_prompt_cache" else ""
            print(f"    {name:<22} {_fmt_cost(pc['modes'][name])}{marker}")
    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS' if result['all_pass'] else 'FAIL'}")
    for name, ok in result["verdict"].items():
        print(f"  {'PASS' if ok is not False else 'FAIL'}  {name}={ok}")
    if result["backend"] == "real":
        rt = result.get("gateway_round_trips", {})
        print(f"  gateway round trips  : preflight={rt.get('preflight')} "
              f"live_calls={rt.get('live_calls')} "
              f"total={rt.get('total')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentISA compiler/hot-path real-gateway validation "
                    "+ priced baseline (W1/W2)")
    parser.add_argument("--backend", choices=["real", "mock"], default="mock")
    parser.add_argument("--scale", choices=["tiny", "paper"], default="tiny")
    parser.add_argument("--tasks", type=int, default=None,
                        help="override the scale's draw count")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--zipf", nargs="+", type=float, default=None)
    parser.add_argument("--population", type=int, default=40,
                        help="task universe size (Zipf support)")
    parser.add_argument("--hot-threshold", type=int, default=5)
    parser.add_argument("--arm-b-cache", choices=["shared", "independent"],
                        default="shared")
    parser.add_argument("--cache-read-rate", type=float, default=0.1,
                        help="MODELED provider prompt-cache read discount on "
                             "input tokens (default 0.1, after Anthropic's "
                             "published prompt-caching economics)")
    parser.add_argument("--price-in-per-mt", type=float, default=3.0,
                        help="list-price stand-in, $/M input tokens")
    parser.add_argument("--price-out-per-mt", type=float, default=15.0,
                        help="list-price stand-in, $/M output tokens")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--ca-cert", type=str, default=DEFAULT_CA_CERT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    preset = SCALE_PRESETS[args.scale]
    tasks = args.tasks if args.tasks is not None else preset["tasks"]
    seeds = args.seeds if args.seeds is not None else preset["seeds"]
    zipf_levels = args.zipf if args.zipf is not None else preset["zipf_levels"]

    if args.backend == "mock":
        backend = DeterministicMockBackend()
        backend_config: Dict[str, Any] = {
            "call_latency_ms": MOCK_CALL_LATENCY_MS,
            "token_accounting": "chars/4 template-size",
        }
        wall_measured = False
    else:
        api_key = resolve_api_key()
        if not api_key:
            raise SystemExit("no API key: set ANTHROPIC_AUTH_TOKEN or "
                             "~/.claude/settings.json")
        backend = RealGatewayBackend(
            base_url=args.base_url, api_key=api_key, ca_cert=args.ca_cert,
            model=args.model, max_tokens=args.max_tokens)
        backend_config = {
            "base_url": args.base_url, "model": args.model,
            "max_tokens": args.max_tokens, "timeout_s": 90.0,
            "retries": "3 bounded (transient only), sequential httpx",
        }
        wall_measured = True

    async def _run() -> Dict[str, Any]:
        preflight: Optional[Dict[str, Any]] = None
        try:
            if args.backend == "real":
                # One preflight round trip: fail fast with a clear error.
                before = backend_snapshot(backend)
                t0 = time.perf_counter()
                pong = await backend.infer("Reply with exactly: pong")
                after = backend_snapshot(backend)
                preflight = {
                    "ok": True,
                    "latency_ms": round(
                        (time.perf_counter() - t0) * 1000.0, 1),
                    "reply": pong[:40],
                    "prompt_tokens": (after["input_tokens"]
                                      - before["input_tokens"]),
                    "completion_tokens": (after["output_tokens"]
                                          - before["output_tokens"]),
                }
                print(f"preflight ok: model={args.model} "
                      f"{preflight['latency_ms']:.0f}ms -> {pong[:40]!r}")
            result = await run_validation(
                backend, tasks=tasks, seeds=seeds, zipf_levels=zipf_levels,
                population=args.population, hot_threshold=args.hot_threshold,
                arm_b_cache=args.arm_b_cache, seed_base=args.seed_base,
                price_in_per_mt=args.price_in_per_mt,
                price_out_per_mt=args.price_out_per_mt,
                cache_read_rate=args.cache_read_rate,
                wall_measured=wall_measured, backend_config=backend_config)
            if preflight is not None:
                result["preflight"] = preflight
                result["gateway_round_trips"] = {
                    "preflight": 1,
                    "live_calls": backend.api_calls - 1,
                    "total": backend.api_calls,
                }
            return result
        finally:
            await backend.close()

    result = asyncio.run(_run())

    out_path = (Path(args.json) if args.json else
                Path(__file__).parent /
                f"paper_real_llm_compiler_results_{args.backend}.json")
    out_path.write_text(json.dumps(result, indent=1) + "\n")

    print_report(result)
    print(f"\nresults: {out_path}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
