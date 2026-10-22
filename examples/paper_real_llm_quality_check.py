#!/usr/bin/env python3
"""Real-gateway QUALITY check of the AgentISA classify step.

Section 5.7.1's "What this does and does not show" paragraph is explicit
about the gap this harness closes: on this alias the reasoning model's
replies bind at ``max_tokens`` in every call (as at ``max_tokens=64`` in
Section 5.7, and as a 1,500-token diagnostic still did), so the classify
step's severity values came from its deterministic fallback; what was
validated against the live model was the mechanism chain (key
computation, cache decisions, call and token accounting, artifact
identity), not classification quality.  No statement about live-model
output quality existed anywhere in the paper.

This harness measures exactly that missing layer, and only that layer:

* the SAME classify step the pipeline runs -- the identical
  ``CLASSIFY_PROMPT``, the identical per-index reply parser and the
  identical deterministic ``fallback_severity`` rule -- IMPORTED from
  ``examples/paper_real_llm_compiler_test.py`` so the mirror cannot
  drift (pinned by tests in ``tests/system/test_real_llm_quality_check.py``);
* a GENEROUS ``max_tokens`` (default 1500, ``--max-tokens``) so replies
  have room to finish: every call's RAW reply text and the gateway's
  ``stop_reason`` are recorded, so a reply that still binds at the
  ceiling is DETECTED (``stop_reason=max_tokens``,
  ``replies_bound_at_ceiling``) instead of silently degrading to
  fallback classifications -- the failure mode Section 5.7.1 hit;
* planted GOLD labels: each synthetic incident signature carries a gold
  severity from an explicit keyword rubric (recorded verbatim in the
  output's provenance), so three agreement rates are computable --
  model-vs-gold (quality), model-vs-rule-fallback (drift from the
  pipeline's own fallback) and rule-vs-gold (the no-model baseline).

Backends (``--backend``):

* ``real``           -- the same local Anthropic-protocol gateway as
  the existing harnesses (alias ``claude-glm-5.3-flash``, internal-CA
  TLS, ``trust_env=False`` sequential httpx, 90 s timeout, three
  bounded retries for transient errors); tokens are gateway-reported,
  latency is measured.
* ``mock``           -- ``DeterministicMockBackend``: same protocol and
  the same deterministic reply function as the compiler harness's mock
  (pinned by tests); ``stop_reason`` is a simulated ``end_turn``.
  Fully deterministic; no network.
* ``mock-truncated`` -- ``TruncatedMockBackend``: simulates the
  ceiling-binding alias (reply cut mid-line, ``stop_reason =
  max_tokens``) to demonstrate and test the detection path offline.

One live call per DISTINCT signature batch (repeats would be
result-cache hits, as in the Section 5.7.1 protocol); at the default
tiny scale (50 Zipf(1.0) draws over 40 tasks) that is 22 live calls --
the same 22 distinct signature batches the Section 5.7.1 reference run
measured tokens on.

What this still does NOT show (kept separate, per the paper's honesty
rules): the gold labels are a planted keyword rubric on synthetic
signatures, not human annotation of real logs -- agreement rates are
rubric-following diagnostics, not deployment-quality claims; no
accuracy threshold is claimed or gated (``all_pass`` gates only the
measurability preconditions: replies complete, claims parsed,
accounting intact); one gateway, one model, one seed, sequential, no
streaming.

Determinism: batches are generated from per-rank seeds, the draw
sequence from a seeded truncated-Zipf generator (same weight formula
as the microbenchmark suite), and the mock backends are pure
functions of their input; no wall-clock value enters mock-mode
output, which is byte-reproducible run to run.

Usage:
    ./venv/bin/python examples/paper_real_llm_quality_check.py \
        --backend mock
    ./venv/bin/python examples/paper_real_llm_quality_check.py \
        --backend real --scale tiny \
        --json examples/paper_real_llm_quality_results_real.json
"""

from __future__ import annotations

import argparse
import asyncio
import httpx
import hashlib
import json
import os
import random
import re
import sys
import time

_RETRIES = max(1, int(os.environ.get("GATEWAY_RETRY_ATTEMPTS", "3")))
_TIMEOUT = float(os.environ.get("GATEWAY_CALL_TIMEOUT", "90"))
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Make the repo package and the sibling example modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# The classify step's prompt, reply parser, fallback rule, cache-key
# convention, token accounting, gateway constants and API-key resolution
# are IMPORTED (not copied) from the Section 5.7.1 harness, so the
# quality measured here is the quality the pipeline's classify step
# would itself get, and the mirrors cannot drift (pinned by tests).
from paper_real_llm_compiler_test import (  # noqa: E402
    CLASSIFY_PROMPT,
    COMPONENTS,
    DEFAULT_BASE_URL,
    DEFAULT_CA_CERT,
    DEFAULT_MODEL,
    MOCK_CALL_LATENCY_MS,
    SEVERITIES,
    _account_tokens,
    cache_key_id,
    fallback_severity,
    parse_classification,
    resolve_api_key,
    zipf_draws,
)

# The generous completion ceiling this harness exists to use: Section 5.7
# bound replies at 64 tokens, Section 5.7.1 at 512 (and a 1,500-token
# diagnostic still bound on the reasoning alias); the default records
# whether 1500 suffices -- if replies still report
# stop_reason=max_tokens, raise --max-tokens and rerun.
DEFAULT_MAX_TOKENS = 1500

DEFAULT_BATCH_SIZE = 8

SCALE_PRESETS = {
    # tiny = the Section 5.7.1 protocol shape: 50 Zipf(1.0) draws over a
    # 40-task universe -> 22 distinct signature batches -> 22 live calls.
    "tiny": {"tasks": 50, "population": 40, "zipf_s": 1.0},
    "paper": {"tasks": 200, "population": 40, "zipf_s": 1.0},
}


# ---------------------------------------------------------------------------
# Planted gold labels: an explicit keyword rubric over the signature universe
# ---------------------------------------------------------------------------

# Ordered (keyword, severity) rules, most specific first; a signature's
# gold label is the first keyword it contains.  The rubric is recorded
# verbatim in the output provenance so the gold labels are auditable.
GOLD_KEYWORD_RULES: List[Tuple[str, str]] = [
    ("heartbeat lost", "critical"),
    ("data loss", "critical"),
    ("replica lag", "critical"),
    ("upstream 5xx", "high"),
    ("timeout waiting for lock", "high"),
    ("queue depth critical", "high"),
    ("token refresh failed", "medium"),
    ("connection reset", "medium"),
    ("schema mismatch", "medium"),
    ("cache miss storm", "medium"),
    ("latency above threshold", "low"),
    ("request completed", "low"),
    ("debug trace volume", "low"),
]

# Message pool per severity class: every message maps back to its class
# through the rubric (pinned by tests), and none of the keywords the
# pipeline's hash-bucket fallback bumps on -- so the two baselines are
# meaningfully different.
SEVERITY_MESSAGES: Dict[str, List[str]] = {
    "critical": [
        "worker heartbeat lost",
        "db data loss detected",
        "replica lag beyond recovery",
    ],
    "high": [
        "upstream 5xx",
        "timeout waiting for lock",
        "queue depth critical",
    ],
    "medium": [
        "token refresh failed",
        "connection reset by peer",
        "schema mismatch on write",
        "cache miss storm",
    ],
    "low": [
        "latency above threshold",
        "request completed",
        "debug trace volume high",
    ],
}


def gold_severity(signature: str) -> str:
    """Planted gold label: first rubric keyword the signature contains.

    Synthetic ground truth on synthetic signatures -- not human
    annotation of real logs (stated in the provenance) -- so
    agreement-vs-gold is a rubric-following diagnostic.
    """
    for keyword, severity in GOLD_KEYWORD_RULES:
        if keyword in signature:
            return severity
    raise ValueError(
        f"signature is not covered by the planted gold rubric: {signature!r}")


def signature_batch(rank: int,
                    batch_size: int = DEFAULT_BATCH_SIZE) -> List[str]:
    """Deterministic incident-signature batch for task ``rank``.

    One signature per severity class is planted first (every batch's
    gold labels cover all four classes, checked in the verdict), the
    rest are seeded fills; the result is sorted and duplicate-free,
    mirroring how the pipeline's classify step sees a batch (sorted
    distinct ``component|message`` signatures over the paper's
    service components).
    """
    if batch_size < len(SEVERITIES):
        raise ValueError(
            f"batch_size must be >= {len(SEVERITIES)} so every batch "
            f"plants one signature per severity class")
    universe_size = (len(COMPONENTS)
                     * sum(len(m) for m in SEVERITY_MESSAGES.values()))
    if batch_size > universe_size:
        # The duplicate-free picker draws from a finite universe; past
        # its size it would never find an unused signature.
        raise ValueError(
            f"batch_size must be <= the signature universe size "
            f"({universe_size} = components x rubric messages)")
    rng = random.Random(30_000 + rank)
    used: set = set()

    def pick(severity: str) -> str:
        while True:
            signature = (f"{rng.choice(COMPONENTS)}|"
                         f"{rng.choice(SEVERITY_MESSAGES[severity])}")
            if signature not in used:
                used.add(signature)
                return signature

    planted = [pick(severity) for severity in SEVERITIES]
    fills = [pick(rng.choice(SEVERITIES))
             for _ in range(batch_size - len(SEVERITIES))]
    return sorted(planted + fills)


# ---------------------------------------------------------------------------
# Per-call outcome protocol (shared by all backends)
# ---------------------------------------------------------------------------

@dataclass
class CallOutcome:
    """One backend round trip, recorded for quality provenance.

    ``stop_reason`` distinguishes a complete reply (``end_turn``) from
    one that bound at the token ceiling (``max_tokens``) -- the
    failure mode Section 5.7.1 hit, which plain string-returning
    backends cannot surface.
    """

    reply: str
    stop_reason: Optional[str]
    input_tokens: int
    completion_tokens: int
    latency_ms: float


class DeterministicMockBackend:
    """Deterministic stand-in with the per-call outcome protocol.

    Reply function, token accounting (chars/4 template sizes) and the
    constant 80 ms simulated latency (``wall_ms`` is accounted, never
    slept) are behaviorally identical to the compiler harness's mock
    (pinned by tests); the additions are the simulated
    ``stop_reason=end_turn`` and the per-call token counts the
    quality records need.
    """

    def __init__(self) -> None:
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_ms = 0.0
        self.latencies_ms: List[float] = []
        self.token_source = "chars/4 template-size accounting (mock)"

    async def infer(self, payload: str) -> CallOutcome:
        reply = self._mock_reply(payload)
        in_tok, out_tok = _account_tokens(payload), _account_tokens(reply)
        self.api_calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.wall_ms += MOCK_CALL_LATENCY_MS
        self.latencies_ms.append(MOCK_CALL_LATENCY_MS)
        return CallOutcome(reply=reply, stop_reason="end_turn",
                           input_tokens=in_tok, completion_tokens=out_tok,
                           latency_ms=MOCK_CALL_LATENCY_MS)

    @staticmethod
    def _mock_reply(payload: str) -> str:
        """Deterministic reply: one ``<index>: <severity>`` line per
        signature line found in the prompt (the fallback rule's own
        severities, mirroring the compiler harness's mock stand-in)."""
        lines = []
        for m in re.finditer(r"^\s*(\d+): ([a-z]+)\|(.+)$", payload, re.MULTILINE):
            signature = f"{m.group(2)}|{m.group(3)}"
            lines.append(f"{m.group(1)}: {fallback_severity(signature)}")
        return "\n".join(lines) if lines else "0: medium"

    async def close(self) -> None:
        pass


class TruncatedMockBackend(DeterministicMockBackend):
    """Simulates the ceiling-binding alias Section 5.7.1 hit.

    The reply is cut to its first third, mid-line, and the outcome's
    ``stop_reason`` reports ``max_tokens`` -- exactly how a
    ceiling-bound completion looks to the parser: early signature lines
    may still parse while the tail (and its claims) never arrives, so
    the parse rate drops and the missing classifications fall back to
    the deterministic rule.  Used to demonstrate and test, fully
    offline, that the harness detects the mode.
    """

    async def infer(self, payload: str) -> CallOutcome:
        full = self._mock_reply(payload)
        cut = full[: max(1, len(full) // 3)]
        in_tok, out_tok = _account_tokens(payload), _account_tokens(cut)
        self.api_calls += 1
        self.input_tokens += in_tok
        self.output_tokens += out_tok
        self.wall_ms += MOCK_CALL_LATENCY_MS
        self.latencies_ms.append(MOCK_CALL_LATENCY_MS)
        return CallOutcome(reply=cut, stop_reason="max_tokens",
                           input_tokens=in_tok, completion_tokens=out_tok,
                           latency_ms=MOCK_CALL_LATENCY_MS)


class RealGatewayBackend:
    """Real ``/v1/messages`` round trips recording raw replies.

    Same gateway conventions as the Section 5.7/5.7.1 harnesses (base
    URL, internal-CA TLS with ``trust_env=False`` to bypass system
    proxies, ``x-api-key`` + ``anthropic-version`` headers, sequential
    httpx, 90 s timeout, three bounded retries for transient errors,
    temperature 0).  Unlike the bench's string-returning client, its
    ``infer`` returns a per-call :class:`CallOutcome` -- the raw reply
    text, the gateway's ``stop_reason`` (``end_turn`` vs
    ``max_tokens``) and usage -- because the quality check must
    distinguish a complete classification from one that bound at the
    token ceiling, and must record what the model actually said.
    """

    def __init__(self, *, base_url: str, api_key: str, ca_cert: Optional[str],
                 model: str, max_tokens: int, timeout: float = _TIMEOUT,
                 temperature: float = 0.0,
                 max_payload_chars: int = 4000) -> None:
        # Imported lazily so mock-only runs never touch httpx at all.
        import httpx
        self._httpx = httpx
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_payload_chars = max_payload_chars
        self._client = httpx.AsyncClient(
            verify=ca_cert or True,
            trust_env=False,  # bypass any system proxy for the local gateway
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=0),  # rotate connections across retries
        )
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_ms = 0.0
        self.latencies_ms: List[float] = []
        self.token_source = "gateway usage field (real)"

    async def infer(self, payload: str) -> CallOutcome:
        body: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content":
                          payload[: self.max_payload_chars]}],
            "temperature": self.temperature,
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        last_error: Optional[Exception] = None
        for attempt in range(_RETRIES):
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
                if attempt < _RETRIES - 1:  # transient: retry with backoff
                    await asyncio.sleep(1.0 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"gateway unreachable after {_RETRIES} attempts: {last_error!r}"
                )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"gateway rejected request: {resp.status_code} "
                    f"{resp.text[:300]}"
                )
            data = resp.json()
            usage = data.get("usage") or {}
            # Text blocks joined verbatim (thinking blocks skipped), the
            # raw reply recorded as the gateway produced it -- even if
            # empty (all budget spent before any text block).
            texts = [block.get("text", "")
                     for block in data.get("content", [])
                     if block.get("type") == "text"]
            reply = "".join(texts)
            in_tok = int(usage.get("input_tokens", 0))
            out_tok = int(usage.get("output_tokens", 0))
            self.api_calls += 1
            self.wall_ms += elapsed_ms
            self.latencies_ms.append(elapsed_ms)
            self.input_tokens += in_tok
            self.output_tokens += out_tok
            return CallOutcome(reply=reply,
                               stop_reason=data.get("stop_reason"),
                               input_tokens=in_tok,
                               completion_tokens=out_tok,
                               latency_ms=elapsed_ms)
        raise RuntimeError(f"gateway unreachable after {_RETRIES} attempts: {last_error!r}")

    async def close(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Reply parsing with claim provenance
# ---------------------------------------------------------------------------

def parse_model_claims(raw: str,
                       signatures: List[str]) -> Dict[str, Optional[str]]:
    """Per-signature parsed claim, or None where the reply is silent.

    Same expected reply format (and the same per-index regex) as the
    pipeline's ``parse_classification`` -- one ``<index>: <severity>``
    line per signature -- but instead of silently filling gaps with the
    fallback rule, it returns ``None`` for signatures the reply never
    mentioned, so each classification's provenance (model claim vs
    fallback fill) can be recorded.  Filling the ``None`` entries with
    ``fallback_severity`` reproduces ``parse_classification`` exactly
    (pinned by tests).
    """
    claims: Dict[str, Optional[str]] = {}
    for n, signature in enumerate(signatures):
        m = re.search(rf"(?<!\d){n}\s*[:=\-]\s*(low|medium|high|critical)\b",
                      raw, re.IGNORECASE)
        claims[signature] = m.group(1).lower() if m else None
    return claims


# ---------------------------------------------------------------------------
# The quality run: one live call per distinct batch, everything recorded
# ---------------------------------------------------------------------------

async def run_quality_check(backend: Any, *, tasks: int, zipf_s: float,
                            population: int, seed_base: int, batch_size: int,
                            max_tokens: int, wall_measured: bool,
                            backend_config: Optional[Dict[str, Any]] = None,
                            progress: bool = False) -> Dict[str, Any]:
    """Run the classify step over a seeded draw sequence and measure.

    Draws ``tasks`` task ranks from the same seeded truncated-Zipf
    generator as the microbenchmark suite, then makes ONE live call per
    DISTINCT signature batch (repeats would be result-cache hits, as in
    the Section 5.7.1 protocol); the batch for a rank is deterministic,
    its gold labels planted by the keyword rubric.  Every call's raw
    reply, stop reason and token counts are recorded, and the three
    agreement rates (model-vs-gold, model-vs-rule-fallback,
    rule-vs-gold) are computed per batch and overall.
    """
    if isinstance(backend, RealGatewayBackend):
        backend_kind = "real"
    elif isinstance(backend, TruncatedMockBackend):
        backend_kind = "mock-truncated"
    else:
        backend_kind = "mock"

    # Backend-independent accounting baseline: differences against this
    # snapshot make the summary robust to calls the backend already
    # made (e.g. the real-mode preflight).
    calls0 = backend.api_calls
    in0, out0, wall0 = backend.input_tokens, backend.output_tokens, backend.wall_ms

    draws = zipf_draws(seed_base, tasks, population, zipf_s)
    distinct_ranks: List[int] = []
    seen: set = set()
    for rank in draws:
        if rank not in seen:
            seen.add(rank)
            distinct_ranks.append(rank)

    records: List[Dict[str, Any]] = []
    t_run0 = time.perf_counter()
    for i, rank in enumerate(distinct_ranks):
        batch = signature_batch(rank, batch_size)
        gold = {s: gold_severity(s) for s in batch}
        rule = {s: fallback_severity(s) for s in batch}
        prompt = CLASSIFY_PROMPT.format(signature_lines="\n".join(
            f"{n}: {s}" for n, s in enumerate(batch)))
        outcome = await backend.infer(prompt)

        claims = parse_model_claims(outcome.reply, batch)
        # Runtime mirror pin: filling the unclaimed entries with the
        # fallback rule must reproduce the pipeline's own
        # classification exactly (same regex, same rule), so the
        # measured quality is the quality the pipeline would get.
        assert {s: (c if c is not None else rule[s])
                for s, c in claims.items()} == parse_classification(
                    outcome.reply, batch)
        per_signature: List[Dict[str, Any]] = []
        for n, s in enumerate(batch):
            claim = claims[s]
            final = claim if claim is not None else rule[s]
            per_signature.append({
                "index": n,
                "signature": s,
                "gold": gold[s],
                "rule_fallback": rule[s],
                "model_claim": claim,
                "final": final,
                "claim_source": ("model" if claim is not None
                                 else "rule_fallback"),
                "final_matches_gold": final == gold[s],
                "claim_matches_gold": (None if claim is None
                                       else claim == gold[s]),
            })
        parsed = sum(1 for r in per_signature if r["model_claim"] is not None)
        matched = sum(1 for r in per_signature if r["final_matches_gold"])
        claimed = [r for r in per_signature if r["model_claim"] is not None]
        count = len(batch)

        records.append({
            "batch_id": cache_key_id(batch),
            "task_rank": rank,
            "signature_count": count,
            "signatures": batch,
            "gold": gold,
            "rule_fallback": rule,
            "model_final": {r["signature"]: r["final"]
                            for r in per_signature},
            "per_signature": per_signature,
            "parsed_claims": parsed,
            "fallback_filled": count - parsed,
            "parse_rate": round(parsed / count, 4) if count else 0.0,
            "final_agreement_vs_gold":
                round(matched / count, 4) if count else 0.0,
            "claims_agreement_vs_gold": (
                round(sum(1 for r in claimed
                          if r["claim_matches_gold"]) / len(claimed), 4)
                if claimed else None),
            "final_agreement_vs_rule_fallback": round(
                sum(1 for r in per_signature
                    if r["final"] == r["rule_fallback"]) / count, 4
            ) if count else 0.0,
            "raw_reply": outcome.reply,
            "raw_reply_sha256": hashlib.sha256(
                outcome.reply.encode("utf-8")).hexdigest(),
            "stop_reason": outcome.stop_reason,
            "reply_bound_at_ceiling": outcome.stop_reason == "max_tokens",
            "prompt_chars": len(prompt),
            "prompt_sha256": hashlib.sha256(
                prompt.encode("utf-8")).hexdigest(),
            "input_tokens": outcome.input_tokens,
            "completion_tokens": outcome.completion_tokens,
            "latency_ms": round(outcome.latency_ms, 1),
        })
        if progress:
            print(f"  [batch {i + 1}/{len(distinct_ranks)}] rank={rank} "
                  f"stop={outcome.stop_reason} "
                  f"reply_chars={len(outcome.reply)} "
                  f"parsed={parsed}/{count}", flush=True)

    # Overall agreement accounting over every recorded signature.
    all_rows = [row for record in records
                for row in record["per_signature"]]
    n_sig = len(all_rows)
    parsed_total = sum(1 for r in all_rows if r["model_claim"] is not None)
    claimed_rows = [r for r in all_rows if r["model_claim"] is not None]
    matched_total = sum(1 for r in all_rows if r["final_matches_gold"])
    rule_matched_total = sum(1 for r in all_rows
                             if r["final"] == r["rule_fallback"])
    rule_vs_gold_total = sum(1 for r in all_rows
                             if r["rule_fallback"] == r["gold"])
    confusion: Dict[str, Dict[str, int]] = {}
    per_gold_class: Dict[str, Dict[str, Any]] = {}
    for severity in SEVERITIES:
        rows = [r for r in all_rows if r["gold"] == severity]
        per_gold_class[severity] = {
            "signatures": len(rows),
            "final_matches_gold": sum(1 for r in rows
                                      if r["final_matches_gold"]),
        }
        per_gold_class[severity]["agreement_rate"] = round(
            per_gold_class[severity]["final_matches_gold"] / len(rows), 4
        ) if rows else 0.0
    for row in all_rows:
        confusion.setdefault(row["gold"], {})
        confusion[row["gold"]][row["final"]] = (
            confusion[row["gold"]].get(row["final"], 0) + 1)

    model_calls = backend.api_calls - calls0
    summary: Dict[str, Any] = {
        "draws": tasks,
        "distinct_batches": len(records),
        "model_calls": model_calls,
        "signatures_total": n_sig,
        "parsed_claims": parsed_total,
        "fallback_filled": n_sig - parsed_total,
        "parse_rate": round(parsed_total / n_sig, 4) if n_sig else 0.0,
        "replies_bound_at_ceiling": sum(
            1 for r in records if r["reply_bound_at_ceiling"]),
        "final_agreement_vs_gold":
            round(matched_total / n_sig, 4) if n_sig else 0.0,
        "claims_agreement_vs_gold": (
            round(sum(1 for r in claimed_rows
                      if r["claim_matches_gold"]) / len(claimed_rows), 4)
            if claimed_rows else None),
        "final_agreement_vs_rule_fallback":
            round(rule_matched_total / n_sig, 4) if n_sig else 0.0,
        "rule_fallback_agreement_vs_gold":
            round(rule_vs_gold_total / n_sig, 4) if n_sig else 0.0,
        "per_gold_class": per_gold_class,
        "confusion_vs_gold": confusion,
        "input_tokens": backend.input_tokens - in0,
        "completion_tokens": backend.output_tokens - out0,
        "accounted_wall_ms": round(backend.wall_ms - wall0, 1),
        "token_source": getattr(backend, "token_source", ""),
    }
    if wall_measured:
        summary["measured_wall_ms"] = round(
            (time.perf_counter() - t_run0) * 1000.0, 1)

    # The verdict gates only the MEASURABILITY preconditions: every
    # distinct batch paid exactly one call, every call's raw reply and
    # every signature's gold label are recorded, no reply bound at the
    # token ceiling, and at least one model claim parsed.  Agreement
    # rates themselves are reported measurements -- no accuracy
    # threshold is claimed or gated.
    verdict = {
        "at_least_one_batch": bool(records),
        "one_call_per_distinct_batch": (
            model_calls == len(records) and bool(records)),
        "raw_reply_recorded_for_every_call": all(
            isinstance(r["raw_reply"], str) and r["raw_reply_sha256"]
            for r in records),
        "gold_labels_for_every_signature": all(
            set(r["gold"]) == set(r["signatures"]) for r in records),
        "planted_gold_classes_covered_per_batch": all(
            set(r["gold"].values()) == set(SEVERITIES) for r in records),
        "replies_not_bound_at_ceiling": all(
            not r["reply_bound_at_ceiling"] for r in records),
        "model_claims_parsed": parsed_total > 0,
    }
    checked = [v for v in verdict.values() if v is not None]

    return {
        "experiment": "classify_step_quality_check",
        "harness": "examples/paper_real_llm_quality_check.py",
        "backend": backend_kind,
        "backend_config": backend_config or {},
        "config": {
            "tasks": tasks,
            "zipf_s": zipf_s,
            "task_population": population,
            "seed_base": seed_base,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "call_protocol": ("one live call per distinct signature batch; "
                              "repeats would be result-cache hits (the "
                              "Section 5.7.1 protocol)"),
        },
        "provenance": {
            "gold_labels": (
                "planted at batch generation from an explicit keyword "
                "rubric (gold_rubric below): synthetic ground truth on "
                "synthetic signatures, not human annotation of real logs"
            ),
            "gold_rubric": [list(rule) for rule in GOLD_KEYWORD_RULES],
            "rule_fallback": (
                "the classify step's deterministic fallback "
                "(paper_real_llm_compiler_test.fallback_severity, "
                "imported -- identical function), used by the pipeline "
                "whenever a reply line does not parse"
            ),
            "prompt": (
                "the pipeline's CLASSIFY_PROMPT verbatim (imported from "
                "paper_real_llm_compiler_test): identical prompt to the "
                "Section 5.7.1 classify step; prompt_sha256 recorded per "
                "call"
            ),
            "parser": (
                "same per-index regex as parse_classification (mirror "
                "pinned by tests); each signature's claim_source records "
                "whether its severity came from the model reply "
                "('model') or the fallback rule ('rule_fallback')"
            ),
            "model_reply": (
                "raw reply text recorded verbatim per call with sha256; "
                "stop_reason from the gateway (end_turn vs max_tokens)"
            ),
            "tokens": getattr(backend, "token_source", ""),
            "latency": ("measured per call (real) / accounted constant "
                        f"{MOCK_CALL_LATENCY_MS:.0f} ms (mock)"),
            "draw_sequence": (
                "seeded truncated-Zipf draws (same weight formula as the "
                "microbenchmark suite) over the fixed task population; "
                "batches are per-rank deterministic and sorted"
            ),
        },
        "summary": summary,
        "verdict": verdict,
        "all_pass": all(checked),
        "data": records,
        "notes": [
            "all_pass gates only the measurability preconditions (one "
            "call per distinct batch, raw replies and gold labels "
            "recorded, no reply bound at the token ceiling, at least one "
            "model claim parsed); agreement rates are reported "
            "measurements -- no accuracy threshold is claimed or gated.",
            "Gold labels are a planted keyword rubric on synthetic "
            "signatures (recorded verbatim in provenance), not human "
            "annotation of real logs; agreement-vs-gold is a "
            "rubric-following diagnostic, not a deployment-quality claim.",
            "If replies report stop_reason=max_tokens, those replies' "
            "classifications are fallback-derived -- exactly the Section "
            "5.7.1 failure mode; raise --max-tokens and rerun before "
            "reading the agreement rates as model quality.",
            "The reply parser accepts the classify step's documented "
            "'<index>: <severity>' line format (same regex as the "
            "pipeline's fallback-filling parse); replies that do not "
            "follow it are counted as unparsed, with each missing "
            "classification attributed to the fallback rule.",
            "Single gateway, one model alias, one seed, sequential, no "
            "streaming; multi-model, multi-seed and real-deployment "
            "behavior remain untested.",
        ],
    }


# ---------------------------------------------------------------------------
# Reporting and CLI
# ---------------------------------------------------------------------------

def print_report(result: Dict[str, Any]) -> None:
    print("=" * 78)
    print(f"AgentISA classify-step QUALITY check "
          f"(backend={result['backend']})")
    print("=" * 78)
    config = result["config"]
    s = result["summary"]
    print(f"\ndraws={config['tasks']} population={config['task_population']} "
          f"zipf={config['zipf_s']} batch_size={config['batch_size']} "
          f"max_tokens={config['max_tokens']}")
    print(f"distinct_batches={s['distinct_batches']} "
          f"model_calls={s['model_calls']}")
    for record in result["data"]:
        claims = (f"{record['claims_agreement_vs_gold']:.1%}"
                  if record["claims_agreement_vs_gold"] is not None
                  else "n/a")
        print(f"  batch {record['batch_id']} rank={record['task_rank']:>3} "
              f"sigs={record['signature_count']} "
              f"parse={record['parse_rate']:.0%} "
              f"vs_gold={record['final_agreement_vs_gold']:.1%} "
              f"claims_vs_gold={claims} "
              f"vs_rule={record['final_agreement_vs_rule_fallback']:.1%} "
              f"stop={record['stop_reason']}")
    print(f"\nsummary: signatures={s['signatures_total']} "
          f"parsed_claims={s['parsed_claims']} "
          f"parse_rate={s['parse_rate']:.1%}")
    print(f"  final agreement vs gold           : "
          f"{s['final_agreement_vs_gold']:.1%} "
          f"(rule fallback alone: {s['rule_fallback_agreement_vs_gold']:.1%})")
    if s["claims_agreement_vs_gold"] is not None:
        print(f"  model claims agreement vs gold    : "
              f"{s['claims_agreement_vs_gold']:.1%} "
              f"(over {s['parsed_claims']} parsed claims)")
    else:
        print("  model claims agreement vs gold    : n/a (no claims parsed)")
    print(f"  final agreement vs rule fallback : "
          f"{s['final_agreement_vs_rule_fallback']:.1%}")
    print(f"  replies bound at ceiling         : "
          f"{s['replies_bound_at_ceiling']}/{s['distinct_batches']}")
    for severity in SEVERITIES:
        cls = s["per_gold_class"][severity]
        print(f"  gold={severity:<9} n={cls['signatures']:<4} "
              f"final_matches={cls['final_matches_gold']:<4} "
              f"rate={cls['agreement_rate']:.1%}")
    print("\n" + "=" * 78)
    print(f"VERDICT: {'PASS' if result['all_pass'] else 'FAIL'}")
    for name, ok in result["verdict"].items():
        print(f"  {'PASS' if ok is not False else 'FAIL'}  {name}={ok}")
    if result["backend"] == "real":
        rt = result.get("gateway_round_trips", {})
        print(f"  gateway round trips  : preflight={rt.get('preflight')} "
              f"live_calls={rt.get('live_calls')} "
              f"total={rt.get('total')}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="AgentISA classify-step real-model QUALITY check: "
                    "generous max_tokens, raw replies recorded, agreement "
                    "vs planted gold and the rule fallback")
    parser.add_argument("--backend",
                        choices=["mock", "mock-truncated", "real"],
                        default="mock")
    parser.add_argument("--scale", choices=["tiny", "paper"], default="tiny")
    parser.add_argument("--tasks", type=int, default=None,
                        help="override the scale's draw count")
    parser.add_argument("--population", type=int, default=None,
                        help="override the scale's task universe size")
    parser.add_argument("--zipf-s", type=float, default=None,
                        help="override the scale's Zipf skew")
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULT_BATCH_SIZE,
                        help="signatures per batch (>= 4, one planted per "
                             "severity class)")
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
                        help="generous completion ceiling (default 1500; "
                             "Section 5.7 used 64, 5.7.1 used 512)")
    parser.add_argument("--base-url", type=str, default=DEFAULT_BASE_URL)
    parser.add_argument("--ca-cert", type=str, default=DEFAULT_CA_CERT)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--json", type=str, default=None)
    args = parser.parse_args(argv)

    if args.batch_size < len(SEVERITIES):
        parser.error(f"--batch-size must be >= {len(SEVERITIES)} "
                     f"(one signature planted per severity class)")

    preset = SCALE_PRESETS[args.scale]
    tasks = (args.tasks if args.tasks is not None else preset["tasks"])
    population = (args.population if args.population is not None
                  else preset["population"])
    zipf_s = args.zipf_s if args.zipf_s is not None else preset["zipf_s"]

    if args.backend == "mock":
        backend = DeterministicMockBackend()
        backend_config: Dict[str, Any] = {
            "reply_function": ("deterministic fallback-severity echo, "
                               "identical to the compiler harness mock "
                               "(pinned by tests)"),
            "call_latency_ms": MOCK_CALL_LATENCY_MS,
            "token_accounting": "chars/4 template-size",
            "stop_reason": "end_turn (simulated)",
        }
        wall_measured = False
    elif args.backend == "mock-truncated":
        backend = TruncatedMockBackend()
        backend_config = {
            "reply_function": "deterministic fallback-severity echo",
            "call_latency_ms": MOCK_CALL_LATENCY_MS,
            "token_accounting": "chars/4 template-size",
            "simulated_failure": ("reply cut to its first third, "
                                  "stop_reason=max_tokens -- the "
                                  "ceiling-binding mode Section 5.7.1 hit"),
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
            "max_tokens": args.max_tokens, "temperature": 0.0,
            "timeout_s": 90.0,
            "retries": "3 bounded (transient only), sequential httpx",
        }
        wall_measured = True

    async def _run() -> Dict[str, Any]:
        preflight: Optional[Dict[str, Any]] = None
        try:
            if isinstance(backend, RealGatewayBackend):
                # One preflight round trip: fail fast with a clear error.
                t0 = time.perf_counter()
                pong = await backend.infer("Reply with exactly: pong")
                preflight = {
                    "ok": True,
                    "latency_ms": round(
                        (time.perf_counter() - t0) * 1000.0, 1),
                    "reply": pong.reply[:40],
                    "stop_reason": pong.stop_reason,
                    "completion_tokens": pong.completion_tokens,
                }
                print(f"preflight ok: model={args.model} "
                      f"{preflight['latency_ms']:.0f}ms -> "
                      f"{pong.reply[:40]!r} "
                      f"(stop={pong.stop_reason})")
            result = await run_quality_check(
                backend, tasks=tasks, zipf_s=zipf_s, population=population,
                seed_base=args.seed_base, batch_size=args.batch_size,
                max_tokens=args.max_tokens, wall_measured=wall_measured,
                backend_config=backend_config,
                progress=isinstance(backend, RealGatewayBackend))
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
                f"paper_real_llm_quality_results_{args.backend}.json")
    out_path.write_text(json.dumps(result, indent=1) + "\n")

    print_report(result)
    print(f"\nresults: {out_path}")
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
