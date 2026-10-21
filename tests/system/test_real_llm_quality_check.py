"""Mock-backend tests for the classify-step QUALITY harness.

The harness (``examples/paper_real_llm_quality_check.py``) closes the
gap Section 5.7.1's "What this does and does not show" paragraph is
explicit about: real-model classification quality was never validated
(replies bound at the token ceiling and every severity came from the
deterministic fallback).  It re-runs the SAME classify step (prompt,
parser, fallback rule imported from the compiler harness, pinned by
tests) at a generous ``max_tokens`` (default 1500), records every raw
reply with its ``stop_reason``, and reports agreement rates against
planted gold labels and the rule fallback.

These tests pin those properties on the deterministic mock backends
ONLY: no network, no live gateway.  The real-backend path is
exercised by the harness itself when the gateway is reachable.
"""

import asyncio
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

import pytest

# Make the harness modules importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "examples"))

from paper_real_llm_quality_check import (  # noqa: E402
    DEFAULT_MAX_TOKENS,
    DeterministicMockBackend,
    GOLD_KEYWORD_RULES,
    SEVERITY_MESSAGES,
    TruncatedMockBackend,
    CallOutcome,
    gold_severity,
    main,
    parse_model_claims,
    run_quality_check,
    signature_batch,
)

from paper_real_llm_compiler_test import (  # noqa: E402
    CLASSIFY_PROMPT,
    COMPONENTS,
    SEVERITIES,
    DeterministicMockBackend as CompilerDeterministicMockBackend,
    cache_key_id,
    fallback_severity,
    parse_classification,
)


def _run(backend, **overrides):
    params = dict(
        tasks=12, zipf_s=1.0, population=10, seed_base=1000,
        batch_size=6, max_tokens=DEFAULT_MAX_TOKENS, wall_measured=False,
    )
    params.update(overrides)
    return asyncio.run(run_quality_check(backend, **params))


class TestGoldRubric:
    """The planted gold labels: an explicit keyword rubric over the
    synthetic signature universe, independent of the pipeline's
    hash-bucket fallback rule."""

    def test_every_pool_message_maps_to_its_class(self):
        for severity, messages in SEVERITY_MESSAGES.items():
            for message in messages:
                assert gold_severity(f"api|{message}") == severity

    def test_signature_outside_the_rubric_is_rejected(self):
        with pytest.raises(ValueError):
            gold_severity("api|something the rubric does not cover")

    def test_rule_fallback_is_not_the_gold_rubric(self):
        """The two baselines must differ, else agreement-with-gold is
        vacuously the same as agreement-with-rule."""
        universe = [f"{c}|{m}" for c in COMPONENTS
                    for messages in SEVERITY_MESSAGES.values() for m in messages]
        agree = sum(1 for s in universe
                    if fallback_severity(s) == gold_severity(s))
        assert 0 < agree < len(universe)

    def test_batch_generation_is_deterministic_sorted_unique(self):
        first, second = signature_batch(7, 6), signature_batch(7, 6)
        assert first == second
        assert first == sorted(first)
        assert len(set(first)) == 6

    def test_batch_plants_every_gold_class(self):
        for rank in range(12):
            batch = signature_batch(rank, 6)
            assert {gold_severity(s) for s in batch} == set(SEVERITIES)

    def test_batch_size_below_class_count_rejected(self):
        with pytest.raises(ValueError):
            signature_batch(0, 3)

    def test_batch_size_above_signature_universe_rejected(self):
        """The duplicate-free picker draws from a finite universe
        (components x rubric messages); past that size it could never
        find an unused signature, so it must refuse loudly."""
        universe = (len(COMPONENTS)
                    * sum(len(m) for m in SEVERITY_MESSAGES.values()))
        with pytest.raises(ValueError):
            signature_batch(0, universe + 1)


class TestParseModelClaims:
    """Per-signature claim parsing: same regex semantics as the
    pipeline's ``parse_classification``, but with claim provenance
    (None = the reply never mentioned that signature)."""

    SIGS = ["api|upstream 5xx", "db|timeout waiting for lock",
            "auth|token refresh failed"]

    def test_full_reply(self):
        claims = parse_model_claims("0: high\n1: critical\n2: low", self.SIGS)
        assert claims == {self.SIGS[0]: "high", self.SIGS[1]: "critical",
                          self.SIGS[2]: "low"}

    def test_partial_reply_marks_unclaimed_as_none(self):
        claims = parse_model_claims("1: critical", self.SIGS)
        assert claims[self.SIGS[0]] is None
        assert claims[self.SIGS[1]] == "critical"
        assert claims[self.SIGS[2]] is None

    def test_garbage_and_empty_replies(self):
        for raw in ("the model rambles without classifying", ""):
            claims = parse_model_claims(raw, self.SIGS)
            assert all(v is None for v in claims.values())

    def test_severity_words_case_insensitive(self):
        claims = parse_model_claims("0: HIGH\n1: Critical\n2: LoW", self.SIGS)
        assert list(claims.values()) == ["high", "critical", "low"]

    @pytest.mark.parametrize("raw", [
        "0: high\n1: critical\n2: low",
        "1: critical",
        "Here you go:\n0: low\n2: medium\n(the rest omitted)",
        "",
        "no classification lines at all",
        "10: high\n11: low",  # indices of a longer batch must not leak
    ])
    def test_filled_claims_match_pipeline_parse_classification(self, raw):
        """Mirror pin: filling the unclaimed entries with the fallback
        rule must reproduce the pipeline's own classification exactly."""
        claims = parse_model_claims(raw, self.SIGS)
        filled = {s: c if c is not None else fallback_severity(s)
                  for s, c in claims.items()}
        assert filled == parse_classification(raw, self.SIGS)


class TestMockReplyMirror:
    def test_reply_matches_the_compiler_harness_mock(self):
        """The quality mock's reply function is behaviorally identical
        to the compiler harness's mock stand-in."""
        payload = ("Signatures:\n0: api|upstream 5xx\n"
                   "1: db|token refresh failed\n2: worker|heartbeat lost")
        assert (DeterministicMockBackend._mock_reply(payload)
                == CompilerDeterministicMockBackend._mock_reply(payload))


@pytest.fixture(scope="module")
def results():
    """One mock run: 12 Zipf(1.0) draws over a 10-task universe."""
    return _run(DeterministicMockBackend())


@pytest.fixture(scope="module")
def summary(results):
    return results["summary"]


class TestMockQualityRun:
    def test_one_live_call_per_distinct_batch(self, results, summary):
        assert summary["distinct_batches"] == len(results["data"]) == 6
        assert summary["model_calls"] == summary["distinct_batches"]
        assert results["config"]["tasks"] == 12

    def test_mock_replies_fully_parsed(self, summary):
        assert summary["parse_rate"] == 1.0
        assert summary["fallback_filled"] == 0
        assert summary["parsed_claims"] == summary["signatures_total"]

    def test_all_claims_sourced_from_the_model(self, results):
        for record in results["data"]:
            assert all(ps["claim_source"] == "model"
                       for ps in record["per_signature"])
            assert all(ps["model_claim"] is not None
                       for ps in record["per_signature"])

    def test_mock_final_agrees_with_rule_fallback_everywhere(
            self, results, summary):
        assert summary["final_agreement_vs_rule_fallback"] == 1.0
        assert all(record["final_agreement_vs_rule_fallback"] == 1.0
                   for record in results["data"])

    def test_mock_vs_gold_equals_rule_vs_gold(self, summary):
        """The mock echoes the fallback rule, so its final
        classifications cannot beat the rule's own gold agreement."""
        assert (summary["final_agreement_vs_gold"]
                == summary["rule_fallback_agreement_vs_gold"])

    def test_rule_vs_gold_baseline_strictly_between(self, summary):
        assert 0 < summary["rule_fallback_agreement_vs_gold"] < 1

    def test_replies_complete_on_mock(self, results, summary):
        assert all(record["stop_reason"] == "end_turn"
                   for record in results["data"])
        assert summary["replies_bound_at_ceiling"] == 0

    def test_batch_ids_mirror_pipeline_cache_keying(self, results):
        for record in results["data"]:
            assert record["batch_id"] == cache_key_id(record["signatures"])

    def test_prompts_are_the_pipeline_prompt(self, results):
        for record in results["data"]:
            prompt = CLASSIFY_PROMPT.format(signature_lines="\n".join(
                f"{n}: {sig}"
                for n, sig in enumerate(record["signatures"])))
            digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
            assert record["prompt_sha256"] == digest

    def test_raw_reply_recorded_verbatim_with_sha(self, results):
        for record in results["data"]:
            assert record["raw_reply"]
            assert record["raw_reply_sha256"] == hashlib.sha256(
                record["raw_reply"].encode("utf-8")).hexdigest()

    def test_gold_covers_all_classes_per_batch(self, results):
        for record in results["data"]:
            assert set(record["gold"]) == set(record["signatures"])
            assert set(record["gold"].values()) == set(SEVERITIES)

    def test_confusion_matrix_totals(self, summary):
        confusion = summary["confusion_vs_gold"]
        total = sum(n for row in confusion.values() for n in row.values())
        assert total == summary["signatures_total"]

    def test_verdict_and_all_pass(self, results):
        assert results["all_pass"] is True
        assert all(results["verdict"].values())
        assert results["verdict"]["one_call_per_distinct_batch"] is True
        assert results["verdict"]["replies_not_bound_at_ceiling"] is True


class _ScriptedBackend:
    """Test double speaking the harness's per-call outcome protocol."""

    def __init__(self, reply_fn):
        self.reply_fn = reply_fn
        self.api_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.wall_ms = 0.0
        self.token_source = "scripted (test)"

    async def infer(self, payload):
        self.api_calls += 1
        self.input_tokens += 10
        self.output_tokens += 10
        self.wall_ms += 1.0
        return CallOutcome(reply=self.reply_fn(payload),
                           stop_reason="end_turn",
                           input_tokens=10, completion_tokens=10,
                           latency_ms=1.0)

    async def close(self):
        pass


def _echo_gold(payload):
    lines = []
    for m in re.finditer(r"^\s*(\d+): ([a-z]+)\|(.+)$", payload, re.MULTILINE):
        lines.append(
            f"{m.group(1)}: {gold_severity(f'{m.group(2)}|{m.group(3)}')}")
    return "\n".join(lines)


class TestAgreementMath:
    """Hand-controlled backends: the agreement arithmetic must be
    exact, not approximate."""

    def test_gold_echoing_model_scores_full_agreement(self):
        res = _run(_ScriptedBackend(_echo_gold))
        s = res["summary"]
        assert s["parse_rate"] == 1.0
        assert s["final_agreement_vs_gold"] == 1.0
        assert s["claims_agreement_vs_gold"] == 1.0
        assert res["all_pass"] is True

    def test_unparseable_reply_falls_back_entirely(self):
        res = _run(_ScriptedBackend(
            lambda payload: "the model rambles and never classifies"))
        s = res["summary"]
        assert s["parsed_claims"] == 0
        assert s["fallback_filled"] == s["signatures_total"]
        assert s["parse_rate"] == 0.0
        assert s["claims_agreement_vs_gold"] is None
        # the final classifications are exactly the rule fallback's
        assert s["final_agreement_vs_rule_fallback"] == 1.0
        assert (s["final_agreement_vs_gold"]
                == s["rule_fallback_agreement_vs_gold"])
        # the run says nothing about the model: the verdict must fail
        assert res["verdict"]["model_claims_parsed"] is False
        assert res["all_pass"] is False


class TestCeilingDetection:
    """The defect this harness exists for: replies binding at the token
    ceiling must be detected, not silently degraded to fallback."""

    @pytest.fixture(scope="class")
    def truncated_results(self):
        return _run(TruncatedMockBackend())

    def test_all_replies_report_max_tokens_stop(self, truncated_results):
        for record in truncated_results["data"]:
            assert record["stop_reason"] == "max_tokens"
            assert record["reply_bound_at_ceiling"] is True

    def test_partial_parse_and_partial_fallback(self, truncated_results):
        s = truncated_results["summary"]
        assert 0 < s["parsed_claims"] < s["signatures_total"]
        assert s["fallback_filled"] > 0
        assert 0.0 < s["parse_rate"] < 1.0

    def test_ceiling_binding_detected_in_summary_and_verdict(
            self, truncated_results):
        s = truncated_results["summary"]
        assert s["replies_bound_at_ceiling"] == s["distinct_batches"]
        assert (truncated_results["verdict"]
                ["replies_not_bound_at_ceiling"] is False)
        assert truncated_results["all_pass"] is False


class TestProvenance:
    def test_provenance_block_complete(self, results):
        provenance = results["provenance"]
        for key in ("gold_labels", "gold_rubric", "rule_fallback",
                    "prompt", "parser", "model_reply", "tokens",
                    "latency", "draw_sequence"):
            assert provenance[key]
        assert provenance["gold_rubric"] == [
            list(rule) for rule in GOLD_KEYWORD_RULES]

    def test_provenance_names_the_pinned_mirrors(self, results):
        provenance = results["provenance"]
        assert "CLASSIFY_PROMPT" in provenance["prompt"]
        assert "fallback_severity" in provenance["rule_fallback"]

    def test_token_source_labeled(self, results):
        assert "template" in results["summary"]["token_source"].lower()

    def test_config_records_the_generous_ceiling(self, results):
        assert results["config"]["max_tokens"] == DEFAULT_MAX_TOKENS == 1500


def _block_network(monkeypatch):
    """Disable outbound networking at the socket layer (IPv4/IPv6);
    AF_UNIX stays available because the asyncio event loop uses a
    Unix-domain socketpair for its internal self-pipe."""
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


class TestCLIMockEndToEnd:
    def test_backend_mock_runs_end_to_end_no_network(
            self, tmp_path, monkeypatch):
        """The CLI must run the full tiny-scale protocol on the mock
        backend with outbound networking disabled at the socket
        layer."""
        _block_network(monkeypatch)
        out = tmp_path / "quality_mock.json"
        rc = main(["--backend", "mock", "--json", str(out)])
        assert rc == 0
        data = json.loads(out.read_text())
        assert data["backend"] == "mock"
        assert data["all_pass"] is True
        assert "preflight" not in data  # preflight is real-backend only
        assert data["config"]["tasks"] == 50
        assert data["config"]["max_tokens"] == 1500
        # tiny scale: 50 Zipf(1.0) draws over 40 tasks -> the same 22
        # distinct signature batches (hence 22 mock calls) the Section
        # 5.7.1 reference run measured tokens on
        assert data["summary"]["distinct_batches"] == 22
        assert len(data["data"]) == 22

    def test_backend_mock_truncated_exits_nonzero(self, tmp_path):
        """The simulated ceiling-binding backend must FAIL the run:
        mock mode is fully offline, and detection is testable."""
        out = tmp_path / "quality_truncated.json"
        rc = main(["--backend", "mock-truncated", "--json", str(out)])
        assert rc == 1
        data = json.loads(out.read_text())
        assert data["backend"] == "mock-truncated"
        assert data["all_pass"] is False
        assert (data["summary"]["replies_bound_at_ceiling"]
                == data["summary"]["distinct_batches"])

    def test_real_backend_requires_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.claude/settings.json
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        with pytest.raises(SystemExit):
            main(["--backend", "real", "--json", str(tmp_path / "x.json")])

    def test_mock_cli_output_byte_reproducible(self, tmp_path):
        outs = []
        for i in range(2):
            out = tmp_path / f"quality_{i}.json"
            assert main(["--backend", "mock", "--json", str(out)]) == 0
            outs.append(out.read_bytes())
        assert outs[0] == outs[1]


class TestMockByteReproducibility:
    def test_two_mock_runs_produce_identical_json(self):
        runs = []
        for _ in range(2):
            res = _run(DeterministicMockBackend(), tasks=8, population=8)
            runs.append(json.dumps(res, indent=1, sort_keys=True))
        assert runs[0] == runs[1]
