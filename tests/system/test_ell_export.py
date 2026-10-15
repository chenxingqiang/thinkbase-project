"""Tests for the AgentISA → ELL export bridge."""

import asyncio
import json
from pathlib import Path

import pytest

from agentflow.core.isa.ell_export import (
    describe_instruction,
    export_stream_to_ell,
    load_ell_trace,
    render_trace,
    save_ell_trace,
    stream_to_messages,
    stream_to_workflow,
)
from agentflow.ell import MessageRole, MessageType
from agentflow.ell.integration import ell_integration

import sys
sys.path.insert(0, "examples")
from system_task_demo import build_stream, RuleClassifierBackend  # noqa: E402


@pytest.fixture
def stream():
    """The demo's 7-instruction stream, executed once (real metrics)."""
    backend = RuleClassifierBackend()
    stream = build_stream(backend, [])
    return stream


@pytest.fixture(autouse=True)
def clean_integration():
    """Isolate the ELLIntegration singleton per test."""
    ell_integration.cleanup()
    yield
    ell_integration.cleanup()


class TestWorkflowExport:
    def test_workflow_has_one_step_per_instruction(self, stream):
        wf = stream_to_workflow(stream, name="t")
        assert wf.name == "t"
        assert len(wf.steps) == len(stream)
        assert [s["step"] for s in wf.steps] == list(range(len(stream)))
        assert wf.to_dict()["metadata"]["instruction_count"] == len(stream)

    def test_steps_carry_descriptors(self, stream):
        wf = stream_to_workflow(stream)
        ids = [s["id"] for s in wf.steps]
        assert "classify_signature" in ids
        cls = next(s for s in wf.steps if s["id"] == "classify_signature")
        assert cls["type"] == "LLM_INTERACTION"
        assert cls["side_effects"] is False

    def test_describe_instruction_shape(self, stream):
        d = describe_instruction(stream[0])
        assert set(d) == {
            "id", "name", "type", "description",
            "dependencies", "outputs", "side_effects",
        }


class TestMessageExport:
    def test_message_count_and_roles(self, stream):
        messages = stream_to_messages(stream)
        # system header + user/assistant pair per instruction
        assert len(messages) == 1 + 2 * len(stream)
        assert messages[0].role == MessageRole.SYSTEM
        for i, m in enumerate(messages[1:], start=1):
            if i % 2 == 1:
                assert m.role == MessageRole.USER
            else:
                assert m.role == MessageRole.ASSISTANT

    def test_unexecuted_instructions_marked(self, stream):
        messages = stream_to_messages(stream)
        assistant_records = [m for m in messages if m.role == MessageRole.ASSISTANT]
        for m in assistant_records:
            assert m.metadata["execution"]["executed"] is False
            assert "not executed" in str(m.content)

    def test_eliminated_flag_propagates(self, stream):
        messages = stream_to_messages(stream, eliminated=["check_syntax"])
        user_msgs = [m for m in messages if m.role == MessageRole.USER]
        guard = next(m for m in user_msgs
                     if m.metadata["instruction"]["id"] == "check_syntax")
        assert guard.metadata["eliminated"] is True
        others = [m for m in user_msgs if m.metadata["instruction"]["id"] != "check_syntax"]
        assert all(m.metadata["eliminated"] is False for m in others)

    def test_execution_summary_in_header(self, stream):
        messages = stream_to_messages(stream, execution_summary={"x": 1})
        assert messages[0].metadata["execution_summary"] == {"x": 1}


class TestIntegration:
    def test_export_records_in_singleton(self, stream):
        wf, messages = asyncio.run(export_stream_to_ell(
            stream, name="service-log-triage"))
        stored = ell_integration.get_messages()
        assert len(stored) == len(messages)
        assert wf.id in ell_integration.list_workflows()
        # The registered workflow object is the one we built.
        assert ell_integration._workflows[wf.id] is wf


class TestTracePersistence:
    def test_save_and_load_round_trip(self, stream, tmp_path: Path):
        wf = stream_to_workflow(stream, name="rt")
        messages = stream_to_messages(stream, eliminated=["check_syntax"])
        path = save_ell_trace(tmp_path / "trace.jsonl", wf, messages)

        loaded_wf, loaded_msgs = load_ell_trace(path)
        assert loaded_wf["name"] == "rt"
        assert loaded_wf["id"] == wf.id
        assert len(loaded_msgs) == len(messages)
        assert loaded_msgs[0]["role"] == "system"
        # JSON-serializable content check: every line parses.
        with path.open() as fh:
            records = [json.loads(line) for line in fh]
        assert records[0]["kind"] == "workflow"
        assert all(r["kind"] == "message" for r in records[1:])

    def test_load_rejects_trace_without_workflow(self, tmp_path: Path):
        bad = tmp_path / "bad.jsonl"
        bad.write_text(json.dumps({"kind": "message", "role": "user"}) + "\n")
        with pytest.raises(ValueError):
            load_ell_trace(bad)

    def test_render_trace_is_readable(self, stream):
        text = render_trace(stream_to_messages(stream))
        assert "[SYSTEM   ]" in text
        assert "[USER     ]" in text
        assert "[ASSISTANT]" in text
        assert "classify_signature" in text


class TestEndToEndWithExecution:
    def test_executed_stream_reports_real_metrics(self, tmp_path: Path):
        """After a real run, the trace's metrics are non-zero and sane."""
        import random
        from system_task_demo import (
            generate_corpus, inject_sandbox, run_stream, write_corpus,
        )
        from agentflow.sandbox.local_executor import LocalSandboxExecutor

        corpus = generate_corpus(random.Random(42), 2, 20)
        backend = RuleClassifierBackend()
        stream = build_stream(backend, sorted(corpus))
        sandbox = LocalSandboxExecutor(name="ell-test")
        try:
            write_corpus(sandbox, corpus)
            inject_sandbox(stream, sandbox)
            for _ in range(3):
                asyncio.run(run_stream(stream))
        finally:
            sandbox.close()

        wf, messages = asyncio.run(export_stream_to_ell(
            stream, name="e2e"))
        path = save_ell_trace(tmp_path / "t.jsonl", wf, messages)
        _, loaded = load_ell_trace(path)

        classify = next(
            r for r in loaded
            if r["metadata"].get("execution") and
            r["metadata"].get("instruction", {}).get("id") == "classify_signature"
        )
        exec_record = classify["metadata"]["execution"]
        assert exec_record["executed"] is True
        assert exec_record["total_executions"] == 3
        assert exec_record["cache_hits"] == 2
        assert exec_record["average_execution_time"] < 1.0  # seconds, not epoch
