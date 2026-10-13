"""Tests for the vendored ELL package (agentflow.ell).

Covers message type coercion, serialization round-trips, the
ELLIntegration singleton lifecycle, decorator instrumentation, and the
ELLWorkflow dataclass used for registration.
"""

import asyncio
import pytest

from agentflow.ell import (
    ELLIntegration,
    ELLWorkflow,
    LMPType,
    Message,
    MessageRole,
    MessageType,
    ell_integration,
)
from agentflow.ell.types import messages_to_dicts


@pytest.fixture(autouse=True)
def cleanup_ell():
    """Reset the singleton around each test."""
    ell_integration.cleanup()
    yield
    ell_integration.cleanup()


class TestMessageTypes:
    def test_string_role_coerced_to_enum(self):
        msg = Message(role="user", content="hi")
        assert msg.role is MessageRole.USER

    def test_unknown_role_string_preserved(self):
        msg = Message(role="custom_role", content="hi")
        assert msg.role == "custom_role"

    def test_default_type_is_text(self):
        msg = Message(role=MessageRole.USER, content="hi")
        assert msg.type is MessageType.TEXT

    def test_none_metadata_normalized_to_dict(self):
        msg = Message(role=MessageRole.USER, content="hi", metadata=None)
        assert msg.metadata == {}

    def test_to_from_dict_roundtrip(self):
        msg = Message(
            role=MessageRole.ASSISTANT,
            content="payload",
            metadata={"k": "v"},
            type=MessageType.RESULT,
        )
        restored = Message.from_dict(msg.to_dict())
        assert restored.role is MessageRole.ASSISTANT
        assert restored.content == "payload"
        assert restored.metadata == {"k": "v"}
        assert restored.type is MessageType.RESULT

    def test_len_of_textual_content(self):
        assert len(Message(role=MessageRole.USER, content="12345")) == 5
        assert len(Message(role=MessageRole.USER, content=None)) == 0

    def test_messages_to_dicts(self):
        msgs = [Message(role=MessageRole.USER, content="a")]
        dicts = messages_to_dicts(msgs)
        assert dicts[0]["role"] == "user"


class TestLMPType:
    def test_lmp_type_members(self):
        assert LMPType.TOOL.value == "tool"
        assert LMPType.PROMPT.value == "prompt"
        assert LMPType.AGENT.value == "agent"


class TestELLIntegrationSingleton:
    def test_singleton_identity(self):
        assert ELLIntegration() is ELLIntegration()
        assert ELLIntegration() is ell_integration

    def test_cleanup_resets_state(self):
        ell_integration.register_workflow("w1", object())
        assert "w1" in ell_integration.list_workflows()
        ell_integration.cleanup()
        assert ell_integration.list_workflows() == []
        assert ell_integration.get_messages() == []

    def test_workflow_registration_roundtrip(self):
        wf = ELLWorkflow(name="wf", description="d")
        ell_integration.register_workflow(wf.id, wf)
        assert wf.id in ell_integration.list_workflows()
        ell_integration.unregister_workflow(wf.id)
        assert wf.id not in ell_integration.list_workflows()

    @pytest.mark.asyncio
    async def test_process_message_returns_result(self):
        msg = Message(role=MessageRole.USER, content="q")
        result = await ell_integration.process_message(msg)
        assert result.role is MessageRole.ASSISTANT
        assert result.type is MessageType.RESULT
        assert result.content == "q"
        assert len(ell_integration.get_messages()) == 1

    def test_mode_config_configure_and_get(self):
        ell_integration.configure({"simple": {"model": "m1", "timeout": 5.0}})
        cfg = ell_integration.get_mode_config("simple")
        assert cfg["model"] == "m1"
        assert cfg["timeout"] == 5.0
        assert ell_integration.get_mode_config("nonexistent") == {}


class TestDecorators:
    @pytest.mark.asyncio
    async def test_with_ell_success(self):
        @ell_integration.with_ell(mode="simple")
        async def ok_fn():
            return "done"

        assert await ok_fn() == "done"
        metrics = ell_integration.get_metrics()
        assert metrics["ok_fn"]["calls"] == 1
        assert metrics["ok_fn"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_with_ell_error_recorded(self):
        @ell_integration.with_ell(mode="simple")
        async def bad_fn():
            raise ValueError("boom")

        with pytest.raises(ValueError):
            await bad_fn()
        metrics = ell_integration.get_metrics()
        assert metrics["bad_fn"]["errors"] == 1
        assert metrics["bad_fn"]["calls"] == 0

    @pytest.mark.asyncio
    async def test_track_function_timing(self):
        @ell_integration.track_function()
        async def slow_fn():
            await asyncio.sleep(0.05)
            return "ok"

        assert await slow_fn() == "ok"
        metrics = ell_integration.get_metrics()
        assert metrics["slow_fn"]["calls"] == 1
        assert metrics["slow_fn"]["total_time"] >= 0.05


class TestELLWorkflow:
    def test_workflow_defaults(self):
        wf = ELLWorkflow(name="n", description="d")
        assert wf.steps == []
        assert wf.metadata == {}
        assert len(wf.id) == 36  # uuid4

    def test_add_and_get_step(self):
        wf = ELLWorkflow(name="n", description="d")
        wf.add_step({"op": "extract"})
        assert wf.get_step(0) == {"op": "extract"}
        assert wf.get_step(1) is None

    def test_to_dict(self):
        wf = ELLWorkflow(name="n", description="d", steps=[{"op": "x"}])
        d = wf.to_dict()
        assert d["name"] == "n"
        assert d["steps"] == [{"op": "x"}]
        assert d["id"] == wf.id
