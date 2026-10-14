"""Tests for the sandbox backend factory (no cloud key required)."""

import pytest

from agentflow.sandbox import (
    LocalSandboxExecutor,
    create_sandbox_executor,
)


def test_default_backend_is_local():
    executor = create_sandbox_executor()
    assert isinstance(executor, LocalSandboxExecutor)


def test_explicit_local():
    executor = create_sandbox_executor("local", timeout=5)
    assert isinstance(executor, LocalSandboxExecutor)
    assert executor.timeout == 5


def test_auto_without_key_falls_back_to_local(monkeypatch):
    monkeypatch.delenv("E2B_API_KEY", raising=False)
    executor = create_sandbox_executor("auto")
    assert isinstance(executor, LocalSandboxExecutor)


def test_auto_with_key_uses_e2b(monkeypatch):
    monkeypatch.setenv("E2B_API_KEY", "test-key")
    executor = create_sandbox_executor("auto")
    assert type(executor).__name__ == "E2BSandboxExecutor"


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown sandbox backend"):
        create_sandbox_executor("nope")


def test_e2b_without_key_raises_on_use():
    executor = create_sandbox_executor("e2b", api_key=None)
    from agentflow.sandbox.e2b_executor import E2BError

    with pytest.raises(E2BError, match="E2B_API_KEY"):
        executor.run_code("print(1)")


def test_sandbox_result_properties():
    from agentflow.sandbox import SandboxResult

    ok = SandboxResult(status="success")
    assert ok.ok
    bad = SandboxResult(status="error")
    assert not bad.ok
