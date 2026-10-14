"""Tests for the E2B sandbox backend.

Cloud execution requires a real ``E2B_API_KEY`` and is skipped otherwise;
the no-key error path and the lazy-import contract are tested for real.
"""

import pytest

from agentflow.sandbox import create_sandbox_executor
from agentflow.sandbox.e2b_executor import E2BError, E2BSandboxExecutor

pytestmark = pytest.mark.skipif(
    __import__("os").environ.get("E2B_API_KEY") is None,
    reason="E2B_API_KEY not set; cloud sandbox tests skipped",
)


def test_lazy_import_contract():
    """Importing agentflow.sandbox must not import the E2B SDK eagerly."""
    import sys

    assert "e2b_code_interpreter" not in sys.modules
    from agentflow.sandbox import E2BSandboxExecutor  # noqa: F401

    # Accessing the class still must not import the SDK.
    assert "e2b_code_interpreter" not in sys.modules


def test_requires_key_for_creation():
    executor = E2BSandboxExecutor(api_key=None)
    with pytest.raises(E2BError, match="E2B_API_KEY"):
        executor.run_code("print('x')")


def test_run_code_cloud(monkeypatch):
    key = pytest.importorskip("os").environ["E2B_API_KEY"]
    with E2BSandboxExecutor(api_key=key) as sandbox:
        result = sandbox.run_code("print(6 * 7)")
        assert result.status == "success"
        assert "42" in result.text or "42" in result.stdout


def test_run_code_error_cloud(monkeypatch):
    key = pytest.importorskip("os").environ["E2B_API_KEY"]
    with E2BSandboxExecutor(api_key=key) as sandbox:
        result = sandbox.run_code("raise ValueError('e2b boom')")
        assert result.status == "error"
        assert "boom" in (result.error or "") + (result.stderr or "")


def test_run_command_cloud(monkeypatch):
    key = pytest.importorskip("os").environ["E2B_API_KEY"]
    with E2BSandboxExecutor(api_key=key) as sandbox:
        result = sandbox.run_command("echo e2b-command-ok")
        assert result.status == "success"
        assert "e2b-command-ok" in result.stdout


def test_filesystem_cloud(monkeypatch):
    key = pytest.importorskip("os").environ["E2B_API_KEY"]
    with E2BSandboxExecutor(api_key=key) as sandbox:
        sandbox.write_file("/tmp/agent_note.txt", b"e2b file data")
        assert sandbox.read_file("/tmp/agent_note.txt") == b"e2b file data"


def test_factory_auto_cloud(monkeypatch):
    key = pytest.importorskip("os").environ["E2B_API_KEY"]
    executor = create_sandbox_executor("auto", api_key=key)
    assert type(executor).__name__ == "E2BSandboxExecutor"
