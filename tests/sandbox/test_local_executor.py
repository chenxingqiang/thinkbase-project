"""Tests for the local subprocess sandbox backend (runs for real)."""

import os
import time

import pytest

from agentflow.sandbox import LocalSandboxExecutor, SandboxResult


@pytest.fixture
def sandbox():
    sbx = LocalSandboxExecutor(timeout=10.0)
    yield sbx
    sbx.close()


def test_run_code_success(sandbox):
    result = sandbox.run_code("print('hello from sandbox')")
    assert result.status == "success"
    assert "hello from sandbox" in result.stdout
    assert result.execution_time >= 0


def test_run_code_computation(sandbox):
    result = sandbox.run_code("print(sum(range(101)))")
    assert result.status == "success"
    assert result.stdout.strip() == "5050"


def test_run_code_syntax_error(sandbox):
    result = sandbox.run_code("def broken(:\n    pass")
    assert result.status == "error"
    assert result.error  # SyntaxError surfaced


def test_run_code_runtime_error(sandbox):
    result = sandbox.run_code("raise ValueError('boom')")
    assert result.status == "error"
    assert "boom" in result.stderr or "boom" in result.error


def test_run_code_timeout(sandbox):
    result = sandbox.run_code("import time; time.sleep(30)", timeout=1)
    assert result.status == "timeout"
    assert "timed out" in result.error


def test_run_code_unsupported_language(sandbox):
    result = sandbox.run_code("echo hi", language="bash")
    assert result.status == "error"
    assert "only supports python" in result.error


def test_run_command_success(sandbox):
    result = sandbox.run_command("echo command-ok")
    assert result.status == "success"
    assert "command-ok" in result.stdout


def test_run_command_failure(sandbox):
    result = sandbox.run_command("exit 3")
    assert result.status == "error"
    assert "exited with code 3" in result.error


def test_run_command_timeout(sandbox):
    result = sandbox.run_command("sleep 30", timeout=1)
    assert result.status == "timeout"


def test_write_and_read_file(sandbox):
    sandbox.write_file("data/note.txt", b"agent data")
    assert sandbox.read_file("data/note.txt") == b"agent data"


def test_envs_are_passed(sandbox):
    result = sandbox.run_code(
        "import os; print(os.environ.get('SANDBOX_MARKER', 'missing'))",
        envs={"SANDBOX_MARKER": "present"},
    )
    assert result.status == "success"
    assert "present" in result.stdout


def test_close_cleans_workdir(sandbox):
    sandbox.write_file("probe.txt", b"x")  # force workdir creation
    workdir = sandbox._workdir
    assert workdir is not None and os.path.isdir(workdir)
    sandbox.close()
    assert not os.path.isdir(workdir)


def test_context_manager(sandbox):
    with LocalSandboxExecutor(timeout=5) as sbx:
        result = sbx.run_code("print('ctx')")
        assert result.status == "success"
    assert sbx._closed


def test_destructor_is_safe():
    sbx = LocalSandboxExecutor(timeout=5)
    sbx.run_code("print('x')")
    sbx.__del__()  # must not raise
    assert sbx._closed
