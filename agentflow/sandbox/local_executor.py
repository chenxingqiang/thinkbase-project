"""Local subprocess-backed sandbox executor.

Runs agent-generated code/commands as a resource-limited subprocess on the
host.  This is the **weakest** isolation tier (no kernel boundary), intended
for local development, unit tests, and as a fallback when no cloud key is
available.  It still enforces:

* a hard timeout (the process is killed when exceeded);
* a private working directory that is torn down on ``close()``;
* a clean environment (only explicitly provided variables are passed).

It must **not** be used to run genuinely untrusted code in production.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, Optional

from .base import SandboxExecutor, SandboxResult

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0


class LocalSandboxExecutor(SandboxExecutor):
    """Resource-limited subprocess backend.

    Parameters
    ----------
    python:
        Interpreter used for ``run_code``.  Defaults to ``sys.executable``.
    """

    def __init__(
        self,
        *,
        python: Optional[str] = None,
        name: str = "local",
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout or DEFAULT_TIMEOUT)
        self.python = python or sys.executable
        self._workdir: Optional[str] = None
        self._closed = False

    def _ensure_workdir(self) -> str:
        if self._workdir is None:
            self._workdir = tempfile.mkdtemp(prefix=f"agentflow_{self.name}_")
        return self._workdir

    @staticmethod
    def _base_env(envs: Optional[Dict[str, str]]) -> Dict[str, str]:
        # Start from a minimal environment so the sandbox does not inherit
        # secrets from the host (PATH is kept so interpreters can spawn).
        base = {"PATH": os.environ.get("PATH", "/usr/bin:/bin:/usr/sbin:/sbin")}
        if envs:
            base.update(envs)
        return base

    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        if language not in ("python", "py"):
            return SandboxResult(
                status="error",
                error=f"local backend only supports python, got {language!r}",
            )
        started = time.perf_counter()
        workdir = self._ensure_workdir()
        script_path = os.path.join(workdir, "_agent_code.py")
        with open(script_path, "w") as handle:
            handle.write(code)
        try:
            proc = subprocess.run(
                [self.python, script_path],
                cwd=workdir,
                env=self._base_env(envs),
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
        except subprocess.TimeoutExpired as error:
            return SandboxResult(
                status="timeout",
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                error=f"timed out after {(timeout or self.timeout):.1f}s",
                execution_time=time.perf_counter() - started,
            )
        except Exception as error:  # noqa: BLE001
            return SandboxResult(
                status="error",
                error=f"{type(error).__name__}: {error}",
                execution_time=time.perf_counter() - started,
            )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            return SandboxResult(
                status="error",
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=f"process exited with code {proc.returncode}",
                execution_time=elapsed,
                metadata={"exit_code": proc.returncode},
            )
        return SandboxResult(
            status="success",
            stdout=proc.stdout,
            stderr=proc.stderr,
            execution_time=elapsed,
            metadata={"exit_code": 0},
        )

    def run_command(
        self,
        command: str,
        *,
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        workdir: Optional[str] = None,
    ) -> SandboxResult:
        started = time.perf_counter()
        cwd = workdir or self._ensure_workdir()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=self._base_env(envs),
                capture_output=True,
                text=True,
                timeout=timeout or self.timeout,
            )
        except subprocess.TimeoutExpired as error:
            return SandboxResult(
                status="timeout",
                stdout=error.stdout or "",
                stderr=error.stderr or "",
                error=f"timed out after {(timeout or self.timeout):.1f}s",
                execution_time=time.perf_counter() - started,
            )
        except Exception as error:  # noqa: BLE001
            return SandboxResult(
                status="error",
                error=f"{type(error).__name__}: {error}",
                execution_time=time.perf_counter() - started,
            )
        elapsed = time.perf_counter() - started
        if proc.returncode != 0:
            return SandboxResult(
                status="error",
                stdout=proc.stdout,
                stderr=proc.stderr,
                error=f"command exited with code {proc.returncode}",
                execution_time=elapsed,
                metadata={"exit_code": proc.returncode},
            )
        return SandboxResult(
            status="success",
            stdout=proc.stdout,
            stderr=proc.stderr,
            execution_time=elapsed,
            metadata={"exit_code": 0},
        )

    def write_file(self, path: str, data: bytes) -> None:
        workdir = self._ensure_workdir()
        full = path if os.path.isabs(path) else os.path.join(workdir, path)
        os.makedirs(os.path.dirname(full) or workdir, exist_ok=True)
        with open(full, "wb") as handle:
            handle.write(data)

    def read_file(self, path: str) -> bytes:
        workdir = self._ensure_workdir()
        full = path if os.path.isabs(path) else os.path.join(workdir, path)
        with open(full, "rb") as handle:
            return handle.read()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._workdir is not None and os.path.isdir(self._workdir):
            shutil.rmtree(self._workdir, ignore_errors=True)
            self._workdir = None
