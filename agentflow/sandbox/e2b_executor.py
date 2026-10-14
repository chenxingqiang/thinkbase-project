"""E2B-backed sandbox executor.

Runs agent-generated code inside a Firecracker microVM in the E2B cloud
(https://e2b.dev).  Strong kernel-level isolation from the host; network
egress is disabled by default (``allow_internet_access=False``) unless the
caller explicitly opts in.

Requires an ``E2B_API_KEY`` environment variable (or an explicit
``api_key`` argument).  The SDK is imported lazily so that importing this
module never pays the SDK import cost.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from .base import SandboxExecutor, SandboxResult

logger = logging.getLogger(__name__)

E2B_API_KEY_ENV = "E2B_API_KEY"


class E2BError(RuntimeError):
    """Raised for E2B configuration/authentication problems."""


class E2BSandboxExecutor(SandboxExecutor):
    """Sandbox backend executing code in an E2B Firecracker microVM.

    Parameters
    ----------
    api_key:
        E2B API key.  Defaults to the ``E2B_API_KEY`` environment variable.
    template:
        E2B sandbox template name or id.  Defaults to the SDK's base
        template (Python interpreter available).
    allow_internet_access:
        Whether the sandbox may reach the public internet.  ``False`` by
        default: a sandbox for untrusted agent code should be egress-scoped.
    sandbox_timeout:
        Maximum lifetime of the sandbox itself in seconds (E2B kills idle
        sandboxes).  Defaults to the SDK default.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        template: Optional[str] = None,
        allow_internet_access: bool = False,
        sandbox_timeout: Optional[int] = None,
        name: str = "e2b",
        timeout: Optional[float] = None,
    ) -> None:
        super().__init__(name=name, timeout=timeout)
        self.api_key = api_key or os.environ.get(E2B_API_KEY_ENV)
        self.template = template
        self.allow_internet_access = allow_internet_access
        self.sandbox_timeout = sandbox_timeout
        self._sandbox: Any = None
        self._e2b_module: Any = None

    # -- lifecycle ---------------------------------------------------------

    def _ensure_sdk(self) -> Any:
        """Import the E2B SDK on first use (lazy import convention)."""
        if self._e2b_module is None:
            try:
                from e2b_code_interpreter import Sandbox  # noqa: PLC0415
            except ImportError as error:  # pragma: no cover - env-dependent
                raise E2BError(
                    "e2b-code-interpreter SDK is not installed; "
                    "run `pip install e2b-code-interpreter`"
                ) from error
            self._e2b_module = Sandbox
        return self._e2b_module

    def _ensure_key(self) -> None:
        if not self.api_key:
            raise E2BError(
                f"E2B_API_KEY is not set; export it or pass api_key= to "
                f"{type(self).__name__} (get one at https://e2b.dev)"
            )

    def _ensure_sandbox(self) -> Any:
        if self._sandbox is None or not self._sandbox.is_running():
            self._ensure_key()
            sandbox_cls = self._ensure_sdk()
            logger.info("Creating E2B sandbox (internet access=%s)", self.allow_internet_access)
            self._sandbox = sandbox_cls.create(
                api_key=self.api_key,
                template=self.template,
                timeout=self.sandbox_timeout,
                allow_internet_access=self.allow_internet_access,
            )
        return self._sandbox

    # -- SandboxExecutor protocol -----------------------------------------

    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        started = time.perf_counter()
        # Configuration/authentication problems (E2BError) propagate to the
        # caller; only *execution* failures are surfaced as an error result.
        sbx = self._ensure_sandbox()
        try:
            execution = sbx.run_code(
                code,
                language=language if language != "python" else None,
                envs=envs,
                timeout=timeout or self.timeout,
            )
        except Exception as error:  # noqa: BLE001 - surface as result
            return SandboxResult(
                status="error",
                error=f"{type(error).__name__}: {error}",
                execution_time=time.perf_counter() - started,
            )
        elapsed = time.perf_counter() - started
        if execution.error:
            return SandboxResult(
                status="error",
                stdout=getattr(execution, "stdout", "") or "",
                stderr=getattr(execution, "stderr", "") or "",
                text=getattr(execution, "text", "") or "",
                error=str(execution.error),
                execution_time=elapsed,
            )
        return SandboxResult(
            status="success",
            stdout=getattr(execution, "stdout", "") or "",
            stderr=getattr(execution, "stderr", "") or "",
            text=getattr(execution, "text", "") or "",
            execution_time=elapsed,
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
        sbx = self._ensure_sandbox()  # config errors (E2BError) propagate
        try:
            proc = sbx.commands.run(
                command,
                envs=envs,
                timeout=timeout or self.timeout,
                cwd=workdir,
            )
        except Exception as error:  # noqa: BLE001
            return SandboxResult(
                status="error",
                error=f"{type(error).__name__}: {error}",
                execution_time=time.perf_counter() - started,
            )
        elapsed = time.perf_counter() - started
        exit_code = getattr(proc, "exit_code", 0)
        stdout = getattr(proc, "stdout", "") or ""
        stderr = getattr(proc, "stderr", "") or ""
        if exit_code != 0:
            return SandboxResult(
                status="error",
                stdout=stdout,
                stderr=stderr,
                error=f"command exited with code {exit_code}",
                execution_time=elapsed,
                metadata={"exit_code": exit_code},
            )
        return SandboxResult(
            status="success",
            stdout=stdout,
            stderr=stderr,
            execution_time=elapsed,
            metadata={"exit_code": exit_code},
        )

    def write_file(self, path: str, data: bytes) -> None:
        sbx = self._ensure_sandbox()
        sbx.files.write(path, data)

    def read_file(self, path: str) -> bytes:
        sbx = self._ensure_sandbox()
        content = sbx.files.read(path)
        if isinstance(content, str):
            return content.encode("utf-8")
        return content

    def close(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.kill()
            except Exception as error:  # noqa: BLE001
                logger.debug("E2B sandbox close failed: %s", error)
            self._sandbox = None
