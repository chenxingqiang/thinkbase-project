"""Abstract sandbox executor for running agent-generated code.

A sandbox executes *untrusted* code (code or commands produced by an
LLM agent) with a hard boundary around resources, network and the host
filesystem.  The concrete backends implement the same protocol:

* :class:`E2BSandboxExecutor` -- Firecracker microVM in the E2B cloud
  (strong kernel-level isolation, filesystem + network scoping);
* :class:`LocalSandboxExecutor` -- a resource-limited subprocess on the
  host, useful when no cloud key is available or for unit tests.

The protocol intentionally mirrors the E2B surface so backends stay
small and swappable via :func:`create_sandbox_executor`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SandboxResult:
    """Outcome of a single code/command execution inside a sandbox."""

    status: str  # "success" | "error" | "timeout"
    stdout: str = ""
    stderr: str = ""
    text: str = ""  # display-friendly primary output (e.g. code interpreter text)
    error: str = ""
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == "success"


class SandboxExecutor(ABC):
    """Base class for sandbox backends.

    Implementations must be cheap to construct and must not import their
    heavy dependency (e.g. the E2B SDK, ``ray``) until first use, so that
    importing ``agentflow.sandbox`` stays fast on every framework entry
    point (see the project's lazy-import convention).
    """

    def __init__(self, *, name: str = "sandbox", timeout: Optional[float] = None) -> None:
        self.name = name
        self.timeout = timeout

    @abstractmethod
    def run_code(
        self,
        code: str,
        *,
        language: str = "python",
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> SandboxResult:
        """Execute a snippet of code (Python by default) in the sandbox."""

    @abstractmethod
    def run_command(
        self,
        command: str,
        *,
        envs: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        workdir: Optional[str] = None,
    ) -> SandboxResult:
        """Execute a shell command in the sandbox."""

    @abstractmethod
    def write_file(self, path: str, data: bytes) -> None:
        """Write a file inside the sandbox filesystem."""

    @abstractmethod
    def read_file(self, path: str) -> bytes:
        """Read a file from the sandbox filesystem."""

    @abstractmethod
    def close(self) -> None:
        """Tear down the sandbox and release its resources."""

    def __enter__(self) -> "SandboxExecutor":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:  # best-effort cleanup
        try:
            self.close()
        except Exception:  # noqa: BLE001 - destructor must not raise
            pass
