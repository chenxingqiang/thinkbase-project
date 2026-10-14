"""Sandbox execution for untrusted, agent-generated code.

Chooses the backend via :func:`create_sandbox_executor`:

* ``e2b``   -- Firecracker microVM in the E2B cloud (strong isolation,
  requires ``E2B_API_KEY``);
* ``local`` -- resource-limited subprocess on the host (development /
  testing fallback; not for untrusted production code).

All heavy dependencies are imported lazily on first use, keeping
``import agentflow.sandbox`` cheap for every framework entry point.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import SandboxExecutor, SandboxResult
from .local_executor import LocalSandboxExecutor

__all__ = [
    "SandboxExecutor",
    "SandboxResult",
    "E2BSandboxExecutor",
    "LocalSandboxExecutor",
    "create_sandbox_executor",
]


def create_sandbox_executor(
    backend: Optional[str] = None,
    *,
    api_key: Optional[str] = None,
    **kwargs: object,
) -> SandboxExecutor:
    """Instantiate a sandbox backend.

    ``backend`` is one of ``"e2b"`` or ``"local"`` (default ``"local"``).
    When ``"auto"`` is requested, E2B is used iff ``E2B_API_KEY`` is set,
    otherwise the local backend is returned.

    Extra keyword arguments are forwarded to the chosen backend's
    constructor (e.g. ``template=``, ``allow_internet_access=`` for E2B;
    ``python=``, ``timeout=`` for both).
    """
    backend = backend or "local"
    if backend == "auto":
        backend = "e2b" if (api_key or os.environ.get("E2B_API_KEY")) else "local"

    if backend == "e2b":
        from .e2b_executor import E2BSandboxExecutor  # noqa: PLC0415

        return E2BSandboxExecutor(api_key=api_key, **kwargs)  # type: ignore[arg-type]
    if backend == "local":
        return LocalSandboxExecutor(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown sandbox backend {backend!r}; choose from 'e2b', 'local', 'auto'"
    )


def __getattr__(name: str):
    # Keep the heavy E2B module out of the eager import path.
    if name == "E2BSandboxExecutor":
        from .e2b_executor import E2BSandboxExecutor  # noqa: PLC0415

        return E2BSandboxExecutor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
