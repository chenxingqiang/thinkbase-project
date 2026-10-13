"""Compatibility shim: ELL integration for the core package.

Re-exports the vendored ELL integration from :mod:`agentflow.ell` so
that historical imports of ``agentflow.core.ell_integration`` keep
working.
"""

from agentflow.ell.integration import ELLIntegration, ell_integration

__all__ = ["ELLIntegration", "ell_integration"]
