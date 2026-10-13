"""Thin compatibility shim exposing the model providers.

The providers historically lived at the ``agentflow.providers`` path;
they now live in :mod:`agentflow.core.providers`. This module re-exports
the public provider classes and registers ``agentflow.providers.deepseek``
as an alias of ``agentflow.core.providers.deepseek`` so that legacy import
paths keep working::

    from agentflow.providers import DeepSeekProvider
    from agentflow.providers.deepseek import DeepSeekConfig
"""

import sys as _sys

from agentflow.core.providers import (
    DeepSeekChoice,
    DeepSeekConfig,
    DeepSeekMessage,
    DeepSeekProvider,
    DeepSeekResponse,
    DeepSeekUsage,
)
from agentflow.core.providers import deepseek as deepseek  # noqa: F401

__all__ = [
    "DeepSeekChoice",
    "DeepSeekConfig",
    "DeepSeekMessage",
    "DeepSeekProvider",
    "DeepSeekResponse",
    "DeepSeekUsage",
    "deepseek",
]

# Alias the submodule so that ``from agentflow.providers.deepseek import X``
# resolves. The alias is only registered when not already present, so a real
# ``agentflow/providers/`` package (if one is ever created) would take over.
_sys.modules.setdefault("agentflow.providers.deepseek", deepseek)
