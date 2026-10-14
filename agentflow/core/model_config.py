"""Model configuration re-export.

``ModelConfig`` is canonically defined in :mod:`agentflow.core.agent_config`
as the LLM model configuration (provider/name/temperature/...).  This
module previously hosted a two-field dataclass (``model_name``/``version``)
that shadowed the LLM configuration at several import sites
(``workflow_engine``, ``agents.agent``, the API server and the unit
tests); it now re-exports the canonical class so every import path
resolves to the same object.

The legacy ``model_name``/``version`` fields are still accepted as
extras and mapped onto ``name`` for backwards compatibility.
"""

from typing import Any

from .agent_config import ModelConfig as _CanonicalModelConfig


class ModelConfig(_CanonicalModelConfig):
    """Backwards-compatible alias of the canonical LLM ModelConfig.

    Accepts the legacy ``model_name`` kwarg as an alias of ``name``.
    """

    def __init__(self, **data: Any) -> None:
        if "model_name" in data and "name" not in data:
            data["name"] = data.pop("model_name")
        super().__init__(**data)


__all__ = ["ModelConfig"]
