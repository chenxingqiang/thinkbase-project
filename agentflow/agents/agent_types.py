"""Agent types module."""

from enum import Enum, auto
from typing import List, Dict, Any, Optional, Union

from ..core.base_types import AgentType, AgentMode, AgentStatus, MessageRole, MessageType
from ..core.model_config import ModelConfig
# ``AgentConfig`` is the canonical pydantic model (name/type/model/
# workflow with validation and model_validate/model_dump); the legacy
# dataclass that used to live in ``agents.agent_config`` is kept only
# for direct historical imports of that module.
from ..core.agent_config import AgentConfig

__all__ = [
    'AgentType',
    'AgentMode',
    'AgentStatus',
    'ModelConfig',
    'AgentConfig',
    'MessageRole',
    'MessageType'
]
