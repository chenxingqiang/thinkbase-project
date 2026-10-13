"""LMP (Language Model Program) type definitions (vendored)."""

from enum import Enum


class LMPType(str, Enum):
    """Classification of language-model programs.

    The classification mirrors the original ELL library's ``LMPType``
    and is used as metadata when routing calls through the ELL
    integration layer.
    """

    PROMPT = "prompt"
    TOOL = "tool"
    FUNCTION = "function"
    AGENT = "agent"
    PIPELINE = "pipeline"
    WORKFLOW = "workflow"


__all__ = ["LMPType"]
