"""Vendored ELL package for AgentFlow.

A dependency-free, API-compatible subset of the ELL (Effective
Language Learning) library used by AgentFlow.  Importing from
``agentflow.ell`` provides the same symbols as the upstream ``ell``
package (``Message``, ``MessageRole``, ``MessageType``, ``LMPType``,
``ELLIntegration``) so call sites never need to change.
"""

from .lmp import LMPType
from .types import Message, MessageRole, MessageType, ContentBlock, ContentWrapper, MessageValidationError, system, user, assistant
from .integration import ELLIntegration, ell_integration
from .workflow import ELLWorkflow

__all__ = [
    "LMPType",
    "Message",
    "MessageRole",
    "MessageType",
    "ContentBlock",
    "ContentWrapper",
    "MessageValidationError",
    "system",
    "user",
    "assistant",
    "ELLIntegration",
    "ell_integration",
    "ELLWorkflow",
]
