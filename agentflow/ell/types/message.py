"""Vendored ELL-compatible message types.

This module provides a dependency-free implementation of the message
types used by the ELL (Effective Language Learning) integration layer.
It keeps the API surface compatible with the original ``ell`` package
(``ell.types.message``) so that AgentFlow code can be migrated to the
real ELL library later without changes.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    """Role of a message author in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    """Kind of payload carried by a message."""

    TEXT = "text"
    RESULT = "result"
    ERROR = "error"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"


class ContentBlock(BaseModel):
    """Content block for structured message payloads.

    Mirrors the canonical ``ContentBlock`` in
    ``agentflow.core.workflow.workflow_types`` so structured content can
    flow through either type system.
    """

    type: str = "text"
    content: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class MessageValidationError(ValueError):
    """Raised when a message cannot be constructed or validated."""


def system(content: Any, **metadata: Any) -> "Message":
    """Build a system-role message."""
    return Message(role=MessageRole.SYSTEM, content=content,
                   metadata=dict(metadata) if metadata else {})


def user(content: Any, **metadata: Any) -> "Message":
    """Build a user-role message."""
    return Message(role=MessageRole.USER, content=content,
                   metadata=dict(metadata) if metadata else {})


def assistant(content: Any, **metadata: Any) -> "Message":
    """Build an assistant-role message."""
    return Message(role=MessageRole.ASSISTANT, content=content,
                   metadata=dict(metadata) if metadata else {})


class ContentWrapper:
    """Groups content blocks into a single renderable unit."""

    def __init__(self, blocks: Optional[List[ContentBlock]] = None):
        self.blocks: List[ContentBlock] = list(blocks or [])

    def add_block(self, block: ContentBlock) -> None:
        """Append a content block."""
        self.blocks.append(block)

    @property
    def content(self) -> str:
        """Rendered content of all blocks."""
        return "\n".join(str(b.content) for b in self.blocks)

    def __str__(self) -> str:
        return self.content

    def __len__(self) -> int:
        return len(self.blocks)

    def __iter__(self):
        return iter(self.blocks)


@dataclass
class Message:
    """A single message in an LLM conversation.

    Args:
        role: Author role (``MessageRole`` or raw string such as
            ``"user"``).  Strings that match a role name are coerced to
            the corresponding enum member.
        content: Message payload (usually text).
        metadata: Arbitrary key/value metadata attached to the message.
        type: Payload type.  Defaults to ``MessageType.TEXT``.
    """

    role: Union[MessageRole, str] = MessageRole.USER
    content: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    type: Optional[MessageType] = MessageType.TEXT

    def __post_init__(self) -> None:
        # Coerce plain-string roles to MessageRole when possible so that
        # ``Message(role="user")`` and ``Message(role=MessageRole.USER)``
        # behave identically.
        if isinstance(self.role, str) and not isinstance(self.role, MessageRole):
            try:
                self.role = MessageRole(self.role)
            except ValueError:
                # Unknown role strings are preserved as-is.
                pass
        if self.type is None:
            self.type = MessageType.TEXT
        if self.metadata is None:
            self.metadata = {}

    @property
    def is_error(self) -> bool:
        """Whether this message carries an error payload."""
        return self.type == MessageType.ERROR

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the message to a plain dictionary."""
        return {
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "metadata": dict(self.metadata),
            "type": self.type.value if isinstance(self.type, MessageType) else self.type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Build a message from a dictionary produced by ``to_dict``."""
        role = data.get("role", MessageRole.USER)
        if isinstance(role, str):
            role = MessageRole(role)
        msg_type = data.get("type", MessageType.TEXT)
        if isinstance(msg_type, str):
            msg_type = MessageType(msg_type)
        return cls(
            role=role,
            content=data.get("content"),
            metadata=dict(data.get("metadata") or {}),
            type=msg_type,
        )

    def __len__(self) -> int:
        """Length of the textual content."""
        return len(self.content) if isinstance(self.content, str) else 0


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, Any]]:
    """Convert a list of messages to a list of plain dictionaries."""
    return [m.to_dict() if isinstance(m, Message) else dict(m) for m in messages]


__all__ = [
    "Message",
    "MessageRole",
    "MessageType",
    "ContentBlock",
    "ContentWrapper",
    "MessageValidationError",
    "system",
    "user",
    "assistant",
    "messages_to_dicts",
]
