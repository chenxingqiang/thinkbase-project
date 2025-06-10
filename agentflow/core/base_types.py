from enum import Enum

class AgentType(Enum):
    DEFAULT = 0

class AgentMode(Enum):
    OFFLINE = 0
    ONLINE = 1

class AgentStatus(Enum):
    INITIALIZING = 0
    RUNNING = 1
    STOPPED = 2

DictKeyType = str

class MessageType(Enum):
    TEXT = 0
    COMMAND = 1

class MessageRole(Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
