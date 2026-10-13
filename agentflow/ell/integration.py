"""ELL integration layer (vendored, dependency-free).

Provides the ``ELLIntegration`` singleton used across AgentFlow to
route messages and workflows through the ELL abstraction.
"""

import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional

from .types.message import Message, MessageRole, MessageType

__all__ = ["ELLIntegration", "ell_integration"]


class ELLIntegration:
    """Singleton class for ELL integration."""

    _instance: Optional["ELLIntegration"] = None
    _initialized: bool = False
    _mode_configs: Dict[str, Dict[str, Any]] = {
        "simple": {"model": "test-model"},
        "complex": {
            "model": "test-model-complex",
            "track_performance": True,
            "track_memory": True,
        },
    }

    def __new__(cls) -> "ELLIntegration":
        """Create or return the singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def __init__(self) -> None:
        """Initialize instance attributes (state lives on the singleton)."""

    def _initialize(self) -> None:
        """Initialize ELL integration state."""
        if not self._initialized:
            self._messages: List[Message] = []
            self._workflows: Dict[str, Any] = {}
            self._metrics: Dict[str, Dict[str, Any]] = {}
            self.config: Dict[str, Any] = {
                "default_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "simple": {
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 30.0,
                },
                "complex": {
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 60.0,
                    "stream": True,
                    "track_performance": True,
                    "track_memory": True,
                },
            }
            ELLIntegration._initialized = True

    def cleanup(self) -> None:
        """Reset ELL integration state."""
        self._messages = []
        self._workflows = {}
        self._metrics = {}
        ELLIntegration._initialized = False

    def configure(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Configure ELL integration.

        Args:
            config: Mapping of mode name (e.g. ``"simple"``) to mode
                configuration, merged over the current mode configs.
        """
        if config:
            self._mode_configs.update(config)

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get configuration for a processing mode."""
        return self._mode_configs.get(mode, {})

    async def process_message(self, message: Message) -> Message:
        """Process a message through ELL.

        Records the message in the shared context and echoes it back as
        an assistant result message.
        """
        self._messages.append(message)
        return Message(
            role=MessageRole.ASSISTANT,
            content=message.content,
            type=MessageType.RESULT,
            metadata=dict(message.metadata or {}),
        )

    def get_messages(self) -> List[Message]:
        """Get all messages in the context."""
        return list(self._messages)

    def clear_messages(self) -> None:
        """Clear all messages from the context."""
        self._messages = []

    def register_workflow(self, workflow_id: str, workflow: Any) -> None:
        """Register a workflow.

        Args:
            workflow_id: Workflow ID.
            workflow: Workflow instance.
        """
        self._workflows[workflow_id] = workflow

    def unregister_workflow(self, workflow_id: str) -> None:
        """Unregister a workflow."""
        self._workflows.pop(workflow_id, None)

    def list_workflows(self) -> List[str]:
        """List registered workflow IDs."""
        return list(self._workflows.keys())

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        return {k: dict(v) for k, v in self._metrics.items()}

    def _ensure_metrics(self, func_name: str) -> Dict[str, Any]:
        """Create the metrics record for a function if missing."""
        if func_name not in self._metrics:
            self._metrics[func_name] = {
                "calls": 0,
                "total_time": 0.0,
                "errors": 0,
            }
        return self._metrics[func_name]

    @staticmethod
    def with_ell(mode: str = "simple") -> Callable:
        """Decorator to enable ELL processing for an async function.

        Args:
            mode: Processing mode (``"simple"`` or ``"complex"``).

        Returns:
            Decorated function.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell = ELLIntegration()
                config = ell.get_mode_config(mode)
                func_name = func.__name__
                metrics = ell._ensure_metrics(func_name)

                try:
                    result = await func(*args, **kwargs)
                    metrics["calls"] += 1

                    message = Message(
                        role=MessageRole.USER,
                        content=str(result),
                        metadata={
                            "mode": mode,
                            "function": func_name,
                            "status": "success",
                            **{k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))},
                        },
                    )
                    await ell.process_message(message)
                    return result
                except Exception as e:
                    metrics["errors"] += 1

                    message = Message(
                        role=MessageRole.USER,
                        content=str(e),
                        metadata={
                            "mode": mode,
                            "function": func_name,
                            "status": "error",
                            "error": str(e),
                        },
                    )
                    await ell.process_message(message)
                    raise

            return wrapper

        return decorator

    @staticmethod
    def track_function() -> Callable:
        """Decorator to track execution metrics of an async function.

        Returns:
            Decorated function.
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell = ELLIntegration()
                func_name = func.__name__
                metrics = ell._ensure_metrics(func_name)

                try:
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    metrics["calls"] += 1
                    metrics["total_time"] += time.time() - start_time
                    return result
                except Exception:
                    metrics["errors"] += 1
                    raise

            return wrapper

        return decorator


# Global singleton instance.
ell_integration = ELLIntegration()
