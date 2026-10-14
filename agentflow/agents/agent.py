"""Agent module.

Provides the runtime classes used across AgentFlow:

* :class:`Agent` -- the workflow agent with message processing, ELL
  integration, ISA/instruction-selection support and lifecycle
  management.
* :class:`AgentState` -- pydantic model tracking status, timing and the
  message/error history of an agent.
* :class:`RemoteAgent` -- Ray actor wrapper around a minimal remote
  agent (with a no-Ray fallback).
* :class:`TransformationPipeline` -- re-exported from
  :mod:`agentflow.transformations.pipeline` for backward compatibility.

The configuration accepted by :class:`Agent` is deliberately permissive:
``None``, a plain dictionary (either flat, e.g. ``{"name": ...,
"type": ...}``, or sectioned, e.g. ``{"AGENT": {...}, "MODEL": {...},
"WORKFLOW": {...}, "DOMAIN_CONFIG": {...}}``) or any configuration
object exposing the usual attributes (``name``, ``type``, ``mode``,
``model``, ``workflow``, ``domain_config``, ...).
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

# Ray is imported lazily (see __getattr__ for RemoteAgent):
# eagerly importing it costs ~2s for every process that loads Agent.
from .state import AgentStatus
from .config import AgentConfig
from ..ell.integration import ELLIntegration
from ..ell.types.message import Message, MessageRole, MessageType
# TransformationPipeline / AdvancedTransformationStrategy are re-exported
# lazily (see module __getattr__ below): importing them eagerly drags in
# pandas+sklearn (~3s) for every process that merely imports Agent.

# ``agentflow.core`` is undergoing an active refactor (the flat modules are
# being turned into shims over ``agentflow.core.workflow``). The symbols
# below are optional for this module, so import them defensively instead of
# hard-coupling to the in-flight chain; ``None`` disables the corresponding
# optional behaviour at runtime.
try:
    from ..core.exceptions import WorkflowExecutionError, ConfigurationError
except Exception:  # pragma: no cover - fallback when core.exceptions is broken
    class WorkflowExecutionError(Exception):  # type: ignore[no-redef]
        """Local fallback for the core workflow execution error."""

    class ConfigurationError(Exception):  # type: ignore[no-redef]
        """Local fallback for the core configuration error."""

try:
    from ..core.model_config import ModelConfig
except Exception:  # pragma: no cover
    ModelConfig = None  # type: ignore[assignment]

try:
    from ..core.workflow_types import WorkflowConfig
except Exception:  # pragma: no cover - flat shim may chain to in-flight modules
    WorkflowConfig = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

__all__ = [
    "Agent",
    "AgentState",
    "AgentStatus",
    "AgentConfig",
    "RemoteAgent",
    "TransformationPipeline",
    "AdvancedTransformationStrategy",
    "WorkflowExecutionError",
    "ConfigurationError",
]

# Config sections used by the sectioned dictionary format
# ({"AGENT": {...}, "MODEL": {...}, "WORKFLOW": {...}, ...}).
_SECTION_KEYS = ("AGENT", "MODEL", "WORKFLOW", "DOMAIN_CONFIG", "TRANSFORMATIONS")

# Attributes that map directly onto a whole section rather than a key
# inside one (e.g. ``config.model`` is the ``MODEL`` section itself).
_SECTION_ATTRIBUTE_MAP = {
    "model": "MODEL",
    "workflow": "WORKFLOW",
    "domain_config": "DOMAIN_CONFIG",
    "transformations": "TRANSFORMATIONS",
}


def _cfg(config: Any, key: str, default: Any = None) -> Any:
    """Read *key* from a config dict, sectioned dict or object.

    Flat dictionaries are checked first, then the classic sections
    (``AGENT``, ``MODEL``, ...), then plain attribute access. Keys that
    name a whole section (``model`` -> ``MODEL``, ...) resolve to that
    section.
    """
    if config is None:
        return default
    if isinstance(config, dict):
        if key in config:
            return config[key]
        section_key = _SECTION_ATTRIBUTE_MAP.get(key)
        if section_key is not None and section_key in config:
            return config[section_key]
        for section_key in _SECTION_KEYS:
            section = config.get(section_key)
            if isinstance(section, dict) and key in section:
                return section[key]
        return default
    return getattr(config, key, default)


class _DictConfigView(dict):
    """Attribute view over a configuration dictionary.

    Subclasses :class:`dict` so that ``isinstance`` checks, ``[]`` access
    and ``.get()`` keep working, while additionally exposing keys as
    attributes (including the ``AGENT``/``MODEL``/``WORKFLOW``/
    ``DOMAIN_CONFIG`` section fallbacks). Nested dictionaries returned
    through attribute access are wrapped so that chains such as
    ``config.model.provider`` work.
    """

    _MISSING = object()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        value = _cfg(self, name, _DictConfigView._MISSING)
        if value is _DictConfigView._MISSING:
            raise AttributeError(name)
        if isinstance(value, dict) and not isinstance(value, _DictConfigView):
            value = _DictConfigView(value)
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class AgentState(BaseModel):
    """Agent state class.

    Tracks execution status, iteration count, processed messages,
    metrics and errors, plus the timing information recorded while a
    message is being processed.
    """

    status: AgentStatus = Field(default=AgentStatus.INITIALIZED)
    iteration: int = Field(default=0)
    last_error: Optional[str] = None
    messages_processed: int = Field(default=0)
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    errors: List[Dict[str, Any]] = Field(default_factory=list)
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def __str__(self) -> str:
        """String representation."""
        return f"AgentState(status={self.status}, iteration={self.iteration})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"AgentState(status={self.status}, iteration={self.iteration}, "
            f"errors={len(self.errors)})"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        status = self.status
        if hasattr(status, "value"):
            status = status.value
        return {
            "status": status,
            "iteration": self.iteration,
            "last_error": self.last_error,
            "messages_processed": self.messages_processed,
            "metrics": self.metrics,
            "errors": self.errors,
            "result": self.result,
            "error": self.error,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }

    @classmethod
    def from_dict(cls, state_dict: Dict[str, Any]) -> "AgentState":
        """Create an AgentState from a dictionary."""
        return cls(**(state_dict or {}))

    async def process_message(self, message: Union[str, Dict[str, Any], Message]) -> str:
        """Process a message and update agent state.

        Records ``start_time``/``end_time`` around the processing step so
        that timing information is available even when processing
        raises.

        Args:
            message: Message to process

        Returns:
            str: Processed message content

        Raises:
            ValueError: If message processing fails or string conversion
                of the message fails.
        """
        # Record start time
        self.start_time = time.time()
        self.status = AgentStatus.PROCESSING

        try:
            # Convert message to string representation (this may raise,
            # e.g. for messages whose __str__ is broken).
            message_str = str(message)

            if isinstance(message, str):
                processed_message = {"content": message, "role": "user"}
            elif isinstance(message, dict):
                processed_message = message
            elif isinstance(message, Message):
                if hasattr(message, "to_dict"):
                    processed_message = message.to_dict()
                else:
                    processed_message = {
                        "content": getattr(message, "content", message_str),
                        "role": str(getattr(message, "role", "user")),
                    }
            else:
                raise ValueError(f"Unsupported message type: {type(message)}")

            # Update messages processed
            self.messages_processed += 1
            self.messages.append(processed_message)

            # Update status
            self.status = AgentStatus.SUCCESS

            # Return the string representation
            return message_str

        except ValueError as error:
            self.status = AgentStatus.FAILED
            self.last_error = str(error)
            self.errors.append({"error": str(error), "timestamp": str(datetime.now())})
            raise

        except Exception as error:  # noqa: BLE001 - record, then re-raise
            self.status = AgentStatus.FAILED
            self.last_error = str(error)
            self.errors.append({"error": str(error), "timestamp": str(datetime.now())})
            raise

        finally:
            # Always record end time, even when processing raised.
            self.end_time = time.time()


class Agent:
    """Base agent class.

    A plain (non-pydantic) class holding configuration, state, history
    and the ELL/ISA integration objects, with an async lifecycle
    (``initialize`` / ``process_message`` / ``execute`` / ``cleanup``).
    """

    def __init__(self, config: Any = None, name: Optional[str] = None, **kwargs):
        """Initialize agent.

        Args:
            config: Agent configuration. May be ``None``, a dictionary
                (flat or sectioned, see module docstring) or any
                configuration object.
            name: Agent name. Falls back to the name stored in the
                configuration, then to a generated UUID.
            **kwargs: Additional overrides, e.g. ``id``, ``type``,
                ``mode``, ``description``, ``workflow``, ``model``,
                ``ell``, ``isa_manager``, ``instruction_selector``.

        Raises:
            ValueError: If the configuration is invalid.
        """
        if config is None:
            config = _DictConfigView(
                {
                    "name": name or str(uuid.uuid4()),
                    "type": kwargs.get("type", "generic"),
                    "mode": kwargs.get("mode", "sequential"),
                }
            )
        elif isinstance(config, dict):
            if not config:
                raise ValueError("Agent must have a configuration")
            config = _DictConfigView(config)
        # else: use the configuration object as-is.

        self.id = kwargs.get("id", str(uuid.uuid4()))
        self.name = name or _cfg(config, "name") or str(uuid.uuid4())
        self.type = kwargs.get("type", _cfg(config, "type", "generic") or "generic")
        self.mode = kwargs.get("mode", _cfg(config, "mode", "sequential") or "sequential")
        self.description = kwargs.get("description", _cfg(config, "description"))
        self.version = _cfg(config, "version", "1.0.0") or "1.0.0"
        self.system_prompt = _cfg(config, "system_prompt")
        self.config = config
        self.domain_config = _cfg(config, "domain_config", {}) or {}
        self.metadata: Dict[str, Any] = dict(_cfg(config, "metadata", {}) or {})
        self.is_distributed = bool(_cfg(config, "is_distributed", False))

        self._initialized = False
        self._status: AgentStatus = AgentStatus.INITIALIZED

        # Ensure max_errors is always an int.
        config_max_errors = _cfg(config, "max_errors", None)
        try:
            self.max_errors: int = int(config_max_errors) if config_max_errors is not None else 10
        except (TypeError, ValueError):
            self.max_errors = 10

        self.errors: List[Dict[str, str]] = []
        self.history: List[Dict[str, Any]] = []
        self.instructions: List[Any] = []

        # Initialize state
        self.state = AgentState(status=AgentStatus.INITIALIZED)

        # Initialize components
        self._ell = kwargs.get("ell", kwargs.get("ell2a", None))
        self._isa_manager = kwargs.get("isa_manager", None)
        self._instruction_selector = kwargs.get("instruction_selector", None)

        # If workflow is provided in kwargs, merge it into the config.
        if "workflow" in kwargs:
            self._merge_config("workflow", self._normalize_workflow(kwargs["workflow"]))

        # If model is provided in kwargs, merge it into the config.
        if "model" in kwargs:
            self._merge_config("model", self._normalize_model(kwargs["model"]))

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _merge_config(self, key: str, value: Any) -> None:
        """Return a new dict-backed config with *key* set to *value*.

        Object configurations that already carry the key are left
        untouched.
        """
        if isinstance(self.config, dict):
            merged = dict(self.config)
            merged[key] = value
            self.config = _DictConfigView(merged)
        elif self.config is None:
            self.config = _DictConfigView({key: value})

    def _normalize_workflow(self, workflow_data: Any) -> Any:
        """Coerce *workflow_data* to a WorkflowConfig when possible.

        Falls back to a dict-backed config view when the core
        WorkflowConfig class is unavailable (defensive import) or the
        payload does not fit its constructor, so that attributes such as
        ``workflow.max_iterations`` remain readable either way.
        """
        if WorkflowConfig is not None and isinstance(workflow_data, WorkflowConfig):
            return workflow_data
        if isinstance(workflow_data, dict):
            if WorkflowConfig is not None:
                try:
                    return WorkflowConfig(**workflow_data)
                except Exception:
                    pass
                try:
                    # WorkflowConfig may accept only a subset of keys;
                    # keep the rest as plain attributes.
                    workflow = WorkflowConfig(steps=workflow_data.get("steps"))
                    for key, value in workflow_data.items():
                        if key != "steps":
                            setattr(workflow, key, value)
                    return workflow
                except Exception:
                    pass
            return _DictConfigView(workflow_data)
        if workflow_data is None:
            return _DictConfigView({"steps": []})
        raise ValueError("workflow must be a dictionary or WorkflowConfig instance")

    def _normalize_model(self, model_data: Any) -> Any:
        """Coerce *model_data* to a ModelConfig when possible."""
        if ModelConfig is not None and isinstance(model_data, ModelConfig):
            return model_data
        if isinstance(model_data, dict):
            if ModelConfig is not None:
                try:
                    return ModelConfig(**model_data)
                except Exception:
                    pass
            return _DictConfigView(model_data)
        if model_data is None:
            return None
        if hasattr(model_data, "provider") or hasattr(model_data, "name"):
            return model_data
        raise ValueError("model must be a dictionary or ModelConfig instance")

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def isa_manager(self):
        """Get ISA manager."""
        return self._isa_manager

    @property
    def instruction_selector(self):
        """Get instruction selector."""
        return self._instruction_selector

    @property
    def status(self) -> AgentStatus:
        """Get agent status."""
        return self._status

    @status.setter
    def status(self, value: AgentStatus) -> None:
        """Set agent status."""
        self._status = value

    # ------------------------------------------------------------------
    # Instructions (compatibility with the instruction-based agents)
    # ------------------------------------------------------------------

    def add_instruction(self, instruction: Any) -> None:
        """Add an instruction to the agent's instruction set."""
        if instruction not in self.instructions:
            self.instructions.append(instruction)

    def remove_instruction(self, instruction: Any) -> None:
        """Remove an instruction from the agent's instruction set."""
        if instruction in self.instructions:
            self.instructions.remove(instruction)

    def get_instructions(self) -> List[Any]:
        """Get all instructions in the agent's instruction set."""
        return list(self.instructions)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        """Initialize agent resources.

        Sets up the ELL integration, the ISA manager and the
        instruction selector. ELL failures mark the agent as failed and
        re-raise; ISA/selector failures are logged and tolerated so the
        agent remains usable without those optional components.
        """
        if self._initialized:
            return

        try:
            # ELL integration (required component).
            if not self._ell:
                self._ell = ELLIntegration()
                ell_config = {
                    "model": _cfg(self.config, "model"),
                    "domain": _cfg(self.config, "domain_config", {}) or {},
                    "enabled": True,
                    "tracking_enabled": True,
                }
                try:
                    self._ell.configure(ell_config)
                except Exception as error:  # noqa: BLE001
                    logger.debug("ELL configuration skipped: %s", error)

            # ISA manager (optional component).  Imported lazily here:
            # isa_manager imports Ray at module level, which costs ~2s
            # for every process that merely constructs an Agent.
            try:
                from ..core.isa.isa_manager import ISAManager
            except Exception:  # pragma: no cover
                ISAManager = None  # type: ignore[assignment]
            try:
                if not self._isa_manager and ISAManager is not None:
                    config_path = None
                    if self.config is not None:
                        config_path = (_cfg(self.config, "domain_config", {}) or {}).get(
                            "isa_config_path"
                        )
                    self._isa_manager = ISAManager(isa_config_path=config_path)
                    await self._isa_manager.initialize()
            except Exception as error:  # noqa: BLE001
                logger.warning("ISA manager initialization failed: %s", error)
                self._isa_manager = None

            # Instruction selector (optional component).
            try:
                from ..core.isa.selector import InstructionSelector
            except Exception:  # pragma: no cover
                InstructionSelector = None  # type: ignore[assignment]
            try:
                if not self._instruction_selector and InstructionSelector is not None:
                    self._instruction_selector = InstructionSelector(self.config)
                    await self._instruction_selector.initialize()
            except Exception as error:  # noqa: BLE001
                logger.warning("Instruction selector initialization failed: %s", error)
                self._instruction_selector = None

            self._initialized = True
            self._status = AgentStatus.INITIALIZED
        except Exception as error:
            self._status = AgentStatus.FAILED
            raise

    def add_error(self, error: str) -> None:
        """Add error to the (bounded) error list."""
        if len(self.errors) >= self.max_errors:
            self.errors.pop(0)
        self.errors.append({"error": error, "timestamp": str(datetime.now())})

    # ------------------------------------------------------------------
    # Message processing
    # ------------------------------------------------------------------

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """Extract readable text from a message content payload."""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if hasattr(block, "text"):
                    parts.append(block.text or "")
                else:
                    parts.append(str(block))
            return " ".join(part for part in parts if part)
        if hasattr(content, "text"):
            return content.text or ""
        return str(content)

    async def process_message(self, message: Union[str, Dict[str, Any], Message]) -> str:
        """Process a message through the ELL integration.

        Args:
            message: Message to process (string, dict or
                :class:`~agentflow.ell.types.message.Message`)

        Returns:
            str: Response content

        Raises:
            Exception: Whatever the ELL integration raises; the agent
                state is updated to ``FAILED`` first.
        """
        if isinstance(message, str):
            message = Message(content=message, role=MessageRole.USER)
        elif isinstance(message, dict):
            message = Message(**message)

        # Optionally add message to history based on a configuration flag.
        disable_history = getattr(self, "_disable_history", False)
        if not disable_history:
            self.history.append(
                {
                    "role": getattr(message, "role", MessageRole.USER),
                    "content": str(getattr(message, "content", message)),
                    "timestamp": str(datetime.now()),
                }
            )

        # Update state
        self.state.messages_processed += 1
        self.state.status = AgentStatus.PROCESSING

        try:
            # Process message using the ELL integration.
            result = await self._ell.process_message(message)
            self.state.status = AgentStatus.SUCCESS

            response_content = result.content if isinstance(result, Message) else result
            response_text = self._content_to_text(response_content)

            # Optionally add response to history.
            if not disable_history:
                self.history.append(
                    {
                        "role": MessageRole.ASSISTANT,
                        "content": response_text,
                        "timestamp": str(datetime.now()),
                    }
                )

            return response_text

        except Exception as error:
            self.state.status = AgentStatus.FAILED
            self.state.last_error = str(error)
            self.add_error(str(error))
            raise

    async def execute(self, input_data: Dict[str, Any]) -> str:
        """Execute agent with input data.

        Args:
            input_data: Input data for execution

        Returns:
            str: Execution result

        Raises:
            WorkflowExecutionError: If execution fails.
        """
        if not isinstance(input_data, dict):
            input_data = {"data": input_data}

        # In test mode, check for should_fail flag.
        if input_data.get("test_mode") and input_data.get("should_fail"):
            self.state.status = AgentStatus.FAILED
            error_msg = "Test error during execution"
            self.state.last_error = error_msg
            self.add_error(error_msg)
            raise WorkflowExecutionError(error_msg)

        # Create message from input data.
        data = input_data.get("data", "")
        if isinstance(data, np.ndarray):
            # Structured payloads keep the raw array so downstream
            # consumers (and analysis pipelines) receive the actual
            # matrix rather than a lossy stringification.
            message_content: Any = {"data": data}
        else:
            message_content = str(data)

        message = Message(
            content=message_content,
            role=MessageRole.USER,
            type=MessageType.RESULT,
            metadata=input_data.get("metadata", {}),
        )

        return await self.process_message(message)

    async def cleanup(self) -> None:
        """Clean up agent resources."""
        try:
            for component in (self._ell, self._isa_manager, self._instruction_selector):
                if component is None:
                    continue
                cleanup_method = getattr(component, "cleanup", None)
                if cleanup_method is None:
                    continue
                if asyncio.iscoroutinefunction(cleanup_method):
                    await cleanup_method()
                else:
                    cleanup_method()

            self._initialized = False
            self._status = AgentStatus.STOPPED
        except Exception as error:
            logger.error("Error during cleanup: %s", error)
            raise

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert agent to dictionary representation."""
        config = self.config
        if hasattr(config, "to_dict"):
            config_data = config.to_dict()
        elif hasattr(config, "model_dump"):
            config_data = config.model_dump()
        elif isinstance(config, dict):
            config_data = dict(config)
        elif config is None:
            config_data = None
        else:
            config_data = str(config)
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "mode": self.mode,
            "description": self.description,
            "state": self.state.to_dict(),
            "metadata": self.metadata,
            "config": config_data,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Agent":
        """Create an Agent instance from a dictionary."""
        config_data = data.get("config")
        if isinstance(config_data, dict):
            config = _DictConfigView(config_data)
        else:
            config = config_data

        state_data = data.get("state")
        state = AgentState(**state_data) if isinstance(state_data, dict) else AgentState()

        agent = cls(
            config=config,
            name=data.get("name"),
            id=data.get("id"),
            description=data.get("description"),
        )
        agent.state = state
        agent.metadata = data.get("metadata", {}) or {}
        return agent


class _RemoteAgentBase:
    """Implementation backing the Ray actor :class:`RemoteAgent`.

    Exposes the minimal remote-agent surface used by the tests:
    ``get_status_remote``, ``initialize`` and ``cleanup``.
    """

    def __init__(self, config: Any = None):
        """Initialize remote agent."""
        self.config = config
        self.id = str(uuid.uuid4())
        self.name = "remote_agent"
        self.type = "generic"
        self.version = "1.0.0"
        self._status = AgentStatus.INITIALIZED
        self._initialized = False

    def get_status_remote(self) -> str:
        """Remote method to get agent status."""
        status = self._status
        if hasattr(status, "value"):
            return str(status.value)
        return str(status)

    def initialize(self) -> bool:
        """Initialize remote agent."""
        self._initialized = True
        return True

    def cleanup(self) -> bool:
        """Clean up remote agent resources."""
        self._status = AgentStatus.STOPPED
        return True

    def set_status(self, value: AgentStatus) -> None:
        """Set agent status."""
        self._status = value


def __getattr__(name):
    """Lazily re-export heavy module members (deps deferred to first use)."""
    if name == "RemoteAgent":
        # Ray is imported only when the remote-agent API is actually used.
        try:
            import ray
        except ImportError:  # pragma: no cover - Ray is installed here
            class _LocalRemoteAgent(_RemoteAgentBase):
                """Local stand-in providing the Ray-style ``.remote()`` API."""

                @classmethod
                def remote(cls, *args: Any, **kwargs: Any) -> "_LocalRemoteAgent":
                    """Return a new instance, mimicking the Ray factory."""
                    return cls(*args, **kwargs)

            RemoteAgent = _LocalRemoteAgent
        else:
            RemoteAgent = ray.remote(_RemoteAgentBase)
        globals()["RemoteAgent"] = RemoteAgent
        return RemoteAgent
    if name in ("TransformationPipeline", "AdvancedTransformationStrategy"):
        if name == "TransformationPipeline":
            from ..transformations.pipeline import TransformationPipeline
            return TransformationPipeline
        from ..transformations.advanced_strategies import (
            AdvancedTransformationStrategy,
        )
        return AdvancedTransformationStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
