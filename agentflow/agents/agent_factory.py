"""Agent factory module for creating instruction-based agents."""

from typing import Dict, Any, Optional, List, Type, Set
from pydantic import BaseModel
import uuid
from enum import Enum
import logging

from .config import AgentType, AgentConfig
from .agent import Agent
from ..core.instructions.base import BaseInstruction as Instruction

logger = logging.getLogger(__name__)

class InstructionSet(BaseModel):
    """Represents a set of instructions for an agent."""
    pattern_learning: Optional[Instruction] = None
    agent_learning: Optional[Instruction] = None
    objective_handling: Optional[Instruction] = None
    custom_instructions: Set[Instruction] = set()

    class Config:
        arbitrary_types_allowed = True

class AgentFactory:
    """Factory class for creating instruction-based agents."""
    
    _instance = None
    _agents: Dict[str, Agent] = {}
    _instruction_sets: Dict[str, InstructionSet] = {}
    _agent_classes: Dict[str, type] = {}
    
    def __init__(self):
        """Initialize agent factory."""
        self.id = str(uuid.uuid4())
        self.name = "AgentFactory"
        self.description = "Factory for creating instruction-based agents"
        self.metadata: Dict[str, Any] = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentFactory, cls).__new__(cls)
        return cls._instance
    
    def register_instruction_set(self, agent_type: str, instruction_set: InstructionSet) -> None:
        """Register an instruction set for an agent type."""
        self._instruction_sets[agent_type] = instruction_set

    @classmethod
    def register_agent(cls, agent_type: str, agent_class) -> None:
        """Register a concrete Agent class for an agent type string.

        Once registered, :meth:`create_agent` instantiates this class
        instead of the base :class:`Agent` for the given type.
        """
        cls._agent_classes[agent_type] = agent_class
    
    @classmethod
    def create_agent(cls, config: Union[Dict[str, Any], Any]) -> Agent:
        """Create an agent for the configured type.

        Callable directly on the class (``AgentFactory.create_agent(config)``)
        as well as on an instance.  The agent type is resolved from either
        a sectioned dict (``config["AGENT"]["type"]``), a flat dict
        (``config["type"]``) or an attribute on a configuration object;
        enum values are normalized to their string form.  When a class was
        registered for the type via :meth:`register_agent`, that class is
        instantiated (it must be an ``Agent`` subclass); otherwise the base
        ``Agent`` is created.

        Args:
            config: Agent configuration (dict or configuration object)

        Returns:
            Agent: Created agent instance
        """
        # Resolve the agent type string from either a sectioned or flat dict
        if isinstance(config, dict):
            agent_type = (
                (config.get("AGENT") or {}).get("type")
                or config.get("type")
                or "generic"
            )
        else:
            agent_type = getattr(config, "type", "generic") or "generic"
        if isinstance(agent_type, Enum):
            agent_type = agent_type.value

        # Instantiate the registered class for this type if present
        agent_cls = cls._agent_classes.get(agent_type)
        if agent_cls is not None and agent_cls is not Agent:
            agent = agent_cls(config)
            if not isinstance(agent, Agent):
                raise ValueError(
                    f"Registered class for {agent_type!r} is not an Agent"
                )
            return agent

        # Default path: base Agent over the configuration object
        agent = Agent(config=config)
        cls._agents[agent.id] = agent
        logger.info(f"Created agent {agent.id} of type {agent_type}")
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return list(self._agents.keys())
    
    def remove_agent(self, agent_id: str) -> None:
        """Remove an agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert factory to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "agents": {k: v.to_dict() for k, v in self._agents.items()},
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentFactory':
        """Create factory from dictionary."""
        factory = cls()
        factory.id = data.get("id")
        factory.name = data.get("name")
        factory.description = data.get("description")
        factory.metadata = data.get("metadata", {})
        
        for agent_id, agent_data in data.get("agents", {}).items():
            agent = Agent.from_dict(agent_data)
            factory._agents[agent_id] = agent
        
        return factory
