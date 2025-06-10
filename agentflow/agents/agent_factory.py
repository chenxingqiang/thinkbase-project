"""Agent factory module for creating instruction-based agents."""

from typing import Dict, Any, Optional, List, Type, Set
from pydantic import BaseModel
import uuid
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
    
    def create_agent(self, config: Dict[str, Any]) -> Agent:
        """Create an agent with specified instructions.
        
        Args:
            config: Agent configuration
            
        Returns:
            Agent: Created agent instance
            
        Raises:
            ValueError: If agent type is not supported
        """
        # Convert dict to AgentConfig
        agent_config = AgentConfig(**config)
        agent_type = agent_config.type
        
        # Create base agent
        agent = Agent(
            id=str(uuid.uuid4()),
            name=agent_config.name,
            type=agent_type,
            config=agent_config
        )
        
        # Add instructions based on agent type
        instruction_set = self._instruction_sets.get(agent_type)
        if instruction_set:
            if instruction_set.pattern_learning:
                agent.add_instruction(instruction_set.pattern_learning)
            if instruction_set.agent_learning:
                agent.add_instruction(instruction_set.agent_learning)
            if instruction_set.objective_handling:
                agent.add_instruction(instruction_set.objective_handling)
            for instruction in instruction_set.custom_instructions:
                agent.add_instruction(instruction)
        
        # Store agent
        self._agents[agent.id] = agent
        
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
