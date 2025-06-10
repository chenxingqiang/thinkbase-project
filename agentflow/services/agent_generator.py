"""Agent configuration generator service."""
import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import asyncio
from ..agents.agent import Agent
from ..core.config import AgentConfig

class AgentGenerator:
    """Service for generating agent configurations through conversation."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize agent generator.
        
        Args:
            config: Service configuration containing:
                - base_path: Base path for storing agent configurations
                - llm_config: Configuration for the LLM used in generation
        """
        self.config = config
        self.base_path = Path(config.get('base_path', './agents'))
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize LLM for generation
        self.llm_config = config.get('llm_config', {})
        
    async def generate_config(self, conversation: list) -> Dict[str, Any]:
        """Generate agent configuration from conversation.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Generated agent configuration
        """
        # Extract requirements from conversation
        requirements = self._extract_requirements(conversation)
        
        # Generate configuration template
        config_template = {
            "agent": {
                "name": requirements.get("name", "generated_agent"),
                "version": "1.0.0",
                "type": requirements.get("type", "general"),
                "description": requirements.get("description", "")
            },
            "input_specification": {
                "modes": requirements.get("input_modes", ["DIRECT"]),
                "validation": requirements.get("validation", {})
            },
            "output_specification": {
                "modes": requirements.get("output_modes", ["RETURN"]),
                "strategies": requirements.get("strategies", {})
            },
            "llm_configuration": {
                "model": requirements.get("model", "gpt-4"),
                "temperature": requirements.get("temperature", 0.7),
                "max_tokens": requirements.get("max_tokens", 1000),
                "system_prompt": requirements.get("system_prompt", "")
            },
            "tools": requirements.get("tools", []),
            "metadata": {
                "created_from": "conversation",
                "conversation_id": requirements.get("conversation_id")
            }
        }
        
        return config_template
        
    def _extract_requirements(self, conversation: list) -> Dict[str, Any]:
        """Extract agent requirements from conversation.
        
        Args:
            conversation: List of conversation messages
            
        Returns:
            Dictionary of requirements
        """
        # TODO: Use LLM to extract requirements
        # For now, return basic requirements
        return {
            "name": "agent_" + str(hash(str(conversation)))[:8],
            "type": "general",
            "description": "Generated from conversation"
        }
        
    async def save_config(self, config: Dict[str, Any], name: Optional[str] = None) -> str:
        """Save agent configuration to file.
        
        Args:
            config: Agent configuration
            name: Optional name for the configuration file
            
        Returns:
            Path to saved configuration file
        """
        if name is None:
            name = config["agent"]["name"]
            
        file_path = self.base_path / f"{name}.json"
        
        with open(file_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        return str(file_path)
        
    async def load_config(self, name: str) -> Dict[str, Any]:
        """Load agent configuration from file.
        
        Args:
            name: Name of the configuration file
            
        Returns:
            Loaded agent configuration
        """
        file_path = self.base_path / f"{name}.json"
        
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file {name}.json not found")
            
        with open(file_path) as f:
            return json.load(f)
            
    async def create_agent(self, config: Dict[str, Any]) -> Agent:
        """Create agent from configuration.
        
        Args:
            config: Agent configuration
            
        Returns:
            Created agent instance
        """
        agent_config = AgentConfig(**config)
        return Agent(agent_config)
        
    async def run_agent(self, name: str, input_data: Dict[str, Any]) -> Any:
        """Run agent with input data.
        
        Args:
            name: Name of the agent configuration
            input_data: Input data for the agent
            
        Returns:
            Agent output
        """
        # Load configuration
        config = await self.load_config(name)
        
        # Create agent
        agent = await self.create_agent(config)
        
        # Run agent
        return await agent.execute(input_data)
        
    def list_agents(self) -> list:
        """List available agent configurations.
        
        Returns:
            List of agent names
        """
        return [f.stem for f in self.base_path.glob("*.json")]
