"""Agent configuration module."""
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    provider: str
    model_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    version: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model_type": self.model_type,
            "parameters": self.parameters,
            "enabled": self.enabled,
            "version": self.version
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create a ModelConfig from a dictionary."""
        return cls(
            name=config_dict.get('name', ''),
            provider=config_dict.get('provider', ''),
            model_type=config_dict.get('model_type', ''),
            parameters=config_dict.get('parameters', {}),
            enabled=config_dict.get('enabled', True),
            version=config_dict.get('version')
        )

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    name: str
    description: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    version: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    models: List[ModelConfig] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "enabled": self.enabled,
            "version": self.version,
            "dependencies": self.dependencies,
            "models": [model.to_dict() for model in self.models]
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'AgentConfig':
        """Create an AgentConfig from a dictionary."""
        return cls(
            name=config_dict.get('name', ''),
            description=config_dict.get('description'),
            metadata=config_dict.get('metadata', {}),
            enabled=config_dict.get('enabled', True),
            version=config_dict.get('version'),
            dependencies=config_dict.get('dependencies', []),
            models=[ModelConfig.from_dict(model_dict) for model_dict in config_dict.get('models', [])]
        )
