"""Configuration module."""

from typing import Dict, Any, Optional, List, Union, Type, Callable, cast, TYPE_CHECKING
from enum import Enum
import asyncio
import numpy as np
from pydantic import BaseModel, Field, field_validator, ValidationError, ConfigDict, model_validator, PrivateAttr
import os
from pathlib import Path
import yaml
import copy
import time
import uuid
from datetime import datetime, date
from typing_extensions import Annotated
import logging
import re
import hashlib
import secrets
import hydra
from omegaconf import OmegaConf, DictConfig, ListConfig
import json

from .retry_policy import RetryPolicy
from .exceptions import ConfigurationError as BaseConfigurationError, WorkflowExecutionError
from .base_types import AgentType, AgentMode, AgentStatus, DictKeyType, MessageType
from .workflow_types import WorkflowConfig, WorkflowStepType
# AgentConfig/ModelConfig are canonically defined in agent_config (which
# has no config dependency); re-exported here for the historical
# ``agentflow.core.config`` import path.
from .agent_config import AgentConfig, ModelConfig

# Re-export the ConfigurationError from exceptions
ConfigurationError = BaseConfigurationError

if TYPE_CHECKING:
    from ..core.isa.selector import InstructionSelector

class ConfigSecurityManager:
    """Manages configuration security and sensitive data masking."""
    
    @staticmethod
    def mask_sensitive_fields(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mask sensitive fields in the configuration.
        
        Masks entire dictionaries if they contain sensitive data.
        Masks specific fields like passwords, api_keys, etc.
        
        Args:
            config (Dict[str, Any]): Configuration dictionary to mask
        
        Returns:
            Dict[str, Any]: Configuration with sensitive fields masked
        """
        if not isinstance(config, dict):
            return config

        masked_config = {}
        sensitive_keywords = ['password', 'secret', 'key', 'token', 'credential']

        for key, value in config.items():
            # Check if the key itself suggests sensitive data
            if any(keyword in key.lower() for keyword in sensitive_keywords):
                masked_config[key] = '***MASKED***'
                continue

            # Recursively mask nested dictionaries
            if isinstance(value, dict):
                # Check if the dictionary contains any sensitive keys
                if any(any(keyword in k.lower() for keyword in sensitive_keywords) for k in value.keys()):
                    masked_config[key] = '***MASKED***'
                else:
                    masked_config[key] = ConfigSecurityManager.mask_sensitive_fields(value)
            else:
                masked_config[key] = value

        return masked_config

    @classmethod
    def hash_sensitive_data(cls, data, salt=None):
        """
        Hash sensitive data with optional salt.
        
        Args:
            data (str): Data to hash
            salt (str, optional): Salt for additional security
        
        Returns:
            str: Hashed data
        """
        if salt is None:
            salt = secrets.token_hex(16)
        
        salted_data = f"{salt}{data}"
        return hashlib.sha256(salted_data.encode()).hexdigest()

class ConfigTypeConverter:
    """Enhanced type conversion with robust error handling."""
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any], schema: Dict[str, type]) -> Dict[str, Any]:
        """Validate configuration against a type schema."""
        validated = {}
        for key, value in config.items():
            if key in schema:
                try:
                    # Attempt type conversion
                    converted = cls.convert_value(value, schema[key], strict=True)
                    validated[key] = converted
                except (ValueError, TypeError):
                    # If conversion fails, preserve original value
                    validated[key] = value
                    logging.warning(f"Type conversion failed for key '{key}'. Preserving original value.")
            else:
                # If no schema for key, preserve original value
                validated[key] = value
        return validated

    @classmethod
    def convert_value(cls, value: Any, target_type: type, schema: Dict[str, Any] = Field(default_factory=dict), strict: bool = False) -> Any:
        """Convert a value to the specified target type."""
        # Handle None values
        if value is None:
            if target_type == dict:
                return {}
            return value
        
        # If the value is already of the target type, return it
        if isinstance(value, target_type):
            return value
        
        try:
            # Handle special cases for boolean conversion
            if target_type == bool:
                if isinstance(value, str):
                    value = value.lower().strip()
                    if value in ('true', 't', 'yes', 'y', '1', 'on'):
                        return True
                    if value in ('false', 'f', 'no', 'n', '0', 'off'):
                        return False
                    if strict:
                        raise ValueError(f"Cannot convert {value} to bool")
                    return False
                elif isinstance(value, (int, float)):
                    return bool(value)
                if strict:
                    raise ValueError(f"Cannot convert {value} to bool")
                return False
            
            # Handle integer conversion
            if target_type == int:
                if isinstance(value, str):
                    value = value.strip()
                    try:
                        float_val = float(value)
                        if float_val.is_integer():
                            return int(float_val)
                        if strict:
                            raise ValueError(f"Cannot convert non-integer float {value} to int")
                        return int(float_val)
                    except ValueError as e:
                        if strict:
                            raise ValueError(f"Cannot convert {value} to int: {e}")
                        return 0
                elif isinstance(value, float):
                    if value.is_integer():
                        return int(value)
                    if strict:
                        raise ValueError(f"Cannot convert non-integer float {value} to int")
                    return int(value)
                
            # Handle float conversion
            if target_type == float:
                if isinstance(value, str):
                    try:
                        return float(value.strip())
                    except ValueError as e:
                        if strict:
                            raise ValueError(f"Cannot convert {value} to float: {e}")
                        return 0.0
                    
            # Handle date conversion
            if target_type == date and isinstance(value, str):
                try:
                    return datetime.strptime(value.strip(), '%Y-%m-%d').date()
                except ValueError as e:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to date: {e}")
                    return None
                
            # Handle nested type conversion for dictionaries
            if target_type == dict:
                if isinstance(value, dict):
                    # Validate and convert nested dictionary according to schema
                    return {str(k): cls.convert_value(v, schema.get(str(k), type(v)), strict=strict)
                       for k, v in value.items()}
                elif isinstance(value, list):
                    # Convert list to dict with indices as keys
                    return {str(i): v for i, v in enumerate(value)}
                elif isinstance(value, str):
                    # Convert string to dict with 'value' key
                    return {'value': value}
                else:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to dict")
                    return {}
                
            # Handle list type conversion
            if target_type == list:
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return [value]  # Convert single string to single-item list
                try:
                    return list(value)  # Try to convert other iterables to list
                except (TypeError, ValueError) as e:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to list: {e}")
                    return [value]  # Return single-item list as fallback
                
            # Attempt direct type conversion
            return target_type(value)
        
        except (TypeError, ValueError) as e:
            if strict:
                raise ValueError(f"Cannot convert {value} to {target_type.__name__}: {e}")
            
            # If conversion fails and strict is False, return empty dict for dict type or original value
            return {} if target_type is dict else value

class ConfigurationInheritanceResolver:
    """
    Resolver for configuration inheritance with advanced merging capabilities.
    """
    @classmethod
    def _load_yaml_config(cls, config_path: str) -> Dict[str, Any]:
        """
        Load a YAML configuration file.
        
        Args:
            config_path (str): Path to the YAML configuration file
        
        Returns:
            dict: Loaded configuration dictionary
        """
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_path}: {e}")

    @classmethod
    def resolve_inheritance(cls, base_config: Union[Dict[str, Any], DictConfig], override_config: Union[Dict[str, Any], DictConfig]) -> DictConfig:
        """
        Resolve configuration inheritance by merging base and override configurations.
        
        Args:
            base_config: Base configuration
            override_config: Configuration to override base with
        
        Returns:
            DictConfig: Merged configuration
        """
        try:
            # Convert inputs to OmegaConf if they aren't already
            if not isinstance(base_config, DictConfig):
                base_config = OmegaConf.create(base_config)
            if not isinstance(override_config, DictConfig):
                override_config = OmegaConf.create(override_config)
            
            # Merge configurations with interpolation
            merged = OmegaConf.merge(base_config, override_config)
            
            # Ensure the merged config is resolved
            merged = OmegaConf.to_container(merged, resolve=True)
            
            # Convert back to DictConfig with proper structure preservation
            result = OmegaConf.create(merged)
            
            # Ensure the result is a DictConfig
            if not isinstance(result, DictConfig):
                result = OmegaConf.create({})
            
            return result
            
        except Exception as e:
            raise ConfigurationError(f"Failed to resolve configuration inheritance: {e}")

    @classmethod
    def load_config_with_inheritance(cls, config_path: str, base_config: str, environment: str) -> Dict[str, Any]:
        """Load configuration with inheritance.
        
        Args:
            config_path: Path to configuration directory
            base_config: Base configuration name
            environment: Environment name
            
        Returns:
            Dict[str, Any]: Merged configuration
        """
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration path {config_path} does not exist")

        base_file = os.path.join(config_path, f"{base_config}.yaml")
        env_file = os.path.join(config_path, f"{base_config}.{environment}.yaml")
        
        if not os.path.exists(base_file):
            raise ConfigurationError(f"Base configuration file {base_file} does not exist")
            
        try:
            with open(base_file, 'r') as f:
                base_config_data = yaml.safe_load(f)
                
            if os.path.exists(env_file):
                with open(env_file, 'r') as f:
                    env_config = yaml.safe_load(f)
                    
                # Deep merge configurations
                merged_config = cls._deep_merge(base_config_data, env_config)
            else:
                merged_config = base_config_data
                
            return merged_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
            
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigurationInheritanceResolver._deep_merge(merged[key], value)
            else:
                merged[key] = value
                
        return merged

class ConfigurationType(str, Enum):
    """Configuration type enumeration."""
    AGENT = "agent"
    MODEL = "model"
    WORKFLOW = "workflow"
    GENERIC = "generic"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    DATA_SCIENCE = "data_science"
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    MONITORING = "monitoring"
    MAINTENANCE = "maintenance"

class StepConfig(BaseModel):
    """Step configuration."""
    strategy: str
    params: Dict[str, Any] = Field(default_factory=dict)
    
class WorkflowStep(BaseModel):
    """Workflow step configuration."""
    id: str
    name: str
    type: WorkflowStepType
    config: StepConfig
    
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the workflow step.
        
        Args:
            context: Execution context
            
        Returns:
            Step execution result
        """
        if self.type == WorkflowStepType.TRANSFORM:
            if self.config.strategy == "feature_engineering":
                data = context["data"]
                params = self.config.params
                method = params.get("method", "standard")
                if method == "standard":
                    with_mean = params.get("with_mean", True)
                    with_std = params.get("with_std", True)
                    mean = np.mean(data, axis=0) if with_mean else 0
                    std = np.std(data, axis=0) if with_std else 1
                    transformed_data = (data - mean) / std
                    return {"output": {"data": transformed_data}}
                elif method == "isolation_forest":
                    threshold = params.get("threshold", 0.1)
                    # Simple outlier detection using mean and std
                    mean = np.mean(data, axis=0)
                    std = np.std(data, axis=0)
                    z_scores = np.abs((data - mean) / std)
                    outliers = np.any(z_scores > threshold, axis=1)
                    transformed_data = data[~outliers]
                    return {"output": {"data": transformed_data}}
                    
        return {"output": context}

def load_global_config(config_path: Optional[str] = None, config_name: str = 'base') -> Dict[str, Any]:
    """
    Load the global configuration using Hydra with enhanced error handling.

    Args:
        config_path: Path to configuration directory
        config_name: Name of the configuration file

    Returns:
        Global configuration as a dictionary
    """
    if config_path is None:
        config_path = os.path.join('..', '..', 'config')
    
    try:
        # Ensure the path is relative to the current file
        base_dir = os.path.dirname(__file__)
        relative_config_path = os.path.relpath(
            os.path.normpath(os.path.join(base_dir, config_path)), 
            base_dir
        )
        
        with hydra.initialize(config_path=relative_config_path, version_base=None):
            config = hydra.compose(config_name=config_name)
            
            # Convert to dictionary to ensure mutability
            config_dict = OmegaConf.to_container(config, resolve=True)
            if not isinstance(config_dict, dict):
                config_dict = {}
            
            # Create a new dictionary with the 'global_' key
            result: Dict[str, Any] = {
                'global_': {
                    'logging_level': config_dict.get('logging_level', 'INFO') if isinstance(config_dict, dict) else 'INFO'
                }
            }
            
            return result
    except Exception as e:
        logging.error(f"Error loading global configuration: {e}")
        raise ConfigurationError(f"Failed to load global configuration: {e}")

def _parse_publication_goals(goals: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse publication goals, converting date strings to date objects.
    
    Args:
        goals: Dictionary of publication goals
    
    Returns:
        Parsed publication goals
    """
    if 'acceptance_deadline' in goals and isinstance(goals['acceptance_deadline'], str):
        try:
            goals['acceptance_deadline'] = datetime.strptime(goals['acceptance_deadline'], '%Y-%m-%d').date()
        except ValueError:
            # Keep original string if parsing fails
            pass
    return goals

def convert_type(value: Any, target_type: type, schema: Optional[Dict[str, Any]] = None, strict: bool = False) -> Any:
    """Convert a value to the specified target type with Hydra-enhanced type conversion."""
    # If using Hydra's OmegaConf, leverage its type conversion
    if isinstance(value, (DictConfig, ListConfig)):
        try:
            value = OmegaConf.to_container(value, resolve=True)
        except Exception as e:
            logging.warning(f"Could not convert OmegaConf to object: {e}")
    
    # If the value is already of the target type, return it
    if isinstance(value, target_type):
        return value
    
    try:
        # Handle nested type conversion for dictionaries
        if target_type is dict and isinstance(value, dict):
            if schema:
                # Validate and convert nested dictionary according to schema
                return {str(k): convert_type(v, schema.get(str(k), type(v)), strict=strict)
                       for k, v in value.items()}
            return value
        
        # Handle list type conversion
        if target_type is list and isinstance(value, list):
            return value
        
        # Handle None values
        if value is None:
            if strict:
                raise TypeError(f"Cannot convert None to {target_type}")
            return {} if target_type is dict else value
        
        # Attempt direct type conversion
        return target_type(value)
    
    except (TypeError, ValueError) as e:
        if strict:
            raise TypeError(f"Could not convert {value} to {target_type}: {e}")
        
        # If conversion fails and strict is False, return empty dict for dict type or original value
        return {} if target_type is dict else value

class ConfigSecurityManagerHydra:
    @staticmethod
    def mask_sensitive_fields(config: Union[Dict[str, Any], DictConfig]) -> Dict[str, Any]:
        """Mask sensitive fields in configuration."""
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        
        masked = {}
        sensitive_fields = {'password', 'key', 'token', 'secret', 'database'}
        
        for key, value in config.items():
            if isinstance(value, dict):
                if any(field in key.lower() for field in sensitive_fields):
                    masked[key] = '***MASKED***'
                else:
                    masked[key] = ConfigSecurityManager.mask_sensitive_fields(value)
            elif isinstance(value, list):
                masked[key] = [
                    ConfigSecurityManager.mask_sensitive_fields(item) if isinstance(item, dict)
                    else '***MASKED***' if any(field in str(item).lower() for field in sensitive_fields)
                    else item
                    for item in value
                ]
            else:
                if any(field in key.lower() for field in sensitive_fields):
                    masked[key] = '***MASKED***'
                else:
                    masked[key] = value
                
        return masked

class ConfigurationInheritanceResolverHydra:
    @classmethod
    def resolve_inheritance(cls, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve configuration inheritance with Hydra's OmegaConf deep merging."""
        # Convert inputs to DictConfig if they aren't already
        base_conf = OmegaConf.create(base_config)
        override_conf = OmegaConf.create(override_config)
            
        # Merge configurations and convert back to dict
        merged = OmegaConf.merge(base_conf, override_conf)
        result = OmegaConf.to_container(merged, resolve=True)
        
        # Ensure we return a dictionary with string keys
        if isinstance(result, dict):
            return {str(k): v for k, v in result.items()}
        return {}

class BaseAgentConfigHydra(BaseModel):
    """Base configuration for agents with Hydra integration."""
    
    @classmethod
    def from_yaml(cls, config_path: str, config_name: str, environment: str = 'default') -> 'BaseAgentConfigHydra':
        """Load configuration from YAML using Hydra's configuration management."""
        try:
            # Validate config path
            if not config_path or not os.path.exists(config_path):
                raise ConfigurationError(f"Configuration path {config_path} does not exist.")

            # Validate config name
            if not config_name:
                raise ConfigurationError("Configuration name cannot be empty.")

            # Use Hydra's composition to merge configurations
            with hydra.initialize(version_base=None, config_path=os.path.relpath(config_path)):
                try:
                    # Compose configuration across environments
                    cfg = hydra.compose(
                        config_name=config_name,
                        overrides=[f'+group={environment}']
                    )
                except Exception as compose_error:
                    raise ConfigurationError(f"Failed to compose configuration: {compose_error}")

                # Convert to dictionary and validate
                try:
                    config_dict = OmegaConf.to_container(cfg, resolve=True)
                    if not isinstance(config_dict, dict):
                        config_dict = {}
                except Exception as convert_error:
                    raise ConfigurationError(f"Failed to convert configuration: {convert_error}")

                # Convert all keys to strings to ensure proper type handling
                str_config = {str(k): v for k, v in config_dict.items()}

                # Validate configuration structure
                try:
                    # Create instance with validated dictionary
                    instance = cls(**str_config)
                except Exception as validation_error:
                    raise ConfigurationError(f"Configuration validation failed: {validation_error}")

                # Mask sensitive data for logging
                masked_config = ConfigSecurityManagerHydra.mask_sensitive_fields(str_config)
                logging.info(f"Loaded configuration: {masked_config}")

                return instance

        except Exception as e:
            # Log the error and raise a ConfigurationError
            logging.error(f"Configuration loading error: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BaseAgentConfigHydra':
        """Create configuration from a dictionary with Hydra-style validation."""
        try:
            # Convert dictionary to OmegaConf for type resolution
            cfg = DictConfig(config_dict)

            # Validate and convert configuration
            validated_config = convert_type(cfg, dict)
            if not isinstance(validated_config, dict):
                validated_config = {}

            # Convert all keys to strings
            str_config = {str(k): v for k, v in validated_config.items()}

            return cls(**str_config)

        except (ValidationError, ValueError) as e:
            logging.error(f"Configuration validation error: {e}")
            raise ConfigurationError(f"Invalid configuration: {e}")

class ResearchAgentConfigHydra(BaseAgentConfigHydra):
    """Specialized configuration for research-oriented agents with Hydra validation."""
    pass

class DynamicConfigGenerator:
    """
    Utility class for generating dynamic configurations with advanced features.
    """
    
    @classmethod
    def generate_config_from_function(cls, config_generator: Callable[[], Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a configuration by calling a configuration generator function.
        
        Args:
            config_generator (callable): A function that returns a configuration dictionary
        
        Returns:
            dict: Generated configuration
        """
        return config_generator()
    
    @classmethod
    def generate_timestamped_config(cls, base_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a timestamped configuration by adding metadata to a base configuration.
        
        Args:
            base_config (dict): Base configuration dictionary
        
        Returns:
            dict: Configuration with added metadata
        """
        timestamped_config = copy.deepcopy(base_config)
        timestamped_config['_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'version': str(uuid.uuid4())[:8]
        }
        return timestamped_config

class ConfigTypeConverterHydra:
    """Enhanced type conversion with Hydra integration."""
    
    @classmethod
    def convert_value(cls, value: Any, target_type: type, schema: Dict[str, Any] = Field(default_factory=dict), strict: bool = False) -> Any:
        """Convert a value to the specified target type with Hydra-enhanced type conversion."""
        # Handle None values
        if value is None:
            if target_type == dict:
                return {}
            return value

        # If using Hydra's OmegaConf, leverage its type conversion
        if isinstance(value, (DictConfig, ListConfig)):
            try:
                value = OmegaConf.to_container(value, resolve=True)
            except Exception as e:
                logging.warning(f"Could not convert OmegaConf to object: {e}")
        
        # If the value is already of the target type, return it
        if isinstance(value, target_type):
            return value
        
        try:
            # Handle special cases for boolean conversion
            if target_type == bool:
                if isinstance(value, str):
                    value = value.lower().strip()
                    if value in ('true', 't', 'yes', 'y', '1', 'on'):
                        return True
                    if value in ('false', 'f', 'no', 'n', '0', 'off'):
                        return False
                    if strict:
                        raise ValueError(f"Cannot convert {value} to bool")
                    return False
                elif isinstance(value, (int, float)):
                    return bool(value)
                if strict:
                    raise ValueError(f"Cannot convert {value} to bool")
                return False

            # Handle integer conversion
            if target_type == int:
                if isinstance(value, str):
                    value = value.strip()
                    try:
                        # Handle strings like "42.0"
                        float_val = float(value)
                        if float_val.is_integer():
                            return int(float_val)
                        if strict:
                            raise ValueError(f"Cannot convert non-integer float {value} to int")
                        return int(float_val)
                    except ValueError as e:
                        if strict:
                            raise ValueError(f"Cannot convert {value} to int: {e}")
                        return 0
                elif isinstance(value, float):
                    if value.is_integer():
                        return int(value)
                    if strict:
                        raise ValueError(f"Cannot convert non-integer float {value} to int")
                    return int(value)

            # Handle date conversion
            if target_type == date and isinstance(value, str):
                try:
                    return datetime.strptime(value.strip(), '%Y-%m-%d').date()
                except ValueError as e:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to date: {e}")
                    return None

            # Handle nested type conversion for dictionaries
            if target_type is dict:
                if isinstance(value, dict):
                    # Validate and convert nested dictionary according to schema
                    return {str(k): cls.convert_value(v, schema.get(str(k), type(v)), strict=strict)
                           for k, v in value.items()}
                elif isinstance(value, list):
                    # Convert list to dict with indices as keys
                    return {str(i): v for i, v in enumerate(value)}
                elif isinstance(value, str):
                    # Convert string to dict with 'value' key
                    return {'value': value}
                else:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to dict")
                    return {}
            
            # Handle list type conversion
            if target_type is list:
                if isinstance(value, list):
                    return value
                if isinstance(value, str):
                    return [value]  # Convert single string to single-item list
                try:
                    return list(value)  # Try to convert other iterables to list
                except (TypeError, ValueError) as e:
                    if strict:
                        raise ValueError(f"Cannot convert {value} to list: {e}")
                    return [value]  # Return single-item list as fallback
            
            # Attempt direct type conversion
            return target_type(value)
        
        except (TypeError, ValueError) as e:
            if strict:
                raise ValueError(f"Cannot convert {value} to {target_type}: {e}")
            
            # If conversion fails and strict is False, return empty dict for dict type or original value
            return {} if target_type is dict else value

class ResearchAgentConfig(BaseModel):
    """Research agent configuration."""
    research_context: Dict[str, Any] = Field(default_factory=dict)
    publication_goals: Dict[str, Any] = Field(default_factory=dict)
    domain_knowledge: Dict[str, Any] = Field(default_factory=dict)
    research_methods: List[str] = Field(default_factory=list)
    analysis_tools: List[str] = Field(default_factory=list)
    name: str = Field(default="Research_Agent", description="Name of the research agent", min_length=3, pattern=r'^[a-zA-Z0-9_\- ]+$')
    type: str = Field(default="research", description="Type of the agent")

    model_config = ConfigDict(
        validate_assignment=True,
        extra='allow'
    )

    @classmethod
    def validate_research_context(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate research context."""
        if not isinstance(v, dict):
            raise ValueError("Research context must be a dictionary")
        return v

    @classmethod
    def validate_publication_goals(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Validate publication goals."""
        if not isinstance(v, dict):
            raise ValueError("Publication goals must be a dictionary")
        
        # Parse acceptance deadline to date if it's a string
        if 'acceptance_deadline' in v and isinstance(v['acceptance_deadline'], str):
            try:
                v['acceptance_deadline'] = datetime.strptime(v['acceptance_deadline'], '%Y-%m-%d').date()
            except ValueError:
                raise ValueError("Invalid date format for acceptance_deadline. Use YYYY-MM-DD")
        
        return v

    def __init__(self, **data):
        """Initialize research agent configuration."""
        # Validate and transform publication goals
        if 'publication_goals' in data:
            data['publication_goals'] = self.validate_publication_goals(data['publication_goals'])
        
        # If no name is provided, use the default
        if 'name' not in data:
            data['name'] = 'Research_Agent'
        
        super().__init__(**data)

    @classmethod
    def from_yaml(cls, config_path: str, config_name: str, environment: str = 'default') -> 'ResearchAgentConfig':
        """Load configuration from ``<config_path>/<config_name>.yaml``.

        The file is loaded directly from the caller-provided directory:
        Hydra's ``initialize`` resolves ``config_path`` relative to the
        calling module, which for a packaged module like this one silently
        points at ``agentflow/core/config/agents`` instead of the requested
        directory.  A direct load keeps the resolution predictable.
        """
        try:
            config_dir = Path(config_path)
            config_file = config_dir / f"{config_name}.yaml"
            if not config_file.is_file():
                raise ConfigurationError(
                    f"Configuration file {config_file} does not exist."
                )
            with open(config_file, "r", encoding="utf-8") as fh:
                config_dict = yaml.safe_load(fh)
            if not isinstance(config_dict, dict):
                raise ConfigurationError(
                    f"Configuration file {config_file} is not a mapping."
                )
            instance = cls(**{str(k): v for k, v in config_dict.items()})
            return instance
        except ConfigurationError:
            raise
        except Exception as e:
            logging.error(f"Configuration loading error: {e}")
            raise ConfigurationError(f"Failed to load configuration: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = {
            "research_context": self.research_context,
            "publication_goals": self.publication_goals,
            "domain_knowledge": self.domain_knowledge,
            "research_methods": self.research_methods,
            "analysis_tools": self.analysis_tools,
            "name": self.name,
            "type": self.type
        }
        return data

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ResearchAgentConfig':
        """Create configuration from a dictionary."""
        return cls(**config_dict)

@staticmethod
def convert_value(value: Any, target_type: type, schema: Optional[Dict[str, type]] = None) -> Any:
    """Convert a value to the target type."""
    if value is None:
        return None
        
    if target_type == str:
        return str(value)
        
    if target_type == int:
        try:
            if isinstance(value, str):
                return int(value.strip())
            return int(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {value} to int")
            
    if target_type == float:
        try:
            if isinstance(value, str):
                return float(value.strip())
            return float(value)
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {value} to float")
            
    if target_type == bool:
        if isinstance(value, str):
            value = value.lower().strip()
            if value in ('true', 't', 'yes', 'y', '1'):
                return True
            if value in ('false', 'f', 'no', 'n', '0'):
                return False
        return bool(value)
        
    if target_type == date:
        try:
            if isinstance(value, str):
                return date.fromisoformat(value)
            return value
        except (ValueError, TypeError):
            raise ValueError(f"Cannot convert {value} to date")
            
    if target_type == list:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]
        
    if target_type == dict:
        if isinstance(value, dict):
            if schema:
                return {k: ConfigTypeConverter.convert_value(v, schema[k]) for k, v in value.items() if k in schema}
            return value
        if isinstance(value, str):
            return {'value': value}
        if isinstance(value, (list, tuple)):
            return {str(i): v for i, v in enumerate(value)}
        return {'value': value}
            
    return value

class DistributedConfig(BaseModel):
    """Configuration for distributed workflow execution."""
    ray_address: Optional[str] = Field(default="auto", description="Ray cluster address")
    num_cpus: Optional[int] = Field(default=None, description="Number of CPUs to use")
    num_gpus: Optional[int] = Field(default=None, description="Number of GPUs to use")
    memory: Optional[int] = Field(default=None, description="Memory limit in bytes")
    object_store_memory: Optional[int] = Field(default=None, description="Object store memory limit in bytes")
    runtime_env: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Runtime environment configuration")
    namespace: Optional[str] = Field(default=None, description="Ray namespace")
    
    class Config:
        """Pydantic configuration."""
        extra = "allow"
        arbitrary_types_allowed = True

# ModelConfig is re-exported from agent_config at the top of this module.