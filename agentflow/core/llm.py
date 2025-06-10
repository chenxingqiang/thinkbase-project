"""Language Model Utilities for Agent Framework.

This module provides base classes and mock implementations 
for language models used in the agent framework.
"""

import json
import logging
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract base class for Language Models."""
    
    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Generate text based on the given prompt.
        
        Args:
            prompt: Input prompt for text generation
            
        Returns:
            Generated text
        """
        pass
    
    def parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from generated text.
        
        Args:
            text: Text potentially containing JSON
            
        Returns:
            Parsed JSON dictionary
        """
        try:
            # Remove any leading/trailing code block markers
            text = text.strip('`')
            return json.loads(text)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse JSON: {text}")
            return {}

class MockLLM(BaseLLM):
    """Mock Language Model for testing purposes."""
    
    async def generate(self, prompt: str) -> str:
        """Generate a mock response for testing.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Mock generated text
        """
        # Basic validation response
        if "validate" in prompt.lower():
            return json.dumps({
                "is_valid": True,
                "validation_errors": []
            })
        
        # Default mock response
        return json.dumps({
            "response": "Mock LLM response",
            "prompt": prompt
        })

class LLMConfig:
    """Configuration for Language Models."""
    
    def __init__(
        self, 
        model: str = "default-model",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        api_key: Optional[str] = None
    ):
        """Initialize LLM configuration.
        
        Args:
            model: Name of the language model
            temperature: Sampling temperature
            max_tokens: Maximum number of tokens to generate
            api_key: Optional API key for the model
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key

class DefaultLLM(BaseLLM):
    """Default LLM implementation."""
    
    async def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        # Default implementation returns empty string
        # Override this in production with actual LLM
        return ""
