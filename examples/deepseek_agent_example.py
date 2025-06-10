"""
Example implementation of a DeepSeek-based agent using the OpenAI SDK with ELL integration.
This example demonstrates code generation and analysis capabilities.
"""

import json
import asyncio
import time
import logging
import sys
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from pathlib import Path
from openai import OpenAI

from agentflow.core.ell_integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   stream=sys.stdout)
logger = logging.getLogger(__name__)

# Get ELL integration instance
ell = ELLIntegration()

@dataclass
class DeepSeekAgentConfig:
    """Configuration for DeepSeek agent."""
    model_name: str = "deepseek-coder-33b-instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    context_window: int = 8192

class DeepSeekAgent:
    """Implementation of an agent using DeepSeek model."""
    
    def __init__(self, config_path: str):
        """Initialize agent with configuration."""
        try:
            logger.info("Loading configuration from %s", config_path)
            self.config = self._load_config(config_path)
            
            # Initialize DeepSeek configuration
            logger.info("Initializing DeepSeek configuration")
            self.model_config = DeepSeekAgentConfig()
            
            logger.info("Initializing ELL integration")
            ell.configure({
                "enabled": True,
                "tracking_enabled": True,
                "store": "./logdir",
                "verbose": True,
                "autocommit": True,
                "model": self.model_config.model_name,
                "default_model": self.model_config.model_name,
                "temperature": self.model_config.temperature,
                "max_tokens": self.model_config.max_tokens,
                "client": OpenAI(
                    base_url="https://api.deepseek.com/v1",
                    api_key="sk-88d0d7467fa84dfc996b9d09962fb75e"
                ),
                "base_url": "https://api.deepseek.com/v1"
            })
            
        except Exception as e:
            logger.error("Error initializing agent: %s", str(e), exc_info=True)
            raise
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load agent configuration from file."""
        with open(config_path, 'r') as f:
            return json.load(f)
    
    @ell.with_ell(mode="simple")
    async def process_input(self, input_text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process input text with optional context."""
        try:
            logger.info("Processing input: %s", input_text)
            start_time = time.time()
            
            # Create message for ELL
            message = Message(
                role=MessageRole.USER,
                content=input_text,
                metadata=context or {}
            )
            
            try:
                # Process message through ELL
                logger.info("Making API request to DeepSeek through ELL")
                response = await ell.process_message(message)
                
                # Calculate metrics
                end_time = time.time()
                latency = end_time - start_time
                
                # Process response
                if response and response.content:
                    result = {
                        "status": "success",
                        "data": {
                            "content": response.content,
                            "model": self.model_config.model_name,
                            "usage": response.metadata.get("usage", {}) if response.metadata else {}
                        },
                        "metrics": {
                            "latency": latency,
                            "tokens": response.metadata.get("usage", {}).get("total_tokens", 0) if response.metadata else 0,
                            "prompt_tokens": response.metadata.get("usage", {}).get("prompt_tokens", 0) if response.metadata else 0,
                            "completion_tokens": response.metadata.get("usage", {}).get("completion_tokens", 0) if response.metadata else 0
                        }
                    }
                    return result
                else:
                    logger.error("Empty response from ELL")
                    return {
                        "status": "error",
                        "data": {
                            "error": "Empty response from ELL"
                        },
                        "metrics": {
                            "latency": latency
                        }
                    }
                    
            except Exception as e:
                logger.error("ELL processing error: %s", str(e), exc_info=True)
                return {
                    "status": "error",
                    "data": {
                        "error": f"ELL processing error: {str(e)}"
                    },
                    "metrics": {
                        "latency": time.time() - start_time
                    }
                }
            
        except Exception as e:
            logger.error("Error in process_input: %s", str(e), exc_info=True)
            return {
                "status": "error",
                "data": {
                    "error": str(e)
                },
                "metrics": {
                    "latency": time.time() - start_time
                }
            }

async def main():
    """Example usage of DeepSeek agent."""
    try:
        logger.info("Starting DeepSeek agent example")
        # Initialize agent
        agent = DeepSeekAgent("examples/deepseek_agent_config.json")
        
        logger.info("Setting up code generation task")
        # Example code generation task
        code_task = """
        Write a Python function that implements a binary search algorithm.
        The function should be well-documented and include type hints.
        """
        
        # Process with context
        context = {
            "language": "python",
            "style": "clean and modern",
            "include_tests": True
        }
        
        logger.info("Processing input with context: %s", context)
        result = await agent.process_input(code_task, context)
        
        # Print results
        logger.info("Processing complete. Status: %s", result["status"])
        if result["status"] == "success":
            print("\nGenerated Code:")
            print("-" * 80)
            print(result["data"].get("content", "No code generated"))
            print("-" * 80)
            print("\nModel:", result["data"].get("model", "unknown"))
            print("\nUsage Metrics:")
            usage = result["data"].get("usage", {})
            print(f"Total Tokens: {usage.get('total_tokens', 0)}")
            print(f"Prompt Tokens: {usage.get('prompt_tokens', 0)}")
            print(f"Completion Tokens: {usage.get('completion_tokens', 0)}")
            print("\nPerformance Metrics:")
            metrics = result.get("metrics", {})
            print(f"Latency: {metrics.get('latency', 0):.2f} seconds")
        else:
            print("\nError details:", result.get("data", {}).get("error", "Unknown error"))
            
    except Exception as e:
        logger.error("Error in main: %s", str(e), exc_info=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 