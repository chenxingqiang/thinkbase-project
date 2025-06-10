"""Research node implementation."""

from typing import Dict, Any, Optional, Protocol, runtime_checkable
import logging
import asyncio

from .node import AgentNode
from ..ell.integration import ELLIntegration
from ..ell.types.message import Message, MessageRole

logger = logging.getLogger(__name__)

@runtime_checkable
class ELLProtocol(Protocol):
    def cleanup(self) -> None: ...  # Not async in actual implementation
    async def process_message(self, message: Message) -> Any: ...
    def configure(self, config: Dict[str, Any]) -> None: ...

class ResearchNode(AgentNode):
    """Node for executing research tasks."""
    
    def __init__(self, name: str, description: str, model_config: Optional[Dict[str, Any]] = None):
        """Initialize research node.
        
        Args:
            name: Node name
            description: Node description
            model_config: Configuration for the language model
        """
        super().__init__(name, description)
        self.model_config = model_config or {
            "provider": "openai",
            "model": "gpt-3.5-turbo",
            "temperature": 0.7
        }
        self._ell: Optional[ELLProtocol] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize ELL integration."""
        if not self._ell:
            self._ell = ELLIntegration()
            self._ell.configure({
                "model": self.model_config,
                "system_prompt": "You are a research assistant helping with academic research."
            })
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a research task.
        
        Args:
            task: Research task to process
            
        Returns:
            Research findings
        """
        try:
            # Initialize if needed
            if not self._ell:
                await self._initialize_impl()
            
            assert isinstance(self._ell, ELLProtocol), "ELL not properly initialized"
            
            # Extract research parameters
            topic = task.get("RESEARCH_TOPIC", "")
            academic_level = task.get("ACADEMIC_LEVEL", "")
            language = task.get("LANGUAGE", {})
            template = task.get("TEMPLATE", "")
            
            # Create research prompt
            prompt = f"""
            Conduct research on the topic: {topic}
            Academic Level: {academic_level}
            Language: {language.get('TYPE', 'English')}
            Style: {language.get('STYLE', 'Academic')}
            Template: {template}
            
            Please provide:
            1. Key findings and insights
            2. Relevant academic sources
            3. Structured analysis
            """
            
            # Create message object
            message = Message(
                role=MessageRole.USER,
                content=prompt,
                metadata={
                    "type": "research",
                    "topic": topic,
                    "academic_level": academic_level
                }
            )
            
            # Process with ELL
            response = await self._ell.process_message(message)
            
            return {
                "findings": response.content,
                "metadata": {
                    "topic": topic,
                    "academic_level": academic_level,
                    "language": language,
                    "template": template
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing research task: {e}")
            raise
    
    async def _cleanup_impl(self) -> None:
        """Clean up ELL resources."""
        if isinstance(self._ell, ELLProtocol) and callable(getattr(self._ell, 'cleanup', None)):
            self._ell.cleanup()  # Not async
            self._ell = None 