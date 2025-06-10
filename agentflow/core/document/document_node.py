"""Document generation node implementation."""

from typing import Dict, Any, Optional
import logging

from core.node import AgentNode
from core.ell.integration import ELLIntegration
from ell import Message, MessageRole

logger = logging.getLogger(__name__)

class DocumentNode(AgentNode):
    """Node for generating documents."""
    
    def __init__(self, name: str, description: str, model_config: Optional[Dict[str, Any]] = None):
        """Initialize document node.
        
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
        self._ell: Optional[ELLIntegration] = None
    
    async def _initialize_impl(self) -> None:
        """Initialize ELL integration."""
        if not self._ell:
            self._ell = ELLIntegration()
            self._ell.configure({
                "model": self.model_config,
                "system_prompt": "You are a document generation assistant specializing in academic writing."
            })
    
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document generation task.
        
        Args:
            task: Document task to process
            
        Returns:
            Generated document
        """
        try:
            # Extract document parameters
            research_findings = task.get("findings", "")
            metadata = task.get("metadata", {})
            
            # Create document prompt
            prompt = f"""
            Generate an academic document based on the following research findings:
            {research_findings}
            
            Topic: {metadata.get('topic', '')}
            Academic Level: {metadata.get('academic_level', '')}
            Language: {metadata.get('language', {}).get('TYPE', 'English')}
            Style: {metadata.get('language', {}).get('STYLE', 'Academic')}
            Template: {metadata.get('template', '')}
            
            Please provide:
            1. A well-structured document following academic standards
            2. Proper citations and references
            3. Clear sections and headings
            4. LaTeX formatting where appropriate
            """
            
            # Create message object
            message = Message(
                role=MessageRole.USER,
                content=prompt,
                metadata={
                    "type": "document",
                    "topic": metadata.get('topic', ''),
                    "academic_level": metadata.get('academic_level', '')
                }
            )
            
            # Process with ELL
            if not self._ell:
                await self._initialize_impl()
            assert self._ell is not None  # Help type checker
            response = await self._ell.process_message(message)
            
            return {
                "document": response.content,
                "metadata": {
                    "format": "Markdown with LaTeX",
                    "source": metadata
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating document: {e}")
            raise
    
    async def _cleanup_impl(self) -> None:
        """Clean up ELL resources."""
        ell: ELLIntegration = self._ell  # type: ignore
        if ell is not None:
            await ell.cleanup()  # type: ignore
            self._ell = None 