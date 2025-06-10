"""
Chatbot Agent Example using AgentISA Framework.

This example demonstrates how to build a chatbot agent that:
1. Processes user messages
2. Maintains conversation context
3. Learns from interactions
4. Optimizes response patterns
"""

import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime

from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Message structure for chat."""
    content: str
    role: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ChatContext:
    """Maintains chat context and history."""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.user_profile: Dict[str, Any] = {}
        self.current_topic: Optional[str] = None
        
    def add_message(self, message: Message):
        """Add message to history."""
        self.messages.append(message)
        
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get most recent messages."""
        return self.messages[-count:]
        
    def update_topic(self, topic: str):
        """Update current conversation topic."""
        self.current_topic = topic

class ProcessMessageInstruction(Instruction):
    """Instruction for processing incoming messages."""
    
    def __init__(self):
        super().__init__(
            id="process_message",
            name="process_message",
            type=InstructionType.LLM_INTERACTION,
            params={
                "message": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=ChatContext)
            },
            description="Process incoming chat message"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        message = input_params["message"]
        context = input_params["context"]
        
        # Process message content
        processed_content = self._process_content(message)
        
        # Update context
        new_message = Message(
            content=message,
            role="user",
            timestamp=datetime.now()
        )
        context.add_message(new_message)
        
        # Detect topic
        topic = self._detect_topic(message, context)
        if topic:
            context.update_topic(topic)
        
        return {
            "processed_content": processed_content,
            "topic": topic,
            "requires_clarification": self._needs_clarification(message)
        }
        
    def _process_content(self, message: str) -> str:
        """Process message content."""
        # Add content processing logic
        return message.strip().lower()
        
    def _detect_topic(self, message: str, context: ChatContext) -> Optional[str]:
        """Detect conversation topic."""
        # Add topic detection logic
        return None
        
    def _needs_clarification(self, message: str) -> bool:
        """Check if message needs clarification."""
        # Add clarification check logic
        return False

class GenerateResponseInstruction(Instruction):
    """Instruction for generating chat responses."""
    
    def __init__(self):
        super().__init__(
            id="generate_response",
            name="generate_response",
            type=InstructionType.LLM_INTERACTION,
            params={
                "processed_content": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=ChatContext),
                "requires_clarification": TypeConstraint(base_type=bool)
            },
            description="Generate chat response"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        processed_content = input_params["processed_content"]
        context = input_params["context"]
        requires_clarification = input_params["requires_clarification"]
        
        if requires_clarification:
            response = self._generate_clarification(processed_content)
        else:
            response = self._generate_reply(processed_content, context)
            
        # Update context
        new_message = Message(
            content=response,
            role="assistant",
            timestamp=datetime.now()
        )
        context.add_message(new_message)
        
        return {
            "response": response,
            "confidence": self._calculate_confidence(response)
        }
        
    def _generate_clarification(self, content: str) -> str:
        """Generate clarification question."""
        return f"Could you please clarify what you mean by '{content}'?"
        
    def _generate_reply(self, content: str, context: ChatContext) -> str:
        """Generate response based on content and context."""
        # Add response generation logic
        return f"I understand you're talking about '{content}'"
        
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for response."""
        # Add confidence calculation logic
        return 0.8

class ChatbotAgent:
    """Main chatbot agent using AgentISA framework."""
    
    def __init__(self):
        # Configure AgentISA
        config = AgentISAConfig(
            enable_learning=True,
            enable_optimization=True,
            min_pattern_support=2,
            min_success_rate=0.8
        )
        self.agent_isa = AgentISA(config)
        self.context = ChatContext()
        
        # Initialize instructions
        asyncio.create_task(self._init_instructions())
        
    async def _init_instructions(self):
        """Initialize and register instructions."""
        self.process_instruction = ProcessMessageInstruction()
        self.generate_instruction = GenerateResponseInstruction()
        
        self.process_id = await self.agent_isa.register_instruction(
            self.process_instruction
        )
        self.generate_id = await self.agent_isa.register_instruction(
            self.generate_instruction
        )
        
    async def handle_message(self, message: str) -> str:
        """Handle incoming message and generate response."""
        try:
            # Process message
            process_result = await self.agent_isa.execute_instruction(
                self.process_id,
                {
                    "message": message,
                    "context": self.context
                }
            )
            
            # Generate response
            response_result = await self.agent_isa.execute_instruction(
                self.generate_id,
                {
                    "processed_content": process_result["processed_content"],
                    "context": self.context,
                    "requires_clarification": process_result["requires_clarification"]
                }
            )
            
            # Learn from interaction
            await self.agent_isa.learn_from_history()
            
            return response_result["response"]
            
        except Exception as e:
            logger.error(f"Error handling message: {str(e)}")
            return "I'm sorry, I encountered an error processing your message."
            
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return self.agent_isa.get_instruction_stats()

async def main():
    """Example usage of ChatbotAgent."""
    # Initialize agent
    agent = ChatbotAgent()
    
    # Example conversation
    messages = [
        "Hello!",
        "How can you help me?",
        "I need information about machine learning.",
        "What's the difference between supervised and unsupervised learning?",
        "Thank you for the explanation!"
    ]
    
    # Process messages
    for message in messages:
        print(f"\nUser: {message}")
        response = await agent.handle_message(message)
        print(f"Bot: {response}")
        
    # Print stats
    print("\nAgent Stats:")
    print(json.dumps(agent.get_stats(), indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 