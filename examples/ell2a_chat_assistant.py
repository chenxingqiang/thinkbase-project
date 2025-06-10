"""
Chat Assistant Example using ELL
Demonstrates multi-turn conversations and tool usage
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime

from agentflow.ell_integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

# Get ELL integration instance
ell = ELLIntegration()

# Configure ELL
ell.configure({
    "simple": {
        "model": "gpt-4",
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 30.0
    },
    "complex": {
        "model": "gpt-4",
        "max_retries": 3,
        "retry_delay": 1.0,
        "timeout": 60.0,
        "stream": True,
        "track_performance": True,
        "track_memory": True
    }
})

# Define tools
async def get_current_time() -> str:
    """Get the current time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

async def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information."""
    # Simulate knowledge base search
    kb = {
        "python": "Python is a high-level programming language.",
        "ell": "ELL is a Language Model Programming Library.",
        "llm": "LLM stands for Large Language Model."
    }
    return kb.get(query.lower(), "No information found.")

# Define chat assistant
@ell.with_ell(mode="complex")
async def chat_assistant(
    user_message: str,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> List[Message]:
    """An intelligent assistant that can engage in conversations and use tools."""
    # Initialize conversation history if None
    if conversation_history is None:
        conversation_history = []
    
    # Build messages list
    messages = [
        Message(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that can engage in natural conversations "
                   "and use tools to provide accurate information. "
                   "Always be concise and friendly.",
            type=MessageType.TEXT
        )
    ]
    
    # Add conversation history
    for msg in conversation_history:
        if msg["role"] == "user":
            messages.append(Message(role=MessageRole.USER, content=msg["content"], type=MessageType.TEXT))
        elif msg["role"] == "assistant":
            messages.append(Message(role=MessageRole.ASSISTANT, content=msg["content"], type=MessageType.TEXT))
    
    # Add current user message
    messages.append(Message(role=MessageRole.USER, content=user_message, type=MessageType.TEXT))
    
    # Return messages list
    return messages

async def interactive_chat():
    """Run an interactive chat session."""
    conversation_history: List[Dict[str, str]] = []
    print("Chat Assistant: Hello! How can I help you today? (type 'exit' to end)")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() == 'exit':
                print("\nChat Assistant: Goodbye!")
                break
            
            # Get assistant response
            messages = await chat_assistant(user_input, conversation_history)
            response = await ell.process_message(messages[-1])
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Process response
            if response and response.content:
                content = str(response.content)
                print(f"\nChat Assistant: {content}")
                conversation_history.append({"role": "assistant", "content": content})
                
                # Add tool responses
                current_time = await get_current_time()
                print(f"[Tool] Current time: {current_time}")
                
                if "python" in user_input.lower() or "ell" in user_input.lower() or "llm" in user_input.lower():
                    result = await search_knowledge_base(user_input.lower())
                    print(f"[Tool] Knowledge base: {result}")
            else:
                print("\nChat Assistant: I apologize, but I encountered an error processing your request.")
                
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Chat Assistant: I apologize for the error. Please try again.")

if __name__ == "__main__":
    print("Starting chat assistant...")
    asyncio.run(interactive_chat()) 