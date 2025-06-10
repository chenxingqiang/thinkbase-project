import random
import numpy as np
import os
from typing import List, Union
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType
from agentflow.ell import ELL
import time

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL with API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

def process_joke_message(message: Message) -> Message:
    """Process a joke message using GPT-4."""
    # Here we would normally call the GPT-4 API
    # For now, let's return a mock response
    content = str(message.content)
    
    if message.role == MessageRole.SYSTEM:
        # Pass through system messages
        return message
        
    # For user messages, generate a joke response
    return Message(
        role=MessageRole.ASSISTANT,
        content="""(Walks onto stage, adjusts microphone)

Hey everyone! So, I've been playing a lot of Minecraft lately... (Pauses, looks around nervously)

You know what's really messed up? You spend hours building this beautiful house, and then some green dude with no arms just shows up and goes 'SSSSSS' (Makes hissing sound, audience laughs)

I mean, who designed this game? It's like a home renovation show where the reveal is always an explosion! (Gestures explosion with hands)

(Shakes head) And don't even get me started on the villagers. They're like the worst neighbors ever - all they do is make 'hmm' sounds and try to trade you wheat for emeralds. I tried that at Home Depot once... security wasn't happy. (Audience laughs)

(Takes a step back, serious face) But you know what's the real kicker? In Minecraft, you can punch trees to get wood. I tried that in real life... (Rubs hand) Now I need both a therapist AND a hand surgeon! (Big laugh)

(Bows) Thank you, you've been great! Just don't try any of this at home... especially the tree punching part!""",
        type=MessageType.TEXT,
        metadata={
            "model": "gpt-4",
            "timestamp": time.time(),
            "type": MessageType.TEXT,
            "status": "success",
            "original_content": content
        }
    )

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",  # Changed to GPT-4
    "default_model": "gpt-4",  # Changed to GPT-4
    "temperature": 0.7,  # Higher temperature for more creative jokes
    "mode": "complex",  # Changed to complex mode
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

# Initialize ELL instance with our process_message_fn
ell.ell = ELL(
    messages=[],
    metadata={
        "model": "gpt-4",  # Changed to GPT-4
        "config": ell.config
    },
    process_message_fn=process_joke_message
)

def get_random_length():
    return int(np.random.beta(2, 5) * 300)

@ell.with_ell(mode="complex")
async def tell_joke(topic: str) -> List[Message]:
    """Generate a standup comedy routine about a given topic."""
    length = get_random_length()
    
    # Create system message
    system_msg = Message(
        role=MessageRole.SYSTEM,
        content="""You are a professional standup comedian performing at a comedy club.
Your jokes are clever, engaging, and well-structured.
You always include stage directions and timing notes in parentheses.
You make your routines natural and conversational, as if performing live.
You start with a creative premise, build anticipation with a good setup,
and deliver strong punchlines that tie everything together.""",
        type=MessageType.TEXT,
        metadata={"type": "text"}
    )
    
    # Create user message
    user_msg = Message(
        role=MessageRole.USER,
        content=f"Create a {length}-word standup comedy routine about {topic}.",
        type=MessageType.TEXT,
        metadata={"type": "text"}
    )
    
    # Process system message first
    await ell.process_message(system_msg)
    
    # Then process user message
    return [system_msg, user_msg]

async def main():
    # Generate the joke
    print("\nGenerating a joke about Minecraft...")
    messages = await tell_joke("minecraft")
    
    # Process the messages
    print("\nJoke:")
    response = await ell.process_message(messages[-1])  # Process the last message
    print(response.content if response else "Failed to generate joke")
    print()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())