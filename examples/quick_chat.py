import random
from typing import List, Tuple
from agentflow.ell.integration import ELLIntegration
import os
import openai

# Get OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Create OpenAI client
client = openai.OpenAI(api_key=api_key)

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.7,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

names_list = [
    "Alice",
    "Bob",
    "Charlie",
    "Diana",
    "Eve",
    "George",
    "Grace",
    "Hank",
    "Ivy",
    "Jack",
]

@ell.with_ell(mode="simple")
async def create_personality() -> str:
    """Create a personality with a backstory."""
    name = random.choice(names_list)
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are backstoryGPT. Create a 3-sentence backstory for a character."
                },
                {
                    "role": "user",
                    "content": f"Create a backstory for {name}"
                }
            ],
            temperature=1.0
        )
        
        if not response or not response.choices or not response.choices[0].message:
            return f"Name: {name}\nBackstory: A mysterious individual with an unknown past."
            
        content = response.choices[0].message.content
        if not content:
            return f"Name: {name}\nBackstory: A mysterious individual with an unknown past."
            
        return f"Name: {name}\nBackstory: {content.strip()}"
        
    except Exception as e:
        print(f"Error creating personality: {str(e)}")
        return f"Name: {name}\nBackstory: A mysterious individual with an unknown past."

def format_message_history(message_history: List[Tuple[str, str]]) -> str:
    return "\n".join([f"{name}: {message}" for name, message in message_history])

@ell.with_ell(mode="simple")
async def chat(message_history: List[Tuple[str, str]], personality: str) -> str:
    """Generate a chat response based on personality and history."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": f"Here is your description.\n{personality}\n\nYour goal is to come up with a response to a chat. Only respond in one sentence (should be like a text message in informality.) Never use Emojis."
                },
                {
                    "role": "user",
                    "content": format_message_history(message_history)
                }
            ],
            temperature=0.3,
            max_tokens=20
        )
        
        if not response or not response.choices or not response.choices[0].message:
            return "..."
            
        content = response.choices[0].message.content
        if not content:
            return "..."
            
        return content.strip()
        
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return "..."

async def main():
    """Main function to run the chat simulation."""
    print("\nStarting chat simulation...")
    print("=" * 50)
    
    # Create personalities
    print("\nCreating personalities...")
    personalities = [await create_personality(), await create_personality()]
    
    # Extract names and backstories
    names = []
    backstories = []
    for personality in personalities:
        parts = list(filter(None, personality.split("\n")))
        names.append(parts[0].split(": ")[1])
        backstories.append(parts[1].split(": ")[1])
    
    print("\nParticipants:")
    for i, (name, backstory) in enumerate(zip(names, backstories)):
        print(f"\nCharacter {i+1}:")
        print(f"Name: {name}")
        print(f"Backstory: {backstory}")
    
    print("\nStarting conversation...")
    print("=" * 50)
    
    messages: List[Tuple[str, str]] = []
    whos_turn = 0
    
    for i in range(10):
        personality_talking = personalities[whos_turn]
        response = await chat(messages, personality=personality_talking)
        messages.append((names[whos_turn], response))
        print(f"\n{names[whos_turn]}: {response}")
        whos_turn = (whos_turn + 1) % len(personalities)
    
    print("\nChat simulation completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
