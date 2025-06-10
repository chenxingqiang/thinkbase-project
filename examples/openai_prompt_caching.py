from typing import List
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

@ell.with_ell(mode="simple")
async def cached_chat(history: List[str], new_message: str) -> str:
    """You are a helpful assistant who chats with the user. 
    Your response should be < 2 sentences."""
    
    prompt = f"Here is the chat history:\n{chr(10).join(history)}\nPlease respond to this message:\n{new_message}"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Keep your responses brief - no more than 2 sentences."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7
        )
        
        if not response or not response.choices or not response.choices[0].message:
            return "I apologize, but I couldn't generate a response at this time."
            
        content = response.choices[0].message.content
        if not content:
            return "I apologize, but I couldn't generate a response at this time."
            
        return content.strip()
        
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        return "I apologize, but I encountered an error while processing your request."

async def main():
    """Main function to demonstrate chat caching."""
    history = []
    simulate_user_messages = [
        "Hello, how are you?",
        "What's the weather like today?",
        "Can you recommend a good book?",
        "Tell me a joke.",
        "What's your favorite color?",
        "How do you make pancakes?",
    ]

    print("\nStarting chat simulation with caching...")
    print("=" * 50)

    for message in simulate_user_messages:
        print(f"\nUser: {message}")
        response = await cached_chat(history, message)
        print(f"Assistant: {response}")
        
        history.append(f"User: {message}")
        history.append(f"Assistant: {response}")
        
    print("\nChat simulation completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

