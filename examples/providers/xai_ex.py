"""
X.AI example: This example demonstrates using X.AI's Grok models.
Make sure you have set the XAI_API_KEY environment variable.
"""
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType
import openai
import os

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

try:
    # Get X.AI API key
    api_key = get_env_var("XAI_API_KEY")
    
    # Create X.AI client
    client = openai.OpenAI(
        base_url="https://api.x.ai/v1",
        api_key=api_key
    )

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- XAI_API_KEY")
    exit(1)

async def chat_with_grok(prompt: str, model: str = "grok-2") -> str:
    """Chat with Grok model."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"Error in conversation: {str(e)}"

async def generate_joke() -> str:
    """Generate a joke using Grok."""
    return await chat_with_grok("Tell me a funny joke! Be creative and original.")

async def answer_question(question: str) -> str:
    """Get an answer from Grok."""
    return await chat_with_grok(question)

async def main():
    # Example 1: Joke Generation
    print("\n=== Joke Generation Example ===")
    print("Asking Grok to tell a joke...")
    joke = await generate_joke()
    print(f"\nJoke:\n{joke}")
    
    # Example 2: Question Answering
    print("\n=== Question Answering Example ===")
    question = "What makes Grok unique compared to other AI models?"
    print(f"Question: {question}")
    answer = await answer_question(question)
    print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())