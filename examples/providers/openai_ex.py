"""
OpenAI example: This example demonstrates using OpenAI to generate stories and chat responses.
Make sure you have set the OPENAI_API_KEY environment variable.
"""
from agentflow.ell.integration import ELLIntegration
import openai
import os

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

try:
    # Get OpenAI API key
    api_key = get_env_var("OPENAI_API_KEY")
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=api_key)

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- OPENAI_API_KEY")
    exit(1)

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4-turbo-preview",
    "default_model": "gpt-4-turbo-preview",
    "client": client,
    "temperature": 0.7
})

async def write_a_story(topic: str) -> str:
    """Write a story about the given topic."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[{
                "role": "user",
                "content": f"""Write a short, creative story about {topic}. 
The story should be imaginative and entertaining, with a clear beginning, middle, and end.
Keep the story between 150-200 words."""
            }],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content or "No story generated."
    except Exception as e:
        return f"Error generating story: {str(e)}"

async def chat_with_assistant(prompt: str) -> str:
    """Have a conversation with the assistant."""
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
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

async def main():
    # Example 1: Story Generation
    print("\n=== Story Generation Example ===")
    story_topic = "a magical library where books come to life at night"
    print(f"Writing a story about: {story_topic}")
    story = await write_a_story(story_topic)
    print(f"\nStory:\n{story}")
    
    # Example 2: Chat Conversation
    print("\n=== Chat Example ===")
    chat_prompt = "What are three interesting facts about libraries?"
    print(f"Question: {chat_prompt}")
    response = await chat_with_assistant(chat_prompt)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



