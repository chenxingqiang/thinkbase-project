"""
OpenRouter example: This example demonstrates using OpenRouter to access various LLM models.
Make sure you have set the OPENROUTER_API_KEY environment variable.
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
    # Get OpenRouter API key
    api_key = get_env_var("OPENROUTER_API_KEY")
    
    # Create OpenRouter client using OpenAI client
    client = openai.OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- OPENROUTER_API_KEY")
    exit(1)

# OpenRouter-specific request parameters
extra_body = {
    "provider": {
        "allow_fallbacks": True,
        "data_collection": "deny",
        "order": ["Anthropic", "Google", "Meta"],
        "quantizations": ["bf16", "fp8"]
    }
}

async def generate_story(topic: str) -> str:
    """Generate a story using a specific model through OpenRouter."""
    try:
        response = client.chat.completions.create(
            model="anthropic/claude-3-opus",  # You can change this to other available models
            messages=[{
                "role": "user",
                "content": f"""Write a short, creative story about {topic}. 
The story should be imaginative and entertaining, with a clear beginning, middle, and end.
Keep the story between 150-200 words."""
            }],
            temperature=0.7,
            max_tokens=300,
            extra_body=extra_body
        )
        return response.choices[0].message.content or "No story generated."
    except Exception as e:
        return f"Error generating story: {str(e)}"

async def chat_with_model(prompt: str, model: str = "meta-llama/llama-2-70b-chat") -> str:
    """Chat with a specific model through OpenRouter."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            temperature=0.7,
            max_tokens=300,
            extra_body=extra_body
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"Error in conversation: {str(e)}"

async def main():
    # Example 1: Story Generation with Claude
    print("\n=== Story Generation Example (Using Claude) ===")
    story_topic = "a time-traveling archaeologist who discovers something unexpected"
    print(f"Writing a story about: {story_topic}")
    story = await generate_story(story_topic)
    print(f"\nStory:\n{story}")
    
    # Example 2: Chat with Llama
    print("\n=== Chat Example (Using Llama) ===")
    chat_prompt = "What are three fascinating discoveries in archaeology?"
    print(f"Question: {chat_prompt}")
    response = await chat_with_model(chat_prompt)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())