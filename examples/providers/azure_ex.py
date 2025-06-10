"""
Azure OpenAI example: pip install ell-ai[azure]
This example demonstrates using Azure OpenAI to generate stories.
"""
from agentflow.ell.integration import ELLIntegration
import openai
import os
from typing import Optional

# Get singleton instance
ell = ELLIntegration()

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",  # Default model
    "default_model": "gpt-4"
})

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

# Your Azure OpenAI credentials
try:
    subscription_key = get_env_var("AZURE_OPENAI_API_KEY")
    azure_endpoint = get_env_var("AZURE_OPENAI_ENDPOINT")
    deployment_name = get_env_var("AZURE_OPENAI_DEPLOYMENT")

    # Create Azure OpenAI client
    azure_client = openai.AzureOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=subscription_key,
        api_version="2024-02-15-preview"
    )

    # Configure ELL to use the Azure model
    ell.configure({
        "model": deployment_name,
        "default_model": deployment_name,
        "client": azure_client
    })

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variables:")
    print("- AZURE_OPENAI_API_KEY")
    print("- AZURE_OPENAI_ENDPOINT")
    print("- AZURE_OPENAI_DEPLOYMENT")
    exit(1)

@ell.with_ell(mode="simple")
async def write_a_story(about: str) -> str:
    """Write a story about the given topic."""
    return f"Write me a creative and engaging story about {about}!"

async def main():
    # Example: Write a story about cats
    print("\n=== Story Generation Example ===")
    story_topic = "cats"
    print(f"Writing a story about: {story_topic}")
    story = await write_a_story(story_topic)
    print(f"\nStory:\n{story}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
