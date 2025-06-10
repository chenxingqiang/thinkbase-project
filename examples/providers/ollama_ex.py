"""
Ollama example: This example demonstrates using Ollama to generate stories.
Make sure you have Ollama running locally (http://localhost:11434).
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
    "autocommit": True
})

# Create Ollama client
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # required but not used
)

def write_a_story(topic: str) -> str:
    """Write a story about the given topic."""
    try:
        response = client.chat.completions.create(
            model="llama2",  # or any other model you have pulled
            messages=[{
                "role": "user",
                "content": f"""Write a short, creative story about {topic}. 
The story should be imaginative and entertaining, with a clear beginning, middle, and end.
Keep the story between 150-200 words."""
            }],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content or "No story was generated."
    except Exception as e:
        return f"Error generating story: {str(e)}"

async def main():
    # Example topics
    topics = [
        "a magical forest",
        "a time-traveling scientist",
        "a friendly robot"
    ]
    
    print("\n=== Story Generation Example ===")
    for i, topic in enumerate(topics, 1):
        print(f"\nStory {i}:")
        print(f"Topic: {topic}")
        story = write_a_story(topic)
        print(f"\nStory:\n{story}")
        print("\n" + "="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

