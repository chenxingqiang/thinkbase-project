"""
vLLM example: This example demonstrates using vLLM to run local language models.
Make sure you have vLLM installed and a model running locally:
pip install vllm
vllm serve --model meta-llama/Llama-2-7b-chat-hf --port 8000
"""
from agentflow.ell.integration import ELLIntegration
import openai
import os

# Create vLLM client using OpenAI client
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",  # vLLM server URL
    api_key="not-needed"  # vLLM doesn't need an API key
)

async def generate_story(topic: str) -> str:
    """Generate a story using a local model through vLLM."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",  # The model you're serving with vLLM
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

async def chat_with_model(prompt: str) -> str:
    """Chat with the local model through vLLM."""
    try:
        response = client.chat.completions.create(
            model="meta-llama/Llama-2-7b-chat-hf",  # The model you're serving with vLLM
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
    story_topic = "a scientist who invents a device that can talk to plants"
    print(f"Writing a story about: {story_topic}")
    story = await generate_story(story_topic)
    print(f"\nStory:\n{story}")
    
    # Example 2: Chat Conversation
    print("\n=== Chat Example ===")
    chat_prompt = "What are three interesting facts about plant communication?"
    print(f"Question: {chat_prompt}")
    response = await chat_with_model(chat_prompt)
    print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())