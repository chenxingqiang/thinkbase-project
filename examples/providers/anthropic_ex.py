""" 
Anthropic example: pip install ell-ai[anthropic]
This example demonstrates using Claude to:
1. Generate creative writing
2. Answer questions
3. Analyze text
"""
from typing import Union, List
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType, ContentBlock
import anthropic
import os

# Create Anthropic client
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

def format_response(response: Union[str, list[ContentBlock], ContentBlock]) -> str:
    """Format the response from the model into a string."""
    if isinstance(response, str):
        return response
    elif isinstance(response, list):
        return "\n".join(str(block) for block in response)
    else:
        return str(response)

async def process_conversation(system_prompt: str, user_prompt: str) -> str:
    """Process a conversation with the model."""
    # Send the messages to Claude
    response = client.messages.create(
        model="claude-3-sonnet-20240229",
        max_tokens=1000,
        temperature=0.7,
        system=system_prompt,
        messages=[
            {"role": "user", "content": user_prompt}
        ]
    )
    
    # Return the response text
    if response.content and len(response.content) > 0:
        return str(response.content[0])
    return "No response generated."

async def creative_writing(prompt: str) -> str:
    """Generate creative writing based on a prompt."""
    system_prompt = "You are a creative writer. Your task is to write engaging and descriptive stories."
    return await process_conversation(system_prompt, prompt)

async def answer_question(question: str) -> str:
    """Answer questions with detailed explanations."""
    system_prompt = "You are a knowledgeable expert. Your task is to provide clear and detailed explanations to questions."
    return await process_conversation(system_prompt, question)

async def analyze_text(text: str) -> str:
    """Analyze the themes, tone, and style of a text."""
    system_prompt = "You are a literary analyst. Your task is to analyze texts, focusing on themes, tone, style, and literary devices."
    return await process_conversation(system_prompt, text)

async def main():
    # Example 1: Creative Writing
    story_prompt = "Write a short story about a robot discovering emotions for the first time."
    print("\n=== Creative Writing Example ===")
    story = await creative_writing(story_prompt)
    print(f"Prompt: {story_prompt}")
    print(f"Response:\n{story}\n")

    # Example 2: Question Answering
    question = "Explain how black holes work and why they're important for understanding the universe."
    print("\n=== Question Answering Example ===")
    answer = await answer_question(question)
    print(f"Question: {question}")
    print(f"Answer:\n{answer}\n")

    # Example 3: Text Analysis
    text_to_analyze = """
    The old house stood silent on the hill, its windows like tired eyes gazing down at the town below. 
    Years of rain and wind had weathered its once-proud facade, but there was still dignity in its bones, 
    a quiet strength that spoke of memories and time passed.
    """
    print("\n=== Text Analysis Example ===")
    analysis = await analyze_text(text_to_analyze)
    print("Text to analyze:", text_to_analyze)
    print(f"Analysis:\n{analysis}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())



