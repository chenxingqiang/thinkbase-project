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

async def get_completion(prompt: str, system_message: str = "") -> str:
    """Get completion from OpenAI API directly."""
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        temperature=0.7
    )
    
    content = response.choices[0].message.content
    if content is None:
        return "No response generated"
    return content

@ell.with_ell(mode="simple")
async def generate_story_ideas(about: str) -> str:
    """Generate a creative story idea."""
    system_message = """You are an expert story ideator. Generate a creative and engaging story idea in a single sentence.
    The idea should be unique, specific, and emotionally resonant."""
    
    prompt = f"""Create a unique story idea about {about}.
Your response should include:
- A main character or characters
- A central conflict or challenge
- An interesting setting or situation
- A potential theme

Respond with ONLY the story idea in a single sentence."""
    
    return await get_completion(prompt, system_message)

@ell.with_ell(mode="simple")
async def write_a_draft_of_a_story(idea: str) -> str:
    """Write a story draft."""
    system_message = """You are an adept story writer. Write a concise but emotionally resonant story in exactly three paragraphs.
    Focus on vivid details, character development, and a satisfying resolution."""
    
    prompt = f"""Write a three-paragraph story based on this idea: {idea}

Structure:
1. First paragraph: Set up the character(s) and situation
2. Second paragraph: Develop the conflict and build tension
3. Third paragraph: Provide a satisfying resolution

Write ONLY the story, no additional text."""
    
    return await get_completion(prompt, system_message)

@ell.with_ell(mode="simple")
async def choose_the_best_draft(drafts: List[str]) -> str:
    """Select the best story draft."""
    system_message = """You are an expert fiction editor with a keen eye for compelling narratives.
    Select the best draft based on story structure, character development, emotional impact, and overall engagement."""
    
    drafts_text = "\n\n---\n\n".join(f"Draft {i+1}:\n{draft}" for i, draft in enumerate(drafts))
    prompt = f"""Read these story drafts and return the text of the most compelling one:

{drafts_text}

Return ONLY the text of the best draft, no explanation or commentary."""
    
    return await get_completion(prompt, system_message)

@ell.with_ell(mode="simple")
async def write_a_really_good_story(about: str) -> str:
    """Write a complete story in Hemingway's style."""
    print("\nGenerating story ideas...")
    ideas = []
    for i in range(4):
        idea = await generate_story_ideas(about)
        print(f"\nIdea {i+1}: {idea}")
        ideas.append(idea)

    print("\nWriting drafts...")
    drafts = []
    for i, idea in enumerate(ideas):
        print(f"\nWriting draft {i+1}...")
        draft = await write_a_draft_of_a_story(idea)
        print(f"Draft {i+1} completed")
        drafts.append(draft)

    print("\nSelecting best draft...")
    best_draft = await choose_the_best_draft(drafts)
    
    print("\nRevising in Hemingway's style...")
    system_message = """You are Ernest Hemingway. Write with:
    - Clear, direct prose
    - Strong imagery
    - Emotional depth through understatement
    - Authentic dialogue
    - Powerful themes"""
    
    prompt = f"""Revise this story in your distinctive style:

{best_draft}

Write ONLY the final story, no additional text."""
    
    return await get_completion(prompt, system_message)

async def main():
    # Write a story about a dog
    story = await write_a_really_good_story("a loyal dog who must choose between two loving families")
    print("\nFinal Story:")
    print("=" * 80)
    print(story)
    print("=" * 80)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
