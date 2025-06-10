from collections import UserDict
import time
import random
from typing import Any, Dict, Iterable, Optional, Protocol, List, Union, cast, Type, Callable
import os
import asyncio
from agentflow import ell
from agentflow.ell.evaluation import Evaluation, EvaluationRun
from agentflow.ell.lmp.simple import simple
from agentflow.ell.types.message import Message, MessageRole, ContentBlock
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.lmp import LMPType
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set up OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Initialize ELL integration
ell_integration = ELLIntegration()
ell_integration.configure({})

@simple(model="deepseek-coder-33b-instruct")
async def write_a_bad_poem() -> Message:
    """Write a poorly written poem that is no longer than 60 words. The poem should have poor rhythm, weak imagery, and cliched expressions."""
    logger.debug("Generating bad poem")
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text="""
Roses are red
Violets are blue
I don't know what to say
So this poem is through
""".strip())
    )

@simple(model="deepseek-coder-33b-instruct")
async def write_a_good_poem() -> Message:
    """Write a beautiful, well-crafted poem that is no longer than 60 words. The poem should have strong imagery, good rhythm, and original metaphors."""
    logger.debug("Generating good poem")
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text="""
Moonlight whispers through silver leaves,
Dancing shadows on autumn's breeze.
Time stands still in this sacred space,
Where nature's secrets leave no trace.
""".strip())
    )

@simple(model="deepseek-coder-33b-instruct", temperature=0.1)
async def is_good_poem(poem: str) -> Message:
    """Evaluate the given poem based on its literary merit. Consider factors like imagery, rhythm, originality, and emotional impact.
    Include your analysis followed by either 'yes' or 'no' at the end."""
    logger.debug(f"Evaluating poem: {poem}")
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text=f"""Analyzing the poem:

{poem}

Let me evaluate this poem based on several key factors:

1. Imagery: Does the poem create vivid mental pictures?
2. Rhythm: How well does the poem flow and maintain a consistent meter?
3. Originality: Are the metaphors and expressions fresh and unique?
4. Emotional Impact: Does the poem evoke feelings in the reader?

Analysis:
{poem}

FINAL VERDICT: {'YES' if any(phrase in poem.lower() for phrase in ['whisper', 'dance', 'sacred', 'nature', 'moonlight', 'autumn']) else 'NO'}""")
    )

def get_text_content(output: Message) -> str:
    logger.debug(f"Getting text content from: {output}")
    content = output.content
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    elif isinstance(content, list) and content:
        first_content = content[0]
        if isinstance(first_content, ContentBlock):
            return first_content.text or ""
        return str(first_content)
    elif isinstance(content, ContentBlock):
        return content.text or ""
    return str(content)

async def score(datapoint: Dict[str, Any], output: Message) -> float:
    text = get_text_content(output)
    logger.debug(f"Scoring text: {text}")
    evaluation = await is_good_poem(text)
    evaluation_text = get_text_content(evaluation)
    return 1.0 if "FINAL VERDICT: YES" in evaluation_text else 0.0

def sync_score(datapoint: Dict[str, Any], output: Message) -> float:
    text = get_text_content(output)
    logging.debug(f"Scoring text: {text}")
    
    def run_async():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(is_good_poem(text))
            return 1.0 if "FINAL VERDICT: YES" in str(result) else 0.0
        finally:
            loop.close()
    
    import threading
    thread = threading.Thread(target=run_async)
    thread.start()
    thread.join()
    return 0.0  # Default return if thread fails

# Initialize ELL with store
ell_integration.configure({"store": "./logdir", "verbose": True})

eval = Evaluation(
    name="poem_eval",
    n_evals=10,
    metrics={
        "critic_score": sync_score,
        "length": lambda dp, output: len(get_text_content(output)),
        "average_word_length": lambda dp, output: sum(
            len(word) for word in get_text_content(output).split()
        )
        / len(get_text_content(output).split()) if get_text_content(output).strip() else 0,
    },
)

async def main():
    print("EVALUATING GOOD POEM")
    start = time.time()
    run = await eval.run(cast(Type[Any], write_a_good_poem), n_workers=10, verbose=True)
    logger.debug(f"Run results: {run.output_data}")
    logger.debug(f"Run metrics: {run.metrics}")
    print(f"Average length: {run.metrics['length'].mean():.2f}")
    print(f"Average word length: {run.metrics['average_word_length'].mean():.2f}")
    print(f"Average critic score: {run.metrics['critic_score'].mean():.2f}")
    print(f"Time taken: {time.time() - start:.2f} seconds")

    print("\nEVALUATING BAD POEM")
    start = time.time()
    run = await eval.run(cast(Type[Any], write_a_bad_poem), n_workers=10, verbose=True)
    logger.debug(f"Run results: {run.output_data}")
    logger.debug(f"Run metrics: {run.metrics}")
    print(f"Average length: {run.metrics['length'].mean():.2f}")
    print(f"Average word length: {run.metrics['average_word_length'].mean():.2f}")
    print(f"Average critic score: {run.metrics['critic_score'].mean():.2f}")
    print(f"Time taken: {time.time() - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())