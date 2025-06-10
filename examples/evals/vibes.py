from typing import Any, Dict, List, Union, cast, Type
import os
import asyncio
from agentflow import ell
from agentflow.ell.evaluation import Evaluation
from agentflow.ell.lmp.simple import simple
from agentflow.ell.types.message import Message, MessageRole, ContentBlock
from agentflow.ell.integration import ELLIntegration
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize ELL integration
ell_integration = ELLIntegration()
ell_integration.configure({"store": "./logdir", "verbose": True})

class TweetInput(BaseModel):
    input: str

@simple(model="deepseek-coder-33b-instruct")
async def tweet(text: str) -> Message:
    """Generate a tweet in roon's style about the given topic."""
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text=f"""just thinking about {text.lower()}... absolutely wild how it changes everything. the implications are staggering. need to meditate on this.""")
    )

class TweetGenerator:
    def __init__(self):
        pass
        
    async def __call__(self, input_data: Dict[str, Any]) -> Message:
        topic = input_data.get("input", "")
        if isinstance(topic, list) and len(topic) > 0 and isinstance(topic[0], dict):
            topic = topic[0].get("input", "")
        if not isinstance(topic, str):
            topic = str(topic)
        return await tweet(topic)

dataset = [
    {"input": [{"input": "Polymath"}]},
    {"input": [{"input": "Dogs"}]},
    {"input": [{"input": "Intelligence"}]},
]

def has_roon_vibes(output: Message) -> bool:
    """Check if the output has roon's style."""
    if not output.content:
        return False
    if isinstance(output.content, ContentBlock):
        text = output.content.text
    else:
        text = str(output.content)
    if not text:
        return False
    text = text.lower()
    return any(phrase in text for phrase in [
        "wild", "staggering", "implications", "meditate", 
        "thinking about", "absolutely", "need to"
    ])

# Initialize evaluation
eval = Evaluation(
    name="vibes_eval",
    description="Evaluation of roon-style tweet generation",
    n_evals=len(dataset),
    samples_per_datapoint=1,
    metrics={
        "has_vibes": lambda dp, output: float(has_roon_vibes(output)),
        "length": lambda dp, output: len(str(output.content.text if isinstance(output.content, ContentBlock) else output.content))
    }
)

async def main():
    # Run the evaluation
    result = await eval.run(
        cast(Type[Any], TweetGenerator),
        data=dataset,
        verbose=True
    )
    
    # Print the results
    print("\nResults:")
    print(f"Mean vibes score: {result.metrics['has_vibes'].mean():.2f}")
    print(f"Mean tweet length: {result.metrics['length'].mean():.2f}")

if __name__ == "__main__":
    asyncio.run(main())
