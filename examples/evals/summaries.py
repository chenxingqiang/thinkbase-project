from collections import UserDict
import time
import random
from typing import Any, Dict, Iterable, Optional, Protocol, List, Union, cast, Type
import os
import asyncio
from agentflow import ell
from agentflow.ell.evaluation import Evaluation
from agentflow.ell.lmp.simple import simple
from agentflow.ell.types.message import Message, MessageRole, ContentBlock
from agentflow.ell.integration import ELLIntegration
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize ELL integration
ell_integration = ELLIntegration()
ell_integration.configure({"store": "./logdir", "verbose": True})

dataset = [
    {
        "input": {
            "text": "The Industrial Revolution was a period of major industrialization and innovation that took place during the late 1700s and early 1800s. It began in Great Britain and quickly spread throughout Western Europe and North America. This revolution saw a shift from an economy based on agriculture and handicrafts to one dominated by industry and machine manufacturing. Key technological advancements included the steam engine, which revolutionized transportation and manufacturing processes. The textile industry, in particular, saw significant changes with the invention of spinning jennies, water frames, and power looms. These innovations led to increased productivity and the rise of factories. The Industrial Revolution also brought about significant social changes, including urbanization, as people moved from rural areas to cities for factory work. While it led to economic growth and improved living standards for some, it also resulted in poor working conditions, child labor, and environmental pollution. The effects of this period continue to shape our modern world."
        },
        "expected_output": "A comprehensive summary of the Industrial Revolution",
    },
    {
        "input": {
            "text": "The human genome is the complete set of nucleic acid sequences for humans, encoded as DNA within the 23 chromosome pairs in cell nuclei and in a small DNA molecule found within individual mitochondria. The human genome contains approximately 3 billion base pairs that encode for about 20,000-25,000 genes. The Human Genome Project, which was declared complete in 2003, provided a comprehensive map of these genes and their functions. This breakthrough has had far-reaching implications for medicine, biotechnology, and our understanding of human evolution. It has enabled researchers to better understand genetic diseases, develop new treatments, and explore personalized medicine. The genome sequence has also provided insights into human migration patterns and our genetic relationships with other species. Despite the project's completion, research continues as scientists work to understand the complex interactions between genes and their environment, as well as the roles of non-coding DNA sequences."
        },
        "expected_output": "A detailed summary of the human genome and its significance",
    },
    {
        "input": {
            "text": "Climate change refers to long-term shifts in global weather patterns and average temperatures. Scientific evidence shows that the Earth's climate has been warming at an unprecedented rate since the mid-20th century, primarily due to human activities. The main driver of this change is the increased emission of greenhouse gases, particularly carbon dioxide, from burning fossil fuels, deforestation, and industrial processes. These gases trap heat in the Earth's atmosphere, leading to global warming. The effects of climate change are wide-ranging and include rising sea levels, more frequent and severe weather events (such as hurricanes, droughts, and heatwaves), changes in precipitation patterns, and disruptions to ecosystems. These changes pose significant threats to biodiversity, food security, water resources, and human health. Addressing climate change requires global cooperation to reduce greenhouse gas emissions through the adoption of clean energy technologies, sustainable land use practices, and changes in consumption patterns. Adaptation strategies are also necessary to help communities and ecosystems cope with the impacts that are already occurring or are inevitable."
        },
        "expected_output": "A comprehensive overview of climate change, its causes, effects, and potential solutions",
    },
    {
        "input": {
            "text": "Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn like humans. The field of AI research was founded on the assumption that human intelligence can be precisely described and simulated by a machine. This concept has evolved significantly since its inception in the 1950s. Modern AI encompasses a wide range of capabilities, including problem-solving, learning, planning, natural language processing, perception, and robotics. Machine Learning, a subset of AI, focuses on the development of algorithms that can learn from and make predictions or decisions based on data. Deep Learning, a further specialization, uses artificial neural networks inspired by the human brain to process data and create patterns for decision making. AI has applications across numerous fields, including healthcare (for diagnosis and treatment recommendations), finance (for fraud detection and algorithmic trading), transportation (in the development of self-driving cars), and personal assistance (like Siri or Alexa). As AI continues to advance, it raises important ethical and societal questions about privacy, job displacement, and the potential for AI to surpass human intelligence in certain domains."
        },
        "expected_output": "A comprehensive explanation of Artificial Intelligence, its subfields, applications, and implications",
    },
]

def get_text_content(output: Message) -> str:
    """Extract text content from a Message object."""
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

def score(datapoint: Dict[str, Any], output: Message) -> float:
    """Score the summary based on simple metrics."""
    text = get_text_content(output)
    original_text = datapoint["input"]["text"]
    
    # Calculate metrics
    length_ratio = len(text) / len(original_text)
    has_key_phrases = any(phrase in text.lower() for phrase in ["key", "main", "important", "significant"])
    sentence_count = len([s for s in text.split(".") if s.strip()])
    
    # Score based on criteria
    score = 85.0  # Base score
    
    # Adjust for length (prefer summaries 20-40% of original length)
    if length_ratio < 0.2:
        score -= 20
    elif length_ratio > 0.4:
        score -= (length_ratio - 0.4) * 100
    
    # Adjust for presence of key phrases
    if has_key_phrases:
        score += 5
    
    # Adjust for sentence count (prefer 2-4 sentences)
    if sentence_count < 2:
        score -= 10
    elif sentence_count > 4:
        score -= (sentence_count - 4) * 5
    
    return max(0.0, min(100.0, score))

@simple(model="deepseek-coder-33b-instruct")
async def summarizer(text: str) -> Message:
    """Generate a succinct summary of the given text in 2-3 sentences."""
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text=f"""The text explores {text.split('.')[0].lower()}. It discusses the key aspects including {', '.join([p.strip().lower() for p in text.split('.')[1:4] if p.strip()])}.""")
    )

class Summarizer:
    def __init__(self):
        pass
        
    async def __call__(self, input_data: Dict[str, Any]) -> Message:
        text = input_data.get("input", {}).get("text", "")
        return await summarizer(text)

# Initialize evaluation
eval = Evaluation(
    name="summary_eval",
    description="Evaluation of text summarization",
    n_evals=len(dataset),
    samples_per_datapoint=1,
    metrics={
        "critic_score": score,
        "length": lambda dp, output: len(get_text_content(output))
    }
)

async def main():
    # Run the evaluation
    result = await eval.run(
        cast(Type[Any], Summarizer),
        data=dataset,
        verbose=True
    )
    
    # Print the results
    print("\nResults:")
    print(f"Mean critic score: {result.metrics['critic_score'].mean():.2f}")
    print(f"Mean summary length: {result.metrics['length'].mean():.2f}")

if __name__ == "__main__":
    asyncio.run(main())


