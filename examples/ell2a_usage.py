"""
Example usage of ELL (Extensible Language Layer)
"""

import os
from typing import List, Dict, Any
import asyncio
from PIL import Image
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS

from agentflow.ell_integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole

# Get ELL integration instance
ell = ELLIntegration()
# Initialize ELL with API key and versioning
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Please set OPENAI_API_KEY environment variable")

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

# Initialize ELL with configuration
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./ell_logs",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4",
    "default_model": "gpt-4",
    "temperature": 0.1,
    "mode": "simple",
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

class SearchResult(BaseModel):
    """Model for search results"""
    query: str
    results: List[Dict[str, str]]

@ell.with_ell(mode="simple")
async def web_search(query: str) -> SearchResult:
    """Real web search using DuckDuckGo"""
    try:
        ddgs = DDGS()
        # Use text search with minimal parameters
        results = []
        for r in ddgs.text(query, max_results=5):
            results.append(r)
            
        # Format results as list of dicts with title and snippet
        formatted_results = [
            {
                "title": result.get("title", "No title"),
                "snippet": result.get("body", "No content available"),
                "link": result.get("link", "")
            }
            for result in results
        ]
        
        return SearchResult(
            query=query,
            results=formatted_results if formatted_results else [
                {
                    "title": "No Results",
                    "snippet": "No search results found. Please try a different query.",
                    "link": ""
                }
            ]
        )
    except Exception as e:
        # Return empty results on error with more detailed error message
        error_msg = f"Search failed: {str(e)}. This might be due to rate limiting or network issues. Please try again in a few moments."
        return SearchResult(
            query=query,
            results=[{"title": "Error", "snippet": error_msg, "link": ""}]
        )

# Simple prompt example
@ell.with_ell(mode="simple")
async def summarize(text: str) -> str:
    """You are a helpful assistant that summarizes text concisely."""
    return f"Please summarize this text in one sentence: {text}"

# Complex prompt example with tool usage
@ell.with_ell(mode="complex")
async def research_topic(topic: str) -> List[Message]:
    """Research a given topic using web search."""
    # First, search for information
    search_results = await web_search(topic)
    
    # Format search results for better readability
    formatted_results = "\n\n".join([
        f"Title: {result['title']}\nSummary: {result['snippet']}\nSource: {result['link']}"
        for result in search_results.results
    ])
    
    messages = [
        Message(
            role=MessageRole.SYSTEM, 
            content="You are a research assistant that provides comprehensive analysis by combining search results with your knowledge. Format your response with clear sections and cite sources when using search results."
        ),
        Message(
            role=MessageRole.USER, 
            content=f"Research this topic: {topic}\n\nSearch Results:\n{formatted_results}\n\nProvide a comprehensive analysis combining these search results with your knowledge. Include citations when referencing search results."
        )
    ]
    return messages

# Multimodal example
@ell.with_ell(mode="complex")
async def analyze_image(image: Image.Image) -> List[Message]:
    """Analyze the given image."""
    messages = [
        Message(
            role=MessageRole.SYSTEM, 
            content="You are a computer vision expert. Analyze the image and describe what you see."
        ),
        Message(
            role=MessageRole.USER, 
            content=["Describe this image in detail:", image]
        )
    ]
    return messages

async def main():
    try:
        # Test simple prompt
        print("\nTesting simple prompt...")
        summary = await summarize(
            "The quick brown fox jumps over the lazy dog. "
            "This pangram contains every letter of the English alphabet."
        )
        print("Summary:", summary)
        
        # Test complex prompt with tool
        print("\nTesting complex prompt with tool...")
        research = await research_topic("artificial intelligence trends 2024")
        print("Research Results:", research)
        
        # Test multimodal (commented out as it requires an image)
        # print("\nTesting multimodal...")
        # image = Image.open("example.jpg")
        # analysis = await analyze_image(image)
        # print("Image Analysis:", analysis)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting ELL examples...")
    asyncio.run(main()) 