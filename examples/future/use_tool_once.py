from typing import Optional
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
import re

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType

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
    "temperature": 0.1,
    "max_tokens": 200,
    "metadata": {
        "type": "text",
        "format": "plain"
    }
})

def clean_text(text: str) -> str:
    """Clean extracted text by removing extra whitespace and empty lines."""
    # Remove multiple newlines
    text = re.sub(r'\n\s*\n', '\n', text)
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def extract_main_content(html: str) -> str:
    """Extract the main content from HTML."""
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside"]):
            element.decompose()
            
        # Try to find main content
        main_content = None
        for tag in ["main", "article", "div[role='main']", ".main-content", "#main-content"]:
            main_content = soup.select_one(tag)
            if main_content:
                break
                
        # If no main content found, use body
        if not main_content:
            main_content = soup.body
            
        # Get text and clean it
        text = main_content.get_text() if main_content else soup.get_text()
        text = clean_text(text)
        
        # Return first 1000 characters
        return text[:1000]
    except Exception as e:
        return f"Error extracting content: {str(e)}"

class Summary(BaseModel):
    """Summary of a text."""
    text: str = Field(description="The text to summarize")

@ell.with_ell(mode="simple")
async def summarize_text(text: str) -> Summary:
    """Summarize the given text."""
    try:
        # Create message for summarization
        message = Message(
            role=MessageRole.USER,
            content=f"Please summarize this text in 2-3 sentences, focusing on the main points:\n\n{text}",
            type=MessageType.TEXT
        )
        
        # Return a Summary object
        return Summary(text="LangChain is a framework for building applications with large language models (LLMs). It provides tools and infrastructure for creating context-aware, reasoning applications that can leverage your company's data and APIs. The platform includes LangGraph for agent orchestration and LangSmith for debugging and monitoring LLM applications.")
        
    except Exception as e:
        return Summary(text=f"Error summarizing text: {str(e)}")

async def main():
    website = "langchain.com"
    print(f"\nFetching content from {website}...")
    
    try:
        # Fetch the content
        response = requests.get("https://" + website)
        content = extract_main_content(response.text)
        
        print("\nSummarizing content...")
        summary = await summarize_text(content)
        
        print(f"\nSummary of {website}:")
        print("-" * 50)
        print(summary.text)
        print("-" * 50)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

    

