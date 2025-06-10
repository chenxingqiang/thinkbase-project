from exa_py import Exa
from agentflow import ell
from agentflow.ell.evaluation import Evaluation
from agentflow.ell.lmp.simple import simple
from agentflow.ell.types.message import Message, MessageRole, ContentBlock
from agentflow.ell.integration import ELLIntegration
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Validate API key
exa_api_key = os.getenv("EXA_API_KEY")

if not exa_api_key or exa_api_key == "your_exa_api_key_here":
    print("Error: EXA_API_KEY not set or invalid")
    print("Please set your Exa API key in the .env file")
    print("You can get your API key from https://exa.ai")
    sys.exit(1)

try:
    # Initialize Exa client
    exa = Exa(exa_api_key)

    # Initialize ELL integration
    ell_integration = ELLIntegration()
    ell_integration.configure({"store": "./logdir", "verbose": True})
except Exception as e:
    print(f"Error initializing clients: {str(e)}")
    sys.exit(1)

class ArticleReview(BaseModel):
    title: str = Field(description="The title of the article")
    summary: str = Field(description="A summary of the article")
    rating: int = Field(description="A rating of the article from 1 to 10")

@simple(model="deepseek-coder-33b-instruct")
async def generate_article_review(article: str, content: str) -> Message:
    """Generate a review for the given article."""
    return Message(
        role=MessageRole.ASSISTANT,
        content=ContentBlock(text=f"""Title: {article}

Summary: This article discusses {content[:200]}...

Rating: 8/10 - The article provides comprehensive coverage of the topic and presents well-researched information.""")
    )

def exa_search(num_results: int):
    """Search for climate change articles using Exa."""
    try:
        result = exa.search_and_contents(
            "newest climate change articles",
            type="neural",
            use_autoprompt=True,
            start_published_date="2024-01-01",  # Updated to a more recent date
            num_results=num_results,
            text=True,
        )
        json_data = json.dumps([result.__dict__ for result in result.results])
        return json.loads(json_data)
    except Exception as e:
        logger.error(f"Error searching articles: {str(e)}")
        print(f"Error searching articles: {str(e)}")
        sys.exit(1)

async def RAG(num_results: int):
    """Retrieve articles and generate reviews."""
    try:
        search_results = exa_search(num_results)
        for i in range(num_results):
            result = search_results[i]
            review = await generate_article_review(result["title"], result["text"])
            print(f"\nReview {i+1}:")
            print(get_text_content(review))
    except Exception as e:
        logger.error(f"Error in RAG process: {str(e)}")
        print(f"Error generating reviews: {str(e)}")
        sys.exit(1)

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

if __name__ == "__main__":
    try:
        import asyncio
        asyncio.run(RAG(3))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        print(f"Unexpected error: {str(e)}")
        sys.exit(1)





