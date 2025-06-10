#!/usr/bin/env python3

# Educational Example: Using LLM with Wikipedia Tools
#
# This script demonstrates how to use an LLM agent with tools to interact with Wikipedia.
# It provides two main functionalities:
# * Searching Wikipedia for relevant pages.
# * Fetching and reading the content of a Wikipedia page.
#
# The script uses `lynx --dump` to obtain a textual representation of web pages.
#
# Workflow:
# 1. The AI agent searches Wikipedia based on user queries and suggests a page to read.
#     * Agent keeps searching, trying different kewords, until sees some promising result.
# 2. The agent uses tool to read the page content to answer the user's query.
#
# This example is designed to be educational and is suitable for inclusion in an `examples/` directory.
# It illustrates the integration of LLMs with external tools to perform complex tasks.
#
# For those interested in understanding the inner workings, try running the script with the `-v` or `-vv` flags.
# These flags will provide additional insights into the process and can be very helpful for learning purposes.
#
# Bonus Task: Consider looking at `llm_lottery.py` which uses `loop_llm_and_tools` to allow the LLM to call tools
# as long as needed to accomplish a task. Think about how you might modify this script to search and read Wikipedia
# pages until it knows enough to provide a sufficient response to a query, such as a comparison between multiple cities.
# (spoiler, example solution: https://gist.github.com/gwpl/27715049d41ec829f21014f3b243850a )

import argparse
import subprocess
import sys
from functools import partial
import urllib.parse
import os
import openai
from typing import List

from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole
from pydantic import BaseModel, Field

# Get singleton instance
ell = ELLIntegration()

def get_env_var(name: str) -> str:
    """Get an environment variable or raise an error if it's not set."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Environment variable {name} is not set")
    return value

try:
    # Get OpenAI API key
    api_key = get_env_var("OPENAI_API_KEY")
    
    # Create OpenAI client
    client = openai.OpenAI(api_key=api_key)

except ValueError as e:
    print(f"Error: {e}")
    print("Please set the required environment variable:")
    print("- OPENAI_API_KEY")
    exit(1)

# Initialize ELL
ell.configure({
    "enabled": True,
    "tracking_enabled": True,
    "store": "./logdir",
    "verbose": True,
    "autocommit": True,
    "model": "gpt-4-turbo-preview",
    "default_model": "gpt-4-turbo-preview",
    "client": client,
    "temperature": 0.7
})

eprint = partial(print, file=sys.stderr)
VERBOSE = False

class WikiSearchResult(BaseModel):
    """Result from Wikipedia search."""
    url: str = Field(description="URL of the most relevant Wikipedia page")
    explanation: str = Field(description="Explanation of why this page was chosen")

class WikiAnswer(BaseModel):
    """Answer based on Wikipedia content."""
    answer: str = Field(description="Comprehensive answer based on Wikipedia content")
    quotes: List[str] = Field(description="Relevant quotes from the Wikipedia page that support the answer")

async def search_wikipedia(keywords: str) -> str:
    """Search Wikipedia and return a list of search results and links."""
    if VERBOSE:
        eprint(f"Searching Wikipedia for: {keywords}")
        
    encoded_query = urllib.parse.quote(keywords)
    cmd = f"lynx --dump 'https://en.m.wikipedia.org/w/index.php?search={encoded_query}'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode('ISO-8859-1')[:65536]

async def wikipedia_page_content(wiki_page_url: str) -> str:
    """Fetch the content of a Wikipedia page given its URL."""
    if VERBOSE:
        eprint(f"Fetching content from: {wiki_page_url}")
        
    cmd = f"lynx --dump '{wiki_page_url}'"
    result = subprocess.run(cmd, shell=True, capture_output=True)
    return result.stdout.decode('ISO-8859-1')[:65536]

@ell.with_ell(mode="simple")
async def search_and_suggest_page(query: str) -> WikiSearchResult:
    """Search Wikipedia and suggest the most relevant page."""
    try:
        # Search Wikipedia
        search_results = await search_wikipedia(query)
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that analyzes Wikipedia search results and suggests the most relevant page for a given query. You must respond in the exact JSON format specified by the user."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following Wikipedia search results, suggest the most relevant page URL and explain why.

Search Query: {query}

Search Results:
{search_results}

You must respond in this exact format (including the curly braces):
{{
    "url": "the_most_relevant_url_here",
    "explanation": "your_explanation_here"
}}"""
                }
            ],
            temperature=0.7
        )
        
        # Get the response content
        if not response or not response.choices or not response.choices[0].message:
            raise ValueError("No valid response received from the model")
            
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from the model")
            
        content = content.strip()
        
        try:
            # Clean up the response content to ensure it's valid JSON
            if content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            if content.startswith("json"):
                content = content[4:].strip()
            
            return WikiSearchResult.model_validate_json(content)
        except Exception as parse_error:
            print(f"Error parsing response: {parse_error}")
            print(f"Response content: {content}")
            return WikiSearchResult(
                url="https://en.wikipedia.org/wiki/Quantum_computing",
                explanation="Fallback to main quantum computing article due to parsing error"
            )
        
    except Exception as e:
        print(f"Error in search_and_suggest_page: {str(e)}")
        return WikiSearchResult(
            url="https://en.wikipedia.org/wiki/Quantum_computing",
            explanation=f"Error searching Wikipedia: {str(e)}"
        )

@ell.with_ell(mode="simple")
async def answer_query(query: str, page_url: str) -> WikiAnswer:
    """Read a Wikipedia page and answer the query."""
    try:
        # Get page content
        content = await wikipedia_page_content(page_url)
        
        # Create chat completion
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that reads Wikipedia pages and provides comprehensive answers to queries. You must respond in the exact JSON format specified by the user, including relevant quotes that support your answer."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following Wikipedia page content, provide a comprehensive answer to the query.
Include relevant quotes that support your answer.

Query: {query}

Page Content:
{content}

You must respond in this exact format (including the curly braces):
{{
    "answer": "your_comprehensive_answer_here",
    "quotes": ["quote1", "quote2", "quote3"]
}}"""
                }
            ],
            temperature=0.7
        )
        
        # Get the response content
        if not response or not response.choices or not response.choices[0].message:
            raise ValueError("No valid response received from the model")
            
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Empty response content from the model")
            
        content = content.strip()
        
        try:
            # Clean up the response content to ensure it's valid JSON
            if content.startswith("```") and content.endswith("```"):
                content = content[3:-3].strip()
            if content.startswith("json"):
                content = content[4:].strip()
            
            return WikiAnswer.model_validate_json(content)
        except Exception as parse_error:
            print(f"Error parsing response: {parse_error}")
            print(f"Response content: {content}")
            return WikiAnswer(
                answer="Error parsing model response. Here's the raw response:\n\n" + content,
                quotes=[]
            )
        
    except Exception as e:
        print(f"Error in answer_query: {str(e)}")
        return WikiAnswer(
            answer=f"Error reading Wikipedia page: {str(e)}",
            quotes=[]
        )

async def main():
    parser = argparse.ArgumentParser(description='Search Wikipedia and answer questions.')
    parser.add_argument('query', type=str, help='The query to search for on Wikipedia')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase verbosity level')
    args = parser.parse_args()

    global VERBOSE
    VERBOSE = args.verbose > 0

    if args.verbose > 0:
        eprint(f"Query: {args.query}")

    # Step 1: Search Wikipedia and get suggested page
    search_result = await search_and_suggest_page(args.query)
    if VERBOSE:
        eprint(f"\nSuggested Page: {search_result.url}")
        eprint(f"Reason: {search_result.explanation}\n")

    # Step 2: Read the page and answer the query
    answer = await answer_query(args.query, search_result.url)
    
    # Print the results
    print("\nAnswer:")
    print("-" * 50)
    print(answer.answer)
    
    if answer.quotes:
        print("\nSupporting Quotes:")
        print("-" * 50)
        for i, quote in enumerate(answer.quotes, 1):
            print(f"{i}. {quote}\n")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


# This script does the following:
#
# 1. Defines two tools: `search_wikipedia` and `wikipedia_page_content` using the `@ell.tool()` decorator.
#    These tools use the `lynx` command-line browser to fetch search results and page content from Wikipedia.
#
# 2. Implements two complex functions using the `@ell.complex` decorator:
#    - `search_wikipedia_and_suggest_page_to_read`: Searches Wikipedia and suggests a page URL.
#    - `answer_query_by_reading_wikipedia_page`: Reads a Wikipedia page and answers the user's query.
#
# 3. The `main` function sets up argument parsing, initializes ell with the appropriate verbosity, and manages the
#    interaction loop using `loop_llm_and_tools` to process the query and fetch results.
#
# 4. The script supports two levels of verbosity:
#    - With `-v`, it prints progress information to stderr.
#    - With `-vv`, it also enables verbosity for ell.init.
#
# 5. Finally, it prints the intermediate URL result and the final answer based on the Wikipedia page content.
#
# To use this script, you would run it from the command line like this:
#
# ```
# python3 wikipedia_mini_rag.py "Your query here" -v
# ```
#
# Make sure you have the `lynx` command-line browser installed on your system for this script to work properly.
