"""
RAG (Retrieval-Augmented Generation) example: This example demonstrates using RAG with ELL.
Make sure you have the required packages installed:
pip install scikit-learn openai
"""
from agentflow.ell.integration import ELLIntegration
from agentflow.ell.types.message import Message, MessageRole, MessageType
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
import os

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

class VectorStore:
    """Simple vector store using TF-IDF and cosine similarity."""
    def __init__(self, vectorizer, tfidf_matrix, documents):
        self.vectorizer = vectorizer
        self.tfidf_matrix = tfidf_matrix
        self.documents = documents

    @classmethod
    def from_documents(cls, documents):
        # Preprocess documents to handle variations in terminology
        processed_docs = [doc.lower() for doc in documents]
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            max_features=1000
        )
        tfidf_matrix = vectorizer.fit_transform(processed_docs)
        return cls(vectorizer, tfidf_matrix, documents)

    def search(self, query: str, k: int = 3) -> list[dict]:
        """Search for the k most relevant documents."""
        # Preprocess query to match document processing
        query = query.lower()
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get top k results with similarity above threshold
        threshold = 0.01
        indices = np.where(similarities > threshold)[0]
        # Convert numpy array values to Python floats for sorting
        top_k_indices = sorted(indices, key=lambda i: float(similarities[i]), reverse=True)[:k]
        
        return [
            {
                "document": self.documents[i],
                "relevance": float(similarities[i]),
                "index": i
            }
            for i in top_k_indices
        ]

async def rag_query(query: str, context: list[dict], temperature: float = 0.7) -> str:
    """Process a query using RAG."""
    try:
        # Format context for the prompt
        context_text = "\n".join(
            f"{i+1}. {doc['document']} (relevance: {doc['relevance']:.2f})" 
            for i, doc in enumerate(context)
        )
        
        # Create the chat completion
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {
                    "role": "system",
                    "content": """You are an AI assistant using Retrieval-Augmented Generation (RAG).
You will be provided with a query and relevant context from a knowledge base.
Use this context to inform your response, but also draw upon your general knowledge when appropriate.
Always strive to provide accurate, helpful, and context-aware answers.
When discussing features or capabilities, try to organize them in a clear, structured way."""
                },
                {
                    "role": "user",
                    "content": f"""Query: {query}

Relevant context (ordered by relevance):
{context_text}

Please provide a comprehensive and accurate response, focusing on information from the provided context."""
                }
            ],
            temperature=temperature,
            max_tokens=500
        )
        return response.choices[0].message.content or "No response generated."
    except Exception as e:
        return f"Error processing query: {str(e)}"

async def main():
    # Example documents (knowledge base)
    documents = [
        "ELL is a powerful framework for building AI applications, created by Will.",
        "Will develops ELL while actively sharing updates on X.com.",
        "ELL aims to be the go-to tool for AI development and automation.",
        "ELL supports multiple LLM providers including OpenAI, Anthropic, and local models.",
        "The framework includes built-in RAG capabilities for enhanced context awareness.",
        "ELL features easy integration with various vector stores and databases.",
        "The framework is designed to be easy to use while maintaining flexibility.",
        "ELL provides robust error handling and logging capabilities.",
        "The framework supports async/await for better performance.",
        "Joe Biden is the current President of the United States, elected in 2020.",
        "The United States has a system of checks and balances between branches.",
        "The US President serves as both head of state and head of government."
    ]
    
    # Create vector store
    print("\nInitializing vector store...")
    vector_store = VectorStore.from_documents(documents)
    
    # Example queries
    queries = [
        "Tell me about ELL and who created it.",
        "Who is the current US President?",
        "What are the main features and capabilities of ELL?"
    ]
    
    # Process each query
    for i, query in enumerate(queries, 1):
        print(f"\n=== Query {i} ===")
        print(f"Question: {query}")
        
        # Get relevant context
        context = vector_store.search(query, k=5)  # Increased k for more context
        
        # Adjust temperature based on query type
        temperature = 0.3 if "features" in query.lower() or "capabilities" in query.lower() else 0.7
        
        # Get response
        response = await rag_query(query, context, temperature)
        print(f"\nResponse:\n{response}")
        print("\n" + "="*50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
