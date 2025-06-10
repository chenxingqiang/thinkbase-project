"""Test DeepSeek provider integration."""

import logging
import os
import pytest
from requests.exceptions import RequestException
from agentflow.ell.providers.deepseek import DeepSeekProvider, DeepSeekConfig

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Retrieve API key with more robust method
def get_deepseek_api_key():
    """Retrieve DeepSeek API key with multiple fallback methods."""
    # Check environment variables
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    # If no API key found, check for a local configuration file
    if not api_key:
        try:
            config_path = os.path.expanduser("~/.config/deepseek/api_key")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    api_key = f.read().strip()
        except Exception as e:
            logger.warning(f"Could not read API key from config file: {e}")
    
    return api_key

# Validate API key
def validate_api_key(api_key):
    """Check if the API key appears to be valid."""
    return api_key and len(api_key) > 10 and api_key.startswith('sk-')

@pytest.fixture(scope="module")
def deepseek_api_key():
    """Fixture to provide API key with validation."""
    api_key = get_deepseek_api_key()
    
    if not validate_api_key(api_key):
        pytest.skip("No valid DeepSeek API key found")
    
    return api_key

@pytest.fixture
def chat_provider(deepseek_api_key):
    """Create a DeepSeek-V3 provider instance."""
    config = {
        "api_key": deepseek_api_key,
        "model": "deepseek-chat-v1",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    return DeepSeekProvider(config=config)

@pytest.fixture
def reasoner_provider(deepseek_api_key):
    """Create a DeepSeek-R1 provider instance."""
    config = {
        "api_key": deepseek_api_key,
        "model": "deepseek-reasoner-v1",
        "temperature": 0.7,
        "max_tokens": 1000
    }
    return DeepSeekProvider(config=config)

@pytest.mark.asyncio
async def test_provider_initialization(chat_provider):
    """Test provider initialization."""
    try:
        await chat_provider.initialize()
        assert chat_provider._session is not None
        assert "Authorization" in chat_provider.session.headers
        assert "Content-Type" in chat_provider.session.headers
    except Exception as e:
        logger.error(f"Provider initialization failed: {e}")
        raise
    finally:
        await chat_provider.cleanup()

@pytest.mark.asyncio
async def test_chat_completion_v3(chat_provider):
    """Test chat completion with DeepSeek-V3."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    
    try:
        response = await chat_provider.chat(messages)
        assert isinstance(response, str)
        assert len(response) > 0
        # The response may contain markdown formatting
        assert "paris" in response.lower().replace("*", "")
    except RequestException as e:
        logger.error(f"Network error in chat completion: {e}")
        pytest.fail(f"Network error in chat completion: {e}")
    except ValueError as e:
        logger.error(f"API error in chat completion: {e}")
        pytest.fail(f"API error in chat completion: {e}")

@pytest.mark.asyncio
async def test_chat_completion_r1(reasoner_provider):
    """Test chat completion with DeepSeek-R1."""
    messages = [
        {"role": "system", "content": "You are a logical reasoning assistant."},
        {"role": "user", "content": "If all cats are mammals, and all mammals are animals, what can we conclude about cats?"}
    ]
    
    try:
        response = await reasoner_provider.chat(messages)
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for logical reasoning in response
        assert any(word in response.lower() for word in ["conclude", "therefore", "logic", "animal"])
    except RequestException as e:
        logger.error(f"Network error in R1 chat completion: {e}")
        pytest.fail(f"Network error in R1 chat completion: {e}")
    except ValueError as e:
        logger.error(f"API error in R1 chat completion: {e}")
        pytest.fail(f"API error in R1 chat completion: {e}")

@pytest.mark.asyncio
async def test_text_generation_v3(chat_provider):
    """Test text generation with DeepSeek-V3."""
    prompt = "Write a short poem about artificial intelligence."
    
    try:
        response = await chat_provider.generate(prompt, use_chat_endpoint=True)
        assert isinstance(response, str)
        assert len(response) > 0
        # Check if the response contains any AI-related keywords
        ai_keywords = ["ai", "artificial", "intelligence", "machine", "neural", "digital", "code", "learn"]
        assert any(keyword in response.lower() for keyword in ai_keywords), "Response should contain AI-related keywords"
    except RequestException as e:
        logger.error(f"Network error in text generation V3: {e}")
        pytest.fail(f"Network error in text generation V3: {e}")
    except ValueError as e:
        logger.error(f"API error in text generation V3: {e}")
        pytest.fail(f"API error in text generation V3: {e}")

@pytest.mark.asyncio
async def test_text_generation_r1(reasoner_provider):
    """Test text generation with DeepSeek-R1."""
    prompt = "Explain the logical steps to solve this problem: If A implies B, and B implies C, what can we say about A and C?"
    
    try:
        response = await reasoner_provider.generate(prompt, use_chat_endpoint=True)
        assert isinstance(response, str)
        assert len(response) > 0
        # Check for logical reasoning terms
        reasoning_keywords = ["therefore", "implies", "conclusion", "logic", "transitive"]
        assert any(keyword in response.lower() for keyword in reasoning_keywords), "Response should contain reasoning keywords"
    except RequestException as e:
        logger.error(f"Network error in text generation R1: {e}")
        pytest.fail(f"Network error in text generation R1: {e}")
    except ValueError as e:
        logger.error(f"API error in text generation R1: {e}")
        pytest.fail(f"API error in text generation R1: {e}")

@pytest.mark.asyncio
@pytest.mark.skip(reason="Embeddings API not available")
async def test_embeddings(chat_provider):
    """Test embeddings generation."""
    text = "This is a test sentence for embeddings."
    
    try:
        embeddings = await chat_provider.embed(text)
        assert isinstance(embeddings, list)
        assert len(embeddings) > 0
        assert all(isinstance(x, float) for x in embeddings)
    except RequestException as e:
        logger.error(f"Network error in embeddings generation: {e}")
        pytest.fail(f"Network error in embeddings generation: {e}")
    except ValueError as e:
        logger.error(f"API error in embeddings generation: {e}")
        pytest.fail(f"API error in embeddings generation: {e}")

@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling with invalid API key."""
    invalid_config = {
        "api_key": "invalid-key",
        "model": "deepseek-chat-v1"
    }
    
    with pytest.raises((ValueError, RequestException), 
                       match=r"(Unauthorized|Invalid API key|Authentication failed)"):
        invalid_provider = DeepSeekProvider(config=invalid_config)
        await invalid_provider.chat([{"role": "user", "content": "Hello"}])

@pytest.mark.asyncio
async def test_provider_cleanup(chat_provider):
    """Test provider cleanup."""
    # Explicitly initialize the session
    await chat_provider.initialize()
    
    # Verify session is not None before cleanup
    assert chat_provider._session is not None
    
    # Perform cleanup
    await chat_provider.cleanup()
    
    # Verify session is None after cleanup
    assert chat_provider._session is None