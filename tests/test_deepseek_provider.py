"""Test suite for DeepSeek provider implementation."""

import pytest
import asyncio
import json
import os
from typing import AsyncGenerator, Dict, Any
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from contextlib import asynccontextmanager

from agentflow.providers.deepseek import DeepSeekProvider, DeepSeekConfig, DeepSeekResponse

# Test timeout
API_TIMEOUT = 30  # seconds

@pytest.fixture
def mock_response() -> Dict[str, Any]:
    """Create a mock API response."""
    return {
        "id": "test-id",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-chat",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Test response"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
            "prompt_tokens_details": {"total": 10},
            "prompt_cache_hit_tokens": 0,
            "prompt_cache_miss_tokens": 10
        },
        "system_fingerprint": "test-fingerprint"
    }

@pytest.fixture
def config() -> DeepSeekConfig:
    """Create a test configuration."""
    return DeepSeekConfig(
        model_name="deepseek-chat",
        api_key="test-key",
        api_base="https://test-api.deepseek.com/v1"
    )

@pytest.fixture
async def provider(config: DeepSeekConfig) -> AsyncGenerator[DeepSeekProvider, None]:
    """Create a DeepSeek provider instance."""
    provider = DeepSeekProvider(config)
    yield provider
    await provider.close()

@asynccontextmanager
async def get_provider(provider_gen: AsyncGenerator[DeepSeekProvider, None]) -> AsyncGenerator[DeepSeekProvider, None]:
    """Helper to get provider from async generator."""
    provider = await anext(provider_gen)
    try:
        yield provider
    finally:
        try:
            await provider_gen.aclose()
        except StopAsyncIteration:
            pass

@pytest.mark.asyncio
async def test_provider_initialization(provider: AsyncGenerator[DeepSeekProvider, None]):
    """Test provider initialization."""
    async with get_provider(provider) as p:
        assert p.config.model_name == "deepseek-chat"
        assert p.config.api_key == "test-key"
        assert p._session is None
        assert p._metrics["total_tokens"] == 0
        assert p._metrics["total_cost"] == 0.0
        assert p._metrics["requests"] == 0
        assert p._metrics["errors"] == 0

@pytest.mark.asyncio
async def test_session_management(provider: AsyncGenerator[DeepSeekProvider, None]):
    """Test session initialization and cleanup."""
    async with get_provider(provider) as p:
        await p._ensure_session()
        assert p._session is not None
        assert "Authorization" in p._session._default_headers
        assert p._session._default_headers["Authorization"] == "Bearer test-key"
        
        await p.close()
        assert p._session is None

@pytest.mark.asyncio
async def test_generate_code_success(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test successful code generation."""
    mock_response["choices"][0]["message"]["content"] = """
def add_numbers(a: int, b: int) -> int:
    return a + b
"""
    
    async def mock_json():
        return mock_response
        
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 200
        mock_context.json = mock_json
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            result = await p.generate_code("Create a function to add two numbers", "python")
            assert "def add_numbers" in result
            assert p._metrics["requests"] == 1
            assert p._metrics["total_tokens"] == 30
            assert p._metrics["errors"] == 0

@pytest.mark.asyncio
async def test_generate_code_retry(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test code generation with retry."""
    mock_response["choices"][0]["message"]["content"] = "def test(): pass"
    
    async def mock_json():
        return mock_response
        
    async def mock_text():
        return "Server Error"
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        # First call fails
        mock_error = AsyncMock()
        mock_error.status = 500
        mock_error.text = mock_text
        
        # Second call succeeds
        mock_success = AsyncMock()
        mock_success.status = 200
        mock_success.json = mock_json
        
        mock_post.return_value.__aenter__.side_effect = [mock_error, mock_success]
        
        async with get_provider(provider) as p:
            result = await p.generate_code("Test function", "python")
            assert "def test()" in result
            assert p._metrics["requests"] == 2
            assert p._metrics["errors"] == 1

@pytest.mark.asyncio
async def test_analyze_requirements_success(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test successful requirements analysis."""
    analysis_result = {
        "functional_requirements": ["Input validation", "Error handling"],
        "non_functional_requirements": ["Performance", "Security"],
        "dependencies": ["None"],
        "constraints": ["Python 3.8+"],
        "risks": ["Input validation failure"],
        "complexity": "Medium"
    }
    
    mock_response["choices"][0]["message"]["content"] = json.dumps(analysis_result)
    
    async def mock_json():
        return mock_response
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 200
        mock_context.json = mock_json
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            result = await p.analyze_requirements("Create a user authentication system")
            assert result == analysis_result
            assert p._metrics["requests"] == 1

@pytest.mark.asyncio
async def test_design_tests_success(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test successful test case design."""
    test_cases = [
        {
            "test_name": "test_valid_input",
            "input_data": {"username": "test", "password": "pass123"},
            "expected_output": True,
            "test_type": "unit",
            "priority": "high"
        }
    ]
    
    mock_response["choices"][0]["message"]["content"] = json.dumps(test_cases)
    
    async def mock_json():
        return mock_response
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 200
        mock_context.json = mock_json
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            result = await p.design_tests("def authenticate(user, pwd): pass")
            assert result == test_cases
            assert p._metrics["requests"] == 1

@pytest.mark.asyncio
async def test_review_code_success(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test successful code review."""
    review_result = {
        "code_quality_issues": [],
        "security_concerns": ["Password stored in plaintext"],
        "performance_considerations": ["O(1) complexity"],
        "best_practice_violations": ["Missing docstring"],
        "suggested_improvements": ["Add input validation"],
        "style_check": "Passed"
    }
    
    mock_response["choices"][0]["message"]["content"] = json.dumps(review_result)
    
    async def mock_json():
        return mock_response
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 200
        mock_context.json = mock_json
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            result = await p.review_code("def authenticate(user, pwd): return True")
            assert result == review_result
            assert p._metrics["requests"] == 1

@pytest.mark.asyncio
async def test_error_handling(provider: AsyncGenerator[DeepSeekProvider, None]):
    """Test error handling."""
    async def mock_text():
        return "Unauthorized"
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 401
        mock_context.text = mock_text
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            with pytest.raises(Exception, match="API request failed: Unauthorized"):
                await p.generate_code("Test", "python")
            
            assert p._metrics["errors"] == 1

@pytest.mark.asyncio
async def test_metrics_tracking(provider: AsyncGenerator[DeepSeekProvider, None], mock_response: Dict[str, Any]):
    """Test metrics tracking."""
    async def mock_json():
        return mock_response
    
    with patch.object(aiohttp.ClientSession, "post") as mock_post:
        mock_context = AsyncMock()
        mock_context.status = 200
        mock_context.json = mock_json
        mock_post.return_value.__aenter__.return_value = mock_context
        
        async with get_provider(provider) as p:
            await p.generate_code("Test", "python")
            metrics = p.get_metrics()
            
            assert metrics["total_tokens"] == 30
            assert metrics["requests"] == 1
            assert metrics["errors"] == 0
            assert isinstance(metrics["total_cost"], float) 