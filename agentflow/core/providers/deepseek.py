"""DeepSeek model provider."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import asyncio
import json
import logging
from pydantic import BaseModel, Field
import aiohttp

logger = logging.getLogger(__name__)

@dataclass
class DeepSeekConfig:
    """Configuration for DeepSeek provider."""
    model_name: str = "deepseek-chat"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.95
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: List[str] = field(default_factory=list)
    api_key: Optional[str] = None
    api_base: str = "https://api.deepseek.com/v1"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DeepSeekConfig":
        """Build a config from a plain dictionary.

        Accepts both the ``model_name`` spelling used by the dataclass and
        the ``model`` spelling used by the DeepSeek API payloads, so callers
        can pass raw configuration dictionaries straight through.
        """
        data = dict(data or {})
        if "model_name" not in data and "model" in data:
            data["model_name"] = data.pop("model")
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})
    
class DeepSeekMessage(BaseModel):
    """Message in DeepSeek API response."""
    role: str
    content: str
    
class DeepSeekChoice(BaseModel):
    """Choice in DeepSeek API response."""
    index: int
    message: DeepSeekMessage
    finish_reason: str
    
class DeepSeekUsage(BaseModel):
    """Usage information from DeepSeek API."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Dict[str, int]
    prompt_cache_hit_tokens: int
    prompt_cache_miss_tokens: int
    
class DeepSeekResponse(BaseModel):
    """Response from DeepSeek API."""
    id: str
    object: str
    created: int
    model: str
    choices: List[DeepSeekChoice]
    usage: DeepSeekUsage
    system_fingerprint: str
    
    @property
    def text(self) -> str:
        """Get the response text."""
        return self.choices[0].message.content
    
class DeepSeekProvider:
    """Provider for interacting with DeepSeek models."""
    
    def __init__(self, config: Optional[Union[DeepSeekConfig, Dict[str, Any]]] = None):
        """Initialize DeepSeek provider.
        
        Args:
            config: Optional configuration for the provider. May be a
                ``DeepSeekConfig`` instance or a plain dictionary (the
                ``model`` key is accepted as an alias of ``model_name``).
        """
        if isinstance(config, dict):
            config = DeepSeekConfig.from_dict(config)
        self.config = config or DeepSeekConfig()
        self._session: Optional[aiohttp.ClientSession] = None
        self._metrics = {
            "total_tokens": 0,
            "total_cost": 0.0,
            "requests": 0,
            "errors": 0
        }
        # Async-iterator bookkeeping: the provider yields itself exactly once
        # so that callers which treat it as an async generator (fixture-style
        # ``anext(provider)`` / ``await provider.aclose()`` usage) work whether
        # or not the surrounding harness already unwrapped the generator.
        self._aiter_exhausted = False

    # ------------------------------------------------------------------
    # Async generator compatibility
    # ------------------------------------------------------------------
    def __aiter__(self) -> "DeepSeekProvider":
        """Return self as an async iterator that yields the provider once."""
        return self

    async def __anext__(self) -> "DeepSeekProvider":
        """Yield the provider itself once, then signal exhaustion."""
        if self._aiter_exhausted:
            raise StopAsyncIteration
        self._aiter_exhausted = True
        return self

    async def aclose(self) -> None:
        """Close the provider session (async-generator style teardown)."""
        await self.close()

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------
    @property
    def session(self) -> Optional[aiohttp.ClientSession]:
        """The underlying aiohttp session, if initialized."""
        return self._session

    async def initialize(self) -> None:
        """Explicitly initialize the HTTP session."""
        await self._ensure_session()

    async def cleanup(self) -> None:
        """Tear down the HTTP session."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session is initialized."""
        if self._session is None:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.config.api_key}"}
            )
            
    async def close(self) -> None:
        """Close the provider session."""
        if self._session:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # API key validation
    # ------------------------------------------------------------------
    def _validate_api_key(self) -> None:
        """Validate that the configured API key looks like a DeepSeek key.

        Raises:
            ValueError: If the key is missing or malformed.
        """
        api_key = self.config.api_key
        if not api_key or not str(api_key).strip():
            raise ValueError("Invalid API key: no API key configured")
        if not str(api_key).startswith("sk-"):
            raise ValueError(
                "Invalid API key: expected a DeepSeek key starting with 'sk-'"
            )

            
    async def _make_request(self,
                            prompt: Optional[str] = None,
                            messages: Optional[List[Dict[str, str]]] = None,
                            is_retry: bool = False) -> DeepSeekResponse:
        """Make a request to the DeepSeek API.
        
        Args:
            prompt: The prompt to send to the model (wrapped as a single
                user message when ``messages`` is not given).
            messages: Full chat message list to send. Takes precedence over
                ``prompt``.
            is_retry: Whether this is a retry attempt.
            
        Returns:
            DeepSeekResponse with the model's response.
            
        Raises:
            Exception if the API request fails.
        """
        await self._ensure_session()
        
        if self._session is None:
            self._metrics["errors"] += 1
            raise RuntimeError("Failed to initialize session")
        
        if messages is None:
            messages = [{"role": "user", "content": prompt or ""}]
        
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stop": self.config.stop_sequences
        }
        
        # Increment requests counter before making the call
        self._metrics["requests"] += 1
        
        try:
            async with self._session.post(
                f"{self.config.api_base}/chat/completions",
                json=payload
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    if not is_retry:
                        self._metrics["errors"] += 1
                    raise Exception(f"API request failed: {error_text}")
                    
                data = await response.json()
                self._metrics["total_tokens"] += data["usage"]["total_tokens"]
                
                return DeepSeekResponse(**data)
                
        except Exception as e:
            if not str(e).startswith("API request failed:") and not is_retry:
                self._metrics["errors"] += 1
            logger.error(f"Error making DeepSeek API request: {str(e)}")
            raise
            
    async def chat(self, messages: List[Dict[str, str]]) -> str:
        """Run a chat completion against the DeepSeek chat endpoint.

        Args:
            messages: Chat messages, e.g.
                ``[{"role": "user", "content": "Hello"}]``.

        Returns:
            The assistant reply text.

        Raises:
            ValueError: If the API key is missing/malformed or the API
                rejects the credentials.
        """
        self._validate_api_key()
        try:
            response = await self._make_request(messages=messages)
        except Exception as e:
            message = str(e)
            if "Unauthorized" in message or "Authentication" in message:
                raise ValueError(f"Authentication failed: {message}") from e
            raise
        return response.text

    async def generate(self, prompt: str, use_chat_endpoint: bool = True) -> str:
        """Generate text for a single prompt.

        Args:
            prompt: The prompt to send.
            use_chat_endpoint: Kept for API compatibility; DeepSeek only
                exposes the chat completions endpoint, so both values use it.

        Returns:
            The generated text.

        Raises:
            ValueError: If the API key is missing/malformed or the request
                fails.
        """
        self._validate_api_key()
        try:
            response = await self._make_request(prompt)
        except Exception as e:
            message = str(e)
            if "Unauthorized" in message or "Authentication" in message:
                raise ValueError(f"Authentication failed: {message}") from e
            raise
        return response.text

    async def generate_code(self, 
                          prompt: str,
                          language: str = "python",
                          max_attempts: int = 3) -> str:
        """Generate code using the DeepSeek model.
        
        Args:
            prompt: The coding prompt.
            language: Target programming language.
            max_attempts: Maximum number of retry attempts.
            
        Returns:
            Generated code as string.
            
        Raises:
            Exception if code generation fails after max attempts.
        """
        formatted_prompt = f"Generate {language} code for: {prompt}\n\nCode:"
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = await self._make_request(formatted_prompt, is_retry=(attempt > 0))
                return response.text.strip()
            except Exception as e:
                last_error = e
                if attempt == max_attempts - 1:
                    break
                await asyncio.sleep(1)  # Wait before retry
        
        raise last_error or RuntimeError("Failed to generate code after all attempts")
                
    async def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze software requirements.
        
        Args:
            requirements: The requirements to analyze.
            
        Returns:
            Dictionary containing analysis results.
        """
        prompt = f"""Analyze the following software requirements and extract key components:
        
Requirements:
{requirements}

Provide analysis in JSON format with:
- Functional requirements
- Non-functional requirements
- Dependencies
- Constraints
- Risks
- Complexity
"""
        
        response = await self._make_request(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse analysis results", "raw": response.text}
            
    async def design_tests(self, 
                          code: str,
                          test_framework: str = "pytest") -> List[Dict[str, Any]]:
        """Design test cases for given code.
        
        Args:
            code: The code to test.
            test_framework: Target test framework.
            
        Returns:
            List of test cases.
        """
        prompt = f"""Design test cases for the following code using {test_framework}:
        
Code:
{code}

Provide test cases in JSON format with:
- Test name
- Input data
- Expected output
- Test type (unit/integration/etc)
- Priority
"""
        
        response = await self._make_request(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return [{"error": "Failed to parse test cases", "raw": response.text}]
            
    async def review_code(self, code: str) -> Dict[str, Any]:
        """Review code for quality and issues.
        
        Args:
            code: The code to review.
            
        Returns:
            Dictionary containing review results.
        """
        prompt = f"""Review the following code and provide detailed feedback:
        
Code:
{code}

Provide review in JSON format with:
- Code quality issues
- Security concerns
- Performance considerations
- Best practice violations
- Suggested improvements
- Style check
"""
        
        response = await self._make_request(prompt)
        try:
            return json.loads(response.text)
        except json.JSONDecodeError:
            return {"error": "Failed to parse review results", "raw": response.text}
            
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get provider usage metrics.
        
        Returns:
            Dictionary containing usage metrics.
        """
        return self._metrics.copy() 