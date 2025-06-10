"""Retry policy module."""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

class RetryPolicy(BaseModel):
    """Retry policy configuration."""
    
    model_config = ConfigDict(frozen=True)
    
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Initial delay between retries in seconds")
    retry_backoff: float = Field(default=2.0, description="Multiplier for delay between retries")
    max_delay: float = Field(default=60.0, description="Maximum delay between retries in seconds")
    retry_on_exceptions: list[str] = Field(default_factory=lambda: ["TimeoutError", "ConnectionError"], description="List of exception names to retry on")
    retry_on_status_codes: list[int] = Field(default_factory=lambda: [408, 429, 500, 502, 503, 504], description="List of HTTP status codes to retry on")
    retry_on_conditions: Dict[str, Any] = Field(default_factory=dict, description="Additional conditions for retrying")

    def should_retry(self, attempt: int, exception: Optional[Exception] = None, status_code: Optional[int] = None) -> bool:
        """Check if a retry should be attempted.
        
        Args:
            attempt: Current attempt number (1-based)
            exception: Optional exception that occurred
            status_code: Optional HTTP status code
            
        Returns:
            bool: True if should retry, False otherwise
        """
        # Check max retries
        if attempt >= self.max_retries:
            return False
            
        # Check exception type
        if exception is not None:
            exception_name = exception.__class__.__name__
            if exception_name in self.retry_on_exceptions:
                return True
                
        # Check status code
        if status_code is not None and status_code in self.retry_on_status_codes:
            return True
            
        return False
        
    def get_delay(self, attempt: int) -> float:
        """Get delay for current retry attempt.
        
        Args:
            attempt: Current attempt number (1-based)
            
        Returns:
            float: Delay in seconds
        """
        delay = self.retry_delay * (self.retry_backoff ** (attempt - 1))
        return min(delay, self.max_delay) 