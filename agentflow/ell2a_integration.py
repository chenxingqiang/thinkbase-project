"""ELL integration module."""

from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from ell.types.message import Message, MessageRole, MessageType

class ELLIntegration:
    """Singleton class for ELL integration."""
    
    _instance = None
    _initialized = False
    _messages = []
    _mode_configs = {
        "simple": {"model": "test-model"},
        "complex": {
            "model": "test-model-complex",
            "track_performance": True,
            "track_memory": True
        }
    }
    _workflows = {}
    _metrics = {}
    
    def __new__(cls):
        """Create or return singleton instance."""
        if cls._instance is None:
            cls._instance = super(ELLIntegration, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def __init__(self):
        """Initialize instance attributes."""
        pass
    
    def _initialize(self):
        """Initialize ELL integration."""
        if not self._initialized:
            self.config = {
                "default_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "simple": {
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 30.0
                },
                "complex": {
                    "max_retries": 3,
                    "retry_delay": 1.0,
                    "timeout": 60.0,
                    "stream": True,
                    "track_performance": True,
                    "track_memory": True
                }
            }
            self._initialized = True
    
    def cleanup(self):
        """Reset ELL integration state."""
        self._workflows.clear()
        self._messages.clear()
        self._metrics.clear()
        self._initialized = False
        
    def configure(self, config: Optional[Dict[str, Any]] = None):
        """Configure ELL integration."""
        if config:
            self._mode_configs.update(config)

    def get_mode_config(self, mode: str) -> Dict[str, Any]:
        """Get mode configuration."""
        return self._mode_configs.get(mode, {})
    
    async def process_message(self, message: Message) -> Message:
        """Process a message through ELL."""
        self._messages.append(message)
        # Return a result message
        return Message(
            role=MessageRole.ASSISTANT,
            content=message.content,
            type=MessageType.RESULT,
            metadata=message.metadata
        )
    
    def get_messages(self) -> List[Message]:
        """Get all messages in the context."""
        return self._messages.copy()
    
    def clear_messages(self):
        """Clear all messages from the context."""
        self._messages.clear()
    
    def register_workflow(self, workflow_id: str, workflow: Any) -> None:
        """Register a workflow.
        
        Args:
            workflow_id: Workflow ID
            workflow: Workflow instance
        """
        self._workflows[workflow_id] = workflow
    
    def unregister_workflow(self, workflow_id: str) -> None:
        """Unregister a workflow.
        
        Args:
            workflow_id: Workflow ID
        """
        self._workflows.pop(workflow_id, None)
    
    def list_workflows(self) -> List[str]:
        """List registered workflows.
        
        Returns:
            List of workflow IDs
        """
        return list(self._workflows.keys())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics.
        
        Returns:
            Current metrics
        """
        return self._metrics.copy()
    
    @staticmethod
    def with_ell(mode: str = "simple"):
        """Decorator to enable ELL for any function.
        
        Args:
            mode: Processing mode (simple or complex)
            
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell = ELLIntegration()
                config = ell.get_mode_config(mode)
                func_name = func.__name__
                
                # Initialize metrics for function if not exists
                if func_name not in ell._metrics:
                    ell._metrics[func_name] = {
                        "calls": 0,
                        "total_time": 0.0,
                        "errors": 0
                    }
                
                try:
                    # Execute original function
                    result = await func(*args, **kwargs)
                    
                    # Update metrics
                    ell._metrics[func_name]["calls"] += 1
                    
                    # Create message from function result
                    message = Message(
                        role=MessageRole.USER,
                        content=str(result),
                        metadata={
                            "mode": mode,
                            "function": func_name,
                            "status": "success",
                            **kwargs
                        }
                    )
                    
                    # Process with ELL
                    processed = await ell.process_message(message)
                    
                    # Return original result
                    return result
                    
                except Exception as e:
                    # Update error metrics
                    ell._metrics[func_name]["errors"] += 1
                    
                    # Create error message
                    message = Message(
                        role=MessageRole.USER,
                        content=str(e),
                        metadata={
                            "mode": mode,
                            "function": func_name,
                            "status": "error",
                            "error": str(e),
                            **kwargs
                        }
                    )
                    
                    # Process error with ELL
                    await ell.process_message(message)
                    
                    # Re-raise the original exception
                    raise
                
            return wrapper
        return decorator
    
    @staticmethod
    def track_function():
        """Decorator to track function execution with ELL.
        
        Returns:
            Decorated function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                ell = ELLIntegration()
                func_name = func.__name__
                
                # Initialize metrics for function if not exists
                if func_name not in ell._metrics:
                    ell._metrics[func_name] = {
                        "calls": 0,
                        "total_time": 0.0,
                        "errors": 0
                    }
                
                try:
                    # Track function execution
                    import time
                    start_time = time.time()
                    result = await func(*args, **kwargs)
                    ell._metrics[func_name]["calls"] += 1
                    ell._metrics[func_name]["total_time"] += time.time() - start_time
                    return result
                except Exception as e:
                    ell._metrics[func_name]["errors"] += 1
                    raise
                
            return wrapper
        return decorator

# Global instance
ell_integration = ELLIntegration() 