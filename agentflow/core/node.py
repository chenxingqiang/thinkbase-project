from enum import Enum
from typing import Any, Dict, Optional, Union, List
from pydantic import BaseModel, Field, ConfigDict
import asyncio
import logging
from abc import ABC, abstractmethod

from ell import LMPType, Message
from agentflow.ell.integration import ELLIntegration

logger = logging.getLogger(__name__)

# Initialize ELL integration
ell_integration = ELLIntegration()

class NodeType(str, Enum):
    """Node type"""
    AGENT = "agent"
    PROCESSOR = "processor"
    WORKFLOW = "workflow"

class NodeState(str, Enum):
    """Node state"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"

class Node(ABC):
    """Base class for all nodes in the workflow."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self.metadata: Dict[str, Any] = {}
        self.status: str = "initialized"
    
    @abstractmethod
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process input context and return output."""
        pass
    
    async def initialize(self) -> None:
        """Initialize the node."""
        self.status = "ready"
    
    async def cleanup(self) -> None:
        """Clean up node resources."""
        self.status = "cleaned"
    
    def get_status(self) -> str:
        """Get the current node status."""
        return self.status
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata to the node."""
        self.metadata[key] = value
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get node metadata."""
        return self.metadata.copy()
    
    def validate(self) -> bool:
        """Validate the node configuration."""
        return True

class AgentNode(ABC):
    """Base class for agent nodes in a workflow."""
    
    def __init__(self, name: str, description: str):
        """Initialize agent node.
        
        Args:
            name: Node name
            description: Node description
        """
        self.name = name
        self.description = description
        self.status = "initialized"
        self.error = None
        self._initialized = False
    
    async def initialize(self) -> None:
        """Initialize node resources."""
        if self._initialized:
            return
        
        try:
            await self._initialize_impl()
            self._initialized = True
            self.status = "ready"
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"Error initializing node {self.name}: {e}")
            raise
    
    async def _initialize_impl(self) -> None:
        """Implementation of node initialization."""
        pass
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            self.status = "processing"
            result = await self.process(task)
            self.status = "ready"
            return result
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"Error processing task in node {self.name}: {e}")
            raise
    
    @abstractmethod
    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task.
        
        Args:
            task: Task to process
            
        Returns:
            Processing result
        """
        pass
    
    async def cleanup(self) -> None:
        """Clean up node resources."""
        try:
            await self._cleanup_impl()
            self._initialized = False
            self.status = "terminated"
        except Exception as e:
            self.status = "failed"
            self.error = str(e)
            logger.error(f"Error cleaning up node {self.name}: {e}")
            raise
    
    async def _cleanup_impl(self) -> None:
        """Implementation of node cleanup."""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get node status.
        
        Returns:
            Status information
        """
        return {
            "name": self.name,
            "status": self.status,
            "error": self.error
        }

class ProcessorNode(Node):
    """Node that executes a processor"""
    type: str = "processor"
    processor: Any  # Can be string (import path) or class
    config: Dict[str, Any] = Field(default_factory=dict)
    metrics: Dict[str, Union[int, float]] = Field(default_factory=lambda: {
        'tokens': 0,
        'latency': 0.0,
        'memory': 0
    })
    use_ell: bool = Field(default=False)  # Flag to use ELL
    ell_mode: str = Field(default="simple")  # ELL mode: "simple" or "complex"

    def __init__(self, **data):
        super().__init__(**data)
        if self.use_ell:
            from .ell_integration import ell_integration
            self.ell = ell_integration

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor with input data"""
        if self.use_ell:
            return await self._execute_with_ell(input_data)
        return await self._execute_standard(input_data)

    async def _execute_with_ell(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor using ELL"""
        # Create ELL message
        message = Message(
            role="user",
            content=str(input_data),
            metadata={
                "type": LMPType.TOOL,
                "config": self.config,
                "processor": self.processor.__name__ if hasattr(self.processor, "__name__") else "processor"
            }
        )
        
        # Execute with ELL based on mode
        try:
            result = await self.ell.process_message(message)
            return result.to_dict()
        except Exception as e:
            logger.error(f"Error executing processor with ELL: {str(e)}")
            raise

    async def _execute_simple_ell(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL simple mode"""
        try:
            # Get processor implementation
            processor = self._get_processor_instance()

            # Execute with global ELL simple mode
            @self.ell.with_ell(mode="simple")
            async def execute_processor():
                if hasattr(processor, 'process_data'):
                    return await processor.process_data({
                        "messages": [message],
                        "config": self.config
                    })
                return await processor.process({
                    "messages": [message],
                    "config": self.config
                })

            return await execute_processor()
        except Exception as e:
            logger.error(f"Error executing processor with ELL simple: {str(e)}")
            raise

    async def _execute_complex_ell(self, message: Any) -> Dict[str, Any]:
        """Execute using ELL complex mode"""
        try:
            # Get processor implementation
            processor = self._get_processor_instance()

            # Execute with global ELL complex mode
            @self.ell.with_ell(mode="complex")
            async def execute_processor():
                if hasattr(processor, 'process_data'):
                    return await processor.process_data({
                        "messages": self.ell.get_messages(),
                        "config": self.config
                    })
                return await processor.process({
                    "messages": self.ell.get_messages(),
                    "config": self.config
                })

            return await execute_processor()
        except Exception as e:
            logger.error(f"Error executing processor with ELL complex: {str(e)}")
            raise

    def _get_processor_instance(self):
        """Get processor instance"""
        if isinstance(self.processor, str):
            module_path, class_name = self.processor.rsplit(".", 1)
            module = __import__(module_path, fromlist=[class_name])
            processor_class = getattr(module, class_name)
            return processor_class(**self.config)
        return self.processor(**self.config)

    async def _execute_standard(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute processor using standard method"""
        processor = self._get_processor_instance()
        if hasattr(processor, 'process_data'):
            return await processor.process_data(input_data)
        return await processor.process(input_data)

class WorkflowNode(BaseModel):
    """Workflow node class."""
    model_config = ConfigDict(frozen=False, validate_assignment=True)
    
    id: str
    name: Optional[str] = None
    node_type: NodeType = Field(default=NodeType.WORKFLOW)
    state: NodeState = Field(default_factory=lambda: NodeState.PENDING)
    input_schema: Optional[Dict[str, Any]] = None
    output_schema: Optional[Dict[str, Any]] = None
    agent_instance: Optional[Any] = None
    processor: Optional[Any] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow node with input data"""
        try:
            self.state = NodeState.RUNNING
            if self.node_type == NodeType.AGENT and self.agent_instance:
                # Add delay if specified in config
                if self.config.get('delay'):
                    await asyncio.sleep(self.config['delay'])
                result = await self.agent_instance.execute(input_data)
                self.state = NodeState.COMPLETED
                return result
            self.state = NodeState.COMPLETED
            return {}
        except Exception as e:
            self.state = NodeState.ERROR
            raise e
