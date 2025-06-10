# AgentFlow API Reference

## Core API

### Agent

#### Agent Configuration

```python
from agentflow.core.config import AgentConfig

class AgentConfig:
    """Agent configuration class."""
    
    def __init__(self, **kwargs):
        """Initialize agent configuration.
        
        Args:
            agent (dict): Agent metadata
            input_specification (dict): Input processing configuration
            output_specification (dict): Output processing configuration
            flow_control (dict, optional): Flow control configuration
            metadata (dict, optional): Additional metadata
        """
        pass
        
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'AgentConfig':
        """Create configuration from dictionary."""
        pass
        
    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        pass
```

#### Agent Class

```python
from agentflow.core.agent import Agent

class Agent:
    """Base agent class."""
    
    def __init__(self, config: AgentConfig):
        """Initialize agent.
        
        Args:
            config: Agent configuration
        """
        pass
        
    async def process(self, input_data: dict) -> dict:
        """Process input data.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Processed output data
        """
        pass
```

### Input Processing

```python
from agentflow.core.input_processor import InputProcessor

class InputProcessor:
    """Input processing class."""
    
    def __init__(self, specification: dict):
        """Initialize input processor.
        
        Args:
            specification: Input processing specification
        """
        pass
        
    def process_input(self, data: dict, mode: str) -> dict:
        """Process input data.
        
        Args:
            data: Input data
            mode: Processing mode
            
        Returns:
            Processed input data
        """
        pass
```

### Output Processing

```python
from agentflow.core.output_processor import OutputProcessor

class OutputProcessor:
    """Output processing class."""
    
    def __init__(self, specification: dict):
        """Initialize output processor.
        
        Args:
            specification: Output processing specification
        """
        pass
        
    def process_output(self, data: dict, mode: str) -> dict:
        """Process output data.
        
        Args:
            data: Output data
            mode: Processing mode
            
        Returns:
            Processed output data
        """
        pass
```

## Visualization API

### Components

```python
from agentflow.visualization.components import VisualGraph, VisualNode, VisualEdge

class VisualGraph:
    """Visual graph class."""
    
    def add_node(self, node: VisualNode):
        """Add node to graph."""
        pass
        
    def add_edge(self, edge: VisualEdge):
        """Add edge to graph."""
        pass
        
    def to_ell_format(self) -> dict:
        """Convert to ell.studio format."""
        pass

class VisualNode:
    """Visual node class."""
    
    def __init__(self, id: str, type: str, label: str, data: dict = None,
                 position: dict = None, style: dict = None):
        """Initialize visual node.
        
        Args:
            id: Node ID
            type: Node type
            label: Node label
            data: Node data
            position: Node position
            style: Node style
        """
        pass

class VisualEdge:
    """Visual edge class."""
    
    def __init__(self, id: str, source: str, target: str, type: str,
                 data: dict = None, style: dict = None):
        """Initialize visual edge.
        
        Args:
            id: Edge ID
            source: Source node ID
            target: Target node ID
            type: Edge type
            data: Edge data
            style: Edge style
        """
        pass
```

### Renderer

```python
from agentflow.visualization.renderer import AgentVisualizer

class AgentVisualizer:
    """Agent visualization class."""
    
    def __init__(self, config: dict = None):
        """Initialize visualizer.
        
        Args:
            config: Visualization configuration
        """
        pass
        
    def visualize_agent(self, agent_config: dict) -> dict:
        """Visualize agent configuration.
        
        Args:
            agent_config: Agent configuration
            
        Returns:
            Visualization data
        """
        pass
        
    def visualize_workflow(self, workflow_config: dict) -> dict:
        """Visualize workflow configuration.
        
        Args:
            workflow_config: Workflow configuration
            
        Returns:
            Visualization data
        """
        pass
```

### Service

```python
from agentflow.visualization.service import VisualizationService

class VisualizationService:
    """Visualization service class."""
    
    def __init__(self, config: dict = None):
        """Initialize service.
        
        Args:
            config: Service configuration
        """
        pass
        
    def start(self, host: str = "0.0.0.0", port: int = 8001):
        """Start service.
        
        Args:
            host: Service host
            port: Service port
        """
        pass
```

## REST API

### Endpoints

#### POST /visualize/agent

Visualize agent configuration.

Request:
```json
{
    "agent": {
        "name": "string",
        "version": "string",
        "type": "string"
    },
    "input_specification": {
        "MODES": ["string"],
        "TYPES": {}
    },
    "output_specification": {
        "MODES": ["string"],
        "STRATEGIES": {}
    }
}
```

Response:
```json
{
    "nodes": [
        {
            "id": "string",
            "type": "string",
            "label": "string",
            "data": {},
            "position": {},
            "style": {}
        }
    ],
    "edges": [
        {
            "id": "string",
            "source": "string",
            "target": "string",
            "type": "string",
            "data": {},
            "style": {}
        }
    ]
}
```

#### POST /visualize/workflow

Visualize workflow configuration.

Request:
```json
{
    "name": "string",
    "steps": [
        {
            "name": "string",
            "type": "string",
            "config": {}
        }
    ]
}
```

Response:
```json
{
    "nodes": [],
    "edges": []
}
```

### WebSocket API

#### WS /live

Real-time visualization updates.

Message Format:
```json
{
    "type": "string",
    "data": {}
}
```

Event Types:
- `status_update`: Update node status
- `progress_update`: Update progress
- `error`: Error notification
- `complete`: Completion notification

## Error Handling

### Error Types

```python
from agentflow.core.exceptions import (
    AgentFlowError,
    ConfigurationError,
    ProcessingError,
    VisualizationError
)

class AgentFlowError(Exception):
    """Base exception class."""
    pass

class ConfigurationError(AgentFlowError):
    """Configuration error."""
    pass

class ProcessingError(AgentFlowError):
    """Processing error."""
    pass

class VisualizationError(AgentFlowError):
    """Visualization error."""
    pass
```

### Error Handling Example

```python
try:
    result = agent.process(input_data)
except ConfigurationError as e:
    print(f"Configuration error: {str(e)}")
except ProcessingError as e:
    print(f"Processing error: {str(e)}")
except VisualizationError as e:
    print(f"Visualization error: {str(e)}")
except AgentFlowError as e:
    print(f"General error: {str(e)}")
```
