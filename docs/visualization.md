# AgentFlow Visualization Guide

## Overview

AgentFlow provides powerful visualization capabilities through integration with ell.studio. This guide explains how to use the visualization components to create interactive and real-time visualizations of your agents and workflows.

## Table of Contents

1. [Components](#components)
2. [Visualization Types](#visualization-types)
3. [Real-time Updates](#real-time-updates)
4. [Customization](#customization)
5. [Integration](#integration)

## Components

### Visual Graph

The `VisualGraph` class is the core component for creating visualizations:

```python
from agentflow.visualization.components import VisualGraph, VisualNode, VisualEdge

# Create graph
graph = VisualGraph()

# Add nodes
graph.add_node(VisualNode(
    id="node1",
    type=NodeType.AGENT,
    label="Research Agent",
    data={"version": "1.0.0"},
    position={"x": 0, "y": 0}
))

# Add edges
graph.add_edge(VisualEdge(
    id="edge1",
    source="node1",
    target="node2",
    type=EdgeType.DATA_FLOW
))
```

### Node Types

Available node types:

- `AGENT`: Represents an agent instance
- `WORKFLOW`: Represents a workflow
- `INPUT`: Input processor node
- `OUTPUT`: Output processor node
- `PROCESSOR`: Data processing node
- `CONNECTOR`: Connection node

### Edge Types

Available edge types:

- `DATA_FLOW`: Represents data flow between nodes
- `CONTROL_FLOW`: Represents control flow
- `MESSAGE`: Represents message passing

## Visualization Types

### Agent Visualization

Visualize agent configuration and state:

```python
from agentflow.visualization.renderer import AgentVisualizer

visualizer = AgentVisualizer()
visual_data = visualizer.visualize_agent({
    "agent": {
        "name": "Research Agent",
        "version": "1.0.0",
        "type": "research"
    },
    "input_specification": {...},
    "output_specification": {...}
})
```

### Workflow Visualization

Visualize workflow structure and execution:

```python
visual_data = visualizer.visualize_workflow({
    "name": "Research Workflow",
    "steps": [
        {
            "name": "Data Collection",
            "type": "input",
            "config": {...}
        },
        {
            "name": "Analysis",
            "type": "processor",
            "config": {...}
        }
    ]
})
```

## Real-time Updates

### WebSocket Integration

```python
from agentflow.visualization.service import VisualizationService

# Create service
service = VisualizationService({
    "api_key": "your_ell_studio_api_key",
    "project_id": "your_project_id"
})

# Start service
service.start()
```

### Client-side Updates

```python
import websockets
import json

async def send_updates():
    async with websockets.connect("ws://localhost:8001/live") as websocket:
        # Send status update
        await websocket.send(json.dumps({
            "type": "status_update",
            "data": {
                "node_id": "agent1",
                "status": "processing",
                "progress": 0.5
            }
        }))
        
        # Receive confirmation
        response = await websocket.recv()
        print(json.loads(response))
```

## Customization

### Styling

Customize node and edge appearance:

```python
from agentflow.visualization.components import DefaultStyles

# Custom node style
custom_style = {
    "backgroundColor": "#6366f1",
    "borderRadius": 8,
    "padding": 16,
    "color": "#ffffff"
}

# Apply style
node = VisualNode(
    id="custom_node",
    type=NodeType.AGENT,
    label="Custom Agent",
    style=custom_style
)
```

### Layout

Configure automatic layout:

```python
from agentflow.visualization.components import VisualLayout

# Create layout configuration
layout_config = {
    "type": "force",
    "options": {
        "strength": 0.5,
        "distance": 100
    }
}

# Apply layout
graph = VisualLayout.auto_layout(graph, layout_config)
```

## Integration

### ell.studio Integration

Configure ell.studio integration:

```python
from agentflow.visualization.service import EllStudioIntegration

# Create integration
integration = EllStudioIntegration({
    "api_key": "your_api_key",
    "project_id": "your_project_id"
})

# Push visualization
await integration.push_visualization(visual_data)

# Update visualization
await integration.update_visualization({
    "node_id": "agent1",
    "status": "completed"
})
```

### Custom Event Handlers

Register custom event handlers:

```python
from agentflow.visualization.components import InteractionHandler

# Create handler
handler = InteractionHandler()

# Register custom handler
def on_node_click(event_data):
    node_id = event_data["node_id"]
    print(f"Node clicked: {node_id}")
    
handler.register_handler("node_click", on_node_click)
```

## Best Practices

1. **Node Organization**
   - Group related nodes together
   - Use clear and descriptive labels
   - Maintain consistent spacing

2. **Edge Management**
   - Minimize edge crossings
   - Use appropriate edge types
   - Consider edge direction

3. **Real-time Updates**
   - Send only necessary updates
   - Batch updates when possible
   - Handle connection errors

4. **Performance**
   - Limit number of visible nodes
   - Use efficient layouts
   - Cache visualization data

## Troubleshooting

Common issues and solutions:

1. **Connection Issues**

   ```python
   try:
       await websocket.connect()
   except Exception as e:
       print(f"Connection failed: {str(e)}")
       # Implement retry logic
   ```

2. **Layout Problems**

   ```python
   # Reset layout
   graph = VisualLayout.auto_layout(graph, {
       "type": "force",
       "options": {"reset": True}
   })
   ```

3. **Performance Issues**

   ```python
   # Optimize updates
   service.config.update({
       "update_throttle": 100,  # ms
       "batch_updates": True
   })
   ```
