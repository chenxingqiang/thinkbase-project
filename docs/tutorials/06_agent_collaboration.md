# Tutorial 6: Agent Collaboration and Workflow Management

This tutorial demonstrates how to implement agent collaboration and workflow management in a distributed AgentFlow system.

## Prerequisites

- Completed Tutorials 1-5
- Understanding of agent collaboration patterns
- Familiarity with workflow management

## 1. Enhanced Node Architecture

Let's update the distributed node architecture to support agent collaboration:

```python
from agentflow.core.isa import AgentISA, AgentISAConfig
from agentflow.core.node import Node, NodeType, NodeState
from agentflow.core.objective_handler import ObjectiveHandler
from typing import Dict, Any, List, Optional, Set
import asyncio
import redis.asyncio as redis
import aiohttp
import json

class CollaborativeNode(Node):
    """Enhanced node implementation with collaboration support."""
    
    def __init__(self, 
                 name: str, 
                 node_type: NodeType, 
                 config: Dict[str, Any],
                 objective_handler: Optional[ObjectiveHandler] = None):
        super().__init__(name=name, node_type=node_type)
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.state = NodeState.PENDING
        self.peers: Dict[str, str] = {}  # node_id -> address
        self.objective_handler = objective_handler or ObjectiveHandler()
        self.capabilities: Set[str] = set()
        self.active_collaborations: Dict[str, Dict[str, Any]] = {}
    
    async def initialize(self) -> None:
        """Initialize node resources."""
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config.get("redis_db", 0)
        )
        
        # Initialize capabilities
        self.capabilities = set(self.config.get("capabilities", []))
        
        # Register node in cluster with enhanced metadata
        await self.redis_client.hset(
            "nodes",
            self.name,
            json.dumps({
                "address": self.config["node_address"],
                "type": self.node_type.value,
                "state": self.state.value,
                "capabilities": list(self.capabilities),
                "current_objectives": list(self.objective_handler.get_active_objectives())
            })
        )
        
        # Start heartbeat and collaboration monitor
        asyncio.create_task(self._heartbeat())
        asyncio.create_task(self._monitor_collaborations())
        
        self.state = NodeState.RUNNING
    
    async def _heartbeat(self):
        """Send periodic heartbeat with enhanced status."""
        while True:
            try:
                status = {
                    "timestamp": int(time.time()),
                    "state": self.state.value,
                    "active_collaborations": len(self.active_collaborations),
                    "capabilities": list(self.capabilities),
                    "load": await self._get_load_metrics()
                }
                
                await self.redis_client.hset(
                    "node_heartbeats",
                    self.name,
                    json.dumps(status)
                )
                
                # Update peer list with capabilities
                nodes = await self.redis_client.hgetall("nodes")
                self.peers = {
                    name.decode(): json.loads(info.decode())
                    for name, info in nodes.items()
                    if name.decode() != self.name
                }
                
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
            await asyncio.sleep(5)
    
    async def _monitor_collaborations(self):
        """Monitor and manage active collaborations."""
        while True:
            try:
                for collab_id, collab_info in list(self.active_collaborations.items()):
                    # Check collaboration health
                    if not await self._verify_collaboration(collab_id):
                        await self._handle_collaboration_failure(collab_id)
                    
                    # Update collaboration status
                    await self._update_collaboration_status(collab_id)
                    
            except Exception as e:
                logger.error(f"Collaboration monitoring error: {str(e)}")
            await asyncio.sleep(10)
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process task with collaboration support."""
        try:
            # Check if task requires collaboration
            if self._requires_collaboration(context):
                return await self._handle_collaborative_task(context)
            
            # Check if task can be processed locally
            if self._can_process_locally(context):
                return await self._process_locally(context)
            
            # Forward to appropriate node
            target_node = await self._select_target_node(context)
            if target_node:
                return await self._forward_task(target_node, context)
            
            raise ValueError("No suitable node found for processing")
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.state = NodeState.ERROR
            raise
    
    def _requires_collaboration(self, context: Dict[str, Any]) -> bool:
        """Check if task requires collaboration."""
        required_capabilities = set(context.get("required_capabilities", []))
        return bool(required_capabilities - self.capabilities)
    
    async def _handle_collaborative_task(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle task that requires collaboration."""
        # Create collaboration session
        collab_id = str(uuid.uuid4())
        
        try:
            # Find collaborators
            collaborators = await self._find_collaborators(context)
            if not collaborators:
                raise ValueError("No suitable collaborators found")
            
            # Initialize collaboration
            self.active_collaborations[collab_id] = {
                "id": collab_id,
                "context": context,
                "collaborators": collaborators,
                "state": "initializing",
                "results": {}
            }
            
            # Set up collaboration
            await self._setup_collaboration(collab_id)
            
            # Execute collaborative task
            result = await self._execute_collaborative_task(collab_id)
            
            # Cleanup collaboration
            await self._cleanup_collaboration(collab_id)
            
            return result
            
        except Exception as e:
            if collab_id in self.active_collaborations:
                await self._cleanup_collaboration(collab_id)
            raise
    
    async def _find_collaborators(self, context: Dict[str, Any]) -> List[str]:
        """Find suitable collaborators for task."""
        required_capabilities = set(context.get("required_capabilities", []))
        collaborators = []
        
        for peer_id, peer_info in self.peers.items():
            peer_capabilities = set(peer_info.get("capabilities", []))
            if peer_capabilities & (required_capabilities - self.capabilities):
                collaborators.append(peer_id)
        
        return collaborators
    
    async def _setup_collaboration(self, collab_id: str):
        """Set up collaboration session."""
        collab = self.active_collaborations[collab_id]
        
        # Notify collaborators
        for collaborator in collab["collaborators"]:
            response = await self._notify_collaborator(
                collaborator,
                {
                    "action": "setup_collaboration",
                    "collab_id": collab_id,
                    "context": collab["context"]
                }
            )
            
            if response.get("status") != "accepted":
                raise ValueError(f"Collaborator {collaborator} rejected collaboration")
        
        collab["state"] = "active"
    
    async def _execute_collaborative_task(self, collab_id: str) -> Dict[str, Any]:
        """Execute collaborative task."""
        collab = self.active_collaborations[collab_id]
        
        # Distribute subtasks
        subtasks = self._create_subtasks(collab["context"])
        results = {}
        
        for subtask in subtasks:
            # Find best collaborator for subtask
            collaborator = await self._select_collaborator(
                collab["collaborators"],
                subtask
            )
            
            # Execute subtask
            result = await self._execute_subtask(collaborator, subtask)
            results[subtask["id"]] = result
        
        # Combine results
        final_result = self._combine_results(results)
        collab["results"] = final_result
        
        return final_result
    
    def _create_subtasks(self, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create subtasks for collaborative execution."""
        # Implement subtask creation logic
        return []
    
    async def _select_collaborator(self, collaborators: List[str],
                                 subtask: Dict[str, Any]) -> str:
        """Select best collaborator for subtask."""
        # Implement collaborator selection logic
        return collaborators[0]
    
    async def _execute_subtask(self, collaborator: str,
                             subtask: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subtask on collaborator."""
        response = await self._notify_collaborator(
            collaborator,
            {
                "action": "execute_subtask",
                "subtask": subtask
            }
        )
        return response.get("result", {})
    
    def _combine_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Combine subtask results."""
        # Implement result combination logic
        return {"combined_result": results}
    
    async def _cleanup_collaboration(self, collab_id: str):
        """Clean up collaboration session."""
        if collab_id not in self.active_collaborations:
            return
            
        collab = self.active_collaborations[collab_id]
        
        # Notify collaborators
        for collaborator in collab["collaborators"]:
            try:
                await self._notify_collaborator(
                    collaborator,
                    {
                        "action": "cleanup_collaboration",
                        "collab_id": collab_id
                    }
                )
            except Exception as e:
                logger.error(f"Error notifying collaborator {collaborator}: {e}")
        
        # Remove collaboration
        del self.active_collaborations[collab_id]
    
    async def _notify_collaborator(self, collaborator: str,
                                 message: Dict[str, Any]) -> Dict[str, Any]:
        """Send notification to collaborator."""
        collaborator_info = self.peers.get(collaborator)
        if not collaborator_info:
            raise ValueError(f"Unknown collaborator: {collaborator}")
            
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{collaborator_info['address']}/collaborate",
                json=message
            ) as response:
                return await response.json()
    
    async def _verify_collaboration(self, collab_id: str) -> bool:
        """Verify collaboration health."""
        collab = self.active_collaborations.get(collab_id)
        if not collab:
            return False
            
        # Check collaborator health
        for collaborator in collab["collaborators"]:
            try:
                response = await self._notify_collaborator(
                    collaborator,
                    {
                        "action": "health_check",
                        "collab_id": collab_id
                    }
                )
                if response.get("status") != "healthy":
                    return False
            except Exception:
                return False
        
        return True
    
    async def _handle_collaboration_failure(self, collab_id: str):
        """Handle collaboration failure."""
        collab = self.active_collaborations.get(collab_id)
        if not collab:
            return
            
        # Implement failure handling logic
        await self._cleanup_collaboration(collab_id)
    
    async def _update_collaboration_status(self, collab_id: str):
        """Update collaboration status."""
        collab = self.active_collaborations.get(collab_id)
        if not collab:
            return
            
        # Update collaboration metrics
        metrics = await self._get_collaboration_metrics(collab_id)
        
        # Store metrics
        await self.redis_client.hset(
            f"collaboration_metrics:{collab_id}",
            mapping=metrics
        )
    
    async def _get_collaboration_metrics(self, collab_id: str) -> Dict[str, Any]:
        """Get collaboration metrics."""
        collab = self.active_collaborations.get(collab_id)
        if not collab:
            return {}
            
        return {
            "active_time": time.time() - collab.get("start_time", time.time()),
            "num_subtasks": len(collab.get("results", {})),
            "collaborators": len(collab["collaborators"]),
            "state": collab["state"]
        }
    
    async def _get_load_metrics(self) -> Dict[str, float]:
        """Get node load metrics."""
        return {
            "cpu_usage": psutil.cpu_percent() / 100.0,
            "memory_usage": psutil.virtual_memory().percent / 100.0,
            "collaboration_load": len(self.active_collaborations) / 10.0  # Normalize to 0-1
        }

class CollaborativeAgentISA:
    """Enhanced AgentISA with collaboration support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, CollaborativeNode] = {}
        self.agent_isa = AgentISA(AgentISAConfig(**config.get("isa_config", {})))
        self.objective_handler = ObjectiveHandler()
    
    async def initialize(self):
        """Initialize collaborative system."""
        # Initialize objective handler
        await self.objective_handler.initialize()
        
        # Create nodes
        for node_config in self.config["nodes"]:
            node = CollaborativeNode(
                name=node_config["name"],
                node_type=NodeType(node_config["type"]),
                config=node_config,
                objective_handler=self.objective_handler
            )
            await node.initialize()
            self.nodes[node.name] = node
    
    async def execute_instruction(self, instruction_id: str,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction with collaboration support."""
        # Create execution context
        context = self._create_execution_context(instruction_id, params)
        
        # Select primary node
        node = await self._select_primary_node(context)
        if not node:
            raise ValueError("No suitable node found")
        
        # Execute with collaboration
        return await node.process(context)
    
    def _create_execution_context(self, instruction_id: str,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Create execution context with collaboration info."""
        instruction = self.agent_isa.instructions.get(instruction_id)
        if not instruction:
            raise ValueError(f"Unknown instruction: {instruction_id}")
            
        return {
            "instruction_id": instruction_id,
            "params": params,
            "required_capabilities": self._get_required_capabilities(instruction),
            "objectives": self.objective_handler.get_relevant_objectives(params)
        }
    
    def _get_required_capabilities(self, instruction: Any) -> List[str]:
        """Get required capabilities for instruction."""
        # Implement capability extraction logic
        return []
    
    async def _select_primary_node(self, context: Dict[str, Any]) -> Optional[CollaborativeNode]:
        """Select primary node for execution."""
        # Implement node selection logic
        return next(iter(self.nodes.values()), None)
```

## 2. Collaboration API

Implement the collaboration API endpoints:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

class CollaborationRequest(BaseModel):
    action: str
    collab_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    subtask: Optional[Dict[str, Any]] = None

class CollaborationResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

def create_collaboration_api(node: CollaborativeNode) -> FastAPI:
    """Create FastAPI application with collaboration endpoints."""
    app = FastAPI()
    
    @app.post("/collaborate", response_model=CollaborationResponse)
    async def handle_collaboration(request: CollaborationRequest):
        try:
            if request.action == "setup_collaboration":
                # Handle collaboration setup
                await node._setup_collaboration(request.collab_id)
                return CollaborationResponse(status="accepted")
                
            elif request.action == "execute_subtask":
                # Execute subtask
                result = await node._process_locally(request.subtask)
                return CollaborationResponse(
                    status="completed",
                    result=result
                )
                
            elif request.action == "cleanup_collaboration":
                # Clean up collaboration
                await node._cleanup_collaboration(request.collab_id)
                return CollaborationResponse(status="success")
                
            elif request.action == "health_check":
                # Return health status
                return CollaborationResponse(status="healthy")
                
            else:
                raise ValueError(f"Unknown action: {request.action}")
                
        except Exception as e:
            return CollaborationResponse(
                status="error",
                error=str(e)
            )
    
    return app
```

## 3. Configuration Example

```yaml
# collaborative_config.yaml
redis_host: localhost
redis_port: 6379
mq_host: localhost
mq_port: 5672
mq_user: guest
mq_password: guest

nodes:
  - name: node1
    type: AGENT
    node_address: http://localhost:8001
    redis_host: localhost
    redis_port: 6379
    capabilities:
      - text_processing
      - data_analysis
      
  - name: node2
    type: PROCESSOR
    node_address: http://localhost:8002
    redis_host: localhost
    redis_port: 6379
    capabilities:
      - image_processing
      - machine_learning

isa_config:
  enable_learning: true
  enable_optimization: true
  min_pattern_support: 2
  min_success_rate: 0.8

collaboration:
  max_active_collaborations: 10
  collaboration_timeout: 300
  health_check_interval: 10
  retry_attempts: 3
```

## 4. Usage Example

```python
async def main():
    # Load configuration
    with open("collaborative_config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize collaborative system
    collaborative_isa = CollaborativeAgentISA(config)
    await collaborative_isa.initialize()
    
    # Create API for each node
    for node in collaborative_isa.nodes.values():
        app = create_collaboration_api(node)
        port = int(node.config["node_address"].split(":")[-1])
        
        # Run API in background
        asyncio.create_task(
            uvicorn.run(app, host="0.0.0.0", port=port)
        )
    
    # Execute collaborative instruction
    result = await collaborative_isa.execute_instruction(
        "complex_task",
        {
            "input_data": "sample data",
            "required_processing": ["text_processing", "image_processing"]
        }
    )
    
    print(f"Execution result: {result}")

if __name__ == "__main__":
    asyncio.run(main())
```

## Next Steps

1. Implement advanced collaboration patterns
2. Add support for dynamic capability discovery
3. Implement collaboration recovery mechanisms
4. Add collaboration metrics and monitoring
5. Implement cross-node learning

Check out the agent collaboration specification for detailed information about implementing collaborative features. 