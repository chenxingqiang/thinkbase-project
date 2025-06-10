# Tutorial 5: Integration and Scaling

This tutorial demonstrates how to integrate AgentISA with external services and implement distributed deployment for scalability.

## Prerequisites

- Completed Tutorials 1-4
- Understanding of distributed systems
- Familiarity with message queues and caching

## 1. Distributed Node Architecture

Let's implement a distributed node system for AgentISA:

```python
from agentflow.core.isa import AgentISA, AgentISAConfig
from agentflow.core.node import Node, NodeType, NodeState
from typing import Dict, Any, List, Optional
import asyncio
import redis.asyncio as redis
import aiohttp
import json

class DistributedNode(Node):
    """Distributed node implementation."""
    
    def __init__(self, name: str, node_type: NodeType, config: Dict[str, Any]):
        super().__init__(name=name, node_type=node_type)
        self.config = config
        self.redis_client: Optional[redis.Redis] = None
        self.state = NodeState.PENDING
        self.peers: Dict[str, str] = {}  # node_id -> address
    
    async def initialize(self) -> None:
        """Initialize node resources."""
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=self.config["redis_host"],
            port=self.config["redis_port"],
            db=self.config.get("redis_db", 0)
        )
        
        # Register node in cluster
        await self.redis_client.hset(
            "nodes",
            self.name,
            json.dumps({
                "address": self.config["node_address"],
                "type": self.node_type.value,
                "state": self.state.value
            })
        )
        
        # Start heartbeat
        asyncio.create_task(self._heartbeat())
        
        self.state = NodeState.RUNNING
    
    async def _heartbeat(self):
        """Send periodic heartbeat."""
        while True:
            try:
                await self.redis_client.hset(
                    "node_heartbeats",
                    self.name,
                    str(int(time.time()))
                )
                # Update peer list
                nodes = await self.redis_client.hgetall("nodes")
                self.peers = {
                    name.decode(): json.loads(info.decode())["address"]
                    for name, info in nodes.items()
                    if name.decode() != self.name
                }
            except Exception as e:
                logger.error(f"Heartbeat error: {str(e)}")
            await asyncio.sleep(5)
    
    async def process(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process task in distributed environment."""
        try:
            # Check if task can be processed locally
            if self._can_process_locally(context):
                return await self._process_locally(context)
            
            # Forward to appropriate node
            target_node = self._select_target_node(context)
            if target_node:
                return await self._forward_task(target_node, context)
            
            raise ValueError("No suitable node found for processing")
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.state = NodeState.ERROR
            raise
    
    def _can_process_locally(self, context: Dict[str, Any]) -> bool:
        """Check if task can be processed locally."""
        # Implement local processing decision logic
        return True
    
    async def _process_locally(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process task locally."""
        # Implement local processing logic
        return {"status": "processed"}
    
    def _select_target_node(self, context: Dict[str, Any]) -> Optional[str]:
        """Select target node for task forwarding."""
        # Implement node selection logic
        return None
    
    async def _forward_task(self, node_address: str,
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Forward task to another node."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{node_address}/process",
                json=context
            ) as response:
                return await response.json()

class DistributedAgentISA:
    """Distributed AgentISA implementation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, DistributedNode] = {}
        self.agent_isa = AgentISA(AgentISAConfig(**config.get("isa_config", {})))
    
    async def initialize(self):
        """Initialize distributed system."""
        # Create nodes
        for node_config in self.config["nodes"]:
            node = DistributedNode(
                name=node_config["name"],
                node_type=NodeType(node_config["type"]),
                config=node_config
            )
            await node.initialize()
            self.nodes[node.name] = node
    
    async def execute_instruction(self, instruction_id: str,
                                params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instruction in distributed environment."""
        # Select node for execution
        node = self._select_node(instruction_id, params)
        if not node:
            raise ValueError("No suitable node found")
        
        # Execute on selected node
        return await node.process({
            "instruction_id": instruction_id,
            "params": params
        })
    
    def _select_node(self, instruction_id: str,
                     params: Dict[str, Any]) -> Optional[DistributedNode]:
        """Select node for instruction execution."""
        # Implement node selection logic
        return next(iter(self.nodes.values()), None)
```

## 2. Message Queue Integration

Implement message queue integration for asynchronous processing:

```python
from aio_pika import connect_robust, Message, ExchangeType
from typing import Callable, Awaitable

class MessageQueue:
    """Message queue integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.connection = None
        self.channel = None
        self.exchange = None
    
    async def initialize(self):
        """Initialize message queue connection."""
        self.connection = await connect_robust(
            host=self.config["mq_host"],
            port=self.config["mq_port"],
            login=self.config["mq_user"],
            password=self.config["mq_password"]
        )
        
        self.channel = await self.connection.channel()
        self.exchange = await self.channel.declare_exchange(
            "agentisa",
            ExchangeType.TOPIC
        )
    
    async def publish(self, routing_key: str, message: Dict[str, Any]):
        """Publish message to queue."""
        await self.exchange.publish(
            Message(
                body=json.dumps(message).encode(),
                content_type="application/json"
            ),
            routing_key=routing_key
        )
    
    async def subscribe(self, routing_key: str,
                       callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """Subscribe to messages."""
        queue = await self.channel.declare_queue(exclusive=True)
        await queue.bind(self.exchange, routing_key)
        
        async def process_message(message):
            async with message.process():
                payload = json.loads(message.body.decode())
                await callback(payload)
        
        await queue.consume(process_message)

class AsyncInstructionProcessor:
    """Asynchronous instruction processor."""
    
    def __init__(self, agent_isa: AgentISA, message_queue: MessageQueue):
        self.agent_isa = agent_isa
        self.mq = message_queue
    
    async def initialize(self):
        """Initialize processor."""
        await self.mq.initialize()
        await self.mq.subscribe("instruction.*", self._process_instruction)
    
    async def submit_instruction(self, instruction_id: str,
                               params: Dict[str, Any]) -> str:
        """Submit instruction for async processing."""
        task_id = str(uuid.uuid4())
        await self.mq.publish(
            f"instruction.{instruction_id}",
            {
                "task_id": task_id,
                "instruction_id": instruction_id,
                "params": params
            }
        )
        return task_id
    
    async def _process_instruction(self, message: Dict[str, Any]):
        """Process instruction from queue."""
        try:
            result = await self.agent_isa.execute_instruction(
                message["instruction_id"],
                message["params"]
            )
            
            # Publish result
            await self.mq.publish(
                f"result.{message['task_id']}",
                {
                    "task_id": message["task_id"],
                    "result": result,
                    "status": "completed"
                }
            )
            
        except Exception as e:
            # Publish error
            await self.mq.publish(
                f"result.{message['task_id']}",
                {
                    "task_id": message["task_id"],
                    "error": str(e),
                    "status": "failed"
                }
            )
```

## 3. Caching and State Management

Implement distributed caching and state management:

```python
from typing import TypeVar, Generic
from dataclasses import dataclass
import pickle

T = TypeVar('T')

@dataclass
class CacheEntry(Generic[T]):
    value: T
    expiry: Optional[float] = None

class DistributedCache:
    """Distributed cache implementation."""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
    
    async def get(self, key: str, default: T = None) -> Optional[T]:
        """Get value from cache."""
        try:
            value = await self.redis.get(key)
            if value is None:
                return default
            return pickle.loads(value)
        except Exception:
            return default
    
    async def set(self, key: str, value: T,
                  expiry: Optional[int] = None) -> bool:
        """Set value in cache."""
        try:
            await self.redis.set(
                key,
                pickle.dumps(value),
                ex=expiry
            )
            return True
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            await self.redis.delete(key)
            return True
        except Exception:
            return False

class StateManager:
    """Distributed state management."""
    
    def __init__(self, cache: DistributedCache):
        self.cache = cache
    
    async def get_instruction_state(self, instruction_id: str) -> Dict[str, Any]:
        """Get instruction state."""
        state = await self.cache.get(f"instruction:{instruction_id}")
        return state or {}
    
    async def update_instruction_state(self, instruction_id: str,
                                     state: Dict[str, Any]):
        """Update instruction state."""
        await self.cache.set(f"instruction:{instruction_id}", state)
    
    async def get_execution_history(self, instruction_id: str) -> List[Dict[str, Any]]:
        """Get instruction execution history."""
        history = await self.cache.get(f"history:{instruction_id}")
        return history or []
    
    async def add_execution_record(self, instruction_id: str,
                                 record: Dict[str, Any]):
        """Add execution record to history."""
        history = await self.get_execution_history(instruction_id)
        history.append(record)
        await self.cache.set(f"history:{instruction_id}", history)
```

## 4. Load Balancing and Scaling

Implement load balancing and auto-scaling:

```python
from dataclasses import dataclass
from typing import List, Dict, Any
import asyncio
import statistics

@dataclass
class NodeMetrics:
    cpu_usage: float
    memory_usage: float
    instruction_count: int
    error_rate: float

class LoadBalancer:
    """Load balancer implementation."""
    
    def __init__(self, nodes: Dict[str, DistributedNode]):
        self.nodes = nodes
        self.metrics: Dict[str, NodeMetrics] = {}
        asyncio.create_task(self._collect_metrics())
    
    async def _collect_metrics(self):
        """Collect node metrics periodically."""
        while True:
            for node in self.nodes.values():
                try:
                    stats = await node.get_stats()
                    self.metrics[node.name] = NodeMetrics(
                        cpu_usage=stats["cpu_usage"],
                        memory_usage=stats["memory_usage"],
                        instruction_count=stats["instruction_count"],
                        error_rate=stats["error_rate"]
                    )
                except Exception as e:
                    logger.error(f"Error collecting metrics from {node.name}: {e}")
            await asyncio.sleep(10)
    
    def select_node(self, instruction_id: str,
                    params: Dict[str, Any]) -> Optional[DistributedNode]:
        """Select node based on load balancing strategy."""
        available_nodes = [
            node for node in self.nodes.values()
            if node.state == NodeState.RUNNING
        ]
        
        if not available_nodes:
            return None
        
        # Calculate node scores
        scores = []
        for node in available_nodes:
            if node.name not in self.metrics:
                continue
                
            metrics = self.metrics[node.name]
            score = self._calculate_node_score(metrics)
            scores.append((node, score))
        
        if not scores:
            return available_nodes[0]
        
        # Select node with best score
        return max(scores, key=lambda x: x[1])[0]
    
    def _calculate_node_score(self, metrics: NodeMetrics) -> float:
        """Calculate node score for selection."""
        # Lower is better
        return (
            0.4 * (1 - metrics.cpu_usage) +
            0.3 * (1 - metrics.memory_usage) +
            0.2 * (1 / (metrics.instruction_count + 1)) +
            0.1 * (1 - metrics.error_rate)
        )

class AutoScaler:
    """Auto-scaling implementation."""
    
    def __init__(self, load_balancer: LoadBalancer,
                 config: Dict[str, Any]):
        self.load_balancer = load_balancer
        self.config = config
        self.scaling_lock = asyncio.Lock()
        asyncio.create_task(self._monitor_scaling())
    
    async def _monitor_scaling(self):
        """Monitor and adjust scaling."""
        while True:
            try:
                await self._check_scaling()
            except Exception as e:
                logger.error(f"Scaling error: {e}")
            await asyncio.sleep(60)
    
    async def _check_scaling(self):
        """Check if scaling is needed."""
        async with self.scaling_lock:
            metrics = list(self.load_balancer.metrics.values())
            if not metrics:
                return
            
            avg_cpu = statistics.mean(m.cpu_usage for m in metrics)
            avg_memory = statistics.mean(m.memory_usage for m in metrics)
            
            if self._should_scale_up(avg_cpu, avg_memory):
                await self._scale_up()
            elif self._should_scale_down(avg_cpu, avg_memory):
                await self._scale_down()
    
    def _should_scale_up(self, cpu: float, memory: float) -> bool:
        """Check if scaling up is needed."""
        return (
            cpu > self.config["cpu_threshold_up"] or
            memory > self.config["memory_threshold_up"]
        )
    
    def _should_scale_down(self, cpu: float, memory: float) -> bool:
        """Check if scaling down is needed."""
        return (
            cpu < self.config["cpu_threshold_down"] and
            memory < self.config["memory_threshold_down"]
        )
    
    async def _scale_up(self):
        """Scale up the cluster."""
        # Implement scaling up logic
        pass
    
    async def _scale_down(self):
        """Scale down the cluster."""
        # Implement scaling down logic
        pass
```

## 5. Deployment Example

Here's an example of deploying the distributed system:

```python
async def main():
    # Load configuration
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    
    # Initialize distributed system
    distributed_isa = DistributedAgentISA(config)
    await distributed_isa.initialize()
    
    # Initialize message queue
    message_queue = MessageQueue(config)
    await message_queue.initialize()
    
    # Initialize async processor
    processor = AsyncInstructionProcessor(
        distributed_isa.agent_isa,
        message_queue
    )
    await processor.initialize()
    
    # Initialize cache and state manager
    redis_client = redis.Redis(
        host=config["redis_host"],
        port=config["redis_port"]
    )
    cache = DistributedCache(redis_client)
    state_manager = StateManager(cache)
    
    # Initialize load balancer and auto-scaler
    load_balancer = LoadBalancer(distributed_isa.nodes)
    auto_scaler = AutoScaler(load_balancer, config)
    
    # Start FastAPI application
    app = FastAPI()
    
    @app.post("/execute")
    async def execute_instruction(request: Dict[str, Any]):
        task_id = await processor.submit_instruction(
            request["instruction_id"],
            request["params"]
        )
        return {"task_id": task_id}
    
    @app.get("/status/{task_id}")
    async def get_task_status(task_id: str):
        result = await cache.get(f"result:{task_id}")
        return result or {"status": "pending"}
    
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    asyncio.run(main())
```

## Configuration Example

```yaml
# config.yaml
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
  - name: node2
    type: PROCESSOR
    node_address: http://localhost:8002
    redis_host: localhost
    redis_port: 6379

isa_config:
  enable_learning: true
  enable_optimization: true
  min_pattern_support: 2
  min_success_rate: 0.8

scaling:
  cpu_threshold_up: 0.8
  memory_threshold_up: 0.8
  cpu_threshold_down: 0.3
  memory_threshold_down: 0.3
  min_nodes: 2
  max_nodes: 10
```

## Next Steps

1. Implement container orchestration
2. Set up service mesh
3. Configure distributed tracing
4. Implement disaster recovery
5. Set up monitoring and alerting

Check out the deployment guide for detailed information about deploying your distributed system. 