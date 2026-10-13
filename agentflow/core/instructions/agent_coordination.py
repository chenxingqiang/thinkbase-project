"""Agent coordination instruction implementations."""

from typing import Dict, Any, List, Optional, Set, Union, cast
from dataclasses import dataclass, field
import asyncio
import logging
import json
import time


from .base import (
    BaseInstruction,
    InstructionType,
    InstructionMetadata,
    ExecutionContext,
    ExecutionResult
)

logger = logging.getLogger(__name__)

@dataclass
class CoordinationMetrics:
    """Metrics for agent coordination."""
    coordination_attempts: int = 0
    successful_coordinations: int = 0
    failed_coordinations: int = 0
    average_coordination_time: float = 0.0
    resource_utilization: float = 0.0

@dataclass
class CoordinationConfig:
    """Configuration for agent coordination."""
    max_participants: int = 10
    timeout_seconds: float = 300.0
    min_success_rate: float = 0.8
    resource_threshold: float = 0.9
    coordination_mode: str = "synchronous"  # synchronous or asynchronous

class AgentSynchronizationInstruction(BaseInstruction):
    """Instruction for synchronizing agent activities."""
    
    def __init__(self,
                 name: str,
                 coordination_config: CoordinationConfig,
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
        )
        self.coordination_config = coordination_config
        self.metrics = CoordinationMetrics()
        self.sync_points: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute synchronization."""
        try:
            # Get synchronization parameters
            sync_id = cast(str, context.params.get("sync_id"))
            agent_id = cast(str, context.params.get("agent_id"))
            
            if not sync_id or not agent_id:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Missing sync_id or agent_id"}]
                )
            
            # Initialize sync point if needed
            if sync_id not in self.sync_points:
                self.sync_points[sync_id] = {
                    "participants": set(),
                    "ready": set(),
                    "data": {},
                    "start_time": time.time()
                }
            
            # Register participant
            sync_point = self.sync_points[sync_id]
            sync_point["participants"].add(agent_id)
            
            # Wait for all participants to be ready
            await self._wait_for_sync(sync_id, agent_id, context)
            
            # Update metrics
            self._update_metrics(sync_id)
            
            return ExecutionResult(
                success=True,
                output={"sync_data": sync_point["data"]},
                state_updates={"sync_status": "completed"},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Synchronization failed: {str(e)}")
            self.metrics.failed_coordinations += 1
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _wait_for_sync(self, sync_id: str, agent_id: str,
                            context: ExecutionContext):
        """Wait for synchronization.

        The barrier width is ``context.params["expected_participants"]``
        when provided, otherwise ``coordination_config.max_participants``.
        Agents are released once every expected participant is ready.
        """
        sync_point = self.sync_points[sync_id]
        sync_point["ready"].add(agent_id)
        
        # Share agent data
        sync_point["data"][agent_id] = context.params.get("sync_data", {})
        
        expected = int(context.params.get(
            "expected_participants", self.coordination_config.max_participants
        ))
        
        start_time = time.time()
        while len(sync_point["ready"]) < expected:
            if time.time() - start_time > self.coordination_config.timeout_seconds:
                raise TimeoutError("Synchronization timeout")
            await asyncio.sleep(0.05)
    
    def _update_metrics(self, sync_id: str):
        """Update coordination metrics."""
        sync_point = self.sync_points[sync_id]
        self.metrics.coordination_attempts += 1
        self.metrics.successful_coordinations += 1
        
        # Update average coordination time
        coordination_time = time.time() - sync_point["start_time"]
        current_total = (
            self.metrics.average_coordination_time *
            (self.metrics.successful_coordinations - 1)
        )
        self.metrics.average_coordination_time = (
            (current_total + coordination_time) /
            self.metrics.successful_coordinations
        )

class ResourceCoordinationInstruction(BaseInstruction):
    """Instruction for coordinating resource usage between agents."""
    
    def __init__(self,
                 name: str,
                 coordination_config: CoordinationConfig,
                 resource_limits: Dict[str, float],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
        )
        self.coordination_config = coordination_config
        self.resource_limits = resource_limits
        self.metrics = CoordinationMetrics()
        self.resource_allocations: Dict[str, Dict[str, float]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute resource coordination."""
        try:
            # Get resource request
            agent_id = cast(str, context.params.get("agent_id"))
            resource_request = context.params.get("resource_request", {})
            
            if not agent_id or not resource_request:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Missing agent_id or resource_request"}]
                )
            
            # Validate and allocate resources
            allocation = await self._allocate_resources(agent_id, resource_request)
            
            # Update metrics
            self._update_metrics(allocation)
            
            return ExecutionResult(
                success=True,
                output={"allocation": allocation},
                state_updates={"resource_allocation": self.resource_allocations},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Resource coordination failed: {str(e)}")
            self.metrics.failed_coordinations += 1
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _allocate_resources(self, agent_id: str,
                                request: Dict[str, float]) -> Dict[str, float]:
        """Allocate resources to agent."""
        # Calculate current utilization
        current_usage = self._calculate_usage()
        
        # Check if request can be satisfied
        for resource, amount in request.items():
            if resource not in self.resource_limits:
                raise ValueError(f"Unknown resource: {resource}")
            
            available = self.resource_limits[resource] - current_usage.get(resource, 0.0)
            if amount > available:
                raise ValueError(f"Insufficient {resource}: requested {amount}, available {available}")
        
        # Allocate resources
        self.resource_allocations[agent_id] = request
        return request
    
    def _calculate_usage(self) -> Dict[str, float]:
        """Calculate current resource usage."""
        usage = {r: 0.0 for r in self.resource_limits}
        
        for allocation in self.resource_allocations.values():
            for resource, amount in allocation.items():
                usage[resource] += amount
        
        return usage
    
    def _update_metrics(self, allocation: Dict[str, float]):
        """Update coordination metrics."""
        self.metrics.coordination_attempts += 1
        self.metrics.successful_coordinations += 1
        
        # Calculate resource utilization
        usage = self._calculate_usage()
        total_utilization = sum(
            usage[r] / self.resource_limits[r]
            for r in self.resource_limits
        ) / len(self.resource_limits)
        
        self.metrics.resource_utilization = total_utilization

class WorkflowCoordinationInstruction(BaseInstruction):
    """Instruction for coordinating agent workflows."""
    
    def __init__(self,
                 name: str,
                 coordination_config: CoordinationConfig,
                 workflow_schema: Dict[str, Any],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            description="Coordinate agent workflows"
        )
        self.workflow_schema = workflow_schema
        self.metrics = CoordinationMetrics()
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute workflow coordination."""
        try:
            # Get workflow parameters
            workflow_id = cast(str, context.params.get("workflow_id"))
            agent_id = cast(str, context.params.get("agent_id"))
            action = cast(str, context.params.get("action"))
            
            if not all([workflow_id, agent_id, action]):
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Missing workflow parameters"}]
                )
            
            # Process workflow action
            result = await self._process_workflow_action(
                workflow_id,
                agent_id,
                action,
                context.params.get("action_data", {})
            )
            
            # Update metrics
            self._update_metrics(result)
            
            return ExecutionResult(
                success=True,
                output=result,
                state_updates={"workflow_state": self.active_workflows},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Workflow coordination failed: {str(e)}")
            self.metrics.failed_coordinations += 1
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _process_workflow_action(self, workflow_id: str,
                                     agent_id: str,
                                     action: str,
                                     action_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process workflow action."""
        # Initialize workflow if needed
        if workflow_id not in self.active_workflows:
            self.active_workflows[workflow_id] = {
                "state": "initialized",
                "participants": set(),
                "actions": [],
                "data": {},
                "start_time": time.time()
            }
        
        workflow = self.active_workflows[workflow_id]
        workflow["participants"].add(agent_id)
        
        # Validate action
        if not self._validate_workflow_action(workflow, action, action_data):
            raise ValueError(f"Invalid workflow action: {action}")
        
        # Process action
        if action == "start":
            workflow["state"] = "running"
        elif action == "complete":
            workflow["state"] = "completed"
            workflow["end_time"] = time.time()
        elif action == "update":
            workflow["data"].update(action_data)
            # Keep a per-agent record so concurrent updates from
            # different agents do not overwrite each other's
            # contributions in the shared data view.
            per_agent = workflow["data"].setdefault("_agent_updates", {})
            per_agent[agent_id] = action_data
        
        # Record action
        workflow["actions"].append({
            "agent_id": agent_id,
            "action": action,
            "data": action_data,
            "timestamp": time.time()
        })
        
        return {
            "workflow_id": workflow_id,
            "state": workflow["state"],
            "data": workflow["data"]
        }
    
    def _validate_workflow_action(self, workflow: Dict[str, Any],
                                action: str,
                                action_data: Dict[str, Any]) -> bool:
        """Validate workflow action."""
        # Check if action is allowed in current state
        current_state = workflow["state"]
        
        if action == "start" and current_state != "initialized":
            return False
        if action == "complete" and current_state != "running":
            return False
        if action == "update" and current_state not in ["running", "initialized"]:
            return False
        
        # Validate action data against schema
        if action == "update" and "data_schema" in self.workflow_schema:
            return self._validate_data(action_data, self.workflow_schema["data_schema"])
        
        return True
    
    def _validate_data(self, data: Dict[str, Any],
                      schema: Dict[str, Any]) -> bool:
        """Validate data against schema."""
        # Implement schema validation
        return True
    
    def _update_metrics(self, result: Dict[str, Any]):
        """Update coordination metrics.

        Every successfully processed action counts as a successful
        coordination; the average-coordination-time statistic is only
        advanced when a workflow reaches the ``completed`` state (the
        only point at which a meaningful end-to-end duration exists).
        """
        self.metrics.coordination_attempts += 1
        self.metrics.successful_coordinations += 1
        
        if result.get("state") == "completed":
            # Update average coordination time
            workflow = self.active_workflows[result["workflow_id"]]
            coordination_time = workflow["end_time"] - workflow["start_time"]
            current_total = (
                self.metrics.average_coordination_time *
                (self.metrics.successful_coordinations - 1)
            )
            self.metrics.average_coordination_time = (
                (current_total + coordination_time) /
                self.metrics.successful_coordinations
            )