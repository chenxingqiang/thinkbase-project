"""Agent collaboration instruction implementations."""

from typing import Dict, Any, List, Optional, Set, Union
from dataclasses import dataclass, field
import asyncio
import logging
import json

from .collaborative import (
    CollaborativeInstruction,
    CollaborationConfig,
    DataSharingInstruction,
    ConsensusInstruction
)
from .base import (
    InstructionMetadata,
    ExecutionContext,
    ExecutionResult
)

logger = logging.getLogger(__name__)

@dataclass
class AgentCollaborationMetrics:
    """Metrics for agent collaboration."""
    interaction_count: int = 0
    message_count: int = 0
    consensus_rounds: int = 0
    collaboration_duration: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)

class AgentTeamFormationInstruction(CollaborativeInstruction):
    """Instruction for forming agent teams."""
    
    def __init__(self,
                 name: str,
                 required_roles: Dict[str, int],
                 team_size_range: tuple[int, int],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            collab_config=CollaborationConfig(
                required_capabilities=set(required_roles.keys()),
                min_collaborators=team_size_range[0],
                max_collaborators=team_size_range[1]
            ),
            metadata=metadata,
            description="Form agent teams based on roles and capabilities"
        )
        self.required_roles = required_roles
        self.team_size_range = team_size_range
    
    async def _execute_task(self, collab_id: str,
                           context: ExecutionContext) -> ExecutionResult:
        """Form agent teams."""
        collab = self.active_collaborations[collab_id]
        
        try:
            # Share agent capabilities
            sharing_instruction = DataSharingInstruction(
                name=f"{self.name}_capability_sharing",
                collab_config=self.collab_config
            )
            
            sharing_result = await sharing_instruction._execute_task(
                collab_id,
                ExecutionContext(
                    params={"data": context.params.get("capabilities", {})},
                    metadata=context.metadata
                )
            )
            
            if not sharing_result.success:
                return sharing_result
            
            # Form teams based on capabilities
            teams = self._form_teams(sharing_result.output)
            
            if not teams:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Could not form valid teams"}]
                )
            
            return ExecutionResult(
                success=True,
                output=teams,
                state_updates={"teams": teams}
            )
            
        except Exception as e:
            logger.error(f"Team formation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}]
            )
    
    def _form_teams(self, agent_capabilities: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Form teams based on agent capabilities."""
        teams = []
        available_agents = set(agent_capabilities.keys())
        
        while available_agents and len(teams) < self.team_size_range[1]:
            team = self._form_single_team(agent_capabilities, available_agents)
            if team:
                teams.append(team)
                available_agents -= set(team["members"].keys())
            else:
                break
        
        return teams
    
    def _form_single_team(self, agent_capabilities: Dict[str, Dict[str, Any]],
                         available_agents: Set[str]) -> Optional[Dict[str, Any]]:
        """Form a single team."""
        team = {"members": {}, "roles": {}}
        unfilled_roles = dict(self.required_roles)
        
        # Try to fill each role
        for role, count in self.required_roles.items():
            suitable_agents = [
                agent_id for agent_id in available_agents
                if role in agent_capabilities[agent_id].get("capabilities", [])
            ]
            
            if len(suitable_agents) < count:
                return None
            
            # Assign agents to roles
            for _ in range(count):
                agent_id = suitable_agents.pop()
                team["members"][agent_id] = agent_capabilities[agent_id]
                team["roles"][role] = team["roles"].get(role, []) + [agent_id]
        
        return team if len(team["members"]) >= self.team_size_range[0] else None

class AgentTaskAllocationInstruction(CollaborativeInstruction):
    """Instruction for allocating tasks between agents."""
    
    def __init__(self,
                 name: str,
                 task_requirements: Dict[str, Dict[str, Any]],
                 allocation_strategy: str = "capability_based",
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            collab_config=CollaborationConfig(
                required_capabilities=self._extract_required_capabilities(task_requirements),
                min_collaborators=1
            ),
            metadata=metadata,
            description="Allocate tasks between collaborative agents"
        )
        self.task_requirements = task_requirements
        self.allocation_strategy = allocation_strategy
    
    def _extract_required_capabilities(self,
                                    task_requirements: Dict[str, Dict[str, Any]]) -> Set[str]:
        """Extract required capabilities from task requirements."""
        capabilities = set()
        for task in task_requirements.values():
            capabilities.update(task.get("required_capabilities", []))
        return capabilities
    
    async def _execute_task(self, collab_id: str,
                           context: ExecutionContext) -> ExecutionResult:
        """Allocate tasks between agents."""
        collab = self.active_collaborations[collab_id]
        
        try:
            # Share agent capabilities and preferences
            sharing_instruction = DataSharingInstruction(
                name=f"{self.name}_agent_info_sharing",
                collab_config=self.collab_config
            )
            
            sharing_result = await sharing_instruction._execute_task(
                collab_id,
                ExecutionContext(
                    params={
                        "data": {
                            "capabilities": context.params.get("capabilities", {}),
                            "preferences": context.params.get("preferences", {})
                        }
                    },
                    metadata=context.metadata
                )
            )
            
            if not sharing_result.success:
                return sharing_result
            
            # Allocate tasks
            if self.allocation_strategy == "capability_based":
                allocation = self._allocate_by_capabilities(
                    sharing_result.output,
                    self.task_requirements
                )
            elif self.allocation_strategy == "preference_based":
                allocation = self._allocate_by_preferences(
                    sharing_result.output,
                    self.task_requirements
                )
            else:
                allocation = self._allocate_round_robin(
                    sharing_result.output,
                    self.task_requirements
                )
            
            if not allocation:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Could not allocate tasks"}]
                )
            
            # Reach consensus on allocation
            consensus_instruction = ConsensusInstruction(
                name=f"{self.name}_allocation_consensus",
                collab_config=self.collab_config
            )
            
            consensus_result = await consensus_instruction._execute_task(
                collab_id,
                ExecutionContext(
                    params={"proposal": allocation},
                    metadata=context.metadata
                )
            )
            
            if not consensus_result.success:
                return consensus_result
            
            return ExecutionResult(
                success=True,
                output=allocation,
                state_updates={"task_allocation": allocation}
            )
            
        except Exception as e:
            logger.error(f"Task allocation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}]
            )
    
    def _allocate_by_capabilities(self,
                                agent_info: Dict[str, Dict[str, Any]],
                                task_requirements: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Allocate tasks based on agent capabilities."""
        allocation = {"tasks": {}, "agents": {}}
        
        # Sort tasks by complexity
        sorted_tasks = sorted(
            task_requirements.items(),
            key=lambda x: len(x[1].get("required_capabilities", [])),
            reverse=True
        )
        
        # Allocate each task
        for task_id, task in sorted_tasks:
            required_capabilities = set(task.get("required_capabilities", []))
            
            # Find best agent
            best_agent = None
            best_score = -1
            
            for agent_id, info in agent_info.items():
                if agent_id in allocation["agents"]:
                    continue
                
                agent_capabilities = set(info.get("capabilities", {}).keys())
                if required_capabilities.issubset(agent_capabilities):
                    score = len(agent_capabilities & required_capabilities)
                    if score > best_score:
                        best_agent = agent_id
                        best_score = score
            
            if best_agent:
                allocation["tasks"][task_id] = best_agent
                allocation["agents"][best_agent] = task_id
            else:
                return None
        
        return allocation
    
    def _allocate_by_preferences(self,
                               agent_info: Dict[str, Dict[str, Any]],
                               task_requirements: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Allocate tasks based on agent preferences."""
        allocation = {"tasks": {}, "agents": {}}
        
        # Create preference matrix
        preferences = {}
        for agent_id, info in agent_info.items():
            agent_prefs = info.get("preferences", {})
            agent_capabilities = set(info.get("capabilities", {}).keys())
            
            preferences[agent_id] = {}
            for task_id, task in task_requirements.items():
                required_capabilities = set(task.get("required_capabilities", []))
                if required_capabilities.issubset(agent_capabilities):
                    preferences[agent_id][task_id] = agent_prefs.get(task_id, 0)
                else:
                    preferences[agent_id][task_id] = -1
        
        # Allocate tasks using stable marriage algorithm
        unallocated_tasks = list(task_requirements.keys())
        task_preferences = {task_id: [] for task_id in task_requirements}
        
        while unallocated_tasks:
            task_id = unallocated_tasks[0]
            
            # Find best available agent
            best_agent = None
            best_pref = -1
            
            for agent_id, prefs in preferences.items():
                if agent_id not in allocation["agents"]:
                    pref = prefs[task_id]
                    if pref > best_pref:
                        best_agent = agent_id
                        best_pref = pref
            
            if best_agent and best_pref >= 0:
                allocation["tasks"][task_id] = best_agent
                allocation["agents"][best_agent] = task_id
                unallocated_tasks.remove(task_id)
            else:
                return None
        
        return allocation
    
    def _allocate_round_robin(self,
                            agent_info: Dict[str, Dict[str, Any]],
                            task_requirements: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Allocate tasks using round-robin strategy."""
        allocation = {"tasks": {}, "agents": {}}
        
        # Get list of capable agents for each task
        task_agents = {}
        for task_id, task in task_requirements.items():
            required_capabilities = set(task.get("required_capabilities", []))
            capable_agents = [
                agent_id for agent_id, info in agent_info.items()
                if required_capabilities.issubset(set(info.get("capabilities", {}).keys()))
            ]
            task_agents[task_id] = capable_agents
        
        # Allocate tasks round-robin
        agent_loads = {agent_id: 0 for agent_id in agent_info}
        
        for task_id, capable_agents in task_agents.items():
            if not capable_agents:
                return None
            
            # Find least loaded capable agent
            best_agent = min(
                capable_agents,
                key=lambda x: agent_loads[x]
            )
            
            allocation["tasks"][task_id] = best_agent
            allocation["agents"][best_agent] = task_id
            agent_loads[best_agent] += 1
        
        return allocation

class AgentCommunicationInstruction(CollaborativeInstruction):
    """Instruction for structured communication between agents."""
    
    def __init__(self,
                 name: str,
                 communication_protocol: str,
                 message_schema: Dict[str, Any],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            collab_config=CollaborationConfig(
                required_capabilities={"communication"},
                min_collaborators=2
            ),
            metadata=metadata,
            description="Structured communication between agents"
        )
        self.communication_protocol = communication_protocol
        self.message_schema = message_schema
    
    async def _execute_task(self, collab_id: str,
                           context: ExecutionContext) -> ExecutionResult:
        """Execute communication task."""
        collab = self.active_collaborations[collab_id]
        
        try:
            # Validate message
            message = context.params.get("message", {})
            if not self._validate_message(message):
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Invalid message format"}]
                )
            
            # Share message
            sharing_instruction = DataSharingInstruction(
                name=f"{self.name}_message_sharing",
                collab_config=self.collab_config
            )
            
            sharing_result = await sharing_instruction._execute_task(
                collab_id,
                ExecutionContext(
                    params={"data": message},
                    metadata=context.metadata
                )
            )
            
            if not sharing_result.success:
                return sharing_result
            
            # Process messages according to protocol
            if self.communication_protocol == "broadcast":
                result = self._handle_broadcast(sharing_result.output)
            elif self.communication_protocol == "request_response":
                result = self._handle_request_response(sharing_result.output)
            elif self.communication_protocol == "publish_subscribe":
                result = self._handle_publish_subscribe(sharing_result.output)
            else:
                result = sharing_result.output
            
            return ExecutionResult(
                success=True,
                output=result,
                state_updates={"messages": result}
            )
            
        except Exception as e:
            logger.error(f"Communication failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}]
            )
    
    def _validate_message(self, message: Dict[str, Any]) -> bool:
        """Validate message against schema."""
        try:
            # Implement schema validation
            return True
        except Exception:
            return False
    
    def _handle_broadcast(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Handle broadcast communication."""
        return {
            "type": "broadcast",
            "messages": messages
        }
    
    def _handle_request_response(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Handle request-response communication."""
        requests = {}
        responses = {}
        
        for agent_id, message in messages.items():
            if message.get("type") == "request":
                requests[agent_id] = message
            elif message.get("type") == "response":
                responses[agent_id] = message
        
        return {
            "type": "request_response",
            "requests": requests,
            "responses": responses
        }
    
    def _handle_publish_subscribe(self, messages: Dict[str, Any]) -> Dict[str, Any]:
        """Handle publish-subscribe communication."""
        topics = {}
        
        for agent_id, message in messages.items():
            topic = message.get("topic")
            if topic:
                if topic not in topics:
                    topics[topic] = []
                topics[topic].append(message)
        
        return {
            "type": "publish_subscribe",
            "topics": topics
        } 