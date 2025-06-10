"""Agent negotiation instruction implementations."""

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
    ExecutionResult,
    InstructionType
)

logger = logging.getLogger(__name__)

@dataclass
class NegotiationMetrics:
    """Metrics for negotiation process."""
    rounds: int = 0
    proposals: int = 0
    counterproposals: int = 0
    concessions: int = 0
    agreement_rate: float = 0.0
    negotiation_time: float = 0.0

@dataclass
class NegotiationConfig:
    """Configuration for negotiation."""
    max_rounds: int = 10
    timeout_per_round: float = 60.0
    min_agreement_rate: float = 0.7
    concession_rate: float = 0.1
    strategy: str = "cooperative"  # cooperative, competitive, or balanced

class AgentNegotiationInstruction(CollaborativeInstruction):
    """Base class for negotiation instructions."""
    
    def __init__(self,
                 name: str,
                 negotiation_config: NegotiationConfig,
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            collab_config=CollaborationConfig(
                required_capabilities={"negotiation"},
                min_collaborators=2
            ),
            metadata=metadata,
            description="Base negotiation instruction"
        )
        self.negotiation_config = negotiation_config
        self.metrics = NegotiationMetrics()
    
    async def _execute_task(self, collab_id: str,
                           context: ExecutionContext) -> ExecutionResult:
        """Execute negotiation task."""
        collab = self.active_collaborations[collab_id]
        
        try:
            # Initialize negotiation
            negotiation_state = await self._initialize_negotiation(collab_id, context)
            
            # Negotiate until agreement or max rounds
            while (self.metrics.rounds < self.negotiation_config.max_rounds and
                   self.metrics.agreement_rate < self.negotiation_config.min_agreement_rate):
                
                # Share proposals
                proposals = await self._share_proposals(collab_id, context, negotiation_state)
                
                # Evaluate proposals
                evaluation = await self._evaluate_proposals(proposals, negotiation_state)
                
                # Update negotiation state
                negotiation_state = await self._update_negotiation(evaluation, negotiation_state)
                
                # Update metrics
                self.metrics.rounds += 1
                self.metrics.agreement_rate = evaluation.get("agreement_rate", 0.0)
            
            # Check if agreement reached
            if self.metrics.agreement_rate >= self.negotiation_config.min_agreement_rate:
                return ExecutionResult(
                    success=True,
                    output=negotiation_state["agreement"],
                    state_updates={"negotiation_result": negotiation_state},
                    metrics=self.metrics.__dict__
                )
            else:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Failed to reach agreement"}],
                    metrics=self.metrics.__dict__
                )
            
        except Exception as e:
            logger.error(f"Negotiation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _initialize_negotiation(self, collab_id: str,
                                    context: ExecutionContext) -> Dict[str, Any]:
        """Initialize negotiation state."""
        return {
            "round": 0,
            "proposals": {},
            "evaluations": {},
            "agreement": None
        }
    
    async def _share_proposals(self, collab_id: str,
                             context: ExecutionContext,
                             state: Dict[str, Any]) -> Dict[str, Any]:
        """Share proposals between agents."""
        sharing_instruction = DataSharingInstruction(
            name=f"{self.name}_proposal_sharing",
            collab_config=self.collab_config
        )
        
        # Generate proposal
        proposal = await self._generate_proposal(state)
        self.metrics.proposals += 1
        
        # Share proposal
        sharing_result = await sharing_instruction._execute_task(
            collab_id,
            ExecutionContext(
                params={"data": proposal},
                metadata=context.metadata
            )
        )
        
        if not sharing_result.success:
            raise ValueError("Failed to share proposals")
        
        return sharing_result.output
    
    async def _generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a proposal based on negotiation state."""
        raise NotImplementedError
    
    async def _evaluate_proposals(self, proposals: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate proposals and calculate agreement rate."""
        raise NotImplementedError
    
    async def _update_negotiation(self, evaluation: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Update negotiation state based on evaluation."""
        raise NotImplementedError

class ResourceNegotiationInstruction(AgentNegotiationInstruction):
    """Instruction for negotiating resource allocation."""
    
    def __init__(self,
                 name: str,
                 resources: Dict[str, float],
                 requirements: Dict[str, Dict[str, float]],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            negotiation_config=NegotiationConfig(
                max_rounds=5,
                timeout_per_round=30.0,
                min_agreement_rate=0.8,
                strategy="balanced"
            ),
            metadata=metadata
        )
        self.resources = resources
        self.requirements = requirements
    
    async def _generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate resource allocation proposal."""
        round_num = state.get("round", 0)
        previous_proposals = state.get("proposals", {})
        
        # Calculate available resources
        available = dict(self.resources)
        for allocation in previous_proposals.values():
            for resource, amount in allocation.items():
                available[resource] = max(0, available[resource] - amount)
        
        # Generate proposal based on strategy
        if self.negotiation_config.strategy == "cooperative":
            return self._generate_cooperative_proposal(available, round_num)
        elif self.negotiation_config.strategy == "competitive":
            return self._generate_competitive_proposal(available, round_num)
        else:
            return self._generate_balanced_proposal(available, round_num)
    
    def _generate_cooperative_proposal(self, available: Dict[str, float],
                                     round_num: int) -> Dict[str, Any]:
        """Generate cooperative resource proposal."""
        proposal = {}
        for resource, amount in available.items():
            required = self.requirements.get(resource, {}).get("min", 0)
            proposal[resource] = min(required, amount)
        return proposal
    
    def _generate_competitive_proposal(self, available: Dict[str, float],
                                     round_num: int) -> Dict[str, Any]:
        """Generate competitive resource proposal."""
        proposal = {}
        for resource, amount in available.items():
            required = self.requirements.get(resource, {}).get("max", amount)
            proposal[resource] = min(required, amount)
        return proposal
    
    def _generate_balanced_proposal(self, available: Dict[str, float],
                                  round_num: int) -> Dict[str, Any]:
        """Generate balanced resource proposal."""
        proposal = {}
        concession = self.negotiation_config.concession_rate * round_num
        
        for resource, amount in available.items():
            min_req = self.requirements.get(resource, {}).get("min", 0)
            max_req = self.requirements.get(resource, {}).get("max", amount)
            target = max_req - (max_req - min_req) * concession
            proposal[resource] = min(target, amount)
            
        return proposal
    
    async def _evaluate_proposals(self, proposals: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate resource allocation proposals."""
        total_resources = {r: 0.0 for r in self.resources}
        valid_proposals = 0
        
        # Sum up resource requests
        for proposal in proposals.values():
            is_valid = True
            for resource, amount in proposal.items():
                if amount < 0 or amount > self.resources[resource]:
                    is_valid = False
                    break
                total_resources[resource] += amount
            
            if is_valid:
                valid_proposals += 1
        
        # Calculate agreement rate
        if not valid_proposals:
            return {"agreement_rate": 0.0}
        
        # Check resource constraints
        constraints_met = all(
            total <= self.resources[r] for r, total in total_resources.items()
        )
        
        # Calculate satisfaction rate
        satisfaction_rates = []
        for proposal in proposals.values():
            rates = []
            for resource, amount in proposal.items():
                min_req = self.requirements.get(resource, {}).get("min", 0)
                max_req = self.requirements.get(resource, {}).get("max", self.resources[resource])
                if max_req > min_req:
                    rate = (amount - min_req) / (max_req - min_req)
                    rates.append(max(0.0, min(1.0, rate)))
            
            if rates:
                satisfaction_rates.append(sum(rates) / len(rates))
        
        avg_satisfaction = sum(satisfaction_rates) / len(satisfaction_rates)
        
        return {
            "agreement_rate": avg_satisfaction if constraints_met else 0.0,
            "total_resources": total_resources,
            "satisfaction_rates": satisfaction_rates
        }
    
    async def _update_negotiation(self, evaluation: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Update resource negotiation state."""
        state["round"] += 1
        state["evaluations"][state["round"]] = evaluation
        
        if evaluation["agreement_rate"] >= self.negotiation_config.min_agreement_rate:
            state["agreement"] = {
                "resource_allocation": evaluation["total_resources"],
                "satisfaction_rates": evaluation["satisfaction_rates"],
                "agreement_rate": evaluation["agreement_rate"]
            }
        
        return state

class TaskNegotiationInstruction(AgentNegotiationInstruction):
    """Instruction for negotiating task assignments."""
    
    def __init__(self,
                 name: str,
                 tasks: Dict[str, Dict[str, Any]],
                 agent_preferences: Dict[str, Dict[str, float]],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            negotiation_config=NegotiationConfig(
                max_rounds=3,
                timeout_per_round=45.0,
                min_agreement_rate=0.75,
                strategy="cooperative"
            ),
            metadata=metadata
        )
        self.tasks = tasks
        self.agent_preferences = agent_preferences
    
    async def _generate_proposal(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generate task assignment proposal."""
        round_num = state.get("round", 0)
        previous_proposals = state.get("proposals", {})
        
        # Get available tasks
        available_tasks = set(self.tasks.keys())
        for proposal in previous_proposals.values():
            available_tasks -= set(proposal.get("assignments", {}).keys())
        
        # Generate assignments based on preferences
        assignments = {}
        preferences = self.agent_preferences.get(state.get("agent_id", ""), {})
        
        sorted_tasks = sorted(
            available_tasks,
            key=lambda t: preferences.get(t, 0.0),
            reverse=True
        )
        
        for task_id in sorted_tasks:
            task = self.tasks[task_id]
            if self._can_handle_task(task):
                assignments[task_id] = {
                    "agent_id": state.get("agent_id", ""),
                    "preference": preferences.get(task_id, 0.0)
                }
        
        return {
            "assignments": assignments,
            "round": round_num
        }
    
    def _can_handle_task(self, task: Dict[str, Any]) -> bool:
        """Check if agent can handle task."""
        # Implement task capability check
        return True
    
    async def _evaluate_proposals(self, proposals: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate task assignment proposals."""
        all_assignments = {}
        conflicts = set()
        
        # Collect all assignments and detect conflicts
        for agent_id, proposal in proposals.items():
            assignments = proposal.get("assignments", {})
            for task_id, assignment in assignments.items():
                if task_id in all_assignments:
                    conflicts.add(task_id)
                all_assignments[task_id] = assignment
        
        # Calculate agreement rate
        total_tasks = len(self.tasks)
        assigned_tasks = len(all_assignments) - len(conflicts)
        agreement_rate = assigned_tasks / total_tasks if total_tasks > 0 else 0.0
        
        # Calculate preference satisfaction
        satisfaction_rates = {}
        for agent_id, preferences in self.agent_preferences.items():
            assigned_prefs = []
            for task_id, assignment in all_assignments.items():
                if assignment["agent_id"] == agent_id and task_id not in conflicts:
                    assigned_prefs.append(preferences.get(task_id, 0.0))
            
            if assigned_prefs:
                satisfaction_rates[agent_id] = sum(assigned_prefs) / len(assigned_prefs)
            else:
                satisfaction_rates[agent_id] = 0.0
        
        return {
            "agreement_rate": agreement_rate,
            "assignments": {
                task_id: assignment
                for task_id, assignment in all_assignments.items()
                if task_id not in conflicts
            },
            "conflicts": list(conflicts),
            "satisfaction_rates": satisfaction_rates
        }
    
    async def _update_negotiation(self, evaluation: Dict[str, Any],
                                state: Dict[str, Any]) -> Dict[str, Any]:
        """Update task negotiation state."""
        state["round"] += 1
        state["evaluations"][state["round"]] = evaluation
        
        if evaluation["agreement_rate"] >= self.negotiation_config.min_agreement_rate:
            state["agreement"] = {
                "task_assignments": evaluation["assignments"],
                "satisfaction_rates": evaluation["satisfaction_rates"],
                "agreement_rate": evaluation["agreement_rate"]
            }
        
        return state 