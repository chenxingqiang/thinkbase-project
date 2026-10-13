"""Multi-agent Coordination and Security System."""
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import time
import numpy as np
from enum import Enum
from .formal import FormalInstruction, InstructionType
from .capability import CapabilityProfile

class CoordinationRole(Enum):
    """Agent roles in coordination."""
    COORDINATOR = "coordinator"
    EXECUTOR = "executor"
    VALIDATOR = "validator"
    MONITOR = "monitor"

class SecurityLevel(Enum):
    """Security levels for operations."""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class CoordinationPolicy:
    """Policy for multi-agent coordination."""
    roles: Dict[str, CoordinationRole]
    permissions: Dict[str, Set[str]]
    trust_levels: Dict[str, float]
    security_requirements: Dict[str, SecurityLevel]

class AgentCoordinator:
    """Manages multi-agent coordination."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.agents: Dict[str, CapabilityProfile] = {}
        self.policies: Dict[str, CoordinationPolicy] = {}
        self.task_history = []
        self.security_monitor = SecurityMonitor(config)
        
    def register_agent(self, 
                      agent_id: str,
                      profile: CapabilityProfile,
                      role: CoordinationRole):
        """Register an agent for coordination."""
        self.agents[agent_id] = profile
        policy = self._create_policy(agent_id, profile, role)
        self.policies[agent_id] = policy
        
    def coordinate_task(self,
                       task: Dict[str, Any],
                       security_level: SecurityLevel) -> Dict[str, Any]:
        """Coordinate task execution across agents."""
        # Validate security requirements
        if not self.security_monitor.validate_task(task, security_level):
            raise SecurityError("Task failed security validation")
        
        # Select agents for task
        selected_agents = self._select_agents(task)
        if not selected_agents:
            raise CoordinationError("No suitable agents found")
        
        # Create execution plan
        plan = self._create_execution_plan(task, selected_agents)
        
        # Execute with monitoring
        try:
            result = self._execute_plan(plan)
            self.task_history.append({
                "task": task,
                "plan": plan,
                "result": result,
                "status": "success"
            })
            return result
        except Exception as e:
            self.task_history.append({
                "task": task,
                "plan": plan,
                "error": str(e),
                "status": "failed"
            })
            raise
    
    def _create_policy(self,
                      agent_id: str,
                      profile: CapabilityProfile,
                      role: CoordinationRole) -> CoordinationPolicy:
        """Create coordination policy for agent."""
        return CoordinationPolicy(
            roles={agent_id: role},
            permissions=self._generate_permissions(role),
            trust_levels={agent_id: self._calculate_trust_level(profile)},
            security_requirements=self._generate_security_requirements(role)
        )
    
    def _select_agents(self, task: Dict[str, Any]) -> List[str]:
        """Select suitable agents for task."""
        requirements = task.get("requirements", {})
        selected = []
        
        for agent_id, profile in self.agents.items():
            if self._meets_requirements(profile, requirements):
                selected.append(agent_id)
                
        return selected
    
    def _create_execution_plan(self,
                             task: Dict[str, Any],
                             agents: List[str]) -> Dict[str, Any]:
        """Create detailed execution plan."""
        plan = {
            "task_id": task.get("id"),
            "stages": [],
            "coordination": {},
            "fallback": {}
        }
        
        # Assign roles
        coordinator = self._select_coordinator(agents)
        validators = self._select_validators(agents, coordinator)
        executors = [a for a in agents if a not in validators + [coordinator]]
        
        # Create stages
        plan["stages"] = [
            {
                "stage": "preparation",
                "agent": coordinator,
                "actions": ["validate_input", "distribute_resources"]
            },
            {
                "stage": "execution",
                "agents": executors,
                "actions": ["execute_task", "report_progress"]
            },
            {
                "stage": "validation",
                "agents": validators,
                "actions": ["validate_results", "verify_consistency"]
            }
        ]
        
        # Define coordination rules
        plan["coordination"] = {
            "sync_points": ["pre_execution", "post_execution"],
            "communication": self._create_communication_rules(agents),
            "conflict_resolution": self._create_conflict_rules()
        }
        
        # Define fallback strategies keyed by stage name (the stage
        # executor looks up "{stage}_failure") and by failure type for
        # direct programmatic access.
        plan["fallback"] = {
            "preparation_failure": self._create_fallback_strategy("coordination_failure"),
            "execution_failure": self._create_fallback_strategy("agent_failure"),
            "validation_failure": self._create_fallback_strategy("validation_failure"),
            "agent_failure": self._create_fallback_strategy("agent_failure"),
            "coordination_failure": self._create_fallback_strategy("coordination_failure"),
        }
        
        return plan
    
    def _execute_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute coordination plan."""
        results = {}
        
        # Execute each stage
        for stage in plan["stages"]:
            stage_results = self._execute_stage(stage, plan["coordination"])
            results[stage["stage"]] = stage_results
            
            # Check for failures and apply fallback if needed
            if not self._validate_stage_results(stage_results):
                fallback = plan["fallback"][f"{stage['stage']}_failure"]
                results[f"{stage['stage']}_fallback"] = self._execute_fallback(fallback)
        
        return results
    
    def _meets_requirements(self,
                          profile: CapabilityProfile,
                          requirements: Dict[str, Any]) -> bool:
        """Check if agent meets task requirements."""
        for cap_type, required_score in requirements.get("capabilities", {}).items():
            if cap_type not in profile.capabilities:
                return False
            if profile.capabilities[cap_type].score < required_score:
                return False
        return True
    
    def _select_coordinator(self, agents: List[str]) -> str:
        """Select best agent for coordinator role."""
        scores = []
        for agent_id in agents:
            profile = self.agents[agent_id]
            score = self._calculate_coordinator_score(profile)
            scores.append((score, agent_id))
        return max(scores, key=lambda x: x[0])[1]
    
    def _select_validators(self,
                         agents: List[str],
                         coordinator: str) -> List[str]:
        """Select agents for validation role."""
        return [
            agent_id for agent_id in agents
            if agent_id != coordinator and
            self._is_suitable_validator(self.agents[agent_id])
        ][:2]  # Select top 2 validators
    
    def _calculate_coordinator_score(self,
                                  profile: CapabilityProfile) -> float:
        """Calculate agent's suitability as coordinator."""
        weights = {
            "reasoning": 0.3,
            "optimization": 0.3,
            "memory": 0.2,
            "specialization": 0.2
        }
        
        score = 0.0
        for cap_type, weight in weights.items():
            if cap_type in profile.capabilities:
                score += profile.capabilities[cap_type].score * weight
        return score
    
    def _is_suitable_validator(self, profile: CapabilityProfile) -> bool:
        """Check if agent is suitable for validation role."""
        return (profile.get_overall_score() > 0.7 and
                "reasoning" in profile.capabilities and
                profile.capabilities["reasoning"].score > 0.8)
    
    @staticmethod
    def _create_communication_rules(agents: List[str]) -> Dict[str, Any]:
        """Create communication rules for agents."""
        return {
            "broadcast": ["status_update", "error_notification"],
            "direct": ["task_assignment", "result_submission"],
            "group": ["validation_consensus", "coordination_sync"]
        }
    
    @staticmethod
    def _create_conflict_rules() -> Dict[str, Any]:
        """Create conflict resolution rules."""
        return {
            "priority": "majority_vote",
            "timeout": 30,
            "retry_limit": 3,
            "escalation": ["warning", "abort", "restart"]
        }
    
    @staticmethod
    def _create_fallback_strategy(failure_type: str) -> Dict[str, Any]:
        """Create fallback strategy for failure type."""
        return {
            "agent_failure": {
                "action": "replace_agent",
                "max_retries": 2
            },
            "validation_failure": {
                "action": "increase_validators",
                "max_validators": 3
            },
            "coordination_failure": {
                "action": "simplify_plan",
                "min_agents": 2
            }
        }.get(failure_type, {"action": "abort"})
    
    def _execute_stage(self,
                      stage: Dict[str, Any],
                      coordination_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single stage of the plan."""
        # Simulate stage execution
        return {
            "status": "completed",
            "metrics": {
                "duration": 1.0,
                "resource_usage": 0.5
            }
        }
    
    def _validate_stage_results(self, results: Dict[str, Any]) -> bool:
        """Validate stage execution results."""
        return results.get("status") == "completed"
    
    def _execute_fallback(self, fallback: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback strategy."""
        # Simulate fallback execution
        return {
            "status": "fallback_completed",
            "action_taken": fallback["action"]
        }
    
    @staticmethod
    def _calculate_trust_level(profile: CapabilityProfile) -> float:
        """Calculate trust level for agent."""
        return min(
            1.0,
            profile.get_overall_score() * 0.7 +
            len(profile.specializations) * 0.1
        )
    
    @staticmethod
    def _generate_permissions(role: CoordinationRole) -> Dict[str, Set[str]]:
        """Generate permission set based on role."""
        base_permissions = {"read_task", "report_status"}
        role_permissions = {
            CoordinationRole.COORDINATOR: {
                "assign_tasks", "modify_plan", "approve_results"
            },
            CoordinationRole.EXECUTOR: {
                "execute_task", "access_resources"
            },
            CoordinationRole.VALIDATOR: {
                "validate_results", "request_verification"
            },
            CoordinationRole.MONITOR: {
                "monitor_execution", "raise_alerts"
            }
        }
        return {
            "base": base_permissions,
            "role": role_permissions.get(role, set())
        }
    
    @staticmethod
    def _generate_security_requirements(role: CoordinationRole) -> Dict[str, SecurityLevel]:
        """Generate security requirements based on role."""
        return {
            CoordinationRole.COORDINATOR: {
                "authentication": SecurityLevel.HIGH,
                "encryption": SecurityLevel.HIGH,
                "isolation": SecurityLevel.MEDIUM
            },
            CoordinationRole.EXECUTOR: {
                "authentication": SecurityLevel.MEDIUM,
                "encryption": SecurityLevel.MEDIUM,
                "isolation": SecurityLevel.HIGH
            },
            CoordinationRole.VALIDATOR: {
                "authentication": SecurityLevel.HIGH,
                "encryption": SecurityLevel.MEDIUM,
                "isolation": SecurityLevel.LOW
            },
            CoordinationRole.MONITOR: {
                "authentication": SecurityLevel.MEDIUM,
                "encryption": SecurityLevel.LOW,
                "isolation": SecurityLevel.LOW
            }
        }.get(role, {
            "authentication": SecurityLevel.LOW,
            "encryption": SecurityLevel.LOW,
            "isolation": SecurityLevel.LOW
        })

class SecurityMonitor:
    """Monitors and enforces security policies."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.security_policies = self._initialize_policies()
        self.violation_history = []
        
    def validate_task(self,
                     task: Dict[str, Any],
                     required_level: SecurityLevel) -> bool:
        """Validate task against security policies."""
        try:
            self._check_authentication(task)
            self._check_authorization(task)
            self._check_resource_access(task)
            self._check_data_sensitivity(task, required_level)
            return True
        except SecurityViolation as e:
            self.violation_history.append({
                "task": task,
                "violation": str(e),
                "timestamp": time.time()
            })
            return False
    
    def _initialize_policies(self) -> Dict[str, Any]:
        """Initialize security policies."""
        return {
            "authentication": {
                "required": True,
                "methods": ["token", "certificate"],
                "expiry": 3600
            },
            "authorization": {
                "role_based": True,
                "mandatory_access": True,
                "delegation": False
            },
            "resource_access": {
                "isolation": True,
                "rate_limiting": True,
                "monitoring": True
            },
            "data_protection": {
                "encryption": True,
                "masking": True,
                "retention": 30
            }
        }
    
    def _check_authentication(self, task: Dict[str, Any]):
        """Check task authentication."""
        if not task.get("authentication"):
            raise SecurityViolation("Missing authentication")
    
    def _check_authorization(self, task: Dict[str, Any]):
        """Check task authorization."""
        if not task.get("authorization"):
            raise SecurityViolation("Missing authorization")
    
    def _check_resource_access(self, task: Dict[str, Any]):
        """Check resource access permissions."""
        if not self._validate_resource_access(task):
            raise SecurityViolation("Invalid resource access")
    
    def _check_data_sensitivity(self,
                              task: Dict[str, Any],
                              required_level: SecurityLevel):
        """Check data sensitivity requirements."""
        if self._get_data_sensitivity(task) > required_level.value:
            raise SecurityViolation("Data sensitivity mismatch")
    
    def _validate_resource_access(self, task: Dict[str, Any]) -> bool:
        """Validate resource access permissions.

        When the coordinator config declares an ``allowed_resources``
        set, every resource requested by the task must fall inside it.
        Without such a declaration no restriction is enforced (open
        default, matching the historical behaviour).
        """
        allowed = self.config.get("allowed_resources")
        if allowed is None:
            return True
        allowed = set(allowed)
        requested = task.get("resources", [])
        if isinstance(requested, str):
            requested = [requested]
        return all(resource in allowed for resource in requested)

    def _get_data_sensitivity(self, task: Dict[str, Any]) -> int:
        """Get data sensitivity level.

        The task may declare sensitivity directly (``data_sensitivity``
        as an int 0-3) or reference classified categories via
        ``data_categories`` (e.g. ``["financial", "medical"]`` count as
        highly sensitive).  0 means no special sensitivity.
        """
        declared = task.get("data_sensitivity")
        if isinstance(declared, (int, float)):
            return int(declared)

        sensitive_categories = {"financial", "medical", "personal", "credentials"}
        categories = task.get("data_categories", [])
        if isinstance(categories, str):
            categories = [categories]
        if categories and set(categories) & sensitive_categories:
            return SecurityLevel.HIGH.value
        if categories:
            return SecurityLevel.MEDIUM.value
        return 0

class SecurityError(Exception):
    """Security-related error."""
    pass

class CoordinationError(Exception):
    """Coordination-related error."""
    pass

class SecurityViolation(Exception):
    """Security violation error."""
    pass
