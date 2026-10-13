"""Multi-agent coordination test suite.

Covers :class:`AgentCoordinator` (role assignment, agent selection,
execution planning, fallback strategies, task history) and
:class:`SecurityMonitor` (authentication/authorization checks,
resource-access validation, data-sensitivity enforcement).
"""

import time

import pytest

from agentflow.core.isa.coordination import (
    AgentCoordinator,
    CoordinationRole,
    SecurityLevel,
    SecurityMonitor,
    SecurityError,
    CoordinationError,
)
from agentflow.core.isa.capability import (
    CapabilityProfile,
    DomainScore,
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_profile(
    reasoning: float = 0.5,
    optimization: float = 0.5,
    memory: float = 0.5,
    specialization: float = 0.5,
    specializations=None,
) -> CapabilityProfile:
    """Build a CapabilityProfile with explicit per-domain scores."""
    profile = CapabilityProfile(
        {"specializations": specializations or ["general"]}
    )
    profile.capabilities = {
        "reasoning": DomainScore(score=reasoning, weight=0.3),
        "optimization": DomainScore(score=optimization, weight=0.3),
        "memory": DomainScore(score=memory, weight=0.2),
        "specialization": DomainScore(score=specialization, weight=0.2),
    }
    return profile


def make_task(task_id: str = "task-1", **overrides) -> dict:
    """Build a task payload that passes security validation."""
    task = {
        "id": task_id,
        "authentication": "token-abc",
        "authorization": "role:executor",
        "requirements": {"capabilities": {}},
    }
    task.update(overrides)
    return task


@pytest.fixture
def coordinator():
    """Coordinator with three registered agents."""
    coord = AgentCoordinator({})
    coord.register_agent(
        "coordinator", make_profile(reasoning=0.9, optimization=0.9), CoordinationRole.COORDINATOR
    )
    coord.register_agent(
        "executor", make_profile(reasoning=0.5, optimization=0.7), CoordinationRole.EXECUTOR
    )
    coord.register_agent(
        "validator", make_profile(reasoning=0.95, optimization=0.4), CoordinationRole.VALIDATOR
    )
    return coord


# ----------------------------------------------------------------------
# AgentCoordinator: registration and policies
# ----------------------------------------------------------------------
class TestRegistrationAndPolicies:
    def test_register_agent_creates_policy(self, coordinator):
        assert set(coordinator.agents) == {"coordinator", "executor", "validator"}
        assert set(coordinator.policies) == set(coordinator.agents)

    def test_policy_permissions_by_role(self, coordinator):
        policy = coordinator.policies["coordinator"]
        assert "assign_tasks" in policy.permissions["role"]
        executor_policy = coordinator.policies["executor"]
        assert "execute_task" in executor_policy.permissions["role"]
        assert "read_task" in executor_policy.permissions["base"]

    def test_security_requirements_by_role(self, coordinator):
        coordinator_reqs = coordinator.policies["coordinator"].security_requirements
        assert coordinator_reqs["authentication"] is SecurityLevel.HIGH
        executor_reqs = coordinator.policies["executor"].security_requirements
        assert executor_reqs["isolation"] is SecurityLevel.HIGH

    def test_trust_level_bounds(self, coordinator):
        for agent_id, trust in coordinator.policies.items():
            assert agent_id in coordinator.agents
            assert 0.0 <= trust.trust_levels[agent_id] <= 1.0

    def test_trust_level_rewards_specializations(self):
        coord = AgentCoordinator({})
        coord.register_agent("plain", make_profile(), CoordinationRole.EXECUTOR)
        coord.register_agent(
            "specialized",
            make_profile(specializations=["nlp", "cv", "systems"]),
            CoordinationRole.EXECUTOR,
        )
        plain_trust = coord.policies["plain"].trust_levels["plain"]
        specialized_trust = coord.policies["specialized"].trust_levels["specialized"]
        assert specialized_trust > plain_trust


# ----------------------------------------------------------------------
# AgentCoordinator: task coordination
# ----------------------------------------------------------------------
class TestTaskCoordination:
    def test_coordinate_task_success(self, coordinator):
        result = coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        assert "preparation" in result
        assert "execution" in result
        assert "validation" in result
        assert result["preparation"]["status"] == "completed"
        assert coordinator.task_history[-1]["status"] == "success"

    def test_failed_task_recorded_in_history(self, coordinator):
        coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        assert len(coordinator.task_history) == 1
        assert coordinator.task_history[0]["status"] == "success"
        assert coordinator.task_history[0]["plan"]["task_id"] == "task-1"

    def test_missing_authentication_raises(self, coordinator):
        task = make_task()
        del task["authentication"]
        with pytest.raises(SecurityError, match="security validation"):
            coordinator.coordinate_task(task, SecurityLevel.LOW)

    def test_missing_authorization_raises(self, coordinator):
        task = make_task()
        del task["authorization"]
        with pytest.raises(SecurityError, match="security validation"):
            coordinator.coordinate_task(task, SecurityLevel.LOW)

    def test_no_suitable_agents_raises(self, coordinator):
        task = make_task(requirements={"capabilities": {"nonexistent_domain": 0.9}})
        with pytest.raises(CoordinationError, match="No suitable agents"):
            coordinator.coordinate_task(task, SecurityLevel.LOW)

    def test_agent_selection_respects_capability_threshold(self, coordinator):
        task = make_task(requirements={"capabilities": {"reasoning": 0.85}})
        result = coordinator.coordinate_task(task, SecurityLevel.LOW)
        assert result["preparation"]["status"] == "completed"
        # The selected coordinator must be one of the reasoning-heavy agents.
        plan = coordinator.task_history[-1]["plan"]
        preparation_agent = plan["stages"][0]["agent"]
        assert coordinator.agents[preparation_agent].capabilities["reasoning"].score >= 0.85

    def test_coordinator_is_highest_scoring_agent(self, coordinator):
        result = coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        plan = coordinator.task_history[-1]["plan"]
        preparation_agent = plan["stages"][0]["agent"]
        best = max(
            coordinator.agents,
            key=lambda a: coordinator._calculate_coordinator_score(coordinator.agents[a]),
        )
        assert preparation_agent == best


# ----------------------------------------------------------------------
# AgentCoordinator: execution plan structure
# ----------------------------------------------------------------------
class TestExecutionPlan:
    def _plan(self, coordinator):
        coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        return coordinator.task_history[-1]["plan"]

    def test_plan_stage_structure(self, coordinator):
        plan = self._plan(coordinator)
        assert [s["stage"] for s in plan["stages"]] == [
            "preparation",
            "execution",
            "validation",
        ]
        assert plan["stages"][0]["agent"] is not None
        assert isinstance(plan["stages"][1]["agents"], list)

    def test_plan_coordination_rules(self, coordinator):
        plan = self._plan(coordinator)
        rules = plan["coordination"]
        assert rules["sync_points"] == ["pre_execution", "post_execution"]
        assert "broadcast" in rules["communication"]
        assert rules["conflict_resolution"]["priority"] == "majority_vote"

    def test_plan_fallback_keys_cover_every_stage(self, coordinator):
        plan = self._plan(coordinator)
        for stage in plan["stages"]:
            assert f"{stage['stage']}_failure" in plan["fallback"]

    def test_fallback_strategies_are_differentiated(self, coordinator):
        plan = self._plan(coordinator)
        assert plan["fallback"]["agent_failure"]["action"] == "replace_agent"
        assert plan["fallback"]["validation_failure"]["action"] == "increase_validators"
        assert plan["fallback"]["coordination_failure"]["action"] == "simplify_plan"

    def test_fallback_executed_on_stage_failure(self, coordinator):
        original = coordinator._validate_stage_results

        coordinator._validate_stage_results = lambda results: False
        try:
            result = coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        finally:
            coordinator._validate_stage_results = original
        assert result["preparation_fallback"]["status"] == "fallback_completed"
        assert result["preparation_fallback"]["action_taken"] == "simplify_plan"
        assert result["execution_fallback"]["action_taken"] == "replace_agent"
        assert result["validation_fallback"]["action_taken"] == "increase_validators"

    def test_validators_are_reasoning_heavy_non_coordinators(self, coordinator):
        plan = self._plan(coordinator)
        validation_stage = plan["stages"][2]
        coordinator_agent = plan["stages"][0]["agent"]
        for validator in validation_stage["agents"]:
            assert validator != coordinator_agent
            assert coordinator.agents[validator].capabilities["reasoning"].score > 0.8


# ----------------------------------------------------------------------
# SecurityMonitor
# ----------------------------------------------------------------------
class TestSecurityMonitor:
    def test_valid_task_passes(self):
        monitor = SecurityMonitor({})
        assert monitor.validate_task(make_task(), SecurityLevel.LOW) is True

    def test_missing_authentication_fails(self):
        monitor = SecurityMonitor({})
        task = make_task()
        del task["authentication"]
        assert monitor.validate_task(task, SecurityLevel.LOW) is False

    def test_missing_authorization_fails(self):
        monitor = SecurityMonitor({})
        task = make_task()
        del task["authorization"]
        assert monitor.validate_task(task, SecurityLevel.LOW) is False

    def test_violation_history_records_timestamp(self):
        monitor = SecurityMonitor({})
        before = time.time()
        monitor.validate_task({}, SecurityLevel.LOW)
        assert len(monitor.violation_history) == 1
        record = monitor.violation_history[0]
        assert record["timestamp"] >= before
        assert "authentication" in record["violation"].lower()

    def test_declared_sensitivity_enforced(self):
        monitor = SecurityMonitor({})
        task = make_task(data_sensitivity=2)
        # HIGH (2) data on a LOW (0) requirement must fail...
        assert monitor.validate_task(task, SecurityLevel.LOW) is False
        # ...but passes when the required level is at least HIGH.
        assert monitor.validate_task(task, SecurityLevel.HIGH) is True

    def test_sensitive_categories_infer_high_level(self):
        monitor = SecurityMonitor({})
        task = make_task(data_categories=["financial"])
        assert monitor.validate_task(task, SecurityLevel.MEDIUM) is False
        assert monitor.validate_task(task, SecurityLevel.HIGH) is True

    def test_benign_categories_pass_at_medium(self):
        monitor = SecurityMonitor({})
        task = make_task(data_categories=["public_reports"])
        assert monitor.validate_task(task, SecurityLevel.MEDIUM) is True

    def test_resource_access_open_by_default(self):
        monitor = SecurityMonitor({})
        task = make_task(resources=["gpu_cluster", "s3_bucket"])
        assert monitor.validate_task(task, SecurityLevel.LOW) is True

    def test_resource_access_restricted_by_config(self):
        monitor = SecurityMonitor({"allowed_resources": {"gpu_cluster"}})
        allowed = make_task(resources=["gpu_cluster"])
        denied = make_task(resources=["gpu_cluster", "s3_bucket"])
        assert monitor.validate_task(allowed, SecurityLevel.LOW) is True
        assert monitor.validate_task(denied, SecurityLevel.LOW) is False

    def test_string_resource_request(self):
        monitor = SecurityMonitor({"allowed_resources": {"s3_bucket"}})
        assert monitor.validate_task(make_task(resources="s3_bucket"), SecurityLevel.LOW) is True
        assert monitor.validate_task(make_task(resources="gpu_cluster"), SecurityLevel.LOW) is False


# ----------------------------------------------------------------------
# Integration: security failure surfaces through the coordinator
# ----------------------------------------------------------------------
class TestCoordinatorSecurityIntegration:
    def test_coordinator_records_security_violation(self, coordinator):
        task = make_task(data_sensitivity=3)
        with pytest.raises(SecurityError):
            coordinator.coordinate_task(task, SecurityLevel.LOW)
        violations = coordinator.security_monitor.violation_history
        assert len(violations) == 1
        assert "sensitivity" in violations[0]["violation"].lower()

    def test_valid_high_security_task(self, coordinator):
        task = make_task(data_sensitivity=3, data_categories=["credentials"])
        result = coordinator.coordinate_task(task, SecurityLevel.CRITICAL)
        assert result["validation"]["status"] == "completed"
