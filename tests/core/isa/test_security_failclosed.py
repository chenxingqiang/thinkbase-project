"""Fail-closed (strict) security mode test suite.

Implements the hardening note of ``docs/AgentISA.md`` Section 6.2 as a
functional matrix: with ``strict=False`` (the default) the
:class:`SecurityMonitor` keeps its historical fail-open posture, while
``strict=True`` denies — and records — unregistered resources, unknown
data sensitivity levels and incomplete resource-access policies.
"""

import time

import pytest

from agentflow.core.isa.coordination import (
    AgentCoordinator,
    CoordinationRole,
    SecurityError,
    SecurityLevel,
    SecurityMonitor,
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
    """Build a task payload that passes the fail-open checks."""
    task = {
        "id": task_id,
        "authentication": "token-abc",
        "authorization": "role:executor",
        "requirements": {"capabilities": {}},
    }
    task.update(overrides)
    return task


def make_monitor(config=None, strict: bool = False) -> SecurityMonitor:
    """Build a SecurityMonitor in the requested enforcement mode."""
    return SecurityMonitor(config or {}, strict=strict)


# ----------------------------------------------------------------------
# Functional matrix: fail-open default vs strict (fail-closed)
# ----------------------------------------------------------------------
class TestModeMatrix:
    """Same scenario through both enforcement modes.

    Each test holds the "other" security dimension at a known-good
    value so exactly one dimension varies per scenario.
    """

    # -- resource-access dimension (sensitivity declared known) -------

    def test_registered_resource_allowed_in_both_modes(self):
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        config = {"allowed_resources": {"gpu_cluster"}}
        assert make_monitor(config).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor(config, strict=True).validate_task(task, SecurityLevel.LOW) is True

    def test_resource_outside_declared_registry_denied_in_both_modes(self):
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        config = {"allowed_resources": {"s3_bucket"}}
        assert make_monitor(config).validate_task(task, SecurityLevel.LOW) is False
        assert make_monitor(config, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_resource_request_without_registry(self):
        # THE fail-open hole: no allowed_resources declared at all.
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_no_resource_request_without_registry_allowed_in_both_modes(self):
        # Nothing requested -> no registry decision needed, even strict.
        task = make_task(data_sensitivity=0)
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is True

    def test_empty_resource_list_without_registry_allowed_in_both_modes(self):
        task = make_task(resources=[], data_sensitivity=0)
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is True

    def test_string_resource_request_registered_allowed_in_both_modes(self):
        task = make_task(resources="gpu_cluster", data_sensitivity=0)
        config = {"allowed_resources": {"gpu_cluster"}}
        assert make_monitor(config).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor(config, strict=True).validate_task(task, SecurityLevel.LOW) is True

    def test_malformed_registry_denied_with_distinct_reasons(self):
        # A string registry degenerates to a char-set; both modes deny
        # but strict mode names the incomplete policy explicitly.
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        config = {"allowed_resources": "gpu_cluster"}
        fail_open = make_monitor(config)
        strict = make_monitor(config, strict=True)
        assert fail_open.validate_task(task, SecurityLevel.LOW) is False
        assert strict.validate_task(task, SecurityLevel.LOW) is False
        assert "incomplete" in strict.violation_history[0]["violation"].lower()

    # -- data-sensitivity dimension (no resources requested) ----------

    def test_declared_sensitivity_within_requirement_allowed_in_both_modes(self):
        task = make_task(data_sensitivity=1)
        assert make_monitor({}).validate_task(task, SecurityLevel.MEDIUM) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.MEDIUM) is True

    def test_declared_sensitivity_above_requirement_denied_in_both_modes(self):
        task = make_task(data_sensitivity=2)
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is False
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_undeclared_sensitivity(self):
        # THE fail-open hole: unknown sensitivity silently reads as 0.
        task = make_task()
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_empty_category_list_is_not_a_declaration(self):
        task = make_task(data_categories=[])
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_out_of_range_negative_sensitivity(self):
        # Fail-open bypass: -1 never exceeds any required level.
        task = make_task(data_sensitivity=-1)
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_out_of_range_high_sensitivity(self):
        task = make_task(data_sensitivity=7)
        assert make_monitor({}).validate_task(task, SecurityLevel.CRITICAL) is False
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.CRITICAL) is False

    def test_non_numeric_sensitivity_declaration(self):
        task = make_task(data_sensitivity="high")
        assert make_monitor({}).validate_task(task, SecurityLevel.LOW) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.LOW) is False

    def test_sensitive_category_inferred_level_in_both_modes(self):
        task = make_task(data_categories=["financial"])
        assert make_monitor({}).validate_task(task, SecurityLevel.MEDIUM) is False
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.MEDIUM) is False
        assert make_monitor({}).validate_task(task, SecurityLevel.HIGH) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.HIGH) is True

    def test_benign_category_inferred_level_in_both_modes(self):
        task = make_task(data_categories=["public_reports"])
        assert make_monitor({}).validate_task(task, SecurityLevel.MEDIUM) is True
        assert make_monitor({}, strict=True).validate_task(task, SecurityLevel.MEDIUM) is True


# ----------------------------------------------------------------------
# Strict mode: every denial is recorded with a distinguishing reason
# ----------------------------------------------------------------------
class TestStrictDenialsAreRecorded:
    def test_unregistered_resource_recorded(self):
        before = time.time()
        monitor = make_monitor({"allowed_resources": {"gpu_cluster"}}, strict=True)
        task = make_task(resources=["s3_bucket"], data_sensitivity=0)
        assert monitor.validate_task(task, SecurityLevel.LOW) is False
        assert len(monitor.violation_history) == 1
        record = monitor.violation_history[0]
        assert record["timestamp"] >= before
        assert "unregistered" in record["violation"].lower()
        assert "s3_bucket" in record["violation"]
        assert record["task"]["id"] == "task-1"

    def test_incomplete_policy_recorded(self):
        monitor = make_monitor({}, strict=True)
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        assert monitor.validate_task(task, SecurityLevel.LOW) is False
        record = monitor.violation_history[-1]
        assert "incomplete" in record["violation"].lower()
        assert "allowed_resources" in record["violation"]

    def test_unknown_sensitivity_recorded(self):
        before = time.time()
        monitor = make_monitor({}, strict=True)
        assert monitor.validate_task(make_task(), SecurityLevel.LOW) is False
        record = monitor.violation_history[-1]
        assert record["timestamp"] >= before
        assert "sensitivity" in record["violation"].lower()

    def test_out_of_range_sensitivity_recorded(self):
        monitor = make_monitor({}, strict=True)
        assert monitor.validate_task(make_task(data_sensitivity=-1), SecurityLevel.LOW) is False
        record = monitor.violation_history[-1]
        assert "sensitivity" in record["violation"].lower()

    def test_boolean_sensitivity_rejected_in_strict_mode(self):
        # bool is an int subclass; strict mode must not coerce True -> 1.
        monitor = make_monitor({}, strict=True)
        assert monitor.validate_task(make_task(data_sensitivity=True), SecurityLevel.LOW) is False
        assert "sensitivity" in monitor.violation_history[-1]["violation"].lower()


# ----------------------------------------------------------------------
# Strict mode: allow path and the always-on checks
# ----------------------------------------------------------------------
class TestStrictAllowPath:
    def test_compliant_task_passes_without_violations(self):
        monitor = make_monitor({"allowed_resources": {"gpu_cluster"}}, strict=True)
        task = make_task(resources=["gpu_cluster"], data_sensitivity=1)
        assert monitor.validate_task(task, SecurityLevel.MEDIUM) is True
        assert monitor.violation_history == []

    def test_authentication_and_authorization_still_enforced(self):
        monitor = make_monitor({}, strict=True)
        no_auth = make_task(data_sensitivity=0)
        del no_auth["authentication"]
        assert monitor.validate_task(no_auth, SecurityLevel.LOW) is False
        no_authz = make_task(data_sensitivity=0)
        del no_authz["authorization"]
        assert monitor.validate_task(no_authz, SecurityLevel.LOW) is False


# ----------------------------------------------------------------------
# Backward compatibility of the flag
# ----------------------------------------------------------------------
class TestDefaultIsFailOpen:
    def test_flag_defaults_to_false(self):
        assert SecurityMonitor({}).strict is False

    def test_historical_constructor_signature_still_fail_open(self):
        monitor = SecurityMonitor({})
        task = make_task(resources=["gpu_cluster", "s3_bucket"], data_sensitivity=0)
        assert monitor.validate_task(task, SecurityLevel.LOW) is True


# ----------------------------------------------------------------------
# Coordinator propagation
# ----------------------------------------------------------------------
class TestCoordinatorStrictPropagation:
    @pytest.fixture
    def strict_coordinator(self):
        """Coordinator with strict security and three registered agents."""
        coord = AgentCoordinator({}, security_strict=True)
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

    def test_flag_defaults_to_false_on_coordinator(self):
        assert AgentCoordinator({}).security_monitor.strict is False

    def test_flag_reaches_security_monitor(self):
        coord = AgentCoordinator({}, security_strict=True)
        assert coord.security_monitor.strict is True

    def test_strict_coordinator_rejects_unclassified_task(self, strict_coordinator):
        with pytest.raises(SecurityError):
            strict_coordinator.coordinate_task(make_task(), SecurityLevel.LOW)
        violations = strict_coordinator.security_monitor.violation_history
        assert len(violations) == 1
        assert "sensitivity" in violations[0]["violation"].lower()

    def test_strict_coordinator_rejects_resource_request_without_registry(self, strict_coordinator):
        task = make_task(resources=["gpu_cluster"], data_sensitivity=0)
        with pytest.raises(SecurityError):
            strict_coordinator.coordinate_task(task, SecurityLevel.LOW)
        violations = strict_coordinator.security_monitor.violation_history
        assert len(violations) == 1
        assert "incomplete" in violations[0]["violation"].lower()

    def test_strict_coordinator_executes_compliant_task(self, strict_coordinator):
        task = make_task(data_sensitivity=0)
        result = strict_coordinator.coordinate_task(task, SecurityLevel.LOW)
        assert result["validation"]["status"] == "completed"
        assert strict_coordinator.security_monitor.violation_history == []
