"""Unit tests for the capability metric extractors (the seventeen
``_measure_*`` functions of ``CapabilityProfile``).

Section 3.5 step 1 specifies that domain metrics are computed from the
agent's instruction set.  These tests pin the extraction semantics
against real instruction metadata — declared input/output counts,
category labels, cost hints, resource footprints, optimization and
security affordances — in three shapes:

* prevalence measures: the fraction of instructions carrying a given
  declared category or affordance;
* complexity measures: declared input arity, saturated at a constant;
* economy measures: declared resource/cost footprint, inverted.

Contract tests pin the invariants every extractor must keep: values in
[0, 1], deterministic, read-only over the instruction list, and
conservative (toward 0) when the relevant metadata is absent.
"""

import math

import pytest

from agentflow.core.isa.capability import CapabilityProfile
from agentflow.core.isa.formal import (
    FormalInstruction,
    InstructionType,
    OptimizationHint,
    ResourceRequirement,
    SecurityConstraint,
)


def make_profile(specializations=None) -> CapabilityProfile:
    return CapabilityProfile(
        {"specializations": specializations or ["general"]}
    )


def instr(**overrides) -> FormalInstruction:
    """Build a FormalInstruction with test-friendly defaults."""
    base = dict(name="step", type=InstructionType.CONTROL, metadata={})
    base.update(overrides)
    return FormalInstruction(**base)


MEASURE_NAMES = [
    "_measure_logical_consistency",
    "_measure_causal_understanding",
    "_measure_abstract_reasoning",
    "_measure_working_memory",
    "_measure_long_term_memory",
    "_measure_creativity",
    "_measure_coherence",
    "_measure_efficiency",
    "_measure_resource_usage",
    "_measure_domain_expertise",
    "_measure_adaptability",
    "_measure_communication",
    "_measure_collaboration",
    "_measure_threat_detection",
    "_measure_policy_compliance",
    "_measure_error_handling",
    "_measure_recovery",
]


# ----------------------------------------------------------------------
# Reasoning metrics
# ----------------------------------------------------------------------
class TestLogicalConsistency:
    def test_well_formed_instructions_score_one(self):
        profile = make_profile()
        value = profile._measure_logical_consistency(
            [instr(), instr(name="other", metadata={"outputs": ["r"]})]
        )
        assert value == 1.0

    def test_non_positive_timeout_breaks_consistency(self):
        profile = make_profile()
        # Five structural checks; one failing check scores 4/5.
        value = profile._measure_logical_consistency(
            [instr(timeout=0.0)]
        )
        assert value == pytest.approx(0.8)

    def test_negative_retries_breaks_consistency(self):
        profile = make_profile()
        value = profile._measure_logical_consistency(
            [instr(max_retries=-1)]
        )
        assert value == pytest.approx(0.8)

    def test_self_dependency_breaks_consistency(self):
        profile = make_profile()
        value = profile._measure_logical_consistency(
            [instr(name="loop", dependencies=["loop"])]
        )
        assert value == pytest.approx(0.8)

    def test_duplicate_dependencies_break_consistency(self):
        profile = make_profile()
        value = profile._measure_logical_consistency(
            [instr(dependencies=["a", "a"])]
        )
        assert value == pytest.approx(0.8)

    def test_duplicate_declared_outputs_break_consistency(self):
        profile = make_profile()
        value = profile._measure_logical_consistency(
            [instr(metadata={"outputs": ["r", "r"]})]
        )
        assert value == pytest.approx(0.8)


class TestCausalUnderstanding:
    def test_fraction_of_instructions_with_declared_dependencies(self):
        profile = make_profile()
        value = profile._measure_causal_understanding(
            [instr(dependencies=["x"]), instr()]
        )
        assert value == 0.5

    def test_all_dependencies_declared_scores_one(self):
        profile = make_profile()
        value = profile._measure_causal_understanding(
            [instr(dependencies=["x"]), instr(dependencies=["y"])]
        )
        assert value == 1.0

    def test_no_dependencies_scores_zero(self):
        profile = make_profile()
        assert profile._measure_causal_understanding([instr()]) == 0.0


class TestAbstractReasoning:
    def test_fraction_with_declared_parameter_schemas(self):
        profile = make_profile()
        value = profile._measure_abstract_reasoning(
            [instr(params={"prompt": "text"}), instr()]
        )
        assert value == 0.5

    def test_formal_conditions_count_as_abstraction(self):
        profile = make_profile()
        value = profile._measure_abstract_reasoning(
            [instr(postconditions=[lambda ctx, result: True])]
        )
        assert value == 1.0

    def test_no_abstraction_scores_zero(self):
        profile = make_profile()
        assert profile._measure_abstract_reasoning([instr()]) == 0.0


# ----------------------------------------------------------------------
# Memory metrics
# ----------------------------------------------------------------------
class TestWorkingMemory:
    def test_declared_inputs_saturate_at_four(self):
        profile = make_profile()
        full = profile._measure_working_memory(
            [instr(metadata={"inputs": ["a", "b", "c", "d"]})]
        )
        half = profile._measure_working_memory(
            [instr(metadata={"inputs": ["a", "b"]})]
        )
        capped = profile._measure_working_memory(
            [instr(metadata={"inputs": ["a", "b", "c", "d", "e", "f", "g", "h"]})]
        )
        assert full == 1.0
        assert half == 0.5
        assert capped == 1.0

    def test_declared_params_fall_back_to_input_arity(self):
        profile = make_profile()
        value = profile._measure_working_memory(
            [instr(parameters={"a": 1, "b": 2})]
        )
        assert value == 0.5

    def test_no_declared_inputs_scores_zero(self):
        profile = make_profile()
        assert profile._measure_working_memory([instr()]) == 0.0

    def test_non_list_inputs_metadata_is_ignored(self):
        profile = make_profile()
        value = profile._measure_working_memory(
            [instr(metadata={"inputs": "not-a-list"})]
        )
        assert value == 0.0


class TestLongTermMemory:
    def test_state_type_instructions_count(self):
        profile = make_profile()
        value = profile._measure_long_term_memory(
            [instr(type=InstructionType.STATE)]
        )
        assert value == 1.0

    def test_cacheable_optimization_hint_counts(self):
        profile = make_profile()
        value = profile._measure_long_term_memory(
            [instr(optimization=OptimizationHint(cacheable=True))]
        )
        assert value == 1.0

    def test_declared_storage_resource_counts(self):
        profile = make_profile()
        value = profile._measure_long_term_memory(
            [instr(resources=ResourceRequirement(storage=2.0))]
        )
        assert value == 1.0

    def test_mixed_list_scores_half(self):
        profile = make_profile()
        value = profile._measure_long_term_memory(
            [instr(type=InstructionType.STATE), instr()]
        )
        assert value == 0.5


# ----------------------------------------------------------------------
# Generation metrics
# ----------------------------------------------------------------------
class TestCreativity:
    def test_fraction_of_llm_instructions(self):
        profile = make_profile()
        value = profile._measure_creativity(
            [instr(type=InstructionType.LLM), instr()]
        )
        assert value == 0.5

    def test_all_llm_scores_one(self):
        profile = make_profile()
        value = profile._measure_creativity(
            [instr(type=InstructionType.LLM)]
        )
        assert value == 1.0

    def test_no_llm_scores_zero(self):
        profile = make_profile()
        assert profile._measure_creativity([instr()]) == 0.0


class TestCoherence:
    def test_fraction_declaring_outputs(self):
        profile = make_profile()
        value = profile._measure_coherence(
            [instr(metadata={"outputs": ["r"]}), instr()]
        )
        assert value == 0.5

    def test_output_prefixed_dependencies_count(self):
        profile = make_profile()
        value = profile._measure_coherence(
            [instr(dependencies=["output:r"])]
        )
        assert value == 1.0

    def test_no_declared_outputs_scores_zero(self):
        profile = make_profile()
        assert profile._measure_coherence([instr()]) == 0.0


# ----------------------------------------------------------------------
# Optimization metrics
# ----------------------------------------------------------------------
class TestEfficiency:
    def test_parallelizable_hint_counts(self):
        profile = make_profile()
        value = profile._measure_efficiency(
            [instr(optimization=OptimizationHint(parallelizable=True))]
        )
        assert value == 1.0

    def test_is_parallel_metadata_counts(self):
        # Real key emitted by the compiler's parallel wrapping pass.
        profile = make_profile()
        value = profile._measure_efficiency(
            [instr(metadata={"is_parallel": True})]
        )
        assert value == 1.0

    def test_cacheable_hint_counts(self):
        profile = make_profile()
        value = profile._measure_efficiency(
            [instr(optimization=OptimizationHint(cacheable=True))]
        )
        assert value == 1.0

    def test_no_optimization_hints_scores_zero(self):
        profile = make_profile()
        assert profile._measure_efficiency([instr()]) == 0.0


class TestResourceUsage:
    def test_no_declared_footprint_is_fully_frugal(self):
        profile = make_profile()
        assert profile._measure_resource_usage([instr()]) == 1.0

    def test_cost_hint_reduces_score(self):
        profile = make_profile()
        value = profile._measure_resource_usage(
            [instr(metadata={"cost": 1.0})]
        )
        assert value == pytest.approx(0.5)

    def test_resource_requirement_reduces_score(self):
        profile = make_profile()
        value = profile._measure_resource_usage(
            [instr(resources=ResourceRequirement(cpu=3.0))]
        )
        assert value == pytest.approx(0.25)

    def test_mixed_footprints_average(self):
        profile = make_profile()
        value = profile._measure_resource_usage(
            [instr(), instr(metadata={"cost": 1.0})]
        )
        assert value == pytest.approx(0.75)

    def test_non_numeric_cost_hint_is_ignored(self):
        profile = make_profile()
        value = profile._measure_resource_usage(
            [instr(metadata={"cost": "expensive"})]
        )
        assert value == 1.0


# ----------------------------------------------------------------------
# Specialization metrics
# ----------------------------------------------------------------------
class TestDomainExpertise:
    def test_declared_specialization_covered_by_tags(self):
        profile = make_profile(specializations=["nlp"])
        value = profile._measure_domain_expertise(
            [instr(metadata={"tags": ["nlp"]})]
        )
        assert value == 1.0

    def test_partial_coverage_scores_half(self):
        profile = make_profile(specializations=["nlp", "cv"])
        value = profile._measure_domain_expertise(
            [instr(metadata={"tags": ["nlp"]})]
        )
        assert value == 0.5

    def test_category_key_matches(self):
        profile = make_profile(specializations=["reasoning"])
        value = profile._measure_domain_expertise(
            [instr(metadata={"category": "reasoning"})]
        )
        assert value == 1.0

    def test_instruction_type_is_a_category_label(self):
        profile = make_profile(specializations=["llm"])
        value = profile._measure_domain_expertise(
            [instr(type=InstructionType.LLM)]
        )
        assert value == 1.0

    def test_uncovered_specializations_score_zero(self):
        profile = make_profile(specializations=["nlp"])
        assert profile._measure_domain_expertise([instr()]) == 0.0

    def test_no_declared_specializations_scores_zero(self):
        profile = make_profile(specializations=[])
        value = profile._measure_domain_expertise(
            [instr(metadata={"tags": ["nlp"]})]
        )
        assert value == 0.0


class TestAdaptability:
    def test_adaptive_type_counts(self):
        profile = make_profile()
        value = profile._measure_adaptability(
            [instr(type=InstructionType.ADAPTIVE)]
        )
        assert value == 1.0

    def test_declared_deadline_counts(self):
        profile = make_profile()
        value = profile._measure_adaptability(
            [instr(optimization=OptimizationHint(deadline=5.0))]
        )
        assert value == 1.0

    def test_declared_locality_counts(self):
        profile = make_profile()
        value = profile._measure_adaptability(
            [instr(optimization=OptimizationHint(locality="node-1"))]
        )
        assert value == 1.0

    def test_adaptive_metadata_counts(self):
        profile = make_profile()
        value = profile._measure_adaptability(
            [instr(metadata={"adaptive": True})]
        )
        assert value == 1.0

    def test_no_adaptation_affordances_scores_zero(self):
        profile = make_profile()
        assert profile._measure_adaptability([instr()]) == 0.0


# ----------------------------------------------------------------------
# Coordination metrics
# ----------------------------------------------------------------------
class TestCommunication:
    def test_agent_and_distributed_types_count(self):
        profile = make_profile()
        value = profile._measure_communication(
            [
                instr(type=InstructionType.AGENT),
                instr(type=InstructionType.DISTRIBUTED),
                instr(),
            ]
        )
        assert value == pytest.approx(2 / 3)

    def test_no_coordination_types_scores_zero(self):
        profile = make_profile()
        assert profile._measure_communication([instr()]) == 0.0


class TestCollaboration:
    def test_producer_and_consumer_roles_count(self):
        profile = make_profile()
        value = profile._measure_collaboration(
            [instr(dependencies=["x"], metadata={"outputs": ["y"]})]
        )
        assert value == 1.0

    def test_declared_inputs_substitute_for_dependencies(self):
        profile = make_profile()
        value = profile._measure_collaboration(
            [instr(metadata={"inputs": ["x"], "outputs": ["y"]})]
        )
        assert value == 1.0

    def test_producer_only_scores_zero(self):
        profile = make_profile()
        value = profile._measure_collaboration(
            [instr(metadata={"outputs": ["y"]})]
        )
        assert value == 0.0

    def test_consumer_only_scores_zero(self):
        profile = make_profile()
        value = profile._measure_collaboration(
            [instr(dependencies=["x"])]
        )
        assert value == 0.0


# ----------------------------------------------------------------------
# Security metrics
# ----------------------------------------------------------------------
class TestThreatDetection:
    def test_security_type_counts(self):
        profile = make_profile()
        value = profile._measure_threat_detection(
            [instr(type=InstructionType.SECURITY)]
        )
        assert value == 1.0

    def test_verification_flag_counts(self):
        profile = make_profile()
        value = profile._measure_threat_detection(
            [instr(security=SecurityConstraint(
                access_level="default", permissions=set(), verification=True
            ))]
        )
        assert value == 1.0

    def test_audit_flag_counts(self):
        profile = make_profile()
        value = profile._measure_threat_detection(
            [instr(security=SecurityConstraint(
                access_level="default", permissions=set(), audit=True
            ))]
        )
        assert value == 1.0

    def test_no_security_affordances_scores_zero(self):
        profile = make_profile()
        assert profile._measure_threat_detection([instr()]) == 0.0


class TestPolicyCompliance:
    def test_declared_permissions_count(self):
        profile = make_profile()
        value = profile._measure_policy_compliance(
            [instr(security=SecurityConstraint(
                access_level="default", permissions={"read"}
            ))]
        )
        assert value == 1.0

    def test_explicit_access_level_counts(self):
        profile = make_profile()
        value = profile._measure_policy_compliance(
            [instr(security=SecurityConstraint(
                access_level="restricted", permissions=set()
            ))]
        )
        assert value == 1.0

    def test_encryption_flag_counts(self):
        profile = make_profile()
        value = profile._measure_policy_compliance(
            [instr(security=SecurityConstraint(
                access_level="default", permissions=set(), encryption=True
            ))]
        )
        assert value == 1.0

    def test_default_security_constraints_score_zero(self):
        profile = make_profile()
        assert profile._measure_policy_compliance([instr()]) == 0.0


# ----------------------------------------------------------------------
# Robustness metrics
# ----------------------------------------------------------------------
class TestErrorHandling:
    def test_default_timeout_retry_contract_scores_one(self):
        profile = make_profile()
        assert profile._measure_error_handling([instr()]) == 1.0

    def test_zero_timeout_halves_score(self):
        profile = make_profile()
        value = profile._measure_error_handling([instr(timeout=0.0)])
        assert value == 0.5

    def test_zero_retries_halves_score(self):
        profile = make_profile()
        value = profile._measure_error_handling([instr(max_retries=0)])
        assert value == 0.5

    def test_no_error_handling_affordances_scores_zero(self):
        profile = make_profile()
        value = profile._measure_error_handling(
            [instr(timeout=0.0, max_retries=0)]
        )
        assert value == 0.0


class TestRecovery:
    def test_postconditions_count(self):
        profile = make_profile()
        value = profile._measure_recovery(
            [instr(postconditions=[lambda ctx, result: True])]
        )
        assert value == 1.0

    def test_invariants_count(self):
        profile = make_profile()
        value = profile._measure_recovery(
            [instr(invariants=[lambda ctx: True])]
        )
        assert value == 1.0

    def test_recovery_metadata_keys_count(self):
        profile = make_profile()
        value = profile._measure_recovery(
            [instr(metadata={"fallback": "retry"})]
        )
        assert value == 1.0

    def test_no_recovery_structure_scores_zero(self):
        profile = make_profile()
        assert profile._measure_recovery([instr()]) == 0.0


# ----------------------------------------------------------------------
# Extraction contracts
# ----------------------------------------------------------------------
class TestExtractionContracts:
    def test_empty_instruction_list_scores_zero_everywhere(self):
        profile = make_profile()
        for name in MEASURE_NAMES:
            assert getattr(profile, name)([]) == 0.0, name

    def test_all_values_stay_in_unit_interval(self):
        profile = make_profile(specializations=["llm"])
        rich = [
            instr(
                name="rich",
                type=InstructionType.LLM,
                params={"prompt": "text"},
                metadata={
                    "inputs": ["a", "b"],
                    "outputs": ["plan"],
                    "cost": 2.0,
                    "adaptive": True,
                },
                dependencies=["a"],
                optimization=OptimizationHint(parallelizable=True),
                security=SecurityConstraint(
                    access_level="restricted", permissions={"read"}
                ),
                resources=ResourceRequirement(cpu=0.5),
                postconditions=[lambda ctx, result: True],
            ),
            instr(name="plain"),
        ]
        for name in MEASURE_NAMES:
            value = getattr(profile, name)(rich)
            assert 0.0 <= value <= 1.0, f"{name}: {value}"

    def test_extraction_is_deterministic(self):
        profile = make_profile(specializations=["llm"])
        rich = [instr(type=InstructionType.LLM, metadata={"outputs": ["r"]})]
        for name in MEASURE_NAMES:
            first = getattr(profile, name)(rich)
            second = getattr(profile, name)(rich)
            assert first == second, name

    def test_extraction_does_not_mutate_instructions(self):
        profile = make_profile(specializations=["llm"])
        target = instr(
            type=InstructionType.LLM,
            metadata={"inputs": ["a"], "outputs": ["r"]},
            dependencies=["a"],
        )
        before_dump = target.model_dump()
        before_details = (
            dict(target.params),
            list(target.dependencies),
            list(target.postconditions),
            list(target.invariants),
            dict(target.metadata),
        )
        for name in MEASURE_NAMES:
            getattr(profile, name)([target])
        assert target.model_dump() == before_dump
        assert (dict(target.params), list(target.dependencies),
                list(target.postconditions), list(target.invariants),
                dict(target.metadata)) == before_details

    def test_assess_capabilities_produces_real_scores_end_to_end(self):
        profile = make_profile()
        rich = instr(
            name="rich",
            type=InstructionType.AGENT,
            params={"prompt": "text"},
            metadata={"outputs": ["plan"], "inputs": ["goal"]},
            dependencies=["goal"],
            postconditions=[lambda ctx, result: True],
        )
        results = profile.assess_capabilities([rich])

        # Domain scores follow from the pinned confidences in
        # capability.py (metric construction) and the extraction
        # semantics above.  reasoning: consistency, causality and
        # abstraction all fully declared -> 1.0.
        assert results["reasoning"]["score"] == pytest.approx(1.0)
        # memory: working_memory = 1 declared input / 4 (saturation),
        # long_term_memory = 0 -> (0.25*0.9 + 0*0.85) / 1.75.
        assert results["memory"]["score"] == pytest.approx(0.225 / 1.75)
        # generation: creativity = 0 (AGENT is not generative),
        # coherence = 1 -> (0*0.8 + 1.0*0.9) / 1.7.
        assert results["generation"]["score"] == pytest.approx(0.9 / 1.7)
        # coordination: communication and collaboration both 1 -> 1.0.
        assert results["coordination"]["score"] == pytest.approx(1.0)
        # robustness: error handling contract and postcondition-based
        # recovery both present -> 1.0.
        assert results["robustness"]["score"] == pytest.approx(1.0)
        # Declared-inputs profile keeps security/specialization at 0
        # (conservative when no affordances are declared).
        assert results["security"]["score"] == 0.0
        assert results["specialization"]["score"] == 0.0
        # Overall: weights applied exactly once, top level only.
        expected = (
            0.2 * 1.0                       # reasoning
            + 0.15 * (0.225 / 1.75)         # memory
            + 0.15 * (0.9 / 1.7)            # generation
            + 0.1 * (0.85 / 1.75)           # optimization (resource frugality)
            + 0.1 * 0.0                     # specialization
            + 0.1 * 1.0                     # coordination
            + 0.1 * 0.0                     # security
            + 0.1 * 1.0                     # robustness
        )
        assert profile.get_overall_score() == pytest.approx(expected)
        # The per-domain DomainScore cache mirrors the results.
        for domain, assessment in results.items():
            cached = profile.capabilities[domain]
            assert cached.score == pytest.approx(assessment["score"])
