"""End-to-end mechanics tests for the capability pipeline.

These tests pin down the pipeline mechanics of
``CapabilityProfile.assess_capabilities``:

1. ``assess_capabilities`` runs end-to-end over a real instruction
   list and produces a well-formed result for all eight domains, with
   metric values extracted from real instruction metadata by the
   seventeen ``_measure_*`` extractors (their unit-level semantics are
   pinned by ``tests/core/isa/test_capability_metrics.py``).
2. Domain aggregation is the confidence-weighted mean and applies the
   domain weight *nowhere* (regression test for the double-weighting
   fix: the weight must enter only in ``get_overall_score``).
3. The overall score is the weighted mean across domains, with the
   documented zero-weight fallback.
"""

import pytest

from agentflow.core.isa.capability import (
    CapabilityDomain,
    CapabilityMetric,
    CapabilityProfile,
    DomainScore,
)
from agentflow.core.isa.formal import FormalInstruction, InstructionType


def _instructions():
    return [
        FormalInstruction(name="llm_step", type=InstructionType.LLM,
                          metadata={"side_effects": False, "outputs": ["r"]}),
        FormalInstruction(name="state_step", type=InstructionType.STATE,
                          metadata={"side_effects": True}),
    ]


class TestPipelineMechanics:

    def test_assess_capabilities_runs_end_to_end(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        results = profile.assess_capabilities(_instructions())

        assert set(results.keys()) == {d.value for d in CapabilityDomain}
        for domain, assessment in results.items():
            # Every domain produces a non-empty metric list (the
            # metrics are real CapabilityMetric objects whose values
            # are extracted from instruction metadata).
            assert assessment["metrics"], f"{domain}: no metrics produced"
            assert assessment["weight"] >= 0.0
            # Real extraction: metric values are evidence-derived and
            # stay within the unit interval.
            assert all(0.0 <= m.value <= 1.0 for m in assessment["metrics"])

        # Spot-check the extraction against the fixture: both steps
        # are structurally well-formed (logical consistency 1.0) but
        # declare no dependencies or parameter schemas (causality and
        # abstraction 0.0).
        reasoning = {m.name: m.value for m in results["reasoning"]["metrics"]}
        assert reasoning["logical_consistency"] == 1.0
        assert reasoning["causal_understanding"] == 0.0
        # The STATE step drives long-term memory to 1/2.
        memory = {m.name: m.value for m in results["memory"]["metrics"]}
        assert memory["long_term_memory"] == 0.5
        # The LLM step drives creativity and output-declaring
        # coherence to 1/2.
        generation = {m.name: m.value for m in results["generation"]["metrics"]}
        assert generation["creativity"] == 0.5
        assert generation["coherence"] == 0.5

    def test_overall_score_reflects_real_extraction(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        profile.assess_capabilities(_instructions())
        # Weighted mean across domains, each domain weight applied
        # exactly once:
        #   reasoning     0.2  * 0.9/2.55    (consistency 1.0, rest 0)
        #   memory        0.15 * 0.425/1.75  (long-term 0.5, working 0)
        #   generation    0.15 * 0.5         (creativity/coherence 0.5)
        #   optimization  0.1  * 0.85/1.75   (frugal resources 1.0)
        #   specialization/coordination/security 0 (nothing declared)
        #   robustness    0.1  * 0.9/1.75   (error handling 1.0)
        expected = (
            0.2 * 0.9 / 2.55
            + 0.15 * 0.425 / 1.75
            + 0.15 * 0.5
            + 0.1 * 0.85 / 1.75
            + 0.1 * 0.9 / 1.75
        )
        assert profile.get_overall_score() == pytest.approx(expected)
        assert profile.get_overall_score() > 0.0


class TestDomainAggregation:

    def _metrics(self, values, confs):
        return [
            CapabilityMetric(name=f"m{i}", domain=CapabilityDomain.REASONING,
                             value=v, confidence=c, metadata={})
            for i, (v, c) in enumerate(zip(values, confs))
        ]

    def test_confidence_weighted_mean(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        score = profile._compute_weighted_score(
            self._metrics([1.0, 0.5], [0.8, 0.2]))
        assert score == pytest.approx((1.0 * 0.8 + 0.5 * 0.2) / (0.8 + 0.2))

    def test_empty_metric_set_scores_zero(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        assert profile._compute_weighted_score([]) == 0.0

    def test_zero_confidence_degrades_to_unweighted_mean(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        score = profile._compute_weighted_score(
            self._metrics([0.4, 0.8], [0.0, 0.0]))
        assert score == pytest.approx(0.6)

    def test_domain_weight_not_applied_in_aggregation(self):
        """Regression test for the double-weighting defect: the domain
        weight must NOT scale the per-domain score.  A weighted-mean
        formula that folded the weight in here would cap the overall
        score below any threshold once get_overall_score multiplied by
        the weights again."""
        profile = CapabilityProfile({"specializations": ["general"]})
        metrics = self._metrics([1.0, 1.0], [1.0, 1.0])
        assert profile._compute_weighted_score(metrics) == pytest.approx(1.0)


class TestOverallScore:

    def test_weighted_mean_across_domains(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        profile.capabilities = {
            "reasoning": DomainScore(score=0.6, weight=0.3),
            "memory": DomainScore(score=0.9, weight=0.7),
        }
        assert profile.get_overall_score() == pytest.approx(0.81)

    def test_zero_weight_falls_back_to_unweighted_mean(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        profile.capabilities = {
            "reasoning": DomainScore(score=0.6, weight=0.0),
            "memory": DomainScore(score=0.9, weight=0.0),
        }
        assert profile.get_overall_score() == pytest.approx(0.75)

    def test_empty_profile_scores_zero(self):
        profile = CapabilityProfile({"specializations": ["general"]})
        assert profile.get_overall_score() == 0.0
