"""Approval-gated auto-promotion pipeline: mining -> optimization (Section 3.3.1).

Covers the disclosed gap that no automated mining-to-optimization
channel existed -- improvement relied on manual decisions (main paper
Section 3.3.1: "no automated mining-to-optimization path exists yet
(promotion is a developer decision)"):

* mined patterns that clear the support / success-rate thresholds
  become deterministic proposals (``AutoPromotionEngine``);
* the approval gate is on by default: proposals stay PENDING and
  nothing reaches the optimizer until ``approve``/``approve_all``
  runs (the --approve gate);
* ``AutoPromotionConfig(auto_approve=True)`` (the --auto mode) runs
  the channel end to end: qualifying proposals are applied at mining
  time, registering the executable fused instruction and installing
  the peephole rewrite rule in the optimizer's rule set;
* applied promotions replay: the fused instruction reproduces its
  mined subsequence's outputs, and the installed rule rewrites
  matching streams through the standard optimization passes;
* the pipeline is deterministic and idempotent: independent engines
  agree on proposals, fused names, rules and instruction ids, and
  re-mining the same traces never duplicates or re-applies.
"""

import pytest
from typing import Any, Dict, List

from agentflow.core.isa.agentic_calling import AgenticTransformer
from agentflow.core.isa.auto_promotion import (
    AutoPromotionConfig,
    AutoPromotionEngine,
    PromotionStatus,
)
from agentflow.core.isa.instruction import Instruction, InstructionType
from agentflow.core.isa.instruction_learning import (
    DialoguePattern,
    InstructionLearner,
    record_trace,
)
from agentflow.core.isa.instruction_optimization import StaticOptimizer


# ----------------------------------------------------------------------
# A small deterministic, pure, ctx-connected instruction vocabulary.
# Every step reads from the shared ctx and merges its dict output back
# into it (the run_stream convention of the system task demo).
# ----------------------------------------------------------------------

class ExtractSeed(Instruction):
    """Pure: derive the raw value list from the seed."""

    def __init__(self):
        super().__init__(
            id="extract", name="extract",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Extract raw values from the seed",
            side_effects=False, outputs=["values"],
        )
        self.dependencies = ["seed"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seed = input_params["ctx"]["seed"]
        return {"values": [(seed + i) % 7 for i in range(4)]}


class TransformValues(Instruction):
    """Pure: shift the raw values."""

    def __init__(self):
        super().__init__(
            id="transform", name="transform",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Transform raw values",
            side_effects=False, outputs=["normalized"],
        )
        self.dependencies = ["values"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        values = input_params["ctx"]["values"]
        return {"normalized": [v * 2 + 1 for v in values]}


class CollectValues(Instruction):
    """Pure: aggregate the transformed values."""

    def __init__(self):
        super().__init__(
            id="collect", name="collect",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Collect transformed values",
            side_effects=False, outputs=["summary"],
        )
        self.dependencies = ["normalized"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        normalized = input_params["ctx"]["normalized"]
        return {"summary": {"total": len(normalized), "sum": sum(normalized)}}


class FetchMeta(Instruction):
    """Pure: derive metadata from the seed (second, independent pattern)."""

    def __init__(self):
        super().__init__(
            id="fetch", name="fetch",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Fetch metadata from the seed",
            side_effects=False, outputs=["meta"],
        )
        self.dependencies = ["seed"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        seed = input_params["ctx"]["seed"]
        return {"meta": {"seed": seed}}


class PersistMeta(Instruction):
    """Pure: render the metadata as persisted."""

    def __init__(self):
        super().__init__(
            id="persist", name="persist",
            type=InstructionType.STATE_MANAGEMENT,
            params={}, description="Persist the metadata",
            side_effects=False, outputs=["saved"],
        )
        self.dependencies = ["meta"]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        meta = input_params["ctx"]["meta"]
        return {"saved": f"saved:{meta['seed']}"}


class ReportConsumer(Instruction):
    """Live consumer bound to the fused instruction by its id.

    The peephole rewrite emits a name-level placeholder; a consumer
    that depends on the fused id keeps the rewritten node alive
    through the optimizer's dead-code pass.
    """

    def __init__(self, consumes_id: str):
        super().__init__(
            id="report", name="report",
            type=InstructionType.CONTROL_FLOW,
            params={}, description="Consume the fused instruction",
            side_effects=True, outputs=[],
        )
        self.dependencies = [consumes_id]

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"reported": True}


ABC_IDS = ["extract", "transform", "collect"]
ABC_NAMES = ["extract", "transform", "collect"]
XY_IDS = ["fetch", "persist"]
XY_NAMES = ["fetch", "persist"]


def build_abc_stream() -> List[Instruction]:
    """The three-step ctx-connected stream, rebuilt fresh on every call."""
    return [ExtractSeed(), TransformValues(), CollectValues()]


def build_xy_stream() -> List[Instruction]:
    """The two-step ctx-connected stream, rebuilt fresh on every call."""
    return [FetchMeta(), PersistMeta()]


async def record_streams(streams: List[List[Instruction]], seed: int = 42):
    """Record one mining-ready trace per stream (shared seed)."""
    return [await record_trace(stream, {"seed": seed}) for stream in streams]


def failed_abc_trace() -> List[Dict[str, Any]]:
    """One [extract, transform, collect] run whose middle step failed."""
    return [
        {"instruction": ExtractSeed(), "success": True,
         "outputs": {"values": [1, 2, 3, 4]}},
        {"instruction": TransformValues(), "success": False, "outputs": {}},
        {"instruction": CollectValues(), "success": True,
         "outputs": {"summary": {"total": 0, "sum": 0}}},
    ]


def make_engine(config: AutoPromotionConfig = None) -> AutoPromotionEngine:
    """A fresh engine over a fresh learner and a fresh optimizer."""
    return AutoPromotionEngine(
        InstructionLearner(AgenticTransformer()),
        StaticOptimizer(),
        config,
    )


# ----------------------------------------------------------------------
# Threshold gate: qualified patterns become proposals
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_threshold_qualified_pattern_becomes_pending_proposal():
    """Qualified pattern -> one PENDING proposal; the gate stays shut."""
    engine = make_engine()
    proposals = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )

    assert len(proposals) == 1
    proposal = proposals[0]
    assert proposal.status is PromotionStatus.PENDING
    assert proposal.instruction_ids == ABC_IDS
    assert proposal.instruction_names == ABC_NAMES
    assert proposal.support == 2
    assert proposal.success_rate == 1.0
    assert proposal.fused_name.startswith("learned_pattern_")
    assert proposal.proposal_id == (
        "proposal_" + proposal.pattern_id.removeprefix("pattern_")
    )
    assert engine.pending_proposals() == proposals

    # The approval gate is on by default: nothing reached the optimizer.
    assert engine.optimizer.peephole_patterns == {}
    assert engine.promoted_instructions == {}


@pytest.mark.asyncio
async def test_pattern_below_support_threshold_is_not_proposed():
    """The proposal gate is independent of (stricter than) mining."""
    engine = make_engine(AutoPromotionConfig(min_support=3))
    traces = await record_streams([build_abc_stream(), build_abc_stream()])

    assert engine.mine_and_propose(traces) == []
    # Mining promoted the pattern (support 2 >= recognizer threshold 2)
    # but the promotion gate (min_support=3) did not.
    assert engine.learner.pattern_recognizer.patterns
    assert engine.proposals == {}
    assert engine.pending_proposals() == []


@pytest.mark.asyncio
async def test_pattern_below_success_threshold_is_not_proposed():
    """A 0.8 success rate passes the default gate but not a 0.9 one."""
    traces = await record_streams([build_abc_stream()] * 4) + [failed_abc_trace()]

    # Default gate (success >= 0.7): proposed with the observed 0.8.
    permissive = make_engine()
    proposals = permissive.mine_and_propose(traces)
    assert len(proposals) == 1
    assert proposals[0].support == 5
    assert proposals[0].success_rate == pytest.approx(0.8)

    # Strict gate (success >= 0.9): the same mined pattern is rejected.
    strict = make_engine(AutoPromotionConfig(min_success_rate=0.9))
    assert strict.mine_and_propose(traces) == []
    assert strict.learner.pattern_recognizer.patterns
    assert strict.proposals == {}


def test_bodyless_pattern_is_not_proposable():
    """Dialogue-mined patterns carry no replayable body, so no proposal."""
    engine = make_engine()
    bodyless = DialoguePattern(
        pattern_id="pattern_0",
        components=["extract", "transform"],
        frequency=9,
        success_rate=1.0,
        context_requirements=set(),
    )

    assert engine.propose_from_patterns([bodyless]) == []
    assert engine.proposals == {}


# ----------------------------------------------------------------------
# The --approve gate
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_approve_applies_and_installs_rewrite_rule():
    """Approving promotes: fused instruction + optimizer rule set entry."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )

    applied = engine.approve(proposal.proposal_id)
    assert applied is proposal
    assert applied.status is PromotionStatus.APPLIED
    assert engine.pending_proposals() == []

    # The optimizer's rule set -- the one mining was never allowed to
    # edit -- now carries the name-level rewrite for this promotion.
    assert engine.optimizer.peephole_patterns == {
        "_".join(ABC_NAMES): (ABC_NAMES, [proposal.fused_name])
    }

    # The fused instruction is executable and replays its constituents.
    fused = engine.promoted_instructions[proposal.fused_name]
    assert fused.id == f"learned_{proposal.fused_name}"
    reference = await record_trace(list(fused.components), {"seed": 42})
    reference_outputs: Dict[str, Any] = {}
    for entry in reference:
        reference_outputs.update(entry["outputs"])
    assert await fused.execute({"ctx": {"seed": 42}}) == reference_outputs


@pytest.mark.asyncio
async def test_repeated_approve_is_idempotent():
    """Approving an applied proposal again changes nothing."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )
    engine.approve(proposal.proposal_id)
    rules_before = dict(engine.optimizer.peephole_patterns)

    again = engine.approve(proposal.proposal_id)
    assert again.status is PromotionStatus.APPLIED
    assert len(engine.proposals) == 1
    assert engine.optimizer.peephole_patterns == rules_before
    assert len(engine.promoted_instructions) == 1


@pytest.mark.asyncio
async def test_gate_rejects_unknown_ids_and_rejected_proposals():
    """Unknown ids raise; a rejected proposal cannot be approved."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )

    with pytest.raises(ValueError, match="not found"):
        engine.approve("proposal_missing")
    with pytest.raises(ValueError, match="not found"):
        engine.reject("proposal_missing")

    assert engine.reject(proposal.proposal_id).status is PromotionStatus.REJECTED
    with pytest.raises(ValueError, match="rejected"):
        engine.approve(proposal.proposal_id)
    # Re-rejecting is an idempotent no-op.
    assert engine.reject(proposal.proposal_id).status is PromotionStatus.REJECTED

    # Nothing was promoted through the failed gate attempts.
    assert engine.optimizer.peephole_patterns == {}
    assert engine.promoted_instructions == {}


@pytest.mark.asyncio
async def test_rejecting_an_applied_proposal_is_refused():
    """Applied promotions are never rolled back by a later reject."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )
    engine.approve(proposal.proposal_id)

    with pytest.raises(ValueError, match="already applied"):
        engine.reject(proposal.proposal_id)
    assert proposal.status is PromotionStatus.APPLIED


# ----------------------------------------------------------------------
# The --auto mode: end to end
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_auto_mode_applies_end_to_end():
    """auto_approve skips the gate: mine -> apply -> rule, in one call."""
    auto = make_engine(AutoPromotionConfig(auto_approve=True))
    proposals = auto.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )

    assert [p.status for p in proposals] == [PromotionStatus.APPLIED]
    proposal = proposals[0]
    assert auto.optimizer.peephole_patterns == {
        "_".join(ABC_NAMES): (ABC_NAMES, [proposal.fused_name])
    }

    # The promoted instruction executes by composite replay.
    fused = auto.promoted_instructions[proposal.fused_name]
    result = await fused.execute({"ctx": {"seed": 42}})
    assert set(result) == {"values", "normalized", "summary"}

    # The standard optimization passes rewrite a matching stream: the
    # constituent run collapses to the fused name, which stays live
    # because a consumer depends on it by id.
    consumer = ReportConsumer(proposal.fused_name)
    optimized = auto.optimizer.optimize(build_abc_stream() + [consumer])
    assert [instr.name for instr in optimized] == [proposal.fused_name, "report"]


@pytest.mark.asyncio
async def test_approve_all_applies_every_pending_proposal_in_order():
    """Two independent patterns, both applied in deterministic order."""
    engine = make_engine()
    traces = await record_streams(
        [build_abc_stream(), build_abc_stream(),
         build_xy_stream(), build_xy_stream()]
    )
    proposals = engine.mine_and_propose(traces)
    assert [p.instruction_ids for p in proposals] == [ABC_IDS, XY_IDS]

    applied = engine.approve_all()
    assert [p.status for p in applied] == [
        PromotionStatus.APPLIED, PromotionStatus.APPLIED,
    ]
    assert set(engine.optimizer.peephole_patterns) == {
        "_".join(ABC_NAMES), "_".join(XY_NAMES),
    }
    assert sorted(engine.promoted_instructions) == sorted(
        p.fused_name for p in proposals
    )
    # Nothing pending remains; a second approve_all is a no-op.
    assert engine.pending_proposals() == []
    assert engine.approve_all() == []


@pytest.mark.asyncio
async def test_promotion_without_optimizer_registers_instruction_only():
    """A missing optimizer degrades to registry-only promotion."""
    engine = AutoPromotionEngine(
        InstructionLearner(AgenticTransformer()), None, None
    )
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )

    assert engine.approve(proposal.proposal_id).status is PromotionStatus.APPLIED
    fused = engine.promoted_instructions[proposal.fused_name]
    assert await fused.execute({"ctx": {"seed": 42}})


# ----------------------------------------------------------------------
# Determinism and replay
# ----------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_is_deterministic_across_engines():
    """Independent engines agree on every observable of the pipeline."""
    traces = await record_streams(
        [build_abc_stream(), build_abc_stream(),
         build_xy_stream(), build_xy_stream()]
    )
    engine_a = make_engine()
    engine_b = make_engine()

    proposals_a = engine_a.mine_and_propose(traces)
    proposals_b = engine_b.mine_and_propose(traces)
    assert proposals_a == proposals_b
    assert [p.instruction_ids for p in proposals_a] == [ABC_IDS, XY_IDS]

    engine_a.approve_all()
    engine_b.approve_all()
    assert engine_a.proposals == engine_b.proposals
    assert set(engine_a.optimizer.peephole_patterns) == set(
        engine_b.optimizer.peephole_patterns
    )
    assert [
        instr.id for instr in engine_a.promoted_instructions.values()
    ] == [
        instr.id for instr in engine_b.promoted_instructions.values()
    ]


@pytest.mark.asyncio
async def test_remining_same_traces_is_idempotent():
    """Re-mining returns the existing proposals; nothing re-applies."""
    traces = await record_streams([build_abc_stream(), build_abc_stream()])

    # --auto: the applied state survives a re-mine untouched.
    auto = make_engine(AutoPromotionConfig(auto_approve=True))
    first = auto.mine_and_propose(traces)
    assert [p.status for p in first] == [PromotionStatus.APPLIED]
    rules_before = dict(auto.optimizer.peephole_patterns)

    second = auto.mine_and_propose(traces)
    assert second == first
    assert len(auto.proposals) == 1
    assert auto.optimizer.peephole_patterns == rules_before

    # Gated: pending proposals survive a re-mine without duplicating.
    gated = make_engine()
    g_first = gated.mine_and_propose(traces)
    g_second = gated.mine_and_propose(traces)
    assert g_second == g_first
    assert len(gated.proposals) == 1
    assert [p.status for p in g_second] == [PromotionStatus.PENDING]


# -- deprecation: the delete half of the repertoire lifecycle ---------


@pytest.mark.asyncio
async def test_deprecate_retires_an_applied_promotion():
    """Deprecating withdraws both the executable and its rewrite rule."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )
    engine.approve(proposal.proposal_id)
    assert engine.active_promotions() == [proposal]

    retired = engine.deprecate(proposal.proposal_id, reason="superseded")

    assert retired.status is PromotionStatus.DEPRECATED
    assert retired.deprecation_reason == "superseded"
    assert engine.promoted_instructions == {}
    assert engine.optimizer.peephole_patterns == {}
    assert engine.active_promotions() == []
    # Idempotent: retiring twice changes nothing.
    assert engine.deprecate(proposal.proposal_id) is proposal


@pytest.mark.asyncio
async def test_deprecate_refuses_proposals_that_were_never_applied():
    """Only an installation can be withdrawn; the record stays honest."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )
    with pytest.raises(ValueError, match="only an applied promotion"):
        engine.deprecate(proposal.proposal_id)
    engine.reject(proposal.proposal_id)
    with pytest.raises(ValueError, match="only an applied promotion"):
        engine.deprecate(proposal.proposal_id)


@pytest.mark.asyncio
async def test_deprecate_stale_retires_only_idle_promotions():
    """Staleness counts observations since last use; absent names are idle."""
    engine = make_engine()
    (proposal,) = engine.mine_and_propose(
        await record_streams([build_abc_stream(), build_abc_stream()])
    )
    engine.approve(proposal.proposal_id)

    assert engine.deprecate_stale({proposal.fused_name: 1}, max_idle=5) == []
    assert engine.active_promotions() == [proposal]

    retired = engine.deprecate_stale({proposal.fused_name: 9}, max_idle=5)
    assert [p.proposal_id for p in retired] == [proposal.proposal_id]
    assert engine.active_promotions() == []
