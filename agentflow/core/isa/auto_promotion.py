"""Automated mining-to-optimization promotion pipeline (paper Section 3.3.1).

The shipped learning stack could *mine* recurring instruction
subsequences (``PatternRecognizer.extract_instruction_patterns``) and
*build* executable fused instructions from them
(``InstructionLearner.create_instruction``), but the step in between
-- deciding which mined patterns actually become instructions and
enter the optimizer's rule set -- was a manual developer decision;
the paper discloses "no automated mining-to-optimization path exists
yet (promotion is a developer decision)".  This module is that path,
behind an approval gate:

    mine traces -> threshold-gated proposals -> (approval | --auto)
        -> executable fused instruction registered
        -> peephole rewrite rule installed in the optimizer

Gate semantics (the ``--approve`` gate is on by default):

* :meth:`AutoPromotionEngine.mine_and_propose` turns every mined
  pattern that clears the configured support / success-rate thresholds
  into a proposal.  Proposals start out ``PENDING``: nothing reaches
  the optimizer until :meth:`approve` / :meth:`approve_all` runs.
* ``AutoPromotionConfig(auto_approve=True)`` (the ``--auto`` mode)
  bypasses the gate: qualifying proposals are applied at mining time,
  so one call carries the channel end to end.

Determinism and replay:

* mining itself is deterministic (see
  ``PatternRecognizer.extract_instruction_patterns``);
* proposal identity is content-derived -- the proposal id and the
  fused-instruction name depend only on the mined instruction-id
  sequence, never on run counters -- and proposals follow mining
  order, so the same traces always yield the same proposals, with
  the same ids, in the same order, across independent engines;
* re-mining the same traces returns the existing proposals unchanged
  (no duplicates, statuses preserved, nothing re-applied); a proposal
  records the statistics of its first qualification and applied
  promotions are never rolled back by later mining runs;
* applying the same approved proposal always installs the same rule
  (constituent names -> fused name) and registers the same fused
  instruction id.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .instruction import Instruction
from .instruction_learning import (
    DialoguePattern,
    InstructionLearner,
    InstructionTemplate,
)
from .instruction_optimization import StaticOptimizer


def _proposal_id(pattern_id: str) -> str:
    """Content-derived proposal id from the mined pattern's id."""
    return "proposal_" + pattern_id.removeprefix("pattern_")


class PromotionStatus(str, Enum):
    """Lifecycle states of a promotion proposal."""
    PENDING = "pending"    # waiting at the approval gate
    APPLIED = "applied"   # approved and promoted
    REJECTED = "rejected"  # declined at the gate


@dataclass
class AutoPromotionConfig:
    """Gate thresholds and mode for the auto-promotion pipeline.

    Defaults mirror ``LearningConfig`` (``min_pattern_support=2``,
    ``min_success_rate=0.7``): with the defaults every pattern the
    recognizer promotes is also proposal-worthy, and a stricter
    promotion policy is expressed by raising these thresholds above
    the mining ones.

    Attributes:
        min_support: minimum number of window occurrences (the
            ``DialoguePattern.frequency`` statistic) a pattern must
            show to be proposed.
        min_success_rate: minimum all-steps-succeeded occurrence rate
            a pattern must show to be proposed.
        auto_approve: the ``--auto`` mode -- when True, qualifying
            proposals skip the approval gate and are applied
            immediately; the default False keeps the gate closed
            until ``approve`` / ``approve_all`` runs (the ``--approve``
            gate).
    """
    min_support: int = 2
    min_success_rate: float = 0.7
    auto_approve: bool = False


@dataclass
class PromotionProposal:
    """A threshold-qualified promotion candidate, pending approval.

    Identity is content-derived: ``proposal_id`` and ``fused_name``
    depend only on the mined instruction-id sequence, never on run
    counters, so the same mined pattern always yields the same
    proposal in any engine instance.
    """
    proposal_id: str
    pattern_id: str
    instruction_ids: List[str]
    instruction_names: List[str]
    fused_name: str
    support: int
    success_rate: float
    status: PromotionStatus = PromotionStatus.PENDING


class AutoPromotionEngine:
    """Approval-gated channel from mined patterns to the optimizer."""

    def __init__(
        self,
        learner: InstructionLearner,
        optimizer: Optional[StaticOptimizer] = None,
        config: Optional[AutoPromotionConfig] = None,
    ):
        self.learner = learner
        self.optimizer = optimizer
        self.config = config or AutoPromotionConfig()
        self.proposals: Dict[str, PromotionProposal] = {}
        # Executable fused instructions, keyed by fused name -- the
        # binding point for the name-level rewrite rules installed on
        # approval (peephole replacements are name-level rewrites;
        # the executable semantics live here).
        self.promoted_instructions: Dict[str, Instruction] = {}
        self._templates: Dict[str, InstructionTemplate] = {}

    # -- mining -> proposals ------------------------------------------

    def mine_and_propose(
        self, traces: List[List[Any]]
    ) -> List[PromotionProposal]:
        """Mine ``traces`` and gate every promoted pattern into proposals.

        Mining runs through the learner's pattern recognizer
        (deterministic; see ``extract_instruction_patterns``).  Each
        mined pattern that clears the support / success-rate
        thresholds becomes a proposal -- ``PENDING`` behind the
        approval gate, or immediately applied under ``auto_approve``.
        """
        patterns = (
            self.learner.pattern_recognizer.extract_instruction_patterns(traces)
        )
        return self.propose_from_patterns(patterns)

    def propose_from_patterns(
        self, patterns: List[DialoguePattern]
    ) -> List[PromotionProposal]:
        """Gate already-mined patterns into proposals, in mining order."""
        proposals: List[PromotionProposal] = []
        for pattern in patterns:
            proposal = self._proposal_for(pattern)
            if proposal is not None:
                proposals.append(proposal)
        return proposals

    def _proposal_for(
        self, pattern: DialoguePattern
    ) -> Optional[PromotionProposal]:
        """Gate one mined pattern; ``None`` when it does not qualify."""
        # Only trace-mined patterns carry the recorded constituent
        # sequence that an executable fused instruction replays;
        # bodyless (dialogue-mined) patterns cannot be promoted here.
        sequence = pattern.instruction_sequence or []
        if not sequence:
            return None
        if pattern.frequency < self.config.min_support:
            return None
        if pattern.success_rate < self.config.min_success_rate:
            return None
        template = self.learner.template_for_pattern(pattern)
        if template is None:
            return None

        proposal_id = _proposal_id(pattern.pattern_id)
        existing = self.proposals.get(proposal_id)
        if existing is not None:
            # Deterministic replay of an earlier run: the proposal
            # already exists.  Keep it (immutable, status preserved)
            # instead of duplicating or re-applying it.
            return existing

        proposal = PromotionProposal(
            proposal_id=proposal_id,
            pattern_id=pattern.pattern_id,
            instruction_ids=[instruction.id for instruction in sequence],
            instruction_names=[instruction.name for instruction in sequence],
            fused_name=template.name,
            support=pattern.frequency,
            success_rate=pattern.success_rate,
        )
        self.proposals[proposal_id] = proposal
        self._templates[proposal_id] = template
        if self.config.auto_approve:
            self._apply(proposal)
        return proposal

    # -- the approval gate --------------------------------------------

    def pending_proposals(self) -> List[PromotionProposal]:
        """Proposals still waiting at the gate, in creation order."""
        return [
            proposal for proposal in self.proposals.values()
            if proposal.status is PromotionStatus.PENDING
        ]

    def approve(self, proposal_id: str) -> PromotionProposal:
        """Approve one pending proposal and apply its promotion.

        Idempotent: approving an already-applied proposal returns it
        unchanged without re-installing anything.
        """
        proposal = self._require_proposal(proposal_id)
        if proposal.status is PromotionStatus.REJECTED:
            raise ValueError(
                f"Proposal {proposal_id!r} was rejected and cannot "
                "be approved"
            )
        if proposal.status is PromotionStatus.PENDING:
            self._apply(proposal)
        return proposal

    def approve_all(self) -> List[PromotionProposal]:
        """Approve every pending proposal, in creation order."""
        return [
            self.approve(proposal.proposal_id)
            for proposal in self.pending_proposals()
        ]

    def reject(self, proposal_id: str) -> PromotionProposal:
        """Decline one pending proposal at the gate (no promotion).

        Idempotent for pending/rejected proposals; applied promotions
        are never rolled back and refuse the transition.
        """
        proposal = self._require_proposal(proposal_id)
        if proposal.status is PromotionStatus.APPLIED:
            raise ValueError(
                f"Proposal {proposal_id!r} is already applied and "
                "cannot be rejected"
            )
        proposal.status = PromotionStatus.REJECTED
        return proposal

    def _require_proposal(self, proposal_id: str) -> PromotionProposal:
        if proposal_id not in self.proposals:
            raise ValueError(f"Proposal not found: {proposal_id}")
        return self.proposals[proposal_id]

    # -- application ----------------------------------------------------

    def _apply(self, proposal: PromotionProposal) -> None:
        """Promote one approved proposal into the optimizer's world.

        Registers the executable fused instruction (composite replay
        of the mined subsequence, built by the learner) and -- when
        an optimizer is attached -- installs the name-level peephole
        rewrite rule "constituent names -> fused name" in its rule
        set: exactly the rule set the shipped stack never let mining
        touch, now written through the approval gate.
        """
        template = self._templates[proposal.proposal_id]
        instruction = self.learner.create_instruction(template)
        self.promoted_instructions[proposal.fused_name] = instruction
        if self.optimizer is not None:
            self.optimizer.add_peephole_pattern(
                list(proposal.instruction_names),
                [proposal.fused_name],
            )
        proposal.status = PromotionStatus.APPLIED
