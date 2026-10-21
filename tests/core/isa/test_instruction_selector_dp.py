"""Tests for tree-pattern instruction selection (MATCH-TREE, Appendix D.1).

The dynamic-programming selector is exposed via
``InstructionSelector(selection_strategy="dp")`` while the default
``"heuristic"`` strategy keeps the shipped name-overlap matcher.  These
tests verify on random small IR trees that

* the DP covering cost never exceeds the heuristic covering cost,
* the DP covering cost equals the brute-force enumeration optimum,
* the produced covering tiles the tree exactly (each node covered once),
* the default strategy and the flat selection path stay backward
  compatible.
"""

import itertools
import math
import random
from collections import Counter
from typing import Iterator, List, Tuple

import pytest

from agentflow.core.isa.instruction_selector import (
    InstructionSelector,
    IRTreeNode,
    TreePattern,
    TreeSelectionResult,
)
from agentflow.core.isa.isa_manager import Instruction, InstructionType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_instruction(name: str, cost: float = 1.0, params=None) -> Instruction:
    """Build a catalogue instruction (isa_manager flavour)."""
    return Instruction(
        id=f"instr-{name}",
        name=name,
        type=InstructionType.COMPUTATION,
        description=f"{name} operation",
        params=params or {},
        cost=cost,
    )


def make_pattern(root_op: str, cost: float, spanned=(), name: str = None) -> TreePattern:
    """Build an explicit tree pattern with the given cost and span."""
    return TreePattern(
        instruction=make_instruction(name or root_op, cost=cost),
        root_op=root_op,
        spanned_child_ops=tuple(spanned),
        cost=cost,
    )


def subtree_nodes(node: IRTreeNode) -> Iterator[IRTreeNode]:
    """Yield every node of the subtree rooted at ``node``."""
    yield node
    for child in node.children:
        yield from subtree_nodes(child)


def assert_valid_tiling(result: TreeSelectionResult, root: IRTreeNode) -> None:
    """The covering must tile the tree: every node covered exactly once."""
    covered: List[IRTreeNode] = []
    for node, pattern in result.covering:
        covered.append(node)
        for child in pattern.spanned_children(node):
            covered.extend(subtree_nodes(child))
    all_nodes = list(subtree_nodes(root))
    assert Counter(id(n) for n in covered) == Counter(id(n) for n in all_nodes)
    # The reported cost is the sum of the chosen patterns' costs.
    assert result.cost == pytest.approx(sum(p.cost for _, p in result.covering))
    # The instruction sequence mirrors the covering, root-first.
    assert result.instructions == [p.instruction for _, p in result.covering]


def brute_force_optimum(root: IRTreeNode, patterns: List[TreePattern]) -> float:
    """Enumerate every valid covering of the tree; return the minimal cost.

    Independent of the DP: recurses over the product of the children's
    covering choices rather than memoizing per-node minima.
    """

    def coverings(node: IRTreeNode) -> Iterator[Tuple[List[Tuple[IRTreeNode, TreePattern]], float]]:
        for pattern in patterns:
            if not pattern.matches(node):
                continue
            spanned_ids = {id(c) for c in pattern.spanned_children(node)}
            rest = [c for c in node.children if id(c) not in spanned_ids]
            child_options = [list(coverings(c)) for c in rest]
            if any(not options for options in child_options):
                continue
            for combo in itertools.product(*child_options):
                entries = [(node, pattern)]
                for sub_entries, _ in combo:
                    entries.extend(sub_entries)
                yield entries, pattern.cost + sum(cost for _, cost in combo)

    return min((cost for _, cost in coverings(root)), default=math.inf)


def random_tree(rng: random.Random, ops: List[str], max_depth: int = 3,
                max_branch: int = 3) -> IRTreeNode:
    """Generate a random small IR tree over ``ops``."""

    def build(depth: int) -> IRTreeNode:
        children = []
        if depth < max_depth and rng.random() < 0.7:
            for _ in range(rng.randint(1, max_branch)):
                children.append(build(depth + 1))
        return IRTreeNode(op=rng.choice(ops), children=children)

    return build(0)


def random_catalogue(rng: random.Random, ops: List[str]) -> List[TreePattern]:
    """Every op gets a single-node pattern; add a few spanning patterns."""
    patterns = [
        make_pattern(op, cost=round(rng.uniform(0.5, 5.0), 3)) for op in ops
    ]
    for _ in range(rng.randint(1, 4)):
        root_op = rng.choice(ops)
        spanned = tuple(rng.choice(ops) for _ in range(rng.randint(1, 2)))
        patterns.append(
            make_pattern(root_op, cost=round(rng.uniform(0.5, 5.0), 3),
                         spanned=spanned, name=f"{root_op}_fused")
        )
    return patterns


# ---------------------------------------------------------------------------
# Strategy configuration
# ---------------------------------------------------------------------------

class TestStrategyConfiguration:
    def test_default_strategy_is_heuristic(self):
        assert InstructionSelector().selection_strategy == "heuristic"

    def test_invalid_strategy_rejected(self):
        with pytest.raises(ValueError):
            InstructionSelector(selection_strategy="bogus")


# ---------------------------------------------------------------------------
# MATCH-TREE dynamic program (Appendix D.1)
# ---------------------------------------------------------------------------

class TestMatchTreeDP:
    def test_fusion_pattern_beats_greedy(self):
        """seq(llm, llm): greedy tiles per node (1 + 2 + 2 = 5); the fusion
        pattern spanning both llm children covers the whole tree for 3."""
        tree = IRTreeNode(op="seq", children=[IRTreeNode("llm"), IRTreeNode("llm")])
        patterns = [
            make_pattern("seq", 1.0),
            make_pattern("llm", 2.0),
            make_pattern("seq", 3.0, spanned=("llm", "llm"), name="seq_fused"),
        ]
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=patterns)
        heur = InstructionSelector().select_for_tree(tree, patterns=patterns)

        assert heur.cost == pytest.approx(5.0)
        assert dp.cost == pytest.approx(3.0)
        assert dp.cost <= heur.cost
        assert [instr.name for instr in dp.instructions] == ["seq_fused"]
        assert dp.costs[id(tree)] == pytest.approx(3.0)
        assert dp.selected[id(tree)].instruction.id == "instr-seq_fused"
        assert_valid_tiling(dp, tree)

    def test_span_requires_every_spanned_op_among_children(self):
        """A spanning pattern needs each spanned op present as a child; with
        only x children the (x, y) span cannot fire."""
        tree = IRTreeNode(op="p", children=[IRTreeNode("x"), IRTreeNode("x")])
        patterns = [
            make_pattern("p", 1.0),
            make_pattern("x", 1.0),
            make_pattern("y", 1.0),
            make_pattern("p", 1.0, spanned=("x", "y"), name="p_fused"),
        ]
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=patterns)

        assert dp.cost == pytest.approx(3.0)  # p + x + x; fusion inapplicable
        assert_valid_tiling(dp, tree)

    def test_span_covers_every_child_with_spanned_op(self):
        """Spanning is by op: the pattern spans all children whose op is
        listed, and the remaining children are covered independently."""
        tree = IRTreeNode(op="p",
                          children=[IRTreeNode("x"), IRTreeNode("x"), IRTreeNode("q")])
        patterns = [
            make_pattern("p", 1.0),
            make_pattern("x", 4.0),
            make_pattern("q", 4.0),
            make_pattern("p", 2.0, spanned=("x",), name="p_fused"),
        ]
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=patterns)

        assert dp.cost == pytest.approx(6.0)  # fused p + both x (2.0) plus q (4.0)
        assert_valid_tiling(dp, tree)

    def test_uncoverable_tree_is_incomplete(self):
        tree = IRTreeNode(op="z")
        patterns = [make_pattern("a", 1.0)]
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=patterns)

        assert dp.complete is False
        assert dp.cost == math.inf
        assert dp.instructions == []
        assert dp.covering == []

    def test_leaf_tree_selects_single_pattern(self):
        tree = IRTreeNode(op="a")
        pattern = make_pattern("a", 2.5)
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=[pattern])

        assert dp.complete is True
        assert dp.cost == pytest.approx(2.5)
        assert dp.instructions == [pattern.instruction]

    def test_cost_ties_break_to_catalogue_order(self):
        tree = IRTreeNode(op="a")
        first = make_pattern("a", 1.0, name="first")
        second = make_pattern("a", 1.0, name="second")
        dp = InstructionSelector(selection_strategy="dp").select_for_tree(
            tree, patterns=[first, second])

        assert dp.cost == pytest.approx(1.0)
        assert dp.instructions[0].id == "instr-first"


# ---------------------------------------------------------------------------
# Randomized comparison against brute force and the heuristic
# ---------------------------------------------------------------------------

class TestRandomTreeOptimality:
    @pytest.mark.parametrize("seed", range(25))
    def test_dp_matches_brute_force_and_beats_heuristic(self, seed):
        rng = random.Random(9000 + seed)
        ops = ["a", "b", "c", "d"]
        tree = random_tree(rng, ops)
        patterns = random_catalogue(rng, ops)

        dp = InstructionSelector(selection_strategy="dp").select_for_tree(tree, patterns=patterns)
        heur = InstructionSelector().select_for_tree(tree, patterns=patterns)

        assert heur.complete is True
        assert dp.complete is True
        assert dp.cost == pytest.approx(brute_force_optimum(tree, patterns))
        assert dp.cost <= heur.cost + 1e-9
        assert_valid_tiling(dp, tree)


# ---------------------------------------------------------------------------
# Backward compatibility and catalogue integration
# ---------------------------------------------------------------------------

class TestSelectInstructionsDispatch:
    def test_dp_strategy_uses_ir_tree_from_input(self):
        tree = IRTreeNode(op="seq", children=[IRTreeNode("llm"), IRTreeNode("llm")])
        available = {
            "seq": make_instruction("seq", 1.0),
            "llm": make_instruction("llm", 2.0),
        }
        selector = InstructionSelector(selection_strategy="dp")
        selected = selector.select_instructions(
            {"ir_tree": tree, "task": "run seq"}, available)

        assert [instr.name for instr in selected] == ["seq", "llm", "llm"]

    def test_default_strategy_ignores_ir_tree(self):
        """Backward compat: the default heuristic keeps the flat name-overlap
        path even when an ir_tree is present in the input."""
        tree = IRTreeNode(op="seq")
        available = {"seq": make_instruction("seq", 1.0)}
        selector = InstructionSelector()  # default strategy
        selected = selector.select_instructions(
            {"ir_tree": tree, "task": "seq", "text": "seq"}, available)

        assert isinstance(selected, list)
        assert len(selected) <= 3
        assert all(isinstance(instr, Instruction) for instr in selected)

    def test_spanning_declared_via_instruction_params(self):
        """An instruction declares a multi-node pattern through
        params["spanned_child_ops"]; its name is the matched root op."""
        tree = IRTreeNode(op="seq", children=[IRTreeNode("llm"), IRTreeNode("llm")])
        available = {
            "seq": make_instruction("seq", 1.0),
            "llm": make_instruction("llm", 2.0),
            "seq_fused": make_instruction(
                "seq", 3.0, params={"spanned_child_ops": ["llm", "llm"]}),
        }
        selector = InstructionSelector(selection_strategy="dp")
        result = selector.select_for_tree(tree, available_instructions=available)

        assert isinstance(result, TreeSelectionResult)
        assert result.cost == pytest.approx(3.0)
        # The declared span lets the fused instruction cover the whole tree.
        assert len(result.covering) == 1
        _, root_pattern = result.covering[0]
        assert root_pattern.instruction is available["seq_fused"]
        assert root_pattern.spanned_child_ops == ("llm", "llm")
        assert_valid_tiling(result, tree)

    def test_select_for_tree_heuristic_strategy_is_deterministic(self):
        """The heuristic tree baseline has no random exploration term, so
        repeated runs return the same covering."""
        tree = IRTreeNode(op="seq", children=[IRTreeNode("llm"), IRTreeNode("llm")])
        patterns = [
            make_pattern("seq", 1.0),
            make_pattern("llm", 2.0),
        ]
        selector = InstructionSelector()
        first = selector.select_for_tree(tree, patterns=patterns)
        second = selector.select_for_tree(tree, patterns=patterns)

        assert first.cost == pytest.approx(5.0)
        assert [instr.name for instr in first.instructions] == [instr.name for instr in second.instructions]
