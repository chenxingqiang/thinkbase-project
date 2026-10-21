"""Instruction selection and optimization module."""

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .instruction import Instruction
from .isa_manager import Instruction

#: Supported selection strategies: ``heuristic`` is the shipped name-overlap
#: matcher (default, unchanged behavior); ``dp`` is the MATCH-TREE dynamic
#: program over IR trees specified in supplementary Appendix D.1.
SELECTION_STRATEGIES = ("heuristic", "dp")


@dataclass
class IRTreeNode:
    """A node of the IR tree consumed by tree-pattern selection.

    ``op`` is the operation name matched against pattern root ops,
    ``children`` are the operands, and ``payload`` carries optional node
    attributes (e.g. parameters) used by the heuristic scorer.
    """
    op: str
    children: List["IRTreeNode"] = field(default_factory=list)
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TreePattern:
    """A context-local selection pattern (Definition 3.8; Appendix D.1).

    The pattern matches a node whose op equals ``root_op``
    (case-insensitive) and spans the whole subtrees of every child whose
    op is listed in ``spanned_child_ops``; it only matches when each
    spanned op occurs among the node's children.  An empty span is a
    single-node pattern covering just the node itself.
    """
    instruction: Optional[Instruction] = None
    root_op: Optional[str] = None
    spanned_child_ops: Tuple[str, ...] = ()
    cost: Optional[float] = None

    def __post_init__(self):
        """Default the root op and cost from the backing instruction."""
        if self.root_op is None:
            self.root_op = getattr(self.instruction, "name", None)
        if self.root_op is None:
            raise ValueError("TreePattern requires root_op or an instruction name")
        if isinstance(self.spanned_child_ops, (list, tuple)):
            self.spanned_child_ops = tuple(str(op) for op in self.spanned_child_ops)
        else:
            self.spanned_child_ops = ()
        if self.cost is None:
            self.cost = float(getattr(self.instruction, "cost", 1.0))

    def matches(self, node: IRTreeNode) -> bool:
        """Whether the pattern can fire at ``node`` (MATCHES of D.1)."""
        if node is None or not isinstance(node.op, str):
            return False
        if node.op.lower() != str(self.root_op).lower():
            return False
        child_ops = {str(child.op).lower() for child in node.children}
        return all(str(op).lower() in child_ops for op in self.spanned_child_ops)

    def spanned_children(self, node: IRTreeNode) -> List[IRTreeNode]:
        """The children whose whole subtrees this pattern covers."""
        spanned = {str(op).lower() for op in self.spanned_child_ops}
        return [child for child in node.children
                if str(child.op).lower() in spanned]


@dataclass
class TreeSelectionResult:
    """Result of a tree-pattern selection run.

    ``costs`` and ``selected`` are the MATCH-TREE dynamic-programming
    tables keyed by ``id(node)``; ``covering`` lists the chosen
    ``(node, pattern)`` pairs in root-first order; ``complete`` is False
    when no covering exists (the cost is infinite).
    """
    instructions: List[Instruction] = field(default_factory=list)
    cost: float = math.inf
    covering: List[Tuple[IRTreeNode, TreePattern]] = field(default_factory=list)
    costs: Dict[int, float] = field(default_factory=dict)
    selected: Dict[int, Optional[TreePattern]] = field(default_factory=dict)
    complete: bool = False


class InstructionSelector:
    """Selects and optimizes instruction sequences.

    ``selection_strategy`` chooses between the shipped heuristic
    name-overlap matcher (``"heuristic"``, the default and unchanged
    behavior) and the MATCH-TREE dynamic program over IR trees
    (``"dp"``, supplementary Appendix D.1).
    """

    def __init__(self, logger: Optional[logging.Logger] = None,
                 selection_strategy: str = "heuristic"):
        """Initialize instruction selector."""
        if selection_strategy not in SELECTION_STRATEGIES:
            raise ValueError(
                f"Unknown selection strategy {selection_strategy!r}; "
                f"expected one of {SELECTION_STRATEGIES}")
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.selection_strategy = selection_strategy

    def select_instructions(self, input_data: Dict[str, Any], available_instructions: Dict[str, Instruction]) -> List[Instruction]:
        """
        Select appropriate instructions based on input data.

        With ``selection_strategy="dp"`` and an ``"ir_tree"`` entry (an
        :class:`IRTreeNode`) in ``input_data``, selection runs the
        MATCH-TREE dynamic program over the tree and returns the chosen
        instructions in covering order.  Otherwise the heuristic
        name-overlap path below applies, unchanged.

        Args:
            input_data: Input data to process
            available_instructions: Dictionary of available instructions

        Returns:
            Selected instruction sequence
        """
        ir_tree = input_data.get("ir_tree") if isinstance(input_data, dict) else None
        if self.selection_strategy == "dp" and ir_tree is not None:
            try:
                result = self.select_for_tree(ir_tree, available_instructions)
                return list(result.instructions) if result.complete else []
            except Exception as e:
                self.logger.error(f"Tree instruction selection failed: {str(e)}")
                return []

        try:
            # Extract task and parameters
            task = input_data.get('task', '')
            text = input_data.get('text', '')
            params = input_data.get('params', {})

            # Score each instruction
            scored_instructions = []
            for instruction in available_instructions.values():
                score = self._score_instruction(instruction, task, text, params)
                scored_instructions.append((score, instruction))

            # Sort by score and select top instructions
            scored_instructions.sort(reverse=True)
            selected = [instr for _, instr in scored_instructions[:3]]

            return selected

        except Exception as e:
            self.logger.error(f"Instruction selection failed: {str(e)}")
            return []

    def select_for_tree(self, tree: IRTreeNode,
                        available_instructions: Optional[Dict[str, Instruction]] = None,
                        patterns: Optional[Sequence[TreePattern]] = None) -> TreeSelectionResult:
        """
        Select a minimal-cost covering of ``tree`` under the configured strategy.

        Args:
            tree: Root of the IR tree to cover
            available_instructions: Instruction catalogue; patterns are
                derived from it (an instruction declares a multi-node
                pattern via ``params["spanned_child_ops"]``)
            patterns: Explicit pattern set; takes precedence over the
                derived catalogue

        Returns:
            A :class:`TreeSelectionResult`; ``complete`` is False when the
            tree cannot be covered.
        """
        if patterns is None:
            patterns = self._build_patterns(available_instructions or {})
        if self.selection_strategy == "dp":
            return self._match_tree(tree, list(patterns))
        return self._greedy_tree_cover(tree, list(patterns))

    def _build_patterns(self, available_instructions: Dict[str, Instruction]) -> List[TreePattern]:
        """Derive tree patterns from the instruction catalogue.

        Each instruction contributes one pattern rooted at its name; an
        instruction declaring ``params["spanned_child_ops"]`` yields a
        multi-node pattern spanning those child ops.
        """
        patterns = []
        for key in sorted(available_instructions.keys()):
            instruction = available_instructions[key]
            params = getattr(instruction, "params", None)
            spanned = params.get("spanned_child_ops", ()) if isinstance(params, dict) else ()
            patterns.append(TreePattern(instruction=instruction,
                                        root_op=instruction.name,
                                        spanned_child_ops=spanned))
        return patterns

    def _match_tree(self, root: IRTreeNode, patterns: List[TreePattern]) -> TreeSelectionResult:
        """MATCH-TREE (Appendix D.1): minimal-cost covering via dynamic
        programming over the IR tree.

        For each node the recurrence takes, over every pattern matching
        the node, the pattern cost plus the optimal costs of the children
        the pattern does not span (spanned subtrees are paid for once by
        the pattern itself); the covering is read off the ``selected``
        table by walking from the root and skipping spanned subtrees.
        """
        costs: Dict[int, float] = {}
        selected: Dict[int, Optional[TreePattern]] = {}

        def match_subtree(node: IRTreeNode) -> float:
            node_id = id(node)
            if node_id in costs:
                return costs[node_id]
            min_cost = math.inf
            best_pattern: Optional[TreePattern] = None
            for pattern in patterns:
                if not pattern.matches(node):
                    continue
                current = float(pattern.cost)
                spanned_ids = {id(child) for child in pattern.spanned_children(node)}
                for child in node.children:
                    if id(child) not in spanned_ids:
                        current += match_subtree(child)
                if current < min_cost:
                    min_cost = current
                    best_pattern = pattern
            costs[node_id] = min_cost
            selected[node_id] = best_pattern
            return min_cost

        total_cost = match_subtree(root)
        covering = self._read_covering(root, selected) if math.isfinite(total_cost) else []
        return TreeSelectionResult(
            instructions=[pattern.instruction for _, pattern in covering
                          if pattern.instruction is not None],
            cost=total_cost,
            covering=covering,
            costs=costs,
            selected=selected,
            complete=math.isfinite(total_cost),
        )

    def _read_covering(self, root: IRTreeNode,
                       selected: Dict[int, Optional[TreePattern]]) -> List[Tuple[IRTreeNode, TreePattern]]:
        """Read the covering off the ``selected`` table: walk from the root
        and skip subtrees spanned by the chosen pattern (Theorem 4.1)."""
        covering: List[Tuple[IRTreeNode, TreePattern]] = []
        stack = [root]
        while stack:
            node = stack.pop()
            pattern = selected.get(id(node))
            if pattern is None:
                return []
            covering.append((node, pattern))
            spanned_ids = {id(child) for child in pattern.spanned_children(node)}
            stack.extend(reversed([child for child in node.children
                                   if id(child) not in spanned_ids]))
        return covering

    def _greedy_tree_cover(self, root: IRTreeNode, patterns: List[TreePattern]) -> TreeSelectionResult:
        """Heuristic baseline over trees: the shipped matcher applied per
        node, in isolation.

        Only single-node patterns are considered (the matcher has no
        notion of multi-node spans and cannot look ahead); each node picks
        the best-scoring pattern by the flat heuristic's name-overlap and
        parameter terms without its random exploration factor, with ties
        resolved to the cheaper instruction and then catalogue order.
        """
        single_node = [pattern for pattern in patterns if not pattern.spanned_child_ops]
        costs: Dict[int, float] = {}
        selected: Dict[int, Optional[TreePattern]] = {}

        stack: List[Tuple[IRTreeNode, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            node_id = id(node)
            if not expanded:
                stack.append((node, True))
                for child in reversed(node.children):
                    stack.append((child, False))
                continue
            candidates = [pattern for pattern in single_node if pattern.matches(node)]
            if not candidates:
                costs[node_id] = math.inf
                selected[node_id] = None
                continue
            best = min(candidates,
                       key=lambda pattern: (-self._tree_score(pattern, node),
                                            float(pattern.cost)))
            children_cost = sum(costs[id(child)] for child in node.children)
            costs[node_id] = float(best.cost) + children_cost
            selected[node_id] = best

        total_cost = costs.get(id(root), math.inf)
        covering = self._read_covering(root, selected) if math.isfinite(total_cost) else []
        return TreeSelectionResult(
            instructions=[pattern.instruction for _, pattern in covering
                          if pattern.instruction is not None],
            cost=total_cost,
            covering=covering,
            costs=costs,
            selected=selected,
            complete=math.isfinite(total_cost),
        )

    def _tree_score(self, pattern: TreePattern, node: IRTreeNode) -> float:
        """Deterministic name-overlap score of a pattern at a node: the
        flat heuristic's name and parameter terms without its random
        exploration factor (compare ``_score_instruction``)."""
        score = 0.0
        if str(pattern.root_op).lower() in node.op.lower():
            score += 1.0
        params = getattr(pattern.instruction, "params", None)
        required = set(params.keys()) if isinstance(params, dict) else set()
        provided = set(node.payload.keys()) if isinstance(node.payload, dict) else set()
        if required:
            score += len(required & provided) / len(required)
        return score

    def _score_instruction(self, instruction: Instruction, task: str, text: str, params: Dict[str, Any]) -> float:
        """Score an instruction based on task, text and parameters."""
        try:
            # Base score
            score = 0.0

            # Match instruction name with task and text
            name = instruction.name.lower()
            if name in task.lower():
                score += 1.0
            if name in text.lower():
                score += 0.5

            # Check parameter compatibility
            required_params = set(instruction.params.keys())
            provided_params = set(params.keys())
            param_match = len(required_params & provided_params) / max(len(required_params), 1)
            score += param_match

            # Add random exploration factor
            score += np.random.normal(0, 0.1)

            return score

        except Exception as e:
            self.logger.error(f"Instruction scoring failed: {str(e)}")
            return 0.0
