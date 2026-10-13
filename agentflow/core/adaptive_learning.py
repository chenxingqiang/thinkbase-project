"""Adaptive learning for agent instruction optimization.

This module implements pattern discovery and instruction optimization on
top of the agentic-calling primitives (:class:`AgentInstruction`,
:class:`FunctionContext`, :class:`ExecutionResult` from
:mod:`agentflow.agents.agentic_calling`):

* :class:`PatternMatcher` aggregates instruction-sequence observations
  (success, latency, context keys) into :class:`InstructionPattern`s.
* :class:`AdaptiveLearner` learns from single instruction executions and
  produces optimized variants of individual instructions.
* :class:`AdaptiveOptimizer` learns from whole instruction sequences and
  rewrites those sequences with optimized instruction variants.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

from agentflow.agents.agentic_calling import AgentInstruction

__all__ = [
    "AdaptiveLearner",
    "AdaptiveMetrics",
    "AdaptiveOptimizer",
    "InstructionPattern",
    "PatternMatcher",
]


@dataclass
class InstructionPattern:
    """A recurring execution pattern observed for an instruction sequence."""

    sequence: List[str]
    frequency: int
    success_rate: float
    avg_latency: float
    context_dependencies: Set[str] = field(default_factory=set)


@dataclass
class AdaptiveMetrics:
    """Metrics tracked during adaptive learning and optimization."""

    pattern_discoveries: int = 0
    instructions_optimized: int = 0
    total_observations: int = 0
    learning_cycles: int = 0


def _extract_name(instruction: Any) -> str:
    """Best-effort extraction of an instruction's name."""
    return getattr(instruction, "name", None) or str(instruction)


def _result_success(result: Any) -> bool:
    """Whether an execution result counts as successful."""
    if result is None:
        return False
    return getattr(result, "status", "success") == "success"


def _result_latency(result: Any) -> float:
    """Extract the execution latency from a result's metrics."""
    metrics = getattr(result, "metrics", None) or {}
    try:
        return float(metrics.get("execution_time", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _context_keys(context: Any) -> Set[str]:
    """Extract the variable names an execution depended on."""
    variables = getattr(context, "variables", None) or {}
    return set(variables.keys())


def _build_optimized_instruction(
    instruction: AgentInstruction,
    pattern: InstructionPattern,
) -> AgentInstruction:
    """Create an optimized variant of ``instruction`` based on ``pattern``."""
    hints: Dict[str, Any] = dict(
        getattr(instruction, "optimization_hints", None) or {}
    )
    hints.update(
        {
            "pattern_optimized": True,
            "original_name": _extract_name(instruction),
            "context_dependencies": sorted(pattern.context_dependencies),
            "pattern_frequency": pattern.frequency,
            "pattern_success_rate": pattern.success_rate,
            "pattern_avg_latency": pattern.avg_latency,
        }
    )
    return AgentInstruction(
        name=f"{_extract_name(instruction)}_optimized",
        func=getattr(instruction, "_func", None)
        or getattr(instruction, "func", None),
        parameters=instruction.parameters,
        has_context_embedding=getattr(instruction, "has_context_embedding", True),
        model_compatibility=list(
            getattr(instruction, "model_compatibility", None) or []
        ),
        optimization_hints=hints,
    )


class PatternMatcher:
    """Aggregates instruction-sequence observations into patterns."""

    def __init__(
        self,
        min_frequency: int = 2,
        min_success_rate: float = 0.7,
    ):
        self.min_frequency = min_frequency
        self.min_success_rate = min_success_rate
        # sequence key -> list of (success, latency, context_keys)
        self._observations: Dict[Tuple[str, ...], List[Tuple[bool, float, Set[str]]]] = {}

    @property
    def pattern_count(self) -> int:
        """Number of distinct sequences observed so far."""
        return len(self._observations)

    def add_sequence(
        self,
        sequence: List[str],
        success: bool,
        latency: float,
        context_keys: Optional[Set[str]] = None,
    ) -> None:
        """Record one observation of an instruction sequence.

        Args:
            sequence: Names of the instructions, in execution order.
            success: Whether that execution succeeded.
            latency: Execution latency in seconds.
            context_keys: Context variable names the execution touched.
        """
        key = tuple(sequence)
        self._observations.setdefault(key, []).append(
            (bool(success), float(latency), set(context_keys or ()))
        )

    def get_pattern(self, sequence: List[str]) -> Optional[InstructionPattern]:
        """Return the raw aggregated pattern for ``sequence`` (no thresholds)."""
        observations = self._observations.get(tuple(sequence))
        if not observations:
            return None
        frequency = len(observations)
        successes = sum(1 for ok, _, _ in observations if ok)
        dependencies: Set[str] = set()
        for _, _, keys in observations:
            dependencies |= keys
        return InstructionPattern(
            sequence=list(sequence),
            frequency=frequency,
            success_rate=successes / frequency,
            avg_latency=sum(lat for _, lat, _ in observations) / frequency,
            context_dependencies=dependencies,
        )

    def find_patterns(self) -> List[InstructionPattern]:
        """Return all patterns that meet the configured thresholds.

        A pattern qualifies when it was observed at least ``min_frequency``
        times with a success rate of at least ``min_success_rate``.
        """
        patterns = []
        for key in self._observations:
            pattern = self.get_pattern(list(key))
            if pattern is None:
                continue
            if (
                pattern.frequency >= self.min_frequency
                and pattern.success_rate >= self.min_success_rate
            ):
                patterns.append(pattern)
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        return patterns


class AdaptiveLearner:
    """Learns execution patterns for individual instructions.

    Unlike :class:`AdaptiveOptimizer`, the learner records *single*
    instruction executions (so a first observation is already enough to
    expose the pattern to :meth:`optimize_instruction`) and optimizes one
    instruction at a time.
    """

    def __init__(self, pattern_matcher: Optional[PatternMatcher] = None):
        self.pattern_matcher = pattern_matcher or PatternMatcher()
        self.learned_patterns: Dict[Tuple[str, ...], InstructionPattern] = {}
        self.metrics = AdaptiveMetrics()

    async def learn_from_execution(
        self,
        instruction: AgentInstruction,
        result: Any,
        context: Any = None,
    ) -> None:
        """Learn from a single instruction execution.

        Args:
            instruction: The instruction that was executed.
            result: The :class:`ExecutionResult` produced.
            context: The preserved execution context.
        """
        name = _extract_name(instruction)
        self.pattern_matcher.add_sequence(
            [name],
            _result_success(result),
            _result_latency(result),
            _context_keys(context if context is not None
                          else getattr(result, "preserved_context", None)),
        )
        self.metrics.total_observations += 1

        key = (name,)
        pattern = self.pattern_matcher.get_pattern([name])
        if pattern is None:
            return
        if key not in self.learned_patterns:
            self.metrics.pattern_discoveries += 1
        self.learned_patterns[key] = pattern

    async def optimize_instruction(
        self, instruction: AgentInstruction
    ) -> Optional[AgentInstruction]:
        """Return an optimized variant of ``instruction``.

        Returns ``None`` when no execution pattern has been learned for the
        instruction yet.
        """
        pattern = self.learned_patterns.get((_extract_name(instruction),))
        if pattern is None:
            return None
        self.metrics.instructions_optimized += 1
        return _build_optimized_instruction(instruction, pattern)

    def get_learned_patterns(self) -> List[InstructionPattern]:
        """Return all learned (raw) patterns."""
        return list(self.learned_patterns.values())

    def reset(self) -> None:
        """Forget everything learned so far."""
        self.pattern_matcher = PatternMatcher(
            min_frequency=self.pattern_matcher.min_frequency,
            min_success_rate=self.pattern_matcher.min_success_rate,
        )
        self.learned_patterns.clear()
        self.metrics = AdaptiveMetrics()


class AdaptiveOptimizer:
    """Optimizes instruction sequences from learned execution patterns."""

    def __init__(self, pattern_matcher: Optional[PatternMatcher] = None):
        self.pattern_matcher = pattern_matcher or PatternMatcher()
        self.metrics = AdaptiveMetrics()
        # original instruction name -> currently active optimized variant
        self._optimized_instructions: Dict[str, AgentInstruction] = {}
        self._active_optimizations: Set[str] = set()
        self._optimization_history: List[Dict[str, Any]] = []

    async def learn_from_sequence(
        self,
        instructions: List[AgentInstruction],
        results: List[Any],
        contexts: List[Any],
    ) -> None:
        """Learn from a full instruction-sequence execution.

        Records one observation per instruction (for per-instruction
        patterns) plus one observation for the whole sequence.

        Args:
            instructions: The instructions that were executed, in order.
            results: Their execution results, in order.
            contexts: Their preserved contexts, in order.
        """
        instructions = list(instructions)
        results = list(results)
        contexts = list(contexts)

        for index, instruction in enumerate(instructions):
            result = results[index] if index < len(results) else None
            context = contexts[index] if index < len(contexts) else None
            self.pattern_matcher.add_sequence(
                [_extract_name(instruction)],
                _result_success(result),
                _result_latency(result),
                _context_keys(context),
            )

        if instructions:
            latencies = [_result_latency(r) for r in results]
            dependencies: Set[str] = set()
            for context in contexts:
                dependencies |= _context_keys(context)
            self.pattern_matcher.add_sequence(
                [_extract_name(i) for i in instructions],
                all(_result_success(r) for r in results) if results else False,
                sum(latencies) / len(latencies) if latencies else 0.0,
                dependencies,
            )

        self.metrics.learning_cycles += 1

    def _pattern_qualifies(self, pattern: Optional[InstructionPattern]) -> bool:
        """Whether a pattern meets the matcher's optimization thresholds."""
        return (
            pattern is not None
            and pattern.frequency >= self.pattern_matcher.min_frequency
            and pattern.success_rate >= self.pattern_matcher.min_success_rate
        )

    async def optimize_instruction_sequence(
        self, instructions: List[AgentInstruction]
    ) -> List[AgentInstruction]:
        """Return the sequence with learned patterns applied.

        Instructions that have a qualifying learned pattern are replaced by
        their optimized variants; all others are returned unchanged.
        """
        optimized: List[AgentInstruction] = []
        optimized_count = 0
        for instruction in instructions:
            name = _extract_name(instruction)
            pattern = self.pattern_matcher.get_pattern([name])
            if self._pattern_qualifies(pattern):
                variant = self._optimized_instructions.get(name)
                if variant is None:
                    variant = _build_optimized_instruction(instruction, pattern)
                    self._optimized_instructions[name] = variant
                self._active_optimizations.add(name)
                optimized.append(variant)
                optimized_count += 1
            else:
                optimized.append(instruction)

        self._optimization_history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "sequence": [_extract_name(i) for i in instructions],
                "total": len(instructions),
                "optimized_count": optimized_count,
            }
        )
        if optimized_count:
            self.metrics.instructions_optimized += optimized_count
        return optimized

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Return a snapshot of the optimizer's statistics."""
        return {
            "total_patterns": self.pattern_matcher.pattern_count,
            "active_optimizations": len(self._active_optimizations),
            "learning_cycles": self.metrics.learning_cycles,
            "instructions_optimized": self.metrics.instructions_optimized,
            "optimization_history": list(self._optimization_history),
        }

    def reset_optimization_state(self) -> None:
        """Reset all learned patterns and optimization state."""
        self.pattern_matcher = PatternMatcher(
            min_frequency=self.pattern_matcher.min_frequency,
            min_success_rate=self.pattern_matcher.min_success_rate,
        )
        self.metrics = AdaptiveMetrics()
        self._optimized_instructions.clear()
        self._active_optimizations.clear()
        self._optimization_history.clear()
