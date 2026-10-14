"""Instruction Learning System implementation following AgentISA paper Section 3.3."""

from typing import Dict, List, Any, Optional, Tuple, Set, Coroutine, Awaitable, Callable, cast
from dataclasses import dataclass, field
import asyncio
import time
import json
import hashlib
from collections import OrderedDict

from .instruction import (
    Instruction,
    InstructionType,
    TypeConstraint,
    InstructionMetrics,
    InstructionCache
)
from .agentic_calling import AgenticTransformer

@dataclass
class DialoguePattern:
    """Dialogue pattern representation."""
    pattern_id: str
    components: List[str]
    frequency: int
    success_rate: float
    context_requirements: Set[str]
    metrics: InstructionMetrics = field(default_factory=InstructionMetrics)

@dataclass
class InstructionTemplate:
    """Template for generating new instructions."""
    name: str
    type: InstructionType
    params: Dict[str, Any]
    preconditions: List[str]
    postconditions: List[str]
    cache_config: Optional[Dict[str, Any]] = None
    parallel_config: Optional[Dict[str, Any]] = None

@dataclass
class LearningConfig:
    """Configuration for learning system."""
    min_pattern_support: int = 2
    min_success_rate: float = 0.7
    enable_caching: bool = True
    cache_size: int = 1000
    cache_ttl: int = 3600
    enable_parallel: bool = True
    parallel_batch_size: int = 10
    max_parallel_tasks: int = 100

class PatternRecognizer:
    """Dialogue pattern recognition system."""
    
    def __init__(self, config: Optional[LearningConfig] = None):
        self.config = config or LearningConfig()
        self.patterns: Dict[str, DialoguePattern] = {}
        self.cache = InstructionCache(
            max_size=self.config.cache_size,
            ttl=self.config.cache_ttl
        ) if self.config.enable_caching else None
        self.metrics = InstructionMetrics()

    def extract_patterns(self, dialogues: List[Dict[str, Any]]) -> List[DialoguePattern]:
        """Extract recurring patterns from dialogues.

        Deterministic grouping: dialogues are bucketed by their action
        signature (the sorted set of system-turn contents), and each
        bucket with at least ``min_pattern_support`` members whose
        success rate clears ``min_success_rate`` is promoted into a
        :class:`DialoguePattern`.  This replaces the earlier
        TF-IDF/DBSCAN clustering, which was non-deterministic across
        sklearn versions and failed small-sample inputs.
        """
        start_time = time.time()

        try:
            # Check cache if enabled
            cache_key = None
            if self.cache:
                cache_key = self._get_cache_key(dialogues)
                cached_patterns = self.cache.get(cache_key)
                if cached_patterns:
                    self._update_metrics(start_time, True, cache_hit=True)
                    return cached_patterns

            groups: Dict[Tuple[str, ...], List[Dict[str, Any]]] = {}
            for dialogue in dialogues:
                signature = self._action_signature(dialogue)
                groups.setdefault(signature, []).append(dialogue)

            patterns: List[DialoguePattern] = []
            for signature, group_dialogues in groups.items():
                pattern = self._extract_pattern_from_cluster(
                    signature, group_dialogues
                )
                if pattern is not None:
                    patterns.append(pattern)

            # Store patterns in instance dictionary
            for pattern in patterns:
                self.patterns[pattern.pattern_id] = pattern

            # Cache results if enabled
            if self.cache and cache_key is not None:
                self.cache.put(cache_key, patterns)

            self._update_metrics(start_time, True)
            return patterns

        except Exception:
            self._update_metrics(start_time, False)
            raise

    def _action_signature(self, dialogue: Dict[str, Any]) -> Tuple[str, ...]:
        """Bucketing key for a dialogue: its sorted system-turn contents."""
        return tuple(sorted(
            turn.get("content", "")
            for turn in dialogue.get("turns", [])
            if turn.get("role") == "system"
        ))

    def _extract_pattern_from_cluster(
        self,
        signature: Tuple[str, ...],
        dialogues: List[Dict[str, Any]],
    ) -> Optional[DialoguePattern]:
        """Extract a pattern from one action-signature group."""
        if not dialogues:
            return None

        components = self._find_common_components(dialogues)
        if not components:
            return None

        pattern = DialoguePattern(
            pattern_id=f"pattern_{len(self.patterns)}",
            components=components,
            frequency=len(dialogues),
            success_rate=(
                sum(1 for d in dialogues if d.get("success", False))
                / len(dialogues)
            ),
            context_requirements=self._extract_context_requirements(dialogues),
        )

        if not self._validate_pattern(pattern):
            return None
        return pattern

    def _find_common_components(
        self, dialogues: List[Dict[str, Any]]
    ) -> List[str]:
        """Turn contents shared by every dialogue in the group.

        The result keeps first-seen order so repeated extraction over
        the same group is stable.
        """
        if not dialogues:
            return []

        common: Optional[Set[str]] = None
        order: List[str] = []
        for dialogue in dialogues:
            contents = {
                turn["content"]
                for turn in dialogue.get("turns", [])
                if self._is_significant_turn(turn)
            }
            if common is None:
                common = contents
                order = [
                    turn["content"]
                    for turn in dialogue.get("turns", [])
                    if self._is_significant_turn(turn)
                ]
            else:
                common &= contents
        return [c for c in order if c in (common or set())]

    def _extract_context_requirements(
        self, dialogues: List[Dict[str, Any]]
    ) -> Set[str]:
        """Union of the context keys demanded by the group's dialogues."""
        requirements: Set[str] = set()
        for dialogue in dialogues:
            context = dialogue.get("context", {})
            if isinstance(context, dict):
                requirements.update(context.keys())
        return requirements

    def _dialogue_to_text(self, dialogue: Dict[str, Any]) -> str:
        """Convert dialogue to text for vectorization."""
        text_parts = []
        for turn in dialogue.get("turns", []):
            if self._is_significant_turn(turn):
                text_parts.append(f"{turn['role']}: {turn['content']}")
        return " ".join(text_parts)

    def _is_significant_turn(self, turn: Dict[str, Any]) -> bool:
        """Check if dialogue turn is significant for pattern extraction."""
        role = cast(Optional[str], turn.get("role"))
        content = cast(Optional[str], turn.get("content"))
        return role in ["system", "user", "assistant"] and bool(content)

    def _validate_pattern(self, pattern: DialoguePattern) -> bool:
        """Validate extracted pattern."""
        if not pattern.components:
            return False
        if pattern.frequency < self.config.min_pattern_support:
            return False
        if pattern.success_rate < self.config.min_success_rate:
            return False
        return True

    def _get_cache_key(self, dialogues: List[Dict[str, Any]]) -> str:
        """Generate cache key for dialogues."""
        dialogue_str = json.dumps(dialogues, sort_keys=True)
        return hashlib.sha256(dialogue_str.encode()).hexdigest()

    def _update_metrics(self, start_time: float, success: bool, cache_hit: bool = False):
        """Update instruction metrics."""
        self.metrics.total_executions += 1
        if success:
            self.metrics.successful_executions += 1
        else:
            self.metrics.failed_executions += 1
        
        self.metrics.total_execution_time += time.time() - start_time
        
        if cache_hit:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1

class InstructionLearner:
    """Instruction learning and optimization system."""
    
    def __init__(self, transformer: AgenticTransformer):
        self.transformer = transformer
        self.pattern_recognizer = PatternRecognizer()
        self.instruction_templates: Dict[str, InstructionTemplate] = {}

    def learn_from_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[InstructionTemplate]:
        """Learn new instructions from dialogues.
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List[InstructionTemplate]: List of learned instruction templates
        """
        # Extract patterns
        patterns = self.pattern_recognizer.extract_patterns(dialogues)
        
        # Convert patterns to instruction templates
        templates: List[InstructionTemplate] = []
        for pattern in patterns:
            template = self._pattern_to_template(pattern)
            if template:
                self.instruction_templates[template.name] = template
                templates.append(template)
        
        return templates

    def _pattern_to_template(self, pattern: DialoguePattern) -> Optional[InstructionTemplate]:
        """Convert dialogue pattern to instruction template."""
        try:
            # Generate template name
            name = f"learned_{pattern.pattern_id}"
            
            # Determine instruction type based on components
            instr_type = self._determine_instruction_type(pattern)
            
            # Extract parameters
            params = self._extract_parameters(pattern)
            
            # Generate conditions
            preconditions = self._generate_preconditions(pattern)
            postconditions = self._generate_postconditions(pattern)
            
            return InstructionTemplate(
                name=name,
                type=instr_type,
                params=params,
                preconditions=preconditions,
                postconditions=postconditions
            )
        except Exception:
            return None

    def _determine_instruction_type(self, pattern: DialoguePattern) -> InstructionType:
        """Determine instruction type from pattern."""
        # Add logic to determine instruction type
        return InstructionType.LLM_INTERACTION

    def _extract_parameters(self, pattern: DialoguePattern) -> Dict[str, Any]:
        """Extract parameters from pattern."""
        params = {}
        # Add parameter extraction logic
        return params

    def _generate_preconditions(self, pattern: DialoguePattern) -> List[str]:
        """Generate preconditions from pattern."""
        conditions = []
        for req in pattern.context_requirements:
            conditions.append(f"requires_{req}")
        return conditions

    def _generate_postconditions(self, pattern: DialoguePattern) -> List[str]:
        """Generate postconditions from pattern."""
        # Add postcondition generation logic
        return []

    def create_instruction(self, template: InstructionTemplate) -> Instruction:
        """Create executable instruction from template.

        Raises:
            ValueError: If the template is structurally invalid (empty
                name or missing type).
        """
        if not template.name:
            raise ValueError("Instruction template requires a non-empty name")
        if not isinstance(template.type, InstructionType):
            raise ValueError("Instruction template requires a valid type")

        # Create a new instruction class for this template
        class LearnedInstruction(Instruction):
            def __init__(self, template: InstructionTemplate):
                super().__init__(
                    id=f"learned_{template.name}",
                    name=template.name,
                    type=template.type,
                    params=template.params,
                    description=f"Learned instruction from pattern {template.name}"
                )
                # Add conditions from template: a precondition named
                # ``requires_X`` (or a bare parameter name) demands the
                # parameter ``X`` be present in the input state.
                for pre in template.preconditions:
                    required = pre.removeprefix("requires_")
                    self.add_precondition(
                        lambda state, required=required: required in state
                    )
                for post in template.postconditions:
                    expected = post.removeprefix("produces_")
                    self.add_postcondition(
                        lambda state, expected=expected: expected in state
                    )

            async def execute(self, input_params: Dict[str, Any]) -> Any:
                # Implement execution logic
                if not self.validate_types(input_params):
                    raise TypeError("Invalid parameter types")
                if not self.verify_conditions(input_params):
                    raise ValueError("Preconditions not met")
                
                # Execute the learned behavior
                # This would be implemented based on the pattern
                return {"status": "executed", "params": input_params}

        return LearnedInstruction(template) 