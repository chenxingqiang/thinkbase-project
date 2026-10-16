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
    # -- Composite-replay provenance (Section 3.3). ----------------------
    # When the pattern was mined from instruction *traces* (rather than
    # from dialogues), these carry the recurring constituent subsequence
    # itself, the dataflow edges between its members, the output names
    # the subsequence was observed to produce, and the external inputs
    # it consumes -- everything a promoted learned instruction needs in
    # order to replay real behavior instead of shipping a stub body.
    # Patterns mined from dialogues leave them at their defaults.
    instruction_sequence: Optional[List[Instruction]] = None
    dataflow_edges: List[Tuple[str, str]] = field(default_factory=list)
    produced_outputs: Optional[List[str]] = None
    external_inputs: Optional[List[str]] = None

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
    # Constituent instruction sequence for composite replay.  ``None``
    # (or empty) means the template was mined without recorded bodies;
    # instructions created from such a template refuse to execute.
    components: Optional[List[Instruction]] = None
    dataflow_edges: List[Tuple[str, str]] = field(default_factory=list)

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

@dataclass
class _TraceEntry:
    """Normalized per-step record of an executed instruction."""
    instruction: Instruction
    success: bool
    # Output names this step produced, in production (first-seen) order:
    # the keys of its observed dict result, or its declared ``outputs``
    # when no result was recorded.
    output_keys: List[str]

async def record_trace(
    stream: List[Instruction],
    initial_ctx: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Execute ``stream`` sequentially and record a mining-ready trace.

    Follows the run_stream execution convention: every instruction
    receives ``{"ctx": <shared state>}`` and its dict outputs are merged
    into the shared state for later steps.  The caller's
    ``initial_ctx`` is copied, never mutated.  Each recorded entry is a
    dict ``{"instruction": ..., "success": True, "outputs": {...}}`` as
    consumed by :meth:`PatternRecognizer.extract_instruction_patterns`;
    a step that raises propagates its exception and is not recorded.
    """
    ctx: Dict[str, Any] = dict(initial_ctx or {})
    trace: List[Dict[str, Any]] = []
    for instr in stream:
        result = await instr.execute({"ctx": ctx})
        entry: Dict[str, Any] = {
            "instruction": instr,
            "success": True,
            "outputs": {},
        }
        if isinstance(result, dict):
            entry["outputs"] = dict(result)
            ctx.update(result)
        trace.append(entry)
    return trace

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

    # ------------------------------------------------------------------
    # Instruction-trace mining (composite replay, Section 3.3)
    # ------------------------------------------------------------------

    def extract_instruction_patterns(
        self, traces: List[List[Any]]
    ) -> List[DialoguePattern]:
        """Mine recurring contiguous instruction subsequences from traces.

        Each trace is a sequence of per-step records -- either a bare
        :class:`Instruction` or a dict as produced by
        :func:`record_trace` (``{"instruction", "success", "outputs"}``).
        All contiguous windows of two or more steps are enumerated and
        keyed by their instruction-id sequence.  A key qualifies when it
        occurs at least ``min_pattern_support`` times across all traces
        and its occurrences succeed at rate at least
        ``min_success_rate`` (an occurrence succeeds when every one of
        its steps did).  Keys that occur contiguously inside another
        qualifying key are pruned, so only *maximal* recurring
        sequences are promoted.

        The procedure is fully deterministic: windows are enumerated in
        (trace, start, length) order, promotion order is longest-first
        then first occurrence, and pattern ids are derived from a
        content hash of the id sequence -- the same traces therefore
        promote the same patterns, with the same ids, in the same
        order, across independent runs.
        """
        start_time = time.time()
        try:
            sequences = [self._normalize_trace(trace) for trace in traces]

            occurrences: Dict[
                Tuple[str, ...], List[Tuple[int, int, List[_TraceEntry]]]
            ] = {}
            for trace_idx, entries in enumerate(sequences):
                for length in range(2, len(entries) + 1):
                    for start in range(len(entries) - length + 1):
                        window = entries[start:start + length]
                        key = tuple(entry.instruction.id for entry in window)
                        occurrences.setdefault(key, []).append(
                            (trace_idx, start, window)
                        )

            qualifying = {
                key: occs for key, occs in occurrences.items()
                if len(occs) >= self.config.min_pattern_support
                and self._window_success_rate(occs)
                >= self.config.min_success_rate
            }
            promoted_keys = [
                key for key in qualifying
                if not any(
                    key != other
                    and self._is_contiguous_subsequence(key, other)
                    for other in qualifying
                )
            ]
            promoted_keys.sort(
                key=lambda k: (-len(k), occurrences[k][0][0], occurrences[k][0][1], k)
            )

            patterns: List[DialoguePattern] = []
            for key in promoted_keys:
                pattern = self._pattern_from_window(key, occurrences[key])
                if pattern is not None:
                    patterns.append(pattern)
            for pattern in patterns:
                self.patterns[pattern.pattern_id] = pattern

            self._update_metrics(start_time, True)
            return patterns
        except Exception:
            self._update_metrics(start_time, False)
            raise

    def _normalize_trace(self, trace: List[Any]) -> List[_TraceEntry]:
        """Normalize a raw trace into per-step records.

        Dict records carry the observed success flag and the observed
        output keys; bare instructions default to success with their
        declared ``outputs``.
        """
        entries: List[_TraceEntry] = []
        for record in trace or []:
            if isinstance(record, dict):
                instruction = record.get("instruction")
                if not isinstance(instruction, Instruction):
                    raise ValueError(
                        "Trace record dicts must carry an 'instruction' entry"
                    )
                success = bool(record.get("success", True))
                outputs = record.get("outputs")
                if isinstance(outputs, dict):
                    output_keys = list(outputs.keys())
                else:
                    output_keys = list(getattr(instruction, "outputs", []) or [])
            else:
                instruction = record
                if not isinstance(instruction, Instruction):
                    raise ValueError(
                        "Trace records must be Instructions or dicts"
                    )
                success = True
                output_keys = list(getattr(instruction, "outputs", []) or [])
            entries.append(_TraceEntry(instruction, success, output_keys))
        return entries

    @staticmethod
    def _window_success_rate(
        occurrences: List[Tuple[int, int, List[_TraceEntry]]]
    ) -> float:
        """Fraction of a window's occurrences whose steps all succeeded."""
        if not occurrences:
            return 0.0
        successful = sum(
            1 for _, _, window in occurrences
            if all(entry.success for entry in window)
        )
        return successful / len(occurrences)

    @staticmethod
    def _is_contiguous_subsequence(
        needle: Tuple[str, ...], haystack: Tuple[str, ...]
    ) -> bool:
        """Whether ``needle`` occurs contiguously inside ``haystack``."""
        n, h = len(needle), len(haystack)
        if n == 0 or n > h:
            return False
        return any(
            tuple(haystack[i:i + n]) == needle for i in range(h - n + 1)
        )

    def _pattern_from_window(
        self,
        key: Tuple[str, ...],
        occurrences: List[Tuple[int, int, List[_TraceEntry]]],
    ) -> DialoguePattern:
        """Build a replayable pattern from a promoted window.

        The constituent sequence is taken from the *first* occurrence
        (deterministic).  The pattern records the dataflow edges
        between its members (producer id -> consumer id), the output
        names it was observed to produce (first-seen order), and its
        external inputs: dependencies satisfied by neither an earlier
        member's outputs nor an earlier member's id.  The pattern id is
        a content hash of the instruction-id sequence, so promotion is
        reproducible across recognizer instances and runs.
        """
        _, _, window = occurrences[0]
        components = [entry.instruction for entry in window]

        producer_of: Dict[str, str] = {}
        dataflow_edges: List[Tuple[str, str]] = []
        produced_outputs: List[str] = []
        external_inputs: List[str] = []

        for position, entry in enumerate(window):
            instruction = entry.instruction
            earlier_ids = {
                prior.instruction.id for prior in window[:position]
            }
            for dep in list(getattr(instruction, "dependencies", []) or []):
                edge: Optional[Tuple[str, str]] = None
                if dep in producer_of:
                    edge = (producer_of[dep], instruction.id)
                elif dep in earlier_ids:
                    edge = (dep, instruction.id)
                elif dep not in external_inputs:
                    external_inputs.append(dep)
                if edge is not None and edge not in dataflow_edges:
                    dataflow_edges.append(edge)
            for out in entry.output_keys:
                if out not in producer_of:
                    producer_of[out] = instruction.id
                if out not in produced_outputs:
                    produced_outputs.append(out)

        pattern_id = "pattern_" + hashlib.sha256(
            json.dumps(list(key)).encode()
        ).hexdigest()[:12]
        return DialoguePattern(
            pattern_id=pattern_id,
            components=[instruction.name for instruction in components],
            frequency=len(occurrences),
            success_rate=self._window_success_rate(occurrences),
            context_requirements=set(external_inputs),
            instruction_sequence=components,
            dataflow_edges=dataflow_edges,
            produced_outputs=produced_outputs,
            external_inputs=external_inputs,
        )

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

    def learn_from_traces(self, traces: List[List[Any]]) -> List[InstructionTemplate]:
        """Learn replayable instructions from execution traces.

        Mirrors :meth:`learn_from_dialogues`, but mines recurring
        contiguous instruction subsequences; each promoted template
        carries its constituent sequence (with the dataflow edges
        between its members), so the instruction built from it executes
        by composite replay instead of refusing.

        Args:
            traces: List of traces; each trace is a list of per-step
                records as produced by :func:`record_trace`.

        Returns:
            List[InstructionTemplate]: Promoted templates, in
                deterministic promotion order.
        """
        patterns = self.pattern_recognizer.extract_instruction_patterns(traces)

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
                postconditions=postconditions,
                components=pattern.instruction_sequence,
                dataflow_edges=pattern.dataflow_edges
            )
        except Exception:
            return None

    def _determine_instruction_type(self, pattern: DialoguePattern) -> InstructionType:
        """Determine instruction type from pattern.

        Trace-mined patterns replay a constituent sequence: the shared
        type when every member agrees, otherwise CONTROL_FLOW (the type
        the composite machinery uses for sequential composition).
        Dialogue-mined patterns keep the existing LLM_INTERACTION
        default.
        """
        sequence = pattern.instruction_sequence
        if sequence:
            types = {instruction.type for instruction in sequence}
            if len(types) == 1:
                return next(iter(types))
            return InstructionType.CONTROL_FLOW
        return InstructionType.LLM_INTERACTION

    def _extract_parameters(self, pattern: DialoguePattern) -> Dict[str, Any]:
        """Extract parameters from pattern."""
        params = {}
        # Trace mining derives no type information for the external
        # inputs; they are captured as preconditions instead.
        return params

    def _generate_preconditions(self, pattern: DialoguePattern) -> List[str]:
        """Generate preconditions from pattern.

        Preconditions name the data keys the instruction requires.
        Trace-mined patterns list their external inputs in mined
        order; dialogue-mined patterns sort their context requirements
        so promotion is reproducible across runs.
        """
        if pattern.external_inputs is not None:
            return [f"requires_{req}" for req in pattern.external_inputs]
        return [
            f"requires_{req}"
            for req in sorted(pattern.context_requirements)
        ]

    def _generate_postconditions(self, pattern: DialoguePattern) -> List[str]:
        """Generate postconditions from pattern.

        Trace-mined patterns declare every output name the mined
        subsequence was observed to produce (first-seen order);
        dialogue-mined patterns declare none.
        """
        if pattern.produced_outputs is not None:
            return [f"produces_{out}" for out in pattern.produced_outputs]
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
                declared_outputs = [
                    post.removeprefix("produces_")
                    for post in template.postconditions
                ]
                super().__init__(
                    id=f"learned_{template.name}",
                    name=template.name,
                    type=template.type,
                    params=template.params,
                    description=f"Learned instruction from pattern {template.name}",
                    cache_config=template.cache_config,
                    parallel_config=template.parallel_config,
                    outputs=declared_outputs,
                )
                self.template = template
                self.components = (
                    list(template.components) if template.components else []
                )
                self.dataflow_edges = list(template.dataflow_edges)
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
                """Execute by composite replay of the mined subsequence.

                When the template carries its constituent instruction
                sequence, replay is plain sequential composition: the
                components run in mined order against one shared
                context built from the inputs (run_stream convention),
                each component's dict outputs merge into that context,
                and the keys the template's ``produces_X``
                postconditions declare are returned.  Nothing smarter
                happens -- no generalization beyond the recorded
                subsequence.  A template mined without bodies refuses
                to execute rather than silently no-opping.
                """
                start_time = time.time()
                try:
                    if not self.validate_types(input_params):
                        raise TypeError("Invalid parameter types")

                    # Shared replay context: the wrapped ``{"ctx": ...}``
                    # convention when the caller uses it, the flat
                    # parameter dict otherwise.  Copied, never mutated
                    # in place for the caller.
                    if isinstance(input_params.get("ctx"), dict):
                        replay_ctx = dict(input_params["ctx"])
                    else:
                        replay_ctx = dict(input_params)

                    # Preconditions speak about the data the sequence
                    # consumes: the replay context, plus any sibling
                    # parameters the caller passed alongside it.
                    condition_state = {
                        **{k: v for k, v in input_params.items() if k != "ctx"},
                        **replay_ctx,
                    }
                    if not self.verify_conditions(condition_state):
                        raise ValueError("Preconditions not met")

                    if not self.components:
                        raise NotImplementedError(
                            f"Learned instruction '{self.name}' has no "
                            "constituent instruction sequence to replay: "
                            "its template was mined without execution "
                            "bodies. Re-mine the pattern from instruction "
                            "traces (InstructionLearner.learn_from_traces) "
                            "to attach a replayable body."
                        )

                    produced: Dict[str, Any] = {}
                    for component in self.components:
                        result = await component.execute({"ctx": replay_ctx})
                        if isinstance(result, dict):
                            replay_ctx.update(result)
                            produced.update(result)

                    declared = [
                        post.removeprefix("produces_")
                        for post in self.template.postconditions
                    ]
                    if declared:
                        missing = [
                            key for key in declared if key not in replay_ctx
                        ]
                        if missing:
                            raise RuntimeError(
                                f"Learned instruction '{self.name}' failed "
                                f"to produce declared output(s) {missing} "
                                f"after replaying {len(self.components)} "
                                "constituent instructions"
                            )
                        result_out = {key: replay_ctx[key] for key in declared}
                    else:
                        result_out = dict(produced)

                    self._update_metrics(start_time, True)
                    return result_out
                except Exception:
                    self._update_metrics(start_time, False)
                    raise

        return LearnedInstruction(template)