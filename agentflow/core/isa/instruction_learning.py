"""Instruction Learning System implementation following AgentISA paper Section 3.3."""

from typing import Dict, List, Any, Optional, Tuple, Set, Coroutine, Awaitable, Callable, cast
from dataclasses import dataclass, field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
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
    min_pattern_support: int = 3
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
        self.vectorizer = TfidfVectorizer()
        self.patterns: Dict[str, DialoguePattern] = {}
        self.cache = InstructionCache(
            max_size=self.config.cache_size,
            ttl=self.config.cache_ttl
        ) if self.config.enable_caching else None
        self.metrics = InstructionMetrics()

    async def extract_patterns(self, dialogues: List[Dict[str, Any]]) -> List[DialoguePattern]:
        """Extract patterns from dialogues using PCFG approach."""
        start_time = time.time()
        
        try:
            # Check cache if enabled
            if self.cache:
                cache_key = self._get_cache_key(dialogues)
                cached_patterns = self.cache.get(cache_key)
                if cached_patterns:
                    self._update_metrics(start_time, True, cache_hit=True)
                    return cached_patterns
            
            # Convert dialogues to feature vectors
            dialogue_texts = [self._dialogue_to_text(d) for d in dialogues]
            vectors = self.vectorizer.fit_transform(dialogue_texts)
            
            # Cluster similar dialogues
            clustering = DBSCAN(
                eps=0.3,
                min_samples=self.config.min_pattern_support
            )
            labels = clustering.fit_predict(vectors)
            
            # Group dialogues by cluster
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:  # Skip noise points
                    continue
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(dialogues[i])
            
            # Process clusters and extract patterns
            patterns: List[DialoguePattern] = []
            
            if self.config.enable_parallel and clusters:
                # Process clusters in parallel
                async def process_clusters():
                    tasks = []
                    for cluster_dialogues in clusters.values():
                        task = asyncio.create_task(
                            self._extract_pattern_from_cluster(cluster_dialogues)
                        )
                        tasks.append(task)
                    results = await asyncio.gather(*tasks)
                    return [p for p in results if p is not None]
                
                patterns = await process_clusters()
            else:
                # Process sequentially
                for cluster_dialogues in clusters.values():
                    pattern = await self._extract_pattern_from_cluster(cluster_dialogues)
                    if pattern is not None:
                        patterns.append(pattern)
            
            # Store patterns in instance dictionary
            for pattern in patterns:
                self.patterns[pattern.pattern_id] = pattern
            
            # Cache results if enabled
            if self.cache:
                self.cache.put(cache_key, patterns)
            
            self._update_metrics(start_time, True)
            return patterns
            
        except Exception as e:
            self._update_metrics(start_time, False)
            raise

    async def _extract_pattern_from_cluster(self, dialogues: List[Dict[str, Any]]) -> Optional[DialoguePattern]:
        """Extract a pattern from a cluster of dialogues."""
        if not dialogues:
            return None
            
        # Extract common elements
        pattern = await self._find_common_elements(dialogues)
        if not pattern:
            return None
            
        # Validate pattern
        if not self._validate_pattern(pattern):
            return None
            
        return pattern

    async def _find_common_elements(self, dialogues: List[Dict[str, Any]]) -> Optional[DialoguePattern]:
        """Find common elements across dialogues."""
        if not dialogues:
            return None
            
        # Extract common components
        components = []
        for dialogue in dialogues:
            for turn in dialogue.get("turns", []):
                if self._is_significant_turn(turn):
                    components.append(turn["content"])
        
        # Create pattern
        return DialoguePattern(
            pattern_id=f"pattern_{len(self.patterns)}",
            components=list(set(components)),
            frequency=len(dialogues),
            success_rate=sum(1 for d in dialogues if d.get("success", False)) / len(dialogues),
            context_requirements=set(components)
        )

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
        return role in ["user", "assistant"] and bool(content)

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

    async def learn_from_dialogues(self, dialogues: List[Dict[str, Any]]) -> List[InstructionTemplate]:
        """Learn new instructions from dialogues.
        
        Args:
            dialogues: List of dialogue dictionaries
            
        Returns:
            List[InstructionTemplate]: List of learned instruction templates
        """
        # Extract patterns
        patterns = await self.pattern_recognizer.extract_patterns(dialogues)
        
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
        """Create executable instruction from template."""
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
                # Add conditions from template
                for pre in template.preconditions:
                    self.add_precondition(lambda state: pre in state)
                for post in template.postconditions:
                    self.add_postcondition(lambda state: post in state)

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