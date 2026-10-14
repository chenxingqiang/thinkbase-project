"""Intelligent instruction selection and optimization"""

from typing import Dict, Any, List, Set, Optional, Union, Type, Callable
import logging
from dataclasses import dataclass
from .adaptive_weights import AdaptiveWeights, WeightConfig
from .isa.instruction import Instruction
from .agent_config import AgentConfig
from .base_types import AgentType, AgentMode, AgentStatus
from .exceptions import ConfigurationError

logger = logging.getLogger(__name__)

@dataclass
class InstructionScore:
    """Score for an instruction based on various factors"""
    name: str
    relevance: float  # Semantic relevance to input
    cost: float      # Computational cost
    success_rate: float  # Historical success rate
    cache_hit_rate: float  # Cache effectiveness
    final_score: float  # Combined score


def _ensure_heavy_deps() -> None:
    """Lazily load numpy/sklearn on first use.

    sklearn costs ~5s at import time; the selector is often constructed
    but rarely needs the TF-IDF machinery, so keep the module import
    cheap and pull the heavy deps in only when the selector is built.
    """
    if "TfidfVectorizer" not in globals():
        import numpy as _np
        from sklearn.feature_extraction.text import TfidfVectorizer as _tfidf
        from sklearn.metrics.pairwise import cosine_similarity as _cos
        globals().update(
            np=_np, TfidfVectorizer=_tfidf, cosine_similarity=_cos
        )


class InstructionSelector:
    """Intelligent instruction selection and optimization"""
    
    def __init__(self):
        _ensure_heavy_deps()
        self.vectorizer = TfidfVectorizer()
        # Fit with a dummy document to ensure it's always fitted
        self.vectorizer.fit(["dummy document"])
        self.instruction_vectors = {}
        self.instruction_history = {}
        
        # Initialize adaptive weights
        self.adaptive_weights = AdaptiveWeights([
            "relevance",
            "success_rate",
            "cost",
            "cache_hit_rate"
        ])
        
        # A/B testing state
        self.ab_testing_enabled = False
        self.variants = {}
    
    def enable_ab_testing(self):
        """Enable A/B testing"""
        self.ab_testing_enabled = True
    
    def register_variant(self, name: str, instructions: Dict[str, Instruction]):
        """Register a variant for A/B testing"""
        self.variants[name] = instructions
    
    def train(self, instructions: Dict[str, Instruction]):
        """Train the selector on available instructions"""
        if not instructions:
            # Initialize with a dummy document to avoid empty vocabulary error
            self.instruction_vectors = self.vectorizer.transform(['dummy'])
            self.instruction_vectors = None  # Clear the vectors after initialization
            return

        # Create TF-IDF vectors for instruction descriptions
        descriptions = [instr.description or "" for instr in instructions.values()]
        if not descriptions:
            descriptions = ['dummy']  # Use dummy doc if no descriptions
        self.instruction_vectors = self.vectorizer.transform(descriptions)
        
        # Initialize history for ALL instructions
        for name in instructions:
            # Ensure history exists for each instruction
            if name not in self.instruction_history:
                self.instruction_history[name] = {
                    "total_executions": 0,
                    "successful_executions": 0,
                    "cache_hits": 0,
                    "avg_execution_time": 0.0
                }
    
    async def initialize(self):
        """Initialize the instruction selector"""
        # Initialize with a dummy document to avoid empty vocabulary error
        self.instruction_vectors = self.vectorizer.fit_transform(['dummy'])
        self.instruction_vectors = None  # Clear the vectors after initialization
        return self
    
    async def cleanup(self):
        """Clean up instruction selector state"""
        # Clear instruction vectors and history
        self.instruction_vectors = {}
        self.instruction_history = {}
        
        # Reset adaptive weights
        self.adaptive_weights = AdaptiveWeights([
            "relevance",
            "success_rate", 
            "cost",
            "cache_hit_rate"
        ])
        
        # Reset A/B testing state
        self.ab_testing_enabled = False
        self.variants = {}
    
    def update_history(self, instruction_name: str, metrics: Dict[str, Any]):
        """Update instruction execution history"""
        history = self.instruction_history[instruction_name]
        history["total_executions"] += 1
        
        if metrics.get("status") == "completed":
            history["successful_executions"] += 1
        if metrics.get("cache_hit", False):
            history["cache_hits"] += 1
            
        # Update average execution time
        curr_avg = history["avg_execution_time"]
        n = history["total_executions"]
        new_time = metrics.get("execution_time", 0.0)
        history["avg_execution_time"] = (curr_avg * (n-1) + new_time) / n
    
    def _calculate_relevance(self, query: str, instruction_desc: str) -> float:
        """Calculate semantic relevance between query and instruction"""
        query_vector = self.vectorizer.transform([query])
        instruction_vector = self.vectorizer.transform([instruction_desc])
        return float(cosine_similarity(query_vector, instruction_vector)[0][0])
    
    def _calculate_success_rate(self, instruction_name: str) -> float:
        _ensure_heavy_deps()
        """Calculate historical success rate"""
        # If instruction history doesn't exist, initialize it
        if instruction_name not in self.instruction_history:
            self.instruction_history[instruction_name] = {
                "total_executions": 0,
                "successful_executions": 0,
                "cache_hits": 0,
                "avg_execution_time": 0.0
            }
        
        history = self.instruction_history[instruction_name]
        total = history["total_executions"]
        if total == 0:
            return 1.0  # Default to optimistic for new instructions
        return history["successful_executions"] / total
    
    def _calculate_cache_effectiveness(self, instruction_name: str) -> float:
        """Calculate cache hit rate"""
        history = self.instruction_history[instruction_name]
        total = history["total_executions"]
        if total == 0:
            return 0.0
        return history["cache_hits"] / total
    
    def _get_feature_scores(self, name: str, instr: Instruction,
                          query: str) -> Dict[str, float]:
        """Get feature scores for instruction"""
        return {
            "relevance": self._calculate_relevance(query, instr.description or ""),
            "success_rate": self._calculate_success_rate(name),
            "cost": 1.0 - (getattr(instr, "cost", 1.0) / 10.0),
            "cache_hit_rate": self._calculate_cache_effectiveness(name)
        }
    
    def select_instructions(self, input_data: Dict[str, Any],
                          available_instructions: Dict[str, Instruction],
                          max_instructions: int = 5) -> List[str]:
        """Select best instructions using adaptive weights"""
        query = input_data.get("text", "")
        if not query:
            return []
        
        if self.ab_testing_enabled and self.variants:
            # Randomly select a variant for A/B testing
            variant_name = np.random.choice(list(self.variants.keys()))
            available_instructions = self.variants[variant_name]
            logger.info(f"Selected variant {variant_name} for A/B testing")
        
        scores: List[InstructionScore] = []
        weights = self.adaptive_weights.get_weights()
        
        for name, instr in available_instructions.items():
            # Get feature scores
            features = self._get_feature_scores(name, instr, query)
            
            # Calculate weighted score
            final_score = sum(
                weights[feature] * score
                for feature, score in features.items()
            )
            
            scores.append(InstructionScore(
                name=name,
                relevance=features["relevance"],
                cost=features["cost"],
                success_rate=features["success_rate"],
                cache_hit_rate=features["cache_hit_rate"],
                final_score=final_score
            ))
        
        # Sort by final score and select top N
        scores.sort(key=lambda x: x.final_score, reverse=True)
        selected = scores[:max_instructions]
        
        # Log selection reasoning with weights
        for score in selected:
            logger.info(
                f"Selected instruction {score.name} with weighted scores:"
                f" relevance={score.relevance:.2f} (w={weights['relevance']:.2f}),"
                f" success_rate={score.success_rate:.2f} (w={weights['success_rate']:.2f}),"
                f" cost={score.cost:.2f} (w={weights['cost']:.2f}),"
                f" cache_hit_rate={score.cache_hit_rate:.2f} (w={weights['cache_hit_rate']:.2f}),"
                f" final_score={score.final_score:.2f}"
            )
        
        return [score.name for score in selected]
    
    def update_weights(self, instruction_name: str,
                      features: Dict[str, float],
                      result: Dict[str, Any]):
        """Update adaptive weights based on execution result"""
        self.adaptive_weights.update(features, result)
    
    def get_selection_stats(self) -> Dict[str, Any]:
        """Get instruction selection statistics"""
        stats = {
            "weights": self.adaptive_weights.get_weights(),
            "performance": self.adaptive_weights.get_performance_stats()
        }
        
        if self.ab_testing_enabled:
            stats["ab_testing"] = {
                variant: {
                    "usage_count": sum(1 for h in self.instruction_history.values()
                                     if h.get("variant") == variant),
                    "avg_performance": np.mean([
                        h.get("performance", 0.0)
                        for h in self.instruction_history.values()
                        if h.get("variant") == variant
                    ])
                }
                for variant in self.variants
            }
        
        return stats
    
    def optimize_sequence(self, selected_instructions: List[str],
                         available_instructions: Dict[str, Instruction]) -> List[str]:
        """Optimize the sequence of selected instructions"""
        if not selected_instructions:
            return []
        
        # Build dependency graph
        dependencies = {
            name: set(getattr(available_instructions[name], 'dependencies', []))
            for name in selected_instructions
        }
        
        # Topological sort with parallel execution consideration
        sorted_instructions = []
        seen = set()
        parallel_groups: List[Set[str]] = []
        
        def visit(name: str, path: Set[str]):
            if name in path:
                raise ValueError(f"Circular dependency detected: {name}")
            if name in seen:
                return
            
            path.add(name)
            deps = dependencies.get(name, set())
            
            # Process all dependencies first
            for dep in deps:
                if dep in selected_instructions:  # Only process selected instructions
                    visit(dep, path)
            
            path.remove(name)
            seen.add(name)
            
            # Try to add to existing parallel group or create new one
            added_to_group = False
            for group in parallel_groups:
                # Check if instruction can be added to group
                if not any(dependencies[name] & dependencies[g] for g in group):
                    group.add(name)
                    added_to_group = True
                    break
            
            if not added_to_group:
                # Create new parallel group
                parallel_groups.append({name})
        
        # Build parallel groups
        for name in selected_instructions:
            if name not in seen:
                visit(name, set())
        
        # Convert parallel groups to final sequence
        for group in parallel_groups:
            sorted_instructions.extend(list(group))
        
        return sorted_instructions
