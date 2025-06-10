"""Pattern learning instruction implementation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN

from .base import BaseInstruction, ExecutionContext, ExecutionResult

@dataclass
class PatternLearningConfig:
    """Configuration for pattern learning."""
    min_pattern_support: int = 3
    min_pattern_confidence: float = 0.7
    max_iterations: int = 100
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.8

class PatternLearningInstruction(BaseInstruction):
    """Instruction for learning patterns from interactions."""
    
    def __init__(self, name: str, config: PatternLearningConfig):
        """Initialize pattern learning instruction.
        
        Args:
            name: Instruction name
            config: Pattern learning configuration
        """
        super().__init__(name)
        self.config = config
        self.patterns: Dict[str, Dict[str, Any]] = {}
        self.vectorizer = TfidfVectorizer()
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute pattern learning.
        
        Args:
            context: Execution context containing:
                - interactions: List of interaction data
                
        Returns:
            ExecutionResult: Pattern learning results
        """
        try:
            # Extract interactions from context
            interactions = context.params.get("interactions", [])
            if not interactions:
                return ExecutionResult(
                    success=False,
                    output={"error": "No interactions provided"},
                    errors=[{"error": "No interactions provided"}]
                )
            
            # Learn patterns
            patterns = await self._learn_patterns(interactions)
            
            # Calculate confidence
            confidence_values = [float(p["confidence"]) for p in patterns.values()]
            pattern_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
            
            return ExecutionResult(
                success=True,
                output={
                    "patterns": patterns,
                    "metrics": {
                        "pattern_count": len(patterns),
                        "pattern_confidence": pattern_confidence
                    }
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}]
            )
    
    async def _learn_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Learn patterns from interactions.
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dictionary of learned patterns
        """
        # Convert interactions to text for vectorization
        texts = [self._interaction_to_text(interaction) for interaction in interactions]
        
        # Vectorize texts
        vectors = self.vectorizer.fit_transform(texts)
        
        # Convert sparse matrix to dense array for clustering
        if isinstance(vectors, csr_matrix):
            dense_vectors = vectors.toarray()
        else:
            dense_vectors = np.array(vectors)
        
        # Cluster interactions
        clusters = DBSCAN(
            eps=0.5,
            min_samples=self.config.min_pattern_support
        ).fit(dense_vectors)
        
        # Extract patterns from clusters
        patterns = {}
        for label in set(clusters.labels_):
            if label == -1:  # Skip noise
                continue
                
            # Get interactions in this cluster
            cluster_interactions = [
                interactions[i] for i in range(len(interactions))
                if clusters.labels_[i] == label
            ]
            
            # Extract pattern
            pattern = self._extract_pattern(cluster_interactions)
            if pattern:
                pattern_id = f"pattern_{label}"
                patterns[pattern_id] = pattern
        
        return patterns
    
    def _interaction_to_text(self, interaction: Dict[str, Any]) -> str:
        """Convert interaction to text representation."""
        # Combine relevant fields into text
        text_parts = []
        
        if "type" in interaction:
            text_parts.append(f"type:{interaction['type']}")
        if "action" in interaction:
            text_parts.append(f"action:{interaction['action']}")
        if "content" in interaction:
            text_parts.append(f"content:{interaction['content']}")
        
        return " ".join(text_parts)
    
    def _extract_pattern(self, interactions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract pattern from a group of similar interactions."""
        if not interactions:
            return None
            
        # Find common components
        common_components = {}
        
        # Extract common fields
        for field in ["type", "action", "content"]:
            values = [
                interaction.get(field)
                for interaction in interactions
                if field in interaction
            ]
            
            if values:
                # If all values are the same, it's a fixed component
                if len(set(values)) == 1:
                    common_components[field] = values[0]
                # Otherwise, it's a variable component
                else:
                    common_components[field] = {
                        "type": "variable",
                        "values": list(set(values))
                    }
        
        if not common_components:
            return None
            
        return {
            "components": common_components,
            "frequency": len(interactions),
            "confidence": len(interactions) / self.config.min_pattern_support
        } 