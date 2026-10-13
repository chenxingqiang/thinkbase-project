"""Adaptive weight adjustment for instruction selection"""

from typing import Dict, Any, List, Optional
import logging
import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize
import time

logger = logging.getLogger(__name__)

@dataclass
class WeightConfig:
    """Configuration for weight adaptation"""
    learning_rate: float = 0.01
    momentum: float = 0.9
    regularization: float = 0.001
    min_weight: float = 0.0
    max_weight: float = 1.0
    
class AdaptiveWeights:
    """Adaptive weight adjustment for instruction selection"""
    
    def __init__(self, feature_names: List[str], config: Optional[WeightConfig] = None):
        self.feature_names = feature_names
        self.config = config or WeightConfig()
        
        # Initialize weights
        self.weights = {
            name: 1.0 / len(feature_names)
            for name in feature_names
        }
        
        # Historical data
        self.history = []
        self.velocity = {name: 0.0 for name in feature_names}
        
        # Performance tracking
        self.performance_history = []
    
    def _normalize_weights(self):
        """Normalize weights to sum to 1"""
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {
                name: weight / total
                for name, weight in self.weights.items()
            }
    
    def _clip_weights(self):
        """Clip weights to configured range"""
        self.weights = {
            name: np.clip(weight, self.config.min_weight, self.config.max_weight)
            for name, weight in self.weights.items()
        }
    
    def _calculate_performance(self, result: Dict[str, Any]) -> float:
        """Calculate performance score from execution result"""
        # Combine multiple metrics into single score
        metrics = result.get("metrics", {})
        
        success_score = float(metrics.get("success", False))
        latency_score = 1.0 / (1.0 + metrics.get("latency", 0.0))
        quality_score = metrics.get("quality_score", 0.0)
        
        # Weighted combination of metrics
        return (
            0.4 * success_score +
            0.3 * latency_score +
            0.3 * quality_score
        )
    
    def _optimize_weights(self, features: Dict[str, float],
                         performance: float) -> Dict[str, float]:
        """Optimize weights using gradient descent with momentum"""
        # Calculate gradients
        gradients = {}
        for name in self.feature_names:
            # Gradient calculation with regularization
            feature_value = features.get(name, 0.0)
            current_weight = self.weights[name]
            
            gradient = (
                performance * feature_value -  # Performance contribution
                self.config.regularization * current_weight  # L2 regularization
            )
            
            # Apply momentum
            self.velocity[name] = (
                self.config.momentum * self.velocity[name] +
                self.config.learning_rate * gradient
            )
            
            # Update weight
            gradients[name] = self.velocity[name]
        
        # Update weights
        new_weights = {
            name: weight + gradients[name]
            for name, weight in self.weights.items()
        }
        
        return new_weights
    
    def update(self, features: Dict[str, float],
               result: Dict[str, Any]):
        """Update weights based on execution result"""
        # Calculate performance
        performance = self._calculate_performance(result)
        
        # Store historical data
        self.history.append({
            "features": features,
            "performance": performance,
            "weights": dict(self.weights),
            "timestamp": time.time()
        })
        
        # Optimize weights
        new_weights = self._optimize_weights(features, performance)
        self.weights = new_weights
        
        # Normalize and clip weights
        self._clip_weights()
        self._normalize_weights()
        
        # Track performance
        self.performance_history.append(performance)
        
        # Log weight updates
        logger.debug("Updated weights: %s", self.weights)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current weights"""
        return dict(self.weights)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.performance_history:
            return {}
        
        recent_window = min(len(self.performance_history), 10)
        recent_performance = self.performance_history[-recent_window:]
        
        return {
            "current_performance": self.performance_history[-1],
            "avg_performance": np.mean(self.performance_history),
            "recent_avg_performance": np.mean(recent_performance),
            "performance_trend": np.mean(np.diff(recent_performance)),
            "weight_stability": {
                name: np.std([h["weights"][name] for h in self.history[-recent_window:]])
                for name in self.feature_names
            }
        }
    
    def reset(self):
        """Reset weights to initial state"""
        self.weights = {
            name: 1.0 / len(self.feature_names)
            for name in self.feature_names
        }
        self.velocity = {name: 0.0 for name in self.feature_names}
        self.history.clear()
        self.performance_history.clear()
