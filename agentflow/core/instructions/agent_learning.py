"""Agent learning instruction implementations."""

from typing import Dict, Any, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
import asyncio
import logging
import json
import numpy as np
import random

from .base import (
    BaseInstruction,
    InstructionType,
    InstructionMetadata,
    ExecutionContext,
    ExecutionResult
)

logger = logging.getLogger(__name__)

@dataclass
class LearningMetrics:
    """Metrics for learning process."""
    patterns_discovered: int = 0
    pattern_confidence: float = 0.0
    learning_iterations: int = 0
    adaptation_rate: float = 0.0
    performance_improvement: float = 0.0

@dataclass
class LearningConfig:
    """Configuration for learning instructions."""
    min_pattern_support: int = 3
    min_pattern_confidence: float = 0.7
    max_iterations: int = 100
    learning_rate: float = 0.1
    adaptation_threshold: float = 0.8
    # Reinforcement-learning parameters (shared defaults)
    discount_factor: float = 0.95
    exploration_rate: float = 0.1

@dataclass
class AgentLearningConfig:
    """Configuration for agent learning."""
    learning_rate: float = 0.1
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    min_success_rate: float = 0.7
    max_iterations: int = 1000
    batch_size: int = 32

class PatternLearningInstruction(BaseInstruction):
    """Instruction for learning interaction patterns."""
    
    def __init__(self,
                 name: str,
                 learning_config: LearningConfig,
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            instruction_type=InstructionType.LLM_INTERACTION,
            metadata=metadata,
            description="Learn interaction patterns"
        )
        self.learning_config = learning_config
        self.metrics = LearningMetrics()
        self.patterns: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute pattern learning."""
        try:
            # Extract interactions from context
            interactions = context.params.get("interactions", [])
            if not interactions:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "No interactions provided"}]
                )
            
            # Learn patterns
            patterns = await self._learn_patterns(interactions)
            
            # Update metrics
            self.metrics.patterns_discovered = len(patterns)
            confidence_values = [float(p["confidence"]) for p in patterns.values()]
            self.metrics.pattern_confidence = float(np.mean(confidence_values)) if confidence_values else 0.0
            
            return ExecutionResult(
                success=True,
                output=patterns,
                state_updates={"learned_patterns": patterns},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Pattern learning failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _learn_patterns(self,
                            interactions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Learn patterns from interactions.

        Successful interactions are grouped by structural signature
        (type plus content keys, with dict values reduced to their key
        sets); recurring structures become patterns. The earlier
        TF-IDF/DBSCAN clustering split near-identical structures over
        numeric payload differences, so signature grouping is used
        instead.
        """
        # Only successful interactions contribute learned patterns
        candidates = [i for i in interactions if i.get("success", False)]
        
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for interaction in candidates:
            signature = self._interaction_signature(interaction)
            groups.setdefault(signature, []).append(interaction)
        
        patterns: Dict[str, Dict[str, Any]] = {}
        for group_interactions in groups.values():
            if len(group_interactions) < self.learning_config.min_pattern_support:
                continue
            
            pattern = self._extract_pattern(group_interactions)
            if pattern:
                pattern_id = f"pattern_{len(patterns)}"
                patterns[pattern_id] = pattern
        
        return patterns
    
    def _interaction_signature(self, interaction: Dict[str, Any]) -> str:
        """Build a structural signature for grouping similar interactions."""
        parts = [str(interaction.get("type", "unknown"))]
        
        content = interaction.get("content", {})
        for key in sorted(content.keys()):
            value = content[key]
            if isinstance(value, dict):
                parts.append(f"{key}:{{{','.join(sorted(value.keys()))}}}")
            elif isinstance(value, (list, tuple)):
                parts.append(f"{key}:list")
            else:
                parts.append(f"{key}:{value}")
        
        return "|".join(parts)
    
    def _extract_pattern(self,
                        interactions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Extract pattern from similar interactions."""
        if not interactions:
            return None
        
        # Calculate pattern frequency and success rate
        total = len(interactions)
        successful = sum(1 for i in interactions if i.get("success", False))
        success_rate = successful / total
        
        if success_rate < self.learning_config.min_pattern_confidence:
            return None
        
        # Extract common components
        components = self._find_common_components(interactions)
        
        # Extract context requirements
        context_reqs = self._extract_context_requirements(interactions)
        
        return {
            "components": components,
            "context_requirements": list(context_reqs),
            "frequency": total,
            "success_rate": success_rate,
            "confidence": success_rate * (total / self.learning_config.min_pattern_support)
        }
    
    def _find_common_components(self,
                              interactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find common components across interactions."""
        if not interactions:
            return []
        
        # Get all component keys
        component_keys = set()
        for interaction in interactions:
            component_keys.update(interaction.get("content", {}).keys())
        
        # Find common components
        common_components = []
        for key in component_keys:
            values = [
                interaction.get("content", {}).get(key)
                for interaction in interactions
            ]
            
            if all(v == values[0] for v in values):
                common_components.append({
                    "key": key,
                    "value": values[0]
                })
        
        return common_components
    
    def _extract_context_requirements(self,
                                    interactions: List[Dict[str, Any]]) -> Set[str]:
        """Extract context requirements from interactions."""
        if not interactions:
            return set()
        
        # Get all context keys
        context_keys = set()
        for interaction in interactions:
            context_keys.update(interaction.get("context", {}).keys())
        
        # Find required context
        required_context = set()
        for key in context_keys:
            if all(key in i.get("context", {}) for i in interactions):
                required_context.add(key)
        
        return required_context

class AdaptiveLearningInstruction(BaseInstruction):
    """Instruction for adaptive learning and behavior modification."""
    
    def __init__(self,
                 name: str,
                 learning_config: LearningConfig,
                 adaptation_rules: Dict[str, Callable],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            instruction_type=InstructionType.LLM_INTERACTION,
            metadata=metadata,
            description="Adaptive learning and behavior modification"
        )
        self.learning_config = learning_config
        self.adaptation_rules = adaptation_rules
        self.metrics = LearningMetrics()
        self.behavior_models: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute adaptive learning."""
        try:
            # Get behavior data
            behavior_data = context.params.get("behavior_data", [])
            if not behavior_data:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "No behavior data provided"}]
                )
            
            # Learn and adapt behavior
            adaptations = await self._adapt_behavior(behavior_data)
            
            # Update metrics
            self.metrics.learning_iterations += 1
            self.metrics.adaptation_rate = len(adaptations) / len(self.adaptation_rules)
            
            return ExecutionResult(
                success=True,
                output=adaptations,
                state_updates={"behavior_adaptations": adaptations},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Adaptive learning failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _adapt_behavior(self,
                            behavior_data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Learn and adapt behavior."""
        adaptations = {}
        
        # Process each adaptation rule
        for rule_name, rule_func in self.adaptation_rules.items():
            try:
                # Apply rule to behavior data
                adaptation = rule_func(behavior_data)
                
                # Validate adaptation
                if self._validate_adaptation(adaptation):
                    adaptations[rule_name] = {
                        "changes": adaptation,
                        "confidence": self._calculate_confidence(adaptation, behavior_data)
                    }
            except Exception as e:
                logger.error(f"Rule {rule_name} failed: {str(e)}")
        
        return adaptations
    
    def _validate_adaptation(self, adaptation: Dict[str, Any]) -> bool:
        """Validate proposed adaptation."""
        if not isinstance(adaptation, dict):
            return False
        
        required_keys = {"type", "parameters", "expected_impact"}
        if not all(key in adaptation for key in required_keys):
            return False
        
        return True
    
    def _calculate_confidence(self, adaptation: Dict[str, Any],
                            behavior_data: List[Dict[str, Any]]) -> float:
        """Calculate confidence in adaptation."""
        # Implement confidence calculation
        return 0.8

class ReinforcementLearningInstruction(BaseInstruction):
    """Instruction for reinforcement learning."""
    
    def __init__(self,
                 name: str,
                 learning_config: LearningConfig,
                 action_space: Optional[Dict[str, Any]] = None,
                 reward_function: Optional[Callable[[Dict[str, Any]], float]] = None,
                 metadata: Optional[InstructionMetadata] = None):
        """Initialize reinforcement learning instruction.
        
        Args:
            name: Instruction name
            learning_config: Learning configuration
            action_space: Available actions and their prior values
            reward_function: Maps an action-history entry to a reward
            metadata: Optional instruction metadata
        """
        super().__init__(
            name=name,
            instruction_type=InstructionType.LLM_INTERACTION,
            metadata=metadata,
            description="Reinforcement learning"
        )
        self.learning_config = learning_config
        # Backward-compatible alias
        self.config = learning_config
        self.action_space: Dict[str, Any] = dict(action_space) if action_space else {}
        self.reward_function = reward_function
        self.q_table: Dict[str, Dict[str, float]] = {}
        self.action_history: List[Dict[str, Any]] = []
        self.metrics = LearningMetrics()
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute reinforcement learning.
        
        Args:
            context: Execution context containing:
                - state: Current state
                - actions: Available actions (defaults to action_space keys)
                - action_history: Optional history to learn from
                
        Returns:
            ExecutionResult: Learning results
        """
        try:
            # Extract state and actions
            state = context.params.get("state")
            actions = context.params.get("actions") or list(self.action_space.keys())
            
            if not state or not actions:
                return ExecutionResult(
                    success=False,
                    output={"error": "Missing state or actions"},
                    errors=[{"error": "Missing state or actions"}]
                )
            
            # Convert state to string for Q-table lookup
            state_key = self._state_to_key(state)
            
            # Learn from provided action history first
            action_history = context.params.get("action_history")
            if action_history:
                await self._update_q_table(action_history)
            
            # Initialize Q-values for new state
            if state_key not in self.q_table:
                self.q_table[state_key] = {
                    action: 0.0 for action in actions
                }
            
            # Select action using epsilon-greedy policy
            action = self._select_action(state_key, actions)
            
            # Record action
            self.action_history.append({
                "state": state,
                "action": action,
                "timestamp": context.metadata.get("timestamp", 0.0)
            })
            
            # Update metrics
            self.metrics.learning_iterations += 1
            self.metrics.performance_improvement = self._calculate_improvement()
            
            return ExecutionResult(
                success=True,
                output={
                    "action": action,
                    "metrics": {
                        "q_values": dict(self.q_table[state_key]),
                        "exploration_rate": self.config.exploration_rate,
                        "improvement": self.metrics.performance_improvement
                    }
                },
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _update_q_table(self, action_history: List[Dict[str, Any]]):
        """Update Q-values based on action history."""
        if not action_history:
            return
        
        all_actions = list(self.action_space.keys())
            
        for i in range(len(action_history)):
            current = action_history[i]
            next_state = current.get("next_state")
            if next_state is None and i + 1 < len(action_history):
                next_state = action_history[i + 1]["state"]
            if next_state is None:
                continue
            
            # Get current state and action
            state_key = self._state_to_key(current["state"])
            action = current["action"]
            
            # Ensure Q-table entries exist for both states
            next_state_key = self._state_to_key(next_state)
            for key in (state_key, next_state_key):
                if key not in self.q_table:
                    self.q_table[key] = {a: 0.0 for a in all_actions}
            if action not in self.q_table[state_key]:
                continue  # Unknown action
            
            # Calculate reward
            reward = self._calculate_reward(current, next_state)
            
            # Get next state's maximum Q-value
            next_max_q = (
                max(self.q_table[next_state_key].values())
                if self.q_table[next_state_key] else 0.0
            )
            
            # Update Q-value
            current_q = self.q_table[state_key][action]
            self.q_table[state_key][action] = current_q + self.config.learning_rate * (
                reward + self.config.discount_factor * next_max_q - current_q
            )
    
    def _select_action(self, state_key: str, actions: List[str]) -> str:
        """Select action using epsilon-greedy policy."""
        if random.random() < self.config.exploration_rate:
            # Exploration: random action
            return random.choice(actions)
        else:
            # Exploitation: best action
            q_values = self.q_table[state_key]
            max_q = max(q_values.values())
            best_actions = [
                action for action, q_value in q_values.items()
                if q_value == max_q
            ]
            return random.choice(best_actions)
    
    def _state_to_key(self, state: Dict[str, Any]) -> str:
        """Convert state dictionary to string key."""
        # Sort keys to ensure consistent string representation
        return str(sorted(state.items()))
    
    def _calculate_reward(self, action_data: Dict[str, Any], next_state: Dict[str, Any]) -> float:
        """Calculate reward for action.

        Prefers the explicit reward function, then a recorded ``reward``
        field, and finally falls back to state improvement.
        """
        if self.reward_function is not None:
            try:
                return float(self.reward_function(action_data))
            except Exception as e:
                logger.warning(f"Reward function failed: {str(e)}")
        
        if "reward" in action_data:
            return float(action_data["reward"])
        
        # Simple reward based on state improvement
        current_value = sum(float(v) for v in action_data["state"].values() if isinstance(v, (int, float)))
        next_value = sum(float(v) for v in next_state.values() if isinstance(v, (int, float)))
        return next_value - current_value
    
    def _calculate_improvement(self) -> float:
        """Calculate performance improvement."""
        if len(self.action_history) < 2:
            return 0.0
            
        # Calculate average reward over recent actions
        recent_rewards = []
        for i in range(len(self.action_history) - 1):
            current = self.action_history[i]
            next_state = self.action_history[i + 1]["state"]
            reward = self._calculate_reward(current, next_state)
            recent_rewards.append(reward)
            
        if not recent_rewards:
            return 0.0
            
        return float(np.mean(recent_rewards)) 