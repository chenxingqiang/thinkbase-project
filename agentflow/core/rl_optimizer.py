"""Reinforcement Learning based optimization for ISA operations."""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, Dict as DictSpace, Discrete
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.env.env_context import EnvContext
from ray.tune.registry import register_env

from agentflow.agents.config import AgentConfig
from agentflow.core.isa_manager import ISAManager, Instruction

logger = logging.getLogger(__name__)

class ISAOptEnv(gym.Env):
    """Environment for optimizing ISA operations."""
    
    def __init__(self, config: EnvContext):
        """Initialize environment.
        
        Args:
            config: Environment configuration containing:
                - max_steps: Maximum steps per episode
                - instruction_set: Set of available instructions
                - optimization_metrics: Metrics to optimize
        """
        self.max_steps = config.get("max_steps", 100)
        self.instruction_set = config.get("instruction_set", {})
        self.current_step = 0
        
        # Define action space for instruction optimization
        self.action_space = DictSpace({
            "instruction_selection": Discrete(len(self.instruction_set)),
            "parallelization": Discrete(8),  # Parallelization level
            "batch_size": Discrete(32),      # Batch size for batched operations
            "caching": Discrete(2),          # Enable/disable caching
            "timeout": Box(low=1.0, high=30.0, shape=(1,), dtype=np.float32)
        })
        
        # Define observation space for ISA state
        self.observation_space = DictSpace({
            "instruction_metrics": Box(
                low=np.array([0, 0, 0, 0]),  # [latency, throughput, success_rate, resource_usage]
                high=np.array([100, 1, 1, 1]),
                dtype=np.float32
            ),
            "system_state": Box(
                low=np.zeros(4),  # [cpu_usage, memory_usage, cache_hit_rate, queue_length]
                high=np.ones(4),
                dtype=np.float32
            )
        })
        
        self.state = self._get_initial_state()
        self.history = []
        
    def _get_initial_state(self) -> Dict[str, np.ndarray]:
        """Get initial environment state."""
        return {
            "instruction_metrics": np.array([
                50.0,  # Initial latency
                0.5,   # Initial throughput
                0.9,   # Initial success rate
                0.5    # Initial resource usage
            ], dtype=np.float32),
            "system_state": np.array([
                0.5,   # Initial CPU usage
                0.5,   # Initial memory usage
                0.5,   # Initial cache hit rate
                0.0    # Initial queue length
            ], dtype=np.float32)
        }
    
    def _calculate_reward(self, action: Dict[str, Any], new_state: Dict[str, np.ndarray]) -> float:
        """Calculate reward based on optimization objectives."""
        metrics = new_state["instruction_metrics"]
        system = new_state["system_state"]
        
        # Performance metrics
        latency_score = 1.0 / (1.0 + metrics[0])
        throughput_score = metrics[1]
        success_score = metrics[2]
        
        # Resource efficiency
        resource_score = 1.0 - metrics[3]
        cache_score = system[2]
        
        # Combined reward with weights
        reward = (
            0.3 * latency_score +      # Minimize latency
            0.2 * throughput_score +   # Maximize throughput
            0.2 * success_score +      # Maintain high success rate
            0.15 * resource_score +    # Minimize resource usage
            0.15 * cache_score        # Maximize cache efficiency
        )
        
        return float(reward)
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset environment to initial state."""
        self.current_step = 0
        self.state = self._get_initial_state()
        return self.state, {}
    
    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one optimization step."""
        self.current_step += 1
        
        # Apply optimization action
        new_state = self._apply_optimization(action)
        
        # Calculate reward
        reward = self._calculate_reward(action, new_state)
        
        # Update state
        self.state = new_state
        
        # Record history
        self.history.append({
            "step": self.current_step,
            "action": action,
            "state": new_state,
            "reward": reward
        })
        
        # Check if episode is done
        done = self.current_step >= self.max_steps
        
        return new_state, reward, done, False, {}
    
    def _apply_optimization(self, action: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Apply optimization action to ISA operations."""
        current_metrics = self.state["instruction_metrics"]
        current_system = self.state["system_state"]
        
        # Simulate optimization effects
        new_metrics = np.array([
            # Latency affected by parallelization and batch size
            max(1.0, current_metrics[0] * (
                1.0 - 0.1 * action["parallelization"] / 8.0 -
                0.05 * action["batch_size"] / 32.0
            )),
            # Throughput affected by parallelization and caching
            min(1.0, current_metrics[1] * (
                1.0 + 0.1 * action["parallelization"] / 8.0 +
                0.05 * action["caching"]
            )),
            # Success rate affected by timeout
            max(0.5, min(1.0, current_metrics[2] * (
                1.0 + 0.05 * action["timeout"][0] / 30.0
            ))),
            # Resource usage affected by all parameters
            min(1.0, current_metrics[3] * (
                1.0 + 0.1 * action["parallelization"] / 8.0 +
                0.05 * action["batch_size"] / 32.0 +
                0.05 * action["caching"]
            ))
        ], dtype=np.float32)
        
        # Update system state
        new_system = np.array([
            # CPU usage
            min(1.0, current_system[0] * (
                1.0 + 0.1 * action["parallelization"] / 8.0
            )),
            # Memory usage
            min(1.0, current_system[1] * (
                1.0 + 0.05 * action["batch_size"] / 32.0
            )),
            # Cache hit rate
            min(1.0, current_system[2] * (
                1.0 + 0.2 * action["caching"]
            )),
            # Queue length
            max(0.0, current_system[3] * (
                1.0 - 0.1 * action["parallelization"] / 8.0
            ))
        ], dtype=np.float32)
        
        return {
            "instruction_metrics": new_metrics,
            "system_state": new_system
        }

class ISAOptimizer:
    """Optimizer for ISA operations using RL."""
    
    def __init__(self, config: Union[Dict[str, Any], AgentConfig]):
        """Initialize optimizer.
        
        Args:
            config: Optimizer configuration
        """
        self.config = config.model_dump() if isinstance(config, AgentConfig) else config
        self._initialized = False
        self.trainer: Optional[Union[PPO, SAC]] = None
        self.isa_manager: Optional[ISAManager] = None
        
        # Register environment
        register_env(
            "ISAOpt-v0",
            lambda config: ISAOptEnv(config)
        )
        
        # Initialize based on algorithm choice
        algorithm = self.config.get("algorithm", "PPO").upper()
        if algorithm == "PPO":
            self._init_ppo()
        elif algorithm == "SAC":
            self._init_sac()
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    def _init_ppo(self):
        """Initialize PPO algorithm."""
        ppo_config = (
            PPOConfig()
            .environment("ISAOpt-v0")
            .framework("torch")
            .training(
                train_batch_size=4000,
                lr=1e-4,
                gamma=0.99
            )
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=4)
        )
        self.trainer = ppo_config.build()
    
    def _init_sac(self):
        """Initialize SAC algorithm."""
        sac_config = (
            SACConfig()
            .environment("ISAOpt-v0")
            .framework("torch")
            .training(train_batch_size=4000)
            .resources(num_gpus=0)
            .rollouts(num_rollout_workers=4)
        )
        self.trainer = sac_config.build()
    
    async def initialize(self, isa_manager: Optional[ISAManager] = None) -> None:
        """Initialize optimizer with ISA manager."""
        if self._initialized:
            return
            
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        self.isa_manager = isa_manager
        self._initialized = True
    
    def optimize_instruction(self, instruction: Instruction) -> Dict[str, Any]:
        """Optimize instruction execution parameters.
        
        Args:
            instruction: Instruction to optimize
            
        Returns:
            Optimized parameters
        """
        if not self._initialized or not self.trainer:
            raise RuntimeError("Optimizer not initialized")
            
        # Get current state
        state = self._get_instruction_state(instruction)
        
        # Get optimized action
        raw_action = self.trainer.compute_single_action(state)
        
        # Convert raw action to dictionary format
        action = self._convert_raw_action(raw_action)
        
        # Convert action to instruction parameters
        params = self._action_to_params(action)
        
        return params
    
    def _get_instruction_state(self, instruction: Instruction) -> Dict[str, np.ndarray]:
        """Get current instruction state."""
        # Get metrics from ISA manager if available
        if self.isa_manager:
            metrics = self.isa_manager.get_instruction_stats()
        else:
            metrics = {}
            
        return {
            "instruction_metrics": np.array([
                metrics.get("latency", 50.0),
                metrics.get("throughput", 0.5),
                metrics.get("success_rate", 0.9),
                metrics.get("resource_usage", 0.5)
            ], dtype=np.float32),
            "system_state": np.array([
                metrics.get("cpu_usage", 0.5),
                metrics.get("memory_usage", 0.5),
                metrics.get("cache_hit_rate", 0.5),
                metrics.get("queue_length", 0.0)
            ], dtype=np.float32)
        }
    
    def _convert_raw_action(self, raw_action: Union[np.ndarray, Tuple, List, Dict[str, Any]]) -> Dict[str, Any]:
        """Convert raw action from trainer to dictionary format.
        
        Args:
            raw_action: Raw action from trainer
            
        Returns:
            Action in dictionary format
        """
        if isinstance(raw_action, dict):
            return raw_action
            
        # Convert array/tuple/list to dictionary
        action_keys = ["instruction_selection", "parallelization", "batch_size", "caching", "timeout"]
        if isinstance(raw_action, (np.ndarray, list, tuple)):
            raw_action = list(raw_action)
            return {
                key: raw_action[i] if i < len(raw_action) else 0
                for i, key in enumerate(action_keys)
            }
            
        raise ValueError(f"Unsupported action type: {type(raw_action)}")
    
    def _action_to_params(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Convert RL action to instruction parameters."""
        return {
            "parallelization_level": int(action["parallelization"]),
            "batch_size": int(action["batch_size"]),
            "use_caching": bool(action["caching"]),
            "timeout": float(action["timeout"][0])
        }
    
    def train(self, num_iterations: int = 100) -> None:
        """Train the optimizer."""
        if not self.trainer:
            raise RuntimeError("Optimizer not initialized")
            
        for i in range(num_iterations):
            result = self.trainer.train()
            if i % 10 == 0:
                logger.info(
                    f"Iteration {i}: "
                    f"episode_reward_mean={result['episode_reward_mean']:.2f}"
                )
    
    async def cleanup(self) -> None:
        """Clean up optimizer resources."""
        if self.trainer:
            self.trainer.stop()
            self.trainer = None
        self._initialized = False

def create_optimizer(config: Optional[Union[Dict[str, Any], AgentConfig]] = None) -> Optional[ISAOptimizer]:
    """Create an ISA optimizer.
    
    Args:
        config: Configuration dictionary or AgentConfig object
        
    Returns:
        ISAOptimizer instance or None if no configuration provided
    """
    if config is None:
        return None
        
    return ISAOptimizer(config)
