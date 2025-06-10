"""Integration tests for agent learning instructions."""

import pytest
import numpy as np
from typing import Dict, Any, List
from agentflow.core.instructions.agent_learning import (
    PatternLearningInstruction,
    AdaptiveLearningInstruction,
    ReinforcementLearningInstruction,
    LearningConfig
)
from agentflow.core.instructions.base import ExecutionContext

@pytest.fixture
def learning_config():
    """Create test learning configuration."""
    return LearningConfig(
        min_pattern_support=2,
        min_pattern_confidence=0.7,
        max_iterations=10,
        learning_rate=0.1,
        adaptation_threshold=0.8
    )

@pytest.fixture
def pattern_instruction(learning_config):
    """Create test pattern learning instruction."""
    return PatternLearningInstruction(
        name="test_pattern",
        learning_config=learning_config
    )

@pytest.fixture
def adaptive_instruction(learning_config):
    """Create test adaptive learning instruction."""
    def adaptation_rule(data: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "type": "behavior_change",
            "parameters": {"threshold": 0.5},
            "expected_impact": {"improvement": 0.2}
        }
    
    return AdaptiveLearningInstruction(
        name="test_adaptive",
        learning_config=learning_config,
        adaptation_rules={"test_rule": adaptation_rule}
    )

@pytest.fixture
def reinforcement_instruction(learning_config):
    """Create test reinforcement learning instruction."""
    def reward_function(entry: Dict[str, Any]) -> float:
        return entry.get("reward", 0.0)
    
    return ReinforcementLearningInstruction(
        name="test_rl",
        learning_config=learning_config,
        action_space={"action1": 1.0, "action2": 2.0},
        reward_function=reward_function
    )

@pytest.mark.asyncio
async def test_pattern_learning(pattern_instruction):
    """Test pattern learning from interactions."""
    # Create test interactions
    interactions = [
        {
            "type": "request",
            "participants": ["agent1", "agent2"],
            "content": {
                "action": "share_data",
                "data": {"value": 42}
            },
            "success": True
        },
        {
            "type": "request",
            "participants": ["agent2", "agent3"],
            "content": {
                "action": "share_data",
                "data": {"value": 43}
            },
            "success": True
        },
        {
            "type": "request",
            "participants": ["agent1", "agent3"],
            "content": {
                "action": "share_data",
                "data": {"value": 44}
            },
            "success": False
        }
    ]
    
    # Execute pattern learning
    context = ExecutionContext(
        params={"interactions": interactions},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await pattern_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert isinstance(result.output, dict)
    assert len(result.output) > 0  # At least one pattern found
    
    # Verify pattern structure
    pattern = next(iter(result.output.values()))
    assert "components" in pattern
    assert "context_requirements" in pattern
    assert "frequency" in pattern
    assert "success_rate" in pattern
    assert "confidence" in pattern
    
    # Verify metrics
    assert pattern_instruction.metrics.patterns_discovered > 0
    assert 0.0 <= pattern_instruction.metrics.pattern_confidence <= 1.0

@pytest.mark.asyncio
async def test_adaptive_learning(adaptive_instruction):
    """Test adaptive learning and behavior modification."""
    # Create test behavior data
    behavior_data = [
        {
            "action": "task1",
            "parameters": {"value": 1},
            "outcome": "success",
            "performance": 0.8
        },
        {
            "action": "task1",
            "parameters": {"value": 2},
            "outcome": "failure",
            "performance": 0.3
        },
        {
            "action": "task2",
            "parameters": {"value": 3},
            "outcome": "success",
            "performance": 0.9
        }
    ]
    
    # Execute adaptive learning
    context = ExecutionContext(
        params={"behavior_data": behavior_data},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await adaptive_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert isinstance(result.output, dict)
    assert "test_rule" in result.output
    
    # Verify adaptation structure
    adaptation = result.output["test_rule"]
    assert "changes" in adaptation
    assert "confidence" in adaptation
    
    # Verify metrics
    assert adaptive_instruction.metrics.learning_iterations > 0
    assert 0.0 <= adaptive_instruction.metrics.adaptation_rate <= 1.0

@pytest.mark.asyncio
async def test_reinforcement_learning(reinforcement_instruction):
    """Test reinforcement learning."""
    # Create test state and action history
    state = {"value": 1}
    action_history = [
        {
            "state": {"value": 0},
            "action": "action1",
            "reward": 1.0,
            "next_state": {"value": 1}
        },
        {
            "state": {"value": 1},
            "action": "action2",
            "reward": 0.5,
            "next_state": {"value": 2}
        }
    ]
    
    # Execute reinforcement learning
    context = ExecutionContext(
        params={
            "state": state,
            "action_history": action_history
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await reinforcement_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert "action" in result.output
    assert result.output["action"] in reinforcement_instruction.action_space
    
    # Verify Q-table
    assert len(reinforcement_instruction.q_table) > 0
    for state_values in reinforcement_instruction.q_table.values():
        assert all(isinstance(v, float) for v in state_values.values())
    
    # Verify metrics
    assert reinforcement_instruction.metrics.learning_iterations > 0
    assert isinstance(reinforcement_instruction.metrics.performance_improvement, float)

@pytest.mark.asyncio
async def test_pattern_learning_insufficient_data(pattern_instruction):
    """Test pattern learning with insufficient data."""
    # Create test interactions (less than min_pattern_support)
    interactions = [
        {
            "type": "request",
            "participants": ["agent1"],
            "content": {"action": "test"},
            "success": True
        }
    ]
    
    # Execute pattern learning
    context = ExecutionContext(
        params={"interactions": interactions},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await pattern_instruction.execute(context)
    
    # Verify results
    assert result.success
    assert len(result.output) == 0  # No patterns found
    assert pattern_instruction.metrics.patterns_discovered == 0

@pytest.mark.asyncio
async def test_adaptive_learning_invalid_data(adaptive_instruction):
    """Test adaptive learning with invalid behavior data."""
    # Execute with empty data
    context = ExecutionContext(
        params={"behavior_data": []},
        state={},
        metadata={"timestamp": 0.0}
    )
    result = await adaptive_instruction.execute(context)
    
    # Verify error handling
    assert not result.success
    assert len(result.errors) > 0
    assert any("no behavior data" in str(error["error"]).lower() for error in result.errors)

@pytest.mark.asyncio
async def test_reinforcement_learning_exploration(reinforcement_instruction):
    """Test reinforcement learning exploration vs exploitation."""
    state = {"value": 0}
    
    # Execute multiple times to test exploration
    actions = set()
    for _ in range(20):
        context = ExecutionContext(
            params={"state": state},
            state={},
            metadata={"timestamp": 0.0}
        )
        result = await reinforcement_instruction.execute(context)
        actions.add(result.output["action"])
    
    # Verify that both actions were selected (exploration)
    assert len(actions) > 1, "Exploration did not try different actions"

@pytest.mark.asyncio
async def test_reinforcement_learning_convergence(reinforcement_instruction):
    """Test reinforcement learning convergence."""
    # Create test data with clear optimal action
    optimal_action = "action2"
    action_history = [
        {
            "state": {"value": i},
            "action": optimal_action,
            "reward": 1.0,
            "next_state": {"value": i + 1}
        } for i in range(10)
    ] + [
        {
            "state": {"value": i},
            "action": "action1",
            "reward": 0.1,
            "next_state": {"value": i + 1}
        } for i in range(10)
    ]
    
    # Train the agent
    context = ExecutionContext(
        params={
            "state": {"value": 0},
            "action_history": action_history
        },
        state={},
        metadata={"timestamp": 0.0}
    )
    await reinforcement_instruction.execute(context)
    
    # Test exploitation (should prefer optimal action)
    optimal_choices = 0
    total_trials = 50
    
    for _ in range(total_trials):
        result = await reinforcement_instruction.execute(context)
        if result.output["action"] == optimal_action:
            optimal_choices += 1
    
    # Verify convergence to optimal action
    optimal_ratio = optimal_choices / total_trials
    assert optimal_ratio > 0.7, "Did not converge to optimal action" 