"""Tests for DeepSeek GitHub Agent."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import json

from examples.deepseek_agent import (
    DeepSeekAgent,
    AgentConfig,
    AgentMetrics
)
from agentflow.core.instructions import InstructionResult
from agentflow.core.isa.types import InstructionStatus

@pytest.fixture
def config_path(tmp_path):
    """Create test configuration file."""
    config = {
        "AGENT": {
            "NAME": "test-agent",
            "VERSION": "0.1.0",
            "TYPE": "code-generation"
        },
        "INPUT_SPECIFICATION": {
            "MODES": ["text"]
        },
        "OUTPUT_SPECIFICATION": {
            "MODES": ["code"]
        },
        "COLLABORATION": {
            "MODE": "autonomous",
            "WORKFLOW": ["analyze", "design", "implement", "review"]
        },
        "PROVIDER": {
            "MODEL": "deepseek-coder-33b-instruct",
            "TEMPERATURE": 0.7,
            "MAX_TOKENS": 4096,
            "API_KEY": "test-key",
            "API_BASE": "https://api.test.com"
        }
    }
    
    config_file = tmp_path / "test_config.json"
    config_file.write_text(json.dumps(config))
    return str(config_file)

@pytest.fixture
def agent(config_path):
    """Create test agent."""
    return DeepSeekAgent(config_path)

@pytest.mark.asyncio
async def test_agent_initialization(agent):
    """Test agent initialization."""
    # Initialize agent
    await agent.initialize()
    
    # Verify state
    assert agent.config is not None
    assert agent.config.name == "test-agent"
    assert agent.config.version == "0.1.0"
    assert agent.pipeline is not None
    assert agent.metrics.tokens_used == 0
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_requirement_analysis(agent):
    """Test requirement analysis."""
    # Initialize agent
    await agent.initialize()
    
    # Test empty requirements
    result = await agent.analyze_requirements("")
    assert result["status"] == "error"
    assert "empty" in result["message"].lower()
    
    # Test valid requirements
    requirements = "Create a function to calculate factorial"
    
    # Mock pipeline execution
    mock_result = Mock()
    mock_result.data = {"analysis": "Test analysis"}
    mock_result.metrics = {"tokens": 100}
    
    with patch.object(agent.pipeline, "execute_instruction", 
                     return_value=mock_result):
        result = await agent.analyze_requirements(requirements)
        
        # Verify result
        assert result["status"] == "success"
        assert "analysis" in result
        assert agent.metrics.tokens_used == 100
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_test_design(agent):
    """Test test case design."""
    # Initialize agent
    await agent.initialize()
    
    # Test empty analysis
    result = await agent.design_tests({})
    assert result["status"] == "error"
    assert "required" in result["message"].lower()
    
    # Test valid analysis
    analysis = {
        "function_name": "factorial",
        "parameters": ["n: int"],
        "return_type": "int"
    }
    
    # Mock pipeline execution
    mock_result = Mock()
    mock_result.data = {"test_cases": ["Test case 1"]}
    mock_result.metrics = {"tokens": 200}
    
    with patch.object(agent.pipeline, "execute_instruction",
                     return_value=mock_result):
        result = await agent.design_tests(analysis)
        
        # Verify result
        assert result["status"] == "success"
        assert "test_cases" in result
        assert agent.metrics.tokens_used == 200
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_code_implementation(agent):
    """Test code implementation."""
    # Initialize agent
    await agent.initialize()
    
    # Test empty test specs
    result = await agent.implement_code([])
    assert result["status"] == "error"
    assert "required" in result["message"].lower()
    
    # Test valid test specs
    test_specs = [{
        "name": "test_factorial_zero",
        "input": {"n": 0},
        "expected": 1
    }]
    
    # Mock pipeline execution
    mock_result = Mock()
    mock_result.data = {
        "implementation": "def factorial(n): return 1"
    }
    mock_result.metrics = {"tokens": 300}
    
    with patch.object(agent.pipeline, "execute_instruction",
                     return_value=mock_result):
        result = await agent.implement_code(test_specs)
        
        # Verify result
        assert result["status"] == "success"
        assert "code" in result
        assert agent.metrics.tokens_used == 300
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_code_review(agent):
    """Test code review."""
    # Initialize agent
    await agent.initialize()
    
    # Test empty implementation
    result = await agent.review_code({})
    assert result["status"] == "error"
    assert "required" in result["message"].lower()
    
    # Test valid implementation
    implementation = {
        "code": "def factorial(n): return 1"
    }
    
    # Mock pipeline execution
    mock_result = Mock()
    mock_result.data = {
        "review": {
            "issues": [],
            "suggestions": []
        }
    }
    mock_result.metrics = {"tokens": 400}
    
    with patch.object(agent.pipeline, "execute_instruction",
                     return_value=mock_result):
        result = await agent.review_code(implementation)
        
        # Verify result
        assert result["status"] == "success"
        assert "review" in result
        assert agent.metrics.tokens_used == 400
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_error_handling(agent):
    """Test error handling."""
    # Initialize agent
    await agent.initialize()
    
    # Mock pipeline to raise error
    with patch.object(agent.pipeline, "execute_instruction",
                     side_effect=RuntimeError("Test error")):
        # Test requirement analysis
        result = await agent.analyze_requirements("Test requirement")
        assert result["status"] == "error"
        assert "Test error" in result["message"]
        assert agent.metrics.error_count == 1
        
        # Test test design
        result = await agent.design_tests({"test": "data"})
        assert result["status"] == "error"
        assert "Test error" in result["message"]
        assert agent.metrics.error_count == 2
        
        # Test code implementation
        result = await agent.implement_code([{"test": "data"}])
        assert result["status"] == "error"
        assert "Test error" in result["message"]
        assert agent.metrics.error_count == 3
        
        # Test code review
        result = await agent.review_code({"code": "test"})
        assert result["status"] == "error"
        assert "Test error" in result["message"]
        assert agent.metrics.error_count == 4
    
    # Cleanup
    await agent.cleanup()

@pytest.mark.asyncio
async def test_metrics_update(agent):
    """Test metrics update."""
    # Initialize agent
    await agent.initialize()
    
    # Update metrics
    agent._update_metrics({
        "latency": 1.0,
        "tokens": 100,
        "success": 1.0
    })
    
    # Verify metrics
    assert agent.metrics.latency == 1.0
    assert agent.metrics.tokens_used == 100
    assert agent.metrics.success_rate == 1.0
    
    # Update again
    agent._update_metrics({
        "latency": 2.0,
        "tokens": 200,
        "success": 0.5
    })
    
    # Verify updated metrics
    assert agent.metrics.latency == 3.0
    assert agent.metrics.tokens_used == 300
    assert agent.metrics.success_rate == 0.75
    
    # Cleanup
    await agent.cleanup() 