"""Tests for DeepSeek GitHub agent."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

from agentflow.core.memory import WorkingMemory
from agentflow.providers.deepseek import DeepSeekProvider
from agentflow.core.instructions import InstructionResult

from examples.deepseek_github_agent import DeepSeekGitHubAgent
from examples.git_handler import GitHandler, GitConfig

@pytest.fixture
def mock_provider():
    """Create a mock DeepSeek provider."""
    provider = MagicMock(spec=DeepSeekProvider)
    provider.analyze_requirements = AsyncMock(return_value={})
    provider.design_tests = AsyncMock(return_value=[])
    provider.generate_code = AsyncMock(return_value="test code")
    provider.review_code = AsyncMock(return_value={})
    provider._metrics = {"total_tokens": 0}
    return provider

@pytest.fixture
def mock_memory():
    """Create a mock working memory."""
    memory = MagicMock(spec=WorkingMemory)
    memory.store = AsyncMock()
    memory.retrieve = AsyncMock()
    return memory

@pytest.fixture
def github_config():
    """Create a test GitHub configuration."""
    return {
        "username": "test-user",
        "email": "test@example.com",
        "repository_url": "https://github.com/test/repo.git",
        "default_branch": "main",
        "commit_style": "conventional",
        "auto_push": False,
        "create_pr": False
    }

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace."""
    return str(tmp_path)

@pytest.fixture
def agent(mock_provider, mock_memory, github_config, temp_workspace):
    """Create a DeepSeek GitHub agent instance."""
    with patch("examples.git_handler.GitHandler") as mock_handler:
        mock_handler.return_value = MagicMock(spec=GitHandler)
        agent = DeepSeekGitHubAgent(
            provider=mock_provider,
            memory=mock_memory,
            workspace=temp_workspace,
            github_config=github_config
        )
        yield agent

@pytest.mark.asyncio
async def test_initialize(agent):
    """Test agent initialization."""
    await agent.initialize()
    
    agent.git_handler.init_repository.assert_called_once()
    agent.memory.store.assert_called_once_with("git_handler", agent.git_handler)

@pytest.mark.asyncio
async def test_analyze_repository(agent):
    """Test repository analysis."""
    analysis = await agent.analyze_repository()
    
    agent.provider.analyze_requirements.assert_called_once()
    assert isinstance(analysis, dict)

@pytest.mark.asyncio
async def test_analyze_requirements(agent):
    """Test requirements analysis."""
    requirements = "Test requirements"
    analysis = await agent.analyze_requirements(requirements)
    
    agent.provider.analyze_requirements.assert_called_once_with(requirements)
    assert isinstance(analysis, dict)

@pytest.mark.asyncio
async def test_design_tests(agent):
    """Test test case design."""
    analysis = {"test": "data"}
    test_cases = await agent.design_tests(analysis)
    
    agent.provider.design_tests.assert_called_once_with(analysis)
    assert isinstance(test_cases, list)

@pytest.mark.asyncio
async def test_implement_code(agent):
    """Test code implementation."""
    test_specs = [{"test": "spec"}]
    code = await agent.implement_code(test_specs, language="python")
    
    agent.provider.generate_code.assert_called_once_with(
        str(test_specs),
        language="python"
    )
    assert isinstance(code, str)

@pytest.mark.asyncio
async def test_review_code(agent):
    """Test code review."""
    code = "def test(): pass"
    review = await agent.review_code(code)
    
    agent.provider.review_code.assert_called_once_with(code)
    assert isinstance(review, dict)

@pytest.mark.asyncio
async def test_commit_changes(agent, mock_memory):
    """Test committing changes."""
    files = ["test.py"]
    message = "test: add test file"
    scope = "test"
    
    mock_git_handler = MagicMock(spec=GitHandler)
    mock_git_handler.create_commit.return_value = "test-hash"
    mock_memory.retrieve.return_value = mock_git_handler
    
    commit_hash = await agent.commit_changes(files, message, scope)
    
    mock_git_handler.create_commit.assert_called_once_with(
        message=message,
        files=files,
        scope=scope
    )
    assert commit_hash == "test-hash"

def test_get_repository_status(agent):
    """Test getting repository status."""
    status = agent.get_repository_status()
    
    agent.git_handler.get_status.assert_called_once()
    assert isinstance(status, dict)

def test_switch_branch(agent):
    """Test switching branches."""
    branch = "feature"
    create = True
    
    agent.switch_branch(branch, create)
    
    agent.git_handler.switch_branch.assert_called_once_with(branch, create)

def test_push_changes(agent):
    """Test pushing changes."""
    agent.push_changes()
    
    agent.git_handler.push_changes.assert_called_once()

def test_create_pull_request(agent):
    """Test creating pull request."""
    title = "Test PR"
    body = "Test description"
    
    agent.create_pull_request(title, body)
    
    agent.git_handler.create_pull_request.assert_called_once_with(title, body) 