"""Tests for Git operations handler."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from examples.git_handler import GitHandler, GitConfig

@pytest.fixture
def git_config():
    """Create a test Git configuration."""
    return GitConfig(
        username="test-user",
        email="test@example.com",
        repository="https://github.com/test/repo.git",
        branch="main",
        commit_style="conventional",
        auto_push=False,
        create_pr=False
    )

@pytest.fixture
def temp_workspace(tmp_path):
    """Create a temporary workspace."""
    return str(tmp_path)

@pytest.fixture
def git_handler(git_config, temp_workspace):
    """Create a Git handler instance."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="")
        handler = GitHandler(git_config, temp_workspace)
        yield handler

def test_init_repository(git_handler):
    """Test repository initialization."""
    with patch("pathlib.Path.exists", return_value=False):
        git_handler.init_repository()
        
        assert len(git_handler._run_git_command.mock_calls) == 3
        commands = [call.args[0] for call in git_handler._run_git_command.mock_calls]
        
        assert ["git", "init"] in commands
        assert ["git", "checkout", "-b", "main"] in commands
        assert ["git", "remote", "add", "origin", "https://github.com/test/repo.git"] in commands

def test_get_status(git_handler):
    """Test getting repository status."""
    mock_status = """A  new.txt
M  modified.txt
 M unstaged.txt
?? untracked.txt"""
    
    with patch.object(git_handler, "_run_git_command") as mock_run:
        mock_run.return_value = MagicMock(stdout=mock_status)
        status = git_handler.get_status()
        
        assert status["staged"] == ["new.txt", "modified.txt"]
        assert status["unstaged"] == ["unstaged.txt"]
        assert status["untracked"] == ["untracked.txt"]
        
        mock_run.assert_called_once_with(["git", "status", "--porcelain"])

def test_stage_files(git_handler):
    """Test staging files."""
    files = ["file1.txt", "file2.txt"]
    git_handler.stage_files(files)
    
    git_handler._run_git_command.assert_called_once_with(
        ["git", "add", "file1.txt", "file2.txt"]
    )

def test_create_commit(git_handler):
    """Test commit creation."""
    message = "feat: add new feature"
    files = ["file1.txt"]
    
    with patch.object(git_handler, "stage_files") as mock_stage:
        git_handler.create_commit(message, files)
        
        mock_stage.assert_called_once_with(files)
        assert len(git_handler._run_git_command.mock_calls) == 2
        
        commit_call = git_handler._run_git_command.mock_calls[0]
        assert commit_call.args[0] == ["git", "commit", "-m", message]

def test_create_commit_with_scope(git_handler):
    """Test commit creation with scope."""
    message = "feat: add new feature"
    scope = "api"
    
    git_handler.create_commit(message, scope=scope)
    
    commit_call = git_handler._run_git_command.mock_calls[0]
    assert commit_call.args[0] == ["git", "commit", "-m", f"{message}(api)"]

def test_push_changes(git_handler):
    """Test pushing changes."""
    git_handler.push_changes()
    
    git_handler._run_git_command.assert_called_once_with(
        ["git", "push", "-u", "origin", "main"]
    )

def test_get_current_branch(git_handler):
    """Test getting current branch."""
    with patch.object(git_handler, "_run_git_command") as mock_run:
        mock_run.return_value = MagicMock(stdout="feature-branch\n")
        branch = git_handler.get_current_branch()
        
        assert branch == "feature-branch"
        mock_run.assert_called_once_with(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        )

def test_switch_branch(git_handler):
    """Test switching branches."""
    git_handler.switch_branch("feature", create=True)
    
    git_handler._run_git_command.assert_called_once_with(
        ["git", "checkout", "-b", "feature"]
    )

def test_switch_existing_branch(git_handler):
    """Test switching to existing branch."""
    git_handler.switch_branch("main")
    
    git_handler._run_git_command.assert_called_once_with(
        ["git", "checkout", "main"]
    ) 