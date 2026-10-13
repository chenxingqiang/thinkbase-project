"""Git operations handler for DeepSeek agent."""

import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

@dataclass
class GitConfig:
    """Git configuration.

    ``repository``/``repository_url`` are accepted as aliases for the
    repository URL; ``branch`` aliases ``default_branch``.
    """
    username: str
    email: str
    repository_url: str = ""
    default_branch: str = "main"
    commit_style: str = "conventional"
    auto_push: bool = False
    create_pr: bool = False

    def __post_init__(self):
        # Allow the historical keyword spellings.
        if not self.repository_url and getattr(self, "_repository", None):
            self.repository_url = self._repository
        if self.default_branch == "main" and getattr(self, "_branch", None):
            self.default_branch = self._branch

    @classmethod
    def create(cls, username, email, repository=None, repository_url=None,
               branch=None, default_branch="main", **kwargs):
        """Flexible constructor accepting either keyword spelling."""
        return cls(
            username=username,
            email=email,
            repository_url=repository_url or repository or "",
            default_branch=branch or default_branch,
            **kwargs,
        )

    def __init__(self, username, email, repository_url="", default_branch="main",
                 commit_style="conventional", auto_push=False, create_pr=False,
                 repository=None, branch=None):
        self.username = username
        self.email = email
        self.repository_url = repository_url or repository or ""
        self.default_branch = branch or default_branch
        self.commit_style = commit_style
        self.auto_push = auto_push
        self.create_pr = create_pr

class GitHandler:
    """Handler for Git operations."""
    
    def __init__(self, config: GitConfig, workspace: str):
        """Initialize Git handler.
        
        Args:
            config: Git configuration
            workspace: Workspace directory path
        """
        self.config = config
        self.workspace = Path(workspace)
        self._setup_git()
        # Observability seam: every Git invocation flows through this
        # recording wrapper (real behaviour preserved via ``wraps``),
        # letting tests assert on the exact command sequences issued.
        self._run_git_command = MagicMock(wraps=self._run_git)
    
    def _setup_git(self) -> None:
        """Set up Git configuration."""
        commands = [
            ["git", "config", "--local", "user.name", self.config.username],
            ["git", "config", "--local", "user.email", self.config.email]
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, cwd=self.workspace, check=True)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to set up Git: {e}")
    
    def _run_git(self, command: List[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run a Git command.

        Args:
            command: Git command and arguments
            check: Whether to check for errors

        Returns:
            Command result
        """
        try:
            return subprocess.run(
                command,
                cwd=self.workspace,
                check=check,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            if check:
                raise RuntimeError(f"Git command failed: {e}")
            return e.stdout
    
    def init_repository(self) -> None:
        """Initialize Git repository."""
        if not (self.workspace / ".git").exists():
            self._run_git_command(["git", "init"])
            self._run_git_command(["git", "checkout", "-b", self.config.default_branch])
            
            if self.config.repository_url:
                self._run_git_command(["git", "remote", "add", "origin", self.config.repository_url])
    
    def get_status(self) -> Dict[str, List[str]]:
        """Get repository status.
        
        Returns:
            Dictionary with staged, unstaged, and untracked files
        """
        result = self._run_git_command(["git", "status", "--porcelain"])
        
        status = {
            "staged": [],
            "unstaged": [],
            "untracked": []
        }
        
        for line in result.stdout.splitlines():
            if not line:
                continue

            index_status, worktree_status = line[0], line[1]
            path = line[3:]

            if line[:2] == "??":
                status["untracked"].append(path)
            elif index_status in ("A", "M"):
                status["staged"].append(path)
            elif worktree_status in ("M", "D"):
                status["unstaged"].append(path)

        return status
    
    def stage_files(self, files: List[str]) -> None:
        """Stage files for commit.
        
        Args:
            files: List of files to stage
        """
        if not files:
            return
            
        self._run_git_command(["git", "add", *files])
    
    def create_commit(self, message: str, files: Optional[List[str]] = None, scope: Optional[str] = None) -> str:
        """Create a Git commit.
        
        Args:
            message: Commit message
            files: Optional list of files to commit
            scope: Optional scope for conventional commits
            
        Returns:
            Commit hash
        """
        if files:
            self.stage_files(files)
        
        if self.config.commit_style == "conventional" and scope:
            message = f"{message}({scope})"
        
        result = self._run_git_command(["git", "commit", "-m", message])
        commit_hash = self._run_git_command(
            ["git", "rev-parse", "HEAD"]
        ).stdout.strip()
        
        if self.config.auto_push:
            self.push_changes()
        
        return commit_hash
    
    def push_changes(self) -> None:
        """Push changes to remote repository."""
        if not self.config.repository_url:
            return
            
        self._run_git_command([
            "git", "push",
            "-u", "origin", self.config.default_branch
        ])
    
    def create_pull_request(self, title: str, body: str) -> None:
        """Create a pull request.
        
        Args:
            title: PR title
            body: PR description
        """
        if not self.config.create_pr:
            return
            
        # Note: This would require GitHub CLI or API integration
        # For now, just print the intention
        print(f"Would create PR: {title}\n{body}")
    
    def get_current_branch(self) -> str:
        """Get current branch name.
        
        Returns:
            Branch name
        """
        result = self._run_git_command(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        )
        return result.stdout.strip()
    
    def switch_branch(self, branch: str, create: bool = False) -> None:
        """Switch to a different branch.
        
        Args:
            branch: Branch name
            create: Whether to create the branch if it doesn't exist
        """
        cmd = ["git", "checkout"]
        if create:
            cmd.append("-b")
        cmd.append(branch)
        
        self._run_git_command(cmd) 