"""Git operations handler for repository management."""

import os
import asyncio
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging
from git import Repo, GitCommandError
from git.remote import Remote
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class GitConfig:
    """Configuration for Git operations.

    ``repository``/``repository_url`` are accepted as aliases (the
    tests use the former); ``branch`` aliases ``default_branch``.
    """

    def __init__(
        self,
        username: str,
        email: Optional[str] = None,
        default_branch: str = "main",
        commit_style: str = "conventional",
        auto_push: bool = True,
        create_pr: bool = False,
        repository: Optional[str] = None,
        repository_url: Optional[str] = None,
        branch: Optional[str] = None,
    ):
        self.username = username
        self.email = email
        self.repository_url = repository_url or repository
        self.default_branch = branch or default_branch
        self.commit_style = commit_style
        self.auto_push = auto_push
        self.create_pr = create_pr

class GitOperationError(Exception):
    """Exception raised for Git operation errors."""
    pass

class GitOperationsHandler:
    """Handler for Git repository operations."""
    
    def __init__(self, config: GitConfig):
        """Initialize Git operations handler.
        
        Args:
            config: Git configuration
        """
        self.config = config
        self._repo: Optional[Repo] = None
        self._metrics = {
            "commits": 0,
            "pushes": 0,
            "errors": 0,
            "file_changes": 0
        }
        
    def initialize_repo(self, path: str) -> None:
        """Initialize or open Git repository.
        
        Args:
            path: Repository path
            
        Raises:
            GitOperationError: If repository initialization fails
        """
        try:
            path = os.path.abspath(path)
            if os.path.exists(os.path.join(path, ".git")):
                self._repo = Repo(path)
            else:
                self._repo = Repo.init(path)
                
            # Configure repository
            with self._repo.config_writer() as git_config:
                git_config.set_value("user", "name", self.config.username)
                if self.config.email:
                    git_config.set_value("user", "email", self.config.email)
                    
        except Exception as e:
            raise GitOperationError(f"Failed to initialize repository: {str(e)}")
            
    def get_repo_status(self) -> Dict[str, Any]:
        """Get repository status.
        
        Returns:
            Dictionary with repository status
            
        Raises:
            GitOperationError: If status check fails
        """
        if not self._repo:
            raise GitOperationError("Repository not initialized")
            
        try:
            return {
                "branch": self._repo.active_branch.name,
                "is_dirty": self._repo.is_dirty(),
                "untracked_files": self._repo.untracked_files,
                "staged_files": [item.a_path for item in self._repo.index.diff("HEAD")],
                "modified_files": [item.a_path for item in self._repo.index.diff(None)]
            }
        except Exception as e:
            raise GitOperationError(f"Failed to get repository status: {str(e)}")
            
    def stage_changes(self, files: Optional[List[str]] = None) -> None:
        """Stage changes for commit.
        
        Args:
            files: Optional list of files to stage. If None, stages all changes.
            
        Raises:
            GitOperationError: If staging fails
        """
        if not self._repo:
            raise GitOperationError("Repository not initialized")
            
        try:
            if files:
                self._repo.index.add(files)
            else:
                self._repo.git.add(A=True)
                
            self._metrics["file_changes"] += len(self._repo.index.diff("HEAD"))
            
        except Exception as e:
            raise GitOperationError(f"Failed to stage changes: {str(e)}")
            
    def create_commit(self, 
                     message: str,
                     files: Optional[List[str]] = None,
                     scope: Optional[str] = None) -> str:
        """Create a commit with staged changes.
        
        Args:
            message: Commit message
            files: Optional list of files to commit
            scope: Optional scope for conventional commits
            
        Returns:
            Commit hash
            
        Raises:
            GitOperationError: If commit creation fails
        """
        if not self._repo:
            raise GitOperationError("Repository not initialized")
            
        try:
            # Stage files if provided
            if files:
                self.stage_changes(files)
                
            # Format commit message
            if self.config.commit_style == "conventional" and scope:
                message = f"{message.split(':')[0]}({scope}): {message.split(':', 1)[1].strip()}"
                
            # Create commit
            commit = self._repo.index.commit(message)
            self._metrics["commits"] += 1
            
            # Auto-push if enabled
            if self.config.auto_push:
                self.push_changes()
                
            return commit.hexsha
            
        except Exception as e:
            raise GitOperationError(f"Failed to create commit: {str(e)}")
            
    def push_changes(self, 
                    remote: str = "origin",
                    branch: Optional[str] = None) -> None:
        """Push changes to remote repository.
        
        Args:
            remote: Remote name
            branch: Branch to push to. If None, uses current branch.
            
        Raises:
            GitOperationError: If push fails
        """
        if not self._repo:
            raise GitOperationError("Repository not initialized")
            
        try:
            if not branch:
                branch = self._repo.active_branch.name
                
            origin: Remote = self._repo.remote(name=remote)
            origin.push(refspec=f"{branch}:{branch}")
            self._metrics["pushes"] += 1
            
        except Exception as e:
            raise GitOperationError(f"Failed to push changes: {str(e)}")
            
    def checkout_branch(self, 
                       branch: str,
                       create: bool = False,
                       base: Optional[str] = None) -> None:
        """Checkout or create a branch.
        
        Args:
            branch: Branch name
            create: Whether to create a new branch
            base: Base branch for new branch
            
        Raises:
            GitOperationError: If checkout fails
        """
        if not self._repo:
            raise GitOperationError("Repository not initialized")
            
        try:
            if create:
                if base:
                    base_branch = self._repo.branches[base]
                else:
                    base_branch = self._repo.active_branch
                    
                new_branch = self._repo.create_head(branch, base_branch)
                new_branch.checkout()
            else:
                self._repo.git.checkout(branch)
                
        except Exception as e:
            raise GitOperationError(f"Failed to checkout branch: {str(e)}")
            
    def get_metrics(self) -> Dict[str, int]:
        """Get Git operations metrics.
        
        Returns:
            Dictionary with metrics
        """
        return self._metrics.copy()
        
    def cleanup(self) -> None:
        """Cleanup Git operations handler."""
        self._repo = None 