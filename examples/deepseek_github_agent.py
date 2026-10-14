"""
DeepSeek GitHub Agent implementation.
This agent specializes in GitHub repository management and code operations:
it wires a DeepSeek provider (requirement analysis, test design, code
generation, code review) together with a GitHandler for repository
operations.
"""

import asyncio
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from agentflow.core.memory import WorkingMemory
from agentflow.providers.deepseek import DeepSeekProvider, DeepSeekConfig
from examples import git_handler as git_handler_module
from examples.git_handler import GitConfig

logger = logging.getLogger(__name__)


class DeepSeekGitHubAgent:
    """An AI agent powered by DeepSeek for GitHub repository operations.

    The agent combines two collaborators:

    * a ``DeepSeekProvider``-compatible LLM provider for the development
      workflow (analysis, test design, implementation, review), and
    * a ``GitHandler`` for repository operations (init, commit, branch,
      push, pull requests).

    Args:
        provider: LLM provider used for the development workflow. Defaults
            to a ``DeepSeekProvider`` with the default configuration.
        memory: ``WorkingMemory``-compatible store used to share the git
            handler across agent operations. Defaults to a new
            ``WorkingMemory``.
        workspace: Path to the working directory for git operations.
        github_config: Git configuration, either a ``GitConfig`` instance
            or a plain dictionary of ``GitConfig`` fields.
    """

    def __init__(
        self,
        provider: Optional[DeepSeekProvider] = None,
        memory: Optional[WorkingMemory] = None,
        workspace: Union[str, Path] = "./workspace",
        github_config: Optional[Union[GitConfig, Dict[str, Any]]] = None,
    ):
        """Initialize the GitHub agent with its collaborators."""
        self.provider = provider if provider is not None else DeepSeekProvider()
        self.memory = memory if memory is not None else WorkingMemory()
        self.workspace = Path(workspace)

        if github_config is None:
            github_config = GitConfig(
                username="deepseek-agent",
                email="deepseek-agent@noreply.github.com",
            )
        if isinstance(github_config, GitConfig):
            git_config = github_config
        else:
            git_config = git_handler_module.GitConfig(**github_config)

        # Reference GitHandler through the module so that
        # ``patch("examples.git_handler.GitHandler")`` takes effect.
        self.git_handler = git_handler_module.GitHandler(
            config=git_config,
            workspace=str(self.workspace),
        )
        self._initialized = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def initialize(self) -> None:
        """Initialize the agent.

        Creates the workspace, initializes the git repository, and
        registers the git handler in working memory.
        """
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.git_handler.init_repository()
        await self.memory.store("git_handler", self.git_handler)
        self._initialized = True

    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.provider is not None and hasattr(self.provider, "close"):
            await self.provider.close()
        await self.memory.clear()
        self._initialized = False

    # ------------------------------------------------------------------
    # Development workflow (provider-backed)
    # ------------------------------------------------------------------
    async def analyze_repository(self) -> Dict[str, Any]:
        """Analyze the current repository to derive requirements.

        Returns:
            Analysis results from the provider.
        """
        requirements = (
            "Analyze the current repository state and identify "
            "improvement opportunities"
        )
        analysis = await self.provider.analyze_requirements(requirements)
        return analysis if isinstance(analysis, dict) else {"analysis": analysis}

    async def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze the given requirements.

        Args:
            requirements: Requirements text to analyze.

        Returns:
            Analysis results from the provider.
        """
        analysis = await self.provider.analyze_requirements(requirements)
        return analysis if isinstance(analysis, dict) else {"analysis": analysis}

    async def design_tests(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Design test cases for the given analysis.

        Args:
            analysis: Requirement analysis produced by
                ``analyze_requirements``.

        Returns:
            List of designed test cases.
        """
        test_cases = await self.provider.design_tests(analysis)
        if isinstance(test_cases, list):
            return test_cases
        return [test_cases] if test_cases else []

    async def implement_code(
        self,
        test_specs: List[Dict[str, Any]],
        language: str = "python",
    ) -> str:
        """Implement code that satisfies the given test specifications.

        Args:
            test_specs: Test specifications from ``design_tests``.
            language: Target programming language.

        Returns:
            Generated code as a string.
        """
        code = await self.provider.generate_code(str(test_specs), language=language)
        return code if isinstance(code, str) else str(code)

    async def review_code(self, code: str) -> Dict[str, Any]:
        """Review the given code.

        Args:
            code: Code to review.

        Returns:
            Review results from the provider.
        """
        review = await self.provider.review_code(code)
        return review if isinstance(review, dict) else {"review": review}

    async def run_workflow(self, requirements: str, language: str = "python") -> Dict[str, Any]:
        """Run the full TDD workflow: analyze, design, implement, review.

        Args:
            requirements: Requirements text to analyze.
            language: Target programming language.

        Returns:
            Dictionary with analysis, test cases, implementation and review.
        """
        analysis = await self.analyze_requirements(requirements)
        test_cases = await self.design_tests(analysis)
        code = await self.implement_code(test_cases, language=language)
        review = await self.review_code(code)
        return {
            "analysis": analysis,
            "test_cases": test_cases,
            "code": code,
            "review": review,
        }

    # ------------------------------------------------------------------
    # Git operations (handler-backed)
    # ------------------------------------------------------------------
    async def commit_changes(
        self,
        files: List[str],
        message: str,
        scope: Optional[str] = None,
    ) -> str:
        """Commit changes to the repository.

        Args:
            files: Files to commit.
            message: Commit message.
            scope: Optional conventional-commit scope.

        Returns:
            The commit hash.
        """
        handler = await self._get_git_handler()
        return handler.create_commit(message=message, files=files, scope=scope)

    def get_repository_status(self) -> Dict[str, Any]:
        """Get the current repository status.

        Returns:
            Dictionary describing staged, unstaged and untracked files.
        """
        status = self.git_handler.get_status()
        return status if isinstance(status, dict) else {"status": status}

    def switch_branch(self, branch: str, create: bool = False) -> None:
        """Switch to a repository branch.

        Args:
            branch: Branch name to switch to.
            create: Whether to create the branch if it does not exist.
        """
        self.git_handler.switch_branch(branch, create)

    def push_changes(self) -> None:
        """Push committed changes to the remote repository."""
        self.git_handler.push_changes()

    def create_pull_request(self, title: str, body: str) -> None:
        """Create a pull request.

        Args:
            title: Pull request title.
            body: Pull request description.
        """
        self.git_handler.create_pull_request(title, body)

    async def _get_git_handler(self):
        """Resolve the git handler from memory, falling back to the local one."""
        handler = await self.memory.retrieve("git_handler")
        return handler if handler is not None else self.git_handler

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------
    def get_metrics(self) -> Dict[str, Any]:
        """Return the underlying provider's usage metrics when available."""
        metrics = getattr(self.provider, "_metrics", None)
        return dict(metrics) if isinstance(metrics, dict) else {}


async def main():
    """Main entry point demonstrating the DeepSeek GitHub agent."""
    agent = DeepSeekGitHubAgent(
        provider=DeepSeekProvider(
            config=DeepSeekConfig(model_name="deepseek-chat")
        ),
        memory=WorkingMemory(),
        workspace="./workspace",
        github_config={
            "username": "deepseek-agent",
            "email": "deepseek-agent@noreply.github.com",
            "repository_url": "https://github.com/example/repo.git",
            "default_branch": "main",
        },
    )

    try:
        await agent.initialize()

        result = await agent.run_workflow(
            "Implement a simple calculator function with division-by-zero handling"
        )
        print(f"Workflow result: {result}")

        status = agent.get_repository_status()
        print(f"Repository status: {status}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
