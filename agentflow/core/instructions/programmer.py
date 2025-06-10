"""Programmer instructions for AI coding operations."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import asyncio

from agentflow.core.instructions.base import (
    BaseInstruction,
    InstructionResult,
    InstructionStatus,
    FileSystemInstruction,
    ProcessInstruction,
    NetworkInstruction,
    CompositeInstruction,
    OptimizableInstruction,
    CacheableInstruction
)
from agentflow.core.instructions.thinking import (
    ThinkingInstruction,
    ThinkingStyle,
    CognitivePhase,
    ThoughtContext,
    ThoughtMetrics
)

logger = logging.getLogger(__name__)

class ProgrammerThinkingInstruction(ThinkingInstruction):
    """Specialized thinking instruction for programming tasks."""
    
    def __init__(self):
        super().__init__()
        self.name = "programmer_thinking"
        self.description = "Execute programmer-specific thinking process"
        self._code_insights = []
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute programmer thinking process."""
        try:
            # Initialize context with programming-specific fields
            context.update({
                "code_patterns": context.get("code_patterns", []),
                "design_patterns": context.get("design_patterns", []),
                "best_practices": context.get("best_practices", []),
                "constraints": context.get("constraints", []) + [
                    "Must follow clean code principles",
                    "Must be maintainable",
                    "Must be testable"
                ]
            })
            
            # Execute base thinking process
            result = await super()._execute_impl(context)
            
            if result.status == str(InstructionStatus.COMPLETED):
                # Add programming-specific insights
                result.data["code_insights"] = self._code_insights
                
            return result
            
        except Exception as e:
            error_msg = f"Error in programmer thinking process: {str(e)}"
            logger.error(error_msg)
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )
    
    async def _analyze(self) -> None:
        """Extended analysis phase for programming tasks."""
        await super()._analyze()
        
        if not self._context or not self._metrics:
            return
            
        # Analyze code patterns
        self._add_thought("Analyzing code patterns...")
        for pattern in self._context.known_facts:
            if "pattern:" in pattern:
                self._add_thought(f"Identified pattern: {pattern}")
                self._code_insights.append({
                    "type": "pattern",
                    "content": pattern
                })
        
        # Analyze design decisions
        self._add_thought("Evaluating design decisions...")
        for constraint in self._context.constraints:
            self._add_thought(f"Considering constraint: {constraint}")
            if "design:" in constraint:
                self._code_insights.append({
                    "type": "design",
                    "content": constraint
                })

class CodeFileInstruction(FileSystemInstruction):
    """Specialized instruction for code file operations."""
    
    def __init__(self):
        super().__init__(
            name="code_file_operations",
            description="Handle code file operations with version control"
        )
        self._file_history = []
    
    async def _read_file(self, file_path: str) -> Dict[str, Any]:
        """Read a code file."""
        try:
            content = Path(file_path).read_text()
            return {
                "content": content,
                "file_path": file_path,
                "size": len(content)
            }
        except Exception as e:
            raise Exception(f"Error reading file {file_path}: {str(e)}")
    
    async def _write_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Write content to a code file."""
        try:
            Path(file_path).write_text(content)
            return {
                "file_path": file_path,
                "size": len(content),
                "operation": "write"
            }
        except Exception as e:
            raise Exception(f"Error writing file {file_path}: {str(e)}")
    
    async def _update_file(self, file_path: str, content: str) -> Dict[str, Any]:
        """Update an existing code file."""
        try:
            if not Path(file_path).exists():
                raise Exception(f"File {file_path} does not exist")
            
            # Read existing content
            old_content = await self._read_file(file_path)
            
            # Write new content
            result = await self._write_file(file_path, content)
            result.update({
                "operation": "update",
                "old_size": len(old_content["content"]),
                "size_diff": len(content) - len(old_content["content"])
            })
            return result
            
        except Exception as e:
            raise Exception(f"Error updating file {file_path}: {str(e)}")
    
    async def _delete_file(self, file_path: str) -> Dict[str, Any]:
        """Delete a code file."""
        try:
            if not Path(file_path).exists():
                raise Exception(f"File {file_path} does not exist")
            
            # Get file info before deletion
            file_info = await self._read_file(file_path)
            
            # Delete file
            Path(file_path).unlink()
            
            return {
                "file_path": file_path,
                "operation": "delete",
                "size": file_info["size"]
            }
        except Exception as e:
            raise Exception(f"Error deleting file {file_path}: {str(e)}")
    
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute code file operation."""
        try:
            operation = context.get("operation")
            file_path = context.get("file_path")
            content = context.get("content", "")  # Default to empty string
            
            if not operation or not file_path:
                return InstructionResult(
                    status=str(InstructionStatus.FAILED),
                    data={"error": "Missing operation or file path"},
                    metrics={}
                )
            
            # Record operation in history
            self._file_history.append({
                "operation": operation,
                "file_path": file_path,
                "timestamp": context.get("timestamp")
            })
            
            # Execute operation
            if operation == "read":
                result = await self._read_file(file_path)
            elif operation == "write":
                if not content:
                    return InstructionResult(
                        status=str(InstructionStatus.FAILED),
                        data={"error": "Content is required for write operation"},
                        metrics={}
                    )
                result = await self._write_file(file_path, content)
            elif operation == "update":
                if not content:
                    return InstructionResult(
                        status=str(InstructionStatus.FAILED),
                        data={"error": "Content is required for update operation"},
                        metrics={}
                    )
                result = await self._update_file(file_path, content)
            elif operation == "delete":
                result = await self._delete_file(file_path)
            else:
                return InstructionResult(
                    status=str(InstructionStatus.FAILED),
                    data={"error": f"Unknown operation: {operation}"},
                    metrics={}
                )
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=result,
                metrics={"history_length": len(self._file_history)}
            )
            
        except Exception as e:
            error_msg = f"Error in code file operation: {str(e)}"
            logger.error(error_msg)
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )

class GitWorkflowInstruction(CompositeInstruction):
    """Instruction for managing Git workflow."""
    
    def __init__(self):
        super().__init__(
            name="git_workflow",
            description="Manage Git workflow operations"
        )
        self._workflow_state = {}
    
    async def _create_commit(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Git commit."""
        try:
            message = context.get("message")
            files = context.get("files", [])
            
            if not message:
                raise Exception("Commit message is required")
            
            # Stage files
            if files:
                await self._stage_files(files)
            
            # Create commit
            result = await self._run_git_command(["commit", "-m", message])
            
            return {
                "operation": "commit",
                "message": message,
                "files": files,
                "hash": result.get("hash", "")
            }
        except Exception as e:
            raise Exception(f"Error creating commit: {str(e)}")
    
    async def _manage_branch(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Manage Git branches."""
        try:
            action = context.get("action")
            branch = context.get("branch")
            
            if not action or not branch:
                raise Exception("Branch action and name are required")
            
            if action == "create":
                result = await self._run_git_command(["checkout", "-b", branch])
            elif action == "switch":
                result = await self._run_git_command(["checkout", branch])
            elif action == "delete":
                result = await self._run_git_command(["branch", "-D", branch])
            else:
                raise Exception(f"Unknown branch action: {action}")
            
            return {
                "operation": "branch",
                "action": action,
                "branch": branch,
                "result": result
            }
        except Exception as e:
            raise Exception(f"Error managing branch: {str(e)}")
    
    async def _stage_files(self, files: List[str]) -> Dict[str, Any]:
        """Stage files for commit."""
        try:
            result = await self._run_git_command(["add"] + files)
            return {
                "operation": "stage",
                "files": files,
                "result": result
            }
        except Exception as e:
            raise Exception(f"Error staging files: {str(e)}")
    
    async def _run_git_command(self, args: List[str]) -> Dict[str, Any]:
        """Run a Git command."""
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                raise Exception(f"Git command failed: {stderr.decode()}")
            
            return {
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "returncode": process.returncode
            }
        except Exception as e:
            raise Exception(f"Error running Git command: {str(e)}")
    
    async def _create_github_pr(
        self,
        title: str,
        body: Optional[str] = None,
        base: str = "main",
        head: str = ""
    ) -> Dict[str, Any]:
        """Create a GitHub pull request."""
        # This is a placeholder - actual implementation would use GitHub API
        return {
            "operation": "create_pr",
            "title": title,
            "body": body or "",
            "base": base,
            "head": head
        }
    
    async def _update_github_pr(
        self,
        pr_number: int,
        title: Optional[str] = None,
        body: Optional[str] = None
    ) -> Dict[str, Any]:
        """Update a GitHub pull request."""
        # This is a placeholder - actual implementation would use GitHub API
        return {
            "operation": "update_pr",
            "pr_number": pr_number,
            "title": title or "",
            "body": body or ""
        }
    
    async def _manage_pr(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Manage pull requests."""
        try:
            action = context.get("action")
            if not action:
                raise Exception("PR action is required")
            
            if action == "create":
                title = context.get("title")
                if not title:
                    raise Exception("PR title is required")
                
                head = context.get("head")
                if not head:
                    raise Exception("Head branch is required")
                
                # Create PR
                result = await self._create_github_pr(
                    title=title,
                    body=context.get("body"),
                    base=context.get("base", "main"),
                    head=head
                )
                
            elif action == "update":
                pr_number = context.get("pr_number")
                if not pr_number:
                    raise Exception("PR number is required for update")
                
                # Update PR
                result = await self._update_github_pr(
                    pr_number=pr_number,
                    title=context.get("title"),
                    body=context.get("body")
                )
                
            else:
                raise Exception(f"Unknown PR action: {action}")
            
            return {
                "operation": "pr",
                "action": action,
                "result": result
            }
        except Exception as e:
            raise Exception(f"Error managing PR: {str(e)}")

class ContextManagerInstruction(BaseInstruction):
    """Instruction for managing programming context."""
    
    def __init__(self):
        super().__init__(
            name="context_manager",
            description="Manage programming context and state"
        )
        self._context_history = []
        self._current_context = {}
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute context management operation."""
        try:
            operation = context.get("operation")
            if not operation:
                return InstructionResult(
                    status=str(InstructionStatus.FAILED),
                    data={"error": "Missing operation"},
                    metrics={}
                )
            
            # Update context history
            self._context_history.append({
                "operation": operation,
                "context": context.copy(),
                "timestamp": context.get("timestamp")
            })
            
            # Execute operation
            if operation == "update":
                self._current_context.update(context.get("updates", {}))
                result = {"context": self._current_context}
            elif operation == "query":
                result = {
                    "context": self._current_context,
                    "history": self._context_history
                }
            elif operation == "reset":
                self._current_context = {}
                result = {"context": self._current_context}
            else:
                return InstructionResult(
                    status=str(InstructionStatus.FAILED),
                    data={"error": f"Unknown operation: {operation}"},
                    metrics={}
                )
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=result,
                metrics={"history_length": len(self._context_history)}
            )
            
        except Exception as e:
            error_msg = f"Error in context management: {str(e)}"
            logger.error(error_msg)
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            ) 