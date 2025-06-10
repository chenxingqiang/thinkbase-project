"""File operation instructions."""

from typing import Dict, Any, Optional, List, Union, Tuple, Set, TypeVar, cast, Sequence
import os
import shutil
import stat
import time
from pathlib import Path
import logging
from datetime import datetime
import re
import textwrap
import difflib
from typing_extensions import TypedDict
import glob
from dataclasses import dataclass
import statistics
from enum import Enum
import fnmatch

from .base import BaseInstruction, InstructionResult, InstructionStatus, InstructionMetrics

logger = logging.getLogger(__name__)

T = TypeVar('T')

@dataclass
class FileChangeMetrics:
    """Metrics for file changes."""
    lines_changed: int
    chars_changed: int
    timestamp: float
    matches_found: Optional[int] = None

@dataclass
class DiffInfo:
    """Information about a diff operation."""
    operation: str
    position: int
    content: List[str]
    lines: Optional[List[str]] = None
    command: Optional[str] = None
    args: Optional[Dict[str, Any]] = None

    def to_sequence(self) -> List[str]:
        """Convert to sequence of strings."""
        return [
            f"Operation: {self.operation}",
            f"Position: {self.position}",
            f"Content: {len(self.content)} lines"
        ]

class FileCreateInstruction(BaseInstruction):
    """Create file with content."""
    
    def __init__(self):
        super().__init__(
            name="file_create",
            description="Create a new file with content"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Create file with specified content."""
        path = context.get("path")
        content = context.get("content", "")
        mode = context.get("mode", "w")  # w for write, a for append
        encoding = context.get("encoding", "utf-8")
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        try:
            with open(path, mode, encoding=encoding) as f:
                f.write(content)
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"path": path, "size": len(content)},
                metrics={"chars_written": len(content)}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileReadInstruction(BaseInstruction):
    """Read file content."""
    
    def __init__(self):
        super().__init__(
            name="file_read",
            description="Read file content"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Read file content with options."""
        path = context.get("path")
        encoding = context.get("encoding", "utf-8")
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        try:
            with open(path, "r", encoding=encoding) as f:
                content = f.read()
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"path": path, "content": content, "size": len(content)},
                metrics={"chars_read": len(content)}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileUpdateInstruction(BaseInstruction):
    """Update file content."""
    
    def __init__(self):
        super().__init__(
            name="file_update",
            description="Update file content"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Update file content with options."""
        path = context.get("path")
        content = context.get("content")
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        if content is None:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "Content not provided"}
            )
            
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"path": path, "size": len(content)},
                metrics={"chars_written": len(content)}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileDeleteInstruction(BaseInstruction):
    """Delete file."""
    
    def __init__(self):
        super().__init__(
            name="file_delete",
            description="Delete file"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Delete file with options."""
        path = context.get("path")
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        try:
            os.remove(path)
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"path": path},
                metrics={"deleted": True}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileSearchInstruction(BaseInstruction):
    """Search for files."""
    
    def __init__(self):
        super().__init__(
            name="file_search",
            description="Search for files matching pattern"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Search for files matching pattern."""
        pattern = context.get("pattern")
        root_dir = context.get("root_dir", ".")
        recursive = context.get("recursive", True)
        
        if not pattern:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "Search pattern not provided"}
            )
            
        try:
            results = []
            for root, _, files in os.walk(root_dir):
                if not recursive and root != root_dir:
                    continue
                    
                for file in files:
                    if fnmatch.fnmatch(file, pattern):
                        results.append(os.path.join(root, file))
                        
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"matches": results, "count": len(results)},
                metrics={"matches_found": len(results)}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileMetadataInstruction(BaseInstruction):
    """Get file metadata."""
    
    def __init__(self):
        super().__init__(
            name="file_metadata",
            description="Get file metadata"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Get file metadata."""
        path = context.get("path")
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        try:
            stat_info = os.stat(path)
            metadata = {
                "path": path,
                "size": stat_info.st_size,
                "created": stat_info.st_ctime,
                "modified": stat_info.st_mtime,
                "accessed": stat_info.st_atime,
                "mode": stat_info.st_mode,
                "is_file": os.path.isfile(path),
                "is_dir": os.path.isdir(path),
                "is_link": os.path.islink(path)
            }
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data=metadata,
                metrics={"metadata_fields": len(metadata)}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

class FileEditInstruction(BaseInstruction):
    """Edit file content."""
    
    def __init__(self):
        super().__init__(
            name="file_edit",
            description="Edit file content"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute file edit operation."""
        path = context.get("path")
        operation = context.get("operation", {})
        
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "File path not provided"}
            )
            
        if not operation:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": "No edit operation specified"}
            )
            
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                
            lines = content.splitlines()
            operation_type = operation.get("type", "")
            
            if operation_type == "insert":
                line_number = operation.get("line", 1)
                text = operation.get("text", "")
                if 1 <= line_number <= len(lines) + 1:
                    lines.insert(line_number - 1, text)
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line number: {line_number}"}
                    )
            elif operation_type == "delete":
                line_number = operation.get("line", 1)
                if 1 <= line_number <= len(lines):
                    lines.pop(line_number - 1)
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line number: {line_number}"}
                    )
            elif operation_type == "replace":
                line_number = operation.get("line", 1)
                text = operation.get("text", "")
                if 1 <= line_number <= len(lines):
                    lines[line_number - 1] = text
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line number: {line_number}"}
                    )
            else:
                return InstructionResult(
                    status=InstructionStatus.FAILED.value,
                    data={},
                    metrics={"error_message": f"Unknown operation type: {operation_type}"}
                )
                
            # Write changes back to file
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                
            return InstructionResult(
                status=InstructionStatus.COMPLETED.value,
                data={"path": path, "lines": lines},
                metrics={"lines_modified": 1}
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={},
                metrics={"error_message": str(e)}
            )

    def _create_backup(self, file_path: str) -> Optional[str]:
        """Create a backup of the file."""
        try:
            backup_dir = os.path.join(os.path.dirname(file_path), ".backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{os.path.basename(file_path)}.{timestamp}.bak"
            backup_path = os.path.join(backup_dir, backup_name)
            
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups
            pattern = f"{os.path.basename(file_path)}.*.bak"
            backup_files = []
            for f in os.listdir(backup_dir):
                if fnmatch.fnmatch(f, pattern):
                    full_path = os.path.join(backup_dir, f)
                    backup_files.append((full_path, os.path.getmtime(full_path)))
            
            # Sort by modification time (newest first) and remove excess backups
            backup_files.sort(key=lambda x: x[1], reverse=True)
            for path, _ in backup_files[5:]:  # Keep only 5 most recent backups
                try:
                    os.remove(path)
                except Exception as e:
                    logger.error(f"Failed to remove old backup {path}: {e}")
            
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
            
    def _cleanup_old_backups(self, file_path: str) -> None:
        """Clean up old backup files, keeping only the most recent ones.
        
        Args:
            file_path: Path to the original file
        """
        try:
            backup_dir = os.path.join(os.path.dirname(file_path), ".backups")
            if not os.path.exists(backup_dir):
                return
                
            # Get all backup files for this file
            base_name = os.path.basename(file_path)
            backup_files = []
            for f in os.listdir(backup_dir):
                if f.startswith(base_name + "."):
                    backup_path = os.path.join(backup_dir, f)
                    timestamp = int(f.split(".")[-1])
                    backup_files.append((backup_path, timestamp))
            
            # Sort by timestamp (newest first) and keep only the most recent ones
            backup_files.sort(key=lambda x: x[1], reverse=True)
            max_backups = 5
            
            # Delete older backups
            for backup_file, _ in backup_files[max_backups:]:
                try:
                    os.remove(backup_file)
                except Exception as e:
                    logger.error(f"Failed to delete old backup {backup_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to cleanup old backups: {e}")
        
    def _validate_path(self, path: str) -> Tuple[bool, Optional[str]]:
        """Validate file path.
        
        Args:
            path: Path to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for path traversal attempts
            if ".." in path or "~" in path:
                return False, "Invalid or inaccessible file"
            
            # Check for command injection attempts
            if any(char in path for char in ";|&"):
                return False, "Invalid or inaccessible file"
            
            if not os.path.exists(path):
                return False, "Invalid or inaccessible file"
            
            if not os.path.isfile(path):
                return False, "Invalid or inaccessible file"
            
            if not os.access(path, os.R_OK):
                return False, "Invalid or inaccessible file"
            
            if not os.access(path, os.W_OK):
                return False, "Invalid or inaccessible file"
            
            return True, None
        except Exception:
            return False, "Invalid or inaccessible file"
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute file edit operation."""
        try:
            path = context.get("path")
            operation = context.get("operation", {})
            
            if not path or not operation:
                return InstructionResult(
                    status=InstructionStatus.FAILED.value,
                    data={"error_message": "Path and operation must be provided"},
                    metrics={}
                )
            
            # Validate file exists and is accessible
            is_valid, error_message = self._validate_path(path)
            if not is_valid:
                return InstructionResult(
                    status=InstructionStatus.FAILED.value,
                    data={"error_message": error_message},
                    metrics={}
                )
            
            # Execute edit operation
            result = await self._execute_impl(context)
            
            # Create backup if needed
            if context.get("create_backup", True):
                backup_path = self._create_backup(path)
                if not backup_path:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={"error_message": "Failed to create backup"},
                        metrics={}
                    )
            
            return result
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED.value,
                data={"error_message": str(e)},
                metrics={}
            )

    def _format_diff(self, old_lines: List[str], new_lines: List[str], diff_info: DiffInfo) -> List[str]:
        """Format a diff between old and new lines.
        
        Args:
            old_lines: Original lines
            new_lines: Modified lines
            diff_info: Information about the diff
            
        Returns:
            List[str]: Formatted diff lines
        """
        # Convert lines to strings for difflib
        old_str = [str(line) for line in old_lines]
        new_str = [str(line) for line in new_lines]
        
        # Get the diff
        diff = list(difflib.unified_diff(
            old_str,
            new_str,
            fromfile="before",
            tofile="after",
            lineterm=""
        ))
        
        # Add operation info
        header = [
            f"{diff_info.operation} at line {diff_info.position + 1}:",
            "---"
        ]
        
        return header + diff

    def _diff_info_to_str(self, diff_info: DiffInfo) -> List[str]:
        """Convert DiffInfo to a list of strings.
        
        Args:
            diff_info: DiffInfo object to convert
            
        Returns:
            List[str]: String representation of the diff info
        """
        return list(diff_info.to_sequence())

    def _update_metrics(self, metrics: Dict[str, int]) -> FileChangeMetrics:
        """Update metrics for file changes."""
        return FileChangeMetrics(
            lines_changed=metrics.get("lines_changed", 0),
            chars_changed=metrics.get("chars_changed", 0),
            timestamp=time.time() * 1000000,  # Convert to microseconds
            matches_found=metrics.get("matches_found")
        )