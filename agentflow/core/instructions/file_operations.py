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
        # Editor state shared by the line-editing command handlers
        self._clipboard: List[str] = []
        self._undo_stack: List[DiffInfo] = []
        self._max_undo: int = 100
        self._indent_size: int = 4
        self._max_line_length: int = 80
        
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
            elif operation_type == "insert_text":
                # Insert ``text`` before line ``position`` (1-based).
                # A trailing newline is dropped so the payload inserts
                # as complete line(s) without adding a blank line.
                position = operation.get("position", 1)
                text = operation.get("text", "")
                if text.endswith("\n"):
                    text = text[:-1]
                insert_lines = text.split("\n") if text else []
                if 1 <= position <= len(lines) + 1:
                    lines[position - 1:position - 1] = insert_lines
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line position: {position}"}
                    )
            elif operation_type == "delete_lines":
                # Delete the inclusive 0-based index range [start, end].
                start = operation.get("start", 0)
                end = operation.get("end", start)
                if 0 <= start <= end < len(lines):
                    del lines[start:end + 1]
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line range: {start}-{end}"}
                    )
            elif operation_type == "change_case":
                # Change case over the inclusive 0-based index range.
                start = operation.get("start", 0)
                end = operation.get("end", start)
                mode = operation.get("mode", "upper")
                case_fns = {
                    "upper": str.upper,
                    "lower": str.lower,
                    "title": str.title,
                    "capitalize": str.capitalize,
                }
                if mode not in case_fns:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Unknown case mode: {mode}"}
                    )
                if 0 <= start <= end < len(lines):
                    for i in range(start, end + 1):
                        lines[i] = case_fns[mode](lines[i])
                else:
                    return InstructionResult(
                        status=InstructionStatus.FAILED.value,
                        data={},
                        metrics={"error_message": f"Invalid line range: {start}-{end}"}
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

    # ------------------------------------------------------------------
    # Vim-style line-editing command handlers
    # ------------------------------------------------------------------

    @staticmethod
    def _command_metrics(lines_changed: int, chars_changed: int) -> Dict[str, Any]:
        """Build the standard metrics payload for a line command."""
        return {
            "lines_changed": lines_changed,
            "chars_changed": chars_changed,
            "timestamp": time.time()
        }

    def _push_undo(self, operation: str, position: int, content: List[str]) -> None:
        """Record an undo snapshot for a line command."""
        self._undo_stack.append(DiffInfo(
            operation=operation,
            position=position,
            content=list(content)
        ))
        if len(self._undo_stack) > self._max_undo:
            self._undo_stack.pop(0)

    async def _yank_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Yank (copy) lines into the clipboard (y).

        Args:
            lines: Current file lines
            args: Command arguments:
                - start: Starting line number (1-indexed)
                - end: Ending line number (1-indexed, exclusive)

        Returns:
            InstructionResult with the yanked content
        """
        try:
            start = args.get("start")
            if start is None:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Start line number must be provided"
                )
            end = args.get("end")
            if end is None:
                end = start + 1

            if not isinstance(start, int) or not isinstance(end, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Line numbers must be integers"
                )

            start_idx = start - 1
            end_idx = end - 1  # end is exclusive

            if not (0 <= start_idx < len(lines) and 0 <= end_idx <= len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Line numbers out of range (1-{len(lines)})"
                )

            yanked = lines[start_idx:max(start_idx, end_idx)]
            self._clipboard = yanked.copy()

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "yanked": yanked,
                    "changes": [f"Yanked {len(yanked)} lines"],
                    "metrics": self._command_metrics(0, 0)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _paste_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Paste clipboard contents after a line (p).

        Args:
            lines: Current file lines
            args: Command arguments:
                - line: Line number (1-indexed) after which to paste

        Returns:
            InstructionResult with modified lines
        """
        try:
            line = args.get("line")
            if not isinstance(line, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Line number must be an integer"
                )

            if not self._clipboard:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Clipboard is empty"
                )

            line_idx = line - 1
            if not (0 <= line_idx < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Line number out of range (1-{len(lines)})"
                )

            pasted = [l if l.endswith("\n") else l + "\n" for l in self._clipboard]
            new_lines = lines[:line_idx + 1] + pasted + lines[line_idx + 1:]

            self._push_undo("paste", line_idx + 1, pasted)

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Pasted {len(pasted)} lines after line {line}"],
                    "metrics": self._command_metrics(len(pasted), sum(len(l) for l in pasted))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _format_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Format plain text (format).

        Strips trailing whitespace and normalises line endings.

        Args:
            lines: Current file lines
            args: Command arguments:
                - style: Formatting style ("text", "plain" or "code")

        Returns:
            InstructionResult with formatted lines
        """
        try:
            style = args.get("style", "text")
            if style not in ("text", "plain", "code"):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Unsupported format style: {style}"
                )

            formatted = []
            for line in lines:
                stripped = line.rstrip("\n").rstrip()
                formatted.append(stripped + "\n" if stripped else "\n")

            chars_changed = sum(
                1 for old, new in zip(lines, formatted) if old != new
            )

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": formatted,
                    "changes": [f"Formatted text with style '{style}'"],
                    "metrics": self._command_metrics(len(formatted), chars_changed)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _auto_indent(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Auto-indent source lines (indent).

        Args:
            lines: Current file lines
            args: Command arguments:
                - language: Source language (currently "python")

        Returns:
            InstructionResult with re-indented lines
        """
        try:
            language = args.get("language", "python")
            indent_size = getattr(self, "_indent_size", 4)

            if language != "python":
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Unsupported language for auto-indent: {language}"
                )

            dedent_tokens = (
                "return", "pass", "raise", "break", "continue",
                "elif", "else", "except", "finally"
            )

            level = 0
            result_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    result_lines.append("\n")
                    continue

                first_token = stripped.split()[0].rstrip(":")
                if first_token in dedent_tokens and level > 0:
                    level -= 1

                result_lines.append(" " * (level * indent_size) + stripped + "\n")

                if stripped.endswith(":"):
                    level += 1

            chars_changed = sum(
                1 for old, new in zip(lines, result_lines) if old != new
            )

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": result_lines,
                    "changes": [f"Auto-indented as {language}"],
                    "metrics": self._command_metrics(len(result_lines), chars_changed)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _align_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Align delimited text into columns (align).

        Args:
            lines: Current file lines
            args: Command arguments:
                - delimiter: Column delimiter (defaults to whitespace)

        Returns:
            InstructionResult with aligned lines
        """
        try:
            delimiter = args.get("delimiter")

            if not delimiter:
                return InstructionResult(
                    status=InstructionStatus.COMPLETED,
                    data={
                        "lines": lines,
                        "changes": ["No delimiter specified; lines unchanged"],
                        "metrics": self._command_metrics(0, 0)
                    }
                )

            rows = [
                [seg.strip() for seg in line.rstrip("\n").split(delimiter)]
                for line in lines
            ]
            if not rows:
                return InstructionResult(
                    status=InstructionStatus.COMPLETED,
                    data={"lines": lines, "changes": ["Aligned text"], "metrics": self._command_metrics(0, 0)}
                )

            max_cols = max(len(r) for r in rows)
            widths = [
                max(len(r[i]) for r in rows if len(r) > i)
                for i in range(max_cols)
            ]

            aligned = []
            for row in rows:
                padded = [row[i].ljust(widths[i]) if i < len(row) else "" for i in range(max_cols)]
                aligned.append(delimiter.join(padded).rstrip() + "\n")

            chars_changed = sum(
                1 for old, new in zip(lines, aligned) if old != new
            )

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": aligned,
                    "changes": [f"Aligned text on delimiter '{delimiter}'"],
                    "metrics": self._command_metrics(len(aligned), chars_changed)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _wrap_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Wrap long lines to a maximum width (wrap).

        Args:
            lines: Current file lines
            args: Command arguments:
                - width: Maximum line width

        Returns:
            InstructionResult with wrapped lines
        """
        try:
            width = args.get("width", getattr(self, "_max_line_length", 80))
            if not isinstance(width, int) or width <= 0:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Width must be a positive integer"
                )

            wrapped = []
            for line in lines:
                text = line.rstrip("\n")
                if not text:
                    wrapped.append("\n")
                    continue
                for segment in textwrap.wrap(text, width=width) or [""]:
                    wrapped.append(segment + "\n")

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": wrapped,
                    "changes": [f"Wrapped text at width {width}"],
                    "metrics": self._command_metrics(len(wrapped), sum(len(l) for l in wrapped))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _sort_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Sort lines (sort).

        Args:
            lines: Current file lines
            args: Command arguments:
                - reverse: Sort in descending order
                - ignore_case: Ignore case while sorting

        Returns:
            InstructionResult with sorted lines
        """
        try:
            reverse = args.get("reverse", False)
            ignore_case = args.get("ignore_case", False)

            key = (lambda l: l.lower()) if ignore_case else (lambda l: l)
            new_lines = sorted(lines, key=key, reverse=reverse)

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Sorted {len(new_lines)} lines"],
                    "metrics": self._command_metrics(len(new_lines), sum(len(l) for l in new_lines))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _join_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Join a range of lines with a separator (join).

        Args:
            lines: Current file lines
            args: Command arguments:
                - start: Starting line number (1-indexed, inclusive)
                - end: Ending line number (1-indexed, inclusive)
                - separator: Text placed between joined lines

        Returns:
            InstructionResult with joined lines
        """
        try:
            start = args.get("start")
            end = args.get("end")
            separator = args.get("separator", " ")

            if not isinstance(start, int) or not isinstance(end, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Line numbers must be integers"
                )

            if not lines:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="File is empty"
                )

            start_idx = start - 1
            end_idx = end  # inclusive end -> exclusive slice bound

            if not (0 <= start_idx < len(lines) and start_idx < end_idx <= len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Line numbers out of range (1-{len(lines)})"
                )

            joined = separator.join(l.rstrip("\n") for l in lines[start_idx:end_idx]) + "\n"
            new_lines = lines[:start_idx] + [joined] + lines[end_idx:]

            self._push_undo("join", start_idx, lines[start_idx:end_idx])

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Joined lines {start}-{end} with '{separator}'"],
                    "metrics": self._command_metrics(1, len(joined))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _block_edit(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Apply a block operation across a line range (block).

        Args:
            lines: Current file lines
            args: Command arguments:
                - start: Starting line number (1-indexed, inclusive)
                - end: Ending line number (1-indexed, inclusive)
                - col_start: Starting column (1-indexed)
                - action: "insert" or "delete"
                - text: Text to insert (or whose length defines the delete span)
                - count: Number of characters to delete (defaults to len(text))

        Returns:
            InstructionResult with modified lines
        """
        try:
            start = args.get("start")
            end = args.get("end")
            col_start = args.get("col_start", 1)
            action = args.get("action", "insert")
            text = args.get("text", "")

            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(col_start, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="Line and column numbers must be integers"
                )

            if not lines:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error="File is empty"
                )

            start_idx = start - 1
            end_idx = end - 1  # inclusive

            if not (0 <= start_idx <= end_idx < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Line numbers out of range (1-{len(lines)})"
                )

            col = col_start - 1
            if col < 0:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Column out of range (1-{len(lines[start_idx]) + 1})"
                )

            new_lines = []
            for i, line in enumerate(lines):
                if start_idx <= i <= end_idx:
                    body = line.rstrip("\n")
                    if col > len(body):
                        return InstructionResult(
                            status=InstructionStatus.FAILED,
                            data={},
                            error=f"Column out of range (1-{len(body) + 1})"
                        )
                    if action == "insert":
                        body = body[:col] + text + body[col:]
                    elif action == "delete":
                        count = args.get("count", len(text) if text else 1)
                        if not isinstance(count, int) or count < 0:
                            return InstructionResult(
                                status=InstructionStatus.FAILED,
                                data={},
                                error="Count must be a non-negative integer"
                            )
                        body = body[:col] + body[col + count:]
                    else:
                        return InstructionResult(
                            status=InstructionStatus.FAILED,
                            data={},
                            error=f"Unsupported block action: {action}"
                        )
                    new_lines.append(body + "\n")
                else:
                    new_lines.append(line)

            self._push_undo("block", start_idx, lines[start_idx:end_idx + 1])

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Block {action} on lines {start}-{end} at column {col_start}"],
                    "metrics": self._command_metrics(end_idx - start_idx + 1, len(text))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _change_case(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Change the case of every line (case).

        Args:
            lines: Current file lines
            args: Command arguments:
                - case: One of "upper", "lower", "title", "capitalize", "swap"

        Returns:
            InstructionResult with re-cased lines
        """
        try:
            case = args.get("case", "lower")
            transforms = {
                "upper": str.upper,
                "lower": str.lower,
                "title": str.title,
                "capitalize": str.capitalize,
                "swap": str.swapcase
            }

            if case not in transforms:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Unsupported case: {case}"
                )

            transform = transforms[case]
            new_lines = [transform(line) for line in lines]

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Changed case to {case}"],
                    "metrics": self._command_metrics(len(new_lines), sum(len(l) for l in new_lines))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _trim_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Trim whitespace from lines (trim).

        Args:
            lines: Current file lines
            args: Command arguments:
                - mode: "left", "right", "both" or "none"

        Returns:
            InstructionResult with trimmed lines
        """
        try:
            mode = args.get("mode", "both")
            if mode not in ("left", "right", "both", "none"):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    data={},
                    error=f"Unsupported trim mode: {mode}"
                )

            new_lines = []
            for line in lines:
                if line.endswith("\n"):
                    body, newline = line[:-1], "\n"
                else:
                    body, newline = line, ""
                if mode == "left":
                    body = body.lstrip()
                elif mode == "right":
                    body = body.rstrip()
                elif mode == "both":
                    body = body.strip()
                new_lines.append(body + newline)

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Trimmed whitespace ({mode})"],
                    "metrics": self._command_metrics(len(new_lines), sum(len(l) for l in new_lines))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _show_diff(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Show a unified diff of the lines (diff).

        Args:
            lines: Current file lines
            args: Command arguments:
                - new_lines: Optional replacement lines to diff against

        Returns:
            InstructionResult containing the diff
        """
        try:
            new_lines = args.get("new_lines")
            if new_lines is None:
                # Fall back to the most recent undo snapshot
                new_lines = self._undo_stack[-1].content if self._undo_stack else lines

            diff = list(difflib.unified_diff(
                [l.rstrip("\n") for l in lines],
                [str(l).rstrip("\n") for l in new_lines],
                fromfile="before",
                tofile="after",
                lineterm=""
            ))

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "diff": diff,
                    "changes": [f"Generated diff ({len(diff)} lines)"],
                    "metrics": self._command_metrics(0, 0)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    @staticmethod
    def _syllable_count(word: str) -> int:
        """Estimate the number of syllables in a word.

        Uses a simple vowel-group heuristic with silent-'e' handling.
        """
        word = re.sub(r"[^a-zA-Z]", "", word).lower()
        if not word:
            return 0

        vowels = "aeiouy"
        count = 0
        prev_was_vowel = False
        for ch in word:
            is_vowel = ch in vowels
            if is_vowel and not prev_was_vowel:
                count += 1
            prev_was_vowel = is_vowel

        if word.endswith("e") and count > 1:
            count -= 1

        return max(count, 1)

    async def _count_syllables(
        self,
        lines_or_word: Any,
        args: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Count syllables.

        Supports two calling conventions:

        * word mode: ``await self._count_syllables("hello")`` returns an int
        * command mode: ``await self._count_syllables(lines, {"word": ...})``
          returns an InstructionResult
        """
        try:
            if isinstance(lines_or_word, str) and (args is None or "word" in (args or {})):
                return self._syllable_count(lines_or_word)

            if isinstance(lines_or_word, str):
                count = self._syllable_count(lines_or_word)
                return InstructionResult(
                    status=InstructionStatus.COMPLETED,
                    data={
                        "lines": [],
                        "syllables": count,
                        "changes": [f"Counted {count} syllables"],
                        "metrics": self._command_metrics(0, 0)
                    }
                )

            # Command mode: operate on the lines of the file
            lines = lines_or_word
            cmd_args = args or {}
            word = cmd_args.get("word")

            if word is not None:
                count = self._syllable_count(str(word))
            else:
                count = sum(
                    self._syllable_count(w)
                    for line in lines
                    for w in line.split()
                )

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "syllables": count,
                    "changes": [f"Counted {count} syllables"],
                    "metrics": self._command_metrics(0, 0)
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))

    async def _format_code(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Format source code (format_code).

        Combines trailing-whitespace cleanup with auto-indentation.

        Args:
            lines: Current file lines
            args: Command arguments:
                - language: Source language (currently "python")

        Returns:
            InstructionResult with formatted code
        """
        try:
            language = args.get("language", "python")

            indent_result = await self._auto_indent(lines, {"language": language})
            if indent_result.status != InstructionStatus.COMPLETED:
                return indent_result

            formatted = []
            for line in indent_result.data["lines"]:
                body = line.rstrip("\n").rstrip()
                formatted.append(body + "\n" if body else "\n")

            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": formatted,
                    "changes": [f"Formatted code as {language}"],
                    "metrics": self._command_metrics(len(formatted), sum(len(l) for l in formatted))
                }
            )
        except Exception as e:
            return InstructionResult(status=InstructionStatus.FAILED, data={}, error=str(e))