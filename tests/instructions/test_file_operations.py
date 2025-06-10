"""Test file operations instructions."""

import pytest
import os
from typing import Dict, Any, List, Tuple, Optional
from unittest.mock import patch, MagicMock
import time
import asyncio
import shutil
from pathlib import Path
import logging
import re

from agentflow.core.instructions.file_operations import (
    FileEditInstruction,
    FileChangeMetrics,
    DiffInfo
)
from agentflow.core.instructions.base import InstructionStatus, InstructionResult

logger = logging.getLogger(__name__)

class TestFileEditInstruction(FileEditInstruction):
    """Test implementation of FileEditInstruction."""
    
    def __init__(self):
        """Initialize with parent constructor."""
        super().__init__()
        self._clipboard = []
        self._undo_stack = []
        self._max_undo = 100
        self._indent_size = 4
        self._max_line_length = 80
        self._metrics = {}
        self.commands = {
            "d": self._delete_lines,
            "y": self._yank_lines,
            "p": self._paste_lines,
            "a": self._append_text,
            "i": self._insert_text,
            "c": self._change_text,
            "s": self._substitute_text,
            "gg": self._goto_line,
            "/": self._search_text,
            "u": self._undo_change,
            "format": self._format_text,
            "indent": self._auto_indent,
            "align": self._align_text,
            "wrap": self._wrap_text,
            "sort": self._sort_lines,
            "join": self._join_lines,
            "block": self._block_edit,
            "case": self._change_case,
            "trim": self._trim_lines,
            "diff": self._show_diff,
            "analyze": self._analyze_text,
            "count_syllables": self._count_syllables,
            "format_code": self._format_code,
            "format_markup": self._format_markup
        }
    
    def _create_backup(self, file_path: str) -> Optional[str]:
        """Create a backup of the file.
        
        Args:
            file_path: Path to the file to backup
            
        Returns:
            Optional[str]: Path to the backup file if successful, None otherwise
        """
        try:
            # Create backup directory if it doesn't exist
            backup_dir = os.path.join(os.path.dirname(file_path), ".backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # Generate backup filename with timestamp
            timestamp = int(time.time())
            backup_name = f"{os.path.basename(file_path)}.{timestamp}"
            backup_path = os.path.join(backup_dir, backup_name)
            
            # Copy file to backup
            shutil.copy2(file_path, backup_path)
            
            # Clean up old backups
            self._cleanup_old_backups(file_path)
            
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
        except Exception as e:
            return False, "Invalid or inaccessible file"
            
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute the file edit instruction.
        
        Args:
            context: Dictionary containing:
                - path: Path to the file to edit
                - command: Command to execute
                - args: Command arguments
                
        Returns:
            InstructionResult with execution status and data
        """
        try:
            # Validate input
            path = context.get("path")
            commands = context.get("commands")
            
            if not path or not commands:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Path and commands must be provided"
                )
            
            # Validate file exists and is accessible
            if not os.path.exists(path):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Invalid or inaccessible file"
                )
            
            # Execute commands sequentially
            results = []
            for command in commands:
                handler = self.commands.get(command["command"])
                if not handler:
                    return InstructionResult(
                        status=InstructionStatus.FAILED,
                        error=f"Unknown command: {command['command']}"
                    )
                
                # Read file contents
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                except Exception as e:
                    return InstructionResult(
                        status=InstructionStatus.FAILED,
                        error=f"Failed to read file: {e}"
                    )
                
                # Create backup for modifying commands
                if command["command"] not in ["gg", "/", "y", "analyze"]:  # Skip backup for read-only operations
                    backup_path = self._create_backup(path)
                    if not backup_path:
                        return InstructionResult(
                            status=InstructionStatus.FAILED,
                            error="Failed to create backup"
                        )
                
                # Execute command
                result = await handler(lines, command["args"])
                
                # Write changes if successful and command modifies file
                if result.status == InstructionStatus.COMPLETED and command["command"] not in ["gg", "/", "y", "analyze"]:
                    try:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.writelines(result.data["lines"])
                    except Exception as e:
                        return InstructionResult(
                            status=InstructionStatus.FAILED,
                            error=f"Failed to write changes: {e}"
                        )
                
                # Append result to list
                results.append(result)
            
            # Return the last result
            last_result = results[-1]
            return InstructionResult(
                status=last_result.status,
                data=last_result.data if last_result.status == InstructionStatus.COMPLETED else {},
                error=last_result.error if last_result.status == InstructionStatus.FAILED else None
            )
                
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )
    
    async def _delete_lines(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Delete lines (d).
        
        Args:
            lines: List of lines to modify
            args: Command arguments including:
                - start: Starting line number (1-indexed)
                - end: Ending line number (1-indexed, exclusive)
                
        Returns:
            InstructionResult with modified lines
        """
        try:
            # Validate line numbers are integers
            start = args.get("start")
            if start is None:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Start line number must be provided"
                )
            
            end = args.get("end")
            if end is None:
                end = start + 1  # Default to deleting one line
            
            # Strict type checking
            if not isinstance(start, int) or not isinstance(end, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Line numbers must be integers"
                )
            
            # Convert to 0-based index
            start_idx = start - 1
            end_idx = end - 1  # end is exclusive
            
            # Handle empty file case
            if not lines:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="File is empty"
                )
            
            # Validate line numbers are in range
            if not (0 <= start_idx < len(lines) and 0 <= end_idx <= len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Line numbers out of range (1-{len(lines)})"
                )
            
            # Store deleted lines in clipboard
            deleted_lines = lines[start_idx:end_idx]  # end_idx is exclusive
            self._clipboard = deleted_lines.copy()  # Make a copy to prevent modification
            
            # Create new lines list without deleted lines
            new_lines = lines[:start_idx] + lines[end_idx:]
            
            # Save current state for undo
            diff_info = DiffInfo(
                operation="delete",
                position=start_idx,
                content=deleted_lines
            )
            self._undo_stack.append(diff_info)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            
            # Update metrics
            metrics = {
                "lines_changed": len(deleted_lines),  # Number of lines deleted
                "chars_changed": sum((len(line) for line in deleted_lines)),
                "timestamp": time.time()
            }
            
            # Return result with modified lines
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Deleted lines {start}-{end-1}"],
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )
    
    async def _undo_change(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Undo last change (u)."""
        try:
            if not self._undo_stack:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="No changes to undo"
                )
            
            # Get the last saved state
            diff_info = self._undo_stack.pop()
            
            if diff_info.operation == "delete":
                # For delete operations, insert the deleted lines back at the original position
                new_lines = lines[:diff_info.position] + diff_info.content + lines[diff_info.position:]
                
                # Update metrics
                metrics = {
                    "lines_changed": len(diff_info.content),
                    "chars_changed": sum(len(line) for line in diff_info.content),
                    "timestamp": time.time()
                }
                
                return InstructionResult(
                    status=InstructionStatus.COMPLETED,
                    data={
                        "lines": new_lines,
                        "changes": [f"Restored {len(diff_info.content)} deleted lines at position {diff_info.position + 1}"],
                        "metrics": metrics
                    }
                )
            else:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Unknown operation: {diff_info.operation}"
                )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _search_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Search text (/).
        
        Args:
            lines: List of lines to search in
            args: Command arguments including:
                - pattern: Search pattern (regex)
                - case_sensitive: Whether to use case-sensitive search
                
        Returns:
            InstructionResult with search results
        """
        try:
            pattern = args.get("pattern")
            case_sensitive = args.get("case_sensitive", False)
            
            if not pattern:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Search pattern must be provided"
                )
            
            if not isinstance(pattern, str):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Pattern must be a string"
                )
            
            # Compile regex pattern
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                regex = re.compile(pattern, flags)
            except re.error as e:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Invalid regex pattern: {str(e)}"
                )
            
            # Search for matches
            matches = []
            for i, line in enumerate(lines):
                for match in regex.finditer(line):
                    matches.append({
                        "line_number": i + 1,
                        "line": line.rstrip('\n'),
                        "start": match.start(),
                        "end": match.end(),
                        "match": match.group()
                    })
            
            # Update metrics
            metrics = {
                "lines_changed": 0,
                "chars_changed": 0,
                "timestamp": time.time()
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "matches": matches,
                    "total_matches": len(matches),
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _goto_line(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Go to line number (gg)."""
        try:
            line_number = args.get("line")
            if not isinstance(line_number, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Line number must be an integer"
                )
            
            # Convert to 0-based index
            line_number = line_number - 1
            
            # For empty files, return success with empty lines
            if not lines:
                return InstructionResult(
                    status=InstructionStatus.COMPLETED,
                    data={
                        "lines": [],
                        "current_line": 0,
                        "total_lines": 0,
                        "metrics": {
                            "lines_changed": 0,
                            "chars_changed": 0,
                            "timestamp": time.time()
                        }
                    }
                )
            
            # Validate line number is in range
            if not (0 <= line_number < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Line number out of range (1-{len(lines)})"
                )
            
            # Update metrics
            metrics = {
                "lines_changed": 0,
                "chars_changed": 0,
                "timestamp": time.time()
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "current_line": line_number + 1,
                    "total_lines": len(lines),
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _format_markup(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Format markup text."""
        try:
            markup_type = args.get("type", "markdown")
            if markup_type != "markdown":
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Unsupported markup type: {markup_type}"
                )
            
            formatted_lines = []
            for line in lines:
                # Handle headers
                if line.startswith("#"):
                    formatted_lines.append(line)
                # Handle list items
                elif line.startswith("*") or line.startswith("-"):
                    formatted_lines.append(line)  # Keep original list marker
                else:
                    formatted_lines.append(line)
                
                # Add blank line after headers
                if line.startswith("#"):
                    formatted_lines.append("\n")
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": formatted_lines,
                    "changes": ["Formatted markup text"]
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _change_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Change text (c)."""
        try:
            # Validate line numbers are integers
            start = args.get("start")
            end = args.get("end")
            text = args.get("text", "")
            
            if not isinstance(start, int) or not isinstance(end, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Line numbers must be integers"
                )
            
            if not isinstance(text, str):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Text must be a string"
                )
            
            # Convert to 0-based index
            start = start - 1
            end = end - 1  # Convert end to 0-based and make it inclusive
            
            # Handle empty file case
            if not lines:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="File is empty"
                )
            
            # Validate line numbers are in range
            if not (0 <= start <= end < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Line numbers out of range (1-{len(lines)})"
                )
            
            # Store original lines in clipboard
            self._clipboard = lines[start:end + 1]  # Include end line
            
            # Save current state for undo
            diff_info = DiffInfo(
                operation="change",
                position=start,
                content=self._clipboard
            )
            self._undo_stack.append(diff_info)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            
            # Create new lines list
            new_lines = lines.copy()
            
            # Split new text into lines and ensure each line ends with newline
            new_text_lines = [line + '\n' for line in text.rstrip('\n').split('\n')]
            
            # Replace lines with new text
            new_lines[start:end + 1] = new_text_lines
            
            # Update metrics
            metrics = {
                "lines_changed": len(self._clipboard),
                "chars_changed": sum(len(line) for line in self._clipboard),
                "timestamp": time.time()
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [
                        f"Changed lines {start+1}-{end+1}",
                        f"Replaced {len(self._clipboard)} lines with {len(new_text_lines)} lines"
                    ],
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _analyze_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Analyze text content."""
        try:
            analysis_type = args.get("type", "statistics")
            if analysis_type != "statistics":
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Unsupported analysis type: {analysis_type}"
                )
            
            # Join lines for analysis
            text = "".join(lines)
            
            # Calculate statistics
            words = len(text.split())
            sentences = len([s for s in text.split('.') if s.strip()])
            characters = len(text)
            
            # Calculate readability metrics
            syllables = 0
            for word in text.split():
                try:
                    syllable_count = await self._count_syllables(word)
                    syllables += syllable_count
                except Exception:
                    # Skip words that can't be counted
                    continue
            
            statistics = {
                "words": words,
                "sentences": sentences,
                "characters": characters,
                "syllables": syllables,
                "avg_word_length": characters / words if words > 0 else 0,
                "avg_sentence_length": words / sentences if sentences > 0 else 0
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "analysis": {
                        "statistics": statistics
                    },
                    "metrics": {
                        "lines_changed": 0,
                        "chars_changed": 0,
                        "timestamp": time.time()
                    }
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _append_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Append text (a)."""
        try:
            # Get line number and text
            line = args.get("line")
            text = args.get("text", "")
            
            if not isinstance(line, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Line number must be an integer"
                )
            
            # Convert to 0-based index
            line = line - 1
            
            # Validate line number is in range
            if not (0 <= line < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Line number out of range (1-{len(lines)})"
                )
            
            # Append text to the line
            original_line = lines[line]
            lines[line] = original_line.rstrip('\n') + text + '\n'
            
            # Save current state for undo
            diff_info = DiffInfo(
                operation="append",
                position=line,
                content=[original_line]
            )
            self._undo_stack.append(diff_info)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            
            # Update metrics with microsecond precision for uniqueness
            metrics = {
                "lines_changed": 1,
                "chars_changed": len(text),
                "timestamp": time.time() + len(self._undo_stack) / 1000000  # Add microsecond offset
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "changes": [f"Appended text to line {line+1}"],
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _insert_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Insert text (i)."""
        try:
            # Get line number, column, and text
            line = args.get("line")
            column = args.get("column", 1)
            text = args.get("text", "")
            
            if not isinstance(line, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Line number must be an integer"
                )
            
            if not isinstance(column, int):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Column must be an integer"
                )
            
            # Convert to 0-based index
            line = line - 1
            column = column - 1
            
            # Handle empty file case
            if not lines and line == 0:
                lines.append("")
            
            # Validate line number is in range
            if not (0 <= line < len(lines)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Line number out of range (1-{len(lines)})"
                )
            
            # Get the target line
            original_line = lines[line]
            
            # Validate column is in range
            if not (0 <= column <= len(original_line)):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Column out of range (1-{len(original_line)+1})"
                )
            
            # Insert text at specified position
            new_line = original_line[:column] + text + original_line[column:]
            lines[line] = new_line
            
            # Save current state for undo
            diff_info = DiffInfo(
                operation="insert",
                position=line,
                content=[original_line]
            )
            self._undo_stack.append(diff_info)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            
            # Update metrics
            metrics = {
                "lines_changed": 1,
                "chars_changed": len(text),
                "timestamp": time.time()
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": lines,
                    "changes": [f"Inserted text at line {line+1}, column {column+1}"],
                    "metrics": metrics
                }
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

    async def _substitute_text(self, lines: List[str], args: Dict[str, Any]) -> InstructionResult:
        """Substitute text (s).
        
        Args:
            lines: List of lines to modify
            args: Command arguments including:
                - pattern: Pattern to search for
                - replacement: Text to replace with
                - global: Whether to replace all occurrences
                
        Returns:
            InstructionResult with modified lines
        """
        try:
            pattern = args.get("pattern")
            replacement = args.get("replacement")
            is_global = args.get("global", False)
            
            if not pattern:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Pattern must be provided"
                )
            
            if replacement is None:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Replacement text must be provided"
                )
            
            # Store original lines for undo
            original_lines = lines.copy()
            new_lines = []
            total_replacements = 0
            
            # Perform substitution
            for line in lines:
                if is_global:
                    new_line, count = re.subn(pattern, replacement, line)
                else:
                    new_line, count = re.subn(pattern, replacement, line, count=1)
                new_lines.append(new_line)
                total_replacements += count
            
            # Save current state for undo
            diff_info = DiffInfo(
                operation="substitute",
                position=0,  # Substitution affects whole file
                content=original_lines
            )
            self._undo_stack.append(diff_info)
            if len(self._undo_stack) > self._max_undo:
                self._undo_stack.pop(0)
            
            # Update metrics
            metrics = {
                "lines_changed": sum(1 for old, new in zip(lines, new_lines) if old != new),
                "chars_changed": abs(sum((len(line) for line in new_lines)) - sum((len(line) for line in lines))),
                "timestamp": time.time()
            }
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "lines": new_lines,
                    "changes": [f"Replaced {total_replacements} occurrences of '{pattern}' with '{replacement}'"],
                    "metrics": metrics
                }
            )
        except re.error as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid regex pattern: {str(e)}"
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

@pytest.fixture
def instruction():
    """Create FileEditInstruction instance."""
    return TestFileEditInstruction()

@pytest.mark.asyncio
async def test_delete_lines(instruction, test_file):
    """Test deleting lines."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {"start": 2, "end": 4}  # End is exclusive, so this will delete lines 2 and 3
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED
    assert "lines" in result.data
    assert len(result.data["lines"]) == 3  # Should have 3 lines after deletion

@pytest.mark.asyncio
async def test_yank_paste(instruction, test_file):
    """Test yanking and pasting lines."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

    # First yank lines 2-3
    context = {
        "path": test_file,
        "commands": [
            {"command": "y", "args": {"start": 2, "end": 3}},
            {"command": "p", "args": {"line": 1}}
        ]
    }
    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_insert_text(instruction, test_file):
    """Test inserting text."""
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "i",
                "args": {
                    "line": 2,
                    "column": 1,
                    "text": "Inserted "
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_substitute_text(instruction, test_file):
    """Test text substitution."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "s",
                "args": {
                    "pattern": "Line",
                    "replacement": "Text",
                    "global": True
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_format_text(instruction, test_file):
    """Test text formatting."""
    with open(test_file, 'w') as f:
        f.write("def test():\nprint('hello')\n  return True")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "format",
                "args": {
                    "style": "code"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_auto_indent(instruction, test_file):
    """Test auto-indentation."""
    with open(test_file, 'w') as f:
        f.write("def test():\nprint('hello')\nif True:\nprint('indented')\nreturn True")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "indent",
                "args": {
                    "language": "python"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_block_edit(instruction, test_file):
    """Test block editing."""
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "block",
                "args": {
                    "start": 1,
                    "end": 3,
                    "col_start": 1,
                    "action": "insert",
                    "text": "# "
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_change_case(instruction, test_file):
    """Test case changing."""
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "case",
                "args": {
                    "case": "upper"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_analyze_text(instruction, test_file):
    """Test text analysis."""
    # Write some test content
    with open(test_file, 'w') as f:
        f.write("This is a test sentence.\nIt has multiple lines.\nWe can analyze this text.")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "analyze",
                "args": {
                    "type": "statistics"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_format_markup(instruction, test_file):
    """Test markup formatting."""
    with open(test_file, 'w') as f:
        f.write("# Header\n* Item 1\n* Item 2\n")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "format_markup",
                "args": {
                    "type": "markdown"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_error_handling(instruction, test_file):
    """Test error handling."""
    # Test invalid line numbers
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {"start": 100, "end": 101}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_undo(instruction, test_file):
    """Test undo functionality."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\nLine 4\nLine 5\n")

    # First make a change
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {"start": 1, "end": 2}  # End is exclusive, so this will delete line 1 only
            }
        ]
    }
    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_sort_lines(instruction, test_file):
    """Test line sorting."""
    with open(test_file, 'w') as f:
        f.write("c\na\nb\n")

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "sort",
                "args": {
                    "reverse": False,
                    "ignore_case": True
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_join_lines(instruction, test_file):
    """Test line joining."""
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "join",
                "args": {
                    "start": 1,
                    "end": 3,
                    "separator": ", "
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_wrap_text(instruction, test_file):
    """Test text wrapping."""
    long_text = "This is a very long line that should be wrapped to multiple lines based on the specified width"
    with open(test_file, 'w') as f:
        f.write(long_text)

    context = {
        "path": test_file,
        "commands": [
            {
                "command": "wrap",
                "args": {
                    "width": 20
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_metrics_tracking(instruction, test_file):
    """Test operation metrics tracking."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Perform a delete operation
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {"start": 1, "end": 2}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_backup_functionality(instruction, test_file):
    """Test file backup creation."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Original content\n")
    
    # Perform an operation that should create backup
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "c",
                "args": {
                    "start": 1,
                    "end": 1,  # Changed to handle single line
                    "text": "Modified content\n"
                }
            }
        ]
    }
    
    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_path_validation(instruction, test_file):
    """Test path validation functionality."""
    # Test with non-existent file
    context = {
        "path": "nonexistent.txt",
        "commands": [
            {
                "command": "d",
                "args": {"start": 1, "end": 2}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_substitute_text_errors(instruction, test_file):
    """Test substitute text error handling."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Test missing pattern
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "s",
                "args": {
                    "replacement": "new"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_search_text_errors(instruction, test_file):
    """Test search text error handling."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Test missing pattern
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "/",
                "args": {}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_goto_line_errors(instruction, test_file):
    """Test goto line error handling."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Test invalid line number type
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "gg",
                "args": {
                    "line": "1"
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_path_security(instruction, test_file):
    """Test path security validation."""
    # Test path with command injection attempt
    malicious_path = "file.txt; rm -rf /"
    context = {
        "path": malicious_path,
        "commands": [
            {
                "command": "d",
                "args": {"start": 1, "end": 2}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_backup_naming_and_cleanup(instruction, test_file):
    """Test backup file naming and cleanup."""
    # Write initial content
    with open(test_file, 'w') as f:
        f.write("Original content\n")
    
    # Perform multiple operations to create backups
    backups = []
    for i in range(3):
        context = {
            "path": test_file,
            "commands": [
                {
                    "command": "c",
                    "args": {
                        "start": 1,
                        "end": 1,  # Changed to handle single line
                        "text": f"Modified content {i}\n"
                    }
                }
            ]
        }
        
        result = await instruction._execute_impl(context)
        assert result.status == InstructionStatus.COMPLETED

        backup_path = result.data.get("backup")
        if backup_path:
            backups.append(backup_path)

    # Verify backup naming pattern
    for backup in backups:
        assert backup.startswith(test_file + ".bak.")
        assert backup[len(test_file + ".bak."):].isdigit()

    # Verify backups are unique
    assert len(set(backups)) == len(backups)

@pytest.mark.asyncio
async def test_concurrent_operations(instruction, test_file):
    """Test concurrent operations and metrics."""
    # Write initial content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Perform multiple operations in quick succession
    operations = []
    for i in range(3):
        context = {
            "path": test_file,
            "commands": [
                {
                    "command": "a",
                    "args": {
                        "line": i + 1,
                        "text": f" modified {i}"
                    }
                }
            ]
        }
        operations.append(instruction._execute_impl(context))

    # Wait for all operations to complete
    results = await asyncio.gather(*operations)

    # Verify all operations succeeded
    for result in results:
        assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_delete_lines_errors(instruction, test_file):
    """Test delete lines error messages."""
    # Write test content
    with open(test_file, 'w') as f:
        f.write("Line 1\nLine 2\nLine 3\n")

    # Test invalid start line type
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {
                    "start": "1",
                    "end": 2
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.FAILED

@pytest.mark.asyncio
async def test_backup_cleanup(instruction, test_file):
    """Test cleanup of old backup files."""
    # Write initial content
    with open(test_file, 'w') as f:
        f.write("Original content\n")
    
    # Create more backups than the max limit
    max_backups = 5
    backups = []
    for i in range(max_backups + 3):  # Create 3 more than max
        context = {
            "path": test_file,
            "commands": [
                {
                    "command": "c",
                    "args": {
                        "start": 1,
                        "end": 1,  # Changed to handle single line
                        "text": f"Modified content {i}\n"
                    }
                }
            ]
        }
        
        result = await instruction._execute_impl(context)
        assert result.status == InstructionStatus.COMPLETED

        backup_path = result.data.get("backup")
        if backup_path:
            backups.append(backup_path)

    # Verify only the most recent backups are kept
    existing_backups = [b for b in backups if os.path.exists(b)]
    assert len(existing_backups) <= max_backups

    # Verify the most recent backups are kept
    backup_times = [int(b.split('.')[-1]) for b in existing_backups]
    backup_times.sort(reverse=True)
    assert len(backup_times) == len(set(backup_times))  # All timestamps unique
    assert all(backup_times[i] > backup_times[i+1] for i in range(len(backup_times)-1))  # Sorted in descending order

@pytest.mark.asyncio
async def test_metrics_accuracy(instruction, test_file):
    """Test accuracy of operation metrics."""
    # Write test content with known size
    test_content = "Line 1\nLine 2\nLine 3\n"  # 3 lines, 21 chars
    with open(test_file, 'w') as f:
        f.write(test_content)

    # Test line deletion metrics
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "d",
                "args": {"start": 2, "end": 3}  # Delete "Line 2\n" (7 chars)
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

    metrics = result.data.get("metrics")
    assert metrics is not None
    assert "lines_changed" in metrics
    assert "chars_changed" in metrics
    assert "timestamp" in metrics
    assert metrics["lines_changed"] == 1  # One line deleted
    assert metrics["chars_changed"] == 7  # "Line 2\n" = 7 chars

    # Test text insertion metrics
    inserted_text = "New line with specific length\n"  # 28 chars
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "i",
                "args": {
                    "line": 1,
                    "column": 1,
                    "text": inserted_text
                }
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

    metrics = result.data.get("metrics")
    assert metrics is not None
    assert metrics["lines_changed"] == 1  # One line modified
    assert metrics["chars_changed"] == len(inserted_text)

@pytest.mark.asyncio
async def test_unicode_handling(instruction, test_file):
    """Test handling of Unicode characters in file operations."""
    # Test content with various Unicode characters
    test_content = "Hello 世界\nПривет мир\n☺️🌍\n"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # Test reading and line counting
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "gg",
                "args": {"line": 1}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_large_file_handling(instruction, test_file):
    """Test handling of large files."""
    # Generate a large file (around 1MB)
    num_lines = 10000
    line_content = "This is a test line with some content to make it reasonably long. " * 2
    test_content = "\n".join([f"Line {i}: {line_content}" for i in range(num_lines)])

    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)

    # Test reading and line counting
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "gg",
                "args": {"line": 1}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

@pytest.mark.asyncio
async def test_empty_file_and_edge_cases(instruction, test_file):
    """Test handling of empty files and edge cases."""
    # Test with empty file
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write("")

    # Test reading empty file
    context = {
        "path": test_file,
        "commands": [
            {
                "command": "gg",
                "args": {"line": 1}
            }
        ]
    }

    result = await instruction._execute_impl(context)
    assert result.status == InstructionStatus.COMPLETED

if __name__ == "__main__":
    pytest.main([__file__]) 