"""File and content search instructions."""

from typing import Dict, Any, Optional, List, Union, Tuple, Set
import os
import fnmatch
import re
from pathlib import Path
import logging
from datetime import datetime
from dataclasses import dataclass
import time

from .base import BaseInstruction, InstructionResult, InstructionStatus

logger = logging.getLogger(__name__)

@dataclass
class SearchMetrics:
    """Metrics for search operations."""
    files_searched: int
    matches_found: int
    execution_time: float
    total_size_searched: int

class FileSearchInstruction(BaseInstruction):
    """Search for files based on various criteria."""
    
    def __init__(self):
        super().__init__(
            name="file_search",
            description="Search for files based on name, pattern, and other criteria"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Internal implementation of file search."""
        return await self.execute(context)
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute file search with various criteria.
        
        Args:
            context: Dictionary containing:
                - root_dir: Root directory to start search from
                - pattern: File name pattern (glob/regex)
                - use_regex: Whether to use regex for pattern matching
                - recursive: Whether to search recursively
                - max_depth: Maximum directory depth to search
                - min_size: Minimum file size in bytes
                - max_size: Maximum file size in bytes
                - modified_after: Files modified after this timestamp
                - modified_before: Files modified before this timestamp
                - file_types: List of file extensions to search
                - exclude_dirs: List of directories to exclude
                - exclude_patterns: List of patterns to exclude
                - follow_symlinks: Whether to follow symbolic links
                - case_sensitive: Whether pattern matching is case sensitive
                
        Returns:
            InstructionResult with search results
        """
        try:
            # Get search parameters
            root_dir = context.get("root_dir", ".")
            pattern = context.get("pattern", "*")
            use_regex = context.get("use_regex", False)
            recursive = context.get("recursive", True)
            max_depth = context.get("max_depth")
            min_size = context.get("min_size")
            max_size = context.get("max_size")
            modified_after = context.get("modified_after")
            modified_before = context.get("modified_before")
            file_types = context.get("file_types", [])
            exclude_dirs = set(context.get("exclude_dirs", []))
            exclude_patterns = context.get("exclude_patterns", [])
            follow_symlinks = context.get("follow_symlinks", False)
            case_sensitive = context.get("case_sensitive", True)
            
            # Validate root directory
            if not os.path.exists(root_dir):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Root directory does not exist: {root_dir}"
                )
            
            # Compile regex pattern if using regex
            if use_regex:
                try:
                    flags = 0 if case_sensitive else re.IGNORECASE
                    pattern_regex = re.compile(pattern, flags)
                except re.error as e:
                    return InstructionResult(
                        status=InstructionStatus.FAILED,
                        error=f"Invalid regex pattern: {str(e)}"
                    )
            
            # Initialize metrics
            metrics = SearchMetrics(
                files_searched=0,
                matches_found=0,
                execution_time=0,
                total_size_searched=0
            )
            
            start_time = time.time()
            results = []
            
            # Walk directory tree
            root_path = Path(root_dir)
            for root, dirs, files in os.walk(root_dir, followlinks=follow_symlinks):
                # Check depth
                if max_depth is not None:
                    current_depth = len(Path(root).relative_to(root_path).parts)
                    if current_depth > max_depth:
                        continue
                
                # Remove excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                # Process files
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Get file stats
                        stats = os.stat(file_path, follow_symlinks=follow_symlinks)
                        
                        # Skip if doesn't match size constraints
                        if min_size is not None and stats.st_size < min_size:
                            continue
                        if max_size is not None and stats.st_size > max_size:
                            continue
                        
                        # Skip if doesn't match modification time constraints
                        if modified_after is not None and stats.st_mtime < modified_after:
                            continue
                        if modified_before is not None and stats.st_mtime > modified_before:
                            continue
                        
                        # Skip if doesn't match file type
                        if file_types and not any(file.endswith(ft) for ft in file_types):
                            continue
                        
                        # Skip if matches exclude patterns
                        if any(fnmatch.fnmatch(file, ep) for ep in exclude_patterns):
                            continue
                        
                        # Check if matches pattern
                        match = False
                        if use_regex:
                            match = bool(pattern_regex.search(file))
                        else:
                            match = fnmatch.fnmatch(file, pattern)
                        
                        if match:
                            results.append({
                                "path": file_path,
                                "size": stats.st_size,
                                "modified": datetime.fromtimestamp(stats.st_mtime).isoformat(),
                                "created": datetime.fromtimestamp(stats.st_ctime).isoformat(),
                                "type": "file"
                            })
                            metrics.matches_found += 1
                        
                        metrics.files_searched += 1
                        metrics.total_size_searched += stats.st_size
                        
                    except Exception as e:
                        logger.warning(f"Error processing file {file_path}: {e}")
                        continue
                
                if not recursive:
                    break
            
            metrics.execution_time = time.time() - start_time
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "results": results,
                    "metrics": {
                        "files_searched": metrics.files_searched,
                        "matches_found": metrics.matches_found,
                        "execution_time": metrics.execution_time,
                        "total_size_searched": metrics.total_size_searched
                    }
                }
            )
            
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

class FileContentSearchInstruction(BaseInstruction):
    """Search file contents for patterns."""
    
    def __init__(self):
        super().__init__(
            name="file_content_search",
            description="Search file contents for patterns"
        )
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Internal implementation of content search."""
        return await self.execute(context)
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute content search across files.
        
        Args:
            context: Dictionary containing:
                - root_dir: Root directory to start search from
                - pattern: Content pattern to search for
                - use_regex: Whether to use regex for pattern matching
                - recursive: Whether to search recursively
                - max_depth: Maximum directory depth to search
                - file_types: List of file extensions to search
                - exclude_dirs: List of directories to exclude
                - exclude_patterns: List of patterns to exclude
                - max_file_size: Maximum file size to search in bytes
                - case_sensitive: Whether pattern matching is case sensitive
                - context_lines: Number of context lines to include
                - max_matches: Maximum number of matches to return
                - encoding: File encoding (default: utf-8)
                
        Returns:
            InstructionResult with search results
        """
        try:
            # Get search parameters
            root_dir = context.get("root_dir", ".")
            pattern = context.get("pattern")
            use_regex = context.get("use_regex", True)
            recursive = context.get("recursive", True)
            max_depth = context.get("max_depth")
            file_types = context.get("file_types", [])
            exclude_dirs = set(context.get("exclude_dirs", []))
            exclude_patterns = context.get("exclude_patterns", [])
            max_file_size = context.get("max_file_size", 10 * 1024 * 1024)  # 10MB default
            case_sensitive = context.get("case_sensitive", True)
            context_lines = context.get("context_lines", 2)
            max_matches = context.get("max_matches")
            encoding = context.get("encoding", "utf-8")
            
            if not pattern:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Search pattern must be provided"
                )
            
            # Validate root directory
            if not os.path.exists(root_dir):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Root directory does not exist: {root_dir}"
                )
            
            # Compile regex pattern
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                if not use_regex:
                    pattern = re.escape(pattern)
                pattern_regex = re.compile(pattern, flags)
            except re.error as e:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Invalid regex pattern: {str(e)}"
                )
            
            # Initialize metrics
            metrics = SearchMetrics(
                files_searched=0,
                matches_found=0,
                execution_time=0,
                total_size_searched=0
            )
            
            start_time = time.time()
            results = []
            
            # Walk directory tree
            root_path = Path(root_dir)
            for root, dirs, files in os.walk(root_dir):
                # Check depth
                if max_depth is not None:
                    current_depth = len(Path(root).relative_to(root_path).parts)
                    if current_depth > max_depth:
                        continue
                
                # Remove excluded directories
                dirs[:] = [d for d in dirs if d not in exclude_dirs]
                
                # Process files
                for file in files:
                    # Check if max matches reached
                    if max_matches and metrics.matches_found >= max_matches:
                        break
                        
                    file_path = os.path.join(root, file)
                    try:
                        # Skip if doesn't match file type
                        if file_types and not any(file.endswith(ft) for ft in file_types):
                            continue
                        
                        # Skip if matches exclude patterns
                        if any(fnmatch.fnmatch(file, ep) for ep in exclude_patterns):
                            continue
                        
                        # Skip if file is too large
                        file_size = os.path.getsize(file_path)
                        if file_size > max_file_size:
                            continue
                        
                        # Read and search file content
                        with open(file_path, 'r', encoding=encoding) as f:
                            lines = f.readlines()
                            
                        file_matches = []
                        for i, line in enumerate(lines):
                            for match in pattern_regex.finditer(line):
                                # Get context lines
                                start_ctx = max(0, i - context_lines)
                                end_ctx = min(len(lines), i + context_lines + 1)
                                context = {
                                    "lines": lines[start_ctx:end_ctx],
                                    "start_line": start_ctx + 1,
                                    "end_line": end_ctx
                                }
                                
                                file_matches.append({
                                    "line_number": i + 1,
                                    "line": line.rstrip('\n'),
                                    "start": match.start(),
                                    "end": match.end(),
                                    "match": match.group(),
                                    "context": context
                                })
                                
                                metrics.matches_found += 1
                                if max_matches and metrics.matches_found >= max_matches:
                                    break
                            
                            if max_matches and metrics.matches_found >= max_matches:
                                break
                        
                        if file_matches:
                            results.append({
                                "file": file_path,
                                "matches": file_matches,
                                "total_matches": len(file_matches)
                            })
                        
                        metrics.files_searched += 1
                        metrics.total_size_searched += file_size
                        
                    except Exception as e:
                        logger.warning(f"Error processing file {file_path}: {e}")
                        continue
                
                if not recursive:
                    break
                
                if max_matches and metrics.matches_found >= max_matches:
                    break
            
            metrics.execution_time = time.time() - start_time
            
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data={
                    "results": results,
                    "metrics": {
                        "files_searched": metrics.files_searched,
                        "matches_found": metrics.matches_found,
                        "execution_time": metrics.execution_time,
                        "total_size_searched": metrics.total_size_searched
                    }
                }
            )
            
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            ) 