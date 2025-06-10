"""Test file and content search instructions."""

import pytest
import os
import time
from pathlib import Path
from typing import Dict, Any
import shutil
import asyncio

from agentflow.core.instructions.search import (
    FileSearchInstruction,
    FileContentSearchInstruction,
    SearchMetrics
)
from agentflow.core.instructions.base import InstructionStatus

@pytest.fixture
def test_dir(tmp_path):
    """Create a temporary directory with test files."""
    # Create test directory structure
    test_files = {
        "file1.txt": "This is a test file\nwith multiple lines\nand some content",
        "file2.py": "def test_function():\n    print('Hello')\n    # TODO: implement",
        "subfolder/file3.js": "console.log('test');\n// TODO: add more",
        "subfolder/file4.txt": "Another test file\nwith TODO: items",
        "large_file.txt": "x" * (1024 * 1024),  # 1MB file
        ".hidden_file": "Hidden file content",
    }
    
    for file_path, content in test_files.items():
        full_path = tmp_path / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
    
    return tmp_path

@pytest.fixture
def file_search():
    """Create FileSearchInstruction instance."""
    return FileSearchInstruction()

@pytest.fixture
def content_search():
    """Create FileContentSearchInstruction instance."""
    return FileContentSearchInstruction()

@pytest.mark.asyncio
async def test_file_search_basic(file_search, test_dir):
    """Test basic file search functionality."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "*.txt",
        "recursive": True
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 3  # file1.txt, file4.txt, large_file.txt
    
    # Verify metrics
    metrics = result.data["metrics"]
    assert metrics["files_searched"] > 0
    assert metrics["matches_found"] == 3
    assert metrics["execution_time"] > 0
    assert metrics["total_size_searched"] > 0

@pytest.mark.asyncio
async def test_file_search_with_size_filter(file_search, test_dir):
    """Test file search with size filters."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "*",
        "min_size": 500 * 1024,  # 500KB
        "max_size": 2 * 1024 * 1024  # 2MB
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 1  # Only large_file.txt

@pytest.mark.asyncio
async def test_file_search_with_type_filter(file_search, test_dir):
    """Test file search with file type filter."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "*",
        "file_types": [".py", ".js"]
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 2  # file2.py and file3.js

@pytest.mark.asyncio
async def test_file_search_with_regex(file_search, test_dir):
    """Test file search with regex pattern."""
    context = {
        "root_dir": str(test_dir),
        "pattern": r"file\d+\.txt",
        "use_regex": True
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 2  # file1.txt and file4.txt

@pytest.mark.asyncio
async def test_file_search_with_exclusions(file_search, test_dir):
    """Test file search with exclusions."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "*",
        "exclude_dirs": ["subfolder"],
        "exclude_patterns": ["*.txt"]
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 2  # file2.py and .hidden_file

@pytest.mark.asyncio
async def test_content_search_basic(content_search, test_dir):
    """Test basic content search functionality."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "TODO:",
        "recursive": True
    }
    
    result = await content_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert len(result.data["results"]) == 3  # Matches in file2.py, file3.js, and file4.txt
    
    # Verify match details
    first_match = result.data["results"][0]
    assert "file" in first_match
    assert "matches" in first_match
    assert "total_matches" in first_match
    
    # Verify match content
    match = first_match["matches"][0]
    assert "line_number" in match
    assert "line" in match
    assert "match" in match
    assert "context" in match
    assert isinstance(match["context"], dict)
    assert "lines" in match["context"]

@pytest.mark.asyncio
async def test_content_search_with_context(content_search, test_dir):
    """Test content search with different context sizes."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "TODO:",
        "context_lines": 1
    }
    
    result = await content_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    
    # Verify context lines
    first_match = result.data["results"][0]["matches"][0]
    assert len(first_match["context"]["lines"]) <= 3  # 1 line before + match line + 1 line after

@pytest.mark.asyncio
async def test_content_search_with_max_matches(content_search, test_dir):
    """Test content search with maximum matches limit."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "test",
        "max_matches": 2
    }
    
    result = await content_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    assert result.data["metrics"]["matches_found"] <= 2

@pytest.mark.asyncio
async def test_content_search_with_case_sensitivity(content_search, test_dir):
    """Test content search with case sensitivity options."""
    # Case sensitive search
    context = {
        "root_dir": str(test_dir),
        "pattern": "Test",
        "case_sensitive": True
    }
    
    result = await content_search.execute(context)
    sensitive_matches = result.data["metrics"]["matches_found"]
    
    # Case insensitive search
    context["case_sensitive"] = False
    result = await content_search.execute(context)
    insensitive_matches = result.data["metrics"]["matches_found"]
    
    assert insensitive_matches > sensitive_matches

@pytest.mark.asyncio
async def test_content_search_with_file_size_limit(content_search, test_dir):
    """Test content search with file size limit."""
    context = {
        "root_dir": str(test_dir),
        "pattern": "x",
        "max_file_size": 1024  # 1KB
    }
    
    result = await content_search.execute(context)
    assert result.status == InstructionStatus.COMPLETED
    # Should not include matches from large_file.txt
    assert all(r["file"].endswith("large_file.txt") == False for r in result.data["results"])

@pytest.mark.asyncio
async def test_search_error_handling(file_search, content_search, test_dir):
    """Test error handling in search instructions."""
    # Test invalid regex pattern
    context = {
        "root_dir": str(test_dir),
        "pattern": "[invalid",
        "use_regex": True
    }
    
    result = await file_search.execute(context)
    assert result.status == InstructionStatus.FAILED
    assert "Invalid regex pattern" in result.error
    
    # Test non-existent directory
    context = {
        "root_dir": "nonexistent",
        "pattern": "*"
    }
    
    result = await content_search.execute(context)
    assert result.status == InstructionStatus.FAILED
    assert "does not exist" in result.error

@pytest.mark.asyncio
async def test_concurrent_searches(file_search, content_search, test_dir):
    """Test concurrent search operations."""
    search_tasks = []
    
    # Create multiple search tasks
    for pattern in ["*.txt", "*.py", "*.js"]:
        context = {
            "root_dir": str(test_dir),
            "pattern": pattern
        }
        search_tasks.append(file_search.execute(context))
    
    # Run searches concurrently
    results = await asyncio.gather(*search_tasks)
    
    # Verify all searches completed successfully
    assert all(r.status == InstructionStatus.COMPLETED for r in results)
    assert sum(len(r.data["results"]) for r in results) == 5  # Total number of matching files

if __name__ == "__main__":
    pytest.main([__file__]) 