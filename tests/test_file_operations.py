"""
Test suite for file operations functionality.
"""

import pytest
import asyncio
from pathlib import Path
import shutil
from typing import AsyncGenerator

from examples.file_operations_example import FileOperationsDemo, FileOperationResult

@pytest.fixture
async def demo() -> AsyncGenerator[FileOperationsDemo, None]:
    """Create a FileOperationsDemo instance with a temporary workspace."""
    workspace = Path("./test_workspace")
    workspace.mkdir(exist_ok=True)
    
    demo = FileOperationsDemo(str(workspace))
    yield demo
    
    # Cleanup
    shutil.rmtree(workspace)

@pytest.mark.asyncio
async def test_create_file(demo: FileOperationsDemo):
    """Test file creation."""
    result = await demo.create_file("test.txt", "Hello, World!")
    assert result.success
    assert result.message == "File test.txt created successfully"
    
    # Verify file exists and content
    filepath = demo.workspace / "test.txt"
    assert filepath.exists()
    assert filepath.read_text() == "Hello, World!"

@pytest.mark.asyncio
async def test_read_file(demo: FileOperationsDemo):
    """Test file reading."""
    # Create a test file first
    await demo.create_file("test.txt", "Hello, World!")
    
    # Test reading
    result = await demo.read_file("test.txt")
    assert result.success
    assert result.message == "File test.txt read successfully"
    assert result.data is not None
    assert result.data.get("content") == "Hello, World!"

@pytest.mark.asyncio
async def test_update_file(demo: FileOperationsDemo):
    """Test file updating."""
    # Create a test file first
    await demo.create_file("test.txt", "Original content")
    
    # Test updating
    result = await demo.update_file("test.txt", "Updated content")
    assert result.success
    assert result.message == "File test.txt updated successfully"
    
    # Verify content was updated
    filepath = demo.workspace / "test.txt"
    assert filepath.read_text() == "Updated content"

@pytest.mark.asyncio
async def test_delete_file(demo: FileOperationsDemo):
    """Test file deletion."""
    # Create a test file first
    await demo.create_file("test.txt", "Delete me")
    
    # Test deletion
    result = await demo.delete_file("test.txt")
    assert result.success
    assert result.message == "File test.txt deleted successfully"
    
    # Verify file was deleted
    filepath = demo.workspace / "test.txt"
    assert not filepath.exists()

@pytest.mark.asyncio
async def test_search_files(demo: FileOperationsDemo):
    """Test file searching."""
    # Create multiple test files
    await demo.create_file("test1.txt", "Content 1")
    await demo.create_file("test2.txt", "Content 2")
    await demo.create_file("other.dat", "Other content")
    
    # Test searching
    result = await demo.search_files("*.txt")
    assert result.success
    assert result.message == "Search completed successfully"
    assert result.data is not None
    
    matches = result.data.get("matches", [])
    assert len(matches) == 2
    assert any("test1.txt" in match for match in matches)
    assert any("test2.txt" in match for match in matches)
    assert not any("other.dat" in match for match in matches)

@pytest.mark.asyncio
async def test_get_file_metadata(demo: FileOperationsDemo):
    """Test getting file metadata."""
    # Create a test file first
    await demo.create_file("test.txt", "Metadata test")
    
    # Test getting metadata
    result = await demo.get_file_metadata("test.txt")
    assert result.success
    assert result.message == "Metadata retrieved successfully"
    assert result.data is not None
    
    metadata = result.data
    assert "size" in metadata
    assert "created_at" in metadata
    assert "modified_at" in metadata
    assert "is_file" in metadata
    assert metadata["is_file"] is True

@pytest.mark.asyncio
async def test_edit_file_content(demo: FileOperationsDemo):
    """Test file content editing."""
    # Create a test file first
    initial_content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    await demo.create_file("test.txt", initial_content)
    
    # Test various edit operations
    edit_operations = [
        {
            "type": "insert_text",
            "position": 1,
            "text": "New Line\n"
        },
        {
            "type": "delete_lines",
            "start": 3,
            "end": 3
        },
        {
            "type": "change_case",
            "start": 1,
            "end": 2,
            "mode": "upper"
        }
    ]
    
    result = await demo.edit_file_content("test.txt", edit_operations)
    assert result.success
    assert result.message == "File editing completed successfully"
    assert result.data is not None
    
    # Verify the final content
    final_content = result.data.get("final_content")
    assert "New Line" in final_content
    assert "LINE 2" in final_content  # Should be uppercase
    assert "Line 3" not in final_content  # Should be deleted

@pytest.mark.asyncio
async def test_error_handling(demo: FileOperationsDemo):
    """Test error handling in file operations."""
    # Test reading non-existent file
    result = await demo.read_file("nonexistent.txt")
    assert not result.success
    assert "Failed to read file" in result.message
    
    # Test deleting non-existent file
    result = await demo.delete_file("nonexistent.txt")
    assert not result.success
    assert "Failed to delete file" in result.message
    
    # Test updating non-existent file
    result = await demo.update_file("nonexistent.txt", "content")
    assert not result.success
    assert "Failed to update file" in result.message
    
    # Test invalid edit operation
    result = await demo.edit_file_content("nonexistent.txt", [
        {"type": "invalid_operation"}
    ])
    assert not result.success
    assert "Failed to edit file" in result.message 