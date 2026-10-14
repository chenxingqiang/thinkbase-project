"""
Example implementation demonstrating comprehensive file operations using AgentFlow.
This example shows how to use all file operation instructions effectively.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional
from pathlib import Path
from dataclasses import dataclass

from agentflow.core.instructions.file_operations import (
    FileCreateInstruction,
    FileReadInstruction,
    FileUpdateInstruction,
    FileDeleteInstruction,
    FileSearchInstruction,
    FileMetadataInstruction,
    FileEditInstruction,
    FileChangeMetrics,
    DiffInfo
)
from agentflow.core.instructions.base import InstructionStatus, InstructionResult

@dataclass
class FileOperationResult:
    """Result of a file operation."""
    success: bool
    message: str
    data: Dict[str, Any]
    metrics: Dict[str, Any]
    
    def __init__(self, success: bool, message: str, data: Optional[Dict[str, Any]] = None, metrics: Optional[Dict[str, Any]] = None):
        self.success = success
        self.message = message
        self.data = data or {}
        self.metrics = metrics or {}

class FileOperationsDemo:
    """Demonstration of comprehensive file operations."""
    
    def __init__(self, workspace_dir: str = "./workspace"):
        self.workspace = Path(workspace_dir)
        self.workspace.mkdir(parents=True, exist_ok=True)
        
        # Initialize instructions
        self.create_instruction = FileCreateInstruction()
        self.read_instruction = FileReadInstruction()
        self.update_instruction = FileUpdateInstruction()
        self.delete_instruction = FileDeleteInstruction()
        self.search_instruction = FileSearchInstruction()
        self.metadata_instruction = FileMetadataInstruction()
        self.edit_instruction = FileEditInstruction()
        
    async def create_file(self, filename: str, content: str) -> FileOperationResult:
        """Create a new file with content."""
        try:
            filepath = self.workspace / filename
            result = await self.create_instruction.execute({
                "path": str(filepath),
                "content": content
            })
            
            return FileOperationResult(
                success=result.status == InstructionStatus.COMPLETED.value,
                message=f"File {filename} created successfully",
                data=result.data,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to create file: {str(e)}",
                data={},
                metrics={}
            )
    
    async def read_file(self, filename: str) -> FileOperationResult:
        """Read file content."""
        try:
            filepath = self.workspace / filename
            result = await self.read_instruction.execute({
                "path": str(filepath)
            })
            
            read_ok = result.status == InstructionStatus.COMPLETED.value
            return FileOperationResult(
                success=read_ok,
                message=(
                    f"File {filename} read successfully"
                    if read_ok
                    else f"Failed to read file {filename}"
                ),
                data=result.data,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to read file: {str(e)}",
                data={},
                metrics={}
            )
    
    async def update_file(self, filename: str, content: str) -> FileOperationResult:
        """Update file content.

        Unlike creation, updating requires the file to already exist:
        a missing file is an error, not an implicit create.
        """
        try:
            filepath = self.workspace / filename
            if not filepath.exists():
                return FileOperationResult(
                    success=False,
                    message=f"Failed to update file: {filename} does not exist",
                    data={},
                    metrics={}
                )
            result = await self.update_instruction.execute({
                "path": str(filepath),
                "content": content
            })
            
            update_ok = result.status == "completed"
            return FileOperationResult(
                success=update_ok,
                message=(
                    f"File {filename} updated successfully"
                    if update_ok
                    else f"Failed to update file {filename}"
                ),
                data=result.data,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to update file: {str(e)}"
            )
    
    async def delete_file(self, filename: str) -> FileOperationResult:
        """Delete a file."""
        try:
            filepath = self.workspace / filename
            result = await self.delete_instruction.execute({
                "path": str(filepath)
            })
            
            delete_ok = result.status == "completed"
            return FileOperationResult(
                success=delete_ok,
                message=(
                    f"File {filename} deleted successfully"
                    if delete_ok
                    else f"Failed to delete file {filename}"
                ),
                data=result.data,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to delete file: {str(e)}"
            )
    
    async def search_files(self, pattern: str) -> FileOperationResult:
        """Search files matching pattern."""
        try:
            result = await self.search_instruction.execute({
                "pattern": pattern,
                "root_dir": str(self.workspace)
            })
            
            return FileOperationResult(
                success=result.status == InstructionStatus.COMPLETED.value,
                message=f"Search completed successfully",
                data=result.data,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to search files: {str(e)}",
                data={},
                metrics={}
            )
    
    async def get_file_metadata(self, filename: str) -> FileOperationResult:
        """Get file metadata."""
        try:
            filepath = self.workspace / filename
            result = await self.metadata_instruction.execute({
                "path": str(filepath)
            })
            
            metadata_ok = result.status == "completed"
            # Normalize stat()-style keys (created/modified) to the
            # *_at naming used by the demo API.
            metadata = dict(result.data or {})
            if "created" in metadata:
                metadata.setdefault("created_at", metadata["created"])
            if "modified" in metadata:
                metadata.setdefault("modified_at", metadata["modified"])
            return FileOperationResult(
                success=metadata_ok,
                message="Metadata retrieved successfully" if metadata_ok else f"Failed to get metadata for {filename}",
                data=metadata,
                metrics=result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to get metadata: {str(e)}"
            )
    
    async def edit_file_content(self, filename: str, edit_operations: List[Dict[str, Any]]) -> FileOperationResult:
        """Perform multiple edit operations on a file."""
        try:
            filepath = self.workspace / filename
            
            # Read current content
            read_result = await self.read_file(filename)
            if not read_result.success:
                # Keep the edit-level failure context (the read failure
                # is a cause, not the operation the caller asked about).
                return FileOperationResult(
                    success=False,
                    message=f"Failed to edit file {filename}: could not read current content",
                    data={},
                    metrics=read_result.metrics
                )
            
            current_content = read_result.data["content"]
            
            # Apply each edit operation
            for operation in edit_operations:
                result = await self.edit_instruction.execute({
                    "path": str(filepath),
                    "content": current_content,
                    "operation": operation
                })
                
                if result.status != "completed":
                    return FileOperationResult(
                        success=False,
                        message=f"Edit operation failed: {operation['type']}",
                        data=result.data
                    )
                
                current_content = result.data.get("content")
                if current_content is None and "lines" in result.data:
                    current_content = "\n".join(result.data["lines"])
            
            # Update file with final content
            update_result = await self.update_file(filename, current_content)
            
            return FileOperationResult(
                success=update_result.success,
                message="File editing completed successfully",
                data={"final_content": current_content},
                metrics=update_result.metrics
            )
        except Exception as e:
            return FileOperationResult(
                success=False,
                message=f"Failed to edit file: {str(e)}"
            )

async def main():
    """Demonstrate file operations usage."""
    demo = FileOperationsDemo("./test_workspace")
    
    # 1. Create a test file
    print("\n1. Creating test file...")
    create_result = await demo.create_file(
        "test.txt",
        "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
    )
    print(f"Create result: {create_result.message}")
    
    # 2. Read the file
    print("\n2. Reading file...")
    read_result = await demo.read_file("test.txt")
    print(f"Read result: {read_result.message}")
    if read_result.success and 'content' in read_result.data:
        print(f"Content:\n{read_result.data['content']}")
    
    # 3. Get file metadata
    print("\n3. Getting metadata...")
    metadata_result = await demo.get_file_metadata("test.txt")
    print(f"Metadata result: {metadata_result.message}")
    if metadata_result.success:
        print(f"Metadata: {metadata_result.data}")
    
    # 4. Perform various edit operations
    print("\n4. Performing edit operations...")
    edit_operations = [
        {
            "type": "insert",
            "line": 1,
            "text": "Inserted Line"
        },
        {
            "type": "delete",
            "line": 3
        }
    ]
    edit_result = await demo.edit_file_content("test.txt", edit_operations)
    print(f"Edit result: {edit_result.message}")
    
    # 5. Search for files
    print("\n5. Searching files...")
    search_result = await demo.search_files("*.txt")
    print(f"Search result: {search_result.message}")
    if search_result.success and 'matches' in search_result.data:
        print(f"Found files: {search_result.data['matches']}")
    
    # 6. Update file
    print("\n6. Updating file...")
    update_result = await demo.update_file(
        "test.txt",
        "Updated content\nNew line"
    )
    print(f"Update result: {update_result.message}")
    
    # 7. Delete file
    print("\n7. Deleting file...")
    delete_result = await demo.delete_file("test.txt")
    print(f"Delete result: {delete_result.message}")

if __name__ == "__main__":
    asyncio.run(main()) 