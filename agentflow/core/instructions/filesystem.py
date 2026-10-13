"""Filesystem operation instructions."""

from typing import Dict, Any, Optional
import os
import shutil
from pathlib import Path

from .base import FileSystemInstruction, InstructionResult, InstructionStatus

class ListDirectoryInstruction(FileSystemInstruction):
    """List directory contents."""
    
    def __init__(self):
        super().__init__(
            name="ls",
            description="List directory contents"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute ls command."""
        path = context.get("path", ".")
        if not self._validate_path(path):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Path not allowed: {path}"
            )
            
        path = self._sanitize_path(path)
        command = f"ls -la {path}"
        return await self._run_command(command)

class MakeDirectoryInstruction(FileSystemInstruction):
    """Create directory."""
    
    def __init__(self):
        super().__init__(
            name="mkdir",
            description="Create directory"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute mkdir command."""
        path = context.get("path")
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Path not provided"
            )
            
        if not self._validate_path(path):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Path not allowed: {path}"
            )
            
        path = self._sanitize_path(path)
        command = f"mkdir -p {path}"
        return await self._run_command(command)

class RemoveInstruction(FileSystemInstruction):
    """Remove files or directories."""
    
    def __init__(self):
        super().__init__(
            name="rm",
            description="Remove files or directories"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute rm command."""
        path = context.get("path")
        if not path:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Path not provided"
            )
            
        if not self._validate_path(path):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Path not allowed: {path}"
            )
            
        recursive = context.get("recursive", False)
        force = context.get("force", False)
        
        path = self._sanitize_path(path)
        options = []
        if recursive:
            options.append("-r")
        if force:
            options.append("-f")
            
        command = f"rm {' '.join(options)} {path}"
        return await self._run_command(command)

class CopyInstruction(FileSystemInstruction):
    """Copy files or directories."""
    
    def __init__(self):
        super().__init__(
            name="cp",
            description="Copy files or directories"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute cp command."""
        source = context.get("source")
        destination = context.get("destination")
        
        if not source or not destination:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Source and destination must be provided"
            )
            
        if not self._validate_path(source) or not self._validate_path(destination):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Invalid path specified"
            )
            
        recursive = context.get("recursive", False)
        
        source = self._sanitize_path(source)
        destination = self._sanitize_path(destination)
        
        options = ["-a"] if recursive else []
        command = f"cp {' '.join(options)} {source} {destination}"
        return await self._run_command(command)

class MoveInstruction(FileSystemInstruction):
    """Move/rename files or directories."""
    
    def __init__(self):
        super().__init__(
            name="mv",
            description="Move/rename files or directories"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute mv command."""
        source = context.get("source")
        destination = context.get("destination")
        
        if not source or not destination:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Source and destination must be provided"
            )
            
        if not self._validate_path(source) or not self._validate_path(destination):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Invalid path specified"
            )
            
        source = self._sanitize_path(source)
        destination = self._sanitize_path(destination)
        
        command = f"mv {source} {destination}"
        return await self._run_command(command)

class ChangePermissionsInstruction(FileSystemInstruction):
    """Change file permissions."""
    
    def __init__(self):
        super().__init__(
            name="chmod",
            description="Change file permissions"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute chmod command."""
        path = context.get("path")
        mode = context.get("mode")
        
        if not path or not mode:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Path and mode must be provided"
            )
            
        if not self._validate_path(path):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Path not allowed: {path}"
            )
            
        # Validate mode format (octal or symbolic)
        if not isinstance(mode, (str, int)):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Invalid mode format"
            )
            
        path = self._sanitize_path(path)
        command = f"chmod {mode} {path}"
        return await self._run_command(command)

class FindInstruction(FileSystemInstruction):
    """Find files and directories."""
    
    def __init__(self):
        super().__init__(
            name="find",
            description="Find files and directories"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute find command."""
        path = context.get("path", ".")
        name = context.get("name")
        type = context.get("type")
        
        if not self._validate_path(path):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Path not allowed: {path}"
            )
            
        path = self._sanitize_path(path)
        command = ["find", path]
        
        if name:
            command.extend(["-name", f"'{name}'"])
        if type:
            command.extend(["-type", type])
            
        return await self._run_command(" ".join(command)) 