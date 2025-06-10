"""Process operation instructions."""

from typing import Dict, Any, Optional
import signal
import psutil

from .base import ProcessInstruction, InstructionResult, InstructionStatus

class ProcessListInstruction(ProcessInstruction):
    """List running processes."""
    
    def __init__(self):
        super().__init__(
            name="ps",
            description="List running processes"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute ps command."""
        command = "ps aux"
        return await self._run_command(command)

class ProcessKillInstruction(ProcessInstruction):
    """Kill a process."""
    
    def __init__(self):
        super().__init__(
            name="kill",
            description="Kill a process"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute kill command."""
        pid = context.get("pid")
        sig = context.get("signal", "SIGTERM")
        
        if not pid:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Process ID not provided"
            )
            
        if not self._validate_pid(pid):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid process ID: {pid}"
            )
            
        if not self._validate_signal(sig):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Signal not allowed: {sig}"
            )
            
        command = f"kill -{sig} {pid}"
        return await self._run_command(command)

class ProcessTopInstruction(ProcessInstruction):
    """Show system processes."""
    
    def __init__(self):
        super().__init__(
            name="top",
            description="Show system processes"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute top command."""
        # Use top in batch mode to get a single snapshot
        command = "top -b -n 1"
        return await self._run_command(command)

class ProcessInfoInstruction(ProcessInstruction):
    """Get detailed process information."""
    
    def __init__(self):
        super().__init__(
            name="pinfo",
            description="Get detailed process information"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Get process information using psutil."""
        pid = context.get("pid")
        
        if not pid:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Process ID not provided"
            )
            
        if not self._validate_pid(pid):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid process ID: {pid}"
            )
            
        try:
            process = psutil.Process(pid)
            info = {
                "name": process.name(),
                "status": process.status(),
                "cpu_percent": process.cpu_percent(),
                "memory_percent": process.memory_percent(),
                "create_time": process.create_time(),
                "cmdline": process.cmdline(),
                "username": process.username(),
                "num_threads": process.num_threads()
            }
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                output=str(info)
            )
        except psutil.NoSuchProcess:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Process {pid} not found"
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

class ProcessTreeInstruction(ProcessInstruction):
    """Show process tree."""
    
    def __init__(self):
        super().__init__(
            name="pstree",
            description="Show process tree"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute pstree command."""
        pid = context.get("pid")
        
        if pid:
            if not self._validate_pid(pid):
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error=f"Invalid process ID: {pid}"
                )
            command = f"pstree {pid}"
        else:
            command = "pstree"
            
        return await self._run_command(command)

class ProcessNiceInstruction(ProcessInstruction):
    """Change process priority."""
    
    def __init__(self):
        super().__init__(
            name="nice",
            description="Change process priority"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute nice/renice command."""
        pid = context.get("pid")
        priority = context.get("priority")
        
        if not pid or priority is None:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Process ID and priority must be provided"
            )
            
        if not self._validate_pid(pid):
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid process ID: {pid}"
            )
            
        # Validate priority range (-20 to 19)
        if not -20 <= priority <= 19:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Priority must be between -20 and 19"
            )
            
        command = f"renice -n {priority} -p {pid}"
        return await self._run_command(command) 