"""System operation instructions."""

from typing import Dict, Any, Optional
import psutil
import os
import logging
from .base import BaseInstruction, InstructionResult, InstructionStatus

logger = logging.getLogger(__name__)

class SystemMonitorInstruction(BaseInstruction):
    """Monitor system resources."""
    
    def __init__(self):
        super().__init__(
            name="sysmon",
            description="Monitor system resources"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Get system resource usage."""
        try:
            metrics = {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": dict(psutil.virtual_memory()._asdict()),
                "disk": {path: dict(psutil.disk_usage(path)._asdict())
                        for path in context.get("paths", ["/"])},
                "network": dict(psutil.net_io_counters()._asdict()),
                "boot_time": psutil.boot_time()
            }
            return InstructionResult(
                status=InstructionStatus.COMPLETED,
                data=metrics
            )
        except Exception as e:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=str(e)
            )

class ServiceControlInstruction(BaseInstruction):
    """Control system services."""
    
    def __init__(self):
        super().__init__(
            name="service",
            description="Control system services"
        )
        self.allowed_actions = ["status", "start", "stop", "restart"]
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute service control command."""
        service = context.get("service")
        action = context.get("action")
        
        if not service:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Service name not provided"
            )
            
        if action not in self.allowed_actions:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid action: {action}"
            )
            
        command = f"systemctl {action} {service}"
        return await self._run_command(command)

class LogViewInstruction(BaseInstruction):
    """View system logs."""
    
    def __init__(self):
        super().__init__(
            name="logview",
            description="View system logs"
        )
        self.allowed_logs = ["/var/log/syslog", "/var/log/messages"]
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """View log file contents."""
        log_file = context.get("log_file")
        lines = context.get("lines", 10)
        
        if not log_file:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Log file not specified"
            )
            
        if log_file not in self.allowed_logs:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Access to log file not allowed: {log_file}"
            )
            
        command = f"tail -n {lines} {log_file}"
        return await self._run_command(command)

class SystemConfigInstruction(BaseInstruction):
    """Manage system configuration."""
    
    def __init__(self):
        super().__init__(
            name="sysconfig",
            description="Manage system configuration"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Get or set system configuration."""
        action = context.get("action")
        param = context.get("param")
        
        if not action or not param:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Action and parameter must be provided"
            )
            
        if action == "get":
            command = f"sysctl {param}"
        elif action == "set":
            value = context.get("value")
            if value is None:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Value must be provided for set action"
                )
            command = f"sysctl -w {param}={value}"
        else:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid action: {action}"
            )
            
        return await self._run_command(command)

class DiskManagementInstruction(BaseInstruction):
    """Manage disk operations."""
    
    def __init__(self):
        super().__init__(
            name="diskman",
            description="Manage disk operations"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute disk management command."""
        action = context.get("action")
        device = context.get("device")
        
        if not action or not device:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Action and device must be provided"
            )
            
        if action == "info":
            command = f"df -h {device}"
        elif action == "usage":
            command = f"du -sh {device}"
        elif action == "mount":
            mount_point = context.get("mount_point")
            if not mount_point:
                return InstructionResult(
                    status=InstructionStatus.FAILED,
                    error="Mount point must be provided"
                )
            command = f"mount {device} {mount_point}"
        elif action == "unmount":
            command = f"umount {device}"
        else:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error=f"Invalid action: {action}"
            )
            
        return await self._run_command(command)

class BackupInstruction(BaseInstruction):
    """Perform system backup operations."""
    
    def __init__(self):
        super().__init__(
            name="backup",
            description="Perform system backup"
        )
        
    async def execute(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute backup command."""
        source = context.get("source")
        destination = context.get("destination")
        
        if not source or not destination:
            return InstructionResult(
                status=InstructionStatus.FAILED,
                error="Source and destination must be provided"
            )
            
        # Use rsync for reliable backup
        command = f"rsync -av --progress {source} {destination}"
        return await self._run_command(command) 