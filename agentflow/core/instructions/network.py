"""Network operation instructions."""

from typing import Dict, Any, Optional, cast
import socket
import ipaddress
import time
import logging
from dataclasses import dataclass

from .base import (
    BaseInstruction,
    ExecutionContext,
    ExecutionResult,
    InstructionType,
    InstructionMetadata
)

logger = logging.getLogger(__name__)

@dataclass
class NetworkMetrics:
    """Network operation metrics."""
    latency: float = 0.0
    bytes_sent: int = 0
    bytes_received: int = 0
    packets_sent: int = 0
    packets_received: int = 0
    errors: int = 0

class NetworkInstruction(BaseInstruction):
    """Base class for network operations."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize network instruction.
        
        Args:
            name: Instruction name
            description: Optional instruction description
            instruction_type: Optional instruction type
            metadata: Optional instruction metadata
        """
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.RESOURCE_MANAGEMENT,
            metadata=metadata or InstructionMetadata(
                capabilities={"network_access"},
                tags={name.lower(), "network"}
            )
        )
        self._network_metrics = NetworkMetrics()
    
    def _validate_address(self, address: str) -> bool:
        """Validate IP address or hostname."""
        try:
            # Try parsing as IP address
            ipaddress.ip_address(address)
            return True
        except ValueError:
            # Try validating as hostname
            try:
                socket.gethostbyname(address)
                return True
            except socket.error:
                return False
    
    def _validate_port(self, port: int) -> bool:
        """Validate port number."""
        return 0 < port < 65536
    
    async def _run_command(self, command: str) -> ExecutionResult:
        """Run network command safely."""
        start_time = time.time()
        
        try:
            # Execute command implementation here
            # This is a placeholder - actual implementation would use subprocess or similar
            
            execution_time = time.time() - start_time
            self.update_metrics(True, execution_time)
            
            return ExecutionResult(
                success=True,
                output={"command": command, "result": "Command executed successfully"},
                errors=[],
                state_updates={},
                metrics={
                    "execution_time": execution_time,
                    "network_metrics": {
                        "latency": self._network_metrics.latency,
                        "bytes_sent": self._network_metrics.bytes_sent,
                        "bytes_received": self._network_metrics.bytes_received,
                        "packets_sent": self._network_metrics.packets_sent,
                        "packets_received": self._network_metrics.packets_received,
                        "errors": self._network_metrics.errors
                    }
                }
            )
            
        except Exception as e:
            logger.error(f"Network command failed: {str(e)}")
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                state_updates={},
                metrics={"execution_time": execution_time}
            )

class NetworkStatusInstruction(NetworkInstruction):
    """Show network interfaces and status."""
    
    def __init__(self):
        super().__init__(
            name="ifconfig",
            description="Show network interfaces and status",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute ifconfig/ip command."""
        # Use ip command as it's more modern
        command = "ip addr show"
        return await self._run_command(command)

class NetworkPingInstruction(NetworkInstruction):
    """Ping a host."""
    
    def __init__(self):
        super().__init__(
            name="ping",
            description="Ping a host",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute ping command."""
        host = context.params.get("host")
        count = context.params.get("count", 4)
        
        if not host:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Host not provided"}]
            )
        
        if not self._validate_address(host):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid host: {host}"}]
            )
        
        command = f"ping -c {count} {host}"
        return await self._run_command(command)

class NetworkTraceRouteInstruction(NetworkInstruction):
    """Trace route to host."""
    
    def __init__(self):
        super().__init__(
            name="traceroute",
            description="Trace route to host",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute traceroute command."""
        host = context.params.get("host")
        
        if not host:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Host not provided"}]
            )
        
        if not self._validate_address(host):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid host: {host}"}]
            )
        
        command = f"traceroute {host}"
        return await self._run_command(command)

class NetworkPortScanInstruction(NetworkInstruction):
    """Scan network ports."""
    
    def __init__(self):
        super().__init__(
            name="nmap",
            description="Scan network ports",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute nmap command."""
        host = context.params.get("host")
        ports = context.params.get("ports")
        
        if not host:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Host not provided"}]
            )
        
        if not self._validate_address(host):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid host: {host}"}]
            )
        
        if ports:
            # Validate ports
            try:
                port_list = [int(p) for p in ports.split(",")]
                if not all(self._validate_port(p) for p in port_list):
                    return ExecutionResult(
                        success=False,
                        output={},
                        errors=[{"error": "Invalid port numbers"}]
                    )
                port_spec = f"-p {ports}"
            except ValueError:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": "Invalid port specification"}]
                )
        else:
            port_spec = ""
        
        command = f"nmap {port_spec} {host}"
        return await self._run_command(command)

class NetworkDNSLookupInstruction(NetworkInstruction):
    """DNS lookup."""
    
    def __init__(self):
        super().__init__(
            name="nslookup",
            description="DNS lookup",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute nslookup command."""
        host = context.params.get("host")
        
        if not host:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Host not provided"}]
            )
        
        if not self._validate_address(host):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid host: {host}"}]
            )
        
        command = f"nslookup {host}"
        return await self._run_command(command)

class NetworkSSHInstruction(NetworkInstruction):
    """SSH connection."""
    
    def __init__(self):
        super().__init__(
            name="ssh",
            description="SSH connection",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT,
            metadata=InstructionMetadata(
                capabilities={"network_access", "ssh_access"},
                tags={"ssh", "network", "remote"}
            )
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute SSH command."""
        host = context.params.get("host")
        user = context.params.get("user")
        port = context.params.get("port", 22)
        command = context.params.get("command")
        
        if not host or not user:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "Host and user must be provided"}]
            )
        
        if not self._validate_address(host):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid host: {host}"}]
            )
        
        if not self._validate_port(port):
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": f"Invalid port: {port}"}]
            )
        
        ssh_command = f"ssh -p {port} {user}@{host}"
        if command:
            ssh_command += f" '{command}'"
        
        return await self._run_command(ssh_command)

class NetworkCurlInstruction(NetworkInstruction):
    """HTTP request using curl."""
    
    def __init__(self):
        super().__init__(
            name="curl",
            description="HTTP request using curl",
            instruction_type=InstructionType.RESOURCE_MANAGEMENT,
            metadata=InstructionMetadata(
                capabilities={"network_access", "http_access"},
                tags={"curl", "http", "network"}
            )
        )
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute curl command."""
        url = context.params.get("url")
        method = context.params.get("method", "GET")
        headers = context.params.get("headers", {})
        data = context.params.get("data")
        
        if not url:
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": "URL not provided"}]
            )
        
        command = ["curl", "-X", method]
        
        # Add headers
        for key, value in headers.items():
            command.extend(["-H", f"'{key}: {value}'"])
        
        # Add data for POST/PUT
        if data and method in ["POST", "PUT"]:
            command.extend(["-d", f"'{data}'"])
        
        command.append(url)
        
        return await self._run_command(" ".join(command)) 