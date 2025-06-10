"""Memory management module for AgentFlow."""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import time
import json

@dataclass
class MemoryEntry:
    """Entry in memory store."""
    key: str
    value: Any
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class WorkingMemory:
    """Memory implementation for storing and retrieving data during agent operations."""
    
    def __init__(self):
        """Initialize working memory."""
        self._memory: Dict[str, Any] = {}
        
    async def store(self, key: str, value: Any) -> None:
        """Store data in memory.
        
        Args:
            key: Key to store data under.
            value: Data to store.
        """
        self._memory[key] = value
        
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve data from memory.
        
        Args:
            key: Key to retrieve data for.
            
        Returns:
            Stored data if found, None otherwise.
        """
        return self._memory.get(key)
        
    async def clear(self) -> None:
        """Clear all stored data."""
        self._memory.clear()
        
    async def list_keys(self) -> list[str]:
        """List all stored keys.
        
        Returns:
            List of stored keys.
        """
        return list(self._memory.keys())
        
    async def dump(self) -> str:
        """Dump memory contents to JSON string.
        
        Returns:
            JSON string of memory contents.
        """
        return json.dumps(self._memory, indent=2)
        
    async def load(self, data: str) -> None:
        """Load memory contents from JSON string.
        
        Args:
            data: JSON string to load.
        """
        self._memory = json.loads(data)
        
    async def delete(self, key: str) -> None:
        """Delete data from memory.
        
        Args:
            key: Key to delete data for.
        """
        if key in self._memory:
            del self._memory[key]
        
    async def retrieve_all(self, pattern: str = "*") -> list[Any]:
        """Retrieve all data from memory matching a pattern.
        
        Args:
            pattern: Optional pattern to match keys. 
                     Uses simple wildcard matching.
                     
        Returns:
            List of all matching values.
        """
        import fnmatch
        
        return [
            value for key, value in self._memory.items() 
            if fnmatch.fnmatch(key, pattern)
        ]

class GlobalMemory:
    """Global memory for persistent storage across instructions."""
    
    def __init__(self):
        self._store: Dict[str, MemoryEntry] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    async def store(
        self,
        key: str,
        value: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Store a value with optional metadata."""
        self._store[key] = MemoryEntry(
            key=key,
            value=value,
            metadata=metadata or {}
        )
    
    async def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve a value and its metadata."""
        entry = self._store.get(key)
        return entry.value if entry else None
    
    async def get_metadata(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a stored value."""
        entry = self._store.get(key)
        return entry.metadata if entry else None
    
    async def clear(self) -> None:
        """Clear all entries from memory."""
        self._store.clear()
        self._metadata.clear() 