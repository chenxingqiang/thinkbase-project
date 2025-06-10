"""Custom error classes for AgentFlow."""

class AgentFlowError(Exception):
    """Base error class for AgentFlow."""
    pass

class ValidationError(AgentFlowError):
    """Error raised when validation fails."""
    pass

class ResourceError(AgentFlowError):
    """Error raised when resource operations fail."""
    pass

class EntityNotFoundError(AgentFlowError):
    """Error raised when an entity is not found."""
    pass

class OperationError(AgentFlowError):
    """Error raised when an operation fails."""
    pass 