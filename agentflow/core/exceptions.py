"""Exceptions module for AgentFlow."""

class AgentFlowError(Exception):
    """Base exception class for AgentFlow."""
    pass

class WorkflowEngineError(AgentFlowError):
    """Exception raised for errors in the workflow engine."""
    pass

class ValidationError(AgentFlowError):
    """Exception raised for validation errors."""
    pass

class WorkflowTimeoutError(AgentFlowError):
    """Exception raised when a workflow execution times out."""
    pass

class ConfigurationError(AgentFlowError):
    """Exception raised for configuration errors."""
    pass

class CommunicationError(AgentFlowError):
    """Exception raised for communication errors between agents."""
    pass

class ExecutionError(AgentFlowError):
    """Exception raised for errors during workflow execution."""
    pass

class WorkflowExecutionError(AgentFlowError):
    """Exception raised when workflow execution fails."""
    pass

class StepExecutionError(AgentFlowError):
    """Exception raised when step execution fails."""
    pass

class StepValidationError(AgentFlowError):
    """Exception raised when step validation fails."""
    pass

class StepTimeoutError(AgentFlowError):
    """Exception raised when step execution times out."""
    pass

class StepRetryError(AgentFlowError):
    """Exception raised when step retry mechanism fails."""
    pass

class WorkflowStateError(AgentFlowError):
    """Exception raised when workflow state is invalid."""
    pass

class WorkflowConfigError(AgentFlowError):
    """Exception raised when workflow configuration is invalid."""
    pass

class WorkflowInputError(AgentFlowError):
    """Exception raised when workflow input is invalid."""
    pass

class WorkflowOutputError(AgentFlowError):
    """Exception raised when workflow output is invalid."""
    pass

class WorkflowValidationError(Exception):
    """Exception raised when workflow validation fails."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

class AgentConfigError(Exception):
    """Agent configuration error."""
    pass

class AgentExecutionError(Exception):
    """Agent execution error."""
    pass

class WorkflowError(Exception):
    """Exception raised for workflow-related errors."""
    pass

class ContractViolationError(AgentFlowError):
    """Exception raised when a contract is violated."""
    pass
