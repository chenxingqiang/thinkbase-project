"""Main AgentFlow class."""

import asyncio
from typing import Dict, Any, Optional, List, Set
from .config import WorkflowConfig, AgentConfig, ModelConfig
from .workflow_executor import WorkflowExecutor
from .exceptions import WorkflowExecutionError

class AgentFlow:
    """AgentFlow class for managing workflows."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize AgentFlow.
        
        Args:
            config: Optional configuration dictionary.
        """
        self.config = config or {}
        self.workflows: Dict[str, WorkflowConfig] = {}
        self.active_workflows: Set[str] = set()
        self.executors: Dict[str, WorkflowExecutor] = {}
        self.agent_config: Optional[AgentConfig] = None
        self.model_config: Optional[ModelConfig] = None
        
        # Initialize configurations if provided
        if config:
            if "agent" in config:
                self.agent_config = AgentConfig(**config["agent"])
            if "model" in config:
                self.model_config = ModelConfig(**config["model"])

    def create_workflow(self, workflow_config: WorkflowConfig) -> str:
        """Create a workflow.
        
        Args:
            workflow_config: Workflow configuration.
            
        Returns:
            Workflow ID.
        """
        # Apply agent and model configs if available
        config_dict = workflow_config.model_dump()
        if self.agent_config:
            config_dict["agent"] = self.agent_config.model_dump()
        if self.model_config:
            config_dict["model"] = self.model_config.model_dump()
        workflow_config = WorkflowConfig(**config_dict)
            
        workflow_id = workflow_config.id
        self.workflows[workflow_id] = workflow_config
        self.executors[workflow_id] = WorkflowExecutor(workflow_config)
        return workflow_id

    async def execute_workflow(self, workflow_id: str, input_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Execute a workflow.
        
        Args:
            workflow_id: Workflow ID.
            input_data: Optional input data.
            
        Returns:
            Workflow execution results.
            
        Raises:
            WorkflowExecutionError: If workflow execution fails.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        try:
            self.active_workflows.add(workflow_id)
            executor = self.executors[workflow_id]
            result = await executor.execute(input_data)
            return result
        except Exception as e:
            self.active_workflows.remove(workflow_id)
            raise WorkflowExecutionError(str(e))

    async def stop_workflow(self, workflow_id: str) -> None:
        """Stop a workflow.
        
        Args:
            workflow_id: Workflow ID.
            
        Raises:
            ValueError: If workflow not found.
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow not found: {workflow_id}")
        
        if workflow_id in self.active_workflows:
            self.active_workflows.remove(workflow_id)

    def get_workflow_status(self, workflow_id: str) -> str:
        """Get workflow status.
        
        Args:
            workflow_id: Workflow ID.
            
        Returns:
            Workflow status.
            
        Raises:
            ValueError: If workflow not found.
        """
        if workflow_id not in self.executors:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        executor = self.executors.get(workflow_id)
        if executor is None:
            return "pending"  # No executor means workflow is created but not started
        
        # Return the actual state from the executor's state manager
        return str(executor.status)

    def list_workflows(self) -> List[str]:
        """List all workflows.
        
        Returns:
            List of workflow IDs.
        """
        return list(self.workflows.keys())
