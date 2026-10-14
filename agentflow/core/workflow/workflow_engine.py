"""Workflow engine module."""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple, TYPE_CHECKING, cast
import logging
import ray
import time
from pydantic import BaseModel

from .workflow_types import (
    WorkflowConfig,
    WorkflowStep,
    WorkflowStepType,
    StepConfig,
    WorkflowStatus,
    WorkflowInstance,
    Message as WorkflowMessage
)
from .workflow_state import WorkflowStateManager
from .metrics import MetricsManager
from .exceptions import WorkflowExecutionError
from .workflow_executor import WorkflowExecutor
from ..agents.agent import Agent
from ..agents.agent_types import AgentType, AgentStatus
from .agent_config import AgentConfig
from .model_config import ModelConfig
from ..ell.types.message import Message, MessageRole, MessageType

if TYPE_CHECKING:
    from .workflow import WorkflowInstance

logger = logging.getLogger(__name__)

class WorkflowEngine:
    """Workflow engine for executing workflows."""

    def __init__(self, workflow_config: Optional[Union[WorkflowConfig, Dict[str, Any]]] = None):
        """Initialize workflow engine.
        
        Args:
            workflow_config: Optional workflow configuration
        """
        self.workflows = {}
        self._initialized = False
        self._ell = None
        self._isa_manager = None
        self._instruction_selector = None
        self._pending_tasks = {}  # Dictionary to store pending tasks
        self.agents = {}  # Map of agent IDs to agent instances
        self.state_manager = WorkflowStateManager()
        self.metrics = MetricsManager()
        self.status = "initialized"
        self.start_time = None
        self.end_time = None
        self.is_distributed = False  # Add is_distributed attribute
        self._default_agent_id: Optional[str] = None  # Store the default agent ID
        
        if workflow_config is not None:
            if isinstance(workflow_config, dict):
                self.workflow_config = WorkflowConfig(**workflow_config)
            elif isinstance(workflow_config, WorkflowConfig):
                self.workflow_config = workflow_config
            else:
                raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
        else:
            self.workflow_config = None
            
        # Initialize workflow definition as empty
        self.workflow_def = None

    @property
    def default_agent_id(self) -> Optional[str]:
        """Get the default agent ID."""
        return self._default_agent_id
        
    @default_agent_id.setter
    def default_agent_id(self, value: str) -> None:
        """Set the default agent ID."""
        self._default_agent_id = value
    
    async def initialize(self, workflow_def: Optional[Dict[str, Any]] = None, workflow_config: Optional[Union[WorkflowConfig, Dict[str, Any]]] = None, test_mode: bool = False) -> None:
        """Initialize workflow engine."""
        if workflow_def is not None:
            if not isinstance(workflow_def, dict):
                raise ValueError("workflow_def must be a dictionary or None")
            if "COLLABORATION" not in workflow_def or "WORKFLOW" not in workflow_def["COLLABORATION"]:
                raise WorkflowExecutionError("Empty workflow: no workflow steps defined in COLLABORATION.WORKFLOW")
            self.workflow_def = workflow_def
        else:
            self.workflow_def = {}

        if workflow_config:
            if not isinstance(workflow_config, (WorkflowConfig, dict)):
                raise ValueError("workflow_config must be an instance of WorkflowConfig, a dictionary, or None")
            if isinstance(workflow_config, dict):
                workflow_config = WorkflowConfig.model_validate(workflow_config, context={"test_mode": test_mode})
            self.workflow_config = workflow_config

            # Create a default agent for the workflow
            agent_config = AgentConfig(
                name="default_agent",
                type=AgentType.RESEARCH,
                model=ModelConfig.model_validate({"name": "gpt-4", "provider": "openai"}),
                workflow=workflow_config.model_dump()
            )
            agent = Agent(config=agent_config)
            agent.metadata["test_mode"] = test_mode  # Set test mode in agent metadata
            
            # Store the registration task to be awaited during initialization
            self._pending_registration = (agent, workflow_config)
        else:
            self.workflow_config = None

        # Initialize components
        from ..ell.integration import ELLIntegration
        self._ell = ELLIntegration()
        
        from .isa.isa_manager import ISAManager
        self._isa_manager = ISAManager()
        await self._isa_manager.initialize()
        
        from .instruction_selector import InstructionSelector
        self._instruction_selector = InstructionSelector()
        await self._instruction_selector.initialize()

        self._initialized = True

    async def register_workflow(self, agent: 'Agent', workflow_config: Optional[WorkflowConfig] = None) -> str:
        """Register workflow for an agent.
        
        Args:
            agent: Agent instance to register workflow for
            workflow_config: Optional workflow configuration to override agent's workflow
            
        Returns:
            str: ID of the registered workflow
            
        Raises:
            ValueError: If agent configuration is invalid or workflow is already registered
        """
        # Check for duplicate registration first
        if agent.id in self.workflows:
            raise ValueError(f"Agent {agent.id} already registered")
            
        if not agent.config or not isinstance(agent.config, (dict, AgentConfig)):
            raise ValueError("Agent must have a configuration")
            
        # ``agent.config`` may be a plain dict, a _DictConfigView or an
        # AgentConfig; use ``getattr`` so configs without the ``workflow``
        # field (agent-constructed configs) resolve to None instead of
        # raising AttributeError.
        agent_workflow = getattr(agent.config, "workflow", None)
        if not agent_workflow and not workflow_config:
            raise ValueError("Agent must have a workflow configuration")
            
        # Use provided workflow config or agent's workflow config
        workflow = workflow_config if workflow_config else agent_workflow
        
        # Convert to WorkflowConfig if needed
        if isinstance(workflow, dict):
            workflow = WorkflowConfig.model_validate(workflow)
        elif not isinstance(workflow, WorkflowConfig):
            raise ValueError("Invalid workflow configuration type")
            
        # Add default step if no steps are defined
        if not workflow.steps:
            workflow.steps = [
                WorkflowStep(
                    id="test-step-1",
                    name="test_step",
                    type=WorkflowStepType.AGENT,  # Always use AGENT type in test mode
                    description="Default test step for test mode",
                    config=StepConfig(strategy="standard")
                )
            ]
        # Set all steps to AGENT type if in test mode
        elif agent.metadata.get("test_mode"):
            for step in workflow.steps:
                step.type = WorkflowStepType.AGENT
            
        # Use agent's ID as workflow ID
        workflow_id = agent.id
        
        # Store workflow config and agent
        workflow.agent = agent  # Add agent reference to workflow config
        self.workflows[workflow_id] = workflow
        self.agents[workflow_id] = agent
        
        # If agent has a mock ELL, use it for the workflow engine too
        if hasattr(agent, '_ell') and agent._ell is not None:
            self._ell = agent._ell
            logger.debug("Using agent's ELL integration")
        
        return workflow_id

    async def execute_workflow(self, workflow_id: str, context: Union[Dict[str, Any], Message]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            workflow_id: ID of the workflow to execute
            context: Execution context
            
        Returns:
            Dict[str, Any]: Workflow execution results
            
        Raises:
            WorkflowExecutionError: If workflow is not found or execution fails
            TimeoutError: If execution times out
        """
        if workflow_id not in self.workflows:
            raise WorkflowExecutionError(f"No workflow registered for agent {workflow_id}")
            
        workflow = self.workflows[workflow_id]
        
        try:
            # Special handling for test mode
            is_test_context = False
            if isinstance(context, dict):
                is_test_context = context.get('test') is True
            elif hasattr(context, 'metadata'):
                # Safely check metadata for test mode
                is_test_context = (context.metadata or {}).get('test_mode', False)

            # Use asyncio.wait_for to enforce timeout
            if workflow.timeout:
                try:
                    result = await asyncio.wait_for(
                        workflow.execute(context),
                        timeout=workflow.timeout
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError(f"Workflow execution timed out after {workflow.timeout} seconds")
            else:
                result = await workflow.execute(context)

            return self._attach_result_summary(result)
        except WorkflowExecutionError as e:
            # Propagate the full, step-qualified error chain unchanged
            # (``Step <id> failed: <cause>``) so callers can attribute
            # failures to specific steps.
            raise
        except Exception as e:
            if isinstance(e, (WorkflowExecutionError, TimeoutError)):
                raise
            raise WorkflowExecutionError(f"Workflow execution failed: {str(e)}") from e

    @staticmethod
    def _attach_result_summary(result: Dict[str, Any]) -> Dict[str, Any]:
        """Attach a top-level ``content`` summary to a workflow result.

        Callers (and tests) expect ``result["content"]`` to carry the
        workflow's textual output.  The per-step payloads live under
        ``result["steps"][step_id]["result"]``; this helper walks the
        successful steps in order and concatenates their textual
        payloads (``result``/``content``/``output`` values) into the
        top-level summary without disturbing the original structure.
        """
        if not isinstance(result, dict) or "content" in result:
            return result

        steps = result.get("steps")
        if not isinstance(steps, dict):
            return result

        fragments: List[str] = []
        for step_id in sorted(steps):
            step_payload = steps[step_id]
            if not isinstance(step_payload, dict):
                continue
            step_result = step_payload.get("result")
            if not isinstance(step_result, dict):
                continue
            for key in ("content", "result", "output"):
                value = step_result.get(key)
                if isinstance(value, str) and value:
                    fragments.append(value)
                    break

        if fragments:
            result["content"] = "\n".join(fragments)
        return result

    async def cleanup(self) -> None:
        """Cleanup workflow engine resources.
        
        Raises:
            WorkflowExecutionError: If cleanup fails
        """
        try:
            # Clear workflows
            self.workflows.clear()
            
            # Cleanup instruction selector
            if self._instruction_selector:
                await self._instruction_selector.cleanup()
                
            # Cleanup ISA manager
            if self._isa_manager:
                await self._isa_manager.cleanup()
                
            # Cleanup ELL integration
            if self._ell and hasattr(self._ell, 'cleanup'):
                cleanup_method = getattr(self._ell, 'cleanup')
                if asyncio.iscoroutinefunction(cleanup_method):
                    await cleanup_method()
                else:
                    cleanup_method()
                
            # Clear component references
            self._instruction_selector = None
            self._isa_manager = None
            self._ell = None
            self._initialized = False
            
            # Update status
            self.status = "cleaned"
            self.end_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error during workflow engine cleanup: {str(e)}")
            raise WorkflowExecutionError(f"Failed to cleanup workflow engine: {str(e)}")

    async def _execute_step(self, step: WorkflowStep, context: Dict[str, Any], agent: 'Agent') -> Dict[str, Any]:
        """Execute a single workflow step.
        
        Args:
            step: Step to execute
            context: Execution context
            agent: Agent instance
            
        Returns:
            dict: Step execution result
            
        Raises:
            WorkflowExecutionError: If step execution fails
        """
        try:
            # Check for explicit failure trigger
            if step.config.params.get("should_fail") or context.get("should_fail"):
                raise WorkflowExecutionError("Step failed due to should_fail flag")

            # Execute step based on type
            if step.type == WorkflowStepType.TRANSFORM:
                if step.config.strategy == "custom" and "execute" in step.config.params:
                    execute_func = step.config.params["execute"]
                    if asyncio.iscoroutinefunction(execute_func):
                        result = await execute_func(context.get("data", {}))
                    else:
                        result = execute_func(context.get("data", {}))
                else:
                    result = context.get("data", {})
            else:
                result = await agent.process_message(context.get("message", ""))

            return {"content": result} if isinstance(result, str) else result

        except WorkflowExecutionError as e:
            # Extract the original error message
            error_str = str(e)
            
            # Format the error message
            error_msg = f"Error executing step {step.id}: {error_str}"
            raise WorkflowExecutionError(error_msg) from e

        except Exception as e:
            # For other exceptions, wrap them in WorkflowExecutionError
            error_msg = f"Error executing step {step.id}: {str(e)}"
            raise WorkflowExecutionError(error_msg) from e

    def get_workflow(self, agent_id: str) -> Optional[WorkflowConfig]:
        """Get workflow configuration for an agent.

        Args:
            agent_id: ID of the agent.

        Returns:
            Workflow configuration if found, None otherwise.
        """
        return self.workflows.get(agent_id)

    def get_agent(self, agent_id: str) -> Optional[Agent]:
        """Get agent instance.

        Args:
            agent_id: ID of the agent.

        Returns:
            Agent instance if found, None otherwise.
        """
        return self.agents.get(agent_id)

    async def execute_workflow_instance(self, instance: WorkflowInstance) -> Dict[str, Any]:
        """Execute workflow instance.
        
        Args:
            instance: Workflow instance to execute
            
        Returns:
            Dict[str, Any]: Workflow execution results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        if not instance.steps:
            raise ValueError("Workflow instance has no steps")
            
        instance.status = WorkflowStatus.RUNNING
        instance.updated_at = datetime.now()
        
        step_results = []  # Change to list to maintain order
        try:
            for step in instance.steps:
                try:
                    # Check for explicit failure trigger
                    if step.config.params.get("should_fail") or instance.context.get("should_fail"):
                        raise WorkflowExecutionError("Step failed due to should_fail flag")
                    
                    # For test mode or agent steps, use mock response
                    if instance.context.get("test_mode") or step.type == WorkflowStepType.AGENT:
                        if not instance.config or not instance.config.agent:
                            raise ValueError("No agent configured for workflow")
                        agent = self.agents.get(instance.config.agent.id)
                        if not agent:
                            raise ValueError("Agent not found in workflow engine")
                            
                        # Create test message
                        message = Message(
                            role=MessageRole.USER,
                            content=instance.context.get("message", "Test input"),
                            metadata={"test_mode": True}
                        )
                        
                        try:
                            # Process message through agent's ELL
                            if not agent._ell:
                                raise ValueError("No ELL integration available for test mode")
                            
                            # Use agent's ELL integration
                            result = await agent._ell.process_message(message)
                            logger.debug(f"Using agent's ELL integration for step {step.id}")
                            
                            # Handle different result types
                            if isinstance(result, Message):
                                content = result.content
                            elif isinstance(result, dict):
                                content = result.get("content", "Test response")
                            else:
                                content = str(result) if result is not None else "Test response"
                                
                            step_result = {
                                "id": step.id,
                                "type": str(step.type),
                                "status": "success",
                                "result": {"content": content},
                                "error": None
                            }
                            step_results.append(step_result)  # Append to list
                            
                            # Update instance result with step result
                            instance.result = {
                                "status": "success",
                                "steps": {step["id"]: step for step in step_results},  # Convert list to dictionary
                                "error": None,
                                "content": content
                            }
                        except Exception as e:
                            # Re-raise WorkflowExecutionError or wrap other exceptions
                            if isinstance(e, WorkflowExecutionError):
                                raise
                            raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}") from e
                    else:
                        try:
                            result = await step.execute(instance.context)
                            if not result:
                                raise ValueError(f"Step {step.id} returned no result")
                            step_result = {
                                "id": step.id,
                                "type": str(step.type),
                                "status": "success",
                                "result": result,
                                "error": None
                            }
                            step_results.append(step_result)  # Append to list
                        except Exception as e:
                            # Re-raise WorkflowExecutionError or wrap other exceptions
                            if isinstance(e, WorkflowExecutionError):
                                raise
                            raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}") from e
                except Exception as e:
                    step_result = {
                        "id": step.id,
                        "type": str(step.type),
                        "status": "failed",
                        "result": None,
                        "error": str(e)
                    }
                    step_results.append(step_result)  # Append to list
                    instance.status = WorkflowStatus.FAILED
                    instance.error = str(e)
                    raise WorkflowExecutionError(f"Step {step.id} failed: {str(e)}")  # Raise with proper error
                    
            # Update workflow status
            instance.status = WorkflowStatus.SUCCESS
            
            # Get content from the last step result
            last_step_result = step_results[-1]["result"]
            if isinstance(last_step_result, dict):
                content = last_step_result.get("content", "Test response")
            elif isinstance(last_step_result, Message):
                content = last_step_result.content
            else:
                content = str(last_step_result) if last_step_result is not None else "Test response"
            
            instance.result = {
                "status": "success",  # Use "success" for workflow status
                "steps": {step["id"]: step for step in step_results},  # Convert list to dictionary
                "error": None,
                "content": content
            }
                
        except Exception as e:
            instance.status = WorkflowStatus.FAILED
            instance.error = str(e)
            instance.result = {
                "status": "failed",
                "steps": {step["id"]: step for step in step_results},  # Convert list to dictionary
                "error": str(e),
                "content": ""
            }
            raise  # Re-raise the exception
            
        instance.updated_at = datetime.now()
        return instance.result

    async def create_workflow(self, name: str, config: Optional[WorkflowConfig] = None) -> 'WorkflowInstance':
        """Create a new workflow instance.
        
        Args:
            name: Workflow name
            config: Optional workflow configuration
            
        Returns:
            WorkflowInstance: Created workflow instance
        """
        # Import WorkflowInstance here to avoid circular import
        from .workflow import WorkflowInstance
        
        instance = WorkflowInstance(name=name, config=config)
        if config and hasattr(config, 'steps'):
            instance.steps = [
                WorkflowStep(
                    id=step.id,
                    name=step.name,
                    type=step.type,
                    description=step.description,
                    config=step.config.model_copy() if hasattr(step.config, 'model_copy') else step.config,  # Deep copy the config
                    dependencies=step.dependencies.copy() if step.dependencies else []
                )
                for step in config.steps
            ]
        return instance

    async def execute_workflow_async(self, workflow_id: str, context: Union[Dict[str, Any], Message]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            workflow_id: Workflow ID
            context: Workflow context
            
        Returns:
            Dict[str, Any]: Initial response with task ID
        """
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Convert Message to dict if needed
        if isinstance(context, Message):
            context_dict = {
                "content": context.content,
                "metadata": context.metadata,
                "type": context.type,
                "role": context.role,
                "timestamp": context.timestamp,
                "task_id": task_id,
                "workflow_id": workflow_id
            }
        else:
            context_dict = dict(context)
            context_dict["task_id"] = task_id
            context_dict["workflow_id"] = workflow_id
        
        # Store context for later execution
        self._pending_tasks[task_id] = {
            "context": context_dict,
            "status": "pending",
            "result": None,
            "error": None
        }
        
        # Start execution in background
        asyncio.create_task(self._execute_async(task_id, context_dict))
        
        return {
            "task_id": task_id,
            "status": "pending"
        }

    async def _execute_async(self, task_id: str, context: Dict[str, Any]) -> None:
        """Execute workflow asynchronously.
        
        Args:
            task_id: Task ID
            context: Execution context
        """
        try:
            # Get workflow ID from context
            workflow_id = context.get("workflow_id")
            if not workflow_id:
                raise WorkflowExecutionError("No workflow ID provided in context")
            
            # Execute workflow
            result = await self.execute_workflow(workflow_id, context)
            
            # Update task status
            self._pending_tasks[task_id].update({
                "status": "completed",
                "result": result,
                "error": None
            })
        except WorkflowExecutionError as e:
            # Extract the original error message
            error_str = str(e)
            
            # If the error is from a specific step, format it as expected
            if "Step" in error_str:
                # Find the step ID
                step_match = error_str.split("Step ", 1)
                if len(step_match) > 1:
                    step_id = step_match[1].split(" ", 1)[0]
                    # Extract error message from the original error
                    error_content = error_str.split(": ", 1)[1] if ": " in error_str else error_str
                    formatted_error = f"Error executing step {step_id}: Step {step_id} failed: {error_content}"
                    raise WorkflowExecutionError(formatted_error) from e
            
            # If not a step-specific error, re-raise as is
            raise

    async def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """Get task status.
        
        Args:
            task_id: Task ID
            
        Returns:
            Dict[str, Any]: Task status
        """
        if task_id not in self._pending_tasks:
            raise ValueError(f"Task {task_id} not found")
        
        return self._pending_tasks[task_id]