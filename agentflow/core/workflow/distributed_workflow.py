"""Distributed workflow implementation using Ray."""

from typing import Dict, Any, List, Optional, Union, Type, cast, NoReturn, TypeVar
from datetime import datetime
import asyncio
import logging
import ray
import uuid
import numpy as np
from dataclasses import dataclass, field
import time
from pydantic import BaseModel
from ray.actor import ActorHandle
from ray.util.actor_pool import ActorPool
from ray._raylet import ObjectRef

from .workflow_types import (
    WorkflowStep,
    WorkflowConfig,
    WorkflowStepType,
    StepConfig,
    WorkflowDefinition
)
from agentflow.core.workflow_state import WorkflowState
from agentflow.core.exceptions import WorkflowExecutionError
from agentflow.core.enums import WorkflowStatus
from agentflow.core.config import DistributedConfig
from .base_types import WorkflowStatus

# Initialize logger
logger = logging.getLogger(__name__)

T = TypeVar('T')

def initialize_ray() -> None:
    """Initialize Ray if not already initialized."""
    if not ray.is_initialized():
        try:
            ray.init(
                ignore_reinit_error=True,
                runtime_env={"pip": ["ray[default]>=2.5.0"]},
                num_cpus=2,  # Use 2 CPUs for distributed testing
                local_mode=False,  # Use distributed mode
                include_dashboard=False  # Disable dashboard for tests
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ray: {e}")
            raise

@ray.remote
class TransformStep:
    """Transform step in workflow."""
    
    def __init__(self):
        """Initialize transform step."""
        self.step_id: str = ""
        self.step_config: Dict[str, Any] = {}
        self.dependencies: List[str] = []
        self.initialized: bool = False
        self.scaler = None  # lazily typed as StandardScaler
        self.logger = logging.getLogger(f"{__name__}.TransformStep")
    
    async def initialize(self, step_id: str, config: Dict[str, Any], dependencies: Optional[List[str]] = None) -> bool:
        """Initialize step with configuration.
        
        Args:
            step_id: Step identifier
            config: Step configuration
            dependencies: List of step dependencies
            
        Returns:
            bool: True if initialization successful
        """
        self.step_id = step_id
        self.step_config = config
        self.dependencies = dependencies or []  # Use empty list as default
        
        # Initialize scaler based on config
        params = self.step_config.get('params', {})
        if params.get('method') == 'standard':
            from sklearn.preprocessing import StandardScaler
            self.scaler = StandardScaler(
                with_mean=params.get('with_mean', True),
                with_std=params.get('with_std', True)
            )
        
        self.initialized = True
        return True
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute transform step.
        
        Args:
            input_data: Input data for transformation
            
        Returns:
            Dict containing transformed data
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        try:
            if not self.initialized:
                raise WorkflowExecutionError("Step not initialized")
            
            # Get data from input
            data = input_data.get('data')
            if data is None:
                raise WorkflowExecutionError("No data provided")
            
            # Convert data to numpy array if needed
            if not isinstance(data, np.ndarray):
                data = np.array(data)
            
            # Reshape data if needed
            if len(data.shape) == 1:
                data = data.reshape(-1, 1)
            
            # Get transformation method
            method = self.step_config.get('params', {}).get('method')
            if method not in ['standard']:
                raise WorkflowExecutionError(f"Invalid transformation method: {method}")
            
            # Transform data
            if method == 'standard' and self.scaler is not None:
                transformed_data = self.scaler.fit_transform(data)
            else:
                raise WorkflowExecutionError(f"Transformation method {method} not properly initialized")
            
            return transformed_data.tolist()
            
        except Exception as e:
            error_msg = f"Failed to execute transform step {self.step_id}: {str(e)}"
            self.logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)
    
    async def health_check(self) -> bool:
        """Check if the actor is healthy."""
        return self.initialized

@ray.remote
class ResearchExecutionStep:
    """Research execution step in workflow."""
    
    def __init__(self):
        """Initialize research execution step."""
        self.step_id: str = ""
        self.step_config: Dict[str, Any] = {}
        self.dependencies: List[str] = []
        self.initialized: bool = False
        self.logger = logging.getLogger(f"{__name__}.ResearchExecutionStep")
    
    async def initialize(self, step_id: str, config: Dict[str, Any], dependencies: Optional[List[str]] = None) -> bool:
        """Initialize step with configuration.
        
        Args:
            step_id: Step identifier
            config: Step configuration
            dependencies: List of step dependencies
            
        Returns:
            bool: True if initialization successful
        """
        self.step_id = step_id
        self.step_config = config
        self.dependencies = dependencies or []  # Use empty list as default
        self.initialized = True
        return True
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research step.
        
        Args:
            input_data: Input data for research execution
            
        Returns:
            Dict containing research results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        try:
            if not self.initialized:
                raise WorkflowExecutionError("Step not initialized")
            
            # Get strategy from config
            strategy = self.step_config.get('strategy')
            if not strategy:
                raise WorkflowExecutionError("No strategy specified in step config")
            
            # Execute research strategy
            # For now, just return a mock result
            result = {
                "status": "success",
                "strategy": strategy,
                "output": f"Executed {strategy} on input data"
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to execute research step {self.step_id}: {str(e)}"
            self.logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def health_check(self) -> bool:
        """Check if the actor is healthy."""
        return self.initialized

class DistributedWorkflow:
    """Distributed workflow class."""

    def __init__(self, config: Optional[Union[Dict[str, Any], WorkflowConfig]] = None):
        """Initialize distributed workflow.
        
        Args:
            config: Workflow configuration
        """
        self.config = config
        self.steps: Dict[str, Dict[str, Any]] = {}
        self.state = WorkflowState(
            workflow_id=str(uuid.uuid4()),
            name="default_workflow",
            status=WorkflowStatus("pending")
        )
        self.workflow_config: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)

    def get_config_dict(self) -> Dict[str, Any]:
        """Get workflow configuration as dictionary."""
        if self.config is None:
            return {}
        
        if isinstance(self.config, dict):
            return self.config
        
        # Handle Pydantic models
        if hasattr(self.config, 'dict'):
            return self.config.dict()  # type: ignore
        
        # Fallback for other objects
        return {k: v for k, v in self.config.__dict__.items() if not k.startswith('_')}

    async def initialize_step(self, step: WorkflowStep) -> Optional[ray.ObjectRef]:
        """Initialize a workflow step."""
        try:
            # Create step actor based on type
            if step.type == WorkflowStepType.TRANSFORM:
                actor = TransformStep.remote()  # type: ignore
            elif step.type == WorkflowStepType.RESEARCH_EXECUTION:
                actor = ResearchExecutionStep.remote()  # type: ignore
            else:
                self.logger.error(f"Unsupported step type: {step.type}")
                return None

            # Initialize step with configuration
            config_dict = {}
            if hasattr(step.config, 'dict'):
                config_dict = step.config.dict()  # type: ignore
            elif isinstance(step.config, dict):
                config_dict = step.config
            
            dependencies = step.dependencies if step.dependencies else []
            
            # Call initialize on the actor
            success = await actor.initialize.remote(  # type: ignore
                step.id,
                config_dict,
                dependencies
            )
            
            if not success:
                self.logger.error(f"Failed to initialize step {step.id}")
                return None
                
            return actor
            
        except Exception as e:
            self.logger.error(f"Error initializing step {step.id}: {str(e)}")
            return None

    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        state = self._ensure_state()
        return {
            "workflow_id": state.workflow_id,
            "name": state.name,
            "status": str(state.status),
            "start_time": state.start_time,
            "end_time": state.end_time
        }

    async def _initialize_step_actors(self, steps: List[Dict[str, Any]]) -> None:
        """Initialize step actors.
        
        Args:
            steps: List of workflow steps
            
        Raises:
            WorkflowExecutionError: If step initialization fails
        """
        for step in steps:
            step_id = step.get('id')
            if step_id:
                self.steps[step_id] = {
                    'config': step.get('config', {}),
                    'dependencies': step.get('dependencies', []),
                    'status': 'pending',
                    'actor': None,
                    'result': None,
                    'error': None
                }
                
                # Create actor for step
                try:
                    # Create a new actor instance
                    actor = ResearchExecutionStep.options(name=step_id).remote()
                    # Initialize the actor
                    init_ref = actor.initialize.remote(  # type: ignore
                        step_id=step_id,
                        config=step.get('config', {}),
                        dependencies=step.get('dependencies', [])
                    )
                    init_result = ray.get(init_ref)  # Wait for initialization to complete
                    if init_result:
                        self.steps[step_id]['actor'] = actor
                    else:
                        raise WorkflowExecutionError(f"Failed to initialize step {step_id}")
                except Exception as e:
                    self.logger.error(f"Failed to initialize step {step_id}: {e}")
                    raise WorkflowExecutionError(f"Failed to initialize step {step_id}: {e}")

    def _ensure_state(self) -> WorkflowState:
        """Ensure workflow state exists and return it.
        
        Returns:
            WorkflowState: The current workflow state
        """
        if not self.state:
            self.state = WorkflowState(
                workflow_id=str(uuid.uuid4()),
                name="default_workflow",
                status=WorkflowStatus("pending")
            )
        return self.state

    def _initialize_step_configs(self, steps: List[Dict[str, Any]]) -> None:
        """Initialize configurations for workflow steps.
        
        Args:
            steps: List of step configurations
        """
        for step in steps:
            step_id = step.get('id')
            if not step_id:
                continue
                
            # Store step configuration
            self.steps[step_id] = {
                'config': step,
                'actor': None,
                'status': 'pending',
                'result': None,
                'error': None
            }

    async def initialize(self, workflow_def: Dict[str, Any], workflow_id: str) -> None:
        """Initialize workflow with definition.

        Args:
            workflow_def: Workflow definition
            workflow_id: Unique identifier for the workflow
            
        Raises:
            WorkflowExecutionError: If initialization fails
        """
        try:
            # Convert workflow definition if needed
            workflow_dict = workflow_def
            if not isinstance(workflow_def, dict) and hasattr(workflow_def, 'model_dump'):
                workflow_dict = workflow_def.model_dump()  # type: ignore

            # Get steps from workflow definition
            workflow = workflow_dict.get("COLLABORATION", {}).get("WORKFLOW", {})
            if not workflow:
                raise WorkflowExecutionError("Invalid workflow definition: missing COLLABORATION.WORKFLOW")
            
            steps = self._convert_workflow_to_steps(workflow)
            
            # Initialize step configurations
            self._initialize_step_configs(steps)
            
            # Initialize step actors
            await self._initialize_step_actors(steps)
            
            # Store workflow config
            self.workflow_config = workflow_dict
            
            # Initialize workflow state
            self.state = WorkflowState(
                workflow_id=workflow_id,
                name=workflow_dict.get("name", "default_workflow"),
                status=WorkflowStatus("pending")
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize workflow: {str(e)}"
            self.logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    def _convert_workflow_to_steps(self, workflow: Union[Dict[str, Any], List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Convert workflow definition to list of steps.
        
        Args:
            workflow: Workflow definition (either a dictionary or a list)
            
        Returns:
            List of step dictionaries
        """
        steps = []
        if isinstance(workflow, list):
            # If workflow is a list, each item is already a step definition
            for step_def in workflow:
                step = {
                    "id": step_def.get("id", str(uuid.uuid4())),
                    "name": step_def.get("name", ""),
                    "type": step_def.get("type", WorkflowStepType.RESEARCH_EXECUTION),
                    "config": {
                        "strategy": step_def.get("config", {}).get("strategy", "feature_engineering"),
                        "params": {
                            "max_retries": step_def.get("config", {}).get("max_retries", 3),
                            "retry_delay": step_def.get("config", {}).get("retry_delay", 1.0),
                            "retry_backoff": step_def.get("config", {}).get("retry_backoff", 2.0),
                            "timeout": step_def.get("config", {}).get("timeout", 30.0),
                            "batch_size": step_def.get("config", {}).get("batch_size", 10),
                            "num_workers": step_def.get("config", {}).get("num_workers", 4)
                        }
                    }
                }
                steps.append(step)
        else:
            # If workflow is a dictionary, convert each item to a step definition
            for step_id, step_def in workflow.items():
                step = {
                    "id": step_def.get("id", step_id),
                    "name": step_def.get("name", ""),
                    "type": step_def.get("type", WorkflowStepType.RESEARCH_EXECUTION),
                    "config": {
                        "strategy": step_def.get("config", {}).get("strategy", "feature_engineering"),
                        "params": {
                            "max_retries": step_def.get("config", {}).get("max_retries", 3),
                            "retry_delay": step_def.get("config", {}).get("retry_delay", 1.0),
                            "retry_backoff": step_def.get("config", {}).get("retry_backoff", 2.0),
                            "timeout": step_def.get("config", {}).get("timeout", 30.0),
                            "batch_size": step_def.get("config", {}).get("batch_size", 10),
                            "num_workers": step_def.get("config", {}).get("num_workers", 4)
                        }
                    }
                }
                steps.append(step)
        return steps
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with input data.
        
        Args:
            input_data: Input data for workflow execution
            
        Returns:
            Dict containing workflow results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        try:
            # Add logging for input validation
            self.logger.info(f"Validating input data: {input_data}")
            
            # Validate input data
            if not isinstance(input_data, dict):
                error_msg = f"Input data must be a dictionary, got {type(input_data)}"
                self.logger.error(error_msg)
                raise WorkflowExecutionError(error_msg)
            if "data" not in input_data:
                error_msg = "Input data must contain 'data' key"
                self.logger.error(error_msg)
                raise WorkflowExecutionError(error_msg)
            if input_data.get("data") is None:
                error_msg = "Input data 'data' value cannot be None"
                self.logger.error(error_msg)
                raise WorkflowExecutionError(error_msg)
            
            # Initialize state if needed
            state = self._ensure_state()

            # Update workflow status
            state.update_status(WorkflowStatus.RUNNING)
            if not hasattr(state, 'start_time') or state.start_time is None:
                state.start_time = datetime.fromtimestamp(time.time())
            
            # Execute steps in sequence
            results = {}
            steps_executed = 0
            for step_id, step in self.steps.items():
                actor = step.get('actor')
                if not actor:
                    continue
                    
                try:
                    # Execute step
                    step['status'] = 'running'
                    result = await actor.execute.remote(input_data)
                    
                    # Store result
                    step['status'] = 'completed'
                    step['result'] = result
                    results[step_id] = result
                    steps_executed += 1
                    
                except Exception as e:
                    error_msg = f"Failed to execute step {step_id}: {str(e)}"
                    self.logger.error(error_msg)
                    step['status'] = 'failed'
                    step['error'] = error_msg
                    raise WorkflowExecutionError(error_msg)
            
            # Update workflow status
            if not hasattr(self, 'state') or self.state is None:
                self.state = WorkflowState(
                    workflow_id=str(uuid.uuid4()),
                    name="default_workflow",
                    status=WorkflowStatus("pending")
                )
            self.state.status = WorkflowStatus.COMPLETED
            if not hasattr(self.state, 'end_time') or self.state.end_time is None:
                self.state.end_time = datetime.fromtimestamp(time.time())
            
            return {
                "status": "success",
                "metrics": {
                    "steps_executed": steps_executed
                },
                "results": results
            }
            
        except Exception as e:
            error_msg = f"Failed to execute workflow: {str(e)}"
            self.logger.error(error_msg)
            if not hasattr(self, 'state') or self.state is None:
                self.state = WorkflowState(
                    workflow_id=str(uuid.uuid4()),
                    name="default_workflow",
                    status=WorkflowStatus("failed")
                )
            self.state.status = WorkflowStatus.FAILED
            self.state.end_time = datetime.fromtimestamp(time.time())
            raise WorkflowExecutionError(error_msg)
            
    async def get_status(self) -> Dict[str, Any]:
        """Get current workflow status.
        
        Returns:
            Dict containing workflow status information
        """
        return {
            'workflow_id': self.state.workflow_id,
            'name': self.state.name,
            'status': self.state.status,
            'start_time': self.state.start_time,
            'end_time': self.state.end_time,
            'steps': {
                step_id: {
                    'status': step['status'],
                    'error': step['error']
                }
                for step_id, step in self.steps.items()
            }
        }
        
    @classmethod
    async def create_remote_workflow(cls, workflow_def: Dict[str, Any], workflow_config: Union[dict, 'WorkflowConfig'], workflow_id: str) -> 'ray.actor.ActorHandle':
        """Create a remote workflow actor.
        
        Args:
            workflow_def: Workflow definition
            workflow_config: Workflow configuration
            workflow_id: Unique identifier for the workflow
            
        Returns:
            Ray actor handle for the workflow
            
        Raises:
            WorkflowExecutionError: If workflow creation fails
        """
        try:
            # Initialize Ray if needed
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Create workflow actor
            workflow = ray.remote(cls).remote(config=workflow_config)  # type: ignore
            
            # Initialize workflow
            init_ref = workflow.initialize.remote(workflow_def, workflow_id)  # type: ignore
            await ray.get(init_ref)  # Wait for initialization to complete
            
            return workflow
            
        except Exception as e:
            error_msg = f"Failed to create remote workflow: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for workflow execution
            
        Returns:
            Dict containing workflow results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        return await self.execute(input_data)

class ResearchDistributedWorkflow(DistributedWorkflow):
    """Research distributed workflow class."""

    def __init__(self, name: Optional[str] = None, config: Optional[Union[Dict[str, Any], WorkflowConfig]] = None, steps: Optional[List[Dict[str, Any]]] = None):
        """Initialize research distributed workflow.
        
        Args:
            name: Workflow name
            config: Workflow configuration
            steps: List of workflow steps
        """
        super().__init__(config)
        self.name = name or "default_research_workflow"
        self.research_context: Dict[str, Any] = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize state properly
        self.state = WorkflowState(
            workflow_id=str(uuid.uuid4()),
            name=self.name,
            status=WorkflowStatus("pending")
        )
        
        # Initialize steps if provided
        if steps:
            self.steps = {}
            for step in steps:
                step_id = step.get('id')
                if step_id:
                    self.steps[step_id] = {
                        'config': step.get('config', {}),
                        'dependencies': step.get('dependencies', []),
                        'status': 'pending',
                        'actor': None,
                        'result': None,
                        'error': None
                    }
                    
                    # Create actor for step
                    try:
                        # Create a new actor instance
                        actor = ResearchExecutionStep.options(name=step_id).remote()
                        # Initialize the actor
                        init_ref = actor.initialize.remote(  # type: ignore
                            step_id=step_id,
                            config=step.get('config', {}),
                            dependencies=step.get('dependencies', [])
                        )
                        init_result = ray.get(init_ref)  # Wait for initialization to complete
                        if init_result:
                            self.steps[step_id]['actor'] = actor
                        else:
                            raise WorkflowExecutionError(f"Failed to initialize step {step_id}")
                    except Exception as e:
                        self.logger.error(f"Failed to initialize step {step_id}: {e}")
                        raise WorkflowExecutionError(f"Failed to initialize step {step_id}: {e}")

    async def initialize(self, workflow_def: Dict[str, Any], workflow_id: str) -> None:
        """Initialize workflow with definition.
        
        Args:
            workflow_def: Workflow definition
            workflow_id: Unique identifier for the workflow
            
        Raises:
            WorkflowExecutionError: If initialization fails
        """
        try:
            # Convert workflow definition if needed
            workflow_dict = workflow_def
            if not isinstance(workflow_def, dict) and hasattr(workflow_def, 'model_dump'):
                workflow_dict = workflow_def.model_dump()

            # Get steps from workflow definition
            workflow = workflow_dict.get("COLLABORATION", {}).get("WORKFLOW", [])
            if not workflow:
                raise WorkflowExecutionError("Invalid workflow definition: missing COLLABORATION.WORKFLOW")
            
            # Convert workflow to steps
            steps = self._convert_workflow_to_steps(workflow)
            
            # Initialize step configurations
            self._initialize_step_configs(steps)
            
            # Initialize step actors
            await self._initialize_step_actors(steps)
            
            # Store workflow config
            self.workflow_config = workflow_dict
            
            # Update workflow state
            self.state = WorkflowState(
                workflow_id=workflow_id,
                name=self.name,
                status=WorkflowStatus("pending")
            )
            
        except Exception as e:
            error_msg = f"Failed to initialize workflow: {str(e)}"
            self.logger.error(error_msg)
            self.state.status = WorkflowStatus.FAILED
            raise WorkflowExecutionError(error_msg)

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow with input data."""
        try:
            # Validate input data
            if not isinstance(input_data, dict):
                error_msg = f"Input data must be a dictionary, got {type(input_data)}"
                self.logger.error(error_msg)
                raise WorkflowExecutionError(error_msg)

            # Update workflow status
            self.state.status = WorkflowStatus.RUNNING
            if not hasattr(self.state, 'start_time') or self.state.start_time is None:
                self.state.start_time = datetime.fromtimestamp(time.time())
            
            # Execute steps in sequence, respecting dependencies
            results = {}
            steps_executed = 0
            
            # Get ordered steps based on dependencies
            step_ids = list(self.steps.keys())
            for step_id in step_ids:
                step = self.steps[step_id]
                actor = step.get('actor')
                if not actor:
                    continue
                
                try:
                    # Prepare input data with dependencies
                    step_input = {"data": input_data.get("data")}
                    step_config = step.get('config', {})
                    dependencies = step_config.get('dependencies', [])
                    
                    if dependencies:
                        dep_results = {}
                        for dep_id in dependencies:
                            if dep_id not in results:
                                raise WorkflowExecutionError(f"Dependent step {dep_id} not executed yet")
                            dep_results[dep_id] = results[dep_id].get("data")
                        step_input["dependencies"] = dep_results
                    
                    # Execute step
                    step['status'] = WorkflowStatus.RUNNING.value
                    result = await actor.execute.remote(step_input)
                    
                    # Store result
                    step['status'] = WorkflowStatus.COMPLETED.value
                    step['result'] = result
                    results[step_id] = result
                    steps_executed += 1
                    
                except Exception as e:
                    error_msg = f"Failed to execute step {step_id}: {str(e)}"
                    self.logger.error(error_msg)
                    step['status'] = WorkflowStatus.FAILED.value
                    step['error'] = error_msg
                    self.state.status = WorkflowStatus.FAILED
                    self.state.end_time = datetime.fromtimestamp(time.time())
                    raise WorkflowExecutionError(error_msg)
            
            # Update workflow status
            self.state.status = WorkflowStatus.COMPLETED
            self.state.end_time = datetime.fromtimestamp(time.time())
            
            # Get final result from last step
            final_step_id = step_ids[-1]
            final_result = results[final_step_id] if results else None
            
            return {
                "status": WorkflowStatus.COMPLETED.value,
                "metrics": {
                    "steps_executed": steps_executed
                },
                "results": results,
                "data": final_result.get("data") if final_result else None
            }
            
        except Exception as e:
            error_msg = f"Failed to execute workflow: {str(e)}"
            self.logger.error(error_msg)
            self.state.status = WorkflowStatus.FAILED
            self.state.end_time = datetime.fromtimestamp(time.time())
            raise WorkflowExecutionError(error_msg)

    async def get_status(self) -> Dict[str, Any]:
        """Get current workflow status."""
        return {
            'workflow_id': self.state.workflow_id,
            'name': self.state.name,
            'status': self.state.status.value if isinstance(self.state.status, WorkflowStatus) else self.state.status,
            'start_time': self.state.start_time,
            'end_time': self.state.end_time,
            'steps': {
                step_id: {
                    'status': step['status'],
                    'error': step.get('error', '')
                }
                for step_id, step in self.steps.items()
            }
        }
        
    @classmethod
    async def create_remote_workflow(cls, workflow_def: Dict[str, Any], workflow_config: Union[dict, 'WorkflowConfig'], workflow_id: str) -> 'ray.actor.ActorHandle':
        """Create a remote workflow actor.
        
        Args:
            workflow_def: Workflow definition
            workflow_config: Workflow configuration
            workflow_id: Unique identifier for the workflow
            
        Returns:
            Ray actor handle for the workflow
            
        Raises:
            WorkflowExecutionError: If workflow creation fails
        """
        try:
            # Initialize Ray if needed
            if not ray.is_initialized():
                ray.init(ignore_reinit_error=True)
            
            # Create workflow actor
            workflow = ray.remote(cls).remote(config=workflow_config)  # type: ignore
            
            # Initialize workflow
            init_ref = workflow.initialize.remote(workflow_def, workflow_id)  # type: ignore
            await ray.get(init_ref)  # Wait for initialization to complete
            
            return workflow
            
        except Exception as e:
            error_msg = f"Failed to create remote workflow: {str(e)}"
            logger.error(error_msg)
            raise WorkflowExecutionError(error_msg)

    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for workflow execution
            
        Returns:
            Dict containing workflow results
            
        Raises:
            WorkflowExecutionError: If execution fails
        """
        return await self.execute(input_data)

@ray.remote
class WorkflowActor:
    """Workflow actor class."""
    
    def __init__(self):
        """Initialize workflow actor."""
        self.workflow: Optional[ResearchDistributedWorkflow] = None
        self.initialized = False
        
    async def initialize(self, workflow_def: Dict[str, Any], workflow_config: Dict[str, Any], workflow_id: str) -> bool:
        """Initialize workflow actor.
        
        Args:
            workflow_def: Workflow definition
            workflow_config: Workflow configuration
            workflow_id: Workflow ID
            
        Returns:
            bool: True if initialization successful
        """
        try:
            self.workflow = ResearchDistributedWorkflow()
            await self.workflow.initialize(workflow_def, workflow_id)
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Failed to initialize workflow actor: {e}")
            raise
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            input_data: Input data for workflow
            
        Returns:
            Dict[str, Any]: Workflow execution results
            
        Raises:
            WorkflowExecutionError: If workflow is not initialized
        """
        if not self.initialized or self.workflow is None:
            raise WorkflowExecutionError("Workflow actor not initialized")
        return await self.workflow.execute(input_data) 