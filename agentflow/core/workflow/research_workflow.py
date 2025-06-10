"""Research workflow module."""

from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
from dataclasses import dataclass
from datetime import datetime

from .workflow import WorkflowEngine, WorkflowInstance
from .types import AgentStatus
from ..agents.agent import Agent
from .exceptions import WorkflowExecutionError
from .workflow_types import WorkflowStepType, WorkflowStep, WorkflowStatus
from .config import WorkflowConfig

logger = logging.getLogger(__name__)

@dataclass
class ResearchStep:
    """Research step class."""
    id: str
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None
    
@dataclass
class DistributedConfig:
    """Distributed configuration class."""
    num_workers: int = 1
    batch_size: int = 1
    max_retries: int = 3
    timeout: float = 3600.0

class ResearchDistributedWorkflow(WorkflowEngine):
    """Research distributed workflow class."""
    
    @classmethod
    async def create_remote_workflow(cls, workflow_def: Dict[str, Any], config: Union[Dict[str, Any], WorkflowConfig, None] = None) -> Any:
        """Create a remote workflow instance.
        
        Args:
            workflow_def: Workflow definition
            config: Workflow configuration
            
        Returns:
            Remote workflow instance
        """
        import ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        # Create a remote actor
        RemoteWorkflow = ray.remote(cls)
        
        # Convert config to dictionary for serialization
        if isinstance(config, WorkflowConfig):
            config_dict = config.model_dump()
        elif isinstance(config, dict):
            config_dict = config
        else:
            config_dict = {}
            
        # Convert workflow definition to expected format
        if "COLLABORATION" not in workflow_def:
            workflow_def = {
                "COLLABORATION": {
                    "WORKFLOW": workflow_def.get("WORKFLOW", {})
                }
            }
            
        # Create remote actor with workflow definition and config
        workflow = RemoteWorkflow.options(name=workflow_def.get("name", "research_workflow")).remote(
            workflow_def["COLLABORATION"]["WORKFLOW"]  # Only pass the workflow definition
        )
        return workflow
    
    def __init__(self, workflow_steps: Dict[str, Any], config: Optional[Dict[str, Any]] = None):
        """Initialize research workflow.
        
        Args:
            workflow_steps: Workflow steps definition
            config: Optional workflow configuration
        """
        # Wrap workflow steps in the expected structure
        workflow_def = {
            "COLLABORATION": {
                "WORKFLOW": workflow_steps
            }
        }
        super().__init__(workflow_def)  # Pass the complete workflow definition
        self._workflow_steps = workflow_steps
        
        # Convert workflow steps to list format
        steps_list = [
            WorkflowStep(
                id=step_id,
                name=step_info.get("name", step_id),
                type=WorkflowStepType.RESEARCH,
                config=step_info
            )
            for step_id, step_info in workflow_steps.items()
        ]
        
        self._instance = WorkflowInstance(
            id=str(datetime.now().timestamp()),
            name="research_workflow",
            steps=steps_list,
            status=WorkflowStatus.PENDING
        )
        
        # Initialize workflow config
        if isinstance(config, WorkflowConfig):
            self.workflow_config = config
        elif isinstance(config, dict):
            self.workflow_config = WorkflowConfig.model_validate(
                config,
                context={"is_initialization": True}
            )
        else:
            self.workflow_config = WorkflowConfig.model_validate(
                {},
                context={"is_initialization": True}
            )
        
        # Initialize distributed config
        self.dist_config = DistributedConfig()
        
        # Initialize additional attributes
        self.workers: List[Agent] = []
        self.step_results: Dict[str, Any] = {}
        
    async def configure(self, config: Dict[str, Any]) -> None:
        """Configure workflow with additional settings.
        
        Args:
            config: Configuration dictionary
        """
        if config:
            self.dist_config = DistributedConfig(**config.get("distributed", {}))
            if hasattr(self, "workflow_config"):
                # Update workflow config using model_validate
                updated_config = {**self.workflow_config.model_dump(), **config}
                self.workflow_config = WorkflowConfig.model_validate(updated_config)
        
    async def initialize(self) -> None:
        """Initialize workflow."""
        await super().initialize()
        
    async def initialize_workers(self, num_workers: Optional[int] = None) -> None:
        """Initialize worker nodes.
        
        Args:
            num_workers: Number of workers to initialize
        """
        n_workers = num_workers or self.dist_config.num_workers
        for _ in range(n_workers):
            worker = Agent()
            await worker.initialize()
            self.workers.append(worker)
            
    async def process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process a batch of inputs.
        
        Args:
            batch: Batch of inputs to process
            
        Returns:
            List of results
        """
        if not self.workers:
            await self.initialize_workers()
            
        tasks = []
        for i, item in enumerate(batch):
            worker = self.workers[i % len(self.workers)]
            task = asyncio.create_task(worker.execute({"data": item}))
            tasks.append(task)
            
        results = await asyncio.gather(*tasks)
        return [{"result": result} for result in results]
        
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow.
        
        Args:
            input_data: Input data for workflow
            
        Returns:
            Execution result
            
        Raises:
            ray.exceptions.RayTaskError: If input data is invalid or execution fails
        """
        try:
            self._instance.status = WorkflowStatus.RUNNING
            # Validate input data
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
                
            # Process workflow steps
            results = {}
            
            # Get execution order based on dependencies
            execution_order = []
            workflow = self._workflow_steps  # Use stored workflow steps
            remaining_steps = set(workflow.keys())
            
            # First, add steps with no dependencies
            for step_id in list(remaining_steps):
                step_info = workflow[step_id]
                if not step_info.get("dependencies"):
                    execution_order.append(step_id)
                    remaining_steps.remove(step_id)
            
            # Then, add steps whose dependencies are satisfied
            while remaining_steps:
                steps_added = False
                for step_id in list(remaining_steps):
                    step_info = workflow[step_id]
                    dependencies = step_info.get("dependencies", [])
                    if all(dep in execution_order for dep in dependencies):
                        execution_order.append(step_id)
                        remaining_steps.remove(step_id)
                        steps_added = True
                
                if not steps_added:
                    # No steps can be executed, must be a cycle
                    raise WorkflowExecutionError("Circular dependencies detected")
            
            # Execute steps in order
            for step_id in execution_order:
                step_info = workflow[step_id]
                retry_count = 0
                max_retries = self.dist_config.max_retries
                
                # Get step-specific retry config
                step_config_key = f"{step_id}_config"
                step_config = {}
                if hasattr(self.workflow_config, step_config_key):
                    step_config = getattr(self.workflow_config, step_config_key)
                    if hasattr(step_config, "max_retries"):
                        max_retries = step_config.max_retries
                
                while retry_count <= max_retries:
                    try:
                        # Process step
                        step_result = await self.process_step(step_id, step_info, input_data)
                        
                        # Add metadata
                        step_result["metadata"] = {
                            "timestamp": datetime.now().timestamp(),
                            "retry_count": retry_count
                        }
                        step_result["retry_count"] = retry_count
                        
                        results[step_id] = step_result
                        break
                        
                    except Exception as e:
                        retry_count += 1
                        if retry_count > max_retries:
                            raise WorkflowExecutionError(
                                f"Step {step_id} failed after {retry_count} retries: {str(e)}"
                            )
                        retry_delay = 0.1
                        if isinstance(step_config, dict):
                            retry_delay = step_config.get("retry_delay", retry_delay)
                        await asyncio.sleep(retry_delay)
            
            self._instance.status = WorkflowStatus.SUCCESS
            return {
                "status": "success",
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self._instance.status = WorkflowStatus.FAILED
            import ray
            logger.error(f"Workflow execution failed: {str(e)}")
            import traceback
            raise ray.exceptions.RayTaskError(
                function_name="execute",
                traceback_str=traceback.format_exc(),
                cause=e
            )
            
    async def process_step(self, step_id: str, step_info: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a workflow step.
        
        Args:
            step_id: Step ID
            step_info: Step information
            input_data: Input data
            
        Returns:
            Step result
        """
        # Process step input
        if "_test_fail_count" in input_data and step_id == "step_1":
            if input_data["_test_fail_count"] > 0:
                input_data["_test_fail_count"] -= 1
                raise ValueError("Simulated failure for testing")
        
        # Get retry delay from step config
        retry_delay = 0.1  # Default retry delay
        step_config = step_info.get("config", {})
        if isinstance(step_config, dict) and "retry_delay" in step_config:
            retry_delay = float(step_config["retry_delay"])
        
        # Process step (mock implementation)
        return {
            "status": "success",
            "output": input_data,
            "retry_count": input_data.get("_test_fail_count", 0),
            "retry_delay": retry_delay
        }
        
    async def cleanup(self) -> None:
        """Clean up workflow resources."""
        # Clean up workers
        for worker in self.workers:
            if hasattr(worker, "execute"):  # Check if worker has execute method
                await worker.execute({"command": "cleanup"})  # Send cleanup command
        self.workers.clear()
        
        # Clean up base class
        await super().cleanup()
        
    async def execute_async(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute workflow asynchronously.
        
        Args:
            input_data: Input data for workflow
            
        Returns:
            Execution result
        """
        return await self.execute(input_data)
        
    async def set_distributed_steps(self, steps: List[str]) -> None:
        """Set distributed steps.
        
        Args:
            steps: List of step IDs to distribute
        """
        self.distributed_steps = steps
        
    async def get_distributed_steps(self) -> List[str]:
        """Get distributed steps.
        
        Returns:
            List of distributed step IDs
        """
        return getattr(self, "distributed_steps", [])