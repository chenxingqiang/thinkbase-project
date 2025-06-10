import sys
import os
import uuid

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

import json
import logging
import ray
import time
import asyncio
import traceback
from typing import Dict, Any, Optional, List

import uvicorn
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, ValidationError

from agentflow.core.distributed_workflow import ResearchDistributedWorkflow, DistributedConfig
from agentflow.core.config import AgentConfig
from agentflow.core.workflow import WorkflowEngine
from agentflow.core.workflow_types import WorkflowConfig
from agentflow.agents.agent import Agent
from agentflow.core.model_config import ModelConfig
from agentflow.agents.agent_types import AgentType

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dictionary to store workflow tasks and their status
task_refs: Dict[str, Dict[str, Any]] = {}

# Conditionally initialize Ray
def initialize_ray():
    """
    Initialize Ray only if it's not already initialized.
    """
    try:
        if not ray.is_initialized():
            ray.init(
                logging_level=logging.INFO,
                dashboard_host='0.0.0.0',
                ignore_reinit_error=True
            )
            logger.info("Ray initialized with basic configuration")
    except Exception as e:
        logger.error(f"Failed to initialize Ray: {e}")
        raise

# Initialize Ray when the module is imported
initialize_ray()

# Create FastAPI app
app = FastAPI(
    title="Distributed Workflow API",
    description="API for executing distributed workflows",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Create a global workflow engine instance
workflow_engine = WorkflowEngine()

# Initialize workflow engine on startup
@app.on_event("startup")
async def startup_event():
    """Initialize workflow engine on startup."""
    await workflow_engine.initialize()
    workflow_engine._pending_tasks = {}  # Initialize pending tasks dictionary

class WorkflowRequest(BaseModel):
    """Workflow request model."""
    workflow: Dict[str, Any] = Field(description="Workflow configuration")
    config: Dict[str, Any] = Field(default_factory=dict, description="Workflow configuration options")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for workflow")
    
    @field_validator('workflow')
    def validate_workflow(cls, v):
        """Validate workflow configuration."""
        try:
            WorkflowConfig(**v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid workflow configuration: {str(e)}")

@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    """Handle Pydantic validation errors"""
    logger.error(f"Validation error: {exc}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    logger.error(f"Exception type: {type(exc)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "detail": str(exc),
            "type": str(type(exc))
        }
    )

@app.post("/workflow/execute")
async def execute_workflow(request: WorkflowRequest) -> Dict[str, Any]:
    """Execute workflow synchronously.
    
    Args:
        request: Workflow request
        
    Returns:
        Dict[str, Any]: Workflow execution results
    """
    try:
        # Create workflow config
        workflow_config = WorkflowConfig(**request.workflow)
        
        # Initialize workflow engine if needed
        if not workflow_engine._initialized:
            await workflow_engine.initialize()
            workflow_engine._pending_tasks = {}  # Initialize pending tasks dictionary
            
        # Create and register an agent for this workflow
        agent_config = AgentConfig(
            id=str(uuid.uuid4()),
            name=f"agent_{workflow_config.id}",
            type=AgentType.RESEARCH,
            model={
                "provider": "openai",
                "name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            workflow=workflow_config
        )
        agent = Agent(config=agent_config)
        await agent.initialize()
        
        # Set test mode in input data
        request.input_data["test_mode"] = True
        
        # Register workflow and execute
        await workflow_engine.register_workflow(agent, workflow_config)
        result = await workflow_engine.execute_workflow(agent.id, request.input_data)
        
        # Update status for consistency
        if result.get("status") == "success":
            result["status"] = "completed"
            
        # Ensure result is included
        if "result" not in result or result["result"] is None:
            # Get the last step's result
            if workflow_config.steps and len(workflow_config.steps) > 0:
                last_step = workflow_config.steps[-1]
                if "steps" in result and last_step.id in result["steps"]:
                    last_step_result = result["steps"][last_step.id]["result"]
                    if isinstance(last_step_result, dict):
                        result["result"] = last_step_result.get("result", last_step_result)
                    else:
                        result["result"] = last_step_result
            
            # If still no result, create a default one
            if "result" not in result or result["result"] is None:
                result["result"] = {
                    "content": result.get("content", ""),
                    "steps": result.get("steps", {})
                }
        
        return result
    except Exception as e:
        logger.error(f"Error executing workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/execute_async")
async def execute_workflow_async(request: WorkflowRequest) -> Dict[str, Any]:
    """Execute workflow asynchronously.
    
    Args:
        request: Workflow request
        
    Returns:
        Dict[str, Any]: Initial response with task ID
    """
    try:
        # Create workflow config
        workflow_config = WorkflowConfig(**request.workflow)
        
        # Initialize workflow engine if needed
        if not workflow_engine._initialized:
            await workflow_engine.initialize()
            
        # Create and register an agent for this workflow
        agent_config = AgentConfig(
            id=str(uuid.uuid4()),
            name=f"agent_{workflow_config.id}",
            type=AgentType.RESEARCH,
            model={
                "provider": "openai",
                "name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            workflow=workflow_config
        )
        agent = Agent(config=agent_config)
        await agent.initialize()
        await workflow_engine.register_workflow(agent, workflow_config)
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        
        # Store task info
        task_refs[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "start_time": time.time()
        }
        
        # Start execution in background
        async def execute_and_store():
            try:
                result = await workflow_engine.execute_workflow(agent.id, request.input_data)
                task_refs[task_id].update({
                    "status": "completed",
                    "result": result,
                    "end_time": time.time()
                })
            except Exception as e:
                task_refs[task_id].update({
                    "status": "failed",
                    "error": str(e),
                    "end_time": time.time()
                })
                
        # Create background task
        background_tasks = BackgroundTasks()
        background_tasks.add_task(execute_and_store)
        
        # Return initial response with task ID
        return {
            "status": "pending",
            "task_id": task_id,
            "message": "Workflow execution started"
        }
    except Exception as e:
        logger.error(f"Error executing workflow asynchronously: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflow/result/{result_ref}")
async def get_workflow_result(result_ref: str) -> Dict[str, Any]:
    """Get workflow result.
    
    Args:
        result_ref: Result reference ID
        
    Returns:
        Dict[str, Any]: Workflow result
    """
    if result_ref not in task_refs:
        raise HTTPException(status_code=404, detail="Result not found")
        
    task_info = task_refs[result_ref]
    
    if task_info["status"] == "error":
        raise HTTPException(status_code=500, detail=task_info["error"])
        
    if task_info["status"] == "pending":
        raise HTTPException(status_code=404, detail="Result not ready")
        
    retrieval_time = time.time() - task_info["start_time"]
    
    return {
        "status": task_info["status"],
        "result": task_info["result"],
        "retrieval_time": retrieval_time
    }

@app.get("/workflow/status/{task_id}")
async def get_workflow_status(task_id: str) -> Dict[str, Any]:
    """Get workflow execution status.
    
    Args:
        task_id: Task ID
        
    Returns:
        Dict[str, Any]: Task status
    """
    try:
        task = workflow_engine._pending_tasks.get(task_id)
        if not task:
            raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
        return task
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def _format_workflow_result(result: Any) -> Any:
    """Format workflow result to ensure consistent structure"""
    if result is None:
        return {'output': {'result': None}}
        
    if not isinstance(result, dict):
        return {'output': {'result': result}}
        
    # If result already has a properly structured output
    if 'output' in result and isinstance(result['output'], dict):
        return result
    
    # If result has step results, format them
    if any(key.startswith('step_') for key in result.keys()):
        formatted_steps = {}
        final_result = None
        for key, value in result.items():
            if key.startswith('step_'):
                step_num = int(key.split('_')[1])
                if isinstance(value, dict):
                    if 'output' in value:
                        formatted_steps[key] = value
                    else:
                        formatted_steps[key] = {'output': value}
                else:
                    formatted_steps[key] = {'output': {'result': value}}
                # Keep track of the last step's result
                final_result = formatted_steps[key]['output']
        # Return both the final result and all step results
        return {
            'output': final_result,
            'step_results': formatted_steps
        }
    
    # Otherwise, wrap the entire result
    return {'output': result}

# Create a Ray actor for handling async workflow execution
@ray.remote
class WorkflowActor:
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def execute_workflow(self, workflow_ref, input_ref):
        """Execute workflow and handle coroutines"""
        # Check if workflow_ref is a Ray object reference
        if isinstance(workflow_ref, ray.ObjectRef):
            workflow = ray.get(workflow_ref)
        else:
            workflow = workflow_ref

        # Check if input_ref is a Ray object reference
        if isinstance(input_ref, ray.ObjectRef):
            input_data = ray.get(input_ref)
        else:
            input_data = input_ref

        # Execute workflow
        result = workflow.execute(input_data)
        
        # If result is a coroutine, run it in the event loop
        if asyncio.iscoroutine(result):
            result = self.loop.run_until_complete(result)
            
        return result

@app.post("/workflow/validate")
async def validate_workflow(request: WorkflowRequest):
    """Validate workflow configuration"""
    try:
        return {"status": "valid"}
        
    except ValueError as e:
        logger.error(f"Workflow validation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Workflow validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def run_server(host='0.0.0.0', port=8000):
    """
    Run the FastAPI server for workflow execution.
    
    :param host: Host to bind the server
    :param port: Port to listen on
    """
    try:
        # Ensure Ray is initialized
        initialize_ray()
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),  # Output to console
                logging.FileHandler('workflow_server.log')  # Output to file
            ]
        )
        
        # Log server startup details
        logger.info(f"Starting workflow server on {host}:{port}")
        
        # Explicitly set the event loop policy for subprocess
        if sys.platform == 'darwin':  # macOS
            import asyncio
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
        
        # Run the server using uvicorn
        uvicorn.run(
            "agentflow.api.workflow_server:app", 
            host=host, 
            port=port, 
            reload=False,
            workers=1,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        # Write error to a file for debugging
        with open('server_startup_error.log', 'w') as f:
            f.write(f"Server startup error: {e}\n")
            traceback.print_exc(file=f)
        raise

if __name__ == "__main__":
    import sys
    import os
    import uvicorn

    # Default port
    port = 8000

    # Check if a port is provided as a command-line argument
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port number: {sys.argv[1]}. Using default port {port}")

    # Run the server
    run_server(host='0.0.0.0', port=port)
