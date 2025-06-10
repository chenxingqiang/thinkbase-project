"""
Task Automation Example using AgentISA Framework.

This example demonstrates how to build a task automation system that:
1. Manages task workflows
2. Handles dependencies between tasks
3. Optimizes task execution
4. Learns from execution patterns
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
import json
import logging
from datetime import datetime
import os

from agentflow.core.isa.instruction import Instruction, InstructionType, TypeConstraint
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskResult:
    """Result of task execution."""
    success: bool
    output: Any
    execution_time: float
    error_message: Optional[str] = None

class TaskContext:
    """Maintains task execution context."""
    
    def __init__(self):
        self.variables: Dict[str, Any] = {}
        self.completed_tasks: Set[str] = set()
        self.execution_history: List[Dict[str, Any]] = []
        
    def set_variable(self, name: str, value: Any):
        """Set context variable."""
        self.variables[name] = value
        
    def get_variable(self, name: str) -> Optional[Any]:
        """Get context variable."""
        return self.variables.get(name)
        
    def mark_completed(self, task_id: str):
        """Mark task as completed."""
        self.completed_tasks.add(task_id)
        
    def is_completed(self, task_id: str) -> bool:
        """Check if task is completed."""
        return task_id in self.completed_tasks
        
    def add_to_history(self, entry: Dict[str, Any]):
        """Add entry to execution history."""
        self.execution_history.append(entry)

class FileProcessingInstruction(Instruction):
    """Instruction for file processing tasks."""
    
    def __init__(self):
        super().__init__(
            id="file_processing",
            name="file_processing",
            type=InstructionType.RESOURCE_MANAGEMENT,
            params={
                "input_path": TypeConstraint(base_type=str),
                "output_path": TypeConstraint(base_type=str),
                "operation": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=TaskContext)
            },
            description="Process files with various operations"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        input_path = input_params["input_path"]
        output_path = input_params["output_path"]
        operation = input_params["operation"]
        context = input_params["context"]
        
        start_time = datetime.now()
        
        try:
            # Validate paths
            if not os.path.exists(input_path):
                raise ValueError(f"Input path does not exist: {input_path}")
                
            # Process file based on operation
            if operation == "copy":
                result = await self._copy_file(input_path, output_path)
            elif operation == "move":
                result = await self._move_file(input_path, output_path)
            elif operation == "delete":
                result = await self._delete_file(input_path)
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
            # Update context
            context.set_variable("last_processed_file", output_path)
            context.mark_completed(f"file_op_{operation}_{input_path}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": TaskResult(
                    success=True,
                    output=result,
                    execution_time=execution_time
                )
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "result": TaskResult(
                    success=False,
                    output=None,
                    execution_time=execution_time,
                    error_message=str(e)
                )
            }
            
    async def _copy_file(self, src: str, dst: str) -> str:
        """Copy file from source to destination."""
        # Add file copy logic
        return f"Copied {src} to {dst}"
        
    async def _move_file(self, src: str, dst: str) -> str:
        """Move file from source to destination."""
        # Add file move logic
        return f"Moved {src} to {dst}"
        
    async def _delete_file(self, path: str) -> str:
        """Delete file."""
        # Add file deletion logic
        return f"Deleted {path}"

class DataTransformationInstruction(Instruction):
    """Instruction for data transformation tasks."""
    
    def __init__(self):
        super().__init__(
            id="data_transform",
            name="data_transform",
            type=InstructionType.STATE_MANAGEMENT,
            params={
                "data": TypeConstraint(base_type=dict),
                "transform_type": TypeConstraint(base_type=str),
                "context": TypeConstraint(base_type=TaskContext)
            },
            description="Transform data in various ways"
        )
        
    async def execute(self, input_params: Dict[str, Any]) -> Dict[str, Any]:
        data = input_params["data"]
        transform_type = input_params["transform_type"]
        context = input_params["context"]
        
        start_time = datetime.now()
        
        try:
            # Apply transformation
            if transform_type == "flatten":
                result = self._flatten_data(data)
            elif transform_type == "normalize":
                result = self._normalize_data(data)
            else:
                raise ValueError(f"Unknown transformation: {transform_type}")
                
            # Update context
            context.set_variable("last_transform_result", result)
            context.mark_completed(f"transform_{transform_type}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return {
                "result": TaskResult(
                    success=True,
                    output=result,
                    execution_time=execution_time
                )
            }
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return {
                "result": TaskResult(
                    success=False,
                    output=None,
                    execution_time=execution_time,
                    error_message=str(e)
                )
            }
            
    def _flatten_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten nested dictionary."""
        # Add flattening logic
        return {"flattened": str(data)}
        
    def _normalize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize data values."""
        # Add normalization logic
        return {"normalized": str(data)}

class TaskAutomationSystem:
    """Main task automation system using AgentISA framework."""
    
    def __init__(self):
        # Configure AgentISA
        config = AgentISAConfig(
            enable_learning=True,
            enable_optimization=True,
            min_pattern_support=2,
            min_success_rate=0.8
        )
        self.agent_isa = AgentISA(config)
        self.context = TaskContext()
        
        # Initialize instructions
        asyncio.create_task(self._init_instructions())
        
    async def _init_instructions(self):
        """Initialize and register instructions."""
        self.file_instruction = FileProcessingInstruction()
        self.transform_instruction = DataTransformationInstruction()
        
        self.file_id = await self.agent_isa.register_instruction(
            self.file_instruction
        )
        self.transform_id = await self.agent_isa.register_instruction(
            self.transform_instruction
        )
        
    async def process_file(self, input_path: str, output_path: str, 
                          operation: str) -> TaskResult:
        """Process a file."""
        result = await self.agent_isa.execute_instruction(
            self.file_id,
            {
                "input_path": input_path,
                "output_path": output_path,
                "operation": operation,
                "context": self.context
            }
        )
        return result["result"]
        
    async def transform_data(self, data: Dict[str, Any], 
                           transform_type: str) -> TaskResult:
        """Transform data."""
        result = await self.agent_isa.execute_instruction(
            self.transform_id,
            {
                "data": data,
                "transform_type": transform_type,
                "context": self.context
            }
        )
        return result["result"]
        
    async def execute_workflow(self, workflow: List[Dict[str, Any]]) -> List[TaskResult]:
        """Execute a workflow of tasks."""
        results = []
        
        for task in workflow:
            if task["type"] == "file":
                result = await self.process_file(
                    task["input_path"],
                    task["output_path"],
                    task["operation"]
                )
            elif task["type"] == "transform":
                result = await self.transform_data(
                    task["data"],
                    task["transform_type"]
                )
            else:
                raise ValueError(f"Unknown task type: {task['type']}")
                
            results.append(result)
            
            # Learn from execution
            await self.agent_isa.learn_from_history()
            
        return results
        
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.agent_isa.get_instruction_stats()

async def main():
    """Example usage of TaskAutomationSystem."""
    # Initialize system
    system = TaskAutomationSystem()
    
    # Define example workflow
    workflow = [
        {
            "type": "file",
            "input_path": "input.txt",
            "output_path": "output.txt",
            "operation": "copy"
        },
        {
            "type": "transform",
            "data": {"key": "value"},
            "transform_type": "flatten"
        },
        {
            "type": "file",
            "input_path": "output.txt",
            "output_path": "final.txt",
            "operation": "move"
        }
    ]
    
    # Execute workflow
    print("Executing workflow...")
    results = await system.execute_workflow(workflow)
    
    # Print results
    for i, result in enumerate(results, 1):
        print(f"\nTask {i} Result:")
        print(f"Success: {result.success}")
        print(f"Output: {result.output}")
        print(f"Execution Time: {result.execution_time:.2f}s")
        if result.error_message:
            print(f"Error: {result.error_message}")
            
    # Print stats
    print("\nSystem Stats:")
    print(json.dumps(system.get_stats(), indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 