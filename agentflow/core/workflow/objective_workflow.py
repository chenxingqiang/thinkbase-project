"""Objective-based workflow implementation."""
from typing import Dict, Any, List, Optional
from datetime import datetime

from .base_workflow import BaseWorkflow
from .objective_handler import ObjectiveHandler, ObjectiveType, ObjectiveStatus
from .context_manager import ContextManager
from .exceptions import WorkflowError

class ObjectiveWorkflow(BaseWorkflow):
    """A workflow that is driven by objectives and their success criteria."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the objective workflow.
        
        Args:
            config: Configuration dictionary containing workflow settings
        """
        super().__init__(config)
        self.objective_handler = ObjectiveHandler(config)
        self.context_manager = ContextManager(config)
        
    def add_objective(
        self,
        objective_id: str,
        objective_type: ObjectiveType,
        description: str,
        success_criteria: List[Dict[str, Any]],
        priority: int = 1,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a new objective to the workflow.
        
        Args:
            objective_id: Unique identifier for the objective
            objective_type: Type of objective
            description: Detailed description
            success_criteria: List of success criteria
            priority: Priority level (1-5, 1 being highest)
            dependencies: List of dependent objective IDs
            metadata: Additional metadata
            
        Returns:
            Created objective
        """
        return self.objective_handler.create_objective(
            objective_id=objective_id,
            objective_type=objective_type,
            description=description,
            success_criteria=success_criteria,
            priority=priority,
            dependencies=dependencies,
            metadata=metadata
        )
        
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the objective-based workflow.
        
        Args:
            context: Workflow context
            
        Returns:
            Updated context with execution results
            
        Raises:
            WorkflowError: If workflow execution fails
        """
        try:
            # Update context with workflow start
            context['workflow_start_time'] = datetime.now().isoformat()
            context['workflow_type'] = 'objective'
            self.context_manager.update_context(context)
            
            # Get all pending objectives
            pending_objectives = self.objective_handler.get_objectives_by_status(
                ObjectiveStatus.PENDING
            )
            
            # Sort by priority and dependencies
            sorted_objectives = self._sort_objectives(pending_objectives)
            
            results = []
            for objective in sorted_objectives:
                # Set as current objective
                self.objective_handler.set_current_objective(objective.objective_id)
                
                # Update status to in progress
                self.objective_handler.update_objective_status(
                    objective.objective_id,
                    ObjectiveStatus.IN_PROGRESS
                )
                
                try:
                    # Execute objective-specific logic
                    objective_result = self._execute_objective(objective, context)
                    results.append(objective_result)
                    
                    # Validate objective completion
                    validation_result = self.objective_handler.validate_objective(
                        objective.objective_id
                    )
                    
                    # Update status based on validation
                    new_status = (
                        ObjectiveStatus.COMPLETED
                        if self._is_objective_successful(validation_result)
                        else ObjectiveStatus.FAILED
                    )
                    
                    self.objective_handler.update_objective_status(
                        objective.objective_id,
                        new_status
                    )
                    
                except Exception as e:
                    # Mark objective as failed
                    self.objective_handler.update_objective_status(
                        objective.objective_id,
                        ObjectiveStatus.FAILED,
                        metadata={"error": str(e)}
                    )
                    raise WorkflowError(f"Objective {objective.objective_id} failed: {e}")
                    
            # Update context with results
            context['workflow_end_time'] = datetime.now().isoformat()
            context['objective_results'] = results
            self.context_manager.update_context(context)
            
            return context
            
        except Exception as e:
            raise WorkflowError(f"Workflow execution failed: {e}")
            
    def _sort_objectives(self, objectives: List[Any]) -> List[Any]:
        """Sort objectives by priority and dependencies.
        
        Args:
            objectives: List of objectives to sort
            
        Returns:
            Sorted list of objectives
        """
        # First sort by priority
        objectives.sort(key=lambda x: x.priority)
        
        # Then handle dependencies
        sorted_objectives = []
        processed = set()
        
        def process_objective(objective):
            if objective.objective_id in processed:
                return
                
            # First process all dependencies
            for dep_id in objective.dependencies:
                dep_obj = next(
                    (obj for obj in objectives if obj.objective_id == dep_id),
                    None
                )
                if dep_obj and dep_obj.objective_id not in processed:
                    process_objective(dep_obj)
                    
            sorted_objectives.append(objective)
            processed.add(objective.objective_id)
            
        # Process all objectives
        for objective in objectives:
            process_objective(objective)
            
        return sorted_objectives
        
    def _execute_objective(
        self,
        objective: Any,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a single objective.
        
        Args:
            objective: Objective to execute
            context: Current context
            
        Returns:
            Execution results
            
        This method should be overridden by subclasses to implement
        objective-specific execution logic.
        """
        return {
            "objective_id": objective.objective_id,
            "type": objective.type.value,
            "status": "executed",
            "timestamp": datetime.now().isoformat()
        }
        
    def _is_objective_successful(self, validation_result: Dict[str, Any]) -> bool:
        """Check if objective was successful based on validation results.
        
        Args:
            validation_result: Results from objective validation
            
        Returns:
            True if successful, False otherwise
        """
        # This is a simple implementation that considers an objective
        # successful if all its criteria are validated
        return all(
            result.get("validated", False)
            for result in validation_result.get("criteria_results", [])
        )
