"""Objective handling instruction implementations."""

from typing import Dict, Any, List, Optional, Set, Union, cast
from dataclasses import dataclass, field
import asyncio
import logging
import json
import numpy as np

from .base import (
    BaseInstruction,
    InstructionType,
    InstructionMetadata,
    ExecutionContext,
    ExecutionResult
)

logger = logging.getLogger(__name__)

@dataclass
class ObjectiveMetrics:
    """Metrics for objective handling."""
    objectives_created: int = 0
    objectives_completed: int = 0
    objectives_failed: int = 0
    average_completion_time: float = 0.0
    success_rate: float = 0.0

@dataclass
class ObjectiveConfig:
    """Configuration for objective handling."""
    max_depth: int = 5
    max_children: int = 10
    timeout_seconds: float = 3600.0
    min_success_rate: float = 0.8
    validation_required: bool = True

class ObjectiveDecompositionInstruction(BaseInstruction):
    """Instruction for decomposing objectives into sub-tasks."""
    
    def __init__(self, name: str, config: ObjectiveConfig):
        """Initialize objective decomposition instruction.
        
        Args:
            name: Instruction name
            config: Objective configuration
        """
        super().__init__(name)
        self.config = config
        self.objectives: Dict[str, Dict[str, Any]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute objective decomposition.
        
        Args:
            context: Execution context containing:
                - objective: Objective to decompose
                
        Returns:
            ExecutionResult: Decomposition results
        """
        try:
            # Extract objective
            objective = context.params.get("objective")
            if not objective:
                return ExecutionResult(
                    success=False,
                    output={"error": "No objective provided"},
                    errors=[{"error": "No objective provided"}]
                )
            
            # Decompose objective
            decomposition = await self._decompose_objective(objective)
            
            # Count objectives
            objectives_count = self._count_objectives(decomposition)
            
            return ExecutionResult(
                success=True,
                output={
                    "decomposition": decomposition,
                    "metrics": {
                        "objectives_count": objectives_count,
                        "max_depth": self.config.max_depth,
                        "max_children": self.config.max_children
                    }
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}]
            )
    
    async def _decompose_objective(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose objective into sub-objectives.
        
        Args:
            objective: Objective to decompose
            
        Returns:
            Dictionary containing decomposition results
        """
        # Check if objective should be decomposed
        if not self._should_decompose(objective):
            return objective
        
        # Create sub-objectives
        sub_objectives = self._create_sub_objectives(objective)
        
        # Extract dependencies
        dependencies = self._extract_dependencies(sub_objectives)
        
        return {
            "id": objective.get("id"),
            "type": objective.get("type"),
            "description": objective.get("description"),
            "sub_objectives": sub_objectives,
            "dependencies": dependencies,
            "depth": objective.get("depth", 0)
        }
    
    def _should_decompose(self, objective: Dict[str, Any]) -> bool:
        """Check if objective should be decomposed."""
        # Check depth limit
        depth = objective.get("depth", 0)
        if depth >= self.config.max_depth:
            return False
        
        # Check complexity
        complexity = self._calculate_complexity(objective)
        return complexity > 1.0
    
    def _create_sub_objectives(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create sub-objectives from objective."""
        # Extract components
        components = self._extract_components(objective)
        
        # Create sub-objectives
        sub_objectives = []
        for i, component in enumerate(components):
            sub_objective = {
                "id": f"{objective.get('id', '')}_sub_{i}",
                "type": component.get("type", "task"),
                "description": component.get("description", ""),
                "requirements": component.get("requirements", {}),
                "depth": objective.get("depth", 0) + 1
            }
            sub_objectives.append(sub_objective)
            
        return sub_objectives
    
    def _extract_components(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract components from objective."""
        # Simple component extraction based on requirements
        requirements = objective.get("requirements", {})
        components = []
        
        for req_type, req_value in requirements.items():
            component = {
                "type": req_type,
                "description": f"Handle {req_type} requirement",
                "requirements": {req_type: req_value}
            }
            components.append(component)
            
        return components[:self.config.max_children]
    
    def _calculate_complexity(self, objective: Dict[str, Any]) -> float:
        """Calculate complexity of objective."""
        # Simple complexity based on requirements
        requirements = objective.get("requirements", {})
        return len(requirements)
    
    def _extract_dependencies(self, sub_objectives: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract dependencies between sub-objectives."""
        dependencies = []
        
        # Check each pair of sub-objectives
        for i, obj_a in enumerate(sub_objectives):
            for j, obj_b in enumerate(sub_objectives[i+1:], i+1):
                if self._has_dependency(obj_a, obj_b):
                    dependencies.append({
                        "from": obj_a["id"],
                        "to": obj_b["id"],
                        "type": "requires"
                    })
                
        return dependencies
    
    def _has_dependency(self, obj_a: Dict[str, Any], obj_b: Dict[str, Any]) -> bool:
        """Check if there is a dependency between objectives."""
        # Simple dependency check based on requirements
        req_a = set(obj_a.get("requirements", {}).keys())
        req_b = set(obj_b.get("requirements", {}).keys())
        return bool(req_a & req_b)
    
    def _count_objectives(self, decomposition: Dict[str, Any]) -> int:
        """Count total number of objectives in decomposition."""
        count = 1  # Count current objective
        
        # Count sub-objectives recursively
        for sub in decomposition.get("sub_objectives", []):
            count += self._count_objectives(sub)
            
        return count

class ObjectiveValidationInstruction(BaseInstruction):
    """Instruction for validating objectives and their completion."""
    
    def __init__(self,
                 name: str,
                 objective_config: ObjectiveConfig,
                 validation_rules: Dict[str, Dict[str, Any]],
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            instruction_type=InstructionType.OBJECTIVE_HANDLING,
            metadata=metadata,
            description="Validate objectives and their completion"
        )
        self.objective_config = objective_config
        self.validation_rules = validation_rules
        self.metrics = ObjectiveMetrics()
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute objective validation."""
        try:
            # Get objective and completion data
            objective = context.params.get("objective")
            completion_data = context.params.get("completion_data")
            
            if not objective or not completion_data:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Missing objective or completion data"}]
                )
            
            # Validate completion
            validation_result = await self._validate_completion(
                objective,
                completion_data,
                context
            )
            
            # Update metrics
            if validation_result.get("success", False):
                self.metrics.objectives_completed += 1
            else:
                self.metrics.objectives_failed += 1
            
            self.metrics.success_rate = (
                self.metrics.objectives_completed /
                (self.metrics.objectives_completed + self.metrics.objectives_failed)
            )
            
            return ExecutionResult(
                success=validation_result.get("success", False),
                output=validation_result,
                state_updates={"validation_result": validation_result},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Objective validation failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _validate_completion(self,
                                 objective: Dict[str, Any],
                                 completion_data: Dict[str, Any],
                                 context: ExecutionContext) -> Dict[str, Any]:
        """Validate objective completion."""
        validation_results = []
        
        # Get applicable rules
        rules = self._get_applicable_rules(objective)
        
        # Apply each rule
        for rule in rules:
            result = await self._apply_validation_rule(
                rule,
                objective,
                completion_data,
                context
            )
            validation_results.append(result)
        
        # Calculate overall success
        success = all(r.get("success", False) for r in validation_results)
        
        return {
            "success": success,
            "objective_id": objective.get("id"),
            "validation_results": validation_results,
            "timestamp": context.metadata.get("timestamp")
        }
    
    def _get_applicable_rules(self,
                            objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get validation rules applicable to objective."""
        applicable_rules = []
        
        for rule_id, rule in self.validation_rules.items():
            if self._is_rule_applicable(rule, objective):
                applicable_rules.append(rule)
        
        return applicable_rules
    
    def _is_rule_applicable(self,
                           rule: Dict[str, Any],
                           objective: Dict[str, Any]) -> bool:
        """Check if validation rule is applicable to objective."""
        # Check objective type
        if "objective_types" in rule:
            if objective.get("type") not in rule["objective_types"]:
                return False
        
        # Check requirements
        if "required_fields" in rule:
            if not all(field in objective for field in rule["required_fields"]):
                return False
        
        return True
    
    async def _apply_validation_rule(self,
                                   rule: Dict[str, Any],
                                   objective: Dict[str, Any],
                                   completion_data: Dict[str, Any],
                                   context: ExecutionContext) -> Dict[str, Any]:
        """Apply validation rule to completion data."""
        try:
            # Extract required data
            required_data = self._extract_required_data(
                rule,
                objective,
                completion_data
            )
            
            # Apply validation logic
            validation_func = rule.get("validation_function")
            if not validation_func:
                raise ValueError("No validation function specified")
            
            result = validation_func(required_data)
            
            return {
                "rule_id": rule.get("id"),
                "success": result.get("success", False),
                "details": result.get("details", {}),
                "timestamp": context.metadata.get("timestamp")
            }
            
        except Exception as e:
            logger.error(f"Rule application failed: {str(e)}")
            return {
                "rule_id": rule.get("id"),
                "success": False,
                "details": {"error": str(e)},
                "timestamp": context.metadata.get("timestamp")
            }
    
    def _extract_required_data(self,
                             rule: Dict[str, Any],
                             objective: Dict[str, Any],
                             completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data required for validation."""
        required_data = {}
        
        # Extract from objective
        for field in rule.get("required_objective_fields", []):
            required_data[f"objective_{field}"] = objective.get(field)
        
        # Extract from completion data
        for field in rule.get("required_completion_fields", []):
            required_data[f"completion_{field}"] = completion_data.get(field)
        
        return required_data

class ObjectiveProgressTrackingInstruction(BaseInstruction):
    """Instruction for tracking objective progress."""
    
    def __init__(self,
                 name: str,
                 objective_config: ObjectiveConfig,
                 metadata: Optional[InstructionMetadata] = None):
        super().__init__(
            name=name,
            instruction_type=InstructionType.OBJECTIVE_HANDLING,
            metadata=metadata,
            description="Track objective progress"
        )
        self.objective_config = objective_config
        self.metrics = ObjectiveMetrics()
        self.progress_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute progress tracking."""
        try:
            # Get objective and progress update
            objective = context.params.get("objective")
            progress_update = context.params.get("progress_update")
            
            if not objective or not progress_update:
                return ExecutionResult(
                    success=False,
                    output=None,
                    errors=[{"error": "Missing objective or progress update"}]
                )
            
            # Update progress
            progress_result = await self._update_progress(
                objective,
                progress_update,
                context
            )
            
            # Calculate metrics
            self._update_metrics(progress_result)
            
            return ExecutionResult(
                success=True,
                output=progress_result,
                state_updates={"progress_tracking": progress_result},
                metrics=self.metrics.__dict__
            )
            
        except Exception as e:
            logger.error(f"Progress tracking failed: {str(e)}")
            return ExecutionResult(
                success=False,
                output=None,
                errors=[{"error": str(e)}],
                metrics=self.metrics.__dict__
            )
    
    async def _update_progress(self,
                             objective: Dict[str, Any],
                             progress_update: Dict[str, Any],
                             context: ExecutionContext) -> Dict[str, Any]:
        """Update objective progress."""
        objective_id = cast(str, objective.get("id"))
        if not objective_id:
            raise ValueError("Objective ID is required")
        
        # Initialize history if needed
        if objective_id not in self.progress_history:
            self.progress_history[objective_id] = []
        
        # Add progress update
        self.progress_history[objective_id].append({
            "timestamp": context.metadata.get("timestamp"),
            "update": progress_update
        })
        
        # Calculate current progress
        current_progress = self._calculate_progress(objective)
        
        # Check completion
        is_completed = current_progress.get("progress", 0) >= 1.0
        if is_completed:
            completion_time = self._calculate_completion_time(objective_id)
            current_progress["completion_time"] = completion_time
        
        return current_progress
    
    def _calculate_progress(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate current progress for objective."""
        objective_id = cast(str, objective.get("id"))
        if not objective_id:
            raise ValueError("Objective ID is required")
        
        history = self.progress_history.get(objective_id, [])
        
        if not history:
            return {
                "objective_id": objective_id,
                "progress": 0.0,
                "status": "not_started",
                "updates": []
            }
        
        # Calculate progress
        latest_update = history[-1]["update"]
        progress = latest_update.get("progress", 0.0)
        
        # Determine status
        status = "in_progress"
        if progress >= 1.0:
            status = "completed"
        elif progress == 0.0:
            status = "not_started"
        
        return {
            "objective_id": objective_id,
            "progress": progress,
            "status": status,
            "updates": history
        }
    
    def _calculate_completion_time(self, objective_id: str) -> float:
        """Calculate time taken to complete objective."""
        history = self.progress_history.get(objective_id, [])
        if not history:
            return 0.0
        
        start_time = cast(float, history[0]["timestamp"])
        end_time = cast(float, history[-1]["timestamp"])
        
        return end_time - start_time
    
    def _update_metrics(self, progress_result: Dict[str, Any]):
        """Update tracking metrics."""
        if progress_result.get("status") == "completed":
            self.metrics.objectives_completed += 1
            
            # Update average completion time
            completion_time = progress_result.get("completion_time", 0.0)
            current_total = (
                self.metrics.average_completion_time *
                (self.metrics.objectives_completed - 1)
            )
            self.metrics.average_completion_time = (
                (current_total + completion_time) /
                self.metrics.objectives_completed
            ) 