"""Objective handler for managing agent goals and success criteria in the CO-STAR framework."""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from .validators import ValidatorRegistry, ValidationResult
from .persistence import PersistenceFactory, BasePersistence

class ObjectiveType(Enum):
    """Objective types supported by the system"""
    TASK = "task"              # 单个任务目标
    WORKFLOW = "workflow"      # 工作流程目标
    CONVERSATION = "conversation"  # 对话交互目标
    ANALYSIS = "analysis"      # 分析类目标
    CREATION = "creation"      # 创作类目标
    OPTIMIZATION = "optimization"  # 优化类目标

class ObjectiveStatus(Enum):
    """Objective execution status"""
    PENDING = "pending"        # 待执行
    IN_PROGRESS = "in_progress"  # 执行中
    COMPLETED = "completed"    # 已完成
    FAILED = "failed"          # 执行失败
    BLOCKED = "blocked"        # 被阻塞

@dataclass
class Success_Criteria:
    """Success criteria for objectives"""
    criteria_type: str
    description: str
    threshold: Optional[float] = None
    validation_method: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Objective:
    """Objective definition"""
    objective_id: str
    type: ObjectiveType
    description: str
    success_criteria: List[Success_Criteria]
    status: ObjectiveStatus = ObjectiveStatus.PENDING
    priority: int = 1
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

class ObjectiveHandler:
    """Handler for managing objectives in the CO-STAR framework"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the objective handler
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.objectives: Dict[str, Objective] = {}
        self.current_objective: Optional[str] = None
        
        # Initialize persistence
        persistence_config = config.get("persistence", {})
        persistence_type = persistence_config.get("type", "file")
        self.persistence = PersistenceFactory.create_persistence(
            persistence_type,
            **persistence_config.get("config", {})
        )
        
    def create_objective(
        self,
        objective_id: str,
        objective_type: Union[ObjectiveType, str],
        description: str,
        success_criteria: List[Dict[str, Any]],
        priority: int = 1,
        dependencies: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Objective:
        """Create a new objective
        
        Args:
            objective_id: Unique identifier for the objective
            objective_type: Type of objective
            description: Detailed description of the objective
            success_criteria: List of success criteria
            priority: Priority level (1-5, 1 being highest)
            dependencies: List of dependent objective IDs
            metadata: Additional metadata
            
        Returns:
            Created objective
            
        Raises:
            ValueError: If objective_id already exists
        """
        if objective_id in self.objectives:
            raise ValueError(f"Objective with ID {objective_id} already exists")
            
        if isinstance(objective_type, str):
            objective_type = ObjectiveType(objective_type.lower())
            
        # Convert success criteria dictionaries to Success_Criteria objects
        criteria_objects = [
            Success_Criteria(
                criteria_type=crit["type"],
                description=crit["description"],
                threshold=crit.get("threshold"),
                validation_method=crit.get("validation_method"),
                metadata=crit.get("metadata", {})
            )
            for crit in success_criteria
        ]
        
        objective = Objective(
            objective_id=objective_id,
            type=objective_type,
            description=description,
            success_criteria=criteria_objects,
            priority=priority,
            dependencies=dependencies or [],
            metadata=metadata or {}
        )
        
        self.objectives[objective_id] = objective
        return objective
        
    def get_objective(self, objective_id: str) -> Optional[Objective]:
        """Get objective by ID
        
        Args:
            objective_id: Objective identifier
            
        Returns:
            Objective if found, None otherwise
        """
        return self.objectives.get(objective_id)
        
    def update_objective_status(
        self,
        objective_id: str,
        status: Union[ObjectiveStatus, str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Objective]:
        """Update objective status
        
        Args:
            objective_id: Objective identifier
            status: New status
            metadata: Additional metadata to update
            
        Returns:
            Updated objective if found, None otherwise
        """
        objective = self.objectives.get(objective_id)
        if not objective:
            return None
            
        if isinstance(status, str):
            status = ObjectiveStatus(status.lower())
            
        objective.status = status
        objective.updated_at = datetime.now().isoformat()
        
        if metadata:
            objective.metadata.update(metadata)
            
        return objective
        
    def set_current_objective(self, objective_id: Optional[str]) -> Optional[Objective]:
        """Set the current active objective
        
        Args:
            objective_id: Objective identifier or None to clear
            
        Returns:
            Current objective if set, None otherwise
        """
        if objective_id is None:
            self.current_objective = None
            return None
            
        if objective_id not in self.objectives:
            return None
            
        self.current_objective = objective_id
        return self.objectives[objective_id]
        
    def get_current_objective(self) -> Optional[Objective]:
        """Get current active objective
        
        Returns:
            Current objective if set, None otherwise
        """
        if not self.current_objective:
            return None
        return self.objectives.get(self.current_objective)
        
    def get_objectives_by_status(self, status: Union[ObjectiveStatus, str]) -> List[Objective]:
        """Get objectives by status
        
        Args:
            status: Status to filter by
            
        Returns:
            List of objectives with matching status
        """
        if isinstance(status, str):
            status = ObjectiveStatus(status.lower())
            
        return [obj for obj in self.objectives.values() if obj.status == status]
        
    def get_dependent_objectives(self, objective_id: str) -> List[Objective]:
        """Get objectives that depend on the given objective
        
        Args:
            objective_id: Objective identifier
            
        Returns:
            List of dependent objectives
        """
        return [
            obj for obj in self.objectives.values()
            if objective_id in obj.dependencies
        ]
        
    def validate_objective(self, objective_id: str) -> Dict[str, Any]:
        """Validate objective against its success criteria.
        
        Args:
            objective_id: Objective identifier
            
        Returns:
            Validation results
            
        Raises:
            ValueError: If objective not found
        """
        objective = self.objectives.get(objective_id)
        if not objective:
            raise ValueError(f"Objective {objective_id} not found")
            
        results = {
            "objective_id": objective_id,
            "status": objective.status.value,
            "criteria_results": []
        }
        
        for criteria in objective.success_criteria:
            validator_type = criteria.validation_method
            if not validator_type:  # Skip if no validation method
                results["criteria_results"].append({
                    "type": criteria.criteria_type,
                    "description": criteria.description,
                    "validated": False,
                    "error": "No validation method specified"
                })
                continue
                
            validator = ValidatorRegistry.get_validator(validator_type)
            
            if not validator:
                results["criteria_results"].append({
                    "type": criteria.criteria_type,
                    "description": criteria.description,
                    "validated": False,
                    "error": f"No validator found for method: {validator_type}"
                })
                continue
                
            # Get validation data from objective metadata
            validation_data = objective.metadata.get("validation_data", {})
            
            # Convert criteria to dict for validator
            criteria_dict = {
                "type": criteria.criteria_type,
                "description": criteria.description,
                "threshold": criteria.threshold,
                "validation_method": criteria.validation_method,
                "metadata": criteria.metadata
            }
            
            # Validate
            validation_result = validator.validate(validation_data, criteria_dict)
            
            results["criteria_results"].append({
                "type": criteria.criteria_type,
                "description": criteria.description,
                "validated": validation_result.is_valid,
                "score": validation_result.score,
                "details": validation_result.details,
                "message": validation_result.message,
                "timestamp": validation_result.timestamp
            })
            
        return results
        
    def get_validation_history(
        self,
        objective_id: str,
        validation_type: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get validation history for an objective.
        
        Args:
            objective_id: ID of the objective
            validation_type: Optional validation type filter
            start_time: Optional start time filter
            end_time: Optional end time filter
            
        Returns:
            List of validation results
        """
        return self.persistence.get_results(
            objective_id=objective_id,
            validation_type=validation_type,
            start_time=start_time,
            end_time=end_time
        )
        
    def clear_validation_history(
        self,
        objective_id: str,
        validation_type: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> bool:
        """Clear validation history for an objective.
        
        Args:
            objective_id: ID of the objective
            validation_type: Optional validation type filter
            timestamp: Optional specific timestamp
            
        Returns:
            True if cleared successfully, False otherwise
        """
        if validation_type:
            return self.persistence.delete_result(
                objective_id=objective_id,
                validation_type=validation_type,
                timestamp=timestamp
            )
        else:
            # Clear all validation types
            success = True
            for criteria in self.objectives[objective_id].success_criteria:
                validator_type = criteria.validation_method
                if validator_type:
                    success = success and self.persistence.delete_result(
                        objective_id=objective_id,
                        validation_type=validator_type,
                        timestamp=timestamp
                    )
            return success
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert handler state to dictionary
        
        Returns:
            Dictionary representation
        """
        return {
            "objectives": {
                obj_id: {
                    "objective_id": obj.objective_id,
                    "type": obj.type.value,
                    "description": obj.description,
                    "success_criteria": [
                        {
                            "criteria_type": crit.criteria_type,
                            "description": crit.description,
                            "threshold": crit.threshold,
                            "validation_method": crit.validation_method,
                            "metadata": crit.metadata
                        }
                        for crit in obj.success_criteria
                    ],
                    "status": obj.status.value,
                    "priority": obj.priority,
                    "dependencies": obj.dependencies,
                    "metadata": obj.metadata,
                    "created_at": obj.created_at,
                    "updated_at": obj.updated_at
                }
                for obj_id, obj in self.objectives.items()
            },
            "current_objective": self.current_objective
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ObjectiveHandler':
        """Create handler from dictionary
        
        Args:
            data: Dictionary representation
            
        Returns:
            ObjectiveHandler instance
        """
        handler = cls(config={})  # We might want to store/restore config too
        
        for obj_id, obj_data in data["objectives"].items():
            success_criteria = [
                Success_Criteria(
                    criteria_type=crit["criteria_type"],
                    description=crit["description"],
                    threshold=crit.get("threshold"),
                    validation_method=crit.get("validation_method"),
                    metadata=crit.get("metadata", {})
                )
                for crit in obj_data["success_criteria"]
            ]
            
            objective = Objective(
                objective_id=obj_data["objective_id"],
                type=ObjectiveType(obj_data["type"]),
                description=obj_data["description"],
                success_criteria=success_criteria,
                status=ObjectiveStatus(obj_data["status"]),
                priority=obj_data["priority"],
                dependencies=obj_data["dependencies"],
                metadata=obj_data["metadata"],
                created_at=obj_data["created_at"],
                updated_at=obj_data["updated_at"]
            )
            
            handler.objectives[obj_id] = objective
            
        handler.current_objective = data.get("current_objective")
        
        return handler
