"""Objective validation instruction implementation."""

from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field

from .base import BaseInstruction, ExecutionContext, ExecutionResult

@dataclass
class ValidationRule:
    """Validation rule for objectives."""
    validation_function: Callable[[Dict[str, Any]], Dict[str, Any]]
    required_objective_fields: List[str]
    required_completion_fields: List[str]

@dataclass
class ValidationConfig:
    """Configuration for objective validation."""
    min_success_rate: float = 0.8
    timeout_seconds: float = 3600.0
    validation_required: bool = True
    max_retries: int = 3

class ObjectiveValidationInstruction(BaseInstruction):
    """Instruction for validating objective completion."""
    
    def __init__(self, name: str, config: ValidationConfig, rules: Dict[str, ValidationRule]):
        """Initialize objective validation instruction.
        
        Args:
            name: Instruction name
            config: Validation configuration
            rules: Dictionary of validation rules
        """
        super().__init__(name)
        self.config = config
        self.rules = rules
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute objective validation.
        
        Args:
            context: Execution context containing:
                - objective: Objective to validate
                - completion_data: Data about objective completion
                
        Returns:
            ExecutionResult: Validation results
        """
        try:
            # Extract objective and completion data
            objective = context.params.get("objective")
            completion_data = context.params.get("completion_data")
            
            if not objective or not completion_data:
                return ExecutionResult(
                    success=False,
                    output={"error": "Missing objective or completion data"},
                    errors=[{"error": "Missing objective or completion data"}]
                )
            
            # Get applicable rules
            applicable_rules = self._get_applicable_rules(objective)
            
            if not applicable_rules and self.config.validation_required:
                return ExecutionResult(
                    success=False,
                    output={"error": "No applicable validation rules"},
                    errors=[{"error": "No applicable validation rules"}]
                )
            
            # Apply validation rules
            validation_results = []
            for rule in applicable_rules:
                result = await self._apply_validation_rule(rule, objective, completion_data)
                validation_results.append(result)
            
            # Calculate overall success
            success = all(result.get("success", False) for result in validation_results)
            
            return ExecutionResult(
                success=True,
                output={
                    "validation_success": success,
                    "validation_results": validation_results,
                    "metrics": {
                        "rules_applied": len(validation_results),
                        "success_rate": sum(1 for r in validation_results if r.get("success", False)) / len(validation_results) if validation_results else 0.0
                    }
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}]
            )
    
    def _get_applicable_rules(self, objective: Dict[str, Any]) -> List[ValidationRule]:
        """Get validation rules applicable to objective."""
        applicable_rules = []
        
        for rule in self.rules.values():
            if self._is_rule_applicable(rule, objective):
                applicable_rules.append(rule)
                
        return applicable_rules
    
    def _is_rule_applicable(self, rule: ValidationRule, objective: Dict[str, Any]) -> bool:
        """Check if validation rule is applicable to objective."""
        # Check if objective has all required fields
        for field in rule.required_objective_fields:
            if field not in objective:
                return False
                
        return True
    
    async def _apply_validation_rule(self, 
                                   rule: ValidationRule,
                                   objective: Dict[str, Any],
                                   completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply validation rule to objective completion."""
        # Check completion data fields
        for field in rule.required_completion_fields:
            if field not in completion_data:
                return {
                    "success": False,
                    "error": f"Missing required completion field: {field}"
                }
        
        # Extract required data
        validation_data = self._extract_required_data(rule, objective, completion_data)
        
        # Apply validation function
        try:
            result = rule.validation_function(validation_data)
            return {
                "success": result.get("success", False),
                "details": result.get("details", {})
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_required_data(self,
                             rule: ValidationRule,
                             objective: Dict[str, Any],
                             completion_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract required data for validation."""
        validation_data = {}
        
        # Extract objective fields
        for field in rule.required_objective_fields:
            validation_data[f"objective_{field}"] = objective.get(field)
            
        # Extract completion fields
        for field in rule.required_completion_fields:
            validation_data[f"completion_{field}"] = completion_data.get(field)
            
        return validation_data 