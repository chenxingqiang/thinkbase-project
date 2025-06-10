"""Objective progress tracking instruction implementation."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import time

from .base import BaseInstruction, ExecutionContext, ExecutionResult

@dataclass
class ProgressConfig:
    """Configuration for progress tracking."""
    update_interval: float = 1.0  # seconds
    min_progress_delta: float = 0.01
    max_stall_time: float = 300.0  # seconds
    completion_threshold: float = 1.0

class ObjectiveProgressTrackingInstruction(BaseInstruction):
    """Instruction for tracking objective progress."""
    
    def __init__(self, name: str, config: ProgressConfig):
        """Initialize progress tracking instruction.
        
        Args:
            name: Instruction name
            config: Progress tracking configuration
        """
        super().__init__(name)
        self.config = config
        self.progress_history: Dict[str, List[Dict[str, Any]]] = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute progress tracking.
        
        Args:
            context: Execution context containing:
                - objective: Objective to track
                - progress_update: Progress update data
                
        Returns:
            ExecutionResult: Progress tracking results
        """
        try:
            # Extract objective and progress update
            objective = context.params.get("objective")
            progress_update = context.params.get("progress_update")
            
            if not objective or not progress_update:
                return ExecutionResult(
                    success=False,
                    output={"error": "Missing objective or progress update"},
                    errors=[{"error": "Missing objective or progress update"}]
                )
            
            # Update progress
            progress_result = await self._update_progress(objective, progress_update)
            
            # Calculate metrics
            metrics = self._calculate_metrics(objective["id"])
            
            return ExecutionResult(
                success=True,
                output={
                    "progress": progress_result,
                    "metrics": metrics
                }
            )
            
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}]
            )
    
    async def _update_progress(self, 
                             objective: Dict[str, Any],
                             progress_update: Dict[str, Any]) -> Dict[str, Any]:
        """Update progress for objective.
        
        Args:
            objective: Objective to update
            progress_update: Progress update data
            
        Returns:
            Updated progress information
        """
        objective_id = objective["id"]
        
        # Initialize history if needed
        if objective_id not in self.progress_history:
            self.progress_history[objective_id] = []
        
        # Add timestamp to update
        update_with_time = {
            **progress_update,
            "timestamp": time.time()
        }
        
        # Add to history
        self.progress_history[objective_id].append(update_with_time)
        
        # Calculate current progress
        current_progress = self._calculate_progress(objective_id)
        
        return {
            "objective_id": objective_id,
            "current_progress": current_progress,
            "update": update_with_time,
            "history_length": len(self.progress_history[objective_id])
        }
    
    def _calculate_progress(self, objective_id: str) -> float:
        """Calculate current progress for objective."""
        if objective_id not in self.progress_history:
            return 0.0
            
        history = self.progress_history[objective_id]
        if not history:
            return 0.0
            
        # Get latest progress value
        latest = history[-1]
        return float(latest.get("progress", 0.0))
    
    def _calculate_metrics(self, objective_id: str) -> Dict[str, Any]:
        """Calculate metrics for objective."""
        if objective_id not in self.progress_history:
            return {
                "updates_count": 0,
                "elapsed_time": 0.0,
                "progress_rate": 0.0,
                "stall_time": 0.0
            }
            
        history = self.progress_history[objective_id]
        if not history:
            return {
                "updates_count": 0,
                "elapsed_time": 0.0,
                "progress_rate": 0.0,
                "stall_time": 0.0
            }
        
        # Calculate metrics
        first_update = history[0]
        last_update = history[-1]
        
        elapsed_time = last_update["timestamp"] - first_update["timestamp"]
        progress_delta = last_update.get("progress", 0.0) - first_update.get("progress", 0.0)
        
        # Calculate stall time
        stall_time = 0.0
        for i in range(1, len(history)):
            prev = history[i-1]
            curr = history[i]
            
            time_delta = curr["timestamp"] - prev["timestamp"]
            progress_change = abs(curr.get("progress", 0.0) - prev.get("progress", 0.0))
            
            if progress_change < self.config.min_progress_delta:
                stall_time += time_delta
        
        return {
            "updates_count": len(history),
            "elapsed_time": elapsed_time,
            "progress_rate": progress_delta / elapsed_time if elapsed_time > 0 else 0.0,
            "stall_time": stall_time
        } 