"""Analysis instructions for code and project analysis."""

from typing import Dict, Any, Optional, cast, Union, Type
from dataclasses import dataclass
import time
import logging
import os
from pathlib import Path

from .base import (
    BaseInstruction,
    ExecutionContext,
    ExecutionResult,
    InstructionType,
    InstructionMetadata
)

logger = logging.getLogger(__name__)

@dataclass
class AnalysisMetrics:
    """Metrics for analysis operations."""
    items_analyzed: int = 0
    patterns_found: int = 0
    issues_found: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0

class AnalysisInstruction(BaseInstruction):
    """Base class for analysis instructions."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize analysis instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.ANALYSIS,
            metadata=metadata or InstructionMetadata(
                capabilities={"analysis"},
                tags={name.lower(), "analysis"},
                cleanup_required=True
            )
        )
        self.analysis_metrics = AnalysisMetrics()
        
        # Register required parameters with proper type validation
        self.add_required_param("input_text", str)
        self.add_required_param("input_data", dict)
        self.add_required_param("analysis_type", str)
        
        # Initialize state
        self._temp_files = set()
        self._analysis_state = {}
        
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute analysis instruction."""
        start_time = time.time()
        
        try:
            # Validate context and parameters
            valid, error_msg = context.validate_required_params(self.get_required_params())
            if not valid:
                return ExecutionResult(
                    success=False,
                    output={},
                    errors=[{"error": error_msg}]
                )
            
            # Validate resource requirements if any
            if self.metadata.resource_requirements:
                valid, error_msg = context.validate_resources(self.metadata.resource_requirements)
                if not valid:
                    return ExecutionResult(
                        success=False,
                        output={},
                        errors=[{"error": error_msg}]
                    )
            
            # Combine inputs for analysis
            analysis_input = {
                "text": context.params.get("input_text", ""),
                "data": context.params.get("input_data", {})
            }
            
            # Perform analysis
            result = await self._analyze(analysis_input)
            
            # Update metrics
            execution_time = time.time() - start_time
            self.analysis_metrics.execution_time = execution_time
            self.update_metrics(True, execution_time)
            
            return ExecutionResult(
                success=True,
                output=result,
                metrics={
                    "analysis_metrics": self.analysis_metrics.__dict__,
                    "execution_time": execution_time
                }
            )
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            execution_time = time.time() - start_time
            self.update_metrics(False, execution_time)
            
            return ExecutionResult(
                success=False,
                output={},
                errors=[{"error": str(e)}],
                metrics={"execution_time": execution_time}
            )
            
        finally:
            await self.cleanup()
            
    async def cleanup(self) -> None:
        """Clean up analysis resources."""
        try:
            # Clean up temporary files
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            # Clear analysis state
            self._analysis_state.clear()
            
            # Call parent cleanup
            await super().cleanup()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
            
    async def _analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Implement analysis logic in subclasses."""
        raise NotImplementedError("Subclasses must implement _analyze")

class RequirementAnalysisInstruction(AnalysisInstruction):
    """Analyze project requirements."""
    
    def __init__(self):
        super().__init__(
            name="requirement_analysis",
            description="Analyze project requirements and dependencies",
            metadata=InstructionMetadata(
                capabilities={"requirement_analysis", "dependency_tracking"},
                tags={"requirements", "analysis", "planning"},
                cleanup_required=True,
                resource_requirements={"memory": 512, "cpu": 0.5}
            )
        )
        
        # Add specific parameters
        self.add_required_param("requirements_file", str)
        self.add_required_param("analysis_depth", int)
        
    async def _analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements."""
        try:
            requirements_file = params["requirements_file"]
            analysis_depth = params["analysis_depth"]
            
            # Validate file exists
            if not os.path.exists(requirements_file):
                raise ValueError(f"Requirements file not found: {requirements_file}")
            
            # Track file for cleanup
            self._temp_files.add(requirements_file)
            
            # Perform analysis
            analysis_result = await self._analyze_requirements(params)
            
            # Update metrics
            self.analysis_metrics.items_analyzed = len(analysis_result.get("requirements", []))
            self.analysis_metrics.patterns_found = len(analysis_result.get("patterns", []))
            self.analysis_metrics.issues_found = len(analysis_result.get("issues", []))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Requirement analysis failed: {str(e)}")
            raise
        
    async def _analyze_requirements(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze requirements implementation."""
        return {
            "requirements": [],
            "patterns": [],
            "issues": [],
            "recommendations": []
        }

class CodeAnalysisInstruction(AnalysisInstruction):
    """Analyze code quality and patterns."""
    
    def __init__(self):
        super().__init__(
            name="code_analysis",
            description="Analyze code quality, patterns and issues",
            metadata=InstructionMetadata(
                capabilities={"code_analysis", "pattern_detection"},
                tags={"code", "analysis", "quality"},
                cleanup_required=True,
                resource_requirements={"memory": 1024, "cpu": 1.0}
            )
        )
        
        # Add specific parameters
        self.add_required_param("code_path", str)
        self.add_required_param("analysis_rules", list)
        
    async def _analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code."""
        try:
            code_path = params["code_path"]
            analysis_rules = params["analysis_rules"]
            
            # Validate path exists
            if not os.path.exists(code_path):
                raise ValueError(f"Code path not found: {code_path}")
            
            # Track path for cleanup
            self._temp_files.add(code_path)
            
            # Perform analysis
            analysis_result = await self._analyze_code(params)
            
            # Update metrics
            self.analysis_metrics.items_analyzed = analysis_result.get("files_analyzed", 0)
            self.analysis_metrics.patterns_found = len(analysis_result.get("patterns", []))
            self.analysis_metrics.issues_found = len(analysis_result.get("issues", []))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Code analysis failed: {str(e)}")
            raise
        
    async def _analyze_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze code implementation."""
        return {
            "files_analyzed": 0,
            "patterns": [],
            "issues": [],
            "metrics": {},
            "recommendations": []
        }

class DependencyAnalysisInstruction(AnalysisInstruction):
    """Analyze project dependencies."""
    
    def __init__(self):
        super().__init__(
            name="dependency_analysis",
            description="Analyze project dependencies and relationships",
            metadata=InstructionMetadata(
                capabilities={"dependency_analysis", "impact_analysis"},
                tags={"dependencies", "analysis", "relationships"},
                cleanup_required=True,
                resource_requirements={"memory": 768, "cpu": 0.75}
            )
        )
        
        # Add specific parameters
        self.add_required_param("project_path", str)
        self.add_required_param("include_dev", bool)
        
    async def _analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies."""
        try:
            project_path = params["project_path"]
            include_dev = params["include_dev"]
            
            # Validate path exists
            if not os.path.exists(project_path):
                raise ValueError(f"Project path not found: {project_path}")
            
            # Track path for cleanup
            self._temp_files.add(project_path)
            
            # Perform analysis
            analysis_result = await self._analyze_dependencies(params)
            
            # Update metrics
            self.analysis_metrics.items_analyzed = len(analysis_result.get("dependencies", []))
            self.analysis_metrics.patterns_found = len(analysis_result.get("patterns", []))
            self.analysis_metrics.issues_found = len(analysis_result.get("issues", []))
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Dependency analysis failed: {str(e)}")
            raise
        
    async def _analyze_dependencies(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze dependencies implementation."""
        return {
            "dependencies": [],
            "dependency_graph": {},
            "issues": [],
            "impact_analysis": {},
            "recommendations": []
        } 