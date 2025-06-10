"""Documentation instructions for generating various types of documentation."""

from typing import Dict, Any, List, Optional, Set, Union, Type, cast
from dataclasses import dataclass, field
import time
import logging
import os
import hashlib
from pathlib import Path

from .base import (
    BaseInstruction,
    ExecutionContext,
    ExecutionResult,
    InstructionType,
    InstructionMetadata,
    InstructionMetrics
)

logger = logging.getLogger(__name__)

@dataclass
class DocumentationMetrics:
    """Metrics for documentation operations."""
    pages_generated: int = 0
    words_generated: int = 0
    sections_covered: int = 0
    images_generated: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

class DocumentationInstruction(BaseInstruction):
    """Base class for documentation instructions."""
    
    def __init__(
        self,
        name: str,
        description: str = "",
        instruction_type: Optional[InstructionType] = None,
        metadata: Optional[InstructionMetadata] = None
    ):
        """Initialize documentation instruction."""
        super().__init__(
            name=name,
            description=description,
            instruction_type=instruction_type or InstructionType.CODE_GENERATION,
            metadata=metadata or InstructionMetadata(
                capabilities={"documentation"},
                tags={name.lower(), "documentation"},
                cleanup_required=True
            )
        )
        self.documentation_metrics = DocumentationMetrics()
        
        # Initialize state
        self._temp_files = set()
        self._documentation_state = {}
        self._cache = {}
        
        # Register required parameters
        self.add_required_param("input_content", str)
        self.add_required_param("output_format", str)
        
    async def cleanup(self) -> None:
        """Clean up documentation resources."""
        try:
            # Clean up temporary files
            for temp_file in self._temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file {temp_file}: {e}")
            
            # Clear documentation state
            self._documentation_state.clear()
            self._cache.clear()
            
            # Call parent cleanup
            await super().cleanup()
            
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

@dataclass
class DocumentationGenerationInstruction(DocumentationInstruction):
    """Generate comprehensive documentation."""
    
    def __init__(self):
        super().__init__(
            name="documentation_generation",
            description="Generate comprehensive project documentation",
            instruction_type=InstructionType.CODE_GENERATION,
            metadata=InstructionMetadata(
                capabilities={"documentation_generation", "content_analysis"},
                tags={"documentation", "generation", "comprehensive"},
                cleanup_required=True,
                resource_requirements={"memory": 512, "cpu": 0.5}
            )
        )
        
        # Add specific parameters
        self.add_required_param("doc_type", str)
        self.add_required_param("template", str)
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute documentation generation."""
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
            
            # Check cache
            cache_key = self._get_cache_key(context)
            if cache_key in self._cache:
                self.documentation_metrics.cache_hits += 1
                return self._cache[cache_key]
            
            self.documentation_metrics.cache_misses += 1
            
            # Generate documentation
            documentation = await self._generate_documentation(context.params)
            
            # Update metrics
            self.documentation_metrics.pages_generated = len(documentation.get("modules", []))
            self.documentation_metrics.sections_covered = len(documentation.get("guides", []))
            self.documentation_metrics.execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                output={"documentation": documentation},
                metrics={
                    "documentation_metrics": self.documentation_metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
            # Cache result
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Documentation generation failed: {str(e)}")
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
    
    def _get_cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key from context."""
        context_str = f"{context.params.get('input_content', '')}:{context.params.get('doc_type', '')}"
        return hashlib.sha256(context_str.encode()).hexdigest()
    
    async def _generate_documentation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive documentation."""
        return {
            "overview": "",
            "modules": [],
            "examples": [],
            "guides": [],
            "metadata": {
                "generated_at": time.time(),
                "version": "1.0.0",
                "template_used": params.get("template", "default")
            }
        }

@dataclass
class APIDocumentationInstruction(DocumentationInstruction):
    """Generate API documentation."""
    
    def __init__(self):
        super().__init__(
            name="api_documentation",
            description="Generate API documentation and references",
            instruction_type=InstructionType.CODE_GENERATION,
            metadata=InstructionMetadata(
                capabilities={"api_documentation", "code_analysis"},
                tags={"api", "documentation", "reference"},
                cleanup_required=True,
                resource_requirements={"memory": 768, "cpu": 0.75}
            )
        )
        
        # Add specific parameters
        self.add_required_param("api_version", str)
        self.add_required_param("include_examples", bool)
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute API documentation generation."""
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
            
            # Check cache
            cache_key = self._get_cache_key(context)
            if cache_key in self._cache:
                self.documentation_metrics.cache_hits += 1
                return self._cache[cache_key]
            
            self.documentation_metrics.cache_misses += 1
            
            # Generate API docs
            api_docs = await self._generate_api_docs(context.params)
            
            # Update metrics
            self.documentation_metrics.pages_generated = len(api_docs.get("endpoints", []))
            self.documentation_metrics.sections_covered = len(api_docs.get("models", []))
            self.documentation_metrics.execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                output={"api_docs": api_docs},
                metrics={
                    "documentation_metrics": self.documentation_metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
            # Cache result
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"API documentation generation failed: {str(e)}")
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
    
    def _get_cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key from context."""
        context_str = f"{context.params.get('input_content', '')}:{context.params.get('api_version', '')}"
        return hashlib.sha256(context_str.encode()).hexdigest()
    
    async def _generate_api_docs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate API documentation."""
        return {
            "endpoints": [],
            "models": [],
            "examples": [],
            "references": {},
            "metadata": {
                "generated_at": time.time(),
                "api_version": params.get("api_version", "1.0.0"),
                "include_examples": params.get("include_examples", True)
            }
        }

@dataclass
class ReadmeGenerationInstruction(DocumentationInstruction):
    """Generate project README."""
    
    def __init__(self):
        super().__init__(
            name="readme_generation",
            description="Generate project README and quick start guides",
            instruction_type=InstructionType.CODE_GENERATION,
            metadata=InstructionMetadata(
                capabilities={"readme_generation", "markdown_generation"},
                tags={"readme", "documentation", "quickstart"},
                cleanup_required=True,
                resource_requirements={"memory": 256, "cpu": 0.25}
            )
        )
        
        # Add specific parameters
        self.add_required_param("project_name", str)
        self.add_required_param("include_badges", bool)
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute README generation."""
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
            
            # Check cache
            cache_key = self._get_cache_key(context)
            if cache_key in self._cache:
                self.documentation_metrics.cache_hits += 1
                return self._cache[cache_key]
            
            self.documentation_metrics.cache_misses += 1
            
            # Generate README
            readme = await self._generate_readme(context.params)
            
            # Update metrics
            self.documentation_metrics.pages_generated = 1
            self.documentation_metrics.sections_covered = len([
                section for section in readme.values() if section
            ])
            self.documentation_metrics.execution_time = time.time() - start_time
            
            result = ExecutionResult(
                success=True,
                output={"readme": readme},
                metrics={
                    "documentation_metrics": self.documentation_metrics.__dict__,
                    "execution_time": time.time() - start_time
                }
            )
            
            # Cache result
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"README generation failed: {str(e)}")
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
    
    def _get_cache_key(self, context: ExecutionContext) -> str:
        """Generate cache key from context."""
        context_str = f"{context.params.get('input_content', '')}:{context.params.get('project_name', '')}"
        return hashlib.sha256(context_str.encode()).hexdigest()
    
    async def _generate_readme(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate project README."""
        return {
            "title": params.get("project_name", ""),
            "description": "",
            "installation": "",
            "usage": "",
            "examples": [],
            "contributing": "",
            "license": "",
            "metadata": {
                "generated_at": time.time(),
                "include_badges": params.get("include_badges", True),
                "project_name": params.get("project_name", "")
            }
        } 