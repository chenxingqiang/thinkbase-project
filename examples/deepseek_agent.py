"""
DeepSeek agent implementation using AgentISA architecture.
This agent specializes in code generation and follows TDD principles.
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from agentflow.core.instructions import (
    InstructionSpec,
    InstructionType,
    Pipeline
)
from agentflow.core.memory import WorkingMemory
from agentflow.providers.deepseek import DeepSeekProvider, DeepSeekConfig
from examples.deepseek_instructions import (
    INSTRUCTION_REGISTRY,
    DeepSeekInstructionType,
    BaseDeepSeekInstruction,
    CodeGenerationInstruction,
    RequirementAnalysisInstruction,
    CodeAnalysisInstruction,
    TestDesignInstruction,
    CodeReviewInstruction
)

@dataclass
class AgentMetrics:
    """Metrics for agent operations."""
    latency: float = 0.0
    tokens_used: int = 0
    success_rate: float = 0.0
    error_count: int = 0
    total_operations: int = 0
    total_successes: int = 0

@dataclass
class AgentConfig:
    """Configuration for DeepSeek agent."""
    name: str
    version: str
    type: str
    input_modes: List[str]
    output_modes: List[str]
    collaboration_mode: str
    workflow: List[str]

class DeepSeekAgent:
    """
    An AI agent powered by DeepSeek model, following AgentISA architecture.
    Implements TDD workflow for code generation and review.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Optional[AgentConfig] = None
        self.memory = WorkingMemory()
        self.provider = None  # Initialize provider after loading config
        self.pipeline = Pipeline(instructions=[])
        self.metrics = AgentMetrics()
        
    async def initialize(self) -> None:
        """Initialize the agent with configuration."""
        try:
            # Load configuration
            with open(self.config_path) as f:
                config_data = json.load(f)
            
            self.config = AgentConfig(
                name=config_data["AGENT"]["NAME"],
                version=config_data["AGENT"]["VERSION"],
                type=config_data["AGENT"]["TYPE"],
                input_modes=config_data["INPUT_SPECIFICATION"]["MODES"],
                output_modes=config_data["OUTPUT_SPECIFICATION"]["MODES"],
                collaboration_mode=config_data["COLLABORATION"]["MODE"],
                workflow=config_data["COLLABORATION"]["WORKFLOW"]
            )
            
            # Initialize provider with config
            provider_config = DeepSeekConfig(
                model_name=config_data["PROVIDER"]["MODEL"],
                temperature=config_data["PROVIDER"]["TEMPERATURE"],
                max_tokens=config_data["PROVIDER"]["MAX_TOKENS"],
                api_key=config_data["PROVIDER"]["API_KEY"],
                api_base=config_data["PROVIDER"]["API_BASE"]
            )
            self.provider = DeepSeekProvider(config=provider_config)
            
            # Initialize pipeline with instructions
            await self._setup_pipeline()
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize agent: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup agent resources."""
        if self.provider:
            await self.provider.close()
        await self.memory.clear()
        await self.pipeline.cleanup()
    
    async def _setup_pipeline(self) -> None:
        """Set up instruction pipeline for TDD workflow."""
        # Create instruction specifications
        specs = [
            InstructionSpec(
                type=InstructionType.REQUIREMENT_ANALYSIS,
                name="analyze_requirements",
                description="Analyze code generation requirements",
                inputs={"requirements": str},
                outputs={"analysis": Dict[str, Any]}
            ),
            InstructionSpec(
                type=InstructionType.TEST_DESIGN,
                name="design_tests",
                description="Design test cases based on analysis",
                inputs={"analysis": Dict[str, Any]},
                outputs={"test_cases": List[Dict[str, Any]]}
            ),
            InstructionSpec(
                type=InstructionType.CODE_GENERATION,
                name="implement_code",
                description="Implement code based on test cases",
                inputs={"test_specs": List[Dict[str, Any]]},
                outputs={"code": str}
            ),
            InstructionSpec(
                type=InstructionType.CODE_REVIEW,
                name="review_code",
                description="Review and improve implemented code",
                inputs={"code": str},
                outputs={"review": Dict[str, Any]}
            )
        ]
        
        # Add instructions to pipeline using registry
        for spec in specs:
            if spec.type == InstructionType.REQUIREMENT_ANALYSIS:
                instruction = RequirementAnalysisInstruction()
            elif spec.type == InstructionType.TEST_DESIGN:
                instruction = TestDesignInstruction()
            elif spec.type == InstructionType.CODE_GENERATION:
                instruction = CodeGenerationInstruction()
            elif spec.type == InstructionType.CODE_REVIEW:
                instruction = CodeReviewInstruction()
            else:
                raise ValueError(f"No instruction registered for type: {spec.type}")
            
            await self.pipeline.add_instruction(instruction)
    
    async def analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze requirements and create development plan."""
        if not requirements.strip():
            return {
                "status": "error",
                "message": "Requirements cannot be empty"
            }
        
        try:
            result = await self.pipeline.execute_instruction(
                "analyze_requirements",
                {
                    "requirements": requirements,
                    "output_format": "json"
                }
            )
            
            # Update metrics
            self._update_metrics(result.metrics)
            
            return {
                "status": "success",
                "analysis": result.data
            }
            
        except Exception as e:
            self.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def design_tests(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Design test cases based on requirement analysis."""
        if not analysis:
            return {
                "status": "error",
                "message": "Analysis data is required"
            }
        
        try:
            result = await self.pipeline.execute_instruction(
                "design_tests",
                {
                    "analysis": analysis,
                    "framework": "pytest",
                    "output_format": "json"
                }
            )
            
            # Update metrics
            self._update_metrics(result.metrics)
            
            return {
                "status": "success",
                "test_cases": result.data
            }
            
        except Exception as e:
            self.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def implement_code(self, test_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Implement code based on test specifications."""
        if not test_specs:
            return {
                "status": "error",
                "message": "Test specifications are required"
            }
        
        try:
            result = await self.pipeline.execute_instruction(
                "implement_code",
                {
                    "test_specs": test_specs,
                    "language": "python",
                    "style": "pep8"
                }
            )
            
            # Update metrics
            self._update_metrics(result.metrics)
            
            return {
                "status": "success",
                "code": result.data["implementation"]
            }
            
        except Exception as e:
            self.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    async def review_code(self, implementation: Dict[str, Any]) -> Dict[str, Any]:
        """Review implemented code and suggest improvements."""
        if not implementation or "code" not in implementation:
            return {
                "status": "error",
                "message": "Implementation code is required"
            }
        
        try:
            result = await self.pipeline.execute_instruction(
                "review_code",
                {
                    "code": implementation["code"],
                    "style": "pep8",
                    "output_format": "json"
                }
            )
            
            # Update metrics
            self._update_metrics(result.metrics)
            
            return {
                "status": "success",
                "review": result.data
            }
            
        except Exception as e:
            self.metrics.error_count += 1
            return {
                "status": "error",
                "message": str(e)
            }
    
    def _update_metrics(self, metrics: Dict[str, Any]) -> None:
        """Update agent metrics."""
        self.metrics.latency += metrics.get("latency", 0.0)
        self.metrics.tokens_used += metrics.get("tokens", 0)
        self.metrics.total_operations += 1
        self.metrics.total_successes += metrics.get("success", 1.0)
        self.metrics.success_rate = (
            self.metrics.total_successes / self.metrics.total_operations
            if self.metrics.total_operations > 0
            else 0.0
        )
    
    def get_metrics(self) -> AgentMetrics:
        """Get current agent metrics."""
        return self.metrics 