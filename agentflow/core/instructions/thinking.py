"""Thinking instructions for AI cognitive processes."""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import time
import logging
from pathlib import Path

from agentflow.core.instructions.base import BaseInstruction, InstructionResult, InstructionStatus

logger = logging.getLogger(__name__)

class ThinkingStyle(Enum):
    """Different styles of thinking."""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    CRITICAL = "critical"
    SYSTEMS = "systems"
    STRATEGIC = "strategic"

class CognitivePhase(Enum):
    """Phases in the cognitive process."""
    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    EVALUATION = "evaluation"
    DECISION = "decision"

@dataclass
class ThoughtMetrics:
    """Metrics for tracking thinking process."""
    start_time: float
    phases_completed: List[str]
    insights_generated: int
    decision_points: int
    reasoning_paths: List[str]
    evidence_points: List[str]
    
    @property
    def thinking_duration(self) -> float:
        """Calculate total thinking duration."""
        return time.time() - self.start_time

@dataclass
class ThoughtContext:
    """Context for thinking process."""
    problem_statement: str
    known_facts: List[str]
    assumptions: List[str]
    constraints: List[str]
    goals: List[str]
    style: ThinkingStyle
    phase: CognitivePhase

class ThinkingInstruction(BaseInstruction):
    """Instruction for structured thinking process."""
    
    def __init__(self):
        super().__init__(
            name="thinking",
            description="Execute structured thinking process"
        )
        self._metrics = None
        self._context = None
        self._thought_stream = []
        
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute thinking process."""
        try:
            # Initialize thinking context
            self._context = ThoughtContext(
                problem_statement=context.get("problem", ""),
                known_facts=context.get("facts", []),
                assumptions=context.get("assumptions", []),
                constraints=context.get("constraints", []),
                goals=context.get("goals", []),
                style=ThinkingStyle(context.get("style", "analytical")),
                phase=CognitivePhase.OBSERVATION
            )
            
            # Initialize metrics
            self._metrics = ThoughtMetrics(
                start_time=time.time(),
                phases_completed=[],
                insights_generated=0,
                decision_points=0,
                reasoning_paths=[],
                evidence_points=[]
            )
            
            # Execute thinking phases
            await self._observe()
            await self._analyze()
            await self._synthesize()
            await self._evaluate()
            await self._decide()
            
            metrics_data = {
                "duration": self._metrics.thinking_duration,
                "phases": self._metrics.phases_completed,
                "insights": self._metrics.insights_generated,
                "decisions": self._metrics.decision_points
            }
            
            result_data = {
                "thought_stream": self._thought_stream,
                "insights": self._get_insights(),
                "decisions": self._get_decisions(),
                "evidence": self._metrics.evidence_points,
                "metrics": metrics_data
            }
            
            return InstructionResult(
                status=str(InstructionStatus.COMPLETED),
                data=result_data,
                metrics=metrics_data
            )
            
        except Exception as e:
            error_msg = f"Error in thinking process: {str(e)}"
            logger.error(error_msg)
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={},
                metrics={}
            )
    
    async def _observe(self) -> None:
        """Observation phase of thinking."""
        if not self._context or not self._metrics:
            return
            
        self._add_thought("Beginning observation phase...")
        
        # Analyze problem statement
        self._add_thought(f"Problem statement: {self._context.problem_statement}")
        
        # Review known facts
        for fact in self._context.known_facts:
            self._add_thought(f"Considering fact: {fact}")
            
        # Note assumptions
        for assumption in self._context.assumptions:
            self._add_thought(f"Working with assumption: {assumption}")
            
        self._metrics.phases_completed.append(CognitivePhase.OBSERVATION.value)
    
    async def _analyze(self) -> None:
        """Analysis phase of thinking."""
        if not self._context or not self._metrics:
            return
            
        self._add_thought("Moving to analysis phase...")
        
        # Apply thinking style
        if self._context.style == ThinkingStyle.ANALYTICAL:
            await self._analytical_analysis()
        elif self._context.style == ThinkingStyle.CREATIVE:
            await self._creative_analysis()
        elif self._context.style == ThinkingStyle.CRITICAL:
            await self._critical_analysis()
        elif self._context.style == ThinkingStyle.SYSTEMS:
            await self._systems_analysis()
        else:
            await self._strategic_analysis()
            
        self._metrics.phases_completed.append(CognitivePhase.ANALYSIS.value)
    
    async def _synthesize(self) -> None:
        """Synthesis phase of thinking."""
        if not self._context or not self._metrics:
            return
            
        self._add_thought("Beginning synthesis phase...")
        
        # Connect observations with analysis
        self._add_thought("Connecting observations with analytical insights...")
        
        # Generate new insights
        self._add_thought("Generating new insights from combined understanding...")
        self._metrics.insights_generated += 1
        
        # Identify patterns
        self._add_thought("Looking for patterns and relationships...")
        
        self._metrics.phases_completed.append(CognitivePhase.SYNTHESIS.value)
    
    async def _evaluate(self) -> None:
        """Evaluation phase of thinking."""
        if not self._context or not self._metrics:
            return
            
        self._add_thought("Moving to evaluation phase...")
        
        # Check against goals
        for goal in self._context.goals:
            self._add_thought(f"Evaluating against goal: {goal}")
            
        # Consider constraints
        for constraint in self._context.constraints:
            self._add_thought(f"Checking constraint: {constraint}")
            
        # Assess evidence
        self._add_thought("Assessing strength of evidence...")
        
        self._metrics.phases_completed.append(CognitivePhase.EVALUATION.value)
    
    async def _decide(self) -> None:
        """Decision phase of thinking."""
        if not self._context or not self._metrics:
            return
            
        self._add_thought("Entering decision phase...")
        
        # Make decisions based on evaluation
        self._add_thought("Formulating decisions based on evaluation...")
        self._metrics.decision_points += 1
        
        # Document reasoning
        self._add_thought("Documenting decision reasoning...")
        
        # Record evidence
        self._metrics.evidence_points.extend(self._get_current_evidence())
        
        self._metrics.phases_completed.append(CognitivePhase.DECISION.value)
    
    async def _analytical_analysis(self) -> None:
        """Analytical thinking approach."""
        if not self._metrics:
            return
            
        self._add_thought("Applying analytical thinking...")
        self._metrics.reasoning_paths.append("analytical")
    
    async def _creative_analysis(self) -> None:
        """Creative thinking approach."""
        if not self._metrics:
            return
            
        self._add_thought("Applying creative thinking...")
        self._metrics.reasoning_paths.append("creative")
    
    async def _critical_analysis(self) -> None:
        """Critical thinking approach."""
        if not self._metrics:
            return
            
        self._add_thought("Applying critical thinking...")
        self._metrics.reasoning_paths.append("critical")
    
    async def _systems_analysis(self) -> None:
        """Systems thinking approach."""
        if not self._metrics:
            return
            
        self._add_thought("Applying systems thinking...")
        self._metrics.reasoning_paths.append("systems")
    
    async def _strategic_analysis(self) -> None:
        """Strategic thinking approach."""
        if not self._metrics:
            return
            
        self._add_thought("Applying strategic thinking...")
        self._metrics.reasoning_paths.append("strategic")
    
    def _add_thought(self, thought: str) -> None:
        """Add a thought to the stream."""
        phase = self._context.phase.value if self._context else "unknown"
        self._thought_stream.append({
            "timestamp": time.time(),
            "phase": phase,
            "content": thought
        })
    
    def _get_insights(self) -> List[Dict[str, Any]]:
        """Get generated insights."""
        return [
            thought for thought in self._thought_stream
            if "insight" in thought["content"].lower()
        ]
    
    def _get_decisions(self) -> List[Dict[str, Any]]:
        """Get made decisions."""
        return [
            thought for thought in self._thought_stream
            if "decision" in thought["content"].lower()
        ]
    
    def _get_current_evidence(self) -> List[str]:
        """Get evidence from current thinking."""
        return [
            thought["content"]
            for thought in self._thought_stream[-5:]  # Last 5 thoughts
            if any(word in thought["content"].lower() 
                  for word in ["because", "therefore", "since", "evidence"])
        ]

class ReflectiveThinkingInstruction(ThinkingInstruction):
    """Extension for reflective thinking process."""
    
    def __init__(self):
        super().__init__()
        self.name = "reflective_thinking"
        self.description = "Execute reflective thinking process"
        self._reflections = []
    
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute reflective thinking process."""
        # First execute normal thinking
        result = await super()._execute_impl(context)
        
        if result.status == InstructionStatus.COMPLETED:
            # Add reflection phase
            await self._reflect()
            
            # Update result with reflections
            result.data["reflections"] = self._reflections
            
        return result
    
    async def _reflect(self) -> None:
        """Reflection phase of thinking."""
        self._add_thought("Beginning reflection phase...")
        
        # Review thought process
        self._add_reflection("Reviewing entire thought process...")
        
        # Analyze decisions
        for decision in self._get_decisions():
            self._add_reflection(f"Reflecting on decision: {decision['content']}")
            
        # Consider alternatives
        self._add_reflection("Considering alternative approaches...")
        
        # Learn from process
        self._add_reflection("Extracting learning points...")
    
    def _add_reflection(self, reflection: str) -> None:
        """Add a reflection."""
        self._reflections.append({
            "timestamp": time.time(),
            "content": reflection
        })

class CollaborativeThinkingInstruction(ThinkingInstruction):
    """Extension for collaborative thinking process."""
    
    def __init__(self):
        super().__init__()
        self.name = "collaborative_thinking"
        self.description = "Execute collaborative thinking process"
        self._perspectives = []
        self._agreements = []
        self._disagreements = []
    
    async def _execute_impl(self, context: Dict[str, Any]) -> InstructionResult:
        """Execute collaborative thinking process."""
        # Get perspectives
        perspectives = context.get("perspectives", [])
        if not perspectives:
            return InstructionResult(
                status=str(InstructionStatus.FAILED),
                data={"error": "No perspectives provided for collaborative thinking"},
                metrics={}
            )
            
        # Consider each perspective
        for perspective in perspectives:
            await self._consider_perspective(perspective)
            
        # Find common ground
        await self._find_common_ground()
        
        # Execute normal thinking with enriched context
        context["assumptions"] = self._agreements
        result = await super()._execute_impl(context)
        
        if result.status == str(InstructionStatus.COMPLETED):
            # Add collaboration results
            result.data.update({
                "perspectives": self._perspectives,
                "agreements": self._agreements,
                "disagreements": self._disagreements
            })
            
        return result
    
    async def _consider_perspective(self, perspective: Dict[str, Any]) -> None:
        """Consider a single perspective."""
        self._add_thought(f"Considering perspective: {perspective['name']}")
        
        # Record perspective
        self._perspectives.append({
            "timestamp": time.time(),
            "name": perspective["name"],
            "points": perspective["points"]
        })
        
        # Analyze points
        for point in perspective["points"]:
            self._add_thought(f"Analyzing point: {point}")
    
    async def _find_common_ground(self) -> None:
        """Find common ground between perspectives."""
        self._add_thought("Looking for common ground between perspectives...")
        
        # Identify agreements
        self._add_thought("Identifying points of agreement...")
        
        # Note disagreements
        self._add_thought("Noting points of disagreement...") 