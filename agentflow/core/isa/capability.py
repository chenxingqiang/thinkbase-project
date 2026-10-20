"""Advanced capability assessment and profiling system."""
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from .formal import FormalInstruction

class CapabilityDomain(Enum):
    """Domains for capability assessment."""
    REASONING = "reasoning"
    MEMORY = "memory"
    GENERATION = "generation"
    OPTIMIZATION = "optimization"
    SPECIALIZATION = "specialization"
    COORDINATION = "coordination"
    SECURITY = "security"
    ROBUSTNESS = "robustness"

@dataclass
class CapabilityMetric:
    """Metric for capability assessment."""
    name: str
    domain: CapabilityDomain
    value: float
    confidence: float
    metadata: Dict[str, Any]

@dataclass
class DomainScore:
    """Aggregated score for one capability domain."""
    score: float
    confidence: float = 1.0
    weight: float = 0.0
    metrics: List[CapabilityMetric] = field(default_factory=list)

class CapabilityProfile:
    """Advanced capability profiling system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics: Dict[str, CapabilityMetric] = {}
        self.history: List[Dict[str, Any]] = []
        # Per-domain aggregate scores keyed by domain value (e.g.
        # "reasoning"); populated by assess_capabilities and consumed
        # by the multi-agent coordinator.
        self.capabilities: Dict[str, "DomainScore"] = {}
        # Declared specializations (e.g. domain expertise tags).
        self.specializations: List[str] = list(
            config.get("specializations", [])
        )
        
    def assess_capabilities(self,
                          instructions: List[FormalInstruction]) -> Dict[str, Any]:
        """Assess capabilities across all domains."""
        results = {}
        for domain in CapabilityDomain:
            assessment = self._assess_domain(domain, instructions)
            results[domain.value] = assessment
            self.capabilities[domain.value] = DomainScore(
                score=assessment["score"],
                weight=assessment["weight"],
                metrics=assessment["metrics"],
            )
        
        # Update history
        self._update_history(instructions, results)
        
        return results

    def get_overall_score(self) -> float:
        """Weighted overall capability score across assessed domains."""
        if not self.capabilities:
            return 0.0
        total_weight = sum(d.weight for d in self.capabilities.values())
        if total_weight <= 0:
            return sum(d.score for d in self.capabilities.values()) / len(
                self.capabilities
            )
        return sum(
            d.score * d.weight for d in self.capabilities.values()
        ) / total_weight
    
    def _assess_domain(self,
                      domain: CapabilityDomain,
                      instructions: List[FormalInstruction]) -> Dict[str, Any]:
        """Assess capabilities for a specific domain."""
        metrics = self._compute_domain_metrics(domain, instructions)
        weight = self._get_capability_weight(domain)
        
        return {
            "metrics": metrics,
            "weight": weight,
            "score": self._compute_weighted_score(metrics)
        }
    
    def _compute_domain_metrics(self,
                              domain: CapabilityDomain,
                              instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Compute metrics for a specific domain."""
        if domain == CapabilityDomain.REASONING:
            return self._assess_reasoning(instructions)
        elif domain == CapabilityDomain.MEMORY:
            return self._assess_memory(instructions)
        elif domain == CapabilityDomain.GENERATION:
            return self._assess_generation(instructions)
        elif domain == CapabilityDomain.OPTIMIZATION:
            return self._assess_optimization(instructions)
        elif domain == CapabilityDomain.SPECIALIZATION:
            return self._assess_specialization(instructions)
        elif domain == CapabilityDomain.COORDINATION:
            return self._assess_coordination(instructions)
        elif domain == CapabilityDomain.SECURITY:
            return self._assess_security(instructions)
        elif domain == CapabilityDomain.ROBUSTNESS:
            return self._assess_robustness(instructions)
        return []
    
    def _get_capability_weight(self, domain: CapabilityDomain) -> float:
        """Get weight for capability domain."""
        weights = {
            CapabilityDomain.REASONING: 0.2,
            CapabilityDomain.MEMORY: 0.15,
            CapabilityDomain.GENERATION: 0.15,
            CapabilityDomain.OPTIMIZATION: 0.1,
            CapabilityDomain.SPECIALIZATION: 0.1,
            CapabilityDomain.COORDINATION: 0.1,
            CapabilityDomain.SECURITY: 0.1,
            CapabilityDomain.ROBUSTNESS: 0.1
        }
        return weights.get(domain, 0.0)
    
    def _compute_weighted_score(self,
                              metrics: List[CapabilityMetric]) -> float:
        """Compute the confidence-weighted mean of metric values.

        The domain weight is deliberately NOT applied here: it enters only
        once, at the top level (``get_overall_score``). Folding it into the
        per-domain score as well double-counts it, which an earlier version
        of this method did.
        """
        if not metrics:
            return 0.0

        total_conf = sum(metric.confidence for metric in metrics)
        if total_conf <= 0:
            return sum(metric.value for metric in metrics) / len(metrics)
        return sum(
            metric.value * metric.confidence for metric in metrics
        ) / total_conf
    
    def _assess_reasoning(self,
                         instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess reasoning capabilities."""
        metrics = []
        
        # Logical consistency
        consistency = self._measure_logical_consistency(instructions)
        metrics.append(CapabilityMetric(
            name="logical_consistency",
            domain=CapabilityDomain.REASONING,
            value=consistency,
            confidence=0.9,
            metadata={}
        ))
        
        # Causal understanding
        causality = self._measure_causal_understanding(instructions)
        metrics.append(CapabilityMetric(
            name="causal_understanding",
            domain=CapabilityDomain.REASONING,
            value=causality,
            confidence=0.85,
            metadata={}
        ))
        
        # Abstract reasoning
        abstraction = self._measure_abstract_reasoning(instructions)
        metrics.append(CapabilityMetric(
            name="abstract_reasoning",
            domain=CapabilityDomain.REASONING,
            value=abstraction,
            confidence=0.8,
            metadata={}
        ))
        
        return metrics
    
    def _assess_memory(self,
                      instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess memory capabilities."""
        metrics = []
        
        # Working memory
        working_memory = self._measure_working_memory(instructions)
        metrics.append(CapabilityMetric(
            name="working_memory",
            domain=CapabilityDomain.MEMORY,
            value=working_memory,
            confidence=0.9,
            metadata={}
        ))
        
        # Long-term memory
        long_term = self._measure_long_term_memory(instructions)
        metrics.append(CapabilityMetric(
            name="long_term_memory",
            domain=CapabilityDomain.MEMORY,
            value=long_term,
            confidence=0.85,
            metadata={}
        ))
        
        return metrics
    
    def _assess_generation(self,
                         instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess generation capabilities."""
        metrics = []
        
        # Creativity
        creativity = self._measure_creativity(instructions)
        metrics.append(CapabilityMetric(
            name="creativity",
            domain=CapabilityDomain.GENERATION,
            value=creativity,
            confidence=0.8,
            metadata={}
        ))
        
        # Coherence
        coherence = self._measure_coherence(instructions)
        metrics.append(CapabilityMetric(
            name="coherence",
            domain=CapabilityDomain.GENERATION,
            value=coherence,
            confidence=0.9,
            metadata={}
        ))
        
        return metrics
    
    def _assess_optimization(self,
                           instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess optimization capabilities."""
        metrics = []
        
        # Efficiency
        efficiency = self._measure_efficiency(instructions)
        metrics.append(CapabilityMetric(
            name="efficiency",
            domain=CapabilityDomain.OPTIMIZATION,
            value=efficiency,
            confidence=0.9,
            metadata={}
        ))
        
        # Resource usage
        resources = self._measure_resource_usage(instructions)
        metrics.append(CapabilityMetric(
            name="resource_usage",
            domain=CapabilityDomain.OPTIMIZATION,
            value=resources,
            confidence=0.85,
            metadata={}
        ))
        
        return metrics
    
    def _assess_specialization(self,
                             instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess specialization capabilities."""
        metrics = []
        
        # Domain expertise
        expertise = self._measure_domain_expertise(instructions)
        metrics.append(CapabilityMetric(
            name="domain_expertise",
            domain=CapabilityDomain.SPECIALIZATION,
            value=expertise,
            confidence=0.85,
            metadata={}
        ))
        
        # Adaptability
        adaptability = self._measure_adaptability(instructions)
        metrics.append(CapabilityMetric(
            name="adaptability",
            domain=CapabilityDomain.SPECIALIZATION,
            value=adaptability,
            confidence=0.8,
            metadata={}
        ))
        
        return metrics
    
    def _assess_coordination(self,
                           instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess coordination capabilities."""
        metrics = []
        
        # Communication
        communication = self._measure_communication(instructions)
        metrics.append(CapabilityMetric(
            name="communication",
            domain=CapabilityDomain.COORDINATION,
            value=communication,
            confidence=0.9,
            metadata={}
        ))
        
        # Collaboration
        collaboration = self._measure_collaboration(instructions)
        metrics.append(CapabilityMetric(
            name="collaboration",
            domain=CapabilityDomain.COORDINATION,
            value=collaboration,
            confidence=0.85,
            metadata={}
        ))
        
        return metrics
    
    def _assess_security(self,
                        instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess security capabilities."""
        metrics = []
        
        # Threat detection
        threats = self._measure_threat_detection(instructions)
        metrics.append(CapabilityMetric(
            name="threat_detection",
            domain=CapabilityDomain.SECURITY,
            value=threats,
            confidence=0.9,
            metadata={}
        ))
        
        # Policy compliance
        compliance = self._measure_policy_compliance(instructions)
        metrics.append(CapabilityMetric(
            name="policy_compliance",
            domain=CapabilityDomain.SECURITY,
            value=compliance,
            confidence=0.95,
            metadata={}
        ))
        
        return metrics
    
    def _assess_robustness(self,
                          instructions: List[FormalInstruction]) -> List[CapabilityMetric]:
        """Assess robustness capabilities."""
        metrics = []
        
        # Error handling
        error_handling = self._measure_error_handling(instructions)
        metrics.append(CapabilityMetric(
            name="error_handling",
            domain=CapabilityDomain.ROBUSTNESS,
            value=error_handling,
            confidence=0.9,
            metadata={}
        ))
        
        # Recovery
        recovery = self._measure_recovery(instructions)
        metrics.append(CapabilityMetric(
            name="recovery",
            domain=CapabilityDomain.ROBUSTNESS,
            value=recovery,
            confidence=0.85,
            metadata={}
        ))
        
        return metrics
    
    def _update_history(self,
                       instructions: List[FormalInstruction],
                       results: Dict[str, Any]) -> None:
        """Update assessment history."""
        self.history.append({
            "instructions": instructions,
            "results": results,
            "timestamp": np.datetime64('now')
        })
    
    # Measurement methods
    def _measure_logical_consistency(self,
                                  instructions: List[FormalInstruction]) -> float:
        """Measure logical consistency."""
        return 0.0  # Implement measurement logic
    
    def _measure_causal_understanding(self,
                                   instructions: List[FormalInstruction]) -> float:
        """Measure causal understanding."""
        return 0.0  # Implement measurement logic
    
    def _measure_abstract_reasoning(self,
                                 instructions: List[FormalInstruction]) -> float:
        """Measure abstract reasoning."""
        return 0.0  # Implement measurement logic
    
    def _measure_working_memory(self,
                             instructions: List[FormalInstruction]) -> float:
        """Measure working memory."""
        return 0.0  # Implement measurement logic
    
    def _measure_long_term_memory(self,
                               instructions: List[FormalInstruction]) -> float:
        """Measure long-term memory."""
        return 0.0  # Implement measurement logic
    
    def _measure_creativity(self,
                         instructions: List[FormalInstruction]) -> float:
        """Measure creativity."""
        return 0.0  # Implement measurement logic
    
    def _measure_coherence(self,
                        instructions: List[FormalInstruction]) -> float:
        """Measure coherence."""
        return 0.0  # Implement measurement logic
    
    def _measure_efficiency(self,
                         instructions: List[FormalInstruction]) -> float:
        """Measure efficiency."""
        return 0.0  # Implement measurement logic
    
    def _measure_resource_usage(self,
                             instructions: List[FormalInstruction]) -> float:
        """Measure resource usage."""
        return 0.0  # Implement measurement logic
    
    def _measure_domain_expertise(self,
                               instructions: List[FormalInstruction]) -> float:
        """Measure domain expertise."""
        return 0.0  # Implement measurement logic
    
    def _measure_adaptability(self,
                           instructions: List[FormalInstruction]) -> float:
        """Measure adaptability."""
        return 0.0  # Implement measurement logic
    
    def _measure_communication(self,
                            instructions: List[FormalInstruction]) -> float:
        """Measure communication."""
        return 0.0  # Implement measurement logic
    
    def _measure_collaboration(self,
                            instructions: List[FormalInstruction]) -> float:
        """Measure collaboration."""
        return 0.0  # Implement measurement logic
    
    def _measure_threat_detection(self,
                               instructions: List[FormalInstruction]) -> float:
        """Measure threat detection."""
        return 0.0  # Implement measurement logic
    
    def _measure_policy_compliance(self,
                                instructions: List[FormalInstruction]) -> float:
        """Measure policy compliance."""
        return 0.0  # Implement measurement logic
    
    def _measure_error_handling(self,
                             instructions: List[FormalInstruction]) -> float:
        """Measure error handling."""
        return 0.0  # Implement measurement logic
    
    def _measure_recovery(self,
                       instructions: List[FormalInstruction]) -> float:
        """Measure recovery."""
        return 0.0  # Implement measurement logic
