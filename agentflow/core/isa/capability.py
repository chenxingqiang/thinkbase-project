"""Advanced capability assessment and profiling system."""
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from enum import Enum
from .formal import FormalInstruction, InstructionType

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

# Declared-input arity that earns full working-memory credit.  An
# instruction declaring this many (or more) inputs is credited with the
# full metric value; fewer scale linearly.  Conservative default —
# typical agent instructions declare at most a handful of inputs.
WORKING_MEMORY_SATURATION = 4.0

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
    
    # ------------------------------------------------------------------
    # Measurement methods
    #
    # Each extractor is *static and side-effect-free* (Section 3.5:
    # assessment is cheap, deterministic, and never probes live
    # execution).  Values are derived exclusively from declared
    # instruction metadata — category labels, declared input/output
    # counts, cost hints, resource requirements, and optimization /
    # security affordances — and stay in [0, 1].  Quantities absent
    # from the metadata are scored conservatively (toward 0), never
    # invented.
    # ------------------------------------------------------------------
    def _measure_logical_consistency(self,
                                   instructions: List[FormalInstruction]) -> float:
        """Measure logical consistency.

        Mean over instructions of the fraction of structural
        well-formedness checks each passes: a positive finite timeout,
        a non-negative retry budget, no self-dependency, no duplicate
        dependencies, and no duplicate declared outputs.
        """
        if not instructions:
            return 0.0
        scores = []
        for instr in instructions:
            outputs = self._declared_outputs(instr)
            checks = [
                0.0 < float(instr.timeout) < float("inf"),
                instr.max_retries >= 0,
                not self._has_self_dependency(instr),
                len(instr.dependencies) == len(set(instr.dependencies)),
                len(outputs) == len(set(outputs)),
            ]
            scores.append(sum(checks) / len(checks))
        return sum(scores) / len(scores)

    def _measure_causal_understanding(self,
                                    instructions: List[FormalInstruction]) -> float:
        """Measure causal understanding.

        Fraction of instructions declaring at least one dependency
        edge — a producer/consumer ordering in the dataflow graph.
        """
        if not instructions:
            return 0.0
        return sum(1 for i in instructions if i.dependencies) / len(instructions)

    def _measure_abstract_reasoning(self,
                                  instructions: List[FormalInstruction]) -> float:
        """Measure abstract reasoning.

        Fraction of instructions declaring a formal abstraction: a
        typed parameter schema or pre/post-conditions/invariants.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.params or i.parameters
            or i.preconditions or i.postconditions or i.invariants
        ) / len(instructions)

    def _measure_working_memory(self,
                              instructions: List[FormalInstruction]) -> float:
        """Measure working memory.

        Mean declared input arity, saturated at
        ``WORKING_MEMORY_SATURATION``: the largest of the metadata
        ``inputs`` list, ``input:``-prefixed dependencies, and declared
        parameter schemas.
        """
        if not instructions:
            return 0.0
        total = sum(
            min(1.0, self._declared_input_count(i) / WORKING_MEMORY_SATURATION)
            for i in instructions
        )
        return total / len(instructions)

    def _measure_long_term_memory(self,
                                instructions: List[FormalInstruction]) -> float:
        """Measure long-term memory.

        Fraction of instructions with persistence affordances: the
        STATE category, a cacheability hint, or a declared storage
        resource requirement.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.type == InstructionType.STATE
            or i.optimization.cacheable
            or i.resources.storage > 0
        ) / len(instructions)

    def _measure_creativity(self,
                          instructions: List[FormalInstruction]) -> float:
        """Measure creativity.

        Fraction of generative (LLM-category) instructions.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions if i.type == InstructionType.LLM
        ) / len(instructions)

    def _measure_coherence(self,
                         instructions: List[FormalInstruction]) -> float:
        """Measure coherence.

        Fraction of instructions declaring their outputs (metadata
        ``outputs`` or ``output:``-prefixed dependencies): coherent
        producers declare what they produce.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions if self._declared_outputs(i)
        ) / len(instructions)

    def _measure_efficiency(self,
                          instructions: List[FormalInstruction]) -> float:
        """Measure efficiency.

        Fraction of instructions declaring execution-efficiency hints:
        the parallelizable or cacheable optimization flags, or the
        compiler's ``is_parallel`` metadata marker.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.optimization.parallelizable
            or i.optimization.cacheable
            or bool(i.metadata.get("is_parallel"))
        ) / len(instructions)

    def _measure_resource_usage(self,
                              instructions: List[FormalInstruction]) -> float:
        """Measure resource usage.

        Declared resource economy: the mean of ``1 / (1 + footprint)``
        per instruction, where the footprint sums the capacity fields
        of the resource requirement and any declared cost hint.  An
        undeclared (zero) footprint is fully frugal; heavier declared
        footprints score lower.
        """
        if not instructions:
            return 0.0
        total = sum(
            1.0 / (1.0 + self._resource_footprint(i))
            for i in instructions
        )
        return total / len(instructions)

    def _measure_domain_expertise(self,
                                instructions: List[FormalInstruction]) -> float:
        """Measure domain expertise.

        Coverage of the profile's declared specializations by the
        category labels carried by the instruction set (instruction
        type plus ``category``/``categories``/``tags``/``domain``/
        ``domains`` metadata).  Zero when no specializations are
        declared or none are covered.
        """
        if not instructions or not self.specializations:
            return 0.0
        labels: Set[str] = set()
        for instr in instructions:
            labels |= self._category_labels(instr)
        covered = sum(
            1 for spec in self.specializations
            if str(spec).strip().lower() in labels
        )
        return covered / len(self.specializations)

    def _measure_adaptability(self,
                            instructions: List[FormalInstruction]) -> float:
        """Measure adaptability.

        Fraction of instructions declaring adaptation affordances:
        the ADAPTIVE category, an ``adaptive`` metadata marker, or
        timing/placement flexibility (a deadline or locality hint).
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.type == InstructionType.ADAPTIVE
            or bool(i.metadata.get("adaptive"))
            or i.optimization.deadline is not None
            or i.optimization.locality is not None
        ) / len(instructions)

    def _measure_communication(self,
                             instructions: List[FormalInstruction]) -> float:
        """Measure communication.

        Fraction of instructions in the inter-agent communication
        categories (AGENT, DISTRIBUTED).
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.type in (InstructionType.AGENT, InstructionType.DISTRIBUTED)
        ) / len(instructions)

    def _measure_collaboration(self,
                             instructions: List[FormalInstruction]) -> float:
        """Measure collaboration.

        Fraction of instructions that both consume and produce — a
        declared dependency (or declared input) plus a declared output:
        the citizens of a collaborative pipeline.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if (i.dependencies or self._declared_input_count(i) > 0)
            and self._declared_outputs(i)
        ) / len(instructions)

    def _measure_threat_detection(self,
                                instructions: List[FormalInstruction]) -> float:
        """Measure threat detection.

        Fraction of instructions with security-inspection
        affordances: the SECURITY category, or declared verification /
        audit flags.
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.type == InstructionType.SECURITY
            or i.security.verification
            or i.security.audit
        ) / len(instructions)

    def _measure_policy_compliance(self,
                                 instructions: List[FormalInstruction]) -> float:
        """Measure policy compliance.

        Fraction of instructions declaring explicit policy
        constraints: a non-default access level, declared permissions,
        or encryption.  Instructions carrying only the default
        security constraint score nothing (conservative).
        """
        if not instructions:
            return 0.0
        return sum(
            1 for i in instructions
            if i.security.access_level not in (None, "", "default")
            or i.security.permissions
            or i.security.encryption
        ) / len(instructions)

    def _measure_error_handling(self,
                              instructions: List[FormalInstruction]) -> float:
        """Measure error handling.

        Mean over instructions of the declared error-handling
        contract: a bounded (positive, finite) timeout and a positive
        retry budget, each worth half.
        """
        if not instructions:
            return 0.0
        total = 0.0
        for instr in instructions:
            score = 0.0
            if 0.0 < float(instr.timeout) < float("inf"):
                score += 0.5
            if instr.max_retries > 0:
                score += 0.5
            total += score
        return total / len(instructions)

    def _measure_recovery(self,
                        instructions: List[FormalInstruction]) -> float:
        """Measure recovery.

        Fraction of instructions declaring recovery structure:
        post-conditions or invariants (failure detection), or an
        explicit recovery metadata key (``fallback``/``recovery``/
        ``rollback``/``compensation``).
        """
        if not instructions:
            return 0.0
        recovery_keys = ("fallback", "recovery", "rollback", "compensation")
        return sum(
            1 for i in instructions
            if i.postconditions or i.invariants
            or any(bool(i.metadata.get(key)) for key in recovery_keys)
        ) / len(instructions)

    # ------------------------------------------------------------------
    # Metadata extraction helpers
    # ------------------------------------------------------------------
    def _declared_outputs(self, instr: FormalInstruction) -> List[str]:
        """Declared output names: the metadata ``outputs`` list, or
        ``output:``-prefixed dependencies."""
        outputs = instr.metadata.get("outputs")
        if isinstance(outputs, (list, tuple)):
            return [str(o) for o in outputs]
        return [d for d in instr.dependencies if d.startswith("output:")]

    def _declared_input_count(self, instr: FormalInstruction) -> int:
        """Declared input arity: the largest of the metadata ``inputs``
        list, ``input:``-prefixed dependencies, and declared parameter
        schemas."""
        inputs = instr.metadata.get("inputs")
        metadata_count = len(inputs) if isinstance(inputs, (list, tuple)) else 0
        dep_count = sum(
            1 for d in instr.dependencies if d.startswith("input:")
        )
        param_count = len(set(instr.params) | set(instr.parameters))
        return max(metadata_count, dep_count, param_count)

    def _category_labels(self, instr: FormalInstruction) -> Set[str]:
        """Category labels carried by an instruction: its type value
        plus category-like metadata (``category``, ``categories``,
        ``tags``, ``domain``, ``domains``), normalized to lowercase."""
        labels = {instr.type.value}
        for key in ("category", "categories", "tags", "domain", "domains"):
            value = instr.metadata.get(key)
            if isinstance(value, str):
                labels.add(value.strip().lower())
            elif isinstance(value, (list, tuple)):
                labels.update(str(v).strip().lower() for v in value)
        return labels

    def _resource_footprint(self, instr: FormalInstruction) -> float:
        """Declared resource footprint: the sum of the capacity fields
        of the resource requirement plus any declared cost hint."""
        resources = instr.resources
        footprint = (
            float(resources.cpu) + float(resources.memory)
            + float(resources.gpu) + float(resources.network)
            + float(resources.storage) + self._declared_cost(instr)
        )
        return max(0.0, footprint)

    @staticmethod
    def _declared_cost(instr: FormalInstruction) -> float:
        """Declared cost hint (metadata ``cost`` or ``cost_hint``);
        0.0 when absent or non-numeric."""
        for key in ("cost", "cost_hint"):
            value = instr.metadata.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return float(value)
        return 0.0

    @staticmethod
    def _has_self_dependency(instr: FormalInstruction) -> bool:
        """Whether an instruction declares a dependency on itself (by
        id or name)."""
        deps = set(instr.dependencies)
        return instr.id in deps or (bool(instr.name) and instr.name in deps)
