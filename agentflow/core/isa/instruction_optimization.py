"""Instruction Optimization System implementation following AgentISA paper Section 3.4."""

from typing import Dict, List, Any, Set, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import networkx as nx
from .instruction import Instruction, InstructionType

@dataclass
class OptimizationMetrics:
    """Metrics for optimization decisions."""
    execution_time: float
    memory_usage: float
    api_calls: int
    success_rate: float

class InstructionDAG:
    """Directed Acyclic Graph for instruction dependencies."""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.execution_order: List[str] = []

    def add_instruction(self, instruction: Instruction, 
                       dependencies: Optional[List[str]] = None):
        """Add instruction to DAG."""
        self.graph.add_node(instruction.id, instruction=instruction)
        if dependencies:
            for dep in dependencies:
                self.graph.add_edge(dep, instruction.id)

    def get_execution_order(self) -> List[str]:
        """Get topologically sorted execution order."""
        try:
            self.execution_order = list(nx.topological_sort(self.graph))
            return self.execution_order
        except nx.NetworkXUnfeasible:
            raise ValueError("Circular dependency detected in instruction graph")

class PipelineScheduler:
    """Instruction pipeline scheduler."""
    
    def __init__(self):
        self.dag = InstructionDAG()
        self.resource_limits = defaultdict(lambda: float('inf'))
        self.stage_latencies = defaultdict(float)

    def set_resource_limit(self, resource: str, limit: float):
        """Set resource usage limit."""
        self.resource_limits[resource] = limit

    def add_stage_latency(self, stage: str, latency: float):
        """Add latency information for pipeline stage."""
        self.stage_latencies[stage] = latency

    def schedule(self, instructions: List[Instruction]) -> List[List[Instruction]]:
        """Schedule instructions into pipeline stages."""
        # Build dependency graph
        for instruction in instructions:
            deps = self._analyze_dependencies(instruction)
            self.dag.add_instruction(instruction, deps)
        
        # Get execution order
        order = self.dag.get_execution_order()
        
        # Allocate to stages
        stages: List[List[Instruction]] = []
        current_stage: List[Instruction] = []
        current_resources = defaultdict(float)
        
        for instr_id in order:
            instruction = self.dag.graph.nodes[instr_id]['instruction']
            resource_needs = self._estimate_resources(instruction)
            
            # Check if instruction can fit in current stage
            can_fit = all(
                current_resources[r] + need <= self.resource_limits[r]
                for r, need in resource_needs.items()
            )
            
            if can_fit:
                current_stage.append(instruction)
                for r, need in resource_needs.items():
                    current_resources[r] += need
            else:
                # Start new stage
                if current_stage:
                    stages.append(current_stage)
                current_stage = [instruction]
                current_resources = defaultdict(float)
                for r, need in resource_needs.items():
                    current_resources[r] = need
        
        # Add last stage if not empty
        if current_stage:
            stages.append(current_stage)
            
        return stages

    def _analyze_dependencies(self, instruction: Instruction) -> List[str]:
        """Analyze instruction dependencies."""
        # Add dependency analysis logic
        return []

    def _estimate_resources(self, instruction: Instruction) -> Dict[str, float]:
        """Estimate resource requirements for instruction."""
        # Add resource estimation logic
        return defaultdict(float)

class _PatternPlaceholderInstruction(Instruction):
    """Placeholder produced by peephole pattern rewrites.

    Carries only a name; executes as a no-op.  Peephole patterns are
    name-level rewrites, so replacements have no backing semantics
    until bound to concrete instructions.
    """

    def __init__(self, id: str, name: str):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Peephole rewrite placeholder for {name}",
            side_effects=False,
            outputs=[],
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"status": "noop", "name": self.name}


class StaticOptimizer:
    """Static instruction optimization."""
    
    def __init__(self):
        self.peephole_patterns: Dict[str, Tuple[List[str], List[str]]] = {}
        self.optimization_stats: Dict[str, int] = defaultdict(int)

    def add_peephole_pattern(self, pattern: List[str], replacement: List[str]):
        """Add peephole optimization pattern."""
        pattern_key = "_".join(pattern)
        self.peephole_patterns[pattern_key] = (pattern, replacement)

    def optimize(self, instructions: List[Instruction]) -> List[Instruction]:
        """Apply static optimizations."""
        # Apply peephole optimization
        optimized = self._apply_peephole(instructions)
        
        # Remove dead code
        optimized = self._eliminate_dead_code(optimized)
        
        # Combine instructions
        optimized = self._combine_instructions(optimized)
        
        return optimized

    def _apply_peephole(self, instructions: List[Instruction]) -> List[Instruction]:
        """Apply peephole optimization patterns.

        A pattern is a sequence of instruction names; when it matches a
        contiguous run of the trace it is rewritten to the replacement
        names.  Replacements are emitted as placeholder instructions so
        the output stays uniformly typed.
        """
        result = []
        i = 0
        while i < len(instructions):
            matched = False
            for pattern, replacement in self.peephole_patterns.values():
                if i + len(pattern) <= len(instructions):
                    # Check if pattern matches
                    if all(getattr(instructions[i+j], "name", None) == pattern[j] 
                          for j in range(len(pattern))):
                        # Apply replacement
                        for name in replacement:
                            result.append(_PatternPlaceholderInstruction(
                                id=name, name=name
                            ))
                        i += len(pattern)
                        matched = True
                        self.optimization_stats["peephole_applied"] += 1
                        break
            if not matched:
                result.append(instructions[i])
                i += 1
        return result

    def _eliminate_dead_code(self, instructions: List[Instruction]) -> List[Instruction]:
        """Eliminate dead code via backward liveness analysis.

        An instruction is live when it (a) has side effects, or (b)
        produces outputs consumed by a later live instruction.  The
        analysis walks the sequence backwards, maintaining the set of
        instruction ids whose results are still needed downstream.
        """
        live_instructions: List[Instruction] = []
        needed_ids: Set[str] = set()
        for instruction in reversed(instructions):
            deps = getattr(instruction, "dependencies", []) or []
            outputs = getattr(instruction, "outputs", []) or []
            has_side_effects = getattr(instruction, "side_effects", True)
            is_live = (
                has_side_effects
                or instruction.id in needed_ids
                or bool(outputs)
            )
            if is_live:
                live_instructions.append(instruction)
                needed_ids.update(deps)
            else:
                self.optimization_stats["dead_code_eliminated"] += 1
        live_instructions.reverse()
        return live_instructions

    def _is_instruction_live(self, instruction: Instruction) -> bool:
        """Check if instruction has effect on output.

        Note: this reflects the instruction in isolation.  Sequence-level
        liveness is computed by :meth:`_eliminate_dead_code`, which also
        accounts for consumers of the instruction's outputs.
        """
        return (
            getattr(instruction, "side_effects", True)
            or bool(getattr(instruction, "outputs", []))
            or bool(getattr(instruction, "dependencies", []))
        )

    def _combine_instructions(self, instructions: List[Instruction]) -> List[Instruction]:
        """Combine compatible instructions."""
        result = []
        current = None
        
        for instruction in instructions:
            if not current:
                current = instruction
            elif self._can_combine(current, instruction):
                current = self._merge_instructions(current, instruction)
                self.optimization_stats["instructions_combined"] += 1
            else:
                result.append(current)
                current = instruction
        
        if current:
            result.append(current)
            
        return result

    def _can_combine(self, instr1: Instruction, instr2: Instruction) -> bool:
        """Check if instructions can be combined.

        Two instructions are combinable when they share the same type,
        neither has side effects, the second does not depend on the
        first's outputs, and their parameter signatures are compatible.
        """
        if instr1 is instr2:
            return False
        if instr1.type != instr2.type:
            return False
        if getattr(instr1, "side_effects", True) or getattr(instr2, "side_effects", True):
            return False
        deps2 = set(getattr(instr2, "dependencies", []) or [])
        if instr1.id in deps2:
            return False
        # Dependencies may also reference *output names* (the compiler's
        # convention): a producer-consumer pair expressed that way is
        # not order-independent and must not fuse.  Without this check,
        # fusing such a pair yields a composite whose second component
        # executes without the first component's outputs.
        if set(getattr(instr1, "outputs", []) or []) & deps2:
            return False
        # Parameter signatures must match for a merged instruction to
        # accept the same inputs.
        if set(instr1.params.keys()) != set(instr2.params.keys()):
            return False
        return True

    def _merge_instructions(self, instr1: Instruction, 
                          instr2: Instruction) -> Instruction:
        """Merge two compatible instructions into a composite.

        The merged instruction executes ``instr1`` then ``instr2``
        sequentially and is itself free of side effects.
        """
        from .instruction import CompositeInstruction
        merged = CompositeInstruction(
            id=f"merged_{instr1.id}_{instr2.id}",
            name=f"merged_{instr1.name}_{instr2.name}",
            components=[instr1, instr2],
            composition_type="sequential",
        )
        # Propagate semantic annotations for downstream analyses.
        merged.side_effects = False
        merged.outputs = list(
            dict.fromkeys(
                list(getattr(instr1, "outputs", []) or [])
                + list(getattr(instr2, "outputs", []) or [])
            )
        )
        merged.dependencies = list(
            dict.fromkeys(
                [d for d in (getattr(instr1, "dependencies", []) or [])
                 if d != instr2.id]
                + [d for d in (getattr(instr2, "dependencies", []) or [])
                   if d != instr1.id]
            )
        )
        return merged

class DynamicOptimizer:
    """Dynamic instruction optimization."""
    
    def __init__(self):
        self.hot_paths: Dict[str, List[Instruction]] = {}
        self.execution_counts: Dict[str, int] = defaultdict(int)
        self.performance_stats: Dict[str, OptimizationMetrics] = {}
        
        # Thresholds
        self.hot_path_threshold = 100
        self.stability_threshold = 0.9
        self.recompilation_threshold = 0.7

        # Trace optimization: reuse the static pass machinery, bounded
        # so pathological traces cannot loop forever.
        self.trace_optimizer = StaticOptimizer()
        self.max_trace_passes = 4

    def record_execution(self, path_id: str, instructions: List[Instruction], 
                        metrics: OptimizationMetrics):
        """Record execution of instruction sequence."""
        self.execution_counts[path_id] += 1
        self.performance_stats[path_id] = metrics
        
        # Check if path is hot
        if (self.execution_counts[path_id] > self.hot_path_threshold and
            self._is_path_stable(path_id)):
            self.hot_paths[path_id] = instructions

    def optimize_hot_path(self, path_id: str) -> Optional[List[Instruction]]:
        """Optimize a hot execution path."""
        if path_id not in self.hot_paths:
            return None
            
        instructions = self.hot_paths[path_id]
        
        # Apply trace-based optimization
        optimized = self._optimize_trace(instructions)
        
        # Verify optimization effectiveness
        if self._verify_optimization(path_id, optimized):
            return optimized
        return None

    def _is_path_stable(self, path_id: str) -> bool:
        """Check if execution path is stable."""
        if path_id not in self.performance_stats:
            return False
            
        metrics = self.performance_stats[path_id]
        return metrics.success_rate >= self.stability_threshold

    def _optimize_trace(self, instructions: List[Instruction]) -> List[Instruction]:
        """Optimize a hot trace with peephole + DCE + combination passes."""
        optimized = list(instructions)
        for _ in range(self.max_trace_passes):
            candidate = self.trace_optimizer.optimize(optimized)
            if len(candidate) == len(optimized):
                break
            optimized = candidate
        return optimized

    def _verify_optimization(self, path_id: str, 
                           optimized: List[Instruction]) -> bool:
        """Verify optimization effectiveness.

        An optimized trace is accepted only when it is non-empty, no
        longer than the original, and every instruction in it still
        satisfies its declared type constraints (basic soundness check).
        """
        if path_id not in self.performance_stats:
            return False
        if not optimized:
            return False
        if len(optimized) > len(self.hot_paths.get(path_id, optimized)):
            return False
        for instr in optimized:
            if not isinstance(instr, Instruction):
                return False
        return True

class OptimizationManager:
    """Manages both static and dynamic optimization."""
    
    def __init__(self):
        self.static_optimizer = StaticOptimizer()
        self.dynamic_optimizer = DynamicOptimizer()
        self.pipeline_scheduler = PipelineScheduler()

    def optimize_instructions(self, instructions: List[Instruction]) -> List[Instruction]:
        """Apply both static and dynamic optimization."""
        # First apply static optimization
        optimized = self.static_optimizer.optimize(instructions)
        
        # Schedule optimized instructions
        scheduled = self.pipeline_scheduler.schedule(optimized)
        
        # Flatten scheduled stages
        final_instructions = []
        for stage in scheduled:
            final_instructions.extend(stage)
        
        return final_instructions

    def record_execution(self, instructions: List[Instruction], 
                        metrics: OptimizationMetrics):
        """Record execution for dynamic optimization."""
        path_id = self._compute_path_id(instructions)
        self.dynamic_optimizer.record_execution(path_id, instructions, metrics)
        
        # Check for hot path optimization
        optimized = self.dynamic_optimizer.optimize_hot_path(path_id)
        if optimized:
            # Update instruction sequence
            return optimized
        return instructions

    def _compute_path_id(self, instructions: List[Instruction]) -> str:
        """Compute unique identifier for instruction sequence."""
        return "_".join(instr.id for instr in instructions) 