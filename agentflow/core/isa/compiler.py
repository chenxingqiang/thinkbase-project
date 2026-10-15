"""Instruction-level Optimization and Compiler Infrastructure."""
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
from .formal import FormalInstruction, InstructionType

class OptimizationLevel(Enum):
    """Optimization levels for instruction compilation."""
    NONE = 0    # No optimization
    BASIC = 1   # Basic optimizations
    ADVANCED = 2  # Advanced optimizations
    AGGRESSIVE = 3  # Aggressive optimizations

class IRNodeType(Enum):
    """Intermediate representation node types."""
    SEQUENCE = "sequence"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    ATOMIC = "atomic"

@dataclass
class IRNode:
    """Intermediate representation node."""
    type: IRNodeType
    instructions: List[FormalInstruction]
    metadata: Dict[str, Any]
    dependencies: List[str]
    optimizations: List[str]

class CompilationContext:
    """Context for instruction compilation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_level = OptimizationLevel[
            config.get("optimization_level", "BASIC").upper()
        ]
        self.target_metrics = config.get("target_metrics", {})
        self.resource_constraints = config.get("resource_constraints", {})
        self.safety_checks = config.get("safety_checks", True)

class InstructionCompiler:
    """Compiles instructions into optimized sequences."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_passes = self._initialize_passes()
        self.ir_builder = IRBuilder()
        self.code_generator = CodeGenerator()
        
    def compile(self,
                instructions: List[FormalInstruction],
                context: CompilationContext) -> List[FormalInstruction]:
        """Compile instruction sequence with optimizations."""
        # Build IR
        ir = self.ir_builder.build(instructions)
        
        # Apply optimization passes
        for pass_name, pass_fn in self.optimization_passes.items():
            if self._should_apply_pass(pass_name, context):
                ir = pass_fn(ir, context)
        
        # Generate optimized code
        return self.code_generator.generate(ir, context)
    
    def _initialize_passes(self) -> Dict[str, callable]:
        """Initialize optimization passes."""
        return {
            "instruction_fusion": self._fuse_instructions,
            "dead_code_elimination": self._eliminate_dead_code,
            "parallel_execution": self._parallelize_execution,
            "resource_optimization": self._optimize_resources,
            "memory_optimization": self._optimize_memory
        }
    
    def _should_apply_pass(self,
                          pass_name: str,
                          context: CompilationContext) -> bool:
        """Determine if optimization pass should be applied."""
        level_requirements = {
            "instruction_fusion": OptimizationLevel.BASIC,
            "dead_code_elimination": OptimizationLevel.BASIC,
            "parallel_execution": OptimizationLevel.ADVANCED,
            "resource_optimization": OptimizationLevel.ADVANCED,
            "memory_optimization": OptimizationLevel.AGGRESSIVE
        }
        return context.optimization_level.value >= level_requirements[pass_name].value
    
    def _fuse_instructions(self,
                         ir: IRNode,
                         context: CompilationContext) -> IRNode:
        """Fuse compatible instructions."""
        if ir.type != IRNodeType.SEQUENCE:
            return ir
            
        fused_instructions = []
        i = 0
        while i < len(ir.instructions):
            if i < len(ir.instructions) - 1:
                fused = self._try_fuse(
                    ir.instructions[i],
                    ir.instructions[i + 1],
                    context
                )
                if fused:
                    fused_instructions.append(fused)
                    i += 2
                    continue
            fused_instructions.append(ir.instructions[i])
            i += 1
            
        ir.instructions = fused_instructions
        return ir
    
    def _eliminate_dead_code(self,
                           ir: IRNode,
                           context: CompilationContext) -> IRNode:
        """Eliminate dead instructions via backward liveness analysis.

        An instruction is live when it has side effects, when any later
        instruction depends on it (by id or by one of its declared
        outputs), or when its outputs are listed in the compilation
        context's required outputs.
        """
        if ir.type == IRNodeType.SEQUENCE:
            required = set(context.target_metrics.get("required_outputs", []) or [])
            live_instructions = []
            needed: Set[str] = set(required)
            for instr in reversed(ir.instructions):
                outputs = list(instr.metadata.get("outputs", []) or [])
                deps = list(getattr(instr, "dependencies", []) or [])
                is_live = (
                    self._has_side_effects(instr)
                    or instr.id in needed
                    or any(out in needed for out in outputs)
                )
                if is_live:
                    live_instructions.append(instr)
                    needed.add(instr.id)
                    needed.update(deps)
                    needed.update(outputs)
            live_instructions.reverse()
            ir.instructions = live_instructions
        return ir
    
    def _parallelize_execution(self,
                             ir: IRNode,
                             context: CompilationContext) -> IRNode:
        """Identify and mark parallel execution opportunities."""
        if ir.type == IRNodeType.SEQUENCE:
            parallel_groups = self._identify_parallel_groups(
                ir.instructions,
                context
            )
            if parallel_groups:
                return IRNode(
                    type=IRNodeType.PARALLEL,
                    instructions=parallel_groups,
                    metadata=ir.metadata,
                    dependencies=ir.dependencies,
                    optimizations=ir.optimizations + ["parallelized"]
                )
        return ir
    
    def _optimize_resources(self,
                          ir: IRNode,
                          context: CompilationContext) -> IRNode:
        """Optimize resource usage."""
        resource_plan = self._create_resource_plan(ir, context)
        ir.metadata["resource_plan"] = resource_plan
        return ir
    
    def _optimize_memory(self,
                        ir: IRNode,
                        context: CompilationContext) -> IRNode:
        """Optimize memory usage."""
        memory_plan = self._create_memory_plan(ir, context)
        ir.metadata["memory_plan"] = memory_plan
        return ir
    
    def _try_fuse(self,
                  instr1: FormalInstruction,
                  instr2: FormalInstruction,
                  context: CompilationContext) -> Optional[FormalInstruction]:
        """Try to fuse two instructions.

        The fused instruction carries the union of the components'
        *external* dependencies: ``instr2``'s dependencies that are
        satisfied by ``instr1``'s outputs are internal to the fusion and
        dropped; everything else must survive, or downstream liveness
        and parallel-grouping analyses would see a node whose data
        dependencies have vanished (it could then, e.g., be grouped
        with its own producer).
        """
        if not self._are_fusible(instr1, instr2):
            return None

        fused = FormalInstruction(
            name=f"fused_{instr1.name}_{instr2.name}",
            type=instr1.type,
            metadata=self._merge_metadata(instr1, instr2)
        )
        internal = set(
            getattr(instr2, "dependencies", []) or []
        ).intersection(instr1.metadata.get("outputs", []) or [])
        fused.dependencies = [
            dep for dep in dict.fromkeys(
                list(getattr(instr1, "dependencies", []) or [])
                + list(getattr(instr2, "dependencies", []) or [])
            )
            if dep not in internal
        ]
        return fused
    
    def _has_side_effects(self, instr: FormalInstruction) -> bool:
        """Check whether an instruction has observable side effects.

        Side effects are declared through instruction metadata:
        ``metadata["side_effects"]`` (bool) or the presence of security
        constraints.  When undeclared, the instruction is conservatively
        treated as having side effects.
        """
        if "side_effects" in instr.metadata:
            return bool(instr.metadata["side_effects"])
        # Security-sensitive instructions always count as effectful.
        if getattr(instr.security, "permissions", None):
            return True
        if getattr(instr.security, "audit", False):
            return True
        return True

    def _has_used_outputs(self,
                          instr: FormalInstruction,
                          context: CompilationContext) -> bool:
        """Check whether any output of the instruction is consumed.

        Outputs are declared via ``metadata["outputs"]``; consumers are
        the dependencies of other instructions in the compilation unit
        plus the context's declared target outputs.
        """
        outputs = instr.metadata.get("outputs") or []
        if not outputs:
            return False
        consumers = set(context.target_metrics.get("required_outputs", []) or [])
        return any(out in consumers for out in outputs)

    def _merge_metadata(self,
                        instr1: FormalInstruction,
                        instr2: FormalInstruction) -> Dict[str, Any]:
        """Merge metadata of two fused instructions.

        Numeric quantities are summed, lists are unioned (order
        preserved), and any other keys prefer the first instruction.
        """
        merged: Dict[str, Any] = {}
        keys = list(instr1.metadata.keys()) + [
            k for k in instr2.metadata.keys() if k not in instr1.metadata
        ]
        for key in keys:
            v1 = instr1.metadata.get(key)
            v2 = instr2.metadata.get(key)
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                merged[key] = v1 + v2
            elif isinstance(v1, list) and isinstance(v2, list):
                merged[key] = list(dict.fromkeys(v1 + v2))
            elif isinstance(v1, list):
                merged[key] = list(v1)
            elif isinstance(v2, list):
                merged[key] = list(v2)
            else:
                merged[key] = v1 if v1 is not None else v2
        return merged

    def _can_parallelize(self,
                          instr: FormalInstruction,
                          group: List[FormalInstruction]) -> bool:
        """Check whether ``instr`` can join the parallel ``group``.

        An instruction can be parallelized with a group when it is free
        of side effects and does not depend on (nor is depended on by)
        any member of the group.
        """
        if self._has_side_effects(instr):
            return False
        instr_deps = set(getattr(instr, "dependencies", []) or [])
        instr_outputs = set(instr.metadata.get("outputs", []) or [])
        for other in group:
            other_deps = set(getattr(other, "dependencies", []) or [])
            other_outputs = set(other.metadata.get("outputs", []) or [])
            # Anti-dependency / flow dependency in either direction ...
            if (instr_deps & other_outputs
                    or other_deps & instr_outputs
                    or other.id in instr_deps
                    or instr.id in other_deps):
                return False
            # ... plus write-write hazards: two pure instructions that
            # declare the same output do not commute, so folding them
            # into one dispatch group would violate non-interference
            # (Theorem 3.5's S3 condition).
            if instr_outputs & other_outputs:
                return False
        return True

    def _calculate_resource_allocation(self,
                                       ir: IRNode,
                                       context: CompilationContext) -> Dict[str, float]:
        """Allocate resources proportionally to per-instruction cost."""
        flat = self._flatten_instructions(ir.instructions)
        total_cost = sum(
            i.metadata.get("cost", 1.0) for i in flat
        ) or 1.0
        return {
            "total_cost": total_cost,
            "budget_per_instruction": (
                context.resource_constraints.get("total_budget", total_cost)
                / total_cost
            ),
        }

    def _create_resource_schedule(self,
                                  ir: IRNode,
                                  context: CompilationContext) -> Dict[str, Any]:
        """Produce a stage schedule from the IR structure."""
        if ir.type == IRNodeType.PARALLEL:
            stages = [
                [getattr(i, "name", str(i)) for i in group]
                if isinstance(group, list) else [getattr(group, "name", str(group))]
                for group in ir.instructions
            ]
        else:
            stages = [[getattr(i, "name", str(i)) for i in ir.instructions]]
        return {"stages": stages, "max_stages": len(stages)}

    def _calculate_memory_allocation(self, ir: IRNode) -> Dict[str, float]:
        """Sum declared memory needs across the compilation unit."""
        flat = self._flatten_instructions(ir.instructions)
        return {
            "total_bytes": float(
                sum(i.metadata.get("memory", 0) for i in flat)
            )
        }

    def _create_memory_lifecycle(self, ir: IRNode) -> Dict[str, Any]:
        """Track when each declared output is produced and last consumed."""
        flat = self._flatten_instructions(ir.instructions)
        consumers: Dict[str, int] = {}
        for pos, instr in enumerate(flat):
            for dep in getattr(instr, "dependencies", []) or []:
                consumers[dep] = pos
        production = {
            out: pos
            for pos, instr in enumerate(flat)
            for out in instr.metadata.get("outputs", []) or []
        }
        return {
            "production": production,
            "last_use": consumers,
            "live_ranges": {
                out: (production.get(out, 0), consumers.get(out, production.get(out, 0)))
                for out in production
            },
        }

    def _identify_memory_optimizations(self, ir: IRNode) -> List[str]:
        """List applicable memory optimizations for the compilation unit."""
        optimizations: List[str] = []
        lifecycle = self._create_memory_lifecycle(ir)
        overlapping = sum(
            1 for out, (p, u) in lifecycle["live_ranges"].items() if u > p
        )
        if overlapping > 1:
            optimizations.append("reuse_disjoint_buffers")
        if len(self._flatten_instructions(ir.instructions)) > 8:
            optimizations.append("stream_small_intermediates")
        return optimizations

    def _are_fusible(self,
                     instr1: FormalInstruction,
                     instr2: FormalInstruction) -> bool:
        """Check if instructions can be fused."""
        # Dependencies live on the FormalInstruction ``dependencies``
        # field (consistent with ``_can_parallelize``); metadata keys
        # are accepted as a legacy fallback.
        instr2_deps = set(
            getattr(instr2, "dependencies", [])
            or instr2.metadata.get("dependencies", [])
        )
        instr1_outputs = set(
            instr1.metadata.get("outputs", [])
            or getattr(instr1, "outputs", [])
        )
        return (
            instr1.type == instr2.type and
            not self._has_side_effects(instr1) and
            not self._has_side_effects(instr2) and
            not instr2_deps.intersection(instr1_outputs)
        )
    
    def _is_instruction_live(self,
                           instr: FormalInstruction,
                           context: CompilationContext) -> bool:
        """Check if instruction has observable effects."""
        return (
            self._has_side_effects(instr) or
            self._has_used_outputs(instr, context)
        )
    
    def _identify_parallel_groups(self,
                                instructions: List[FormalInstruction],
                                context: CompilationContext) -> List[List[FormalInstruction]]:
        """Identify groups of instructions that can run in parallel.

        A group only ever contains side-effect-free instructions: an
        effectful instruction cannot *join* a group (``_can_parallelize``
        rejects it) and, crucially, cannot *start* one either — the
        previous version let an effectful instruction open a group that
        subsequent pure instructions then joined, producing dispatch
        groups whose non-interference (Theorem 3.5's S3) was never
        established.  Effectful instructions are always dispatched
        alone.
        """
        groups = []
        current_group = []

        for instr in instructions:
            if current_group and self._can_parallelize(instr, current_group):
                current_group.append(instr)
            else:
                if current_group:
                    groups.append(current_group)
                current_group = [instr]
                if self._has_side_effects(instr):
                    groups.append(current_group)
                    current_group = []

        if current_group:
            groups.append(current_group)

        return groups
    
    @staticmethod
    def _flatten_instructions(instructions: List[Any]) -> List[FormalInstruction]:
        """Flatten (possibly nested) IR instruction groups."""
        flat: List[FormalInstruction] = []
        for item in instructions:
            if isinstance(item, list):
                flat.extend(InstructionCompiler._flatten_instructions(item))
            else:
                flat.append(item)
        return flat

    def _create_resource_plan(self,
                            ir: IRNode,
                            context: CompilationContext) -> Dict[str, Any]:
        """Create optimized resource allocation plan."""
        return {
            "allocation": self._calculate_resource_allocation(ir, context),
            "scheduling": self._create_resource_schedule(ir, context),
            "constraints": context.resource_constraints
        }
    
    def _create_memory_plan(self,
                          ir: IRNode,
                          context: CompilationContext) -> Dict[str, Any]:
        """Create optimized memory usage plan."""
        return {
            "allocation": self._calculate_memory_allocation(ir),
            "lifecycle": self._create_memory_lifecycle(ir),
            "optimization": self._identify_memory_optimizations(ir)
        }

class IRBuilder:
    """Builds intermediate representation from instructions."""
    
    def build(self,
             instructions: List[FormalInstruction]) -> IRNode:
        """Build IR from instruction sequence."""
        return IRNode(
            type=IRNodeType.SEQUENCE,
            instructions=instructions,
            metadata=self._collect_metadata(instructions),
            dependencies=self._collect_dependencies(instructions),
            optimizations=[]
        )
    
    def _collect_metadata(self,
                         instructions: List[FormalInstruction]) -> Dict[str, Any]:
        """Collect metadata from instructions."""
        return {
            "cost": sum(instr.metadata.get("cost", 0) for instr in instructions),
            "memory": sum(instr.metadata.get("memory", 0) for instr in instructions),
            "complexity": self._calculate_complexity(instructions)
        }
    
    def _collect_dependencies(self,
                            instructions: List[FormalInstruction]) -> List[str]:
        """Collect all dependencies."""
        deps = set()
        for instr in instructions:
            deps.update(instr.metadata.get("dependencies", []))
        return list(deps)
    
    def _calculate_complexity(self,
                            instructions: List[FormalInstruction]) -> float:
        """Calculate computational complexity."""
        return sum(
            instr.metadata.get("complexity", 1.0)
            for instr in instructions
        )

class CodeGenerator:
    """Generates optimized instruction sequences from IR."""
    
    def generate(self,
                ir: IRNode,
                context: CompilationContext) -> List[FormalInstruction]:
        """Generate optimized instruction sequence."""
        if ir.type == IRNodeType.SEQUENCE:
            return ir.instructions
        elif ir.type == IRNodeType.PARALLEL:
            return self._generate_parallel(ir, context)
        elif ir.type == IRNodeType.CONDITIONAL:
            return self._generate_conditional(ir, context)
        elif ir.type == IRNodeType.LOOP:
            return self._generate_loop(ir, context)
        return []
    
    def _generate_parallel(self,
                         ir: IRNode,
                         context: CompilationContext) -> List[FormalInstruction]:
        """Generate code for parallel execution.

        Only groups with more than one member are wrapped: a singleton
        group has nothing to run in parallel with, and wrapping it
        would add dispatch overhead without exposing any concurrency.
        """
        instructions = []
        for group in ir.instructions:
            if isinstance(group, list):
                if len(group) > 1:
                    instructions.extend(self._wrap_parallel(group))
                else:
                    instructions.extend(group)
            else:
                instructions.append(group)
        return instructions
    
    def _generate_conditional(self,
                            ir: IRNode,
                            context: CompilationContext) -> List[FormalInstruction]:
        """Generate code for conditional execution."""
        return ir.instructions  # Implement conditional logic
    
    def _generate_loop(self,
                      ir: IRNode,
                      context: CompilationContext) -> List[FormalInstruction]:
        """Generate code for loop execution."""
        return ir.instructions  # Implement loop logic
    
    def _wrap_parallel(self,
                      instructions: List[FormalInstruction]) -> List[FormalInstruction]:
        """Wrap instructions for parallel execution."""
        return [
            FormalInstruction(
                name="parallel_block",
                type=InstructionType.CONTROL,
                metadata={
                    "parallel_instructions": instructions,
                    "is_parallel": True
                }
            )
        ]
