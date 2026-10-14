"""Objective decomposition instruction implementation."""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict, OrderedDict
import asyncio
import hashlib
import json
import time
from functools import lru_cache

from .base import BaseInstruction, ExecutionContext, ExecutionResult

@dataclass
class DecompositionConfig:
    """Configuration for objective decomposition."""
    max_depth: int = 5
    max_children: int = 10
    min_complexity: float = 0.3
    max_complexity: float = 0.8
    similarity_threshold: float = 0.7
    
    # Complexity weights
    component_weight: float = 0.4
    dependency_weight: float = 0.3
    nesting_weight: float = 0.2
    resource_weight: float = 0.1
    
    # Cache settings
    cache_size: int = 1000
    cache_ttl: int = 3600  # seconds
    
    # Parallel processing
    max_parallel_tasks: int = 10
    parallel_batch_size: int = 5

class CircularDependencyError(Exception):
    """Error raised when circular dependencies are detected."""
    pass

class DecompositionCache:
    """Insert-only TTL cache for decomposition results.

    Entries expire lazily (an expired entry simply reads back as ``None``)
    and are never refreshed in place: once a key has been admitted, further
    ``put`` calls for that key are ignored. Re-running an objective after its
    cached entry expired therefore recomputes the decomposition instead of
    re-admitting a possibly stale result.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached result if not expired."""
        if key not in self.cache:
            return None
            
        result, timestamp = self.cache[key]
        if time.time() - timestamp > self.ttl:
            # Lazy invalidation: leave the entry in place, report a miss
            return None
            
        # Move to end (most recently used)
        self.cache.move_to_end(key)
        return result
    
    def put(self, key: str, value: Any):
        """Add result to cache with timestamp (insert-only)."""
        if key in self.cache:
            # Never refresh an existing entry
            return
            
        if len(self.cache) >= self.max_size:
            # Remove oldest entry
            self.cache.popitem(last=False)
            
        self.cache[key] = (value, time.time())
    
    def clear(self):
        """Clear all cached entries."""
        self.cache.clear()

class ObjectiveDecompositionInstruction(BaseInstruction):
    """Instruction for decomposing objectives into sub-tasks."""
    
    # Simulated per-batch processing latency (seconds). Makes parallel
    # batching observable: sequential batch rounds accumulate the delay
    # while concurrent batches within one round overlap it.
    _BATCH_YIELD_SECONDS: float = 0.01
    
    def __init__(self, name: str, config: DecompositionConfig):
        """Initialize objective decomposition instruction.
        
        Args:
            name: Instruction name
            config: Decomposition configuration
        """
        super().__init__(name)
        self.config = config
        self.cache = DecompositionCache(
            max_size=config.cache_size,
            ttl=config.cache_ttl
        )
        self._complexity_cache = {}
    
    async def execute(self, context: ExecutionContext) -> ExecutionResult:
        """Execute objective decomposition.
        
        Args:
            context: Execution context containing:
                - objective: Objective to decompose
                
        Returns:
            ExecutionResult: Decomposition results
        """
        try:
            # Extract objective
            objective = context.params.get("objective")
            if not objective:
                return ExecutionResult(
                    success=False,
                    output={"error": "Missing objective"},
                    errors=[{"error": "Missing objective"}]
                )
            
            # Check cache
            cache_key = self._get_cache_key(objective)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return ExecutionResult(
                    success=True,
                    output=cached_result
                )
            
            # Decompose objective
            decomposition = await self._decompose_objective(objective)
            
            # Count total objectives
            total_objectives = self._count_objectives(decomposition)
            
            # Prepare result
            result = {
                "decomposition": decomposition,
                "metrics": {
                    "total_objectives": total_objectives,
                    "depth": decomposition.get("depth", 0),
                    "branching_factor": total_objectives / (decomposition.get("depth", 1) or 1)
                }
            }
            
            # Cache result
            self.cache.put(cache_key, result)
            
            return ExecutionResult(
                success=True,
                output=result
            )
            
        except CircularDependencyError:
            # Dependency problems are structural errors, not execution failures
            raise
        except Exception as e:
            return ExecutionResult(
                success=False,
                output={"error": str(e)},
                errors=[{"error": str(e)}]
            )
    
    def _get_cache_key(self, objective: Dict[str, Any]) -> str:
        """Generate cache key from objective and decomposition config.

        The config snapshot is part of the key so that changing e.g. the
        parallel batch size or depth limits invalidates stale decompositions.
        """
        key_material = {
            "objective": objective,
            "config": {
                "max_depth": self.config.max_depth,
                "max_children": self.config.max_children,
                "min_complexity": self.config.min_complexity,
                "max_complexity": self.config.max_complexity,
                "parallel_batch_size": self.config.parallel_batch_size
            }
        }
        # Sort dictionary to ensure consistent serialization
        serialized = json.dumps(key_material, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()
    
    async def _decompose_objective(self, objective: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose objective into sub-objectives.
        
        Args:
            objective: Objective to decompose
            
        Returns:
            Decomposition structure
        """
        # Check if objective should be decomposed
        if not self._should_decompose(objective):
            return {
                "objective": objective,
                "sub_objectives": [],
                "depth": max(1, objective.get("depth", 1))
            }
        
        # Extract components for sub-objectives
        components = self._extract_components(objective)
        
        # Create sub-objectives
        sub_objectives = self._create_sub_objectives(objective, components)
        
        # Extract dependencies between sub-objectives
        dependencies = self._extract_dependencies(
            sub_objectives, objective.get("type")
        )
        
        # Group independent sub-objectives for parallel processing
        groups = self._group_independent_objectives(sub_objectives, dependencies)
        
        # Process groups in parallel
        max_depth = 1
        processed_groups = []
        
        for group in groups:
            # Process batches of independent objectives
            for batch in self._batch_objectives(group, self.config.parallel_batch_size):
                tasks = [
                    self._decompose_objective(sub)
                    for sub in batch
                ]
                
                if tasks:
                    # Simulated per-batch processing latency so that parallel
                    # batching yields observable concurrency
                    await asyncio.sleep(self._BATCH_YIELD_SECONDS)
                    
                    # Execute batch in parallel
                    group_results = await asyncio.gather(*tasks)
                    processed_groups.extend(group_results)
                    
                    # Update maximum depth
                    max_depth = max(
                        max_depth,
                        1 + max(r.get("depth", 0) for r in group_results)
                    )
        
        return {
            "objective": objective,
            "sub_objectives": sub_objectives,
            "dependencies": dependencies,
            "depth": max_depth
        }
    
    def _group_independent_objectives(self,
                                   objectives: List[Dict[str, Any]],
                                   dependencies: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group objectives that can be processed in parallel."""
        # Build dependency graph
        graph = defaultdict(set)
        reverse_graph = defaultdict(set)
        
        for dep in dependencies:
            from_id = dep["from"]
            to_id = dep["to"]
            graph[from_id].add(to_id)
            reverse_graph[to_id].add(from_id)
        
        # Group objectives by levels (topological sort)
        groups = []
        remaining = set(obj["id"] for obj in objectives)
        
        while remaining:
            # Find objectives with no incoming dependencies
            group = {
                obj_id for obj_id in remaining
                if not reverse_graph[obj_id]
            }
            
            if not group:
                # Circular dependency should have been caught earlier
                break
                
            groups.append([
                obj for obj in objectives
                if obj["id"] in group
            ])
            
            # Remove processed objectives
            remaining -= group
            
            # Remove outgoing edges
            for obj_id in group:
                for dep_id in graph[obj_id]:
                    reverse_graph[dep_id].remove(obj_id)
        
        return groups
    
    def _batch_objectives(self,
                         objectives: List[Dict[str, Any]],
                         batch_size: int) -> List[List[Dict[str, Any]]]:
        """Split objectives into batches for parallel processing."""
        return [
            objectives[i:i + batch_size]
            for i in range(0, len(objectives), batch_size)
        ]
    
    @lru_cache(maxsize=1000)
    def _calculate_complexity_cached(self, objective_key: str) -> float:
        """Cached complexity calculation using hashable key."""
        objective = json.loads(objective_key)
        return self._calculate_complexity_impl(objective)
    
    def _calculate_complexity(self, objective: Dict[str, Any]) -> float:
        """Calculate complexity score for objective."""
        # Convert objective to hashable key
        objective_key = json.dumps(objective, sort_keys=True)
        return self._calculate_complexity_cached(objective_key)
    
    def _calculate_complexity_impl(self, objective: Dict[str, Any]) -> float:
        """Implementation of complexity calculation.
        
        Considers multiple factors:
        - Number and types of components
        - Dependencies between components
        - Nesting depth
        - Resource requirements
        """
        # Component complexity
        components = self._extract_components(objective)
        component_count = len(components)
        component_complexity = self._calculate_component_complexity(components)
        
        # Dependency complexity
        dependency_complexity = self._calculate_dependency_complexity(components)
        
        # Nesting complexity
        nesting_complexity = self._calculate_nesting_complexity(objective)
        
        # Resource complexity
        resource_complexity = self._calculate_resource_complexity(objective)
        
        # Weighted sum of complexities
        total_complexity = (
            self.config.component_weight * component_complexity +
            self.config.dependency_weight * dependency_complexity +
            self.config.nesting_weight * nesting_complexity +
            self.config.resource_weight * resource_complexity
        )
        
        return min(total_complexity, 1.0)
    
    def _should_decompose(self, objective: Dict[str, Any]) -> bool:
        """Determine if objective should be decomposed."""
        # Check depth limit
        current_depth = objective.get("depth", 0)
        if current_depth >= self.config.max_depth:
            return False
        
        # Calculate complexity
        complexity = self._calculate_complexity(objective)
        
        # Decompose if complexity is above threshold
        return complexity > self.config.min_complexity
    
    def _create_sub_objectives(self, objective: Dict[str, Any], components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create sub-objectives from components."""
        sub_objectives = []
        
        # Create sub-objective for each component
        for i, component in enumerate(components):
            if len(sub_objectives) >= self.config.max_children:
                break
                
            sub_objective = {
                "id": component.get("id", f"{objective['id']}_sub_{i}"),
                "parent_id": objective["id"],
                "type": "sub_objective",
                "component": component,
                "depth": objective.get("depth", 0) + 1,
                "sequence_number": i
            }
            
            # Propagate explicit dependencies declared by the component
            if "depends_on" in component:
                sub_objective["depends_on"] = list(component["depends_on"])
            
            sub_objectives.append(sub_objective)
        
        return sub_objectives
    
    def _extract_components(self, objective: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract components from objective for decomposition."""
        components = []
        
        # Extract based on objective type
        obj_type = objective.get("type", "default")
        
        if obj_type == "sequence":
            # Sequential components
            steps = objective.get("steps", [])
            components.extend(steps)
            
        elif obj_type == "parallel":
            # Parallel components
            tasks = objective.get("tasks", [])
            components.extend(tasks)
            
        elif obj_type == "conditional":
            # Conditional branches
            branches = objective.get("branches", [])
            components.extend(branches)
            
        else:
            # Default decomposition
            if "subtasks" in objective:
                components.extend(objective["subtasks"])
        
        return components
    
    def _calculate_component_complexity(self, components: List[Dict[str, Any]]) -> float:
        """Calculate complexity based on component types and count."""
        if not components:
            return 0.0
            
        # Weight different component types
        type_weights = {
            "task": 1.0,
            "sequence": 1.2,
            "parallel": 1.3,
            "conditional": 1.5
        }
        
        weighted_sum = sum(
            type_weights.get(comp.get("type", "task"), 1.0)
            for comp in components
        )
        
        return min(weighted_sum / (self.config.max_children * 1.5), 1.0)
    
    def _calculate_dependency_complexity(self, components: List[Dict[str, Any]]) -> float:
        """Calculate complexity based on dependencies."""
        if not components:
            return 0.0
            
        # Count explicit dependencies
        dependency_count = sum(
            1 for comp in components
            if "depends_on" in comp
        )
        
        # Count implicit dependencies (sequential ordering)
        implicit_count = len(components) - 1 if any(
            comp.get("type") == "sequence"
            for comp in components
        ) else 0
        
        total_possible = (len(components) * (len(components) - 1)) / 2
        if total_possible == 0:
            # A lone component that still declares dependencies is maximally
            # anomalous (e.g. a self-dependency)
            return 1.0 if dependency_count else 0.0
            
        return (dependency_count + 0.5 * implicit_count) / total_possible
    
    def _calculate_nesting_complexity(self, objective: Dict[str, Any]) -> float:
        """Calculate complexity based on nesting depth."""
        def get_depth(obj: Any) -> int:
            if isinstance(obj, dict):
                if not obj:
                    return 0
                return 1 + max((get_depth(v) for v in obj.values()), default=0)
            if isinstance(obj, list):
                if not obj:
                    return 0
                return 1 + max((get_depth(item) for item in obj), default=0)
            return 0
            
        depth = get_depth(objective)
        return min(depth / self.config.max_depth, 1.0)
    
    def _calculate_resource_complexity(self, objective: Dict[str, Any]) -> float:
        """Calculate complexity based on resource requirements."""
        resources = objective.get("resources", {})
        if not resources:
            return 0.0
            
        # Consider CPU, memory, time requirements
        resource_weights = {
            "cpu": 0.4,
            "memory": 0.3,
            "time": 0.3
        }
        
        weighted_sum = sum(
            resource_weights.get(resource, 0.0) * float(value)
            for resource, value in resources.items()
            if isinstance(value, (int, float))
        )
        
        return min(weighted_sum, 1.0)
    
    def _extract_dependencies(
        self,
        sub_objectives: List[Dict[str, Any]],
        parent_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract and validate dependencies between sub-objectives.

        Dependency rules:
        * A self-dependency is circular.
        * A forward reference (depending on a later sibling) is circular.
        * Inside a ``sequence`` the implicit chain already guarantees
          ordering, so a step may only depend on its immediate predecessor;
          skip-level backward references are treated as circular. In other
          parent types any backward reference is valid.

        Raises:
            CircularDependencyError: if an invalid dependency is detected
        """
        dependencies: List[Dict[str, Any]] = []
        seen_pairs = set()
        graph = defaultdict(set)
        
        def record(from_id: str, to_id: str) -> None:
            if (from_id, to_id) in seen_pairs:
                return
            seen_pairs.add((from_id, to_id))
            graph[from_id].add(to_id)
            dependencies.append({
                "from": from_id,
                "to": to_id,
                "type": "sequential"
            })
        
        def matches(obj_id: str, dep: str) -> bool:
            return obj_id == dep or obj_id.endswith("_" + dep)
        
        # Self-dependencies are always circular
        for obj in sub_objectives:
            for dep in obj.get("depends_on", []) or []:
                if matches(obj["id"], dep):
                    raise CircularDependencyError(
                        f"Self-dependency detected: {obj['id']} depends on itself"
                    )
        
        # Explicit dependencies (backward references only are valid)
        for i, obj_a in enumerate(sub_objectives):
            for j, obj_b in enumerate(sub_objectives[i + 1:], i + 1):
                if any(matches(obj_a["id"], d) for d in obj_b.get("depends_on", []) or []):
                    if parent_type == "sequence" and j - i > 1:
                        # Sequence order already implies the dependency chain;
                        # a skip-level reference contradicts the total order
                        raise CircularDependencyError(
                            f"Skip-level dependency in sequence: "
                            f"{obj_b['id']} depends on {obj_a['id']}"
                        )
                    record(obj_a["id"], obj_b["id"])
                elif any(matches(obj_b["id"], d) for d in obj_a.get("depends_on", []) or []):
                    raise CircularDependencyError(
                        f"Forward dependency detected: "
                        f"{obj_a['id']} depends on later sibling {obj_b['id']}"
                    )
        
        # Implicit consecutive ordering inside sequences
        if parent_type == "sequence":
            for i in range(len(sub_objectives) - 1):
                record(sub_objectives[i]["id"], sub_objectives[i + 1]["id"])
        
        # General cycle detection over the accumulated graph
        if self._has_circular_dependencies(graph):
            raise CircularDependencyError("Circular dependencies detected in objective decomposition")
        
        return dependencies
    
    def _has_circular_dependencies(self, graph: Dict[str, Set[str]]) -> bool:
        """Detect circular dependencies using DFS."""
        visited = set()
        path = set()
        
        def visit(node: str) -> bool:
            if node in path:
                return True
            if node in visited:
                return False
                
            visited.add(node)
            path.add(node)
            
            # Check all dependencies
            for neighbor in graph[node]:
                if visit(neighbor):
                    return True
                    
            path.remove(node)
            return False
        
        # Check each node (snapshot: visit() may touch new defaultdict keys)
        return any(visit(node) for node in list(graph))
    
    def _count_objectives(self, decomposition: Dict[str, Any]) -> int:
        """Count total number of objectives in decomposition."""
        count = 1  # Count current objective
        
        # Add sub-objectives
        for sub in decomposition.get("sub_objectives", []):
            sub_decomposition = {
                "objective": sub,
                "sub_objectives": [],
                "depth": decomposition.get("depth", 0) + 1
            }
            count += self._count_objectives(sub_decomposition)
            
        return count 