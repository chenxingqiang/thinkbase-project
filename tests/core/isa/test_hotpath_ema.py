"""Unit tests for the EMA windowed hot-path detector and the tracked-path cap.

Covers the defect that hot-path detection keyed on exact sequence identity
never accumulated trajectory diversity and that hot-path state grew without
bound (main paper Section 4.2.2, supplementary Appendix D.4):

* an EMA-based windowed detector (``detection_mode="ema"``) that keys a
  path's hotness on its *recent execution share* (counter[h] / sum of
  counters, decay gamma = 2/(window+1)), so repeated paths inside a
  diverse trajectory stream become hot even though no single exact
  sequence ever crosses the absolute count threshold;
* an LRU cap on tracked paths (``max_tracked_paths``, default 1000)
  bounding every per-path mapping (counts, EMA counters, stats, hot
  paths).

Default behavior (exact identity matching, cap never reached) must stay
backward compatible with the shipped counter/threshold detector.
"""

import pytest
from typing import Any, Dict, List

from agentflow.core.isa.instruction import Instruction, InstructionType
from agentflow.core.isa.instruction_optimization import (
    DynamicOptimizer,
    OptimizationManager,
    OptimizationMetrics,
)
from agentflow.core.isa.agent_isa import AgentISA, AgentISAConfig


class TestInstruction(Instruction):
    """Test instruction for hot-path detection testing."""

    def __init__(self, id: str, name: str):
        super().__init__(
            id=id,
            name=name,
            type=InstructionType.CONTROL_FLOW,
            params={},
            description=f"Test instruction {name}"
        )

    async def execute(self, input_params: Dict[str, Any]) -> Any:
        return {"status": "executed", "id": self.id}


def _metrics(success_rate: float = 1.0) -> OptimizationMetrics:
    """Execution metrics, stable by default."""
    return OptimizationMetrics(
        execution_time=1.0,
        memory_usage=50.0,
        api_calls=1,
        success_rate=success_rate
    )


def _path(prefix: str, size: int = 3) -> List[Instruction]:
    """A distinct instruction sequence (distinct ids => distinct path)."""
    return [
        TestInstruction(f"{prefix}_{i}", f"{prefix}_{i}")
        for i in range(size)
    ]


def test_default_detector_is_exact_with_cap_1000():
    """Defaults must keep the shipped exact-identity behavior and cap."""
    optimizer = DynamicOptimizer()
    assert optimizer.detection_mode == "exact"
    assert optimizer.max_tracked_paths == 1000
    assert optimizer.hot_path_threshold == 100


def test_exact_mode_still_triggers_above_absolute_threshold():
    """The shipped counter/threshold rule is unchanged in exact mode."""
    optimizer = DynamicOptimizer()
    instructions = _path("classic")
    for _ in range(optimizer.hot_path_threshold + 1):
        optimizer.record_execution("classic", instructions, _metrics())
    assert "classic" in optimizer.hot_paths


def test_ema_decay_follows_window_formula():
    """gamma = 2/(window+1) per the specified detector (supplementary D.4)."""
    optimizer = DynamicOptimizer(detection_mode="ema", ema_window=10)
    assert optimizer.ema_decay == pytest.approx(2.0 / 11.0)


def test_diverse_stream_ema_triggers_where_exact_does_not():
    """Required regression: trajectory diversity accumulates under EMA.

    A stream interleaves three distinct trajectory variants, each far
    below the absolute count threshold (40 < 100).  Exact-identity
    counting never accumulates any variant past the threshold, so
    nothing becomes hot; the EMA detector sees each variant holding
    1/3 of the recent execution share (> theta = 0.3) and promotes
    all three.
    """
    variants = ["v1", "v2", "v3"]
    rounds = 40  # observations per variant, well below threshold 100

    exact = DynamicOptimizer()  # shipped default
    ema = DynamicOptimizer(
        detection_mode="ema", ema_window=10, ema_threshold=0.3
    )

    instructions = {v: _path(v) for v in variants}
    for _ in range(rounds):
        for variant in variants:
            for optimizer in (exact, ema):
                optimizer.record_execution(variant, instructions[variant],
                                           _metrics())

    # Exact identity matching: diversity never accumulates.
    assert exact.hot_paths == {}
    assert all(exact.execution_counts[v] == rounds for v in variants)

    # EMA: every variant's recent execution share (1/3) exceeds theta.
    assert set(ema.hot_paths) == set(variants)


def test_single_path_below_exact_threshold_triggers_ema_only():
    """A path repeated 30 times: EMA promotes it, exact counting does not."""
    exact = DynamicOptimizer()
    ema = DynamicOptimizer(detection_mode="ema", ema_threshold=0.3)
    instructions = _path("core")
    for _ in range(30):
        metrics = _metrics()
        exact.record_execution("core", instructions, metrics)
        ema.record_execution("core", instructions, metrics)

    assert "core" not in exact.hot_paths
    assert "core" in ema.hot_paths


def test_pure_diversity_never_becomes_hot_under_ema():
    """One-off trajectories must not accumulate into hot paths.

    A single recent observation (EMA counter == 1) is not a hot path
    even when its share of a young tracker exceeds theta.
    """
    optimizer = DynamicOptimizer(detection_mode="ema", ema_threshold=0.3)
    for i in range(50):
        optimizer.record_execution(f"p{i}", _path(f"p{i}"), _metrics())

    assert optimizer.hot_paths == {}
    assert len(optimizer.ema_counters) == 50


def test_ema_requires_stability():
    """High recent share alone is not enough: the stability gate applies."""
    flaky = DynamicOptimizer(detection_mode="ema", ema_threshold=0.3)
    stable = DynamicOptimizer(detection_mode="ema", ema_threshold=0.3)
    instructions = _path("flaky")
    for _ in range(10):
        flaky.record_execution("flaky", instructions,
                               _metrics(success_rate=0.5))
        stable.record_execution("flaky", instructions,
                                _metrics(success_rate=1.0))

    assert "flaky" not in flaky.hot_paths
    assert "flaky" in stable.hot_paths


def test_cap_bounds_all_per_path_state():
    """With a small cap, every per-path mapping stays bounded."""
    optimizer = DynamicOptimizer(max_tracked_paths=5)
    for i in range(12):
        optimizer.record_execution(f"p{i}", _path(f"p{i}"), _metrics())

    assert len(optimizer.execution_counts) <= 5
    assert len(optimizer.performance_stats) <= 5
    assert len(optimizer.hot_paths) <= 5
    # Most recently recorded paths survive; the oldest are evicted.
    assert "p11" in optimizer.execution_counts
    assert "p0" not in optimizer.execution_counts
    assert "p0" not in optimizer.performance_stats


def test_cap_evicts_least_recently_recorded_path():
    """Eviction is LRU, not FIFO: refreshing a path protects it."""
    optimizer = DynamicOptimizer(max_tracked_paths=3)
    for path_id in ("a", "b", "c"):
        optimizer.record_execution(path_id, _path(path_id), _metrics())
    # Refresh "a" so "b" becomes the least recently recorded path.
    optimizer.record_execution("a", _path("a"), _metrics())
    optimizer.record_execution("d", _path("d"), _metrics())

    assert set(optimizer.execution_counts) == {"a", "c", "d"}
    assert "b" not in optimizer.performance_stats
    assert optimizer.execution_counts["a"] == 2


def test_cap_bounds_ema_counters_and_hot_paths():
    """Hot paths promoted by the EMA detector are also LRU-bounded."""
    optimizer = DynamicOptimizer(
        detection_mode="ema",
        ema_window=10,
        ema_threshold=0.3,
        max_tracked_paths=4,
    )
    # Three interleaved variants each hold 1/3 of recent share: all hot.
    for _ in range(10):
        for variant in ("v1", "v2", "v3"):
            optimizer.record_execution(variant, _path(variant), _metrics())
    assert set(optimizer.hot_paths) == {"v1", "v2", "v3"}

    # A burst of new diverse paths evicts the stale variants.
    for i in range(5):
        optimizer.record_execution(f"u{i}", _path(f"u{i}"), _metrics())

    assert len(optimizer.ema_counters) <= 4
    assert len(optimizer.execution_counts) <= 4
    assert optimizer.hot_paths == {}


def test_evicted_path_restarts_ema_accumulation():
    """After LRU eviction, a path's EMA accumulation starts over."""
    optimizer = DynamicOptimizer(
        detection_mode="ema", ema_threshold=0.3, max_tracked_paths=2
    )
    instructions = _path("hot")
    for _ in range(5):
        optimizer.record_execution("hot", instructions, _metrics())
    assert "hot" in optimizer.hot_paths

    # Crowd "hot" out of the tracked set.
    for i in range(2):
        optimizer.record_execution(f"other{i}", _path(f"other{i}"), _metrics())
    assert "hot" not in optimizer.ema_counters
    assert "hot" not in optimizer.hot_paths

    # Re-recording starts a fresh counter: a single observation is not hot.
    optimizer.record_execution("hot", instructions, _metrics())
    assert optimizer.ema_counters["hot"] == pytest.approx(1.0)
    assert "hot" not in optimizer.hot_paths


def test_optimization_manager_forwards_detector_options():
    """The manager passes detection options through to the dynamic optimizer."""
    manager = OptimizationManager(
        hot_path_threshold=42,
        stability_threshold=0.85,
        detection_mode="ema",
        ema_window=5,
        ema_threshold=0.4,
        max_tracked_paths=7,
    )
    dynamic = manager.dynamic_optimizer
    assert dynamic.hot_path_threshold == 42
    assert dynamic.stability_threshold == 0.85
    assert dynamic.detection_mode == "ema"
    assert dynamic.ema_window == 5
    assert dynamic.ema_threshold == 0.4
    assert dynamic.max_tracked_paths == 7


def test_manager_end_to_end_ema_promotion():
    """Recording through the manager promotes the path in EMA mode."""
    manager = OptimizationManager(
        detection_mode="ema", ema_threshold=0.3
    )
    instructions = _path("mgr")
    result = None
    for _ in range(5):
        result = manager.record_execution(instructions, _metrics())

    assert manager.dynamic_optimizer.hot_paths
    assert isinstance(result, list)


def test_agent_isa_config_wires_detector():
    """AgentISAConfig fields reach the dynamic optimizer."""
    config = AgentISAConfig(
        hot_path_detection="ema",
        ema_window=8,
        ema_threshold=0.25,
        max_tracked_paths=11,
        hot_path_threshold=42,
        stability_threshold=0.85,
    )
    agent = AgentISA(config)
    dynamic = agent.optimization_manager.dynamic_optimizer
    assert dynamic.detection_mode == "ema"
    assert dynamic.ema_window == 8
    assert dynamic.ema_threshold == 0.25
    assert dynamic.max_tracked_paths == 11
    assert dynamic.hot_path_threshold == 42
    assert dynamic.stability_threshold == 0.85


def test_agent_isa_config_defaults_backward_compatible():
    """New config fields default to the shipped exact-mode behavior."""
    config = AgentISAConfig()
    assert config.hot_path_detection == "exact"
    assert config.ema_window == 10
    assert config.ema_threshold == 0.5
    assert config.max_tracked_paths == 1000

    agent = AgentISA()
    dynamic = agent.optimization_manager.dynamic_optimizer
    assert dynamic.detection_mode == "exact"
    assert dynamic.hot_path_threshold == 100
    assert dynamic.stability_threshold == 0.9


def test_invalid_detector_options_rejected():
    """Misconfiguration fails fast instead of silently misbehaving."""
    with pytest.raises(ValueError):
        DynamicOptimizer(detection_mode="windowed")
    with pytest.raises(ValueError):
        OptimizationManager(detection_mode="ema-ish")
    with pytest.raises(ValueError):
        DynamicOptimizer(max_tracked_paths=0)
    with pytest.raises(ValueError):
        DynamicOptimizer(detection_mode="ema", ema_window=0)
    with pytest.raises(ValueError):
        DynamicOptimizer(detection_mode="ema", ema_threshold=0.0)
    with pytest.raises(ValueError):
        DynamicOptimizer(detection_mode="ema", ema_threshold=1.5)
