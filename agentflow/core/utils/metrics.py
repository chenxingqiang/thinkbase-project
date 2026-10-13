"""Metrics utilities for tracking inference performance."""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import time

@dataclass
class InferenceMetrics:
    """Metrics for tracking inference performance."""
    
    # Processing metrics
    total_processed: int = 0
    processing_time: float = 0.0
    average_latency: float = 0.0
    
    # Model metrics
    average_confidence: float = 0.0
    total_entities: int = 0
    inference_depth: int = 0
    
    # Resource metrics
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, metrics_dict: Dict[str, Any]) -> None:
        """Update metrics from a dictionary."""
        for key, value in metrics_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.custom_metrics[key] = value
    
    def reset(self) -> None:
        """Reset all metrics to default values."""
        for field_name in self.__dataclass_fields__:
            if field_name != "custom_metrics":
                setattr(self, field_name, 0)
        self.custom_metrics.clear()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        metrics_dict = {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
            if field_name != "custom_metrics"
        }
        metrics_dict.update(self.custom_metrics)
        return metrics_dict
    
    def calculate_average_latency(self) -> None:
        """Calculate average latency from total time and processed items."""
        if self.total_processed > 0:
            self.average_latency = self.processing_time / self.total_processed

@dataclass
class BatchMetrics:
    """Metrics for batch processing."""
    
    batch_size: int = 0
    batch_start_time: float = 0.0
    batch_processing_time: float = 0.0
    batch_memory_usage: float = 0.0
    batch_success_count: int = 0
    batch_error_count: int = 0
    
    def start_batch(self, batch_size: int) -> None:
        """Start tracking a new batch."""
        self.reset()
        self.batch_size = batch_size
        self.batch_start_time = time.time()
    
    def end_batch(self) -> None:
        """End batch tracking and calculate metrics."""
        self.batch_processing_time = time.time() - self.batch_start_time
    
    def reset(self) -> None:
        """Reset batch metrics."""
        self.batch_size = 0
        self.batch_start_time = 0.0
        self.batch_processing_time = 0.0
        self.batch_memory_usage = 0.0
        self.batch_success_count = 0
        self.batch_error_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch metrics to dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        }

@dataclass
class ModelMetrics:
    """Metrics for model performance tracking."""
    
    model_name: str = ""
    model_version: str = ""
    
    # Performance metrics
    inference_time: float = 0.0
    preprocessing_time: float = 0.0
    postprocessing_time: float = 0.0
    
    # Resource usage
    memory_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    
    # Model specific metrics
    parameters_count: int = 0
    flops: int = 0
    
    # Cache metrics
    cache_size: int = 0
    cache_hit_rate: float = 0.0
    
    def update_timing(
        self,
        inference_time: float,
        preprocessing_time: Optional[float] = None,
        postprocessing_time: Optional[float] = None
    ) -> None:
        """Update timing metrics."""
        self.inference_time = inference_time
        if preprocessing_time is not None:
            self.preprocessing_time = preprocessing_time
        if postprocessing_time is not None:
            self.postprocessing_time = postprocessing_time
    
    def update_resource_usage(
        self,
        memory_usage: Optional[float] = None,
        gpu_memory_usage: Optional[float] = None
    ) -> None:
        """Update resource usage metrics."""
        if memory_usage is not None:
            self.memory_usage = memory_usage
        if gpu_memory_usage is not None:
            self.gpu_memory_usage = gpu_memory_usage
    
    def update_cache_metrics(
        self,
        cache_size: int,
        hits: int,
        total_requests: int
    ) -> None:
        """Update cache-related metrics."""
        self.cache_size = cache_size
        self.cache_hit_rate = hits / total_requests if total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model metrics to dictionary."""
        return {
            field_name: getattr(self, field_name)
            for field_name in self.__dataclass_fields__
        } 

class MetricsCollector:
    """Accumulates named metric values over time.

    Used by the distributed and federated optimizers to gather
    per-worker / per-client measurements.
    """

    def __init__(self):
        self._values: Dict[str, List[Any]] = {}
        self._latest: Dict[str, Any] = {}

    def record(self, name: str, value: Any) -> None:
        """Record a value for the named metric."""
        self._values.setdefault(name, []).append(value)
        self._latest[name] = value

    def get_values(self, name: str) -> List[Any]:
        """Return all recorded values for the named metric."""
        return list(self._values.get(name, []))

    def get_metrics(self) -> Dict[str, Any]:
        """Return the latest value of every recorded metric."""
        return dict(self._latest)

    def summary(self) -> Dict[str, Dict[str, float]]:
        """Return per-metric min/mean/max summaries where numeric."""
        out = {}
        for name, vals in self._values.items():
            nums = [v for v in vals if isinstance(v, (int, float))]
            if nums:
                out[name] = {
                    "count": len(vals),
                    "min": min(nums),
                    "mean": sum(nums) / len(nums),
                    "max": max(nums),
                }
        return out

    def clear(self) -> None:
        """Drop all recorded values."""
        self._values = {}
        self._latest = {}
