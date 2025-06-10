"""Advanced Monitoring and Diagnostics System."""
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import time
from .formal import FormalInstruction, InstructionType

class MetricType(Enum):
    """Types of monitored metrics."""
    PERFORMANCE = "performance"
    RESOURCE = "resource"
    QUALITY = "quality"
    SECURITY = "security"
    RELIABILITY = "reliability"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class MetricThreshold:
    """Threshold configuration for metrics."""
    warning: float
    error: float
    critical: float
    window_size: int
    aggregation: str  # mean, max, min, percentile

class DiagnosticResult:
    """Results from diagnostic analysis."""
    
    def __init__(self):
        self.issues: List[Dict[str, Any]] = []
        self.recommendations: List[Dict[str, Any]] = []
        self.root_causes: List[Dict[str, Any]] = []
        
    def add_issue(self, issue: Dict[str, Any]):
        """Add identified issue."""
        self.issues.append(issue)
        
    def add_recommendation(self, recommendation: Dict[str, Any]):
        """Add recommendation for improvement."""
        self.recommendations.append(recommendation)
        
    def add_root_cause(self, cause: Dict[str, Any]):
        """Add identified root cause."""
        self.root_causes.append(cause)

class PerformanceMonitor:
    """Advanced performance monitoring system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history: Dict[str, List[Tuple[float, float]]] = {}
        self.thresholds: Dict[str, float] = {}
        self.alerts = []
        
    def collect_metrics(self, metrics: Dict[str, Any]) -> None:
        """Record metrics.
        
        Args:
            metrics: Metrics to record
        """
        timestamp = time.time()
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = []
            self.metrics_history[name].append((timestamp, float(value)))
            
            # Check thresholds
            if name in self.thresholds and value > self.thresholds[name]:
                self.alerts.append({
                    "metric": name,
                    "value": value,
                    "threshold": self.thresholds[name],
                    "timestamp": timestamp
                })
    
    def get_metrics(self,
                   metric_names: Optional[List[str]] = None,
                   window: Optional[int] = None) -> Dict[str, Any]:
        """Get metrics within specified window.
        
        Args:
            metric_names: Optional list of metric names to retrieve
            window: Optional time window in seconds
            
        Returns:
            Dictionary of metrics
        """
        current_time = time.time()
        metrics = {}
        
        for name in (metric_names or self.metrics_history.keys()):
            if name in self.metrics_history:
                values = self.metrics_history[name]
                if window:
                    values = [
                        (t, v) for t, v in values
                        if current_time - t <= window
                    ]
                metrics[name] = {
                    "current": values[-1][1] if values else None,
                    "mean": np.mean([v for _, v in values]) if values else None,
                    "max": max(v for _, v in values) if values else None,
                    "min": min(v for _, v in values) if values else None,
                    "std": np.std([v for _, v in values]) if values else None
                }
        return metrics
    
    def set_threshold(self, metric_name: str, threshold: float) -> None:
        """Set threshold for metric.
        
        Args:
            metric_name: Metric name
            threshold: Threshold value
        """
        self.thresholds[metric_name] = threshold
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get recorded alerts.
        
        Returns:
            List of alerts
        """
        return self.alerts.copy()
    
    def clear_metrics(self) -> None:
        """Clear all recorded metrics."""
        self.metrics_history.clear()
        self.alerts.clear()

class AnomalyDetector:
    """Anomaly detection system."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.history: Dict[str, List[float]] = {}
        self.window_size = config.get("anomaly_window", 100)
        self.threshold = config.get("anomaly_threshold", 3.0)
    
    def check_anomaly(self, metric_name: str, value: float) -> bool:
        """Check if value is anomalous."""
        if metric_name not in self.history:
            self.history[metric_name] = []
            
        self.history[metric_name].append(value)
        
        # Keep fixed window
        if len(self.history[metric_name]) > self.window_size:
            self.history[metric_name].pop(0)
            
        # Need minimum history
        if len(self.history[metric_name]) < 10:
            return False
            
        # Calculate z-score
        mean = np.mean(self.history[metric_name])
        std = np.std(self.history[metric_name])
        
        if std == 0:
            return False
            
        z_score = abs((value - mean) / std)
        return z_score > self.threshold

class DiagnosticEngine:
    """System diagnostics engine."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.performance_monitor = PerformanceMonitor(config)
        self.diagnostic_rules = self._initialize_rules()
        
    def diagnose(self,
                context: Dict[str, Any],
                focus_areas: Optional[List[str]] = None) -> DiagnosticResult:
        """Perform system diagnosis."""
        result = DiagnosticResult()
        
        # Get relevant metrics
        metrics = self.performance_monitor.get_metrics()
        
        # Apply diagnostic rules
        for rule in self._get_relevant_rules(focus_areas):
            if rule["condition"](metrics, context):
                result.add_issue({
                    "type": rule["type"],
                    "description": rule["description"],
                    "severity": rule["severity"]
                })
                
                # Add recommendations
                for rec in rule["recommendations"]:
                    result.add_recommendation(rec)
                
                # Analyze root cause
                root_cause = self._analyze_root_cause(
                    rule["type"],
                    metrics,
                    context
                )
                if root_cause:
                    result.add_root_cause(root_cause)
        
        return result
    
    def _initialize_rules(self) -> List[Dict[str, Any]]:
        """Initialize diagnostic rules."""
        return [
            {
                "type": "high_latency",
                "condition": lambda m, c: self._check_latency(m),
                "description": "System experiencing high latency",
                "severity": AlertSeverity.ERROR,
                "recommendations": [
                    {
                        "action": "optimize_pipeline",
                        "description": "Optimize instruction pipeline"
                    },
                    {
                        "action": "scale_resources",
                        "description": "Increase available resources"
                    }
                ]
            },
            {
                "type": "resource_exhaustion",
                "condition": lambda m, c: self._check_resources(m),
                "description": "Resource usage near capacity",
                "severity": AlertSeverity.CRITICAL,
                "recommendations": [
                    {
                        "action": "scale_up",
                        "description": "Increase resource limits"
                    },
                    {
                        "action": "optimize_usage",
                        "description": "Optimize resource usage"
                    }
                ]
            },
            {
                "type": "quality_degradation",
                "condition": lambda m, c: self._check_quality(m),
                "description": "Output quality degrading",
                "severity": AlertSeverity.WARNING,
                "recommendations": [
                    {
                        "action": "retrain",
                        "description": "Retrain models"
                    },
                    {
                        "action": "validate_input",
                        "description": "Improve input validation"
                    }
                ]
            }
        ]
    
    def _get_relevant_rules(self,
                          focus_areas: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Get rules relevant to focus areas."""
        if not focus_areas:
            return self.diagnostic_rules
        return [
            rule for rule in self.diagnostic_rules
            if rule["type"] in focus_areas
        ]
    
    def _analyze_root_cause(self,
                          issue_type: str,
                          metrics: Dict[str, Any],
                          context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze root cause of issue."""
        if issue_type == "high_latency":
            return self._analyze_latency_cause(metrics, context)
        elif issue_type == "resource_exhaustion":
            return self._analyze_resource_cause(metrics, context)
        elif issue_type == "quality_degradation":
            return self._analyze_quality_cause(metrics, context)
        return None
    
    def _check_latency(self, metrics: Dict[str, Any]) -> bool:
        """Check for latency issues."""
        if "latency" in metrics:
            return metrics["latency"]["current"] > self.config.get("max_latency", 1.0)
        return False
    
    def _check_resources(self, metrics: Dict[str, Any]) -> bool:
        """Check for resource issues."""
        if "resource_usage" in metrics:
            return metrics["resource_usage"]["current"] > 0.9
        return False
    
    def _check_quality(self, metrics: Dict[str, Any]) -> bool:
        """Check for quality issues."""
        if "quality_score" in metrics:
            return metrics["quality_score"]["current"] < 0.8
        return False
    
    def _analyze_latency_cause(self,
                             metrics: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cause of latency issues."""
        causes = []
        if metrics.get("cpu_usage", {}).get("current", 0) > 0.8:
            causes.append("high_cpu_usage")
        if metrics.get("memory_usage", {}).get("current", 0) > 0.8:
            causes.append("high_memory_usage")
        if metrics.get("io_wait", {}).get("current", 0) > 0.3:
            causes.append("io_bottleneck")
            
        return {
            "type": "latency",
            "causes": causes,
            "correlation": self._calculate_correlation(metrics, "latency")
        }
    
    def _analyze_resource_cause(self,
                              metrics: Dict[str, Any],
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cause of resource issues."""
        causes = []
        for resource in ["cpu", "memory", "disk", "network"]:
            if metrics.get(f"{resource}_usage", {}).get("current", 0) > 0.9:
                causes.append(f"high_{resource}_usage")
                
        return {
            "type": "resource",
            "causes": causes,
            "correlation": self._calculate_correlation(metrics, "resource_usage")
        }
    
    def _analyze_quality_cause(self,
                             metrics: Dict[str, Any],
                             context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cause of quality issues."""
        causes = []
        if metrics.get("error_rate", {}).get("current", 0) > 0.1:
            causes.append("high_error_rate")
        if metrics.get("accuracy", {}).get("current", 1) < 0.9:
            causes.append("low_accuracy")
            
        return {
            "type": "quality",
            "causes": causes,
            "correlation": self._calculate_correlation(metrics, "quality_score")
        }
    
    def _calculate_correlation(self,
                             metrics: Dict[str, Any],
                             target: str) -> Dict[str, float]:
        """Calculate correlation between metrics."""
        correlations = {}
        if target in metrics:
            target_values = [v for _, v in metrics[target]]
            for name, values in metrics.items():
                if name != target:
                    metric_values = [v for _, v in values]
                    if len(target_values) == len(metric_values):
                        correlation = np.corrcoef(target_values, metric_values)[0, 1]
                        correlations[name] = correlation
        return correlations
