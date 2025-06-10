"""Output specification processor for AgentFlow."""
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass

from litellm import Field

class OutputMode(Enum):
    RETURN = "RETURN"
    FORWARD = "FORWARD"
    STORE = "STORE"
    TRIGGER = "TRIGGER"

@dataclass
class TransformationConfig:
    enabled: bool = False
    methods: List[str] = Field(default_factory=list)

class OutputProcessor:
    """处理Agent输出的核心组件"""
    
    def __init__(self, output_spec: Dict[str, Any]):
        self.modes = [OutputMode(mode) for mode in output_spec.get("MODES", [])]
        self.strategies = output_spec.get("STRATEGIES", {})
        self.transformation = TransformationConfig(
            enabled=output_spec.get("TRANSFORMATION", {}).get("ENABLED", False),
            methods=output_spec.get("TRANSFORMATION", {}).get("METHODS", [])
        )
        
        # 注册转换方法
        self.transform_methods = {
            "FILTER": self._filter_output,
            "MAP": self._map_output,
            "REDUCE": self._reduce_output,
            "AGGREGATE": self._aggregate_output
        }

    def process_output(self, output_data: Any, mode: OutputMode) -> Dict[str, Any]:
        """处理输出数据
        
        Args:
            output_data: 输出数据
            mode: 输出模式
            
        Returns:
            处理后的输出数据
        """
        if mode not in self.modes:
            raise ValueError(f"Unsupported output mode: {mode}")
            
        # 应用转换
        if self.transformation.enabled:
            output_data = self._apply_transformations(output_data)
            
        processor = self._get_processor(mode)
        return processor(output_data)

    def _get_processor(self, mode: OutputMode) -> Callable[[Any], Dict[str, Any]]:
        """Get corresponding mode processor."""
        processors = {
            OutputMode.RETURN: self._process_return,
            OutputMode.FORWARD: self._process_forward,
            OutputMode.STORE: self._process_store,
            OutputMode.TRIGGER: self._process_trigger
        }
        if processor := processors.get(mode):
            return processor
        raise ValueError(f"No processor found for mode: {mode}")

    def _process_return(self, output_data: Any) -> Dict[str, Any]:
        """处理返回输出"""
        return_config = self.strategies.get("RETURN", {})
        options = return_config.get("options", ["FULL_RESULT"])
        
        if "SUMMARY" in options:
            return self._generate_summary(output_data)
        elif "PARTIAL_RESULT" in options:
            return self._extract_partial_result(output_data)
        else:
            # Ensure full string representation is returned
            return {
                "result": output_data,
                "summary": str(output_data)
            }

    def _process_forward(self, output_data: Any) -> Dict[str, Any]:
        """处理转发输出"""
        forward_config = self.strategies.get("FORWARD", {})
        routing = forward_config.get("routing_options", ["DIRECT_PASS"])
        
        if "TRANSFORM" in routing:
            output_data = self._transform_for_forward(output_data)
        elif "SELECTIVE_FORWARD" in routing:
            output_data = self._selective_forward(output_data)
            
        return {"forward_data": output_data}

    def _process_store(self, output_data: Any) -> Dict[str, Any]:
        """处理存储输出"""
        store_config = self.strategies.get("STORE", {})
        storage_types = store_config.get("storage_types", [])
        
        storage_data = {}
        for storage_type in storage_types:
            storage_data[storage_type] = self._prepare_for_storage(output_data, storage_type)
            
        return {"storage_data": storage_data}

    def _process_trigger(self, output_data: Any) -> Dict[str, Any]:
        """处理触发输出"""
        trigger_config = self.strategies.get("TRIGGER", {})
        trigger_types = trigger_config.get("trigger_types", [])
        
        triggers = {}
        for trigger_type in trigger_types:
            triggers[trigger_type] = self._prepare_trigger(output_data, trigger_type)
            
        return {"triggers": triggers}

    def _apply_transformations(self, data: Any) -> Any:
        """应用转换方法"""
        for method in self.transformation.methods:
            if method in self.transform_methods:
                data = self.transform_methods[method](data)
        return data

    def _filter_output(self, data: Any) -> Any:
        """过滤输出"""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if v is not None}
        return data

    def _map_output(self, data: Any) -> Any:
        """映射输出"""
        if isinstance(data, dict):
            return {k: str(v) for k, v in data.items()}
        return data

    def _reduce_output(self, data: Any) -> Any:
        """归约输出"""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in self.strategies.get("key_fields", [])}
        return data

    def _aggregate_output(self, data: Any) -> Any:
        """聚合输出"""
        if isinstance(data, list):
            return {"count": len(data), "items": data}
        return data

    def _generate_summary(self, data: Any) -> Dict[str, Any]:
        """生成输出摘要"""
        return {"summary": str(data)}

    def _extract_partial_result(self, data: Any) -> Dict[str, Any]:
        """提取部分结果"""
        if isinstance(data, dict):
            return {"partial": {k: data[k] for k in list(data.keys())[:5]}}
        return {"partial": data}

    def _transform_for_forward(self, data: Any) -> Any:
        """转换用于转发的数据"""
        return {"transformed": data}

    def _selective_forward(self, data: Any) -> Any:
        """选择性转发"""
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if k in self.strategies.get("forward_fields", [])}
        return data

    def _prepare_for_storage(self, data: Any, storage_type: str) -> Any:
        """准备用于存储的数据"""
        return {
            "type": storage_type,
            "data": data,
            "timestamp": "2024-01-19T00:00:00Z"  # 使用实际时间戳
        }

    def _prepare_trigger(self, data: Any, trigger_type: str) -> Any:
        """准备触发数据"""
        return {
            "type": trigger_type,
            "payload": data,
            "timestamp": "2024-01-19T00:00:00Z"  # 使用实际时间戳
        }
