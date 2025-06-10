"""Input specification processor for AgentFlow."""
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

from litellm import Field
from .exceptions import ValidationError

class InputMode(Enum):
    DIRECT_INPUT = "DIRECT_INPUT"
    CONTEXT_INJECTION = "CONTEXT_INJECTION" 
    STREAM_INPUT = "STREAM_INPUT"
    REFERENCE_INPUT = "REFERENCE_INPUT"

class InputType(Enum):
    DIRECT = "DIRECT"
    CONTEXT = "CONTEXT"
    STREAM = "STREAM"
    REFERENCE = "REFERENCE"

@dataclass
class ValidationConfig:
    strict_mode: bool = False
    schema_validation: bool = True
    transform_strategies: List[str] = Field(default_factory=list)

class InputProcessor:
    """处理Agent输入的核心组件"""
    
    def __init__(self, input_spec: Dict[str, Any]):
        self.modes = [InputMode(mode) for mode in input_spec.get("MODES", [])]
        self.types = input_spec.get("TYPES", {})
        self.validation = ValidationConfig(
            strict_mode=input_spec.get("VALIDATION", {}).get("STRICT_MODE", False),
            schema_validation=input_spec.get("VALIDATION", {}).get("SCHEMA_VALIDATION", True),
            transform_strategies=input_spec.get("VALIDATION", {}).get("TRANSFORM_STRATEGIES", [])
        )

    def process_input(self, input_data: Any, mode: InputMode) -> Dict[str, Any]:
        """处理输入数据
        
        Args:
            input_data: 输入数据
            mode: 输入模式
            
        Returns:
            处理后的输入数据
        """
        if mode not in self.modes:
            raise ValueError(f"Unsupported input mode: {mode}")
            
        processor = self._get_processor(mode)
        if processor is None:
            raise ValueError(f"No processor found for mode: {mode}")
        return processor(input_data)

    def _get_processor(self, mode: InputMode):
        """获取对应模式的处理器"""
        processors = {
            InputMode.DIRECT_INPUT: self._process_direct_input,
            InputMode.CONTEXT_INJECTION: self._process_context_injection,
            InputMode.STREAM_INPUT: self._process_stream_input,
            InputMode.REFERENCE_INPUT: self._process_reference_input
        }
        return processors.get(mode)

    def _process_direct_input(self, input_data: Any) -> Dict[str, Any]:
        """处理直接输入"""
        if self.validation.strict_mode:
            self._validate_direct_input(input_data)
        return {"direct_data": input_data}

    def _process_context_injection(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """处理上下文注入"""
        context_config = self.types.get("CONTEXT", {})
        allowed_sources = context_config.get("sources", [])
        
        if not all(source in allowed_sources for source in context.keys()):
            raise ValidationError("Invalid context source")
            
        return context

    def _process_stream_input(self, stream_data: Any) -> Dict[str, Any]:
        """处理流式输入"""
        stream_config = self.types.get("STREAM", {})
        modes = stream_config.get("modes", [])
        
        if not modes:
            raise ValidationError("No stream modes configured")
            
        return {"stream_data": stream_data}

    def _process_reference_input(self, reference: str) -> Dict[str, Any]:
        """处理引用输入"""
        reference_config = self.types.get("REFERENCE", {})
        allowed_types = reference_config.get("types", [])
        
        if not any(ref_type in reference for ref_type in allowed_types):
            raise ValidationError("Invalid reference type")
            
        return {"reference": reference}

    def _validate_direct_input(self, input_data: Any):
        """验证直接输入"""
        if self.validation.schema_validation:
            # 实现具体的schema验证逻辑
            pass
            
        if "TYPE_COERCION" in self.validation.transform_strategies:
            # 实现类型转换逻辑
            pass
