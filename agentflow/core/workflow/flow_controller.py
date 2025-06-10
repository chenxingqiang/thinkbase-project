"""Data flow controller for AgentFlow."""
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import logging

class ErrorStrategy(Enum):
    SKIP = "SKIP"
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    COMPENSATE = "COMPENSATE"

@dataclass
class RouteCondition:
    when: str
    action: str

@dataclass
class ErrorConfig:
    strategies: List[ErrorStrategy]
    max_retries: int = 3

class FlowController:
    """控制Agent数据流的核心组件"""
    
    def __init__(self, flow_config: Dict[str, Any]):
        self.default_behavior = flow_config.get("ROUTING_RULES", {}).get("DEFAULT_BEHAVIOR", "FORWARD_ALL")
        self.conditions = [
            RouteCondition(**cond) 
            for cond in flow_config.get("ROUTING_RULES", {}).get("CONDITIONAL_ROUTING", {}).get("CONDITIONS", [])
        ]
        
        error_config = flow_config.get("ERROR_HANDLING", {})
        self.error_config = ErrorConfig(
            strategies=[ErrorStrategy(s) for s in error_config.get("STRATEGIES", [])],
            max_retries=error_config.get("MAX_RETRIES", 3)
        )
        
        self.logger = logging.getLogger(__name__)
        self._retry_count = 0

    def route_data(self, data: Any) -> Dict[str, Any]:
        """路由数据流
        
        Args:
            data: 需要路由的数据
            
        Returns:
            路由结果
        """
        try:
            if self.conditions:
                return self._apply_routing_rules(data)
            return self._apply_default_behavior(data)
        except Exception as e:
            return self._handle_error(e, data)

    def _apply_routing_rules(self, data: Any) -> Dict[str, Any]:
        """应用路由规则"""
        for condition in self.conditions:
            if self._evaluate_condition(condition.when, data):
                return self._execute_action(condition.action, data)
                
        return self._apply_default_behavior(data)

    def _apply_default_behavior(self, data: Any) -> Dict[str, Any]:
        """应用默认行为"""
        if self.default_behavior == "FORWARD_ALL":
            return {"forward": data}
        elif self.default_behavior == "FILTER":
            return self._filter_data(data)
        else:
            return {"data": data}

    def _evaluate_condition(self, condition: str, data: Any) -> bool:
        """评估条件
        
        Args:
            condition: 条件表达式
            data: 评估数据
            
        Returns:
            条件评估结果
        """
        try:
            # 这里可以实现更复杂的条件评估逻辑
            if isinstance(data, dict):
                return eval(condition, {"data": data})
            return False
        except Exception as e:
            self.logger.error(f"Error evaluating condition: {str(e)}")
            return False

    def _execute_action(self, action: str, data: Any) -> Dict[str, Any]:
        """执行路由动作
        
        Args:
            action: 动作描述
            data: 处理数据
            
        Returns:
            处理结果
        """
        actions = {
            "FORWARD": lambda x: {"forward": x},
            "TRANSFORM": self._transform_data,
            "FILTER": self._filter_data,
            "AGGREGATE": self._aggregate_data
        }
        
        action_func = actions.get(action)
        if action_func:
            return action_func(data)
        else:
            raise ValueError(f"Unsupported action: {action}")

    def _handle_error(self, error: Exception, data: Any) -> Dict[str, Any]:
        """错误处理
        
        Args:
            error: 异常对象
            data: 相关数据
            
        Returns:
            错误处理结果
        """
        for strategy in self.error_config.strategies:
            try:
                if strategy == ErrorStrategy.SKIP:
                    return self._skip_error(error)
                elif strategy == ErrorStrategy.RETRY:
                    return self._retry_operation(data)
                elif strategy == ErrorStrategy.FALLBACK:
                    return self._use_fallback(data)
                elif strategy == ErrorStrategy.COMPENSATE:
                    return self._compensate_error(error, data)
            except Exception as e:
                self.logger.error(f"Error handling strategy {strategy} failed: {str(e)}")
                continue
                
        raise error

    def _skip_error(self, error: Exception) -> Dict[str, Any]:
        """跳过错误"""
        self.logger.warning(f"Skipping error: {str(error)}")
        return {"status": "skipped", "error": str(error)}

    def _retry_operation(self, data: Any) -> Dict[str, Any]:
        """重试操作"""
        if self._retry_count >= self.error_config.max_retries:
            raise ValueError("Max retries exceeded")
            
        self._retry_count += 1
        return self.route_data(data)

    def _use_fallback(self, data: Any) -> Dict[str, Any]:
        """使用后备方案"""
        return {
            "status": "fallback",
            "data": self._get_fallback_data(data)
        }

    def _compensate_error(self, error: Exception, data: Any) -> Dict[str, Any]:
        """错误补偿"""
        return {
            "status": "compensated",
            "original_error": str(error),
            "compensated_data": self._get_compensation_data(data)
        }

    def _transform_data(self, data: Any) -> Dict[str, Any]:
        """转换数据"""
        if isinstance(data, dict):
            return {"transformed": {k: str(v) for k, v in data.items()}}
        return {"transformed": str(data)}

    def _filter_data(self, data: Any) -> Dict[str, Any]:
        """过滤数据"""
        if isinstance(data, dict):
            return {"filtered": {k: v for k, v in data.items() if v is not None}}
        return {"filtered": data}

    def _aggregate_data(self, data: Any) -> Dict[str, Any]:
        """聚合数据"""
        if isinstance(data, list):
            return {"aggregated": len(data)}
        return {"aggregated": data}

    def _get_fallback_data(self, data: Any) -> Any:
        """获取后备数据"""
        if isinstance(data, dict):
            return {k: None for k in data.keys()}
        return None

    def _get_compensation_data(self, data: Any) -> Any:
        """获取补偿数据"""
        if isinstance(data, dict):
            return {k: "compensated" for k in data.keys()}
        return "compensated"
