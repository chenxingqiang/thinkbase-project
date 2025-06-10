"""Dynamic context manager for Agent LLM interactions with CO-STAR framework support."""
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
import json
from datetime import datetime
from enum import Enum

class COSTARComponent(Enum):
    """CO-STAR framework components"""
    CONTEXT = "context"
    OBJECTIVE = "objective"
    STYLE = "style"
    TONE = "tone"
    AUDIENCE = "audience"
    RESPONSE = "response"

@dataclass
class COSTARContext:
    """CO-STAR framework context"""
    component: COSTARComponent
    content: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class MessageContext:
    """消息上下文"""
    role: str
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    costar_context: Optional[COSTARContext] = None

@dataclass
class ConversationMemory:
    """对话记忆"""
    messages: List[MessageContext] = field(default_factory=list)
    max_tokens: int = 4000
    memory_window: int = 10

class ContextManager:
    """Dynamic context manager for Agent LLM interactions with CO-STAR framework support."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory = ConversationMemory(
            max_tokens=config.get("max_tokens", 4000),
            memory_window=config.get("memory_window", 10)
        )
        self.system_prompt = config.get("system_prompt", "")
        self.context_variables = {}
        self.costar_contexts: Dict[COSTARComponent, List[COSTARContext]] = {
            component: [] for component in COSTARComponent
        }
        self.context: Dict[str, Any] = {}
        
    def add_costar_context(self, component: Union[COSTARComponent, str], content: Any, metadata: Optional[Dict[str, Any]] = None):
        """添加CO-STAR上下文
        
        Args:
            component: CO-STAR组件
            content: 上下文内容
            metadata: 元数据
        """
        if isinstance(component, str):
            component = COSTARComponent(component.lower())
            
        context = COSTARContext(
            component=component,
            content=content,
            metadata=metadata or {}
        )
        self.costar_contexts[component].append(context)
        
    def get_costar_context(self, component: Union[COSTARComponent, str]) -> List[COSTARContext]:
        """获取指定CO-STAR组件的上下文
        
        Args:
            component: CO-STAR组件
            
        Returns:
            组件上下文列表
        """
        if isinstance(component, str):
            component = COSTARComponent(component.lower())
        return self.costar_contexts[component]
        
    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None, costar_context: Optional[COSTARContext] = None):
        """添加消息到上下文
        
        Args:
            role: 消息角色
            content: 消息内容
            metadata: 消息元数据
            costar_context: CO-STAR上下文
        """
        message = MessageContext(
            role=role,
            content=content,
            metadata=metadata or {},
            costar_context=costar_context
        )
        self.memory.messages.append(message)
        self._trim_memory()
        
    def get_context(self, include_system: bool = True, include_costar: bool = True) -> List[Dict[str, Any]]:
        """获取格式化的上下文
        
        Args:
            include_system: 是否包含系统提示词
            include_costar: 是否包含CO-STAR上下文
            
        Returns:
            格式化的上下文列表
        """
        context = []
        
        # 添加系统提示词
        if include_system and self.system_prompt:
            context.append({
                "role": "system",
                "content": self._format_system_prompt()
            })
            
        # 添加CO-STAR上下文
        if include_costar:
            for component in COSTARComponent:
                contexts = self.costar_contexts[component]
                if contexts:
                    context.append({
                        "role": "system",
                        "content": f"[{component.value.upper()}]\n" + 
                                 "\n".join(str(ctx.content) for ctx in contexts)
                    })
            
        # 添加历史消息
        for message in self.memory.messages[-self.memory.memory_window:]:
            msg_context: Dict[str, Union[str, Dict[str, Any]]] = {
                "role": message.role,
                "content": message.content
            }
            if message.costar_context and include_costar:
                msg_context["costar_context"] = {
                    "component": message.costar_context.component.value,
                    "content": message.costar_context.content,
                    "metadata": message.costar_context.metadata
                }
            context.append(msg_context)
            
        return context
        
    def update_system_prompt(self, prompt: str):
        """更新系统提示词
        
        Args:
            prompt: 新的系统提示词
        """
        self.system_prompt = prompt
        
    def set_context_variable(self, key: str, value: Any):
        """设置上下文变量
        
        Args:
            key: 变量名
            value: 变量值
        """
        self.context_variables[key] = value
        
    def get_context_variable(self, key: str) -> Any:
        """获取上下文变量
        
        Args:
            key: 变量名
            
        Returns:
            变量值
        """
        return self.context_variables.get(key)
        
    def clear_memory(self):
        """清空对话记忆"""
        self.memory.messages.clear()
        
    def _trim_memory(self):
        """裁剪对话记忆以符合token限制"""
        while len(self.memory.messages) > self.memory.memory_window:
            self.memory.messages.pop(0)
            
    def _format_system_prompt(self) -> str:
        """格式化系统提示词"""
        prompt = self.system_prompt
        
        # 替换上下文变量
        for key, value in self.context_variables.items():
            placeholder = f"{{{key}}}"
            if placeholder in prompt:
                prompt = prompt.replace(placeholder, str(value))
                
        return prompt
        
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "system_prompt": self.system_prompt,
            "context_variables": self.context_variables,
            "memory": {
                "messages": [
                    {
                        "role": msg.role,
                        "content": msg.content,
                        "timestamp": msg.timestamp,
                        "metadata": msg.metadata,
                        "costar_context": {
                            "component": msg.costar_context.component.value,
                            "content": msg.costar_context.content,
                            "metadata": msg.costar_context.metadata
                        } if msg.costar_context else None
                    }
                    for msg in self.memory.messages
                ],
                "max_tokens": self.memory.max_tokens,
                "memory_window": self.memory.memory_window
            },
            "costar_contexts": {
                component.value: [
                    {
                        "component": ctx.component.value,
                        "content": ctx.content,
                        "metadata": ctx.metadata,
                        "timestamp": ctx.timestamp
                    }
                    for ctx in contexts
                ]
                for component, contexts in self.costar_contexts.items()
            }
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ContextManager':
        """从字典创建实例
        
        Args:
            data: 字典数据
            
        Returns:
            ContextManager实例
        """
        manager = cls(config={
            "max_tokens": data["memory"]["max_tokens"],
            "memory_window": data["memory"]["memory_window"],
            "system_prompt": data["system_prompt"]
        })
        
        # 恢复上下文变量
        manager.context_variables = data["context_variables"]
        
        # 恢复CO-STAR上下文
        for component, contexts in data["costar_contexts"].items():
            for ctx_data in contexts:
                manager.add_costar_context(
                    component=COSTARComponent(component),
                    content=ctx_data["content"],
                    metadata=ctx_data["metadata"]
                )
        
        # 恢复消息历史
        for msg_data in data["memory"]["messages"]:
            costar_context = None
            if msg_data["costar_context"]:
                costar_context = COSTARContext(
                    component=COSTARComponent(msg_data["costar_context"]["component"]),
                    content=msg_data["costar_context"]["content"],
                    metadata=msg_data["costar_context"]["metadata"]
                )
            manager.add_message(
                role=msg_data["role"],
                content=msg_data["content"],
                metadata=msg_data["metadata"],
                costar_context=costar_context
            )
            
        return manager

    def update_context(self, context: Dict[str, Any]) -> None:
        """Update the current context.
        
        Args:
            context: New context data to update
        """
        self.context.update(context)
