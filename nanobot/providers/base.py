"""大模型提供商的基础抽象接口。

这个模块定义了三件最核心的事情：
1. 工具调用请求的数据结构。
2. 大模型响应的统一格式。
3. 所有提供商必须实现的抽象方法。

这样无论底层接 OpenAI、Anthropic 还是 LiteLLM 网关，上层代码都只面向统一接口编程。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCallRequest:
    """大模型返回的工具调用请求。"""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """大模型返回的标准化响应。"""

    content: str | None
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)
    reasoning_content: str | None = None
    thinking_blocks: list[dict] | None = None

    @property
    def has_tool_calls(self) -> bool:
        """判断响应中是否包含工具调用。"""
        return len(self.tool_calls) > 0


class LLMProvider(ABC):
    """LLM 提供商抽象基类。"""

    def __init__(self, api_key: str | None = None, api_base: str | None = None):
        self.api_key = api_key
        self.api_base = api_base

    @staticmethod
    def _sanitize_empty_content(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """清洗空内容消息，避免部分厂商因空字符串直接报错。

        处理策略如下：
        1. 空字符串内容：替换成占位文本或空字符串。
        2. 多模态列表：过滤掉空文本块。
        3. 字典格式内容：标准化为列表，方便后续统一处理。
        """
        result: list[dict[str, Any]] = []
        for msg in messages:
            content = msg.get("content")

            if isinstance(content, str) and not content:
                clean = dict(msg)
                if clean.get("role") == "assistant" and msg.get("tool_calls"):
                    clean["content"] = ""  # 部分厂商不接受 null，改用空字符串
                else:
                    clean["content"] = "(空内容)"
                result.append(clean)
                continue

            if isinstance(content, list):
                filtered = [
                    item
                    for item in content
                    if not (
                        isinstance(item, dict)
                        and item.get("type") in ("text", "input_text", "output_text")
                        and not item.get("text")
                    )
                ]
                if len(filtered) != len(content):
                    clean = dict(msg)
                    if filtered:
                        clean["content"] = filtered
                    elif msg.get("role") == "assistant" and msg.get("tool_calls"):
                        clean["content"] = ""  # 部分厂商不接受 null，改用空字符串
                    else:
                        clean["content"] = "(空内容)"
                    result.append(clean)
                    continue

            if isinstance(content, dict):
                clean = dict(msg)
                clean["content"] = list(content)
                result.append(clean)
                continue

            # 处理 content 为 None 的情况
            if content is None and msg.get("role") == "assistant" and msg.get("tool_calls"):
                clean = dict(msg)
                clean["content"] = ""  # 部分厂商不接受 null，改用空字符串
                result.append(clean)
                continue

            result.append(msg)

        return result

    @staticmethod
    def _sanitize_request_messages(
        messages: list[dict[str, Any]],
        allowed_keys: frozenset[str],
    ) -> list[dict[str, Any]]:
        """清洗请求消息字段，只保留厂商支持的字段。"""
        sanitized = []
        for msg in messages:
            clean = {key: value for key, value in msg.items() if key in allowed_keys}
            # 确保 assistant 消息有有效的 content（部分 API 不接受 None）
            if clean.get("role") == "assistant" and "content" not in clean:
                clean["content"] = ""
            sanitized.append(clean)
        return sanitized

    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """发送聊天请求并返回标准化响应。"""
        raise NotImplementedError

    @abstractmethod
    def get_default_model(self) -> str:
        """返回当前提供商默认模型名。"""
        raise NotImplementedError
