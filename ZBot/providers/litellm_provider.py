"""
基于LiteLLM的LLM提供商实现
核心作用：通过LiteLLM统一调用所有大模型API，100%兼容注册表配置
无需为每个厂商写独立代码，所有逻辑由注册表驱动

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
【为什么大模型会返回工具参数而不是普通文本？】
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

这是 OpenAI 兼容 API 的 Function Calling（工具调用）机制决定的：

1. API 层面的约定
   - 当请求中包含 `tools` 参数时，模型知道它可以调用工具
   - `tool_choice: "auto"` 让模型自己决定何时调用工具
   - 这是 API 协议层面的约定，不是提示词决定的

2. 模型训练层面
   - 模型在训练时学习了"当有 tools 参数时，用特定格式返回工具调用"
   - 返回格式是：{"tool_calls": [{"function": {"name": "xxx", "arguments": "..."}}]}
   - 这个格式是 OpenAI 定义的，所有兼容厂商都遵循

3. 工具 Schema 的作用
   - tools 参数包含每个工具的 JSON Schema（名称、描述、参数类型）
   - 模型根据 Schema 知道"有哪些工具可用"、"参数是什么类型"
   - 模型根据 Schema 生成符合格式的参数 JSON

4. 提示词的作用
   - 提示词只影响"何时调用工具"的策略决策
   - 提示词不定义"如何返回工具调用"的格式
   - 格式是 API 协议 + 模型训练决定的

【完整流程】
用户消息 → 构建消息 + 工具 Schema → 发给 API
                                        ↓
                    API 返回 tool_calls ← 模型决定调用工具
                                        ↓
                    执行工具 → 结果写回消息 → 再次调用模型
                                        ↓
                    最终返回普通文本 ← 模型完成任务

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
import json
import secrets                                  # 安全随机，random伪随机
import string
from typing import Any,Optional
import litellm
from litellm import acompletion                 # async，调用大模型，返回回答
from loguru import logger 

from ZBot.providers.base import DEFAULT_CONTEXT_WINDOW, LLMProvider, LLMResponse, ToolCallRequest
from ZBot.providers.registry import find_by_model, find_gateway
from ZBot.providers.registry import ProviderSpec

_ALLOWED_MSG_KEYS = frozenset({"role","content","tool_calls","tool_call_id", "name", "reasoning_content"}) # 允许的消息标准字段（所有厂商通用）
_ALNUM = string.ascii_letters + string.digits  # 字母数字字符集（生成工具ID）


def _short_tool_id() ->str:
    """
    生成9位字母数字工具ID,兼容所有厂商
    """
    return "".join(secrets.choice(_ALNUM) for _ in range(9))


class LiteLLMProvider(LLMProvider):
    """LiteLLM统一调用实现类,为什么_instance这样写，这就是cls._instance就是一个实例"""

    _instance: Optional["LiteLLMProvider"] = None
    default_model: str
    _std_provider: Optional["ProviderSpec"] = None
    _gateway: Optional["ProviderSpec"] = None

    def __new__(
            cls,
            api_key: str | None = None,
            api_base: str | None = None,
            default_model: str = "GLM-5.0-Pro",  # 默认模型名称
            provider_name: str | None = None,        
    ):
        """创建或复用 LiteLLM 提供商单例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls) # 创建了实例，并赋值给了cls的_instance属性，如果再次cls._instance.某某 = 某某，就是在给实例写属性
            # 初始化父类（API密钥/地址）
            cls._instance.api_key = api_key
            cls._instance.api_base = api_base
            cls._instance.default_model = default_model
            # 检测网关,返回ProviderSpec类
            cls._instance._gateway = find_gateway(provider_name)
            # 监测标准提供商，返回ProviderSpec类
            cls._instance._std_provider = find_by_model(default_model)
            litellm.api_key = api_key
            litellm.api_base = api_base    
            # LiteLLM基础配置
            litellm.suppress_debug_info = True  # 关闭调试日志
            litellm.drop_params = True          # 自动删除不支持的参数
        return cls._instance   # 返回实例对象

    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        default_model: str = "GLM-5.0-Pro",
        provider_name: str | None = None,
    ):
        """匹配 __new__ 的参数签名，避免实例化时落到父类 __init__ 报多余参数。"""
        # 实际初始化在 __new__ 中完成；这里保留空实现是为了兼容现有单例写法。
        pass


    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        model: str | None = None,
        max_tokens: int = 4396,
        temperature: float = 0.1,
        reasoning_effort: str | None = None,
    ) -> LLMResponse:
        """【核心方法】异步发送聊天请求给 LiteLLM，并返回标准化响应。"""
        if model is None:
            model = self.default_model
            
        model = self._resolve_model(model)

        # 确保 max_tokens 至少为1，避免传入0导致SDK报错
        max_tokens = max(1, max_tokens)
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": self._sanitize_messages(self._sanitize_empty_content(messages)),
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        kwargs["api_key"] = self.api_key
        kwargs["api_base"] = self.api_base
        
        # 推理强度参数
        if reasoning_effort:
            kwargs["reasoning_effort"] = reasoning_effort
            kwargs["drop_params"] = True
        

        if tools:
            kwargs["tools"] = tools   # 工具列表（JSON Schema 格式）
            kwargs["tool_choice"] = "auto"

        try:
            logger.debug("发送给模型的消息：{}", json.dumps(kwargs.get("messages", []), ensure_ascii=False, indent=2))
            response = await acompletion(**kwargs)
            return self._parse_response(response)
        except Exception as e:
            # 捕获调用中的任意异常并以 LLMResponse 错误形式返回，避免上层抛出
            return LLMResponse(
                content=f"调用大模型失败：{str(e)}",
                finish_reason="error",
            )

    def get_context_window(self, model: str | None = None) -> int:
        """优先读取 LiteLLM 的模型元数据，失败时回退到 128K。"""
        target_model = model or self.default_model
        try:
            resolved_model = self._resolve_model(target_model)
            info = litellm.get_model_info(resolved_model)
        except Exception:
            logger.debug("无法读取模型 {} 的上下文窗口，使用默认值 {}", target_model, DEFAULT_CONTEXT_WINDOW)
            return DEFAULT_CONTEXT_WINDOW

        for key in ("max_input_tokens", "max_context_tokens", "max_tokens"):
            value = info.get(key)
            if isinstance(value, int) and value > 0:
                return value
        return DEFAULT_CONTEXT_WINDOW



    def _resolve_model(self, model: str) -> str:
        """
        模型名称标准化：根据注册表自动添加前缀，便于litellm使用。
        """
        if self._gateway is not None:
            # 网关模式：保证路由到网关内部的模型池
            model = f"{self._gateway.litellm_prefix}/{model}"
            return model

        # 标准厂商：自动添加前缀
        elif self._std_provider is not None:
            model = f"{self._std_provider.litellm_prefix}/{model}"
            return model

        raise ValueError(f"无法解析模型名称 '{model}'，请检查配置是否正确，或模型名称是否包含已注册的关键词。")
    

    @staticmethod
    def _sanitize_messages(messages: list[dict[str, Any]], extra_keys: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
        """
        消息最终清洗：
        1. 保留标准+厂商专属字段
        2. 移除 tool 消息中的 name 字段（OpenAI 兼容 API 不接受）
        """
        allowed = _ALLOWED_MSG_KEYS | extra_keys
        sanitized = LLMProvider._sanitize_request_messages(messages, allowed)

        for clean in sanitized:
            # OpenAI 兼容 API 的 tool 消息不应包含 name 字段
            if clean.get("role") == "tool":
                clean.pop("name", None)
        return sanitized



    def _parse_response(self, response: Any) -> LLMResponse:
        """
        解析LiteLLM响应 → 标准化LLMResponse,本质就是一个提取返回对象的最底层的属性值的过程。
        """
        choice = response.choices[0]
        message = choice.message
        content = message.content
        finish_reason = choice.finish_reason
        tool_calls = []

        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                args = tc.function.arguments
                # 有些厂商会返回不合规的字符串 JSON，需要恢复为 dict
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                tool_calls.append(ToolCallRequest(
                    id=_short_tool_id(),
                    name=tc.function.name,
                    arguments=args if isinstance(args, dict) else {},
                ))

        # Token统计
        usage = {}
        if hasattr(response, "usage") and response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        # 思考内容
        reasoning_content = getattr(message, "reasoning_content", None)

        # 返回标准化响应
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason or "stop",
            usage=usage,
            reasoning_content=reasoning_content,
        )
