# 通用的Agent基类
# 通用的方法和属性 + 必须实现的抽象方法

# 子类具体怎么去执行，我不管，我只把通用的方法和属性提取给子类写好就行了，作为父类，就是这么宽广的心胸

from __future__ import annotations

import copy
import asyncio
import json
import re
import time
from abc import ABC, abstractmethod
from contextlib import AsyncExitStack    # 感觉这个像函数一样，创建的时候有各种资源，当结束的时候，就把这些资源释放掉
from typing import Any, Awaitable, Callable

from loguru import logger

from ZBot.agent.tools.cron import CronTool
from ZBot.agent.tools.base import format_tool_error
from ZBot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from ZBot.agent.tools.registry import ToolRegistry
from ZBot.agent.tools.shell import ExecTool
from ZBot.agent.tools.web import WebFetchTool, WebSearchTool
from ZBot.config.agent_runtime import AgentRuntimeConfig
from ZBot.cron.service import CronService
from ZBot.providers.base import LLMProvider, ToolCallRequest


# ==================== 模块级常量 ====================
# 正则表达式：用于匹配大模型输出中的思考块（）<think><think>aaa</think>bbb</think>，它会匹配第一个 </think> 就停止，而不是一直匹配到最后一个。
_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)  # 方括号 [ ] 表示字符类（character class），用来匹配方括号内列出的任意一个字符。


class BaseAgent(ABC):
    """
    通用的Agent抽象基类

    这个类只放所有 Agent 都需要的运行基础设施：
    模型调用参数、默认工具、MCP 连接、工具循环、消息链辅助方法。
    长期会话、记忆归档这些职责放在 CoreAgent；一次性子任务执行放在 SubAgent。
    """

    # 工具返回结果的最大字符数限制（防止会话历史无限膨胀）
    _TOOL_RESULT_MAX_CHARS = 2000

    # 上下文压缩摘要标记；再次压缩时会替换旧摘要，避免摘要越叠越厚。
    _COMPACTION_MARKER = "【上下文压缩摘要】"
    # 连续失败且没有新观察信息时，给模型一次明确的换策略反馈。
    _NO_PROGRESS_FAILURE_LIMIT = 3
    # 压缩后保留的最近用户消息数量；旧工具链由摘要承接，避免留下不完整 tool_call。
    _RECENT_USER_MESSAGES_AFTER_COMPACTION = 1
    # 摘要里的单条片段上限，防止压缩摘要比原工具结果还膨胀。
    _COMPACTION_SNIPPET_CHARS = 240

    def __init__(
        self,
        provider: LLMProvider,
        runtime_config: AgentRuntimeConfig,
        cron_service: CronService | None = None,
    ):
        """初始化所有 Agent 共享的运行依赖。

        `provider` 负责真正调用大模型；`runtime_config` 是从全局配置派生出的
        Agent 运行快照；`cron_service` 可选，只有主 Agent 需要定时任务工具时传入。
        """

        # ==================== 大模型配置（后续可以改为子类的属性） ====================
        self.provider = provider      # 可能后续改为子类的属性，毕竟不同的agent可以调用不同的大模型
        self.runtime_config = runtime_config
        self.workspace = runtime_config.workspace    # 可能后续改成子类的属性，毕竟不同的agent有不同的工作区
        self.model = runtime_config.model            # 可能后续改为子类的属性，毕竟可以调用不同的大模型
        self.temperature = runtime_config.temperature
        self.max_tokens = runtime_config.max_tokens
        self.reasoning_effort = runtime_config.reasoning_effort
        self.agent_timeout_seconds = runtime_config.agent_timeout_seconds
        self.context_compaction_threshold = runtime_config.context_compaction_threshold
        self.context_window = provider.get_context_window(self.model)

        # ==================== 工具配置（后续可以考虑不同的agent，有不同的工具配置） ====================
        self.web_search_config = runtime_config.web_search_config
        self.web_proxy = runtime_config.web_proxy
        self.exec_config = runtime_config.exec_config
        self.cron_service = cron_service
        self.restrict_to_workspace = runtime_config.restrict_to_workspace

        # 工具注册中心：统一管理所有可用工具
        self.tools = ToolRegistry()

        # ==================== MCP 相关状态 ====================
        self._mcp_servers: dict[str, Any] = runtime_config.mcp_servers
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected: bool = False
        self._mcp_connecting: bool = False

        # ==================== 其他配置 ====================
        self.score_threshold = runtime_config.score_threshold
        self._current_messages_for_subagent: list[dict[str, Any]] | None = None
        self._active_progress_callback: Callable[..., Awaitable[None]] | None = None

        # ==================== 注册默认工具 ====================
        self._register_default_tools()


    async def connect_mcp(self) -> None:
        """连接 MCP 服务器，注册 MCP 工具。

        BaseAgent 只负责连接和注册；CoreAgent 控制什么时候调用它。
        子 Agent 不主动连接 MCP，而是通过父 Agent 已注册的 MCP 工具引用复用连接。
        """
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return

        from ZBot.agent.tools.mcp import connect_mcp_servers

        self._mcp_connecting = True
        try:
            self._mcp_stack = AsyncExitStack()
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as exc:
            logger.error("连接 MCP 服务器失败（下次收到消息时会重试）：{}", exc)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False


    # 公共 Agent Loop：CoreAgent 和 SubAgent 都复用这段模型-工具循环。
    async def run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]],
        progress_label: str = "主agent",
    ) -> tuple[str | None, list[str], list[dict[str, Any]]]:
        """
        核心方法：执行模型与工具的交互循环。

        Args:
            initial_messages: 初始消息列表（已包含 system prompt + 历史对话 + 当前用户消息）
            on_progress: 外部传入的进度回调，用于向 CLI/前端推送进度（如"正在搜索..."提示）

        Returns:
            (final_content, tools_used, messages) 三元组：
            - final_content: 最终返回给用户的文本回复
            - tools_used: 本轮对话中实际使用过的工具名称列表
            - messages: 完整的消息链（包含所有中间工具调用和结果）
        """

        messages: list[dict[str, Any]] = initial_messages
        tools_used: list[str] = []
        final_content: str | None = None
        loop_started_at: float = time.monotonic() # 获取系统单调时钟的时间戳（浮点秒数），核心特点：只递增、不会被系统时间校准 / 时区修改 / 手动改系统时间影响，记录 Agent loop 开始时间用于超时判断
        turn_index: int = 0
        consecutive_no_progress_failures: int = 0
        self._current_messages_for_subagent: list[dict[str, Any]] | None = None
        self._active_progress_callback = on_progress

        # ========== 主交互循环 ==========
        try:
            while True:
                turn_index += 1
                if self._is_agent_timeout(loop_started_at):
                    final_content = self._timeout_message()
                    self._add_assistant_message(messages, final_content)
                    break

                messages = await self._compact_messages_if_needed(messages, on_progress)
                logger.debug("Agent循环迭代: {}, 消息长度: {}", turn_index, len(messages))

                try:
                    response = await asyncio.wait_for(
                        self.provider.chat(
                            messages=messages,
                            tools=self.tools.get_definitions(),
                            model=self.model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                            reasoning_effort=self.reasoning_effort,  # 推理努力程度（仅部分模型支持）
                        ),
                        timeout=self._remaining_agent_seconds(loop_started_at),
                    )
                except asyncio.TimeoutError:
                    final_content = self._timeout_message()
                    self._add_assistant_message(messages, final_content)
                    break

                # 记录调试日志：模型响应详情
                logger.debug(
                    "模型回复: 是否包含工具调用={}, 结束原因={}, 回复内容的前100字符={}",
                    response.has_tool_calls,
                    response.finish_reason,
                    (response.content or "")[:100] if response.content else None,
                )

                if response.has_tool_calls:
                    if self._has_mixed_create_sub_agent_calls(response.tool_calls):
                        messages.append(
                            {
                                "role": "system",
                                "content": self._mixed_subagent_tool_call_message(),
                            }
                        )
                        continue

                    # 提取思考内容（去除 <think>...</think> 块，只保留可见文本）
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(
                        self._tool_hint(response.tool_calls),
                        tool_hint=True,
                        agent_label=progress_label,
                    )

                    # 将模型返回的 tool_calls 转换为标准格式
                    tool_call_dicts = [
                        {
                            "id": tool_call.id,
                            "type": "function",
                            "function": {
                                "name": tool_call.name,
                                "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                            },
                        }
                        for tool_call in response.tool_calls
                    ]

                    
                    # 将 assistant 的工具调用意图写入消息链
                    self._add_assistant_message(
                        messages,
                        response.content,
                        tool_call_dicts,
                        reasoning_content=response.reasoning_content,
                    )

                    # 逐个执行工具调用
                    for tool_call in response.tool_calls:
                        tools_used.append(tool_call.name)
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.info("调用工具：{}({})", tool_call.name, args_str[:200])

                        try:
                            if tool_call.name == "create_sub_agent":
                                # 工具执行前的上下文可以安全交给子 agent；它不包含尚未配对的 tool_call，复制三份了已经。
                                self._current_messages_for_subagent = copy.deepcopy(messages)
                            # 执行工具。用户中断会向上传播，确保上层 finally 清理资源。
                            result: str = await asyncio.wait_for(
                                self.tools.execute(tool_call.name, tool_call.arguments),
                                timeout=self._remaining_agent_seconds(loop_started_at),
                            )
                        except asyncio.TimeoutError:
                            result: str = self._tool_timeout_result(tool_call.name)

                        consecutive_no_progress_failures = self._count_no_progress_failures(
                            consecutive_no_progress_failures,
                            result,
                        )

                        if consecutive_no_progress_failures >= self._NO_PROGRESS_FAILURE_LIMIT:
                            result = (
                                result
                                + "\n\n[进展判断：连续多次工具结果都是失败，且没有提供新的可用观察信息。"
                                "请停止重复当前路径，改用不同工具/参数；如果没有新路径，请总结已知信息并给出最终回复。]"
                            )
                            consecutive_no_progress_failures = 0

                        # 将工具执行结果追加到消息链
                        self._add_tool_result(messages, tool_call.id, tool_call.name, result)

                    # 继续下一轮迭代
                    continue

                # ========== 处理最终回复 ==========
                # _strip_think 会移除模型可能包含的<think>...</think> 思考块，只保留对外输出的文本
                clean = self._strip_think(response.content)

                if response.finish_reason == "error":
                    logger.error("大模型返回错误：{}", (clean or "")[:200])
                    final_content = clean or "抱歉，调用大模型时发生了错误。 "
                    break

                # 将最终回复写入消息链
                self._add_assistant_message(
                    messages,
                    clean,
                    reasoning_content=response.reasoning_content,
                )
                final_content = clean
                break
        finally:
            self._current_messages_for_subagent = None
            self._active_progress_callback = None

        return final_content, tools_used, messages

    def _register_default_tools(self) -> None:
        """注册默认工具：文件、Exec、Web，若提供 CronService 则注册 CronTool。

        这些是 CoreAgent 和 SubAgent 都可以使用的普通工具。CoreAgent 会额外注册
        create_sub_agent；SubAgent 不会注册它，避免递归创建子 Agent。
        """
        # 如果限制了工作区，文件工具只能访问工作区内的文件
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        # 注册文件操作工具
        for tool_cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(tool_cls(workspace=self.workspace, allowed_dir=allowed_dir))

        # 注册 Shell 执行工具
        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
            )
        )

        # 注册网页搜索工具
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))

        # 注册网页抓取工具
        self.tools.register(WebFetchTool(proxy=self.web_proxy))

        # 如果提供了定时任务服务，注册定时任务工具
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

        logger.info("默认工具注册完成")

    @staticmethod
    def _has_mixed_create_sub_agent_calls(tool_calls: list[ToolCallRequest]) -> bool:
        """判断 create_sub_agent 是否和普通工具混在同一轮调用。"""
        names = [tool_call.name for tool_call in tool_calls]
        return "create_sub_agent" in names and len(names) > 1

    @staticmethod
    def _mixed_subagent_tool_call_message() -> str:
        """生成 create_sub_agent 并列调用时的停止提示。"""
        return (
            "本轮模型同时请求了 create_sub_agent 和其他工具。"
            "为避免子 Agent 拿到未配对完成的工具调用链，create_sub_agent 必须单独一轮调用。"
            "请先完成必要的普通工具调用，再单独创建子 Agent。"
        )

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """移除模型输出中的思考块并返回清理后的文本。<think>和<think>包裹。

        这些内容对用户通常没有价值，反而会增加上下文长度和 token 费用。
        例如输入：
        "我来帮你写这个程序。首先需要..."
        返回：
        "我来帮你写这个程序。"
        """
        if not text:
            return None
        cleaned = _THINK_BLOCK_RE.sub("", text).strip()
        return cleaned or None

    @staticmethod
    def _tool_hint(tool_calls: list[ToolCallRequest]) -> str:
        """把工具调用列表压缩成一行简短提示，便于在 CLI 中展示进度。"""
        hints: list[str] = []

        for tool_call in tool_calls:
            args = tool_call.arguments
            preview = next((value for value in args.values() if isinstance(value, str) and value), None)
            if preview is None:
                hints.append(tool_call.name)
            elif len(preview) > 40:
                hints.append(f'{tool_call.name}("{preview[:40]}...")')
            else:
                hints.append(f'{tool_call.name}("{preview}")')

        return ",".join(hints)

    @staticmethod
    def _add_assistant_message(
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
    ) -> None:
        """向消息链追加 assistant 消息。

        统一封装这个格式，避免 CoreAgent/SubAgent 各自手写 tool_calls、
        reasoning_content 等字段时格式不一致。
        """
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        messages.append(message)

    def _is_agent_timeout(self, started_at: float) -> bool:
        """判断本轮 Agent loop 是否超过总运行时间。"""
        return time.monotonic() - started_at >= self.agent_timeout_seconds

    def _remaining_agent_seconds(self, started_at: float) -> float:
        """返回本轮任务还剩多少秒，用于包住模型调用和工具调用。"""
        return max(0.001, self.agent_timeout_seconds - (time.monotonic() - started_at))

    def _timeout_message(self) -> str:
        """生成主 Agent 总超时后的最终回复。"""
        return (
            f"本轮任务已运行超过 {self.agent_timeout_seconds} 秒，为避免继续占用资源，我先停止执行。"
            "你可以根据当前结果继续追问，或者把任务拆成更小的步骤。"
        )

    def _tool_timeout_result(self, tool_name: str) -> str:
        """在总超时已发生但需要补齐 tool result 时使用，保证消息链合法。"""
        return format_tool_error(
            "Agent 总运行时间已超过限制，停止继续执行工具",
            attempted=f"调用工具 {tool_name}",
            observed=f"当前任务运行时间已达到 {self.agent_timeout_seconds} 秒",
            do_not_repeat="不要继续调用新的工具",
            next_action="总结当前已知信息并给出最终回复",
        )

    async def _compact_messages_if_needed(
        self,
        messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]],
    ) -> list[dict[str, Any]]:
        """上下文接近模型窗口时压缩旧消息，保留任务状态继续执行。"""
        estimated_tokens = self._estimate_messages_tokens(messages)
        threshold_tokens = int(self.context_window * self.context_compaction_threshold)
        if estimated_tokens < threshold_tokens:
            return messages

        logger.info(
            "上下文接近模型窗口，开始压缩：estimated_tokens={}, threshold={}, context_window={}",
            estimated_tokens,
            threshold_tokens,
            self.context_window,
        )
        await on_progress("上下文接近模型窗口，正在压缩历史工具链和中间过程。")

        compacted = self._compact_messages(messages)
        if self._estimate_messages_tokens(compacted) >= estimated_tokens:
            compacted = self._minimal_compacted_messages(messages)
        logger.info(
            "上下文压缩完成：{} 条消息 -> {} 条消息，估算 token {} -> {}",
            len(messages),
            len(compacted),
            estimated_tokens,
            self._estimate_messages_tokens(compacted),
        )
        return compacted

    def _compact_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """把旧工具链压成一条 system 摘要，避免简单截断导致 Agent 失忆。"""
        system_messages = [
            message
            for message in messages
            if message.get("role") == "system"
            and not str(message.get("content", "")).startswith(self._COMPACTION_MARKER)
        ]
        recent_user_messages = [
            message for message in messages if message.get("role") == "user"
        ][-self._RECENT_USER_MESSAGES_AFTER_COMPACTION :]

        summary = self._build_compaction_summary(messages)
        return [*copy.deepcopy(system_messages), {"role": "system", "content": summary}, *copy.deepcopy(recent_user_messages)]

    def _minimal_compacted_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """常规摘要没有变小时，退回最小摘要，保证压缩一定减少上下文。"""
        system_messages = [
            message
            for message in messages
            if message.get("role") == "system"
            and not str(message.get("content", "")).startswith(self._COMPACTION_MARKER)
        ]
        latest_user = self._latest_content(messages, "user") or "未识别到明确用户任务"
        summary = (
            f"{self._COMPACTION_MARKER}\n"
            f"本轮任务目标：\n{latest_user[: self._COMPACTION_SNIPPET_CHARS]}\n\n"
            "已完成：见此前工具链，已因上下文预算压缩。\n"
            "关键事实：保留最近用户目标，后续如信息不足请重新获取新的有效观察。\n"
            "重要文件/路径：暂无可安全保留的完整列表。\n"
            "工具调用结论：旧工具链已压缩。\n"
            "失败尝试：不要重复刚才无效路径。\n"
            "不要重复：不要重复已失败的同参数工具调用。\n"
            "剩余待办：继续完成用户任务；如缺信息，先获取新的有效观察。"
        )
        return [*copy.deepcopy(system_messages), {"role": "system", "content": summary}]

    def _build_compaction_summary(self, messages: list[dict[str, Any]]) -> str:
        """生成给模型继续工作的压缩摘要，只保留任务推进所需的信息。"""
        task_goal = (self._latest_content(messages, "user") or "未识别到明确用户任务")[: self._COMPACTION_SNIPPET_CHARS]
        assistant_notes = self._collect_role_snippets(messages, "assistant", limit=4)
        tool_successes, tool_failures = self._collect_tool_snippets(messages)

        return (
            f"{self._COMPACTION_MARKER}\n"
            f"本轮任务目标：\n{task_goal}\n\n"
            "已完成：\n"
            f"{self._format_snippets(assistant_notes) or '暂无明确完成项'}\n\n"
            "关键事实：\n"
            f"{self._format_snippets(tool_successes) or '暂无可保留事实'}\n\n"
            "重要文件/路径：\n"
            f"{self._extract_paths(messages) or '暂无'}\n\n"
            "工具调用结论：\n"
            f"{self._format_snippets(tool_successes) or '暂无'}\n\n"
            "失败尝试：\n"
            f"{self._format_snippets(tool_failures) or '暂无'}\n\n"
            "不要重复：\n"
            f"{self._collect_do_not_repeat(messages) or '暂无'}\n\n"
            "剩余待办：\n根据当前摘要继续完成用户任务；如信息不足，优先获取新的有效观察，不要重复失败路径。"
        )

    @staticmethod
    def _estimate_messages_tokens(messages: list[dict[str, Any]]) -> int:
        """粗略估算 token：第一版不引入 tokenizer，用字符数做保守近似。"""
        total_chars = 0
        for message in messages:
            total_chars += len(str(message.get("role", "")))
            total_chars += len(str(message.get("content", "")))
            if "tool_calls" in message:
                total_chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))
        return max(1, total_chars // 2)

    @staticmethod
    def _latest_content(messages: list[dict[str, Any]], role: str) -> str | None:
        """?????????????????"""
        for message in reversed(messages):
            if message.get("role") == role and isinstance(message.get("content"), str):
                content = message["content"].strip()
                if content:
                    return content[:1200]
        return None

    @staticmethod
    def _collect_role_snippets(messages: list[dict[str, Any]], role: str, *, limit: int) -> list[str]:
        """????????????????"""
        snippets: list[str] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") != role or not isinstance(content, str) or not content.strip():
                continue
            snippets.append(content.strip()[: BaseAgent._COMPACTION_SNIPPET_CHARS])
        return snippets[-limit:]

    @staticmethod
    def _collect_tool_snippets(messages: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
        """?????????????????"""
        successes: list[str] = []
        failures: list[str] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                continue
            snippet = f"{message.get('name', 'tool')}: {content.strip()[: BaseAgent._COMPACTION_SNIPPET_CHARS]}"
            if content.startswith("错误："):
                failures.append(snippet)
            elif content.strip():
                successes.append(snippet)
        return successes[-5:], failures[-5:]

    @staticmethod
    def _format_snippets(snippets: list[str]) -> str:
        """????????? Markdown ???"""
        return "\n".join(f"- {snippet}" for snippet in snippets)

    @staticmethod
    def _extract_paths(messages: list[dict[str, Any]]) -> str:
        """??????????????????"""
        text = "\n".join(str(message.get("content", "")) for message in messages)
        paths = re.findall(r"(?:[A-Za-z]:\\[^\s\"'<>|]+|[\w./-]+/[\w./-]+)", text)
        unique_paths = list(dict.fromkeys(paths))
        return "\n".join(f"- {path}" for path in unique_paths[-6:])

    @staticmethod
    def _collect_do_not_repeat(messages: list[dict[str, Any]]) -> str:
        """???????????????"""
        items: list[str] = []
        for message in messages:
            content = message.get("content")
            if message.get("role") != "tool" or not isinstance(content, str):
                continue
            for line in content.splitlines():
                if line.startswith("不要重复："):
                    items.append(line.removeprefix("不要重复：").strip())
        return "\n".join(f"- {item}" for item in list(dict.fromkeys(items))[-5:])

    def _count_no_progress_failures(self, current_count: int, result: str) -> int:
        """连续失败且没有新观察信息时计数，用来提醒模型换策略。"""
        if not result.startswith("错误："):
            return 0
        has_observation = "观察结果：" in result
        return 0 if has_observation else current_count + 1

    @staticmethod
    def _add_tool_result(
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> None:
        """向消息链追加工具返回结果。

        OpenAI 兼容协议要求 tool 消息包含 tool_call_id；这里统一写入，
        让后续 provider 清洗时能保持正确的工具调用链。
        """
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
