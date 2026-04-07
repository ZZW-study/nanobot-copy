"""Agent 主循环与单轮消息处理。

`AgentLoop` 是运行时入口，负责把下面几件事串起来：
1. 从直接调用拿到用户输入。
2. 找到对应 session，并构造要发给模型的上下文。
3. 处理模型的普通回复或工具调用。
4. 把本轮消息安全地写回 session。
5. 在合适的时机触发长期记忆归档。

因此它既是调度层，也是"消息 -> 模型 -> 工具 -> 会话落盘"这条链路的汇总点。
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from loguru import logger

from nanobot.agent.context import ContextBuilder
from nanobot.agent.tools.cron import CronTool
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry
from nanobot.agent.tools.shell import ExecTool
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool
from nanobot.providers.base import LLMProvider
from nanobot.session.manager import Session, SessionManager

if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig, WebSearchConfig
    from nanobot.cron.service import CronService
    from nanobot.providers.base import ToolCallRequest


_THINK_BLOCK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


class AgentLoop:
    """运行中的 Agent 实例。

    这个类故意把核心流程收敛成少量 helper：
    - `process_direct` 负责直接处理用户输入。
    - `_run_turn` 负责单轮正常对话。
    - `_run_agent_loop` 负责模型与工具之间的循环。

    这样每个 helper 都只回答一个问题，便于维护和排错。
    """

    # 工具结果如果原样全部写进 session，历史会迅速膨胀，因此对 tool 消息做截断。
    # 注意：这个截断只影响写入 session 的历史记录，不影响发送给模型的内容
    _TOOL_RESULT_MAX_CHARS = 2000

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        model: str | None = None,
        max_iterations: int = 40,
        temperature: float = 0.1,
        max_tokens: int = 4096,
        memory_window: int = 100,
        reasoning_effort: str | None = None,
        web_search_config: WebSearchConfig | None = None,
        web_proxy: str | None = None,
        exec_config: ExecToolConfig | None = None,
        cron_service: CronService | None = None,
        restrict_to_workspace: bool = False,
        session_manager: SessionManager | None = None,
        mcp_servers: dict[str, Any] | None = None,
    ):
        """初始化运行时依赖与内部状态。"""
        from nanobot.config.schema import ExecToolConfig, WebSearchConfig

        self.provider = provider                            # LLM 提供者抽象（负责与模型交互）
        self.workspace = workspace                          # 工作目录，文件工具等会基于此限制或执行文件操作
        self.model = model or provider.get_default_model()
        self.max_iterations = max_iterations                # 单轮最大工具调用/迭代次数，防止无限循环
        self.temperature = temperature                      # 模型的采样温度，决定回答的随机性
        self.max_tokens = max_tokens                        # 模型返回最大 token 限制
        self.memory_window = memory_window                  # 回话历史窗口大小，用于构建上下文

        self.reasoning_effort = reasoning_effort
        self.web_search_config = web_search_config or WebSearchConfig()
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        self.context = ContextBuilder(workspace)            # ContextBuilder 负责把会话历史/当前消息转换为模型输入的 messages

        self.sessions = session_manager or SessionManager(workspace)    # Session 管理器负责会话的创建/保存/加载

        self.tools = ToolRegistry()                         # 工具注册中心，包含文件/网络/执行等工具实例


        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False

        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task[Any]] = set()
        self._consolidation_locks: dict[str, asyncio.Lock] = {}
        self._processing_lock = asyncio.Lock()
        # 全局消息处理锁，保证单条消息的处理是串行的（避免会话竞争）

        self._register_default_tools()
        # `_consolidation_*` 系列字段负责长期记忆归档任务的并发控制与回收。

    # ---- 工具注册与初始化 ----

    def _register_default_tools(self) -> None:
        """注册默认工具集。

        文件工具、shell、web 工具默认都会加载；
        cron 工具只有在外部确实提供了 `CronService` 时才会注册。
        """
        # 如果配置了 restrict_to_workspace，则把工作目录作为文件工具的白名单目录
        # 否则 allowed_dir 为 None，工具不做目录限制（由工具自身进一步校验）
        allowed_dir = self.workspace if self.restrict_to_workspace else None

        for tool_cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(tool_cls(workspace=self.workspace, allowed_dir=allowed_dir))

        self.tools.register(
            ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            )
        )
        # ExecTool: 负责在工作区运行 shell 命令，受 timeout 与路径限制保护，避免越权执行。
        self.tools.register(WebSearchTool(config=self.web_search_config, proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))

        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    async def _connect_mcp(self) -> None:
        """懒连接 MCP 服务器。

        MCP 是扩展工具来源，不是每次启动都一定需要。
        因此这里只在首次真正用到 Agent 时才连接，减少启动成本。
        """
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return

        from nanobot.agent.tools.mcp import connect_mcp_servers

        self._mcp_connecting = True
        try:
            # 使用 AsyncExitStack 管理 MCP 连接生命周期，确保在关闭时能统一释放所有资源
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            # connect_mcp_servers 会把来自 MCP 的工具注册进 ToolRegistry
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

    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """移除模型输出中的 `<think>...</think>` 思维块。"""
        if not text:
            return None
        cleaned = _THINK_BLOCK_RE.sub("", text).strip()
        return cleaned or None

    @staticmethod
    def _tool_hint(tool_calls: list[ToolCallRequest]) -> str:
        """把工具调用列表压缩成适合进度展示的短提示。"""
        hints: list[str] = []
        for tool_call in tool_calls:
            args = tool_call.arguments
            if isinstance(args, list) and args:
                args = args[0]

            preview: str | None = None
            if isinstance(args, dict):
                preview = next((value for value in args.values() if isinstance(value, str) and value), None)

            if preview is None:
                hints.append(tool_call.name)
            elif len(preview) > 40:
                hints.append(f'{tool_call.name}("{preview[:40]}...")')
            else:
                hints.append(f'{tool_call.name}("{preview}")')
        return ", ".join(hints)

    async def _run_agent_loop(
        self,
        initial_messages: list[dict[str, Any]],
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> tuple[str | None, list[str], list[dict[str, Any]]]:
        """驱动"模型回复 -> 工具执行 -> 再喂回模型"的循环。

        返回三样东西：
        1. 最终用户可见回复
        2. 本轮实际调用过的工具列表
        3. 完整消息链，用于后续写回 session
        """

        messages = list(initial_messages)       # 复制初始消息链到局部变量，后续会在循环中追加 assistant/tool 的中间消息
        tools_used: list[str] = []              # 本轮实际上被调用过的工具名称（按顺序，会在写回历史时去重）
        final_content: str | None = None        # 最终返回给用户的文本（清理过 <think> 标签的回复），初始为 None

        for _ in range(self.max_iterations):
            # 每一轮都把"当前消息链 + 工具 schema"发给模型，让模型自行决定
            # 是直接回答，还是继续调用工具。
            # 把当前消息链交给 LLMProvider，请求模型决定是回答还是调用工具
            logger.debug("Agent loop iteration {}, messages count: {}", _ + 1, len(messages))
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(),
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )
            logger.debug("Model response: has_tool_calls={}, finish_reason={}, content_preview={}",
                        response.has_tool_calls, response.finish_reason,
                        (response.content or "")[:100] if response.content else None)

            if response.has_tool_calls:
                # 如果模型一边思考一边决定调用工具，这里把精简后的状态向外发送，
                # 让 CLI 或前端能够展示"正在做什么"。
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                # 将模型返回的 tool_calls 转为写入消息链的"函数调用"结构，
                # 这样在执行工具前，消息链中就包含了 assistant 的调用意图，
                # 便于下一轮模型看到自己的调用历史。
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
                # 先把 assistant 的调用意图写入消息链，再执行工具。
                # 这样下一轮模型能看到"自己刚才调用了什么"。
                self.context.add_assistant_message(
                    messages,
                    response.content,
                    tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                for tool_call in response.tool_calls:
                    # 逐个执行工具，并把结果回填给模型。工具可能涉及网络/IO/子进程，故需 await。
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("调用工具：{}({})", tool_call.name, args_str[:200])
                    # 执行工具，得到任意可序列化的结果（字符串或结构化对象）
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # 将工具执行结果作为一条 role=tool 的消息追加到 messages，供模型下一轮消费
                    self.context.add_tool_result(messages, tool_call.id, tool_call.name, result)
                continue

            # 没有工具调用时，本轮对话结束，clean 后的文本就是最终回复。
            # `_strip_think` 会移除模型可能包含的 <think>...</think> 思考块，只保留最终输出。
            clean = self._strip_think(response.content)
            if response.finish_reason == "error":
                logger.error("大模型返回错误：{}", (clean or "")[:200])
                final_content = clean or "抱歉，调用大模型时发生了错误。 "
                break

            self.context.add_assistant_message(
                messages,
                clean,
                reasoning_content=response.reasoning_content,
                thinking_blocks=response.thinking_blocks,
            )
            final_content = clean
            break

        if final_content is None:
            # 如果循环结束仍未产生最终内容，说明已达到 max_iterations 限制。
            # 这是为了防止模型与工具进入无穷回路。向用户说明原因并建议拆分任务。
            logger.warning("已达到最大工具迭代次数：{}", self.max_iterations)
            final_content = (
                f"我已经达到最大工具调用轮数（{self.max_iterations} 次），仍未完成任务。"
                "你可以把任务拆成更小的步骤后再试。"
            )

        return final_content, tools_used, messages

    async def close_mcp(self) -> None:
        """关闭 MCP 连接栈，通常在进程退出前调用。"""
        if not self._mcp_stack:
            return
        try:
            await self._mcp_stack.aclose()
        except BaseException as exc:
            if not (isinstance(exc, RuntimeError) or exc.__class__.__name__ == "BaseExceptionGroup"):
                raise
        finally:
            self._mcp_stack = None
            self._mcp_connected = False

    async def _process_message(
        self,
        content: str,
        session_key: str,
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """处理单条消息，并返回最终回复。

        这里会处理内建命令，例如 `/new`、`/help`。
        """
        preview = content[:80] + "..." if len(content) > 80 else content
        logger.info("正在处理消息：{}", preview)

        session = self.sessions.get_or_create(session_key)
        command = content.strip().lower()

        if command == "/new":
            # /new 的语义不是简单清空，而是"先归档，再开始新会话"。
            if not await self._archive_and_reset_session(session):
                return "长期记忆归档失败，会话未清空，请稍后重试。"
            return "已开始新的会话。"

        if command == "/help":
            return "nanobot 可用命令：\n/new - 开始新会话\n/help - 查看帮助"

        # 只有会话累计到一定长度时，才在后台触发长期记忆归档。
        self._schedule_consolidation(session)

        final_content = await self._run_turn(
            session,
            content=content,
            on_progress=on_progress,
        )
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("回复：{}", preview)

        return final_content

    async def _run_turn(
        self,
        session: Session,
        *,
        content: str,
        media: list[str] | None = None,
        on_progress: Callable[..., Awaitable[None]] | None = None,
    ) -> str:
        """执行一轮标准对话。

        这层是主路径：
        1. 从 session 取历史
        2. 构造模型请求
        3. 执行模型/工具循环
        4. 把本轮结果写回 session
        """
        history = session.get_history(max_messages=self.memory_window)
        initial_messages = self.context.build_messages(
            history=history,
            current_message=content,
            media=media,
        )

        final_content, tools_used, all_messages = await self._run_agent_loop(
            initial_messages,
            on_progress=on_progress,
        )
        final_content = final_content or "我已经完成处理，但没有需要额外返回的内容。"

        self._save_turn(session, all_messages, 1 + len(history), tools_used)
        self.sessions.save(session)
        return final_content

    async def _archive_and_reset_session(self, session: Session) -> bool:
        """归档当前会话剩余消息，并把会话清空重置。"""
        lock = self._get_consolidation_lock(session.key)
        self._consolidating.add(session.key)
        try:
            async with lock:
                snapshot = session.messages[session.last_consolidated :]
                if snapshot:
                    temp = Session(key=session.key, messages=list(snapshot))
                    if not await self._consolidate_memory(temp, archive_all=True):
                        return False
        except Exception:
            logger.exception("会话 {} 在执行 /new 归档时失败", session.key)
            return False
        finally:
            self._consolidating.discard(session.key)

        session.clear()
        self.sessions.save(session)
        self.sessions.invalidate(session.key)
        return True

    def _schedule_consolidation(self, session: Session) -> None:
        """在满足阈值时，为会话安排后台记忆归档任务。"""
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated < self.memory_window or session.key in self._consolidating:
            return

        self._consolidating.add(session.key)
        task = asyncio.create_task(self._run_consolidation(session))
        self._consolidation_tasks.add(task)
        task.add_done_callback(self._consolidation_tasks.discard)

    async def _run_consolidation(self, session: Session) -> None:
        """真正执行后台归档任务，并保证状态标记能被回收。"""
        try:
            async with self._get_consolidation_lock(session.key):
                await self._consolidate_memory(session)
        finally:
            self._consolidating.discard(session.key)

    def _get_consolidation_lock(self, session_key: str) -> asyncio.Lock:
        """获取某个 session 专属的归档锁。"""
        lock = self._consolidation_locks.get(session_key)
        if lock is None:
            lock = asyncio.Lock()
            self._consolidation_locks[session_key] = lock
        return lock

    def _save_turn(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        skip: int,
        tools_used: list[str] | None = None,
    ) -> None:
        """把本轮新增消息写回 session。

        `messages` 里包含 system prompt、历史消息和当前轮消息，
        因此这里通过 `skip` 跳过前半段，只保留本轮真正新增的消息。
        """
        from datetime import datetime

        turn_messages = [dict(message) for message in messages[skip:]]
        # `skip` 通常等于 1 + len(history)，用于跳过 system + 已有历史，
        # 仅把本轮新增的 assistant/tool/user 消息写进 session。
        self._annotate_tools_used(turn_messages, tools_used or [])

        for entry in turn_messages:
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue

            # tool 结果通常最容易失控增长，落盘前在这里做统一截断。
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n……（内容已截断）"
            elif role == "user":
                # user 消息里会混入当前轮的运行时元信息，写回历史前必须去掉。
                stripped = self._strip_runtime_context(content)
                if stripped is None:
                    continue
                entry["content"] = stripped

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)

        session.updated_at = datetime.now()

    @staticmethod
    def _annotate_tools_used(messages: list[dict[str, Any]], tools_used: list[str]) -> None:
        """把本轮使用过的工具集合挂到最后一条 assistant 消息上。"""
        if not tools_used:
            return

        unique_tools = list(dict.fromkeys(tools_used))
        for message in reversed(messages):
            if message.get("role") == "assistant":
                message["tools_used"] = unique_tools
                return

    @staticmethod
    def _strip_runtime_context(content: Any) -> str | list[dict[str, Any]] | None:
        """从 user 消息里移除运行时元信息。

        运行时信息只对当前轮推理有意义，长期保留在 session 里会污染历史，
        所以这里在落盘前主动清理。
        """
        if isinstance(content, str):
            if content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG) or content.startswith(ContextBuilder._LEGACY_RUNTIME_CONTEXT_TAG):
                parts = content.split("\n\n", 1)
                return parts[1] if len(parts) > 1 and parts[1].strip() else None
            return content

        if not isinstance(content, list):
            return content

        # 对于 list 形式的混合内容（例如图片+文本），逐项过滤运行时上下文并把图片替换为占位
        filtered: list[dict[str, Any]] = []
        for item in content:
            if (
                item.get("type") == "text"
                and isinstance(item.get("text"), str)
                and (
                    item["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG)
                    or item["text"].startswith(ContextBuilder._LEGACY_RUNTIME_CONTEXT_TAG)
                )
            ):
                continue
            if (
                item.get("type") == "image_url"
                and item.get("image_url", {}).get("url", "").startswith("data:image/")
            ):
                filtered.append({"type": "text", "text": "[image]"})
            else:
                filtered.append(item)
        return filtered or None

    async def _consolidate_memory(self, session: Session, archive_all: bool = False) -> bool:
        """把会话交给 `MemoryStore` 做长期记忆归档。"""
        # 这里把 session 的未归档段落交给 MemoryStore 处理，
        # MemoryStore 负责生成摘要、向持久化/向量库落盘并决定是否归档到长期记忆。
        return await self.context.memory.consolidate(
            session,
            self.provider,
            self.model,
            archive_all=archive_all,
            memory_window=self.memory_window,
        )

    async def process_direct(
        self,
        content: str,
        session_key: str = "cli:direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """供 CLI 或脚本直接调用的一次性入口。"""
        # 用于脚本/测试/CLI 的同步入口：确保 MCP 已连接（若需要），然后同步处理一条消息并返回文本。
        await self._connect_mcp()
        return await self._process_message(content, session_key=session_key, on_progress=on_progress)
