"""Agent loop: the core processing engine."""
# 核心作用：这是 nanobot 的**大脑核心**，负责接收消息、调用AI、执行工具、返回响应
from __future__ import annotations

# 标准库导入：异步、JSON、正则、弱引用、上下文管理器、路径、类型注解
import asyncio
import json
import re
import weakref
from contextlib import AsyncExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Any, Awaitable, Callable

# 日志库：打印程序运行日志
from loguru import logger

# 导入项目核心模块
from nanobot.agent.context import ContextBuilder    # 上下文构建器：组装AI需要的对话历史/系统提示
from nanobot.agent.memory import MemoryStore        # 记忆存储：管理长期对话记忆
from nanobot.agent.tools.cron import CronTool       # 定时任务工具
# 文件操作工具：读/写/编辑/列出文件
from nanobot.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from nanobot.agent.tools.registry import ToolRegistry           # 工具注册器：管理所有AI可用工具
from nanobot.agent.tools.shell import ExecTool                  # Shell命令执行工具
from nanobot.agent.tools.web import WebFetchTool, WebSearchTool # 网页抓取/搜索工具
from nanobot.bus.events import InboundMessage, OutboundMessage  # 入站/出站消息模型
from nanobot.bus.queue import MessageBus                        # 消息总线：消息收发的中央管道
from nanobot.providers.base import LLMProvider                  # 大模型接口：对接GPT/DeepSeek等AI
from nanobot.session.manager import Session, SessionManager     # 会话管理：保存用户对话记录

# 类型检查：仅开发时校验类型，不运行
if TYPE_CHECKING:
    from nanobot.config.schema import ExecToolConfig
    from nanobot.cron.service import CronService

# ====================== 核心类：AI大脑引擎 ======================
class AgentLoop:
    """
    AI核心处理引擎（AgentLoop）
    核心工作流程：
    1. 从消息总线接收用户消息
    2. 组装对话上下文（历史+记忆+工具）
    3. 调用大模型（LLM）生成回答
    4. 如果AI需要工具（读文件/搜网页），自动执行
    5. 将最终响应返回给用户
    """

    # 常量：工具执行结果最大字符数（超长自动截断，避免AI上下文溢出）
    _TOOL_RESULT_MAX_CHARS = 500

    # ====================== 初始化方法：创建AI大脑 ======================
    def __init__(
        self,
        bus: MessageBus,          # 消息总线：收发消息
        provider: LLMProvider,    # AI模型提供商（GPT/DeepSeek等）
        workspace: Path,          # 工作目录：文件操作的根目录
        model: str | None = None, # AI模型名称（不填则用默认）
        max_iterations: int = 40, # 最大工具调用次数（防止无限循环）
        temperature: float = 0.1, # AI温度：0=严谨，1=创意
        max_tokens: int = 4096,   # AI最大响应长度
        memory_window: int = 100, # 对话历史窗口（保留最近100条）
        reasoning_effort: str | None = None, # 推理强度
        brave_api_key: str | None = None,    # 网页搜索API密钥
        web_proxy: str | None = None,        # 网页代理
        exec_config: ExecToolConfig | None = None, # Shell命令配置
        cron_service: CronService | None = None,   # 定时任务服务
        restrict_to_workspace: bool = False,       # 是否限制文件操作仅在工作目录内
        session_manager: SessionManager | None = None, # 会话管理器
        mcp_servers: dict | None = None,               # MCP扩展工具服务器
    ):
        # 导入配置类
        from nanobot.config.schema import ExecToolConfig
        # 绑定核心依赖
        self.bus = bus
        self.provider = provider
        self.workspace = workspace
        # AI模型：不传则用提供商默认模型
        self.model = model or provider.get_default_model()
        # AI行为参数
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_window = memory_window
        self.reasoning_effort = reasoning_effort
        # 工具配置
        self.brave_api_key = brave_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.cron_service = cron_service
        self.restrict_to_workspace = restrict_to_workspace

        # 上下文构建器：负责给AI组装对话上下文
        self.context = ContextBuilder(workspace)
        # 会话管理器：管理每个用户的对话记录
        self.sessions = session_manager or SessionManager(workspace)
        # 工具注册器：存储AI可用的所有工具
        self.tools = ToolRegistry()

        # 状态标志：引擎是否运行
        self._running = False
        # MCP扩展工具相关
        self._mcp_servers = mcp_servers or {}
        self._mcp_stack: AsyncExitStack | None = None
        self._mcp_connected = False
        self._mcp_connecting = False
        # 记忆 Consolidate（压缩对话历史）相关
        self._consolidating: set[str] = set()
        self._consolidation_tasks: set[asyncio.Task] = set()
        self._consolidation_locks: weakref.WeakValueDictionary[str, asyncio.Lock] = weakref.WeakValueDictionary()
        # 活跃任务：存储每个会话的正在运行的任务（用于/stop停止）
        self._active_tasks: dict[str, list[asyncio.Task]] = {}
        # 处理锁：保证消息串行处理，避免并发混乱
        self._processing_lock = asyncio.Lock()
        # 注册默认工具（文件/Shell/网页/消息等）
        self._register_default_tools()

    # ====================== 注册AI默认可用工具 ======================
    def _register_default_tools(self) -> None:
        """注册AI默认可用的所有工具，AI可以直接调用这些功能"""
        # 如果限制工作目录，则文件操作仅允许在workspace内
        allowed_dir = self.workspace if self.restrict_to_workspace else None
        # 注册文件操作工具
        for cls in (ReadFileTool, WriteFileTool, EditFileTool, ListDirTool):
            self.tools.register(cls(workspace=self.workspace, allowed_dir=allowed_dir))
        # 注册Shell命令执行工具
        self.tools.register(ExecTool(
            working_dir=str(self.workspace),
            timeout=self.exec_config.timeout,
            restrict_to_workspace=self.restrict_to_workspace,
            path_append=self.exec_config.path_append,
        ))
        # 注册网页搜索/抓取工具
        self.tools.register(WebSearchTool(proxy=self.web_proxy))
        self.tools.register(WebFetchTool(proxy=self.web_proxy))
        # 如果有定时任务服务，注册定时工具
        if self.cron_service:
            self.tools.register(CronTool(self.cron_service))

    # ====================== 连接MCP扩展工具 ======================
    async def _connect_mcp(self) -> None:
        """连接MCP服务器（扩展AI工具，一次性懒加载）"""
        # 如果已连接/正在连接/无MCP配置，直接返回
        if self._mcp_connected or self._mcp_connecting or not self._mcp_servers:
            return
        self._mcp_connecting = True
        from nanobot.agent.tools.mcp import connect_mcp_servers
        try:
            self._mcp_stack = AsyncExitStack()
            await self._mcp_stack.__aenter__()
            # 连接所有MCP服务器，加载扩展工具
            await connect_mcp_servers(self._mcp_servers, self.tools, self._mcp_stack)
            self._mcp_connected = True
        except Exception as e:
            logger.error("Failed to connect MCP servers (will retry next message): {}", e)
            if self._mcp_stack:
                try:
                    await self._mcp_stack.aclose()
                except Exception:
                    pass
                self._mcp_stack = None
        finally:
            self._mcp_connecting = False

    # ====================== 设置工具上下文（渠道/聊天ID） ======================
    def _set_tool_context(self, channel: str, chat_id: str, message_id: str | None = None) -> None:
        """给工具设置上下文：告诉工具消息来自哪个渠道、哪个聊天"""
        for name in ("cron",):
            if tool := self.tools.get(name):
                if hasattr(tool, "set_context"):
                    tool.set_context(channel, chat_id)

    # ====================== 清理AI思考标签 ======================
    @staticmethod
    def _strip_think(text: str | None) -> str | None:
        """删除AI返回的思考标签，只保留最终回答"""
        if not text:
            return None
        return re.sub(r"[\s\S]*?", "", text).strip() or None

    # ====================== 格式化工具调用提示 ======================
    @staticmethod
    def _tool_hint(tool_calls: list) -> str:
        """将AI的工具调用格式化为简洁提示，例如：web_search("Python")"""
        def _fmt(tc):
            args = (tc.arguments[0] if isinstance(tc.arguments, list) else tc.arguments) or {}
            val = next(iter(args.values()), None) if isinstance(args, dict) else None
            if not isinstance(val, str):
                return tc.name
            return f'{tc.name}("{val[:40]}…")' if len(val) > 40 else f'{tc.name}("{val}")'
        return ", ".join(_fmt(tc) for tc in tool_calls)

    # ====================== 核心AI循环：调用LLM+执行工具 ======================
    async def _run_agent_loop(
        self,
        initial_messages: list[dict], # 初始上下文（对话历史+当前消息）
        on_progress: Callable[..., Awaitable[None]] | None = None, # 进度回调
    ) -> tuple[str | None, list[str], list[dict]]:
        """
        运行AI核心循环
        返回：(最终回答, 使用的工具列表, 完整对话消息)
        """
        messages = initial_messages
        iteration = 0
        final_content = None
        tools_used: list[str] = []

        # 循环调用AI，最多执行max_iterations次（防止无限工具调用）
        while iteration < self.max_iterations:
            iteration += 1

            # 调用大模型AI
            response = await self.provider.chat(
                messages=messages,
                tools=self.tools.get_definitions(), # 传给AI所有可用工具
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                reasoning_effort=self.reasoning_effort,
            )

            # 如果AI需要调用工具
            if response.has_tool_calls:
                # 发送进度提示：AI的思考内容+工具调用提示
                if on_progress:
                    thought = self._strip_think(response.content)
                    if thought:
                        await on_progress(thought)
                    await on_progress(self._tool_hint(response.tool_calls), tool_hint=True)

                # 格式化工具调用消息，加入对话上下文
                tool_call_dicts = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments, ensure_ascii=False)
                        }
                    }
                    for tc in response.tool_calls
                ]
                messages = self.context.add_assistant_message(
                    messages, response.content, tool_call_dicts,
                    reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )

                # 遍历所有工具调用，逐个执行
                for tool_call in response.tool_calls:
                    tools_used.append(tool_call.name)
                    args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                    logger.info("Tool call: {}({})", tool_call.name, args_str[:200])
                    # 执行工具（读文件/搜网页/执行命令）
                    result = await self.tools.execute(tool_call.name, tool_call.arguments)
                    # 将工具执行结果加入上下文，让AI看到结果
                    messages = self.context.add_tool_result(
                        messages, tool_call.id, tool_call.name, result
                    )
            # 如果AI不需要工具，直接返回回答
            else:
                clean = self._strip_think(response.content)
                # 如果AI返回错误，不保存到历史，避免污染上下文
                if response.finish_reason == "error":
                    logger.error("LLM returned error: {}", (clean or "")[:200])
                    final_content = clean or "Sorry, I encountered an error calling the AI model."
                    break
                # 保存AI的最终回答
                messages = self.context.add_assistant_message(
                    messages, clean, reasoning_content=response.reasoning_content,
                    thinking_blocks=response.thinking_blocks,
                )
                final_content = clean
                break

        # 达到最大迭代次数，返回提示
        if final_content is None and iteration >= self.max_iterations:
            logger.warning("Max iterations ({}) reached", self.max_iterations)
            final_content = (
                f"I reached the maximum number of tool call iterations ({self.max_iterations}) "
                "without completing the task. You can try breaking the task into smaller steps."
            )

        return final_content, tools_used, messages

    # ====================== 启动AI引擎 ======================
    async def run(self) -> None:
        """启动AI核心循环，持续消费消息总线的消息"""
        self._running = True
        # 连接MCP扩展工具
        await self._connect_mcp()
        logger.info("Agent loop started")

        # 无限循环：监听消息总线
        while self._running:
            try:
                # 每1秒监听一次消息
                msg = await asyncio.wait_for(self.bus.consume_inbound(), timeout=1.0)
            except asyncio.TimeoutError:
                continue

            # 如果用户发送 /stop 命令，停止当前任务
            if msg.content.strip().lower() == "/stop":
                await self._handle_stop(msg)
            # 否则，异步处理消息（不阻塞主线程）
            else:
                task = asyncio.create_task(self._dispatch(msg))
                self._active_tasks.setdefault(msg.session_key, []).append(task)
                # 任务完成后，从活跃任务列表移除
                task.add_done_callback(lambda t, k=msg.session_key: self._active_tasks.get(k, []) and self._active_tasks[k].remove(t) if t in self._active_tasks.get(k, []) else None)

    # ====================== 处理 /stop 命令：停止当前任务 ======================
    async def _handle_stop(self, msg: InboundMessage) -> None:
        """取消当前会话的所有正在运行的任务"""
        tasks = self._active_tasks.pop(msg.session_key, [])
        cancelled = sum(1 for t in tasks if not t.done() and t.cancel())
        # 等待任务取消完成
        for t in tasks:
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        # 返回停止提示
        content = f"⏹ Stopped {cancelled} task(s)." if cancelled else "No active task to stop."
        await self.bus.publish_outbound(OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=content,
        ))

    # ====================== 消息分发：加锁保证串行处理 ======================
    async def _dispatch(self, msg: InboundMessage) -> None:
        """分发消息，加锁保证同一时间只处理一条消息（避免并发混乱）"""
        async with self._processing_lock:
            try:
                # 处理消息，获取响应
                response = await self._process_message(msg)
                # 将响应发送到消息总线
                if response is not None:
                    await self.bus.publish_outbound(response)
                elif msg.channel == "cli":
                    await self.bus.publish_outbound(OutboundMessage(
                        channel=msg.channel, chat_id=msg.chat_id,
                        content="", metadata=msg.metadata or {},
                    ))
            except asyncio.CancelledError:
                logger.info("Task cancelled for session {}", msg.session_key)
                raise
            except Exception:
                logger.exception("Error processing message for session {}", msg.session_key)
                # 异常时返回错误提示
                await self.bus.publish_outbound(OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Sorry, I encountered an error.",
                ))

    # ====================== 关闭MCP连接 ======================
    async def close_mcp(self) -> None:
        """关闭MCP扩展工具连接"""
        if self._mcp_stack:
            try:
                await self._mcp_stack.aclose()
            except (RuntimeError, BaseExceptionGroup):
                pass
            self._mcp_stack = None

    # ====================== 停止AI引擎 ======================
    def stop(self) -> None:
        """停止AI核心循环"""
        self._running = False
        logger.info("Agent loop stopping")

    # ====================== 核心：处理单条用户消息 ======================
    async def _process_message(
        self,
        msg: InboundMessage,          # 入站用户消息
        session_key: str | None = None, # 会话ID
        on_progress: Callable[[str], Awaitable[None]] | None = None, # 进度回调
    ) -> OutboundMessage | None:
        """处理单条消息，返回出站响应（核心业务逻辑）"""
        # 处理系统消息
        if msg.channel == "system":
            channel, chat_id = (msg.chat_id.split(":", 1) if ":" in msg.chat_id
                                else ("cli", msg.chat_id))
            logger.info("Processing system message from {}", msg.sender_id)
            key = f"{channel}:{chat_id}"
            session = self.sessions.get_or_create(key)
            self._set_tool_context(channel, chat_id, msg.metadata.get("message_id"))
            # 获取对话历史
            history = session.get_history(max_messages=self.memory_window)
            # 构建AI上下文
            messages = self.context.build_messages(
                history=history,
                current_message=msg.content, channel=channel, chat_id=chat_id,
            )
            # 运行AI核心循环
            final_content, _, all_msgs = await self._run_agent_loop(messages)
            # 保存对话记录
            self._save_turn(session, all_msgs, 1 + len(history))
            self.sessions.save(session)
            return OutboundMessage(channel=channel, chat_id=chat_id,
                                  content=final_content or "Background task completed.")

        # 日志：打印收到的消息
        preview = msg.content[:80] + "..." if len(msg.content) > 80 else msg.content
        logger.info("Processing message from {}:{}: {}", msg.channel, msg.sender_id, preview)

        # 获取/创建用户会话（每个用户独立对话记录）
        key = session_key or msg.session_key
        session = self.sessions.get_or_create(key)

        # ====================== 处理用户命令 ======================
        cmd = msg.content.strip().lower()
        # 命令1：/new 新建会话（清空对话历史）
        if cmd == "/new":
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())
            self._consolidating.add(session.key)
            try:
                async with lock:
                    snapshot = session.messages[session.last_consolidated:]
                    if snapshot:
                        temp = Session(key=session.key)
                        temp.messages = list(snapshot)
                        if not await self._consolidate_memory(temp, archive_all=True):
                            return OutboundMessage(
                                channel=msg.channel, chat_id=msg.chat_id,
                                content="Memory archival failed, session not cleared. Please try again.",
                            )
            except Exception:
                logger.exception("/new archival failed for {}", session.key)
                return OutboundMessage(
                    channel=msg.channel, chat_id=msg.chat_id,
                    content="Memory archival failed, session not cleared. Please try again.",
                )
            finally:
                self._consolidating.discard(session.key)

            session.clear() # 清空会话
            self.sessions.save(session)
            self.sessions.invalidate(session.key)
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="New session started.")
        # 命令2：/help 帮助
        if cmd == "/help":
            return OutboundMessage(channel=msg.channel, chat_id=msg.chat_id,
                                  content="🐈 nanobot commands:\n/new — 新建对话\n/stop — 停止任务\n/help — 帮助")

        # 自动压缩对话历史（防止过长）
        unconsolidated = len(session.messages) - session.last_consolidated
        if (unconsolidated >= self.memory_window and session.key not in self._consolidating):
            self._consolidating.add(session.key)
            lock = self._consolidation_locks.setdefault(session.key, asyncio.Lock())

            async def _consolidate_and_unlock():
                try:
                    async with lock:
                        await self._consolidate_memory(session)
                finally:
                    self._consolidating.discard(session.key)
                    _task = asyncio.current_task()
                    if _task is not None:
                        self._consolidation_tasks.discard(_task)

            _task = asyncio.create_task(_consolidate_and_unlock())
            self._consolidation_tasks.add(_task)

        # 设置工具上下文
        self._set_tool_context(msg.channel, msg.chat_id, msg.metadata.get("message_id"))

        # 获取对话历史
        history = session.get_history(max_messages=self.memory_window)
        # 构建AI上下文（支持图片/文本消息）
        initial_messages = self.context.build_messages(
            history=history,
            current_message=msg.content,
            media=msg.media if msg.media else None,
            channel=msg.channel, chat_id=msg.chat_id,
        )

        # 进度回调：实时发送AI思考/工具调用提示
        async def _bus_progress(content: str, *, tool_hint: bool = False) -> None:
            meta = dict(msg.metadata or {})
            meta["_progress"] = True
            meta["_tool_hint"] = tool_hint
            await self.bus.publish_outbound(OutboundMessage(
                channel=msg.channel, chat_id=msg.chat_id, content=content, metadata=meta,
            ))

        # 运行AI核心循环，生成最终回答
        final_content, _, all_msgs = await self._run_agent_loop(
            initial_messages, on_progress=on_progress or _bus_progress,
        )

        # 兜底：如果AI无响应，返回默认文本
        if final_content is None:
            final_content = "I've completed processing but have no response to give."

        # 保存对话记录到会话
        self._save_turn(session, all_msgs, 1 + len(history))
        self.sessions.save(session)

        # 日志：打印AI响应
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("Response to {}:{}: {}", msg.channel, msg.sender_id, preview)
        # 返回最终响应
        return OutboundMessage(
            channel=msg.channel, chat_id=msg.chat_id, content=final_content,
            metadata=msg.metadata or {},
        )

    # ====================== 保存对话轮次 ======================
    def _save_turn(self, session: Session, messages: list[dict], skip: int) -> None:
        """保存一轮对话（用户消息+AI响应），截断超长工具结果"""
        from datetime import datetime
        for m in messages[skip:]:
            entry = dict(m)
            role, content = entry.get("role"), entry.get("content")
            # 跳过空消息
            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue
            # 截断超长工具结果
            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[:self._TOOL_RESULT_MAX_CHARS] + "\n... (truncated)"
            # 清理用户消息的上下文标签
            elif role == "user":
                if isinstance(content, str) and content.startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                    parts = content.split("\n\n", 1)
                    if len(parts) > 1 and parts[1].strip():
                        entry["content"] = parts[1]
                    else:
                        continue
                if isinstance(content, list):
                    filtered = []
                    for c in content:
                        if c.get("type") == "text" and isinstance(c.get("text"), str) and c["text"].startswith(ContextBuilder._RUNTIME_CONTEXT_TAG):
                            continue
                        if (c.get("type") == "image_url"
                                and c.get("image_url", {}).get("url", "").startswith("data:image/")):
                            filtered.append({"type": "text", "text": "[image]"})
                        else:
                            filtered.append(c)
                    if not filtered:
                        continue
                    entry["content"] = filtered
            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)
        session.updated_at = datetime.now()

    # ====================== 记忆 Consolidate：压缩对话历史 ======================
    async def _consolidate_memory(self, session, archive_all: bool = False) -> bool:
        """压缩对话历史，将长对话总结为记忆，节省上下文空间"""
        return await MemoryStore(self.workspace).consolidate(
            session, self.provider, self.model,
            archive_all=archive_all, memory_window=self.memory_window,
        )

    # ====================== 直接调用AI（CLI/定时任务） ======================
    async def process_direct(
        self,
        content: str,                # 消息内容
        session_key: str = "cli:direct",
        channel: str = "cli",
        chat_id: str = "direct",
        on_progress: Callable[[str], Awaitable[None]] | None = None,
    ) -> str:
        """直接处理消息（用于CLI命令行或定时任务）"""
        await self._connect_mcp()
        msg = InboundMessage(channel=channel, sender_id="user", chat_id=chat_id, content=content)
        response = await self._process_message(msg, session_key=session_key, on_progress=on_progress)
        return response.content if response else ""
    
    