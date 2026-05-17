"""主 Agent 会话处理模块。"""
from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from loguru import logger

from ZBot.agent.base_agent import BaseAgent
from ZBot.agent.context import ContextBuilder
from ZBot.agent.subagent.subagent_pool import SubAgentPool
from ZBot.agent.tools.create_sub_agent import CreateSubAgentTool
from ZBot.config.agent_runtime import AgentRuntimeConfig
from ZBot.cron.service import CronService
from ZBot.providers.base import LLMProvider
from ZBot.session.manager import Session, SessionManager


class CoreAgent(BaseAgent):
    """主 Agent：负责用户会话、记忆、工具调度和最终回复。"""

    def __init__(
        self,
        provider: LLMProvider,
        runtime_config: AgentRuntimeConfig,
        cron_service: CronService | None = None,
    ):
        """初始化主 Agent 的长期会话能力。"""
        super().__init__(
            provider=provider,
            runtime_config=runtime_config,
            cron_service=cron_service,
        )

        self.context = ContextBuilder(self.workspace)
        self.sessions = SessionManager(self.workspace)
        self.recent_history_token_budget_ratio = runtime_config.recent_history_token_budget_ratio
        self.recent_history_max_tokens = runtime_config.recent_history_max_tokens
        self.memory_consolidation_interval = runtime_config.memory_consolidation_interval
        self.session_memory_keep_recent_tokens = runtime_config.session_memory_keep_recent_tokens
        self._is_consolidating: bool = False
        self.subagent_pool: SubAgentPool | None = None
        self._register_core_tools()

    async def process_message(
        self,
        message: str,
        session_name: str = "default",
        *,
        on_progress: Callable[..., Awaitable[None]],
    ) -> str:
        """处理一条用户消息，是 CLI 和上层调用 CoreAgent 的主入口。"""
        await self.connect_mcp()
        self.subagent_pool = self.ensure_subagent_pool()

        logger.info("正在处理消息：{}", message[:80] + "..." if len(message) > 80 else message)

        session, is_load = await self.sessions.get_or_create(session_name)
        if is_load:
            logger.info("会话 '{}' 已加载，包含 {} 条历史消息", session_name, len(session.messages))
            await self.context.session_memory.write_session_memory(session.memory_snapshot or "无记忆快照")

        self._schedule_consolidation(session)

        final_content = await self._run_turn(
            session,
            content=message,
            on_progress=on_progress,
        )
        preview = final_content[:120] + "..." if len(final_content) > 120 else final_content
        logger.info("回复：{}", preview)
        return final_content

    async def close_mcp(self) -> None:
        """关闭子 Agent 池和 MCP 连接栈。"""
        if self.subagent_pool is not None:
            await self.subagent_pool.close()
            self.subagent_pool = None
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

    async def consolidate_all_session_memory(self, session_name: str) -> None:
        """对指定会话执行完整会话记忆归档。"""
        session, _ = await self.sessions.get_or_create(session_name)
        await self.context.session_memory.consolidate(
            session,
            self.provider,
            self.model,
            keep_recent_tokens=self.session_memory_keep_recent_tokens,
            consolidate_all=True,
        )
        await self.sessions.save(session)

    async def consolidate_daily_memory(self, session_name: str) -> None:
        """把指定会话整理进日常记忆。"""
        session, _ = await self.sessions.get_or_create(session_name)
        consolidate_daily_memory_result = await self.context.daily_memory.add_daily_memory(
            self.provider,
            self.model,
            session,
        )
        if not consolidate_daily_memory_result:
            logger.error("日常记忆归档失败")
        else:
            logger.info("日常记忆归档成功")

    def ensure_subagent_pool(self) -> SubAgentPool:
        """确保子 Agent 池已创建，并返回池实例。"""
        if self.subagent_pool is None:
            self.subagent_pool = SubAgentPool(self)
        return self.subagent_pool

    def _register_core_tools(self) -> None:
        """注册只有主 Agent 才能使用的工具。"""
        sub_agent_tool = CreateSubAgentTool()
        sub_agent_tool.bind_agent(self)
        self.tools.register(sub_agent_tool)

    async def _run_turn(
        self,
        session: Session,
        *,
        content: str,
        on_progress: Callable[..., Awaitable[None]],
    ) -> str:
        """执行一轮对话：构造上下文、运行 Agent loop、写回会话。"""
        history = session.get_history_by_token_budget(self._recent_history_token_budget())

        initial_messages = await self.context.build_messages(
            history=history,
            user_message=content,
            score_threshold=self.score_threshold,
        )

        final_content, tools_used, all_messages = await self.run_agent_loop(
            initial_messages,
            on_progress=on_progress,
        )
        final_content = final_content or "我已经完成处理，但没有需要额外返回的内容。"

        self._save_turn(session, all_messages, 1 + len(history), tools_used)
        await self.sessions.save(session)
        return final_content

    def _schedule_consolidation(self, session: Session) -> None:
        """当未归档消息达到阈值时，安排后台会话记忆归档任务。"""
        unconsolidated = len(session.messages) - session.last_consolidated
        if unconsolidated < self.memory_consolidation_interval or self._is_consolidating:
            return

        async def _run_consolidation() -> None:
            """执行后台归档，并在结束时回收归档状态标记。"""
            try:
                await self.context.session_memory.consolidate(
                    session,
                    self.provider,
                    self.model,
                    keep_recent_tokens=self.session_memory_keep_recent_tokens,
                )
            finally:
                self._is_consolidating = False

        self._is_consolidating = True
        asyncio.create_task(_run_consolidation())

    def _recent_history_token_budget(self) -> int:
        """按模型上下文窗口计算短期历史预算，并设置 64K 默认硬上限。"""
        ratio_budget = int(self.context_window * self.recent_history_token_budget_ratio)
        return max(1, min(ratio_budget, self.recent_history_max_tokens))

    def _save_turn(
        self,
        session: Session,
        messages: list[dict[str, Any]],
        skip: int,
        tools_used: list[str] | None = None,
    ) -> None:
        """把本轮新增消息写回 session。"""
        from datetime import datetime

        turn_messages = [dict(message) for message in messages[skip:]]
        self._annotate_tools_used(turn_messages, tools_used or [])

        for entry in turn_messages:
            role = entry.get("role")
            content = entry.get("content")

            if role == "assistant" and not content and not entry.get("tool_calls"):
                continue

            if role == "tool" and isinstance(content, str) and len(content) > self._TOOL_RESULT_MAX_CHARS:
                entry["content"] = content[: self._TOOL_RESULT_MAX_CHARS] + "\n……（内容已截断）"
            elif role == "user":
                entry["content"] = self._strip_runtime_context(content)

            entry.setdefault("timestamp", datetime.now().isoformat())
            session.messages.append(entry)

        session.updated_at = datetime.now()

    @staticmethod
    def _strip_runtime_context(content: str | None) -> str:
        """剥离用户消息中的运行时上下文标签，只保留纯净用户输入。"""
        if not content:
            return content or ""

        runtime_tag = ContextBuilder._RUNTIME_CONTEXT_TAG
        if content.startswith(runtime_tag):
            lines = content.split("\n\n", 1)
            if len(lines) > 1:
                return lines[1].strip()
            return ""

        return content

    @staticmethod
    def _annotate_tools_used(messages: list[dict[str, Any]], tools_used: list[str]) -> None:
        """把本轮使用过的工具名挂到最后一条 assistant 消息上。"""
        if not tools_used:
            return

        unique_tools = list(dict.fromkeys(tools_used))
        for message in reversed(messages):
            if message.get("role") == "assistant":
                message["tools_used"] = unique_tools
                return
