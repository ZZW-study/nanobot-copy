"""会话记忆与历史归档。

这个模块处理的是"会话太长之后，如何把旧消息压缩成长期可用的信息"。
维护一份简单但稳定的文件：
- `SESSION_MEMORY.md`：持续演进的会话记忆（可被模型读取和更新）

核心类：
    SessionMemoryStore: 封装 memory/ 目录中的读写与归档逻辑
"""

from __future__ import annotations
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any
from loguru import logger
from ZBot.utils.helpers import format_messages, normalize_tool_args,ensure_dir


if TYPE_CHECKING:
    from ZBot.providers.base import LLMProvider
    from ZBot.session.manager import Session



# 系统提示词 + 用户提示词 + 工具定义 --> 大模型返回给的工具定义的参数内容
# 工具定义，一定是你写你想要大模型返回什么的内容,然后你去解析工具参数内容，拿到你想要的结果。
_SAVE_SESSION_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "保存更新后的会话记忆",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_update": {
                        "type": "string",
                        "description": (
                            "更新后的 SESSION_MEMORY.md 内容。\n"
                            "Markdown 格式，按 ## 二级标题分区组织。"
                        ),
                    },
                },
                "required": ["memory_update"],
            },
        },
    }
]


class SessionMemoryStore:
    """
    封装 `memory/` 目录中的读写与归档逻辑。
    """

    def __init__(self, workspace: Path):
        """
        Args:
            workspace: 工作区根目录路径
        """
        self.memory_file = workspace / "memory" / "SESSION_MEMORY.md"     # 会话记忆文件路径
        ensure_dir(self.memory_file.parent)  # 确保目录存在

    

    async def write_session_memory(self, content: str) -> None:
        """覆盖写入 `SESSION_MEMORY.md`。Path.write_text() 方法在文件不存在时会自动创建文件"""
        await asyncio.to_thread(self.memory_file.write_text, content, encoding="utf-8")


    async def get_session_memory_context(self) -> str:
        """
        返回适合直接注入 prompt 的会话记忆文本,给上下文构造用的。
        """
        memory = await self._read_session_memory()
        return f"## SESSION_MEMORY.md\n{memory}" if memory else ""



    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        keep_recent_tokens: int = 16_000,
        consolidate_all: bool = False,
    ) -> bool:
        """
        每次对话的时候进行归档，把会话中的旧消息归档进会话记忆。
        """
        # 确定本次要归档的消息区间和需要保留的尾部消息数量
        messages, keep_count = self._messages_to_archive(session, keep_recent_tokens, consolidate_all)
        if not messages:
            return True  # 没有消息需要归档，直接返回成功

        # 读取当前的会话记忆内容
        current_memory = await self._read_session_memory()
        # 构造归档提示词（包含当前记忆和待归档消息）
        prompt = self._build_prompt(current_memory, messages)

        try:
            # 调用大模型进行历史压缩
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是会话记忆归档助手，负责压缩对话历史以解决上下文过长问题。\n"
                            "⚠️ 必须调用 save_memory 工具返回结果。\n"
                            "只提取当前会话专属信息，不提取跨会话通用信息。"
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_SESSION_MEMORY_TOOL,  # 强制模型使用 save_memory 工具
                model=model,
            )
        except Exception:
            logger.exception("会话记忆归档失败")
            return False

        # 检查模型是否调用了 save_memory 工具
        if not response.has_tool_calls:
            logger.warning("会话记忆归档被跳过：模型没有调用 save_memory 工具")
            return False

        # 规范化工具参数（处理不同格式的返回值）
        args = normalize_tool_args(response.tool_calls[0].arguments)
        if args is None:
            logger.warning("会话记忆归档失败：模型返回的工具参数格式不正确")
            return False

        # 处理会话记忆更新（覆盖写入 SESSION_MEMORY.md）
        memory_update = args.get("memory_update", "").strip()
        await self.write_session_memory(memory_update)

        # 更新会话的归档标记和记忆快照
        session.last_consolidated = len(session.messages) - keep_count

        session.memory_snapshot = memory_update

        logger.info(
            "会话记忆归档完成：本次归档 {} 条消息，last_consolidated={}",
            len(messages),
            session.last_consolidated,
        )
        return True



    def _build_prompt(self, current_memory: str, messages: list[dict[str, Any]]) -> str:
        """把会话记忆和待归档对话整理成提示词。"""
        # 格式化消息列表为转录文本
        transcript = "\n".join(format_messages(messages))
        return (
            "请从以下待归档对话中提取当前会话专属信息，生成更新后的 SESSION_MEMORY.md。\n\n"

            "【提取范围】\n"
            "- 项目信息：目录结构、技术栈、架构设计\n"
            "- 任务进度：待办事项、当前进度\n"
            "- 临时要求：用户本次会话的临时要求\n"
            "- 环境配置：路径、服务地址等\n\n"

            "【不提取】跨会话通用信息（用户长期偏好、通用知识）—— 由日常记忆处理\n\n"

            "【合并规则】\n"
            "- 已有内容无变化的，完整保留\n"
            "- 已有内容有更新/推翻的，原地覆盖更新\n"
            "- 新增信息插入对应分区\n\n"

            "## 当前 SESSION_MEMORY.md 已有内容\n"
            f"{current_memory or '(当前会话记忆为空，首次生成)'}\n\n"
            "## 本次待归档的对话内容\n"
            f"{transcript}"
        )

    @staticmethod
    def _messages_to_archive(
        session: Session,
        keep_recent_tokens: int,
        consolidate_all: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        确定本次归档的消息区间，以及本轮需要保留多少尾部消息。
        """
        if consolidate_all:
            return session.messages[session.last_consolidated :], 0  # 归档所有剩余消息，不保留尾部

        # 归档是会话记忆维护；尾部按 token 保留原文，避免近期细节只剩摘要。
        keep_count = SessionMemoryStore._count_recent_messages_by_token(
            session.messages,
            keep_recent_tokens,
        )
        if len(session.messages) <= keep_count:
            # 消息总数不超过保留数量，无需归档
            return [], keep_count

        # 计算归档范围：从上次归档位置到倒数 keep_count 条消息
        start = session.last_consolidated  # 上次归档结束的位置
        end = len(session.messages) - keep_count  # 保留尾部 keep_count 条
        if end <= start:
            # 归档范围无效（已经归档过了），无需归档
            return [], keep_count

        # 返回要归档的消息片段和保留数量
        return session.messages[start:end], keep_count

    @staticmethod
    def _count_recent_messages_by_token(messages: list[dict[str, Any]], token_budget: int) -> int:
        """从尾部开始按 token 预算计算要保留的最近原文消息数量。"""
        used_tokens = 0
        keep_count = 0
        for message in reversed(messages):
            cost = SessionMemoryStore._estimate_message_tokens(message)
            if keep_count and used_tokens + cost > token_budget:
                break
            keep_count += 1
            used_tokens += cost
        return keep_count

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        """粗略估算消息 token，避免为归档引入 tokenizer 依赖。"""
        import json

        total_chars = len(str(message.get("role", ""))) + len(str(message.get("content", "")))
        if "tool_calls" in message:
            total_chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))
        if "tool_call_id" in message:
            total_chars += len(str(message["tool_call_id"]))
        return max(1, total_chars // 2)

    async def _read_session_memory(self) -> str:
        """读取会话记忆全文（给合并用的）；文件不存在时返回空字符串。"""
        if not self.memory_file.exists():
            return ""
        return await asyncio.to_thread(self.memory_file.read_text, encoding="utf-8")

