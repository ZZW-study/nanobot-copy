"""会话持久化管理。

这个模块负责管理对话会话（Session）的：
1. 内存缓存：加速频繁访问
2. 磁盘持久化：使用 JSONL 格式存储
3. 生命周期管理：创建、加载、保存、删除

文件格式：
- 每个会话存储在独立的 .jsonl 文件中
- 第一行是 metadata 元数据
- 后续每行是一条消息（JSON 格式）
- 优点：易于追加写入，适合对话历史场景

核心类：
    Session: 单个会话对象，包含消息列表和元信息
    SessionManager: 会话管理器，负责多会话的 CRUD 操作
"""

from __future__ import annotations
import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from loguru import logger

from ZBot.utils.helpers import ensure_dir, safe_filename

# 历史消息中需要特别保留的字段
# 这些字段在转换历史记录时需要特殊处理
_HISTORY_FIELDS = ("tool_calls", "tool_call_id", "name")


@dataclass
class Session:
    """单个会话对象。"""

    session_name: str                                               # 会话名称
    messages: list[dict[str, Any]] = field(default_factory=list)    # 消息列表
    created_at: datetime = field(default_factory=datetime.now)      # 创建时间，打印才是年-月-日-时-分-秒，不然就是datatime对象，如果要保存不能用这个，必须在后面加isoformat（），变成字符串对象，保存
    updated_at: datetime = field(default_factory=datetime.now)      # 更新时间
    last_consolidated: int = 0                                      # 已归档的消息索引（用于会话记忆）
    memory_snapshot: str | None = None                              # 记忆快照（归档时保存的摘要信息）


    def get_history_by_token_budget(self, token_budget: int) -> list[dict[str, Any]]:
        """按 token 预算返回最近历史，避免固定消息条数误判上下文大小。"""
        messages = self.messages[self.last_consolidated :]
        if not messages:
            return []

        selected: list[dict[str, Any]] = []
        used_tokens = 0
        for message in reversed(messages):
            cost = self._estimate_message_tokens(message)
            if selected and used_tokens + cost > token_budget:
                break
            selected.append(message)
            used_tokens += cost

        selected.reverse()
        selected = self._trim_to_first_user(selected)
        selected = self._drop_unpaired_tool_prefix(selected)
        return [self._history_entry(message) for message in selected]


    def clear(self) -> None:
        """清空会话。"""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()
        self.memory_snapshot = None

    @staticmethod
    def _estimate_message_tokens(message: dict[str, Any]) -> int:
        """粗略估算单条消息 token，和 Agent loop 的轻量估算保持同一思路。"""
        total_chars = len(str(message.get("role", ""))) + len(str(message.get("content", "")))
        if "tool_calls" in message:
            total_chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))
        if "tool_call_id" in message:
            total_chars += len(str(message["tool_call_id"]))
        return max(1, total_chars // 2)

    @staticmethod
    def _trim_to_first_user(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """历史上下文从第一条 user 消息开始，避免 assistant/tool 无来源地开头。"""
        first_user = next((index for index, message in enumerate(messages) if message.get("role") == "user"), None)
        return messages[first_user:] if first_user is not None else messages

    @staticmethod
    def _drop_unpaired_tool_prefix(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """去掉开头没有 assistant tool_call 配对的 tool 消息，保持协议合法。"""
        while messages and messages[0].get("role") == "tool":
            messages = messages[1:]
        return messages

    @staticmethod
    def _history_entry(message: dict[str, Any]) -> dict[str, Any]:
        """构造发给模型的历史消息，只保留标准字段和必要工具链字段。"""
        entry = {"role": message["role"], "content": message.get("content", "")}
        for field_name in _HISTORY_FIELDS:
            if field_name in message:
                entry[field_name] = message[field_name]
        return entry


class SessionManager:
    """
    会话文件管理器。
    """

    def __init__(self, workspace: Path | str):
        """初始化 SessionManager"""
        self.sessions_dir = ensure_dir(Path(workspace) / "sessions")
        # 内存缓存：key -> Session 映射，用于加速频繁访问
        self._cache: dict[str, Session] = {}


    async def get_or_create(self, session_name: str) -> tuple[Session, bool]:
        """获取或创建会话。先从缓存查找，如果没有则从磁盘加载；如果磁盘上也没有，则创建一个新的空会话。"""
        session = self._cache.get(session_name)
        if session is None:
            session = await self._load(session_name)
            if session:
                return session,True  #加载成功，返回会话和标记
            session = Session(session_name=session_name)  # 创建新会话
            self._cache[session_name] = session
        return session,False  #返回会话和标记，标记为False表示新创建的会话


    async def save(self, session: Session) -> None:
        """保存会话到磁盘,同一会话名字，都是追加写入。"""
        path: Path = self._session_path(session.session_name)
        lines: list[str] = [json.dumps(self._metadata_line(session), ensure_ascii=False)]
        lines.extend(json.dumps(message, ensure_ascii=False) for message in session.messages)
        # asyncio.to_thread：将同步 IO 操作放到线程池执行，避免阻塞事件循环
        # 原理：
        #   1. Python 事件循环自带默认线程池，首次使用时自动创建
        #   2. to_thread 会从线程池取一个线程，在其中执行同步函数
        #   3. 主协程 await 等待线程完成，期间事件循环可处理其他协程
        #   4. 线程完成后返回结果，主协程继续执行
        def write_file():
            """在线程池中把会话内容追加写入磁盘。"""
            with open(path, mode="a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        await asyncio.to_thread(write_file)
        self._cache[session.session_name] = session


    async def _load(self, session_name: str) -> Session | None:
        """从磁盘加载会话。"""
        path = self._session_path(session_name)
        if not path.exists():
            return None

        try:
            # 异步读取文件内容
            content = await asyncio.to_thread(path.read_text, encoding="utf-8")
            messages: list[dict[str, Any]] = []
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated: int = 0
            memory_snapshot: str | None = None

            # 逐行解析 JSONL 文件
            for line in content.splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)

                # 第一行是元数据（_type == "metadata"）
                if data.get("_type") == "metadata":
                    created_at = self._parse_datetime(data.get("created_at"))
                    updated_at = self._parse_datetime(data.get("updated_at"))
                    last_consolidated = data.get("last_consolidated", 0)
                    memory_snapshot = data.get("memory_snapshot")
                    continue

                # 其他行是消息记录
                messages.append(data)

            # 构造 Session 对象，时间字段若无则用当前时间
            now = datetime.now()
            return Session(
                session_name=session_name,
                messages=messages,
                created_at=created_at or now,
                updated_at=updated_at or created_at or now,
                last_consolidated=last_consolidated,
                memory_snapshot=memory_snapshot,
            )
        except Exception as exc:
            logger.warning("加载会话失败 {}: {}", session_name, exc)
            return None

    def _session_path(self, session_name: str) -> Path:
        """返回会话文件的完整路径。"""
        safe_name = safe_filename(session_name)  # 转义非法字符
        return self.sessions_dir / f"{safe_name}.jsonl"

    @staticmethod
    def _metadata_line(session: Session) -> dict[str, Any]:
        """
        生成元数据行的字典。元数据包含会话的基本信息，但不包含大量消息内容，便于快速读取和使用。
        """
        return {
            "_type": "metadata",          # 标记这是元数据行
            "name": session.session_name, # 会话名称
            "created_at": session.created_at.isoformat(),  # 创建时间
            "updated_at": session.updated_at.isoformat(),  # 更新时间
            "last_consolidated": session.last_consolidated,  # 归档索引
            "memory_snapshot": session.memory_snapshot,  # 记忆快照
        }

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        """
        解析 ISO 格式的时间字符串。

        Args:
            value: ISO 格式的时间字符串（如 "2024-01-15T14:30:00"）

        Returns:
            datetime 对象，如果值为 None 则返回 None
        """
        return datetime.fromisoformat(value) if value else None
