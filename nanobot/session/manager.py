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

from __future__ import annotations  # 启用未来版本的类型注解特性

import json  # 用于 JSON 格式的数据读写
from dataclasses import dataclass, field  # 用于定义数据结构
from datetime import datetime  # 用于时间戳
from pathlib import Path  # 用于路径操作
from typing import Any  # 用于类型注解

from loguru import logger  # 用于日志记录

from nanobot.utils.helpers import ensure_dir, safe_filename  # 工具函数


# 历史消息中需要特别保留的字段
# 这些字段在转换历史记录时需要特殊处理
_HISTORY_FIELDS = ("tool_calls", "tool_call_id", "name")


@dataclass
class Session:
    """
    单个会话对象。

    表示一次完整的对话会话，包含：
    - key: 会话唯一标识符（如"cli:default"或"user:123"）
    - messages: 消息列表（每条消息包含 role、content 等）
    - created_at: 会话创建时间
    - updated_at: 最后更新时间
    - metadata: 可选的元数据字典
    - last_consolidated: 已归档的消息索引（用于长期记忆机制）

    用法示例：
        session = Session(key="user:123")
        session.add_message("user", "你好")
        session.add_message("assistant", "你好！有什么可以帮助你的吗？")
    """

    key: str  # 会话的唯一标识符
    messages: list[dict[str, Any]] = field(default_factory=list)  # 消息列表
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间
    metadata: dict[str, Any] = field(default_factory=dict)  # 自定义元数据
    last_consolidated: int = 0  # 已归档的消息索引（用于长期记忆）

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        """
        向会话追加消息。

        Args:
            role: 消息角色（"user"、"assistant"、"tool"）
            content: 消息内容
            **kwargs: 其他可选字段（如 tool_calls、tool_call_id 等）
        """
        self.messages.append(
            {
                "role": role,           # 消息角色
                "content": content,     # 消息内容
                "timestamp": datetime.now().isoformat(),  # 时间戳
                **kwargs,               # 其他字段（如工具调用信息）
            }
        )
        # 更新最后修改时间
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        """
        返回历史消息列表（用于构造模型上下文）。

        获取逻辑：
        1. 从 last_consolidated 索引开始截取（跳过已归档的历史）
        2. 限制最多返回 max_messages 条
        3. 如果第一条不是 user 消息，则截断到第一个 user 消息
           （避免把 assistant/tool 消息作为上下文的起点）

        Args:
            max_messages: 最多返回的消息数量（默认 500）

        Returns:
            过滤后的历史消息列表
        """
        # 从上次归档位置到末尾，再取最近的 max_messages 条
        messages = self.messages[self.last_consolidated :][-max_messages:]

        # 找到第一条 user 消息的位置
        first_user = next((index for index, message in enumerate(messages) if message.get("role") == "user"), None)
        if first_user is not None:
            # 从第一条 user 消息开始截断
            messages = messages[first_user:]

        history: list[dict[str, Any]] = []
        for message in messages:
            # 构造标准格式的消息条目
            entry = {"role": message["role"], "content": message.get("content", "")}
            # 额外保留特殊字段（如工具调用信息）
            for f in _HISTORY_FIELDS:
                if f in message:
                    entry[f] = message[f]
            history.append(entry)
        return history

    def clear(self) -> None:
        """清空会话。"""
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    """
    会话文件管理器。

    负责任务多会话的持久化存储和管理：
    1. 加载会话（从磁盘文件）
    2. 保存会话（写回磁盘）
    3. 缓存管理（内存加速访问）
    4. 列出所有会话

    文件布局：
    sessions/
      cli_default.jsonl   # CLI 默认会话
      user_123.jsonl      # 用户 123 的会话
      ...

    每个文件包含：
    - 第一行：元数据（JSON 对象）
    - 后续行：消息（每行一个 JSON 对象）
    """

    def __init__(self, workspace: Path | str):
        """
        初始化 SessionManager。

        Args:
            workspace: 工作区根目录路径
        """
        # 会话目录：workspace/sessions/
        self.sessions_dir = ensure_dir(Path(workspace) / "sessions")
        # 内存缓存：key -> Session 映射，用于加速频繁访问
        self._cache: dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        """
        获取或创建会话。

        先从缓存查找，如果没有则从磁盘加载；
        如果磁盘上也没有，则创建一个新的空会话。

        Args:
            key: 会话标识符（如"cli:default"）

        Returns:
            会话对象（从缓存或磁盘加载，或新建）
        """
        session = self._cache.get(key)
        if session is None:
            # 尝试从磁盘加载
            session = self._load(key) or Session(key=key)
            # 加入缓存（下次直接访问内存即可）
            self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        """
        保存会话到磁盘。

        文件格式：
        Line 1: {"_type": "metadata", "key": "...", ...}
        Line 2: {"role": "user", "content": "...", ...}  (消息 1)
        Line 3: {"role": "assistant", "content": "..."}   (消息 2)
        ...

        Args:
            session: 要保存的会话对象
        """
        path = self._session_path(session.key)
        lines = [json.dumps(self._metadata_line(session), ensure_ascii=False)]
        lines.extend(json.dumps(message, ensure_ascii=False) for message in session.messages)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        # 同时更新缓存
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        """
        让会话强制重载。

        从缓存中移除指定的会话，下次访问时会重新从磁盘加载。
        用于确保获取最新的会话数据（如外部修改后）。

        Args:
            key: 要使无效的会话标识符
        """
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        """
        列出所有会话（仅元数据，不加载完整消息）。

        用于会话选择界面或 API 端点。

        Returns:
            会话元数据列表，按更新时间倒序排列
        """
        sessions: list[dict[str, Any]] = []
        # 遍历 sessions 目录下所有 .jsonl 文件
        for path in self.sessions_dir.glob("*.jsonl"):
            metadata = self._read_metadata(path)
            if metadata:
                sessions.append(metadata)
        # 按更新时间倒序（最新的在前面）
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)

    def _load(self, key: str) -> Session | None:
        """
        从磁盘加载会话。

        解析流程：
        1. 读取 JSONL 文件
        2. 第一行解析为元数据
        3. 后续行解析为消息列表

        Args:
            key: 会话标识符

        Returns:
            解析后的 Session 对象，如果加载失败则返回 None
        """
        path = self._session_path(key)
        if not path.exists():
            return None

        try:
            metadata: dict[str, Any] = {}
            messages: list[dict[str, Any]] = []
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0

            # 逐行解析文件
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue  # 跳过空行
                data = json.loads(line)

                if data.get("_type") == "metadata":
                    # 第一行是元数据
                    metadata = data.get("metadata", {})
                    created_at = self._parse_datetime(data.get("created_at"))
                    updated_at = self._parse_datetime(data.get("updated_at"))
                    last_consolidated = data.get("last_consolidated", 0)
                    continue

                # 其他行是消息
                messages.append(data)

            now = datetime.now()
            return Session(
                key=key,
                messages=messages,
                created_at=created_at or now,
                updated_at=updated_at or created_at or now,
                metadata=metadata,
                last_consolidated=last_consolidated,
            )
        except Exception as exc:
            logger.warning("Failed to load session {}: {}", key, exc)
            return None

    def _session_path(self, key: str) -> Path:
        """
        计算会话文件的完整路径。

        文件名由 key 转换而来：
        - "cli:default" -> "cli_default.jsonl"
        - "user:123" -> "user_123.jsonl"

        Args:
            key: 会话标识符

        Returns:
            会话文件的绝对路径
        """
        safe_key = safe_filename(key)  # 转义非法字符
        return self.sessions_dir / f"{safe_key}.jsonl"

    @staticmethod
    def _metadata_line(session: Session) -> dict[str, Any]:
        """
        生成元数据行的字典。

        元数据包含会话的基本信息，但不包含大量消息内容，
        便于快速读取和使用。

        Args:
            session: 会话对象

        Returns:
            元数据字典（可被 json.dumps 序列化）
        """
        return {
            "_type": "metadata",          # 标记这是元数据行
            "key": session.key,           # 会话 ID
            "created_at": session.created_at.isoformat(),  # 创建时间
            "updated_at": session.updated_at.isoformat(),  # 更新时间
            "metadata": session.metadata,  # 自定义元数据
            "last_consolidated": session.last_consolidated,  # 归档索引
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

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any] | None:
        """
        仅读取会话的元数据（不加载全部消息）。

        用于 list_sessions() 方法，快速获取会话列表概览。

        Args:
            path: 会话文件路径

        Returns:
            会话元数据字典，如果读取失败则返回 None
        """
        try:
            # 只读取第一行（元数据行）
            first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception:
            return None

        if not first_line:
            return None

        try:
            data = json.loads(first_line)
        except json.JSONDecodeError:
            return None

        # 验证是否为元数据行
        if data.get("_type") != "metadata":
            return None

        return {
            "key": data.get("key") or path.stem,  # 会话 ID
            "created_at": data.get("created_at"),  # 创建时间
            "updated_at": data.get("updated_at"),  # 更新时间
            "path": str(path),                     # 文件路径
        }