"""长期记忆与历史归档。

这个模块处理的是"会话太长之后，如何把旧消息压缩成长期可用的信息"。
目标不是做复杂知识库，而是维护两份简单但稳定的文件：
1. `MEMORY.md`：持续演进的长期记忆（可被模型读取和更新）
2. `HISTORY.md`：只追加不回写的历史摘要（用于人工查阅和调试）

核心类：
    MemoryStore: 封装 memory/ 目录中的读写与归档逻辑
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from nanobot.utils.helpers import ensure_dir

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import Session


_SAVE_MEMORY_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "save_memory",
            "description": "保存一条压缩后的历史摘要，并返回更新后的长期记忆内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": (
                            "用 2 到 5 句话总结本次归档内容，并以 [YYYY-MM-DD HH:MM] 时间戳开头，"
                            "方便后续用 grep 或关键字检索。"
                        ),
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "更新后的完整 MEMORY.md 内容。",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    """
    封装 `memory/` 目录中的读写与归档逻辑。

    这个类负责管理两个关键文件：
    - MEMORY.md：长期记忆文件，会被注入到 system prompt 中供模型参考
    - HISTORY.md：历史归档文件，只追加不修改，用于人工查阅对话历史

    主要功能：
    1. 读写 MEMORY.md 文件
    2. 追加 HISTORY.md 文件
    3. 调用大模型压缩会话历史并更新长期记忆
    """

    def __init__(self, workspace: Path):
        """
        初始化 MemoryStore 实例。

        Args:
            workspace: 工作区根目录路径
        """
        # 确保 memory 目录存在，如果不存在则创建
        self.memory_dir = ensure_dir(workspace / "memory")
        # 长期记忆文件路径
        self.memory_file = self.memory_dir / "MEMORY.md"
        # 历史归档文件路径
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        """
        读取长期记忆全文；文件不存在时返回空字符串。

        Returns:
            MEMORY.md 文件的内容，如果文件不存在则返回空字符串
        """
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    def write_long_term(self, content: str) -> None:
        """
        覆盖写入 `MEMORY.md`。

        Args:
            content: 要写入的长期记忆内容
        """
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        """
        向 `HISTORY.md` 追加一条阶段性摘要。

        Args:
            entry: 要追加的历史摘要内容（会自动添加换行符）
        """
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(entry.strip() + "\n\n")

    def get_memory_context(self) -> str:
        """
        返回适合直接注入 prompt 的长期记忆文本。

        如果 MEMORY.md 有内容，则返回格式化的文本块；
        如果为空，则返回空字符串。

        Returns:
            格式化的长期记忆文本，或空字符串
        """
        memory = self.read_long_term()
        return f"## MEMORY.md\n{memory}" if memory else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        """
        把会话中的旧消息归档进长期记忆。

        这是长期记忆的核心方法，执行流程：
        1. 确定要归档的消息范围（_messages_to_archive）
        2. 构造归档提示词（_build_prompt）
        3. 调用大模型压缩历史并生成更新建议
        4. 处理模型返回的结果并更新文件
        5. 更新会话的 last_consolidated 标记

        Args:
            session: 当前会话对象
            provider: LLM 提供商实例（用于调用大模型）
            model: 使用的模型名称
            archive_all: 是否强制归档所有消息（用于 /new 命令）
            memory_window: 记忆窗口大小（决定保留多少最新消息）

        Returns:
            True 表示归档成功，False 表示失败
        """
        # 确定本次要归档的消息区间和需要保留的尾部消息数量
        messages, keep_count = self._messages_to_archive(session, archive_all, memory_window)
        if not messages:
            return True  # 没有消息需要归档，直接返回成功

        # 读取当前的长期记忆内容
        current_memory = self.read_long_term()
        # 构造归档提示词（包含当前记忆和待归档消息）
        prompt = self._build_prompt(current_memory, messages)

        try:
            # 调用大模型进行历史压缩
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "你负责压缩对话历史，且必须调用 save_memory 工具返回结构化结果。",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,  # 强制模型使用 save_memory 工具
                model=model,
            )
        except Exception:
            logger.exception("长期记忆归档失败")
            return False

        # 检查模型是否调用了 save_memory 工具
        if not response.has_tool_calls:
            logger.warning("长期记忆归档被跳过：模型没有调用 save_memory 工具")
            return False

        # 规范化工具参数（处理不同格式的返回值）
        args = self._normalize_tool_args(response.tool_calls[0].arguments)
        if args is None:
            logger.warning("长期记忆归档失败：模型返回的工具参数格式不正确")
            return False

        # 处理历史摘要（追加到 HISTORY.md）
        history_entry = self._coerce_text(args.get("history_entry"))
        if history_entry:
            self.append_history(history_entry)

        # 处理长期记忆更新（覆盖写入 MEMORY.md）
        memory_update = self._coerce_text(args.get("memory_update"))
        if memory_update is not None and memory_update != current_memory:
            self.write_long_term(memory_update)

        # 更新会话的归档标记
        # archive_all=True 时归档所有消息，last_consolidated 设为 0
        # 否则保留 keep_count 条最新消息，其余标记为已归档
        session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
        logger.info(
            "长期记忆归档完成：本次归档 {} 条消息，last_consolidated={}",
            len(messages),
            session.last_consolidated,
        )
        return True

    @staticmethod
    def _messages_to_archive(
        session: Session,
        archive_all: bool,
        memory_window: int,
    ) -> tuple[list[dict[str, Any]], int]:
        """
        确定本次要归档的消息区间，以及本轮需要保留多少尾部消息。

        默认策略是"保留最近一半窗口，归档更早的部分"，
        这样下一轮模型还能看到足够新的上下文，而老消息不会无限膨胀。

        Args:
            session: 当前会话对象
            archive_all: 是否归档所有消息
            memory_window: 记忆窗口大小

        Returns:
            (messages_to_archive, keep_count) 元组：
            - messages_to_archive: 要归档的消息列表
            - keep_count: 需要保留的尾部消息数量
        """
        if archive_all:
            # 强制归档所有消息，保留 0 条
            return list(session.messages), 0

        # 默认保留最近一半窗口的消息（至少保留 1 条）
        keep_count = max(1, memory_window // 2)
        if len(session.messages) <= keep_count:
            # 消息总数不超过保留数量，无需归档
            return [], keep_count

        # 计算归档范围：从上次归档位置到倒数 keep_count 条消息
        start = session.last_consolidated
        end = len(session.messages) - keep_count
        if end <= start:
            # 归档范围无效，无需归档
            return [], keep_count

        # 返回要归档的消息片段和保留数量
        return session.messages[start:end], keep_count

    def _build_prompt(self, current_memory: str, messages: list[dict[str, Any]]) -> str:
        """
        把长期记忆和待归档对话整理成提示词。

        构造的提示词包含：
        1. 当前 MEMORY.md 的内容
        2. 待归档的对话历史（格式化后的转录文本）

        Args:
            current_memory: 当前长期记忆内容
            messages: 待归档的消息列表

        Returns:
            完整的归档提示词字符串
        """
        # 格式化消息列表为转录文本
        transcript = "\n".join(self._format_messages(messages))
        return (
            "请整理下面这些旧对话，把需要长期保留的信息写入 MEMORY.md，"
            "并把本段历史压缩成一条可检索的摘要。\n\n"
            "## 当前 MEMORY.md\n"
            f"{current_memory or '(当前为空)'}\n\n"
            "## 待归档对话\n"
            f"{transcript}"
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[str]:
        """
        把消息列表格式化成适合归档模型阅读的转录文本。

        每条消息的格式：[timestamp] ROLE[tools_used]: content

        Args:
            messages: 消息列表

        Returns:
            格式化后的消息行列表
        """
        lines: list[str] = []
        for message in messages:
            content = message.get("content")
            if not content:
                continue  # 跳过空内容消息
            # 获取使用的工具列表（如果有）
            tools = message.get("tools_used") or []
            tool_suffix = f" [使用工具: {', '.join(tools)}]" if tools else ""
            # 截取时间戳的前 16 个字符（YYYY-MM-DD HH:MM）
            timestamp = str(message.get("timestamp", "?"))[:16]
            # 构造格式化行：[2024-01-15 14:30] USER [使用工具: web_search]: 用户消息内容
            lines.append(f"[{timestamp}] {message['role'].upper()}{tool_suffix}: {content}")
        return lines

    @staticmethod
    def _normalize_tool_args(arguments: Any) -> dict[str, Any] | None:
        """
        把模型返回的工具参数统一规整成字典。

        处理多种可能的返回格式：
        1. 字符串（JSON 格式）-> 解析为字典
        2. 列表 -> 取第一个字典元素
        3. 字典 -> 直接返回

        Args:
            arguments: 模型返回的工具参数（可能是字符串、列表或字典）

        Returns:
            规范化的参数字典，或 None（如果无法解析）
        """
        # 如果是字符串，尝试 JSON 解析
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        # 如果是列表，取第一个字典元素
        if isinstance(arguments, list):
            arguments = arguments[0] if arguments and isinstance(arguments[0], dict) else None

        # 确保返回字典类型
        return arguments if isinstance(arguments, dict) else None

    @staticmethod
    def _coerce_text(value: Any) -> str | None:
        """
        把工具结果字段规范成字符串，便于直接写文件。

        处理不同类型的值：
        - None -> None
        - 字符串 -> 直接返回
        - 其他类型 -> JSON 序列化

        Args:
            value: 工具返回的字段值

        Returns:
            规范化的字符串，或 None
        """
        if value is None:
            return None
        if isinstance(value, str):
            return value
        # 其他类型（如数字、布尔值、列表、字典）转换为 JSON 字符串
        return json.dumps(value, ensure_ascii=False)
