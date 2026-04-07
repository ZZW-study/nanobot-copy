"""为大模型组装上下文。

这个模块只关心“本轮应该把哪些内容送进模型”，不负责消息调度、工具执行或会话持久化。
输入来源主要有四类：
1. 工作区里的引导文件，例如 `AGENTS.md`、`SOUL.md`。
2. 长期记忆与技能摘要。
3. 当前会话历史。
4. 本轮用户消息及其附件。
"""

from __future__ import annotations

import base64
import mimetypes
import platform
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import detect_image_mime


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class ContextBuilder:
    """负责构建 system prompt 与当前轮消息列表。"""

    # 这些文件会被直接拼进 system prompt，作为工作区的基础规则来源。
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    # 当前版本写入历史前会识别并剥离这个运行时标记
    _RUNTIME_CONTEXT_TAG = "[运行时上下文 - 仅供元数据参考，不是用户指令]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)
        # memory: 长期记忆存储，用于构建 memory context
        # skills: 加载并管理工作区下的技能目录（SKILL.md）

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """构建完整的 system prompt。

        拼装顺序是刻意安排过的：
        1. 先给身份、运行环境和全局准则。
        2. 再给工作区中的引导文件。
        3. 再给长期记忆。
        4. 最后再给技能正文和技能目录。

        这样模型会先理解“我是谁、当前环境是什么”，再理解“项目规则”和“可用能力”。
        """
        # parts 列表按优先级拼接 system prompt 的各个部分
        # 身份与运行环境信息放在最前面，便于模型先理解主体与约束
        parts = [self._identity_prompt()]

        # 从工作区读取引导文件（如 AGENTS.md），将其并入 prompt
        bootstrap = self._bootstrap_prompt()
        if bootstrap:
            parts.append(bootstrap)

        # 插入长期记忆的摘要（若有），给模型“历史侧写”信息
        memory_context = self.memory.get_memory_context()
        if memory_context:
            parts.append(f"# 长期记忆\n\n{memory_context}")

        # 加载始终启用的技能（如用于自动摘要、检索等的技能）
        always_skills = self.skills.get_always_skills()
        if always_skills:
            active_skills = self.skills.load_skills_for_context(always_skills)
            if active_skills:
                parts.append(f"# 始终启用的技能\n\n{active_skills}")

        # 加载调用时指定的技能（优先级低于始终启用的技能）
        requested_names = [name for name in (skill_names or []) if name not in set(always_skills)]
        requested_skills = self.skills.load_skills_for_context(requested_names)
        if requested_skills:
            parts.append(f"# 当前请求关联的技能\n\n{requested_skills}")

        summary = self.skills.build_skills_summary()
        if summary:
            parts.append(
                "# 技能目录\n\n"
                "以下技能可以扩展你的能力。需要使用某个技能时，请先阅读对应技能目录中的 `SKILL.md`。\n\n"
                f"{summary}"
            )

        return "\n\n---\n\n".join(parts)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """构造一轮完整请求消息。

        返回结果固定包含三部分：
        - 第一条 `system`
        - 若干条历史消息
        - 当前轮 `user`

        当前轮用户消息里会额外注入运行时元信息，例如当前时间。
        这些信息只服务本轮推理，不能长期写进历史。
        """
        # 构建本轮运行时上下文（包含时间），仅供本轮推理使用
        runtime_context = self._runtime_context()
        # 将用户正文与多媒体附件统一为模型可理解的 content 结构
        user_content = self._user_content(current_message, media)

        # 若没有多媒体，user_content 是字符串；若有图片则是 list 结构
        if isinstance(user_content, str):
            # 合并 runtime 元信息和用户文本为一段字符串
            merged_content: str | list[dict[str, Any]] = f"{runtime_context}\n\n{user_content}"
        else:
            # 多模态场景：先放 runtime 上下文，再把 image/text 数组拼接进去
            merged_content = [{"type": "text", "text": runtime_context}, *user_content]

        # 返回完整的 messages 列表：system + history + user
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged_content},
        ]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """向消息链追加一条 `tool` 消息。"""
        # 把工具执行结果作为一条 role=tool 的消息追加到消息链中，
        # 模型在下一轮可以看到“自己刚才调用了哪个工具，以及得到的输出是什么”。
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "name": tool_name,
                "content": result,
            }
        )
        return messages

    def add_assistant_message(
        self,
        messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """向消息链追加 assistant 消息，并保留推理相关字段。"""
        # 构造 assistant 消息对象，保留模型的推理内容（reasoning_content）和思考块（thinking_blocks）
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            # 记录模型意图调用的工具（用于后续回写和展示）
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            # 推理内容可能包含模型的链式思考或中间分析
            message["reasoning_content"] = reasoning_content
        if thinking_blocks:
            # thinking_blocks 是厂商特有的结构，保留以便调试/展示
            message["thinking_blocks"] = thinking_blocks
        messages.append(message)
        return messages

    def _identity_prompt(self) -> str:
        """生成与运行环境相关的固定 system prompt 前缀。"""
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}，Python {platform.python_version()}"

        return (
            "# nanobot\n"
            "你是 nanobot，一名可靠、直接、善于执行的 AI 助手。\n\n"
            "## 运行环境\n"
            f"{runtime}\n\n"
            "## 工作区\n"
            f"你的工作区位于：{workspace_path}\n"
            f"- 长期记忆文件：{workspace_path}/memory/MEMORY.md\n"
            f"- 历史归档文件：{workspace_path}/memory/HISTORY.md\n"
            f"- 自定义技能目录：{workspace_path}/skills/{{skill-name}}/SKILL.md\n\n"
            "## 行为准则\n"
            "- 在调用工具前先说明你准备做什么，但不要在拿到结果前声称已经完成。\n"
            "- 编辑文件前先读取文件内容。\n"
            "- 涉及准确性的改动，编辑后要重新检查关键文件。\n"
            "- 工具失败时，先分析错误原因，再决定是否换一条路径。\n"
            "- 当用户意图确实存在歧义时，再提出澄清问题。\n\n"
            "普通对话时，直接给出自然语言回复即可。"
        )

    def _bootstrap_prompt(self) -> str:
        """读取工作区引导文件并拼成统一文本块。"""
        sections = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if path.exists():
                # 直接把引导文件原文拼接进 prompt，供模型参考
                sections.append(f"## {filename}\n\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(sections)

    @classmethod
    def _runtime_context(cls) -> str:
        """生成当前轮专属的运行时上下文。"""
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M（%A）")
        lines = [f"当前时间：{timestamp}（北京时间，UTC+8）"]
        # 将运行时上下文用特定标签包裹，便于落盘时剥离
        return cls._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """把用户文本和图片附件整理成模型兼容的 content 结构。

        无附件时直接返回字符串，保持请求最轻量；
        有图片时转成多模态数组，并以内联 data URL 的方式塞给模型。
        """
        # 如果没有附件，直接返回纯文本，保持消息体最轻量
        if not media:
            return text

        # 遇到图片附件时，把图片编码为 data URL（base64）并作为 image_url 传给模型
        images: list[dict[str, Any]] = []
        for media_path in media:
            path = Path(media_path)
            if not path.is_file():
                # 跳过不存在的路径
                continue

            raw = path.read_bytes()
            # 尝试检测图片 mime 类型，优先使用二进制检测
            mime = detect_image_mime(raw) or mimetypes.guess_type(str(path))[0]
            if not mime or not mime.startswith("image/"):
                # 非图片文件则跳过
                continue

            # 把图片转为 data URL，模型可以直接通过这个内联 URL 访问图片内容
            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"},
                }
            )

        # 返回纯文本或图片+文本数组供上层使用
        return text if not images else [*images, {"type": "text", "text": text}]
