# 核心作用：上下文构建器 → 专门为AI大模型组装【系统提示词+对话历史+用户消息】的工具类
# 简单说：AI要回答问题，必须先看这个类组装好的"完整说明书+对话上下文"

import base64        # 图片转base64编码（让AI能识别图片）
import mimetypes     # 识别文件类型
import platform      # 获取系统信息（Windows/Mac/Linux）
import time
from datetime import datetime  # 获取当前时间
from pathlib import Path       # 路径处理
from typing import Any

# 导入项目模块：记忆存储、技能加载、工具函数
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader
from nanobot.utils.helpers import detect_image_mime


class ContextBuilder:
    """构建AI所需的完整上下文（系统提示词 + 对话消息）"""

    # 常量：工作区中需要自动加载的引导文件（AI的核心规则文档）
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    # 常量：运行时上下文标签（标记这是元数据，不是AI指令）
    _RUNTIME_CONTEXT_TAG = "[Runtime Context — metadata only, not instructions]"

    def __init__(self, workspace: Path):
        """
        初始化上下文构建器
        :param workspace: 工作区路径（AI操作文件、存储记忆的根目录）
        """
        self.workspace = workspace  # 绑定工作区路径
        self.memory = MemoryStore(workspace)  # 初始化记忆管理器（读取长期记忆）
        self.skills = SkillsLoader(workspace) # 初始化技能加载器（读取AI技能）

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        【核心函数】构建AI的系统提示词（AI的身份、规则、能力、记忆）
        :param skill_names: 可选技能名称列表
        :return: 完整的系统提示字符串
        """
        # 1. 先加入AI的核心身份信息（我是谁、运行环境、规则）
        parts = [self._get_identity()]

        # 2. 加载工作区中的引导文件（AGENTS.md等核心规则）
        bootstrap = self._load_bootstrap_files()
        if bootstrap:
            parts.append(bootstrap)

        # 3. 加入AI的长期记忆（之前总结的重要信息）
        memory = self.memory.get_memory_context()
        if memory:
            parts.append(f"# Memory\n\n{memory}")

        # 4. 加入常驻技能（AI默认必须掌握的技能）
        always_skills = self.skills.get_always_skills()
        if always_skills:
            always_content = self.skills.load_skills_for_context(always_skills)
            if always_content:
                parts.append(f"# Active Skills\n\n{always_content}")

        # 5. 加入技能总结（告诉AI有哪些扩展技能可用）
        skills_summary = self.skills.build_skills_summary()
        if skills_summary:
            parts.append(f"""# Skills
The following skills extend your capabilities. To use a skill, read its SKILL.md file using the read_file tool.
Skills with available="false" need dependencies installed first - you can try installing them with apt/brew.

{skills_summary}""")

        # 用分隔符拼接所有部分，返回完整系统提示词
        return "\n\n---\n\n".join(parts)

    def _get_identity(self) -> str:
        """
        【内部函数】生成AI的核心身份信息（AI的"自我介绍+运行环境+行为准则"）
        :return: 身份说明字符串
        """
        # 获取工作区绝对路径
        workspace_path = str(self.workspace.expanduser().resolve())
        # 获取系统信息（Windows/Mac/Linux + 架构 + Python版本）
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        # 返回AI的身份、运行环境、工作区、行为准则
        return f"""# nanobot 🐈
You are nanobot, a helpful AI assistant.

## Runtime
{runtime}

## Workspace
Your workspace is at: {workspace_path}
- Long-term memory: {workspace_path}/memory/MEMORY.md (write important facts here)
- History log: {workspace_path}/memory/HISTORY.md (grep-searchable). Each entry starts with [YYYY-MM-DD HH:MM].
- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md

## nanobot Guidelines
- State intent before tool calls, but NEVER predict or claim results before receiving them.
- Before modifying a file, read it first. Do not assume files or directories exist.
- After writing or editing a file, re-read it if accuracy matters.
- If a tool call fails, analyze the error before retrying with a different approach.
- Ask for clarification when the request is ambiguous.

Reply directly with text for conversations."""

    @staticmethod
    def _build_runtime_context(channel: str | None, chat_id: str | None) -> str:
        """
        【内部函数】构建运行时元数据（当前时间、消息渠道、聊天ID）
        :param channel: 消息渠道（WhatsApp/CLI/微信）
        :param chat_id: 聊天ID（区分不同用户）
        :return: 运行时上下文字符串
        """
        # 获取当前时间（格式化：年-月-日 时:分 (星期)）
        now = datetime.now().strftime("%Y-%m-%d %H:%M (%A)")
        # 获取时区
        tz = time.strftime("%Z") or "UTC"
        # 组装元数据行
        lines = [f"Current Time: {now} ({tz})"]
        if channel and chat_id:
            lines += [f"Channel: {channel}", f"Chat ID: {chat_id}"]
        # 返回带标签的运行时上下文
        return ContextBuilder._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _load_bootstrap_files(self) -> str:
        """
        【内部函数】自动加载工作区中的引导文件（AGENTS.md/SOUL.md等）
        这些文件是自定义AI性格、规则、能力的核心文档
        :return: 拼接后的引导文件内容
        """
        parts = []

        # 遍历所有引导文件，存在则读取内容
        for filename in self.BOOTSTRAP_FILES:
            file_path = self.workspace / filename
            if file_path.exists():
                content = file_path.read_text(encoding="utf-8")
                parts.append(f"## {filename}\n\n{content}")

        # 拼接所有内容，无文件则返回空
        return "\n\n".join(parts) if parts else ""

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        【终极核心函数】构建AI大模型需要的【完整消息列表】
        格式：[系统提示, 历史消息1, 历史消息2, ..., 当前用户消息]
        :param history: 对话历史列表
        :param current_message: 用户当前发送的文本
        :param skill_names: 技能列表
        :param media: 图片/媒体路径列表
        :param channel: 消息渠道
        :param chat_id: 聊天ID
        :return: 完整的消息列表（直接传给LLM）
        """
        # 1. 生成运行时上下文（时间、渠道）
        runtime_ctx = self._build_runtime_context(channel, chat_id)
        # 2. 处理用户消息（文本+图片）
        user_content = self._build_user_content(current_message, media)

        # 3. 合并「运行时上下文」和「用户消息」（避免大模型报错：连续相同角色消息）
        if isinstance(user_content, str):
            # 纯文本：直接拼接
            merged = f"{runtime_ctx}\n\n{user_content}"
        else:
            # 图片+文本：分开组装
            merged = [{"type": "text", "text": runtime_ctx}] + user_content

        # 4. 返回最终格式：系统提示 + 对话历史 + 当前用户消息
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},
            *history,
            {"role": "user", "content": merged},
        ]

    def _build_user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """
        【内部函数】处理用户消息：支持【纯文本】和【文本+图片】
        图片会转为base64编码，让多模态AI能识别
        :param text: 用户文本消息
        :param media: 图片路径列表
        :return: 处理后的用户消息（字符串 或 多模态数组）
        """
        # 无图片：直接返回文本
        if not media:
            return text

        images = []
        # 遍历所有图片路径
        for path in media:
            p = Path(path)
            if not p.is_file():
                continue  # 文件不存在则跳过
            # 读取图片二进制数据
            raw = p.read_bytes()
            # 识别真实图片格式（优先从二进制判断，而非后缀名）
            mime = detect_image_mime(raw) or mimetypes.guess_type(path)[0]
            if not mime or not mime.startswith("image/"):
                continue  # 非图片则跳过
            # 图片转base64编码
            b64 = base64.b64encode(raw).decode()
            # 组装成多模态AI支持的图片格式
            images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

        # 无有效图片：返回纯文本
        if not images:
            return text
        # 有图片：返回 [图片1, 图片2, 文本] 格式
        return images + [{"type": "text", "text": text}]

    def add_tool_result(
        self, messages: list[dict[str, Any]],
        tool_call_id: str, tool_name: str, result: str,
    ) -> list[dict[str, Any]]:
        """
        【工具函数】把【工具执行结果】加入消息列表（让AI看到工具执行的结果）
        例：AI调用读文件工具后，把文件内容加入上下文
        """
        messages.append({
            "role": "tool",          # 角色：工具
            "tool_call_id": tool_call_id, # 工具调用ID
            "name": tool_name,       # 工具名称
            "content": result        # 工具执行结果
        })
        return messages

    def add_assistant_message(
        self, messages: list[dict[str, Any]],
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
        reasoning_content: str | None = None,
        thinking_blocks: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """
        【工具函数】把【AI的回复/工具调用】加入消息列表
        支持：纯文本回复、工具调用、AI思考过程
        """
        # 构建助手消息
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        # 如果有工具调用，加入字段
        if tool_calls:
            msg["tool_calls"] = tool_calls
        # 如果有思考过程，加入字段
        if reasoning_content is not None:
            msg["reasoning_content"] = reasoning_content
        if thinking_blocks:
            msg["thinking_blocks"] = thinking_blocks
            
        # 加入消息列表并返回
        messages.append(msg)
        return messages





