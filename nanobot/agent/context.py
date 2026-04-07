"""为大模型组装上下文。

这个模块只关心"本轮应该把哪些内容送进模型"，不负责消息调度、工具执行或会话持久化。
输入来源主要有四类：
1. 工作区里的引导文件，例如 `AGENTS.md`、`SOUL.md`。
2. 长期记忆与技能摘要。
3. 当前会话历史。
4. 本轮用户消息及其附件。

核心类：
    ContextBuilder: 负责构建 system prompt 与当前轮消息列表
"""

from __future__ import annotations  # 启用未来版本的类型注解特性

import base64  # 用于 base64 编码（图片转 data URL）
import mimetypes  # 用于根据文件扩展名猜测 MIME 类型
import platform  # 用于获取运行平台信息（系统、架构、Python 版本）
from datetime import datetime  # 用于获取当前时间
from pathlib import Path  # 用于路径操作
from typing import Any  # 用于类型注解
from zoneinfo import ZoneInfo  # 用于时区处理（北京时间）

from nanobot.agent.memory import MemoryStore  # 长期记忆存储模块
from nanobot.agent.skills import SkillsLoader  # 技能加载器模块
from nanobot.utils.helpers import detect_image_mime  # 图片 MIME 类型检测工具


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class ContextBuilder:
    """
    负责构建 system prompt 与当前轮消息列表。

    这个类的职责很单一：把各种输入源（引导文件、长期记忆、技能、历史对话、当前消息）
    组装成大模型可以理解的 messages 格式。它不负责：
    - 消息调度（由 AgentLoop 负责）
    - 工具执行（由 ToolRegistry 负责）
    - 会话持久化（由 SessionManager 负责）
    """

    # 引导文件列表：这些文件会被直接拼进 system prompt，作为工作区的基础规则来源
    # 按顺序读取并拼接，形成项目特定的行为准则
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]

    # 运行时上下文标签：当前版本写入历史前会识别并剥离这个标记
    # 用于标记那些只对当前轮推理有意义的元信息（如当前时间）
    _RUNTIME_CONTEXT_TAG = "[运行时上下文 - 仅供元数据参考，不是用户指令]"

    def __init__(self, workspace: Path):
        """
        初始化 ContextBuilder。

        Args:
            workspace: 工作区根目录路径（Path 对象）
        """
        self.workspace = workspace  # 工作区根目录
        self.memory = MemoryStore(workspace)  # 长期记忆存储，用于构建 memory context
        self.skills = SkillsLoader(workspace)  # 技能加载器，用于读取和管理技能目录

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        """
        构建完整的 system prompt。

        拼装顺序是刻意安排过的：
        1. 先给身份、运行环境和全局准则（让模型理解”我是谁”）
        2. 再给工作区中的引导文件（让模型理解”项目规则”）
        3. 再给长期记忆（让模型理解”历史侧写”）
        4. 最后再给技能正文和技能目录（让模型理解”可用能力”）

        这样模型会先理解”我是谁、当前环境是什么”，再理解”项目规则”和”可用能力”。

        Args:
            skill_names: 可选的技能名称列表，用于加载特定技能

        Returns:
            完整的 system prompt 字符串
        """
        # parts 列表按优先级拼接 system prompt 的各个部分
        # 身份与运行环境信息放在最前面，便于模型先理解主体与约束
        parts = [self._identity_prompt()]

        # 从工作区读取引导文件（如 AGENTS.md），将其并入 prompt
        # 这些文件包含项目特定的规则和准则
        bootstrap = self._bootstrap_prompt()
        if bootstrap:
            parts.append(bootstrap)

        # 插入长期记忆的摘要（若有），给模型”历史侧写”信息
        # 让模型了解之前的对话历史和重要信息
        memory_context = self.memory.get_memory_context()
        if memory_context:
            parts.append(f"# 长期记忆\n\n{memory_context}")

        # 加载始终启用的技能（如用于自动摘要、检索等的技能）
        # 这些技能无条件生效，提供基础能力扩展
        always_skills = self.skills.get_always_skills()
        if always_skills:
            active_skills = self.skills.load_skills_for_context(always_skills)
            if active_skills:
                parts.append(f"# 始终启用的技能\n\n{active_skills}")

        # 加载调用时指定的技能（优先级低于始终启用的技能）
        # 过滤掉已经在 always_skills 中的技能，避免重复加载
        requested_names = [name for name in (skill_names or []) if name not in set(always_skills)]
        requested_skills = self.skills.load_skills_for_context(requested_names)
        if requested_skills:
            parts.append(f"# 当前请求关联的技能\n\n{requested_skills}")

        # 构建技能目录摘要
        summary = self.skills.build_skills_summary()
        if summary:
            parts.append(
                "# 技能目录\n\n"
                "以下技能可以扩展你的能力。需要使用某个技能时，请先阅读对应技能目录中的 `SKILL.md`。\n\n"
                f"{summary}"
            )

        # 用 “---” 分隔符连接所有部分，形成完整的 system prompt
        # “---” 是 Markdown 中常用的分隔符，视觉上清晰区分不同内容块
        return "\n\n---\n\n".join(parts)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        构造一轮完整请求消息。

        这是组装消息链的核心方法，返回结果固定包含三部分：
        1. 第一条 `system` 消息（包含身份、规则、记忆、技能等）
        2. 若干条历史消息（之前的对话内容）
        3. 当前轮 `user` 消息（用户输入 + 运行时元信息 + 可选媒体附件）

        当前轮用户消息里会额外注入运行时元信息（如当前时间），
        这些信息只服务本轮推理，不能长期写进历史（会在保存时被剥离）。

        Args:
            history: 历史消息列表（每条消息包含 role 和 content）
            current_message: 当前用户消息文本
            skill_names: 可选的技能名称列表（用于加载特定技能）
            media: 可选的媒体文件路径列表（如图片）

        Returns:
            完整的 messages 列表，可直接传给大模型 API
        """
        # 构建本轮运行时上下文（包含时间），仅供本轮推理使用
        # 这个上下文会在保存历史时被剥离，避免污染长期记忆
        runtime_context = self._runtime_context()

        # 将用户正文与多媒体附件统一为模型可理解的 content 结构
        # 无附件时返回字符串，有图片时返回多模态数组
        user_content = self._user_content(current_message, media)

        # ========== 合并运行时上下文和用户内容 ==========
        # 若没有多媒体，user_content 是字符串，直接拼接
        if isinstance(user_content, str):
            # 合并 runtime 元信息和用户文本为一段字符串
            merged_content: str | list[dict[str, Any]] = f"{runtime_context}\n\n{user_content}"
        else:
            # 多模态场景：先放 runtime 上下文，再把 image/text 数组拼接进去
            merged_content = [{"type": "text", "text": runtime_context}, *user_content]

        # 返回完整的 messages 列表：system + history + user
        # 使用 *history 解构语法将历史消息插入中间
        return [
            {"role": "system", "content": self.build_system_prompt(skill_names)},  # system 消息
            *history,  # 历史消息（user/assistant/tool 的对话记录）
            {"role": "user", "content": merged_content},  # 当前用户消息
        ]

    def add_tool_result(
        self,
        messages: list[dict[str, Any]],
        tool_call_id: str,
        tool_name: str,
        result: str,
    ) -> list[dict[str, Any]]:
        """
        向消息链追加一条 `tool` 消息。

        当工具执行完毕后，需要将结果回填给模型，让模型理解工具执行的结果。
        这条消息会被追加到消息链中，模型在下一轮请求时可以看到：
        - 自己之前调用了哪个工具（通过 tool_call_id 关联）
        - 工具返回了什么结果（通过 content 字段）

        Args:
            messages: 当前消息链列表
            tool_call_id: 工具调用的唯一标识符（与 assistant 消息中的 tool_calls.id 对应）
            tool_name: 工具名称（如 “web_search”、”read_file”）
            result: 工具执行结果（字符串或 JSON）

        Returns:
            更新后的消息链（原列表被修改，返回引用便于链式调用）
        """
        # 构造 tool 消息对象
        messages.append(
            {
                "role": "tool",           # 固定为 “tool” 角色
                "tool_call_id": tool_call_id,  # 关联到 assistant 的 tool_call
                "name": tool_name,        # 工具名称（用于日志和调试）
                "content": result,        # 工具执行结果
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
        """
        向消息链追加 assistant 消息，并保留推理相关字段。

        当模型返回响应时（无论是否包含工具调用），都需要将响应写入消息链，
        以便下一轮请求时模型能看到自己的回复历史。此方法同时保留：
        - 基础回复内容（content）
        - 工具调用意图（tool_calls）
        - 推理内容（reasoning_content，部分模型支持）
        - 思考块（thinking_blocks，原始思考内容）

        Args:
            messages: 当前消息链列表
            content: 模型回复文本（可能为 None，如只调用工具不说话）
            tool_calls: 可选的工具调用列表（每项包含 id、type、function）
            reasoning_content: 可选的推理内容（部分模型支持的中间思考）
            thinking_blocks: 可选的思考块列表（厂商特有的结构）

        Returns:
            更新后的消息链（原列表被修改，返回引用便于链式调用）
        """
        # 构造 assistant 消息对象，保留模型的推理内容和思考块
        message: dict[str, Any] = {"role": "assistant", "content": content}

        # 如果有工具调用，记录模型意图调用的工具
        if tool_calls:
            message["tool_calls"] = tool_calls  # 用于后续执行工具并关联 tool 结果

        # 如果有推理内容，保留以便调试和展示
        # 推理内容可能包含模型的链式思考或中间分析过程
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content

        # 如果有思考块，保留以便调试和展示
        # thinking_blocks 是厂商特有的结构，保留原始内容
        if thinking_blocks:
            message["thinking_blocks"] = thinking_blocks

        # 将消息追加到消息链
        messages.append(message)
        return messages

    def _identity_prompt(self) -> str:
        """
        生成与运行环境相关的固定 system prompt 前缀。

        这是 system prompt 的第一部分，包含：
        1. Agent 的身份定义（"你是 nanobot..."）
        2. 运行环境信息（操作系统、架构、Python 版本）
        3. 工作区路径信息（长期记忆、历史归档、技能目录的位置）
        4. 行为准则（调用工具前先说明、先读后改、检查结果等）

        这部分内容是固定的，不依赖外部文件或配置。

        Returns:
            身份和运行环境的 system prompt 文本块
        """
        # 获取工作区的绝对路径（展开 ~ 用户目录符号，转换为绝对路径）
        workspace_path = str(self.workspace.expanduser().resolve())

        # 获取操作系统名称（Darwin = macOS，Linux = Linux，Windows = Windows）
        system = platform.system()

        # 构建运行环境描述字符串
        # macOS 显示为 "macOS"，其他系统显示原名称
        # 包含 CPU 架构（如 arm64、x86_64）和 Python 版本
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}，Python {platform.python_version()}"

        # 返回完整的身份 prompt（包含身份定义、运行环境、工作区和行为准则）
        return (
            "# nanobot\n"
            "你是 nanobot，一名可靠、直接、善于执行的 AI 助手。\n\n"
            "## 运行环境\n"
            f"{runtime}\n\n"  # 插入运行环境信息
            "## 工作区\n"
            f"你的工作区位于：{workspace_path}\n"
            f"- 长期记忆文件：{workspace_path}/memory/MEMORY.md\n"  # 长期记忆存储位置
            f"- 历史归档文件：{workspace_path}/memory/HISTORY.md\n"  # 历史归档存储位置
            f"- 自定义技能目录：{workspace_path}/skills/{{skill-name}}/SKILL.md\n\n"  # 技能目录位置模板
            "## 行为准则\n"
            "- 在调用工具前先说明你准备做什么，但不要在拿到结果前声称已经完成。\n"  # 先声明再执行
            "- 编辑文件前先读取文件内容。\n"  # 先读后改，避免盲目修改
            "- 涉及准确性的改动，编辑后要重新检查关键文件。\n"  # 改后复查，确保正确
            "- 工具失败时，先分析错误原因，再决定是否换一条路径。\n"  # 失败时分析而非重试
            "- 当用户意图确实存在歧义时，再提出澄清问题。\n\n"  # 只有真正歧义才提问
            "普通对话时，直接给出自然语言回复即可。"  # 简单对话直接回复
        )

    def _bootstrap_prompt(self) -> str:
        """
        读取工作区引导文件并拼成统一文本块。

        引导文件（如 AGENTS.md、SOUL.md）包含项目特定的规则和准则，
        这些文件会被直接拼接进 system prompt，让模型理解项目的特殊要求。

        文件列表由 BOOTSTRAP_FILES 类常量定义：
        - AGENTS.md: Agent 行为准则
        - SOUL.md: Agent 人格/风格定义
        - USER.md: 用户偏好设置
        - TOOLS.md: 工具使用指南
        - IDENTITY.md: 身份相关信息

        Returns:
            拼接后的引导文件内容，如果没有任何文件存在则返回空字符串
        """
        sections = []  # 存储每个文件的内容片段
        # 遍历引导文件列表，尝试读取每个文件
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename  # 构建文件完整路径
            if path.exists():
                # 直接把引导文件原文拼接进 prompt，供模型参考
                # 使用 UTF-8 编码读取，确保中文内容正确解析
                sections.append(f"## {filename}\n\n{path.read_text(encoding='utf-8')}")
        # 用双换行连接所有片段，形成统一的文本块
        return "\n\n".join(sections)

    @classmethod
    def _runtime_context(cls) -> str:
        """
        生成当前轮专属的运行时上下文。

        运行时上下文包含只在当前轮推理中有意义的信息（如当前时间），
        这些信息会注入到用户消息中，但在保存历史时会被剥离，
        防止污染长期记忆。

        目前包含的信息：
        - 当前时间（北京时间，包含日期、时间、星期）

        Returns:
            带有运行时标签的上下文文本
        """
        # 获取当前时间并格式化为易读的格式
        # 格式：2024-01-15 14:30（星期一）
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M（%A）")

        # 构建运行时上下文内容行
        lines = [f"当前时间：{timestamp}（北京时间，UTC+8）"]

        # 将运行时上下文用特定标签包裹，便于落盘时剥离
        # 标签格式："[运行时上下文 - 仅供元数据参考，不是用户指令]"
        return cls._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        """
        把用户文本和图片附件整理成模型兼容的 content 结构。

        模型的 content 字段有两种格式：
        1. 纯文本：字符串类型（最简单，无附件时使用）
        2. 多模态：列表类型，包含多个 text/image_url 块

        此方法负责将用户输入统一转换为这两种格式之一。

        Args:
            text: 用户文本消息
            media: 可选的媒体文件路径列表（如图片路径）

        Returns:
            如果无附件：返回纯文本字符串
            如果有图片：返回多模态数组（图片 + 文本块）
        """
        # 如果没有附件，直接返回纯文本，保持消息体最轻量
        if not media:
            return text

        # 遇到图片附件时，把图片编码为 data URL（base64）并作为 image_url 传给模型
        images: list[dict[str, Any]] = []

        # 遍历所有媒体文件路径
        for media_path in media:
            path = Path(media_path)  # 转换为 Path 对象
            if not path.is_file():
                # 跳过不存在的路径（可能是无效输入）
                continue

            # 读取文件的原始二进制内容
            raw = path.read_bytes()

            # 尝试检测图片 MIME 类型
            # 优先使用二进制检测（通过文件头魔数判断），更可靠
            # 如果二进制检测失败，则尝试通过文件扩展名猜测
            mime = detect_image_mime(raw) or mimetypes.guess_type(str(path))[0]
            if not mime or not mime.startswith("image/"):
                # 非图片文件则跳过（只处理图片类型）
                continue

            # 把图片转为 data URL 格式
            # 格式：data:image/png;base64,iVBORw0KGgo...
            # 模型可以直接通过这个内联 URL 访问图片内容
            images.append(
                {
                    "type": "image_url",  # 固定为 image_url 类型
                    "image_url": {
                        "url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"
                    },
                }
            )

        # 返回纯文本或图片 + 文本数组供上层使用
        # 如果没有成功加载任何图片，返回纯文本
        # 如果有图片，返回数组（先放图片，再放文本，顺序便于模型先看图再回答）
        return text if not images else [*images, {"type": "text", "text": text}]
