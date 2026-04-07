"""技能发现、读取与可用性判断。

这里不追求实现完整的包管理器，而是提供一套足够直接的技能装载逻辑：
1. 从内置目录和工作区目录发现技能。
2. 读取 `SKILL.md` 内容。
3. 解析少量 frontmatter 元数据。
4. 判断技能当前依赖是否满足。

核心概念：
- 技能（Skill）：以 SKILL.md 文件形式存在的功能模块
- 内置技能：位于项目内置 skills/ 目录下的技能
- 工作区技能：位于用户工作区 skills/ 目录下的技能（可覆盖内置技能）
- Frontmatter：SKILL.md 文件顶部的 YAML 格式元数据块

核心类：
    SkillsLoader: 统一管理工作区技能和内置技能的加载器
"""

from __future__ import annotations  # 启用未来版本的类型注解特性

import json  # 用于解析技能元数据中的 JSON 配置
import os  # 用于检查环境变量
import re  # 用于正则表达式匹配 frontmatter
import shutil  # 用于检查系统命令是否存在
from pathlib import Path  # 用于路径操作

# 内置技能目录路径：相对于当前文件的上级目录的 skills/ 子目录
# 例如：nanobot/agent/../skills/ = nanobot/skills/
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# Frontmatter 正则表达式：匹配 SKILL.md 文件顶部的 --- ... --- 块
# re.DOTALL 标志使 . 能匹配换行符，确保多行内容都能被捕获
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


class SkillsLoader:
    """
    统一管理工作区技能和内置技能。

    这个类负责：
    1. 发现技能（扫描内置目录和工作区目录）
    2. 读取技能内容（SKILL.md 文件）
    3. 解析元数据（frontmatter 中的配置）
    4. 检查依赖状态（命令行工具、环境变量等）
    5. 构建技能摘要（用于 system prompt）

    技能优先级：工作区技能 > 内置技能（同名时工作区覆盖内置）
    """

    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        """
        初始化 SkillsLoader 实例。

        Args:
            workspace: 用户工作区根目录路径
            builtin_skills_dir: 可选的内置技能目录路径（默认使用 BUILTIN_SKILLS_DIR）
        """
        # 工作区技能目录：workspace/skills/
        self.workspace_skills = workspace / "skills"
        # 内置技能目录：默认为项目内置目录，可被覆盖
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        """
        列出当前可发现的技能。

        默认会过滤掉依赖不满足的技能，因为这类技能虽然存在，
        但直接暴露给模型只会增加误用概率。

        Args:
            filter_unavailable: 是否过滤掉不可用的技能（依赖缺失）

        Returns:
            技能信息列表，每个技能包含 name、path、source 字段
        """
        # 获取所有发现的技能（包括内置和工作区）
        skills = list(self._discover_skills().values())
        if not filter_unavailable:
            return skills

        # 过滤掉依赖不满足的技能
        # _requirements_status 返回 (可用, 缺失依赖列表)
        return [skill for skill in skills if self._requirements_status(skill["name"])[0]]

    def load_skill(self, name: str) -> str | None:
        """
        按名称读取单个技能原文。

        按优先级查找技能文件：
        1. 工作区技能目录
        2. 内置技能目录

        Args:
            name: 技能名称（对应技能目录名）

        Returns:
            SKILL.md 文件的完整内容，如果找不到则返回 None
        """
        path = self._skill_path(name)
        return path.read_text(encoding="utf-8") if path else None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        """
        把多个技能拼成可直接注入 system prompt 的文本块。

        对于每个技能：
        1. 读取 SKILL.md 内容
        2. 移除 frontmatter（只保留正文）
        3. 添加标题前缀

        Args:
            skill_names: 要加载的技能名称列表

        Returns:
            拼接后的技能文本块，用 "---" 分隔符分隔
        """
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                # 移除 frontmatter，只保留真正要给模型看的正文
                parts.append(f"### 技能：{name}\n\n{self._strip_frontmatter(content)}")
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        """
        构建一份紧凑的技能目录摘要。

        这份摘要不是完整技能内容，而是给模型一个"目录索引"，
        让它知道有哪些技能、哪些可用、哪些因为依赖缺失暂时不可用。

        输出格式为 XML，包含每个技能的：
        - name: 技能名称
        - description: 技能描述
        - location: 文件路径
        - requires: 缺失的依赖（如果有）
        - available: 是否可用（true/false）

        Returns:
            XML 格式的技能目录摘要字符串
        """
        skills = self.list_skills(filter_unavailable=False)
        if not skills:
            return ""

        def escape_xml(value: str) -> str:
            """转义 XML 特殊字符，防止注入攻击。"""
            return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # 构建 XML 结构
        lines = ["<skills>"]
        for skill in skills:
            metadata = self.get_skill_metadata(skill["name"]) or {}
            # 检查技能依赖状态
            available, missing = self._requirements_status(skill["name"], metadata)

            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{escape_xml(skill['name'])}</name>")
            lines.append(f"    <description>{escape_xml(self._skill_description(skill['name'], metadata))}</description>")
            lines.append(f"    <location>{skill['path']}</location>")
            if missing:
                lines.append(f"    <requires>{escape_xml('，'.join(missing))}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def get_always_skills(self) -> list[str]:
        """
        返回被标记为始终注入上下文的技能名称。

        某些技能（如自动摘要、基础工具）需要始终可用，
        这些技能在 frontmatter 中标记了 always: true。

        Returns:
            始终启用的技能名称列表
        """
        always: list[str] = []
        # 只考虑可用的技能（依赖满足）
        for skill in self.list_skills(filter_unavailable=True):
            metadata = self.get_skill_metadata(skill["name"]) or {}
            skill_meta = self._skill_meta(metadata)
            # 检查是否标记为 always（支持两种格式）
            if skill_meta.get("always") or metadata.get("always"):
                always.append(skill["name"])
        return always

    def get_skill_metadata(self, name: str) -> dict[str, str] | None:
        """
        读取 `SKILL.md` 顶部 frontmatter 中的简单键值元数据。

        Frontmatter 格式示例：
        ---
        name: web_search
        description: 网络搜索工具
        requires:
          bins: [curl, jq]
        ---

        Args:
            name: 技能名称

        Returns:
            解析后的元数据字典，如果文件不存在或没有 frontmatter 则返回 None
        """
        content = self.load_skill(name)
        if not content:
            return None

        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None

        # 解析 frontmatter 中的键值对
        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        return metadata

    def _discover_skills(self) -> dict[str, dict[str, str]]:
        """
        扫描内置目录和工作区目录，收集技能清单。

        工作区和内置技能可能重名。这里有意让后扫描到的条目覆盖前者，
        这样用户可以在工作区自然覆盖内置技能版本。

        扫描顺序：先内置，再工作区（工作区覆盖内置）

        Returns:
            技能字典，键为技能名称，值为技能信息（name、path、source）
        """
        skills: dict[str, dict[str, str]] = {}
        # 扫描两个来源：builtin（内置）和 workspace（工作区）
        for source, root in (("builtin", self.builtin_skills), ("workspace", self.workspace_skills)):
            if not root or not root.exists():
                continue
            # 遍历技能目录下的所有子目录
            for skill_dir in root.iterdir():
                skill_file = skill_dir / "SKILL.md"
                # 只有包含 SKILL.md 文件的目录才被认为是技能
                if skill_dir.is_dir() and skill_file.exists():
                    skills[skill_dir.name] = {
                        "name": skill_dir.name,
                        "path": str(skill_file),
                        "source": source,
                    }
        return skills

    def _skill_path(self, name: str) -> Path | None:
        """
        查找技能文件路径，优先工作区，其次内置目录。

        Args:
            name: 技能名称

        Returns:
            SKILL.md 文件的完整路径，如果找不到则返回 None
        """
        # 按优先级查找：先工作区，再内置
        for root in (self.workspace_skills, self.builtin_skills):
            if not root:
                continue
            path = root / name / "SKILL.md"
            if path.exists():
                return path
        return None

    def _requirements_status(
        self,
        name: str,
        metadata: dict[str, str] | None = None,
    ) -> tuple[bool, list[str]]:
        """
        判断技能当前是否满足运行依赖。

        依赖类型：
        1. bins: 命令行工具（通过 shutil.which 检查）
        2. env: 环境变量（通过 os.environ 检查）

        Args:
            name: 技能名称
            metadata: 可选的元数据字典（避免重复读取）

        Returns:
            (available, missing) 元组：
            - available: 是否所有依赖都满足
            - missing: 缺失的依赖列表（错误信息字符串）
        """
        skill_meta = self._skill_meta(metadata or self.get_skill_metadata(name) or {})
        missing: list[str] = []

        # 检查命令行工具依赖
        requires = skill_meta.get("requires", {})
        for binary in requires.get("bins", []):
            if not shutil.which(binary):
                missing.append(f"缺少命令行工具：{binary}")

        # 检查环境变量依赖
        for env_name in requires.get("env", []):
            if not os.environ.get(env_name):
                missing.append(f"缺少环境变量：{env_name}")

        return not missing, missing

    def _skill_description(self, name: str, metadata: dict[str, str] | None = None) -> str:
        """
        优先取元数据里的描述，没有则退回技能名。

        Args:
            name: 技能名称
            metadata: 可选的元数据字典

        Returns:
            技能描述字符串
        """
        metadata = metadata or self.get_skill_metadata(name) or {}
        return metadata.get("description") or name

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        """
        移除 frontmatter，只保留真正要给模型看的正文。

        Args:
            content: 完整的 SKILL.md 内容

        Returns:
            移除 frontmatter 后的正文内容
        """
        match = _FRONTMATTER_RE.match(content)
        return content[match.end() :].strip() if match else content

    @staticmethod
    def _skill_meta(metadata: dict[str, str]) -> dict:
        """
        解析 `metadata` 字段中嵌套的 JSON 配置。

        支持两种格式：
        1. 直接的 metadata 字段（旧格式）
        2. metadata.nanobot 或 metadata.openclaw（新格式）

        Args:
            metadata: 基础元数据字典

        Returns:
            解析后的技能元数据字典
        """
        raw = metadata.get("metadata")
        if isinstance(raw, dict):
            payload = raw
        elif isinstance(raw, str):
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return {}
        else:
            return {}

        if not isinstance(payload, dict):
            return {}
        # 兼容新旧格式：nanobot 或 openclaw
        return payload.get("nanobot", payload.get("openclaw", {}))
