"""技能发现与加载。

从多个目录（内置/用户/工作区）发现技能，解析 SKILL.md 元数据，
构建技能目录供大模型自行选择加载。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# 内置技能目录：ZBot/skills/
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# 技能来源优先级：数值越大优先级越高，同名技能会被覆盖
SOURCE_PRIORITY = {
    "builtin": 1,
    "user": 2,
    "workspace": 3,
}


@dataclass(slots=True)
class SkillManifest:
    """技能元数据。"""

    name: str  # 技能名称
    description: str  # 技能描述
    source: str  # 来源：builtin/user/workspace
    base_dir: Path  # 技能目录

    @property
    def skill_file(self) -> Path:
        """SKILL.md 文件路径。"""
        return self.base_dir / "SKILL.md"


def _extract_frontmatter_and_body(content: str) -> tuple[str | None, str]:
    """分离 SKILL.md 的 frontmatter 和正文。"""
    lines = content.splitlines()

    if not lines or lines[0].strip() != "---":
        return None, content.strip()

    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :]).strip()
            return frontmatter, body

    raise ValueError("SKILL.md frontmatter 缺少结束分隔线 ---")


def _load_frontmatter(frontmatter_text: str) -> dict[str, Any]:
    """解析 YAML frontmatter。"""
    loaded = yaml.safe_load(frontmatter_text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("SKILL.md frontmatter 必须是一个 YAML 对象")
    return loaded


def _normalize_manifest(skill_dir: Path, source: str) -> SkillManifest:
    """解析技能目录，返回 SkillManifest。"""
    skill_file = skill_dir / "SKILL.md"
    if not skill_file.exists():
        raise ValueError(f"目录 {skill_dir} 下缺少 SKILL.md")

    content = skill_file.read_text(encoding="utf-8")
    frontmatter_text, _body = _extract_frontmatter_and_body(content)
    if frontmatter_text is None:
        raise ValueError(f"{skill_file} 缺少 YAML frontmatter")

    frontmatter = _load_frontmatter(frontmatter_text)

    name = str(frontmatter.get("name", "")).strip()
    description = str(frontmatter.get("description", "")).strip()

    if not name:
        raise ValueError(f"{skill_file} 缺少 name")
    if not description:
        raise ValueError(f"{skill_file} 缺少 description")
    if name != skill_dir.name:
        raise ValueError(
            f"{skill_file} 的 name 为 '{name}'，但目录名是 '{skill_dir.name}'，两者必须一致"
        )

    return SkillManifest(
        name=name,
        description=description,
        source=source,
        base_dir=skill_dir,
    )


class SkillsLoader:
    """技能加载器：发现技能并构建目录。"""

    def __init__(
        self,
        workspace: Path | None = None,
        builtin_skills_dir: Path | None = None,
        user_skills_dir: Path | None = None,
    ):
        """初始化技能加载目录和缓存状态。"""
        self.builtin_skills_dir = builtin_skills_dir or BUILTIN_SKILLS_DIR
        self.user_skills_dir = user_skills_dir or (Path.home() / ".ZBot" / "skills")
        self.workspace_skills_dir = workspace / "skills" if workspace else None
        self._registry_cache: dict[str, SkillManifest] | None = None

    def build_catalog_for_prompt(self) -> str:
        """构建技能目录摘要，注入到 system prompt。"""
        skills = self.list_skills()
        if not skills:
            return ""

        lines = [
            "以下是当前可用的技能目录。",
            "根据技能描述判断是否需要；需要时读取对应 SKILL.md 正文。",
            "",
        ]

        for manifest in skills:
            lines.append(
                f"- `{manifest.name}`：{manifest.description}（路径：{manifest.skill_file}）"
            )

        return "\n".join(lines)

    def list_skills(self) -> list[SkillManifest]:
        """列出所有技能，按名称排序。"""
        skills = list(self._registry().values())
        skills.sort(key=lambda item: item.name)
        return skills

    def _registry(self) -> dict[str, SkillManifest]:
        """返回缓存的技能注册表。"""
        if self._registry_cache is None:
            self._registry_cache = self._discover_registry()
        return self._registry_cache

    def _discover_registry(self) -> dict[str, SkillManifest]:
        """扫描所有目录，构建技能注册表。"""
        registry: dict[str, SkillManifest] = {}

        for source_name, source_dir in self._iter_sources():
            if not source_dir.exists():
                continue

            for skill_dir in source_dir.iterdir():
                if not skill_dir.is_dir():
                    continue

                try:
                    manifest = _normalize_manifest(skill_dir, source_name)
                except Exception:
                    continue

                existing = registry.get(manifest.name)
                if existing is None:
                    registry[manifest.name] = manifest
                elif SOURCE_PRIORITY[manifest.source] >= SOURCE_PRIORITY[existing.source]:
                    registry[manifest.name] = manifest

        return registry

    def _iter_sources(self) -> list[tuple[str, Path]]:
        """返回所有要扫描的技能目录。"""
        sources: list[tuple[str, Path]] = [("builtin", self.builtin_skills_dir)]
        if self.user_skills_dir:
            sources.append(("user", self.user_skills_dir))
        if self.workspace_skills_dir:
            sources.append(("workspace", self.workspace_skills_dir))
        return sources
