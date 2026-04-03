"""Skill discovery and loading."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path


BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"
_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


class SkillsLoader:
    def __init__(self, workspace: Path, builtin_skills_dir: Path | None = None):
        self.workspace_skills = workspace / "skills"
        self.builtin_skills = builtin_skills_dir or BUILTIN_SKILLS_DIR

    def list_skills(self, filter_unavailable: bool = True) -> list[dict[str, str]]:
        skills = list(self._discover_skills().values())
        if not filter_unavailable:
            return skills
        return [skill for skill in skills if self._requirements_status(skill["name"])[0]]

    def load_skill(self, name: str) -> str | None:
        path = self._skill_path(name)
        return path.read_text(encoding="utf-8") if path else None

    def load_skills_for_context(self, skill_names: list[str]) -> str:
        parts = []
        for name in skill_names:
            content = self.load_skill(name)
            if content:
                parts.append(f"### Skill: {name}\n\n{self._strip_frontmatter(content)}")
        return "\n\n---\n\n".join(parts)

    def build_skills_summary(self) -> str:
        skills = self.list_skills(filter_unavailable=False)
        if not skills:
            return ""

        def escape_xml(value: str) -> str:
            return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        lines = ["<skills>"]
        for skill in skills:
            metadata = self.get_skill_metadata(skill["name"]) or {}
            available, missing = self._requirements_status(skill["name"], metadata)

            lines.append(f'  <skill available="{str(available).lower()}">')
            lines.append(f"    <name>{escape_xml(skill['name'])}</name>")
            lines.append(f"    <description>{escape_xml(self._skill_description(skill['name'], metadata))}</description>")
            lines.append(f"    <location>{skill['path']}</location>")
            if missing:
                lines.append(f"    <requires>{escape_xml(', '.join(missing))}</requires>")
            lines.append("  </skill>")
        lines.append("</skills>")
        return "\n".join(lines)

    def get_always_skills(self) -> list[str]:
        always: list[str] = []
        for skill in self.list_skills(filter_unavailable=True):
            metadata = self.get_skill_metadata(skill["name"]) or {}
            skill_meta = self._skill_meta(metadata)
            if skill_meta.get("always") or metadata.get("always"):
                always.append(skill["name"])
        return always

    def get_skill_metadata(self, name: str) -> dict[str, str] | None:
        content = self.load_skill(name)
        if not content:
            return None

        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None

        metadata: dict[str, str] = {}
        for line in match.group(1).splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip().strip("\"'")
        return metadata

    def _discover_skills(self) -> dict[str, dict[str, str]]:
        skills: dict[str, dict[str, str]] = {}
        for source, root in (("builtin", self.builtin_skills), ("workspace", self.workspace_skills)):
            if not root or not root.exists():
                continue
            for skill_dir in root.iterdir():
                skill_file = skill_dir / "SKILL.md"
                if skill_dir.is_dir() and skill_file.exists():
                    skills[skill_dir.name] = {
                        "name": skill_dir.name,
                        "path": str(skill_file),
                        "source": source,
                    }
        return skills

    def _skill_path(self, name: str) -> Path | None:
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
        skill_meta = self._skill_meta(metadata or self.get_skill_metadata(name) or {})
        missing: list[str] = []

        requires = skill_meta.get("requires", {})
        for binary in requires.get("bins", []):
            if not shutil.which(binary):
                missing.append(f"CLI: {binary}")
        for env_name in requires.get("env", []):
            if not os.environ.get(env_name):
                missing.append(f"ENV: {env_name}")

        return not missing, missing

    def _skill_description(self, name: str, metadata: dict[str, str] | None = None) -> str:
        metadata = metadata or self.get_skill_metadata(name) or {}
        return metadata.get("description") or name

    @staticmethod
    def _strip_frontmatter(content: str) -> str:
        match = _FRONTMATTER_RE.match(content)
        return content[match.end() :].strip() if match else content

    @staticmethod
    def _skill_meta(metadata: dict[str, str]) -> dict:
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
        return payload.get("nanobot", payload.get("openclaw", {}))
