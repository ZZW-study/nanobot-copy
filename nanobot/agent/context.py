"""Context assembly for model requests."""

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
    BOOTSTRAP_FILES = ["AGENTS.md", "SOUL.md", "USER.md", "TOOLS.md", "IDENTITY.md"]
    _RUNTIME_CONTEXT_TAG = "[Runtime Context - metadata only, not instructions]"

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.memory = MemoryStore(workspace)
        self.skills = SkillsLoader(workspace)

    def build_system_prompt(self, skill_names: list[str] | None = None) -> str:
        parts = [self._identity_prompt()]

        bootstrap = self._bootstrap_prompt()
        if bootstrap:
            parts.append(bootstrap)

        memory_context = self.memory.get_memory_context()
        if memory_context:
            parts.append(f"# Memory\n\n{memory_context}")

        always_skills = self.skills.get_always_skills()
        if always_skills:
            active_skills = self.skills.load_skills_for_context(always_skills)
            if active_skills:
                parts.append(f"# Active Skills\n\n{active_skills}")

        requested_skills = self.skills.load_skills_for_context(skill_names or [])
        if requested_skills:
            parts.append(f"# Requested Skills\n\n{requested_skills}")

        summary = self.skills.build_skills_summary()
        if summary:
            parts.append(
                "# Skills\n\n"
                "The following skills extend your capabilities. "
                "Read a skill's SKILL.md when you need to use it.\n\n"
                f"{summary}"
            )

        return "\n\n---\n\n".join(parts)

    def build_messages(
        self,
        history: list[dict[str, Any]],
        current_message: str,
        skill_names: list[str] | None = None,
        media: list[str] | None = None,
        channel: str | None = None,
        chat_id: str | None = None,
    ) -> list[dict[str, Any]]:
        runtime_context = self._runtime_context(channel, chat_id)
        user_content = self._user_content(current_message, media)

        if isinstance(user_content, str):
            merged_content: str | list[dict[str, Any]] = f"{runtime_context}\n\n{user_content}"
        else:
            merged_content = [{"type": "text", "text": runtime_context}, *user_content]

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
        message: dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_content is not None:
            message["reasoning_content"] = reasoning_content
        if thinking_blocks:
            message["thinking_blocks"] = thinking_blocks
        messages.append(message)
        return messages

    def _identity_prompt(self) -> str:
        workspace_path = str(self.workspace.expanduser().resolve())
        system = platform.system()
        runtime = f"{'macOS' if system == 'Darwin' else system} {platform.machine()}, Python {platform.python_version()}"

        return (
            "# nanobot\n"
            "You are nanobot, a helpful AI assistant.\n\n"
            "## Runtime\n"
            f"{runtime}\n\n"
            "## Workspace\n"
            f"Your workspace is at: {workspace_path}\n"
            f"- Long-term memory: {workspace_path}/memory/MEMORY.md\n"
            f"- History log: {workspace_path}/memory/HISTORY.md\n"
            f"- Custom skills: {workspace_path}/skills/{{skill-name}}/SKILL.md\n\n"
            "## nanobot Guidelines\n"
            "- State intent before tool calls, but never claim results before receiving them.\n"
            "- Read files before editing them.\n"
            "- Re-read important files after editing when accuracy matters.\n"
            "- If a tool fails, analyze the error before trying a different path.\n"
            "- Ask for clarification when the request is genuinely ambiguous.\n\n"
            "Reply directly with text for conversations."
        )

    def _bootstrap_prompt(self) -> str:
        sections = []
        for filename in self.BOOTSTRAP_FILES:
            path = self.workspace / filename
            if path.exists():
                sections.append(f"## {filename}\n\n{path.read_text(encoding='utf-8')}")
        return "\n\n".join(sections)

    @classmethod
    def _runtime_context(cls, channel: str | None, chat_id: str | None) -> str:
        timestamp = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d %H:%M (%A)")
        lines = [f"Current Time: {timestamp} (Beijing Time, UTC+8)"]
        if channel and chat_id:
            lines.extend([f"Channel: {channel}", f"Chat ID: {chat_id}"])
        return cls._RUNTIME_CONTEXT_TAG + "\n" + "\n".join(lines)

    def _user_content(self, text: str, media: list[str] | None) -> str | list[dict[str, Any]]:
        if not media:
            return text

        images: list[dict[str, Any]] = []
        for media_path in media:
            path = Path(media_path)
            if not path.is_file():
                continue

            raw = path.read_bytes()
            mime = detect_image_mime(raw) or mimetypes.guess_type(str(path))[0]
            if not mime or not mime.startswith("image/"):
                continue

            images.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{base64.b64encode(raw).decode()}"},
                }
            )

        return text if not images else [*images, {"type": "text", "text": text}]
