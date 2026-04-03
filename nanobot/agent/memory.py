"""Long-term memory and history consolidation."""

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
            "description": "Persist a condensed history entry and updated long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "history_entry": {
                        "type": "string",
                        "description": (
                            "A 2-5 sentence summary that starts with a timestamp like "
                            "[YYYY-MM-DD HH:MM] so it remains grep-friendly."
                        ),
                    },
                    "memory_update": {
                        "type": "string",
                        "description": "The full updated MEMORY.md content.",
                    },
                },
                "required": ["history_entry", "memory_update"],
            },
        },
    }
]


class MemoryStore:
    def __init__(self, workspace: Path):
        self.memory_dir = ensure_dir(workspace / "memory")
        self.memory_file = self.memory_dir / "MEMORY.md"
        self.history_file = self.memory_dir / "HISTORY.md"

    def read_long_term(self) -> str:
        return self.memory_file.read_text(encoding="utf-8") if self.memory_file.exists() else ""

    def write_long_term(self, content: str) -> None:
        self.memory_file.write_text(content, encoding="utf-8")

    def append_history(self, entry: str) -> None:
        with open(self.history_file, "a", encoding="utf-8") as handle:
            handle.write(entry.strip() + "\n\n")

    def get_memory_context(self) -> str:
        memory = self.read_long_term()
        return f"## Long-term Memory\n{memory}" if memory else ""

    async def consolidate(
        self,
        session: Session,
        provider: LLMProvider,
        model: str,
        *,
        archive_all: bool = False,
        memory_window: int = 50,
    ) -> bool:
        messages, keep_count = self._messages_to_archive(session, archive_all, memory_window)
        if not messages:
            return True

        current_memory = self.read_long_term()
        prompt = self._build_prompt(current_memory, messages)

        try:
            response = await provider.chat(
                messages=[
                    {
                        "role": "system",
                        "content": "You condense conversation history and must call save_memory.",
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=_SAVE_MEMORY_TOOL,
                model=model,
            )
        except Exception:
            logger.exception("Memory consolidation failed")
            return False

        if not response.has_tool_calls:
            logger.warning("Memory consolidation skipped because save_memory was not called")
            return False

        args = self._normalize_tool_args(response.tool_calls[0].arguments)
        if args is None:
            logger.warning("Memory consolidation returned invalid tool arguments")
            return False

        history_entry = self._coerce_text(args.get("history_entry"))
        if history_entry:
            self.append_history(history_entry)

        memory_update = self._coerce_text(args.get("memory_update"))
        if memory_update is not None and memory_update != current_memory:
            self.write_long_term(memory_update)

        session.last_consolidated = 0 if archive_all else len(session.messages) - keep_count
        logger.info(
            "Memory consolidated: archived {} messages, last_consolidated={}",
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
        if archive_all:
            return list(session.messages), 0

        keep_count = max(1, memory_window // 2)
        if len(session.messages) <= keep_count:
            return [], keep_count

        start = session.last_consolidated
        end = len(session.messages) - keep_count
        if end <= start:
            return [], keep_count

        return session.messages[start:end], keep_count

    def _build_prompt(self, current_memory: str, messages: list[dict[str, Any]]) -> str:
        transcript = "\n".join(self._format_messages(messages))
        return (
            "Consolidate the following conversation.\n\n"
            "## Current MEMORY.md\n"
            f"{current_memory or '(empty)'}\n\n"
            "## Conversation To Archive\n"
            f"{transcript}"
        )

    @staticmethod
    def _format_messages(messages: list[dict[str, Any]]) -> list[str]:
        lines: list[str] = []
        for message in messages:
            content = message.get("content")
            if not content:
                continue
            tools = message.get("tools_used") or []
            tool_suffix = f" [tools: {', '.join(tools)}]" if tools else ""
            timestamp = str(message.get("timestamp", "?"))[:16]
            lines.append(f"[{timestamp}] {message['role'].upper()}{tool_suffix}: {content}")
        return lines

    @staticmethod
    def _normalize_tool_args(arguments: Any) -> dict[str, Any] | None:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None

        if isinstance(arguments, list):
            arguments = arguments[0] if arguments and isinstance(arguments[0], dict) else None

        return arguments if isinstance(arguments, dict) else None

    @staticmethod
    def _coerce_text(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)
