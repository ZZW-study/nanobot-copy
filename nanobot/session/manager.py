"""Session persistence and cache management."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config import get_legacy_sessions_dir
from nanobot.utils.helpers import ensure_dir, safe_filename


_HISTORY_FIELDS = ("tool_calls", "tool_call_id", "name")


@dataclass
class Session:
    key: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)
    last_consolidated: int = 0

    def add_message(self, role: str, content: str, **kwargs: Any) -> None:
        self.messages.append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
                **kwargs,
            }
        )
        self.updated_at = datetime.now()

    def get_history(self, max_messages: int = 500) -> list[dict[str, Any]]:
        messages = self.messages[self.last_consolidated :][-max_messages:]
        first_user = next((index for index, message in enumerate(messages) if message.get("role") == "user"), None)
        if first_user is not None:
            messages = messages[first_user:]

        history: list[dict[str, Any]] = []
        for message in messages:
            entry = {"role": message["role"], "content": message.get("content", "")}
            for field in _HISTORY_FIELDS:
                if field in message:
                    entry[field] = message[field]
            history.append(entry)
        return history

    def clear(self) -> None:
        self.messages = []
        self.last_consolidated = 0
        self.updated_at = datetime.now()


class SessionManager:
    def __init__(self, workspace: Path | str):
        self.sessions_dir = ensure_dir(Path(workspace) / "sessions")
        self.legacy_sessions_dir = get_legacy_sessions_dir()
        self._cache: dict[str, Session] = {}

    def get_or_create(self, key: str) -> Session:
        session = self._cache.get(key)
        if session is None:
            session = self._load(key) or Session(key=key)
            self._cache[key] = session
        return session

    def save(self, session: Session) -> None:
        path = self._session_path(session.key)
        lines = [json.dumps(self._metadata_line(session), ensure_ascii=False)]
        lines.extend(json.dumps(message, ensure_ascii=False) for message in session.messages)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._cache[session.key] = session

    def invalidate(self, key: str) -> None:
        self._cache.pop(key, None)

    def list_sessions(self) -> list[dict[str, Any]]:
        sessions: list[dict[str, Any]] = []
        for path in self.sessions_dir.glob("*.jsonl"):
            metadata = self._read_metadata(path)
            if metadata:
                sessions.append(metadata)
        return sorted(sessions, key=lambda item: item.get("updated_at", ""), reverse=True)

    def _load(self, key: str) -> Session | None:
        path = self._session_path(key)
        if not path.exists():
            self._migrate_legacy_session(key, path)
        if not path.exists():
            return None

        try:
            metadata: dict[str, Any] = {}
            messages: list[dict[str, Any]] = []
            created_at: datetime | None = None
            updated_at: datetime | None = None
            last_consolidated = 0

            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if data.get("_type") == "metadata":
                    metadata = data.get("metadata", {})
                    created_at = self._parse_datetime(data.get("created_at"))
                    updated_at = self._parse_datetime(data.get("updated_at"))
                    last_consolidated = data.get("last_consolidated", 0)
                    continue
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

    def _migrate_legacy_session(self, key: str, path: Path) -> None:
        legacy_path = self._legacy_session_path(key)
        if not legacy_path.exists():
            return
        try:
            shutil.move(str(legacy_path), str(path))
            logger.info("Migrated legacy session {}", key)
        except Exception:
            logger.exception("Failed to migrate legacy session {}", key)

    def _session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.sessions_dir / f"{safe_key}.jsonl"

    def _legacy_session_path(self, key: str) -> Path:
        safe_key = safe_filename(key.replace(":", "_"))
        return self.legacy_sessions_dir / f"{safe_key}.jsonl"

    @staticmethod
    def _metadata_line(session: Session) -> dict[str, Any]:
        return {
            "_type": "metadata",
            "key": session.key,
            "created_at": session.created_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "metadata": session.metadata,
            "last_consolidated": session.last_consolidated,
        }

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime | None:
        return datetime.fromisoformat(value) if value else None

    @staticmethod
    def _read_metadata(path: Path) -> dict[str, Any] | None:
        try:
            first_line = path.read_text(encoding="utf-8").splitlines()[0].strip()
        except Exception:
            return None

        if not first_line:
            return None

        try:
            data = json.loads(first_line)
        except json.JSONDecodeError:
            return None

        if data.get("_type") != "metadata":
            return None

        return {
            "key": data.get("key") or path.stem.replace("_", ":", 1),
            "created_at": data.get("created_at"),
            "updated_at": data.get("updated_at"),
            "path": str(path),
        }
