"""Shared helpers used across the project."""

from __future__ import annotations

import re
from pathlib import Path


_IMAGE_SIGNATURES: tuple[tuple[bytes, slice | None, bytes | None, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", None, None, "image/png"),
    (b"\xff\xd8\xff", None, None, "image/jpeg"),
    (b"GIF87a", None, None, "image/gif"),
    (b"GIF89a", None, None, "image/gif"),
    (b"RIFF", slice(8, 12), b"WEBP", "image/webp"),
)
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def detect_image_mime(data: bytes) -> str | None:
    for prefix, check_slice, expected, mime in _IMAGE_SIGNATURES:
        if not data.startswith(prefix):
            continue
        if check_slice is not None and data[check_slice] != expected:
            continue
        return mime
    return None


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    return _UNSAFE_CHARS.sub("_", name).strip()


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    from importlib.resources import files as pkg_files

    try:
        templates = pkg_files("nanobot") / "templates"
    except Exception:
        return []

    if not templates.is_dir():
        return []

    added: list[str] = []

    def write_template(src, dest: Path) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        content = "" if src is None else src.read_text(encoding="utf-8")
        dest.write_text(content, encoding="utf-8")
        added.append(str(dest.relative_to(workspace)))

    for item in templates.iterdir():
        if item.name.endswith(".md"):
            write_template(item, workspace / item.name)

    write_template(templates / "memory" / "MEMORY.md", workspace / "memory" / "MEMORY.md")
    write_template(None, workspace / "memory" / "HISTORY.md")
    (workspace / "skills").mkdir(parents=True, exist_ok=True)

    if added and not silent:
        from rich.console import Console

        console = Console()
        for name in added:
            console.print(f"[dim]创建了{name}[/dim]")

    return added
