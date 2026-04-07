"""通用辅助函数"""

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
    """根据二进制文件头识别图片 MIME 类型"""
    for prefix, check_slice, expected, mime in _IMAGE_SIGNATURES:
        if not data.startswith(prefix):
            continue
        if check_slice is not None and data[check_slice] != expected:
            continue
        return mime
    return None


def ensure_dir(path: Path) -> Path:
    """确保目录存在"""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """将不安全字符替换为下划线"""
    return _UNSAFE_CHARS.sub("_", name).strip()


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """创建工作区必要目录"""
    added: list[str] = []

    # 创建必要目录
    dirs = [
        workspace / "memory",
        workspace / "skills",
        workspace / "sessions",
    ]

    for d in dirs:
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            added.append(str(d.relative_to(workspace)))

    return added
