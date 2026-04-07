"""通用辅助函数。

本模块提供一些零散但各处都会用到的工具函数，
例如图片类型检测、目录创建、文件名清理、工作区初始化等。
"""

from __future__ import annotations

import re  # 正则表达式模块，用于模式匹配
from pathlib import Path  # 面向对象的文件路径处理


# 图片文件的二进制签名（Magic Number）对照表。
# 每种图片格式的文件开头都有固定的字节序列，用于识别类型。
# 元组格式：(前缀字节, 需要额外检查的位置, 期望的字节值, MIME 类型)
_IMAGE_SIGNATURES: tuple[tuple[bytes, slice | None, bytes | None, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", None, None, "image/png"),        # PNG 图片
    (b"\xff\xd8\xff", None, None, "image/jpeg"),            # JPEG 图片
    (b"GIF87a", None, None, "image/gif"),                   # GIF 图片（87a 版本）
    (b"GIF89a", None, None, "image/gif"),                   # GIF 图片（89a 版本）
    (b"RIFF", slice(8, 12), b"WEBP", "image/webp"),         # WebP 图片（RIFF 容器 + WEBP 标记）
)
# 文件名中不允许出现的不安全字符（Windows/Unix 系统保留字符）
_UNSAFE_CHARS = re.compile(r'[<>:"/\\|?*]')


def detect_image_mime(data: bytes) -> str | None:
    """根据二进制文件头识别图片 MIME 类型。

    每种图片格式在文件开头都有固定的"魔数"（Magic Number），
    通过比对文件开头的字节就能判断图片类型。

    参数：
        data: 图片文件的二进制数据

    返回：
        MIME 类型字符串（如 "image/png"），无法识别则返回 None
    """
    # 遍历所有已知的图片签名
    for prefix, check_slice, expected, mime in _IMAGE_SIGNATURES:
        # 先检查文件开头是否匹配前缀
        if not data.startswith(prefix):
            continue
        # 某些格式（如 WebP）还需要在特定偏移位置检查额外标记
        if check_slice is not None and data[check_slice] != expected:
            continue
        # 匹配成功，返回 MIME 类型
        return mime
    return None  # 所有签名都不匹配


def ensure_dir(path: Path) -> Path:
    """确保目录存在，如果不存在则递归创建。

    参数：
        path: 目标目录路径

    返回：
        传入的 path（方便链式调用）
    """
    # parents=True 递归创建所有缺失的父目录
    # exist_ok=True 目录已存在时不报错
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(name: str) -> str:
    """将文件名中的不安全字符替换为下划线。

    某些字符（如 / \ ? * 等）在文件系统中是非法的，
    此函数将这些字符替换为下划线，保证文件名可用。

    参数：
        name: 原始文件名

    返回：
        清理后的安全文件名
    """
    # 用正则将所有不安全字符替换为下划线，并去除首尾空白
    return _UNSAFE_CHARS.sub("_", name).strip()


def sync_workspace_templates(workspace: Path, silent: bool = False) -> list[str]:
    """创建工作区所需的必要目录。

    在首次使用 nanobot 或新建工作区时，
    需要创建 memory（记忆）、skills（技能）、sessions（会话）等目录。

    参数：
        workspace: 工作区根目录路径
        silent: 是否静默模式（目前未使用，保留供后续扩展）

    返回：
        本次新创建的目录相对路径列表
    """
    added: list[str] = []  # 记录本次新增的目录

    # 创建工作区必须存在的目录列表
    dirs = [
        workspace / "memory",    # 长期记忆和归档目录
        workspace / "skills",    # 自定义技能目录
        workspace / "sessions",  # 会话历史记录目录
    ]

    for d in dirs:
        # 目录不存在时创建
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            # 记录相对路径
            added.append(str(d.relative_to(workspace)))

    return added  # 返回新增目录列表
