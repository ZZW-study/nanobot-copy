"""文件系统工具集：读取、写入、编辑、列出目录"""

import difflib
import re
from pathlib import Path
from typing import Any

from nanobot.agent.tools.base import Tool


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """解析文件路径，确保在允许的目录范围内"""
    p = Path(path).expanduser()
    if not p.is_absolute() and workspace:
        p = workspace / p
    resolved = p.resolve()

    if allowed_dir:
        all_dirs = [allowed_dir] + (extra_allowed_dirs or [])
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(f"路径 {path} 超出了允许访问的目录范围：{allowed_dir}")
    return resolved


def _strip_code_fence(content: str) -> str:
    """去除 Markdown 代码围栏"""
    m = re.match(r'^\s*(```|~~~)[^\n]*\n([\s\S]*?)\n\1\s*$', content, re.DOTALL)
    if m:
        return m.group(2)
    return content


def _is_under(path: Path, directory: Path) -> bool:
    """判断 path 是否位于 directory 目录之下"""
    try:
        path.relative_to(directory.resolve())
        return True
    except ValueError:
        return False


class ReadFileTool(Tool):
    """读取文件内容"""

    _MAX_CHARS = 128_000
    _DEFAULT_LIMIT = 2000

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "读取文件内容。可使用 offset 和 limit 分页读取大文件。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
                "offset": {"type": "integer", "description": "起始行号（1 索引）", "minimum": 1},
                "limit": {"type": "integer", "description": "最多读取的行数（默认 2000）", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(self, path: str, offset: int = 1, limit: int | None = None, **kwargs: Any) -> str:
        try:
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            if not fp.exists():
                return f"错误：文件不存在：{path}"
            if not fp.is_file():
                return f"错误：目标不是文件：{path}"

            all_lines = fp.read_text(encoding="utf-8").splitlines()
            total = len(all_lines)

            if offset < 1:
                offset = 1
            if total == 0:
                return f"（空文件：{path}）"
            if offset > total:
                return f"错误：起始行号 {offset} 超出了文件末尾（总行数 {total}）"

            start = offset - 1
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            if end < total:
                result += f"\n\n（当前显示第 {offset} 到 {end} 行，共 {total} 行；如需继续，请使用 offset={end + 1}）"
            else:
                result += f"\n\n（文件结束，共 {total} 行）"
            return result

        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"错误：读取文件失败：{e}"


class WriteFileTool(Tool):
    """写入文件"""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "将内容写入文件，自动创建父目录。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, path: str, content: str, **kwargs: Any) -> str:
        try:
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fp.write_text(content, encoding="utf-8")
            return f"已成功写入文件：{fp}（共 {len(content)} 个字符）"
        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"错误：写入文件失败：{e}"


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """在内容中定位旧文本，支持宽松匹配"""
    if old_text in content:
        return old_text, content.count(old_text)

    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0

    stripped_old = [line.strip() for line in old_lines]
    content_lines = content.splitlines()

    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        window = content_lines[i : i + len(stripped_old)]
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))

    if candidates:
        return candidates[0], len(candidates)
    return None, 0


class EditFileTool(Tool):
    """编辑文件"""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "edit_file"

    @property
    def description(self) -> str:
        return "编辑文件，将 old_text 替换为 new_text。支持轻微的空白差异。设置 replace_all=true 可替换所有出现。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要编辑的文件路径"},
                "old_text": {"type": "string", "description": "要查找并替换的文本"},
                "new_text": {"type": "string", "description": "替换成的新文本"},
                "replace_all": {"type": "boolean", "description": "是否替换所有出现（默认 false）"},
            },
            "required": ["path", "old_text", "new_text"],
        }

    async def execute(
        self, path: str, old_text: str, new_text: str,
        replace_all: bool = False, **kwargs: Any,
    ) -> str:
        try:
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            if not fp.exists():
                return f"错误：文件不存在：{path}"

            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw
            content = raw.decode("utf-8").replace("\r\n", "\n")

            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            if match is None:
                return self._not_found_msg(old_text, content, path)

            if count > 1 and not replace_all:
                return f"警告：old_text 在文件中出现了 {count} 次。请补充更多上下文或传入 replace_all=true。"

            norm_new = new_text.replace("\r\n", "\n")
            new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)

            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            fp.write_bytes(new_content.encode("utf-8"))
            return f"已成功编辑文件：{fp}"

        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"错误：编辑文件失败：{e}"

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        lines = content.splitlines(keepends=True)
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)

        best_ratio, best_start = 0.0, 0
        for i in range(max(1, len(lines) - window + 1)):
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        if best_ratio > 0.5:
            diff = "\n".join(difflib.unified_diff(
                old_lines, lines[best_start : best_start + window],
                fromfile="old_text（输入内容）",
                tofile=f"{path}（文件实际内容，第 {best_start + 1} 行起）",
                lineterm="",
            ))
            return f"错误：在 {path} 中找不到 old_text。\n最接近的片段位于第 {best_start + 1} 行起（相似度 {best_ratio:.0%}）：\n{diff}"
        return f"错误：在 {path} 中找不到 old_text，且没有发现足够接近的片段。"


class ListDirTool(Tool):
    """列出目录内容"""

    _DEFAULT_MAX = 200
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        self._workspace = workspace
        self._allowed_dir = allowed_dir

    @property
    def name(self) -> str:
        return "list_dir"

    @property
    def description(self) -> str:
        return "列出目录内容。设置 recursive=true 可递归显示。常见噪声目录会被自动忽略。"

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要列出的目录路径"},
                "recursive": {"type": "boolean", "description": "是否递归列出（默认 false）"},
                "max_entries": {"type": "integer", "description": "最多返回的条目数（默认 200）", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(
        self, path: str, recursive: bool = False,
        max_entries: int | None = None, **kwargs: Any,
    ) -> str:
        try:
            dp = _resolve_path(path, self._workspace, self._allowed_dir)
            if not dp.exists():
                return f"错误：目录不存在：{path}"
            if not dp.is_dir():
                return f"错误：目标不是目录：{path}"

            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []
            total = 0

            if recursive:
                for item in sorted(dp.rglob("*")):
                    if any(p in self._IGNORE_DIRS for p in item.parts):
                        continue
                    total += 1
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                for item in sorted(dp.iterdir()):
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    if len(items) < cap:
                        pfx = "[DIR] " if item.is_dir() else "[FILE] "
                        items.append(f"{pfx}{item.name}")

            if not items and total == 0:
                return f"目录为空：{path}"

            result = "\n".join(items)
            if total > cap:
                result += f"\n\n（结果已截断，当前显示前 {cap} 项，共 {total} 项）"
            return result

        except PermissionError as e:
            return f"错误：{e}"
        except Exception as e:
            return f"错误：列出目录失败：{e}"
