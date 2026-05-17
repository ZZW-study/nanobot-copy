"""文件系统工具集：读取、写入、编辑、列出目录"""

import difflib
from pathlib import Path
from typing import Any

from ZBot.agent.tools.base import Tool, format_tool_error


def _resolve_path(
    path: str,
    workspace: Path | None = None,
    allowed_dir: Path | None = None,
    extra_allowed_dirs: list[Path] | None = None,
) -> Path:
    """解析文件路径，确保在允许的目录范围内"""
    # 将字符串路径转为 Path 对象，并展开 ~ 为家目录
    p = Path(path).expanduser()
    # 如果是相对路径且提供了工作区，则拼接工作区目录
    if not p.is_absolute() and workspace:
        p = workspace / p
    # 解析为绝对路径（消除 .. 等相对符号）
    resolved = p.resolve()

    # 如果设置了允许访问的目录，则检查路径是否在其范围内
    if allowed_dir:
        # 合并主允许目录和额外允许目录
        all_dirs = [allowed_dir] + (extra_allowed_dirs or [])
        # 只要路径不在任何一个允许目录下，就拒绝访问
        if not any(_is_under(resolved, d) for d in all_dirs):
            raise PermissionError(f"路径 {path} 超出了允许访问的目录范围：{allowed_dir}")
    return resolved


def _is_under(path: Path, directory: Path) -> bool:
    """判断 path 是否位于 directory 目录之下"""
    try:
        # 尝试计算 path 相对于 directory 的相对路径,path 必须是 directory 的子路径
        path.relative_to(directory.resolve())
        # 如果没抛异常，说明 path 在 directory 下面
        return True
    except ValueError:
        # 抛出 ValueError 说明 path 不在 directory 的子树中
        return False


def _preview_dir(path: Path, limit: int = 8) -> str:
    """返回目录下少量条目，给失败结果提供可行动线索。"""
    if not path.exists() or not path.is_dir():
        return ""
    try:
        names = sorted(item.name + ("/" if item.is_dir() else "") for item in path.iterdir())
    except OSError:
        return ""
    if not names:
        return "目录为空"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview += f", ...（共 {len(names)} 项）"
    return preview


def _path_failure_hint(path: str, resolved: Path, *, expected: str, workspace: Path | None) -> str:
    """为路径类失败生成观察信息。"""
    parent = resolved.parent
    parts = [
        f"请求路径：{path}",
        f"解析后路径：{resolved}",
    ]
    if workspace is not None:
        parts.append(f"工作区：{workspace}")
    parts.append(f"期望类型：{expected}")
    if parent.exists():
        parts.append(f"父目录存在：{parent}")
        preview = _preview_dir(parent)
        if preview:
            parts.append(f"父目录条目预览：{preview}")
    else:
        parts.append(f"父目录不存在：{parent}")
    return "；".join(parts)


class ReadFileTool(Tool):
    """读取文件内容"""

    _MAX_CHARS = 128_000  # 返回内容的最大字符数，超出则截断.数字下划线写法,下划线只用来分隔数字、方便阅读，不影响数值大小
    _DEFAULT_LIMIT = 2000  # 默认读取行数

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        """初始化读取工具的工作区和访问边界。"""
        self._workspace = workspace         # 工作区根目录
        self._allowed_dir = allowed_dir     # 允许访问的目录

    @property
    def name(self) -> str:
        """返回读取文件工具名称。"""
        return "read_file"

    @property
    def description(self) -> str:
        """返回读取文件工具说明。"""
        return "读取文件内容。可使用 offset 和 limit 分页读取大文件。"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回读取文件工具参数 Schema。"""
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要读取的文件路径"},
                "offset": {"type": "integer", "description": "起始行号（1 索引）", "minimum": 1},
                "limit": {"type": "integer", "description": "最多读取的行数（默认 2000）", "minimum": 1},
            },
            "required": ["path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """读取指定文件内容并按需分页返回。"""
        try:
            # 解析路径并检查是否在允许范围内
            path: str = kwargs.get("path", "")
            offset: int = kwargs.get("offset", 1)
            limit = kwargs.get("limit", None)
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            # 检查文件是否存在
            if not fp.exists():
                return format_tool_error(
                    "文件不存在",
                    attempted=f"读取文件 {path}",
                    observed=_path_failure_hint(path, fp, expected="文件", workspace=self._workspace),
                    do_not_repeat=f"不要继续用相同路径调用 read_file：{path}",
                    next_action=f"先调用 list_dir 查看父目录：{fp.parent}",
                )
            # 检查是否为普通文件（非目录）
            if not fp.is_file():
                next_action = f"如果要查看目录内容，请调用 list_dir：{path}" if fp.is_dir() else "请确认目标路径类型"
                return format_tool_error(
                    "目标不是文件",
                    attempted=f"读取文件 {path}",
                    observed=_path_failure_hint(path, fp, expected="文件", workspace=self._workspace),
                    do_not_repeat=f"不要继续用 read_file 读取该路径：{path}",
                    next_action=next_action,
                )

            # 读取文件全部行
            all_lines = fp.read_text(encoding="utf-8").splitlines()   
            # fp.read_text(encoding="utf-8")
            # 打开文件 fp
            # 以 UTF-8 编码读取全部内容
            # 返回一个大字符串（包含所有换行、空格）
            # splitlines()
            # 把上面那个大字符串按换行符切割
            # 自动去掉换行符 \n、\r\n
            # 返回一个列表，每一项是文件的一行内容
            total = len(all_lines)  # 总行数

            # 校正 offset 为最小值 1
            if offset < 1:
                offset = 1
            # 空文件直接返回提示
            if total == 0:
                return f"（空文件：{path}）"
            # offset 超出总行数则返回错误
            if offset > total:
                return format_tool_error(
                    "起始行号超出文件末尾",
                    attempted=f"读取 {path} 的第 {offset} 行起",
                    observed=f"文件总行数为 {total}",
                    do_not_repeat=f"不要继续使用 offset={offset} 读取该文件",
                    next_action="改用更小的 offset，或根据已读取内容继续分析",
                )

            # 计算实际读取范围（offset 是 1 索引，需减 1）
            start = offset - 1
            # 结束位置取 limit 默认值和总行数的较小值
            end = min(start + (limit or self._DEFAULT_LIMIT), total)
            # 给每行加上行号前缀
            numbered = [f"{start + i + 1}| {line}" for i, line in enumerate(all_lines[start:end])]
            result = "\n".join(numbered)

            # 如果内容超过最大字符数，截断到限制以内
            if len(result) > self._MAX_CHARS:
                trimmed, chars = [], 0
                for line in numbered:
                    chars += len(line) + 1
                    if chars > self._MAX_CHARS:
                        break
                    trimmed.append(line)
                end = start + len(trimmed)
                result = "\n".join(trimmed)

            # 提示用户是否还有更多内容需要读取
            if end < total:
                result += f"\n\n（当前显示第 {offset} 到 {end} 行，共 {total} 行；如需继续，请使用 offset={end + 1}）"
            else:
                result += f"\n\n（文件结束，共 {total} 行）"
            return result

        except PermissionError as e:
            return format_tool_error(
                str(e),
                attempted=f"读取文件 {kwargs.get('path', '')}",
                do_not_repeat="不要重复访问同一路径",
                next_action="改用工作区内允许访问的路径，或先 list_dir 确认可访问目录",
            )
        except Exception as e:
            return format_tool_error(
                f"读取文件失败：{e}",
                attempted=f"读取文件 {kwargs.get('path', '')}",
                do_not_repeat="不要用相同参数重复读取",
                next_action="先确认路径、编码和文件类型；必要时改用 list_dir 定位文件",
            )


class WriteFileTool(Tool):
    """写入文件"""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        """初始化写入工具的工作区和访问边界。"""
        self._workspace = workspace  # 工作区根目录
        self._allowed_dir = allowed_dir  # 允许访问的目录

    @property
    def name(self) -> str:
        """返回写入文件工具名称。"""
        return "write_file"

    @property
    def description(self) -> str:
        """返回写入文件工具说明。"""
        return "将内容写入文件，自动创建父目录。"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回写入文件工具参数 Schema。"""
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "要写入的文件路径"},
                "content": {"type": "string", "description": "要写入的内容"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs: Any) -> str:
        """把指定内容写入目标文件。"""
        path = kwargs.get("path", "")
        content = kwargs.get("content", "")
        try:
            # 解析路径(包含了文件)并检查是否在允许范围内
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            # 自动创建所有缺失的父目录
            fp.parent.mkdir(parents=True, exist_ok=True)
            # 写入文件内容, write_text() 会自动创建不存在的文件。只会创建文件本身，不会自动创建父级文件夹
            fp.write_text(content, encoding="utf-8")
            return f"已成功写入文件：{fp}（共 {len(content)} 个字符）"
        except PermissionError as e:
            return format_tool_error(
                str(e),
                attempted=f"写入文件 {path}",
                do_not_repeat="不要重复写入同一路径",
                next_action="改用工作区内允许写入的路径",
            )
        except Exception as e:
            return format_tool_error(
                f"写入文件失败：{e}",
                attempted=f"写入文件 {path}",
                do_not_repeat="不要用相同参数重复写入",
                next_action="检查父目录、权限和内容大小后再决定是否重试",
            )


def _find_match(content: str, old_text: str) -> tuple[str | None, int]:
    """在内容中定位旧文本，支持宽松匹配（忽略缩进差异）"""
    # 先尝试精确匹配
    if old_text in content:
        return old_text, content.count(old_text)  # 返回匹配文本和出现次数

    # 精确匹配失败，按行分割进行宽松匹配
    old_lines = old_text.splitlines()
    if not old_lines:
        return None, 0  # 空文本无法匹配

    # 对旧文本的每行去除首尾空白
    stripped_old = [line.strip() for line in old_lines]
    # 同样分割文件内容
    content_lines = content.splitlines()

    # 滑动窗口遍历文件内容的每一处可能匹配的位置
    candidates = []
    for i in range(len(content_lines) - len(stripped_old) + 1):
        # 取与 old_lines 等长的窗口
        window = content_lines[i : i + len(stripped_old)]
        # 对比去除空白后的行是否一致
        if [line.strip() for line in window] == stripped_old:
            candidates.append("\n".join(window))  # 记录原始匹配内容

    # 返回第一个匹配和总匹配数
    if candidates:
        return candidates[0], len(candidates)
    return None, 0  # 未找到匹配


class EditFileTool(Tool):
    """编辑文件（查找并替换文本）"""

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        """初始化编辑工具的工作区和访问边界。"""
        self._workspace = workspace  # 工作区根目录
        self._allowed_dir = allowed_dir  # 允许访问的目录

    @property
    def name(self) -> str:
        """返回编辑文件工具名称。"""
        return "edit_file"

    @property
    def description(self) -> str:
        """返回编辑文件工具说明。"""
        return "编辑文件，将 old_text 替换为 new_text。支持轻微的空白差异。设置 replace_all=true 可替换所有出现。"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回编辑文件工具参数 Schema。"""
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
        self, **kwargs: Any
    ) -> str:
        """查找并替换文件中的指定文本。"""
        path = kwargs.get("path", "")
        old_text = kwargs.get("old_text", "")
        new_text = kwargs.get("new_text", "")
        replace_all = kwargs.get("replace_all", False)

        try:
            # 解析路径并检查是否在允许范围内
            fp = _resolve_path(path, self._workspace, self._allowed_dir)
            # 检查文件是否存在
            if not fp.exists():
                return format_tool_error(
                    "文件不存在",
                    attempted=f"编辑文件 {path}",
                    observed=_path_failure_hint(path, fp, expected="文件", workspace=self._workspace),
                    do_not_repeat=f"不要继续编辑不存在的路径：{path}",
                    next_action=f"先调用 list_dir 查看父目录：{fp.parent}",
                )

            # 以二进制读取，返回字节串（bytes），检测换行符类型（CRLF 还是 LF）
            raw = fp.read_bytes()
            uses_crlf = b"\r\n" in raw  # Windows 换行符
            # 统一转为 LF 处理
            content = raw.decode("utf-8").replace("\r\n", "\n")

            # 查找要替换的旧文本（也统一转为 LF）
            match, count = _find_match(content, old_text.replace("\r\n", "\n"))

            # 未找到匹配文本
            if match is None:
                return self._not_found_msg(old_text, content, path)

            # 多次出现但没开启全部替换，提示用户补充上下文
            if count > 1 and not replace_all:
                return f"警告：old_text 在文件中出现了 {count} 次。请补充更多上下文或传入 replace_all=true。"

            # 规范化新文本的换行符
            norm_new = new_text.replace("\r\n", "\n")
            # 执行替换：replace_all 替换所有，否则只替换第一次
            new_content = content.replace(match, norm_new) if replace_all else content.replace(match, norm_new, 1)

            # 如果原文件是 CRLF 换行，则恢复回去
            if uses_crlf:
                new_content = new_content.replace("\n", "\r\n")

            # 写回文件
            fp.write_bytes(new_content.encode("utf-8"))
            return f"已成功编辑文件：{fp}"

        except PermissionError as e:
            return format_tool_error(
                str(e),
                attempted=f"编辑文件 {path}",
                do_not_repeat="不要重复编辑同一路径",
                next_action="改用工作区内允许访问的路径",
            )
        except Exception as e:
            return format_tool_error(
                f"编辑文件失败：{e}",
                attempted=f"编辑文件 {path}",
                do_not_repeat="不要用相同参数重复编辑",
                next_action="先 read_file 确认当前内容，再构造更准确的 old_text",
            )

    @staticmethod
    def _not_found_msg(old_text: str, content: str, path: str) -> str:
        """生成未找到匹配文本时的详细错误消息（含最接近片段的 diff）"""
        lines = content.splitlines(keepends=True)  # 保留换行符
        old_lines = old_text.splitlines(keepends=True)
        window = len(old_lines)  # 滑动窗口大小

        best_ratio, best_start = 0.0, 0  # 最佳相似度和起始行
        # 滑动窗口遍历，找到与 old_text 最相似的片段
        for i in range(max(1, len(lines) - window + 1)):
            # 使用 difflib 计算文本相似度
            ratio = difflib.SequenceMatcher(None, old_lines, lines[i : i + window]).ratio()
            if ratio > best_ratio:
                best_ratio, best_start = ratio, i

        # 相似度超过 50% 则展示 diff 差异
        if best_ratio > 0.5:
            diff = "\n".join(difflib.unified_diff(
                old_lines, lines[best_start : best_start + window],
                fromfile="old_text（输入内容）",
                tofile=f"{path}（文件实际内容，第 {best_start + 1} 行起）",
                lineterm="",
            ))
            return format_tool_error(
                "找不到 old_text",
                attempted=f"在 {path} 中替换指定文本",
                observed=f"最接近的片段位于第 {best_start + 1} 行起（相似度 {best_ratio:.0%}）：\n{diff}",
                do_not_repeat="不要用相同 old_text 再次调用 edit_file",
                next_action="先 read_file 查看目标行附近内容，再用文件中的精确文本重试",
            )
        return format_tool_error(
            "找不到 old_text",
            attempted=f"在 {path} 中替换指定文本",
            observed="没有发现足够接近的片段",
            do_not_repeat="不要用相同 old_text 再次调用 edit_file",
            next_action="先 read_file 或搜索目标符号，确认当前文件内容后再编辑",
        )


class ListDirTool(Tool):
    """列出目录内容"""

    _DEFAULT_MAX = 200  # 默认最多返回条目数
    # 需要忽略的噪声目录（版本控制、缓存、虚拟环境等）
    _IGNORE_DIRS = {
        ".git", "node_modules", "__pycache__", ".venv", "venv",
        "dist", "build", ".tox", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", ".coverage", "htmlcov",
    }

    def __init__(self, workspace: Path | None = None, allowed_dir: Path | None = None):
        """初始化列目录工具的工作区和访问边界。"""
        self._workspace = workspace  # 工作区根目录
        self._allowed_dir = allowed_dir  # 允许访问的目录

    @property
    def name(self) -> str:
        """返回列目录工具名称。"""
        return "list_dir"

    @property
    def description(self) -> str:
        """返回列目录工具说明。"""
        return "列出目录内容。设置 recursive=true 可递归显示。常见噪声目录会被自动忽略。"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回列目录工具参数 Schema。"""
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
        self, **kwargs: Any
    ) -> str:
        """列出目录内容并按上限截断返回。"""
        path = kwargs.get("path", "")
        recursive = kwargs.get("recursive", False)
        max_entries = kwargs.get("max_entries", None)

        try:
            # 解析路径并检查是否在允许范围内
            dp = _resolve_path(path, self._workspace, self._allowed_dir)
            # 检查目录是否存在
            if not dp.exists():
                return format_tool_error(
                    "目录不存在",
                    attempted=f"列出目录 {path}",
                    observed=_path_failure_hint(path, dp, expected="目录", workspace=self._workspace),
                    do_not_repeat=f"不要继续用相同路径调用 list_dir：{path}",
                    next_action=f"先列出存在的父目录：{dp.parent}",
                )
            # 检查是否为目录
            if not dp.is_dir():
                next_action = f"如果要读取文件，请调用 read_file：{path}" if dp.is_file() else "请确认目标路径类型"
                return format_tool_error(
                    "目标不是目录",
                    attempted=f"列出目录 {path}",
                    observed=_path_failure_hint(path, dp, expected="目录", workspace=self._workspace),
                    do_not_repeat=f"不要继续用 list_dir 读取该路径：{path}",
                    next_action=next_action,
                )

            # 确定返回条目上限
            cap = max_entries or self._DEFAULT_MAX
            items: list[str] = []  # 收集结果条目
            total = 0  # 总条目计数

            if recursive:
                # 递归遍历所有子文件/目录
                for item in sorted(dp.rglob("*")):
                    # 跳过忽略目录
                    if any(p in self._IGNORE_DIRS for p in item.parts):
                        continue
                    total += 1
                    # 未达上限才加入结果
                    if len(items) < cap:
                        rel = item.relative_to(dp)
                        # 目录加 / 后缀，文件直接用名称
                        items.append(f"{rel}/" if item.is_dir() else str(rel))
            else:
                # 只列出顶层目录
                for item in sorted(dp.iterdir()):
                    # 跳过忽略目录
                    if item.name in self._IGNORE_DIRS:
                        continue
                    total += 1
                    # 未达上限才加入结果
                    if len(items) < cap:
                        pfx = "[DIR] " if item.is_dir() else "[FILE] "
                        items.append(f"{pfx}{item.name}")

            # 空目录处理
            if not items and total == 0:
                return f"目录为空：{path}"

            result = "\n".join(items)
            # 如果实际条目超过上限，提示截断信息
            if total > cap:
                result += f"\n\n（结果已截断，当前显示前 {cap} 项，共 {total} 项）"
            return result

        except PermissionError as e:
            return format_tool_error(
                str(e),
                attempted=f"列出目录 {path}",
                do_not_repeat="不要重复访问同一路径",
                next_action="改用工作区内允许访问的目录",
            )
        except Exception as e:
            return format_tool_error(
                f"列出目录失败：{e}",
                attempted=f"列出目录 {path}",
                do_not_repeat="不要用相同参数重复列目录",
                next_action="检查路径是否存在、是否为目录，再决定下一步",
            )
