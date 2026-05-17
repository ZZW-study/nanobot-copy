"""Shell 命令执行工具。

这个工具是整个系统里风险最高的一类能力，因此实现重点不在"跑命令",
而在"先做足够严格的安全限制，再去跑命令"。

安全策略包括：
1. 危险命令黑名单（阻止 rm -rf、format、shutdown 等高危操作）
2. 工作区路径限制（可选，阻止命令访问工作区外的路径）
3. 执行超时限制（防止命令无限运行）
4. 输出长度限制（防止超大输出占用内存）

核心类：
    ExecTool: 执行 shell 命令的工具类，继承自 Tool 基类
"""

from __future__ import annotations  # 启用未来版本的类型注解特性

import asyncio                      # 用于异步执行 shell 命令
import os  
import re                           # 用于正则表达式匹配危险命令模式
from pathlib import Path  
from typing import Any  

from ZBot.agent.tools.base import Tool, format_tool_error


class ExecTool(Tool):
    """
    执行 shell 命令，并在执行前做安全拦截。
    这是整个系统风险最高的工具，因为 shell 命令可以：
    - 删除文件（rm -rf）
    - 格式化磁盘（format、mkfs）
    - 关闭系统（shutdown、reboot）
    - 执行任意代码
    因此在执行任何命令前，必须通过多层安全检查：
    1. 黑名单检查：阻止已知危险命令
    2. 路径限制：可选，阻止访问工作区外的路径
    3. 超时限制：防止命令无限运行
    4. 输出截断：防止超大输出占用内存
    """

    # 最大超时时间（秒）：防止命令运行过久
    _MAX_TIMEOUT = 600  # 10 分钟上限

    # 最大输出长度（字符数）：防止超大输出占用内存
    _MAX_OUTPUT = 10_000  # 约 10KB

    def __init__(
        self,
        timeout: int = 60,
        working_dir: str | None = None,
        deny_patterns: list[str] | None = None,
        restrict_to_workspace: bool = False,
    ):
        """
        初始化 ExecTool 工具实例。

        Args:
            timeout: 默认命令执行超时时间（秒），默认 60 秒
            working_dir: 默认工作目录，None 表示使用当前进程目录
            deny_patterns: 危险命令黑名单正则表达式列表，None 使用默认黑名单
            restrict_to_workspace: 是否限制只能访问工作区内的路径
        """
        self.timeout = timeout          # 默认超时时间
        self.working_dir = working_dir  # 默认工作目录
        # 默认危险命令模式覆盖删除磁盘、关机、fork bomb 等高风险操作
        # 这些正则表达式会匹配命令字符串，阻止执行
        self.deny_patterns = deny_patterns or [
            r"\brm\s+-[rf]{1,2}\b",  # rm -rf / rm -fr：强制递归删除，\b，单词边界
            r"\bdel\s+/[fq]\b",  # del /f /q：Windows 强制静默删除
            r"\brmdir\s+/s\b",  # rmdir /s：Windows 递归删除目录
            r"(?:^|[;&|]\s*)format\b",  # format：格式化磁盘
            r"\b(mkfs|diskpart)\b",  # mkfs/diskpart：磁盘分区和格式化工具
            r"\bdd\s+if=",  # dd if=：磁盘复制工具（可擦除磁盘）
            r">\s*/dev/sd",  # > /dev/sd：直接写入磁盘设备
            r"\b(shutdown|reboot|poweroff)\b",  # shutdown/reboot/poweroff：关机/重启命令
            r":\(\)\s*\{.*\};\s*:",  # :(){ :|:& };:：Fork bomb（进程爆炸攻击）
        ]

        self.restrict_to_workspace = restrict_to_workspace  # 是否限制工作区路径

    @property
    def name(self) -> str:
        """返回 shell 执行工具名称。"""
        return "exec"

    @property
    def description(self) -> str:
        """返回 shell 执行工具说明。"""
        return "执行 shell 命令并返回结果。使用前请确认命令安全且必要。"

    @property
    def parameters(self) -> dict[str, Any]:
        """返回 shell 执行工具参数 Schema。"""
        return {
            "type": "object",  # 参数必须是一个对象，具体是什么对象，在参数里面定义，parameters 整体是一个对象
            "properties": {
                "command": {
                    "type": "string",  # command 参数必须是字符串
                    "description": "要执行的 shell 命令。",
                },
                "working_dir": {
                    "type": "string",  # working_dir 参数可选，字符串类型
                    "description": "可选。覆盖默认工作目录。",
                },
                "timeout": {
                    "type": "integer",  # timeout 参数可选，整数类型
                    "description": "超时时间，单位秒。默认 60 秒，最大 600 秒。",
                    "minimum": 1,    # 最小值 1 秒
                    "maximum": 600,  # 最大值 600 秒（10 分钟）
                },
            },
            "required": ["command"],  # command 是必须参数，其他是可选
        }

    async def execute(
        self,
        **kwargs: Any,
    ) -> str:
        """
        执行 shell 命令并返回标准化后的输出。

        这是工具的核心方法，执行流程：
        1. 确定工作目录（参数优先级：调用参数 > 工具初始化参数 > 当前进程目录）
        2. 执行安全检查（黑名单、路径限制）
        3. 异步执行命令（使用 asyncio.create_subprocess_shell）
        4. 等待命令完成或超时
        5. 处理输出（截断过长内容）

        Args:
            command: 要执行的 shell 命令字符串
            working_dir: 可选，覆盖默认工作目录
            timeout: 可选，覆盖默认超时时间（秒）
            **kwargs: 其他未使用的参数（工具框架兼容）

        Returns:
            命令执行结果字符串，包含：
            - 标准输出内容
            - 标准错误输出（如果有）
            - 退出码
            - 或错误信息（如果被拦截或执行失败）
        """
        # 工作目录优先级：调用参数 > 工具初始化参数 > 当前进程目录
        command = kwargs.get("command", "")
        working_dir = kwargs.get("working_dir", None)
        timeout = kwargs.get("timeout", None)

        # 空命令检查
        if not command.strip():
            return format_tool_error(
                "命令不能为空",
                attempted="执行空 shell 命令",
                do_not_repeat="不要再次用空 command 调用 exec",
                next_action="提供明确的 shell 命令，或改用 read_file/list_dir 获取信息",
            )

        cwd = working_dir or self.working_dir or os.getcwd()

        # 执行安全检查（黑名单、路径限制）
        guard_error = self._guard_command(command, cwd)
        if guard_error:
            return guard_error  # 如果被拦截，直接返回错误信息

        # 计算实际超时时间（不超过最大上限）
        effective_timeout = min(timeout or self.timeout, self._MAX_TIMEOUT)

        try:
            # 使用 asyncio 异步创建子进程执行 shell 命令
            process = await asyncio.create_subprocess_shell(
                command,                         # 要执行的命令
                stdout=asyncio.subprocess.PIPE,  # 捕获标准输出
                stderr=asyncio.subprocess.PIPE,  # 捕获标准错误
                cwd=cwd,                         # 工作目录
            )

            try:
                # 等待进程完成，同时捕获输出
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),  # communicate() 返回 (stdout, stderr) 元组
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                # 命令执行超时，强制终止进程
                process.kill()
                try:
                    # 等待进程真正结束（最多等 5 秒）
                    await asyncio.wait_for(process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass  # 进程可能已经结束，忽略超时
                return format_tool_error(
                    f"命令执行超时（{effective_timeout} 秒）",
                    attempted=f"在 {cwd} 执行：{command}",
                    observed="进程已被终止，未得到完整输出",
                    do_not_repeat="不要用相同命令和相同超时时间重复执行",
                    next_action="缩小命令范围、增加过滤条件，或改用更具体的文件/搜索工具",
                )

            # 组装输出内容
            output_parts = []
            if stdout:
                # 解码标准输出，替换不可解码字节以避免抛出异常
                output_parts.append(stdout.decode("utf-8", errors="replace"))
            if stderr:
                # 解码标准错误输出
                stderr_text = stderr.decode("utf-8", errors="replace")
                if stderr_text.strip():
                    # 如果有非空的错误输出，添加到结果中
                    output_parts.append(f"标准错误输出：\n{stderr_text}")
            # 添加退出码信息
            output_parts.append(f"\n退出码：{process.returncode}")

            # 合并所有输出部分
            result = "\n".join(output_parts) if output_parts else "（命令没有输出内容）"

            # 如果输出过长，截断中间部分（保留开头和结尾）
            if len(result) > self._MAX_OUTPUT:
                half = self._MAX_OUTPUT // 2  # 计算保留的头尾长度
                result = (
                    result[:half]  # 保留开头
                    + f"\n\n......（已截断 {len(result) - self._MAX_OUTPUT:,} 个字符）......\n\n"
                    + result[-half:]  # 保留结尾
                )

            return result

        except Exception as exc:
            # 捕获其他异常（如命令不存在、权限不足等）
            return format_tool_error(
                f"执行命令失败：{str(exc)}",
                attempted=f"在 {cwd} 执行：{command}",
                do_not_repeat="不要用相同命令重复执行",
                next_action="检查命令是否存在、工作目录是否正确，或改用更小的观察命令定位问题",
            )

    def _guard_command(self, command: str, cwd: str) -> str | None:
        """
        执行前的安全检查。

        拦截策略包括两层：
        1. 匹配危险命令黑名单（阻止 rm -rf、shutdown 等）
        2. 如果启用了工作区限制，则阻止路径越界

        Args:
            command: 要检查的命令字符串
            cwd: 当前工作目录路径

        Returns:
            None 表示检查通过，可以执行
            str 表示被拦截，返回的错误信息
        """
        cmd = command.strip()  # 去除首尾空白
        lower = cmd.lower()  # 转为小写（便于匹配，不区分大小写）

        # ========== 第一层：黑名单检查 ==========
        for pattern in self.deny_patterns:
            if re.search(pattern, lower):
                # 如果匹配到危险命令模式，拦截并返回错误
                return format_tool_error(
                    "命令被安全策略拦截，检测到高风险模式",
                    attempted=f"在 {cwd} 执行：{command}",
                    observed=f"匹配危险命令规则：{pattern}",
                    do_not_repeat="不要继续尝试执行相同或等价的高风险命令",
                    next_action="改用只读观察命令，或使用受控的文件工具完成必要操作",
                )

        # ========== 第二层：路径限制检查 ==========
        if self.restrict_to_workspace:
            # 检查路径穿越攻击（如 ../../）
            # ../：Linux/macOS 系统的上级目录写法
            # ..\\：Windows 系统的上级目录写法（双反斜杠是 Python 转义写法）
            if "..\\" in cmd or "../" in cmd:
                return format_tool_error(
                    "命令被安全策略拦截，检测到路径穿越",
                    attempted=f"在 {cwd} 执行：{command}",
                    observed="命令中包含 ../ 或 ..\\",
                    do_not_repeat="不要继续用路径穿越访问工作区外内容",
                    next_action="把目标路径改成工作区内路径，或先用 list_dir 确认可访问目录",
                )

            # 获取工作目录的绝对路径
            cwd_path = Path(cwd).resolve()

            # 从命令中提取所有绝对路径，检查是否越界
            for raw in self._extract_absolute_paths(cmd):
                try:
                    # 展开环境变量（如 $HOME）和用户目录符号（如 ~）
                    expanded = Path.expanduser(Path(raw.strip()))
                    path = Path(expanded).expanduser().resolve()
                except Exception:
                    continue  # 路径解析失败，跳过检查

                # 检查路径是否在工作区外
                # path.parents 包含所有父目录，如果 cwd_path 不在其中，说明越界
                if path.is_absolute() and (
                    cwd_path not in path.parents and path != cwd_path
                ):
                    return format_tool_error(
                        "命令被安全策略拦截，访问路径超出了当前工作目录",
                        attempted=f"在 {cwd} 执行：{command}",
                        observed=f"越界路径：{path}；当前工作目录：{cwd_path}",
                        do_not_repeat="不要继续访问工作区外绝对路径",
                        next_action="改用工作区内路径，或先 list_dir 当前工作目录确认可访问文件",
                    )

        return None  # 所有检查通过，允许执行

    @staticmethod
    def _extract_absolute_paths(command: str) -> list[str]:
        r"""
        从命令字符串中提取绝对路径，供路径越界检查使用。

        支持三种路径格式：
        1. Windows 风格：C:\path\to\file
        2. POSIX 风格：/path/to/file
        3. 用户目录风格：~/path/to/file

        Args:
            command: 命令字符串

        Returns:
            提取到的绝对路径列表
        """
        # 提取 Windows 风格路径（如 C:\Users\admin\file.txt）
        win_paths = re.findall(r"[A-Za-z]:\\[^\s\"'|><;]+", command)

        # 提取 POSIX 风格路径（如 /home/user/file.txt）
        # 匹配以 / 开头的路径，排除引号和特殊字符内的内容
        posix_paths = re.findall(r"(?:^|[\s|>'\"])(/[^\s\"'>;|<]+)", command)

        # 提取用户目录风格路径（如 ~/Documents/file.txt）
        # ~ 符号在 POSIX 系统中表示用户主目录
        home_paths = re.findall(r"(?:^|[\s|>'\"])(~[^\s\"'>;|<]*)", command)

        # 返回所有提取到的路径
        return win_paths + posix_paths + home_paths
