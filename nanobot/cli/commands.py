"""nanobot 命令行入口。

本模块是 nanobot 的 CLI（命令行界面）入口，
使用 Typer 框架构建命令行工具，提供以下子命令：
1. `nanobot agent`: 与 AI 智能体对话（单次或交互模式）
2. `nanobot onboard`: 初始化配置和工作区
3. `nanobot status`: 查看配置状态
"""

from __future__ import annotations

import asyncio  # 异步编程支持，用于并发执行异步任务
import os  # 操作系统接口，用于环境变量、终端配置等
import select  # I/O 多路复用，用于检测终端输入
import signal  # 信号处理，用于优雅退出程序
import sys  # 系统特定参数和函数（标准输入输出等）
from typing import Optional  # 可选类型注解

import typer  # 命令行框架，用于构建 CLI
from prompt_toolkit import PromptSession  # 高级终端输入，支持历史记录、补全等
from prompt_toolkit.formatted_text import HTML  # 支持 HTML 格式的提示文本
from prompt_toolkit.history import FileHistory  # 将输入历史保存到文件
from prompt_toolkit.patch_stdout import patch_stdout  # 修复异步输出与终端输入的冲突
from rich.console import Console  # 富文本终端输出
from rich.markdown import Markdown  # Markdown 渲染
from rich.text import Text  # 纯文本输出

from nanobot import __logo__, __version__  # 版本号与 Logo
from nanobot.config.paths import get_workspace_path, get_runtime_subdir  # 路径工具
from nanobot.config.schema import Config  # 配置 schema
from nanobot.utils.helpers import sync_workspace_templates  # 工作区模板同步

# Windows 平台特殊处理：强制使用 UTF-8 编码
# Windows 默认使用 GBK 编码，会导致中文输出乱码
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"  # 设置 Python I/O 编码为 UTF-8
    try:
        # 重新配置标准输入/输出/错误的编码
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass  # 如果配置失败，静默忽略（不影响核心功能）


# 创建 Typer CLI 应用实例
# name: 程序名称；help: 帮助信息；no_args_is_help=True: 无参数时显示帮助
app = typer.Typer(name="nanobot", help="nanobot -- 你的个人 AI 助手", no_args_is_help=True)
# 创建 Rich 控制台输出实例
console = Console()

# 退出指令集合（包含多种常见退出方式）
EXIT_COMMAND = {"exit", "quit", "/exit", "/quit", ":q", "退出", "再见"}

# 全局变量：交互式输入会话（延迟初始化）
_PROMPT_SESSION: PromptSession | None = None
# 全局变量：保存的终端属性（用于程序退出时恢复）
_SAVED_TERM_ATTRS = None


def _flush_pending_tty_input() -> None:
    """清理标准输入中残留的内容。

    在某些情况下（如信号中断后），终端输入缓冲区中可能残留未处理的字符。
    此函数通过三种方式清理缓冲区：
    1. 使用 termios.tcflush（Unix 专用）
    2. 使用 select + os.read 手动读取并丢弃
    3. 都不是则静默忽略
    """
    try:
        fd = sys.stdin.fileno()  # 获取标准输入的文件描述符
        if not os.isatty(fd):  # 如果不是终端设备（如管道输入），则无需清理
            return
    except Exception:
        return  # 获取文件描述符失败，静默忽略

    # 方式1：使用 termios 的 tcflush 清空输入缓冲区（Unix 专用）
    try:
        import termios
        # TCIFLUSH 表示清空未读取的输入数据
        termios.tcflush(fd, termios.TCIFLUSH)
        return  # 成功清理，直接返回
    except Exception:
        pass  # termios 不可用，尝试方式2

    # 方式2：使用 select 检测是否有待读数据，然后手动读取丢弃
    try:
        while True:
            # select 检查是否有数据可读（超时设为 0，非阻塞）
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break  # 没有数据，退出循环
            if not os.read(fd, 4096):  # 读取最多 4096 字节
                break  # 读到 EOF，退出循环
    except Exception:
        return  # 读取失败，静默忽略


def _restore_terminal() -> None:
    """恢复终端原始状态。

    程序在运行时可能修改了终端属性（如关闭回显、启用原始模式等），
    此函数在程序退出前将终端属性恢复为保存的原始状态，
    避免退出后终端处于异常状态（如输入不显示）。
    """
    # 如果没有保存过终端属性，说明无需恢复
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        # TCSADRAIN 表示等待所有输出写入后再恢复属性
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass  # 恢复失败时静默忽略


def _init_prompt_session() -> None:
    """初始化交互式输入会话。

    创建 prompt_toolkit 的 PromptSession 实例，提供：
    1. 输入历史记录（上下键翻阅）
    2. 彩色提示文本
    3. 更好的行编辑体验
    """
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS  # 修改全局变量

    # 保存当前终端属性，以便后续恢复
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass  # Unix 系统专用，Windows 不支持 termios

    # 获取历史记录文件路径并确保父目录存在
    from nanobot.config.paths import get_cli_history_path
    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)  # 递归创建目录

    # 创建 PromptSession 实例
    try:
        _PROMPT_SESSION = PromptSession(
            history=FileHistory(str(history_file)),  # 文件持久化的输入历史
            enable_open_in_editor=False,  # 禁用外部编辑器（防止复杂化）
            multiline=False,  # 不支持多行输入（每行是一条完整消息）
        )
    except Exception as exc:
        # Windows 上 prompt_toolkit 可能不可用，此时设为 None
        if sys.platform == "win32":
            _PROMPT_SESSION = None  # 退回到使用内置的 input()
        else:
            raise exc  # 非 Windows 平台则抛出异常


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """打印 nanobot 的回复到终端。

    使用 Rich 库进行格式化输出：
    - 如果 render_markdown 为 True，则将回复按 Markdown 渲染（标题、代码块等）
    - 否则以纯文本形式输出

    参数：
        response: AI 返回的回复文本
        render_markdown: 是否按 Markdown 格式渲染输出
    """
    content = response or ""  # 确保不为 None
    # 根据参数选择渲染方式：Markdown 渲染或纯文本
    body = Markdown(content) if render_markdown else Text(content)
    console.print()  # 输出一个空行，增加视觉间隔
    console.print(f"[cyan]{__logo__}nanobot[/cyan]")  # 输出带颜色的 nanobot 标识
    console.print(body)  # 输出回复内容
    console.print()  # 再输出一个空行


def _is_exit_command(command: str) -> bool:
    """判断用户输入是否为退出指令。

    通过检查输入是否匹配预定义的退出命令集合来判断。

    参数：
        command: 用户输入的字符串

    返回：
        True 表示是退出指令，False 表示不是
    """
    # 同时检查小写和原始形式，确保各种大小写都能识别
    return command.lower() in EXIT_COMMAND or command in EXIT_COMMAND


def version_callback(value: bool) -> None:
    """处理 --version 参数的回调函数。

    当用户传入 -v 或 --version 时，Typer 会调用此函数。

    参数：
        value: 如果用户传入了 --version，则为 True
    """
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")  # 打印版本信息
        raise typer.Exit()  # 优雅退出程序


async def _read_interactive_input_async() -> str:
    """异步读取用户输入。

    在交互模式下，此函数负责等待并读取用户输入的一行文本。
    如果 prompt_toolkit 可用，使用其 PromptSession 提供丰富的编辑功能；
    否则退回到使用内置的 input() 函数。

    返回：
        用户输入的字符串

    异常：
        当遇到 EOF（文件结束，如 Ctrl+D）时，转为 KeyboardInterrupt
    """
    # 如果 PromptSession 不可用（如 Windows 上的兼容问题），使用内置 input
    if _PROMPT_SESSION is None:
        try:
            # asyncio.to_thread 将阻塞的 input() 放入线程池执行，不阻塞事件循环
            return await asyncio.to_thread(input, "你：")
        except EOFError as exc:
            # EOF（如 Ctrl+D）转为 KeyboardInterrupt（如 Ctrl+C），统一处理
            raise KeyboardInterrupt from exc

    # 使用 prompt_toolkit 的异步输入
    try:
        with patch_stdout():  # 修复异步输出与终端输入的冲突
            # 显示蓝色加粗的 "你：" 提示符，异步等待用户输入
            return await _PROMPT_SESSION.prompt_async(HTML("<b fg='ansiblue'>你：</b> "))
    except EOFError as exc:
        raise KeyboardInterrupt from exc


# CLI 入口回调：当不带子命令运行时触发
# invoke_without_command=True 表示即使没有子命令也执行此函数
@app.callback(invoke_without_command=True)
def main(
    # 版本号选项：-v 或 --version，is_eager=True 表示优先处理此参数
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True)
):
    """nanobot 主入口，默认不带参数时显示帮助信息。"""
    return None


# onboard 子命令：初始化配置和工作区
@app.command()
def onboard():
    """初始化配置文件和工作区。

    首次使用 nanobot 时运行此命令，它会：
    1. 创建默认配置文件（config.json）
    2. 创建工作区目录（含 memory/skills/sessions 等子目录）
    3. 提供后续操作建议
    """
    # 延迟导入配置相关函数
    from nanobot.config.loader import get_path_config, load_config, save_config

    config_path = get_path_config()  # 获取配置文件路径
    if config_path.exists():
        # 已有配置文件，询问用户是否覆盖
        console.print(f"[blue]检测到已有配置文件：{config_path}[/blue]")
        console.print("[bold]y[/bold] = 覆盖现有配置")
        console.print("[bold]N[/bold] = 仅刷新缺失字段")
        if typer.confirm("是否覆盖现有配置？"):
            # 用户选择覆盖：创建全新默认配置
            config = Config()
            save_config(config)
        else:
            # 用户选择不覆盖：加载现有配置并保存（补全缺失字段）
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] 已刷新配置：{config_path}")
    else:
        # 无配置文件：创建全新默认配置
        config = Config()
        save_config(config)
        console.print(f"[green]✓[/green] 已创建配置文件：{config_path}")

    # 准备工作区目录
    workspace = get_workspace_path(str(config.workspace_path))
    console.print(f"[green]✓[/green] 已准备工作区：{workspace}")

    # 同步工作区模板（创建 memory/skills/sessions 等必要目录）
    sync_workspace_templates(workspace=workspace)

    # 打印欢迎信息和后续操作建议
    console.print(f"\n{__logo__} nanobot 已准备就绪！")
    console.print("\n建议下一步：")
    console.print(f"  1. 在 [cyan]{config_path}[/cyan] 中填写 API 密钥")
    console.print("  2. 如果使用 OpenRouter，可在 https://openrouter.ai/keys 获取密钥")
    console.print('  3. 开始对话： [cyan]nanobot agent -m "你好！"[/cyan]')


def _make_provider(config: Config):
    """创建 LLM 提供商实例。

    根据配置文件中选择的模型和提供商，创建对应的 LiteLLMProvider 实例。
    此实例是后续与 AI 大模型通信的核心组件。

    参数：
        config: 已加载的配置对象

    返回：
        LiteLLMProvider 实例

    异常：
        如果无法匹配提供商或未配置 API 密钥，则退出程序
    """
    model = config.model  # 获取配置的模型名称
    # 根据模型名称查找对应的提供商配置
    provider_config, provider_name = config.get_provider(model)

    # 延迟导入（避免循环依赖）
    from nanobot.config.loader import get_path_config
    from nanobot.providers.litellm_provider import LiteLLMProvider

    config_path = get_path_config()  # 获取配置文件路径
    # 检查是否成功匹配到提供商
    if not provider_name or provider_config is None:
        console.print(f"[red]错误：无法为模型 {model} 自动匹配提供商。[/red]")
        console.print("[red]请检查 provider 配置，或改用受支持的模型前缀。[/red]")
        raise typer.Exit(1)  # 退出程序，错误码 1

    # 检查 API 密钥是否已配置
    if not provider_config.api_key:
        console.print(f"[red]错误：尚未配置 {provider_name} 的 API 密钥。[/red]")
        console.print(f"[red]请在配置文件中补充密钥：{config_path}[/red]")
        raise typer.Exit(1)  # 退出程序，错误码 1

    # 创建并返回 LiteLLMProvider 实例
    return LiteLLMProvider(
        api_key=provider_config.api_key,       # API 密钥
        api_base=provider_config.api_base,     # API 地址（可为 None）
        default_model=model,                   # 默认模型名称
        provider_name=provider_name,           # 提供商名称
    )


# agent 子命令：启动与 nanobot 的对话
@app.command()
def agent(
    # --message 或 -m：发送给 AI 的单次消息（单次模式）
    message: Optional[str] = typer.Option(None, "--message", "-m", help="发送给智能体的单次消息"),
    # --session 或 -s：会话 ID，用于区分不同对话
    session_id: str = typer.Option("default", "--session", "-s", help="会话 ID"),
    # --markdown/--no-markdown：控制是否渲染 Markdown 格式输出
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="是否按 Markdown 渲染回复"),
    # --logs/--no-logs：控制是否显示调试日志
    logs: bool = typer.Option(False, "--logs/--no-logs", help="是否显示调试日志"),
):
    """启动与 nanobot 的对话。

    支持两种运行模式：
    1. 单次模式：传入 -m 参数，发送一条消息后等待回复并退出
    2. 交互模式：不带 -m 参数，进入持续对话直到用户输入 exit 或 Ctrl+C
    """
    # 延迟导入运行时依赖模块
    from loguru import logger  # 日志库
    from nanobot.agent.loop import AgentLoop  # AI 智能体核心循环
    from nanobot.config.loader import load_config  # 配置加载
    from nanobot.cron.service import CronService  # 定时任务服务

    # 加载配置文件
    config = load_config()
    # 创建 LLM 提供商实例（用于调用大模型 API）
    provider = _make_provider(config)

    # 初始化定时任务服务，存储路径为 ~/.nanobot/cron/jobs.json
    cron_store_path = get_runtime_subdir("cron") / "jobs.json"
    cron = CronService(cron_store_path)

    # 配置日志：--logs 启用则显示 nanobot 日志，否则静默
    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    # 创建 AgentLoop 实例（AI 智能体运行时的核心）
    agent_loop = AgentLoop(
        provider=provider,                    # LLM 提供商
        workspace=config.workspace_path,      # 工作区目录
        model=config.model,                   # 使用的模型
        temperature=config.temperature,       # 采样温度（控制随机性）
        max_tokens=config.max_tokens,         # 最大输出 token 数
        max_iterations=config.max_tool_iterations,  # 工具调用最大迭代次数
        memory_window=config.memory_window,   # 记忆窗口大小（保留历史条数）
        reasoning_effort=config.reasoning_effort,   # 推理强度参数
        web_search_config=config.tools.web.search,  # 网页搜索配置
        web_proxy=config.tools.web.proxy or None,   # 网页代理
        exec_config=config.tools.exec,        # Shell 执行配置
        cron_service=cron,                    # 定时任务服务
        restrict_to_workspace=config.tools.restrict_to_workspace,  # 是否限制工作区
        mcp_servers=config.tools.mcp_servers, # MCP 服务器配置
    )

    # 思考状态显示上下文（有日志时不显示 spinner）
    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()  # 有日志时不使用 spinner
        # 无日志时显示"nanobot 正在思考..."的动态提示
        return console.status("[dim]nanobot 正在思考...[/dim]", spinner="dots")

    # 进度回调函数：在 CLI 中显示工具调用进度
    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        prefix = "正在调用工具：" if tool_hint else "进度："
        console.print(f"[dim]↳ {prefix}{content}[/dim]")

    # ========== 单次模式：传入 -m 参数 ==========
    if message:
        async def run_once() -> None:
            """执行单次对话：发送消息 → 等待回复 → 打印 → 退出"""
            with _thinking_ctx():  # 显示思考状态
                # 处理用户消息并获取 AI 回复
                response = await agent_loop.process_direct(
                    message,
                    session_id,
                    on_progress=_cli_progress  # 进度回调
                )
            # 打印 AI 回复（支持 Markdown 渲染）
            _print_agent_response(response, render_markdown=markdown)
            # 关闭 MCP 连接（如有）
            await agent_loop.close_mcp()

        asyncio.run(run_once())  # 运行异步主函数
        return  # 单次模式执行完毕直接返回

    # ========== 交互模式：持续对话 ==========
    _init_prompt_session()  # 初始化终端输入会话
    console.print(f"{__logo__} 已进入交互模式（输入 [bold]exit[/bold] 或按 [bold]Ctrl+C[/bold] 结束）\n")

    # 信号处理函数：优雅处理中断信号
    def _handle_signal(signum, _frame):
        """处理系统信号（如 Ctrl+C、SIGTERM），恢复终端并退出"""
        sig_name = signal.Signals(signum).name  # 信号名称
        _restore_terminal()  # 恢复终端原始状态
        console.print(f"\n收到信号 {sig_name}，程序退出。")
        sys.exit(0)  # 优雅退出

    # 注册信号处理器
    signal.signal(signal.SIGINT, _handle_signal)   # Ctrl+C
    signal.signal(signal.SIGTERM, _handle_signal)  # kill 命令
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)  # 终端断开连接
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)  # 忽略管道破裂信号

    # 交互模式主循环
    async def run_interactive() -> None:
        """持续读取用户输入 → 发送给 AI → 打印回复，直到用户退出"""
        try:
            while True:  # 无限循环，直到用户输入 exit 或中断
                try:
                    _flush_pending_tty_input()  # 清理终端残留输入
                    user_input = await _read_interactive_input_async()  # 读取用户输入
                    command = user_input.strip()  # 去除首尾空白

                    # 空输入则跳过
                    if not command:
                        continue
                    # 检查是否为退出指令
                    if _is_exit_command(command):
                        _restore_terminal()
                        console.print("\n再见！")
                        break

                    # 处理用户消息并获取 AI 回复
                    with _thinking_ctx():
                        response = await agent_loop.process_direct(
                            command,
                            session_id,
                            on_progress=_cli_progress
                        )

                    # 打印 AI 回复
                    _print_agent_response(response, render_markdown=markdown)

                except KeyboardInterrupt:
                    # Ctrl+C 中断
                    _restore_terminal()
                    console.print("\n再见！")
                    break
                except EOFError:
                    # Ctrl+D 或管道结束
                    _restore_terminal()
                    console.print("\n再见！")
                    break
        finally:
            # 无论是否正常退出，都关闭 MCP 连接
            await agent_loop.close_mcp()

    asyncio.run(run_interactive())  # 启动交互循环


# status 子命令：查看当前配置状态
@app.command()
def status():
    """查看当前配置状态。

    此命令用于诊断 nanobot 的配置情况，显示：
    1. 配置文件是否存在
    2. 工作区目录是否存在
    3. 当前使用的模型
    4. 各 LLM 提供商的 API 密钥/地址配置状态
    """
    # 延迟导入配置相关函数
    from nanobot.config.loader import get_path_config, load_config

    config_path = get_path_config()  # 获取配置文件路径
    config = load_config()  # 加载配置对象
    workspace = config.workspace_path  # 工作区路径

    # 打印状态标题
    console.print(f"{__logo__} nanobot 状态\n")

    # 显示配置文件状态（存在✓ / 不存在✗）
    console.print(f"配置文件：{config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    # 显示工作区目录状态
    console.print(f"工作区：{workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    # 如果配置文件存在，显示更多详细信息
    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS  # 导入提供商注册表

        # 显示当前使用的模型名称
        console.print(f"当前模型：{config.model}")

        # 遍历所有注册的提供商，显示其配置状态
        for spec in PROVIDERS:
            provider = getattr(config.providers, spec.name, None)
            if provider is None:
                continue  # 跳过不存在的提供商

            # 网关类提供商（如 OpenRouter）显示 API 地址
            if spec.is_gateway:
                if provider.api_base:
                    console.print(f"{spec.display_name}： [green]✓ {provider.api_base}[/green]")
                else:
                    console.print(f"{spec.display_name}： [dim]未设置[/dim]")
            else:
                # 标准厂商只显示 API 密钥是否已配置
                has_key = bool(provider.api_key)
                console.print(f"{spec.display_name}： {'[green]✓[/green]' if has_key else '[dim]未设置[/dim]'}")


# 程序入口：当此文件被直接运行时执行
if __name__ == "__main__":
    app()  # 启动 Typer CLI 应用
