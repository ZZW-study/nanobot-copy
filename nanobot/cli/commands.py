"""nanobot 命令行入口"""

from __future__ import annotations

import asyncio
import os
import select
import signal
import sys
from typing import Optional

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.config.paths import get_workspace_path, get_runtime_subdir
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"
    try:
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


app = typer.Typer(name="nanobot", help="nanobot -- 你的个人 AI 助手", no_args_is_help=True)
console = Console()

EXIT_COMMAND = {"exit", "quit", "/exit", "/quit", ":q", "退出", "再见"}

_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None


def _flush_pending_tty_input() -> None:
    """清理标准输入中残留的内容"""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):
            return
    except Exception:
        return

    try:
        import termios
        termios.tcflush(fd, termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0)
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return


def _restore_terminal() -> None:
    """恢复终端原始状态"""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() -> None:
    """初始化交互式输入会话"""
    global _PROMPT_SESSION, _SAVED_TERM_ATTRS

    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass

    from nanobot.config.paths import get_cli_history_path
    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        _PROMPT_SESSION = PromptSession(
            history=FileHistory(str(history_file)),
            enable_open_in_editor=False,
            multiline=False,
        )
    except Exception as exc:
        if sys.platform == "win32":
            _PROMPT_SESSION = None
        else:
            raise exc


def _print_agent_response(response: str, render_markdown: bool) -> None:
    """打印 nanobot 回复"""
    content = response or ""
    body = Markdown(content) if render_markdown else Text(content)
    console.print()
    console.print(f"[cyan]{__logo__}nanobot[/cyan]")
    console.print(body)
    console.print()


def _is_exit_command(command: str) -> bool:
    """判断是否为退出指令"""
    return command.lower() in EXIT_COMMAND or command in EXIT_COMMAND


def version_callback(value: bool) -> None:
    """处理 --version 参数"""
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()


async def _read_interactive_input_async() -> str:
    """异步读取用户输入"""
    if _PROMPT_SESSION is None:
        try:
            return await asyncio.to_thread(input, "你：")
        except EOFError as exc:
            raise KeyboardInterrupt from exc

    try:
        with patch_stdout():
            return await _PROMPT_SESSION.prompt_async(HTML("<b fg='ansiblue'>你：</b> "))
    except EOFError as exc:
        raise KeyboardInterrupt from exc


@app.callback(invoke_without_command=True)
def main(
    version: bool = typer.Option(None, "--version", "-v", callback=version_callback, is_eager=True)
):
    return None


@app.command()
def onboard():
    """初始化配置文件和工作区"""
    from nanobot.config.loader import get_path_config, load_config, save_config

    config_path = get_path_config()
    if config_path.exists():
        console.print(f"[blue]检测到已有配置文件：{config_path}[/blue]")
        console.print("[bold]y[/bold] = 覆盖现有配置")
        console.print("[bold]N[/bold] = 仅刷新缺失字段")
        if typer.confirm("是否覆盖现有配置？"):
            config = Config()
            save_config(config)
        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] 已刷新配置：{config_path}")
    else:
        config = Config()
        save_config(config)
        console.print(f"[green]✓[/green] 已创建配置文件：{config_path}")

    workspace = get_workspace_path(str(config.workspace_path))
    console.print(f"[green]✓[/green] 已准备工作区：{workspace}")

    sync_workspace_templates(workspace=workspace)

    console.print(f"\n{__logo__} nanobot 已准备就绪！")
    console.print("\n建议下一步：")
    console.print(f"  1. 在 [cyan]{config_path}[/cyan] 中填写 API 密钥")
    console.print("  2. 如果使用 OpenRouter，可在 https://openrouter.ai/keys 获取密钥")
    console.print('  3. 开始对话： [cyan]nanobot agent -m "你好！"[/cyan]')


def _make_provider(config: Config):
    """创建 LLM 提供商实例"""
    model = config.model
    provider_config, provider_name = config.get_provider(model)

    from nanobot.config.loader import get_path_config
    from nanobot.providers.litellm_provider import LiteLLMProvider

    config_path = get_path_config()
    if not provider_name or provider_config is None:
        console.print(f"[red]错误：无法为模型 {model} 自动匹配提供商。[/red]")
        console.print("[red]请检查 provider 配置，或改用受支持的模型前缀。[/red]")
        raise typer.Exit(1)

    if not provider_config.api_key:
        console.print(f"[red]错误：尚未配置 {provider_name} 的 API 密钥。[/red]")
        console.print(f"[red]请在配置文件中补充密钥：{config_path}[/red]")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=provider_config.api_key,
        api_base=provider_config.api_base,
        default_model=model,
        provider_name=provider_name,
    )


@app.command()
def agent(
    message: Optional[str] = typer.Option(None, "--message", "-m", help="发送给智能体的单次消息"),
    session_id: str = typer.Option("default", "--session", "-s", help="会话 ID"),
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="是否按 Markdown 渲染回复"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="是否显示调试日志"),
):
    """启动与 nanobot 的对话

    支持两种模式：
    1. 单次模式：nanobot agent -m "你好"
    2. 交互模式：nanobot agent
    """
    from loguru import logger
    from nanobot.agent.loop import AgentLoop
    from nanobot.config.loader import load_config
    from nanobot.cron.service import CronService

    config = load_config()
    provider = _make_provider(config)

    cron_store_path = get_runtime_subdir("cron") / "jobs.json"
    cron = CronService(cron_store_path)

    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    agent_loop = AgentLoop(
        provider=provider,
        workspace=config.workspace_path,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        max_iterations=config.max_tool_iterations,
        memory_window=config.memory_window,
        reasoning_effort=config.reasoning_effort,
        web_search_config=config.tools.web.search,
        web_proxy=config.tools.web.proxy or None,
        exec_config=config.tools.exec,
        cron_service=cron,
        restrict_to_workspace=config.tools.restrict_to_workspace,
        mcp_servers=config.tools.mcp_servers,
    )

    def _thinking_ctx():
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        return console.status("[dim]nanobot 正在思考...[/dim]", spinner="dots")

    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        prefix = "正在调用工具：" if tool_hint else "进度："
        console.print(f"[dim]↳ {prefix}{content}[/dim]")

    if message:
        async def run_once() -> None:
            with _thinking_ctx():
                response = await agent_loop.process_direct(
                    message,
                    session_id,
                    on_progress=_cli_progress
                )
            _print_agent_response(response, render_markdown=markdown)
            await agent_loop.close_mcp()

        asyncio.run(run_once())
        return

    _init_prompt_session()
    console.print(f"{__logo__} 已进入交互模式（输入 [bold]exit[/bold] 或按 [bold]Ctrl+C[/bold] 结束）\n")

    def _handle_signal(signum, _frame):
        sig_name = signal.Signals(signum).name
        _restore_terminal()
        console.print(f"\n收到信号 {sig_name}，程序退出。")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _handle_signal)
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    async def run_interactive() -> None:
        try:
            while True:
                try:
                    _flush_pending_tty_input()
                    user_input = await _read_interactive_input_async()
                    command = user_input.strip()

                    if not command:
                        continue
                    if _is_exit_command(command):
                        _restore_terminal()
                        console.print("\n再见！")
                        break

                    with _thinking_ctx():
                        response = await agent_loop.process_direct(
                            command,
                            session_id,
                            on_progress=_cli_progress
                        )

                    _print_agent_response(response, render_markdown=markdown)

                except KeyboardInterrupt:
                    _restore_terminal()
                    console.print("\n再见！")
                    break
                except EOFError:
                    _restore_terminal()
                    console.print("\n再见！")
                    break
        finally:
            await agent_loop.close_mcp()

    asyncio.run(run_interactive())


@app.command()
def status():
    """查看当前配置状态"""
    from nanobot.config.loader import get_path_config, load_config

    config_path = get_path_config()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot 状态\n")
    console.print(f"配置文件：{config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    console.print(f"工作区：{workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"当前模型：{config.model}")

        for spec in PROVIDERS:
            provider = getattr(config.providers, spec.name, None)
            if provider is None:
                continue
            if spec.is_gateway:
                if provider.api_base:
                    console.print(f"{spec.display_name}： [green]✓ {provider.api_base}[/green]")
                else:
                    console.print(f"{spec.display_name}： [dim]未设置[/dim]")
            else:
                has_key = bool(provider.api_key)
                console.print(f"{spec.display_name}： {'[green]✓[/green]' if has_key else '[dim]未设置[/dim]'}")


if __name__ == "__main__":
    app()
