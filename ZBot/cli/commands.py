"""ZBot 命令行入口。"""

from __future__ import annotations

import asyncio                                          
import signal                                           # 信号处理，用于优雅退出程序
import sys                                              
from typing import Optional         

import typer                                            
from prompt_toolkit import PromptSession                # 高级终端输入，支持历史记录、补全等
from prompt_toolkit.formatted_text import HTML          # 支持 HTML 格式的提示文本
from prompt_toolkit.history import FileHistory          # 将输入历史保存到文件
from prompt_toolkit.patch_stdout import patch_stdout    # 修复异步输出与终端输入的冲突
from rich.console import Console                        # 富文本终端输出
from rich.markdown import Markdown                      # Markdown 渲染

from ZBot import __logo__, __version__                                   
from ZBot.config.paths import  get_runtime_subdir     
from ZBot.config.schema import Config                                   
from ZBot.utils.helpers import ensure_workspace_dirs                    


# 创建 Typer CLI 应用实例
app = typer.Typer(name="ZBot", help="ZBot -- 你的个人 AI 助手", no_args_is_help=True)
console = Console()                                                         # 创建 Rich 控制台输出实例
EXIT_COMMAND = {"exit", "quit", "/exit", "/quit", ":q", "退出", "再见"}     # 退出指令集合
_PROMPT_SESSION: PromptSession                                


def _init_prompt_session() -> None:
    """初始化交互式输入会话。

    创建 prompt_toolkit 的 PromptSession 实例，提供：
    1. 输入历史记录（上下键翻阅）
    2. 彩色提示文本
    3. 更好的行编辑体验
    """
    global _PROMPT_SESSION              
    # 获取历史记录文件路径
    from ZBot.config.paths import get_cli_history_path
    from ZBot.utils.helpers import ensure_dir

    history_file = get_cli_history_path()
    ensure_dir(history_file.parent)

    # 创建 PromptSession 实例（FileHistory 可自动创建文件，但不会自动创建父目录）
    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),
        multiline=False,
    )


def _print_agent_response(response: str) -> None:
    """打印 ZBot 的回复到终端。"""
    content = response                            
    # 如果回复内容包含 Markdown 语法，则使用 Rich 的 Markdown 渲染
    body = Markdown(content)
    console.print()                                     
    console.print(f"[cyan]{__logo__} ZBot[/cyan]")      
    console.print(body)                                 # 输出回复内容
    console.print()                                    


def _is_exit_command(command: str) -> bool:
    """判断用户输入是否为退出指令。"""
    return command.lower() in EXIT_COMMAND


def version_callback(value: bool) -> None:
    """处理 --version 参数的回调函数。"""
    if value:
        console.print(f"{__logo__} ZBot 版本 [cyan]{__version__}[/cyan]")  
        raise typer.Exit()  


async def _read_interactive_input_async() -> str:
    """异步读取用户输入。
    返回：
        用户输入的字符串
    """
    # 使用 prompt_toolkit 的异步输入,必须要异步，不然我创建的实例，会一直阻塞当前线程，AI无法工作。
    try:
        with patch_stdout():   # 它让输出绕过当前输入行，显示在上方。修复异步输出与终端输入的冲突
            return await _PROMPT_SESSION.prompt_async(HTML("<b fg='ansiblue'>你：</b> "))
    except EOFError as exc:
        raise KeyboardInterrupt from exc


def make_provider(config: Config):
    """创建 LLM 提供商实例。根据配置文件中选择的模型和提供商，创建对应的 LiteLLMProvider 实例。"""
    from ZBot.config.paths import get_config_path
    from ZBot.providers.litellm_provider import LiteLLMProvider
    from pathlib import Path

    config_path: Path = get_config_path()                              

    model: str = config.model                                         # 获取配置的模型名称
    if not model:
        console.print(f"[red]未填写模型名称，请到配置{config_path}中填写模型名称[/red]")
        raise typer.Exit(1)
    
    provider_config, provider_name,is_gateway = config.get_provider(model)  # 根据模型名称查找对应的提供商配置

    
    # 检查是否成功匹配到提供商
    if not provider_name or provider_config is None:
        console.print(f"[red]错误:无法为模型 {model} 自动匹配提供商。[/red]")
        console.print("[red]请检查 provider 配置，将模型名称前缀改为受支持的提供商。[/red]")
        raise typer.Exit(1)             

    # 检查 API 密钥是否已配置
    if not provider_config.api_key:
        console.print(f"[red]错误：尚未配置 {provider_name} 的 API 密钥。[/red]")
        console.print(f"[red]请在配置文件中补充密钥：{config_path}[/red]")
        raise typer.Exit(1)             

    # 检查 API_Base 密钥是否已配置
    if not provider_config.api_base:
        console.print(f"[red]错误：尚未配置 {provider_name} 的 API 地址。[/red]")
        console.print(f"[red]请在配置文件中补充地址：{config_path}[/red]")
        raise typer.Exit(1)             

    # 创建并返回 LiteLLMProvider 实例
    return LiteLLMProvider(
        api_key=provider_config.api_key,      
        api_base=provider_config.api_base,     
        default_model=model.split("/",1)[1] if is_gateway else model,   # 默认模型名称 # type: ignore
        provider_name=provider_name,                                    # 提供商名称 # type: ignore
    )


# CLI 入口回调
@app.callback()
def main(
            # is_eager=True 表示优先处理此参数。单项flag机制。
    version: bool = typer.Option(False, "--version", "-v", callback=version_callback, is_eager=True,help="显示版本信息")
):
    """ZBot 主入口，默认不带参数时显示帮助信息。"""
    pass




# onboard 子命令
@app.command()
def onboard():
    """初始化配置文件和工作区。
    首次使用 ZBot 时运行此命令，它会：
    1. 创建默认配置文件（config.json）
    2. 创建工作区目录（含 memory/skills/sessions 等子目录）
    3. 提供后续操作建议
    """
    from ZBot.config.loader import load_config, save_config
    from ZBot.config.paths import get_config_path

    config_path = get_config_path()  # 获取配置文件路径
    if config_path.exists():
        # 已有配置文件，询问用户是否覆盖
        console.print(f"[blue]检测到已有配置文件：{config_path}[/blue]")
        console.print("[bold]y[/bold] = 创建全新配置来覆盖现有配置")
        console.print("[bold]N[/bold] = 仅刷新缺失字段")
        if typer.confirm("是否覆盖现有配置？"): 
            # 用户选择覆盖：创建全新默认配置
            config = Config()
            save_config(config)
        else:
            # 用户选择不覆盖：加载现有配置并保存（补全缺失字段）
            config = load_config(config_path=config_path)
            if config is None:
                console.print(f"[red]警告：无法加载现有配置，已创建全新默认配置。[/red]")
                config = Config()
            save_config(config)
            console.print(f"[green]✓[/green] 已刷新配置：{config_path}")
    else:
        # 无配置文件：创建全新默认配置
        config = Config()
        save_config(config)
        console.print(f"[green]✓[/green] 已创建配置文件：{config_path}")

    # 准备工作区目录
    workspace = config.workspace_path
    # 创建 memory/skills/sessions 等必要目录
    ensure_workspace_dirs(workspace=workspace)
    console.print(f"[green]✓[/green] 已准备工作区：{workspace}")


    # 打印欢迎信息和后续操作建议
    console.print(f"\n{__logo__} ZBot 已准备就绪！")
    console.print("\n建议下一步：")
    console.print(f"1. 在[cyan]{config_path}[/cyan]中配置模型名称")
    console.print(f"2. 在[cyan]{config_path}[/cyan]中填写 API 密钥和 API 地址")
    console.print("3. 如果使用[cyan]siliconflow[/cyan]，可在[cyan]https://cloud.siliconflow.cn/account/ak[/cyan]上获取密钥")
    console.print('4. 开始对话：[cyan]python -m ZBot agent -m "你好！"[/cyan]')
    console.print("\n[bold yellow]💡 提示：[/bold yellow]使用 [cyan]-s[/cyan] 参数指定会话 ID，可以为不同话题创建独立对话：")
    console.print('   [cyan]python -m ZBot agent -s "work"[/cyan]    → 工作相关对话')
    console.print('   [cyan]python -m ZBot agent -s "study"[/cyan]   → 学习相关对话')
    console.print("   不同会话的历史记录独立存储，模型回答更精准！")





# agent 子命令：启动与 ZBot 的对话
@app.command()
def agent(
    message: Optional[str] = typer.Option(None, "--message", "-m", help="发送给智能体的单次消息"),
    session_name: str = typer.Option("default", "--session", "-s", help="会话名称"),
    logs: bool = typer.Option(False, "--logs/--no-logs", help="是否显示调试日志"),
):
    """启动与 ZBot 的对话。"""

    from loguru import logger  
    from ZBot.agent.core_agent import CoreAgent       
    from ZBot.config.agent_runtime import AgentRuntimeConfig
    from ZBot.config.loader import load_config   
    from ZBot.cron.service import CronService    

    if not logs:
        logger.disable("ZBot")
    
    config = load_config()
    if config is None:
        console.print("[red]错误：无法加载配置文件，请先运行 'python -m ZBot onboard' 来初始化配置。[/red]")
        raise typer.Exit(1)
    # 创建 LLM 提供商实例
    provider = make_provider(config)

    # 定时任务执行回调
    from ZBot.agent.tools.cron import set_cron_context, reset_cron_context

    async def _on_cron_job(job):
        """定时任务触发时的回调函数"""
        token = set_cron_context(True)
        try:
            console.print(f"\n[yellow]⏰ 提醒：{job.message}[/yellow]")
        finally:
            reset_cron_context(token)
    # 初始化定时服务
    cron_store_path = get_runtime_subdir("cron") / "jobs.json"
    cron = CronService(cron_store_path, on_job=_on_cron_job)

    
    runtime_config = AgentRuntimeConfig.from_app_config(
        config=config,
        model=provider.default_model,
    )
    
    # 创建 CoreAgent 实例
    core_agent = CoreAgent(
        provider=provider,                    # LLM 提供商
        runtime_config=runtime_config,        # 从全局配置派生的 Agent 运行时配置
        cron_service=cron,                    # 定时任务服务
    )

    # 思考状态显示上下文
    def _thinking_ctx():
        """返回 CLI 思考状态的显示上下文。"""
        return console.status("[bold green] 🤖 ZBot 正在思考...[/bold green]", spinner="dots") # 返回上下文对象


    # 进度回调函数：在 CLI 中显示工具调用进度（就是打印一下而已，没什么神奇的），* 后面的参数必须使用关键字形式传递
    async def _cli_progress(content: str, *, tool_hint: bool = False, agent_label: str | None = None) -> None:
        """把 Agent 或工具执行进度输出到 CLI。"""
        label = agent_label or "主agent"
        prefix = "正在调用工具：" if tool_hint else "进度："
        console.print(f"[bold green]↳ {label} {prefix}{content}[/bold green]")


    # ========== 单次模式：传入 -m 参数 ==========
    if message:
        async def run_once() -> None:
            """执行单次对话：发送消息 → 等待回复 → 打印 → 退出"""
            # 启动定时任务调度器（恢复之前保存的任务）
            await cron.start()
            with _thinking_ctx():  # 显示思考状态
                # 处理用户消息并获取 AI 回复
                response = await core_agent.process_message(
                    message,
                    session_name,
                    on_progress=_cli_progress
                )
            _print_agent_response(response)
            # 停止调度器并关闭 MCP 连接
            cron.stop()
            await core_agent.close_mcp()
            await core_agent.consolidate_all_session_memory(session_name=session_name)
            await core_agent.consolidate_daily_memory(session_name=session_name) 

        asyncio.run(run_once())     
        return                      # 单次模式执行完毕直接返回

    # ========== 交互模式：持续对话 ==========
    _init_prompt_session()     # 初始化终端输入会话
    console.print(f"{__logo__} 已进入交互模式（输入 [bold]exit[/bold] 或按 [bold]Ctrl+C[/bold] 结束）\n")


    # 交互模式主循环
    async def run_interactive() -> None:
        """持续读取用户输入 → 发送给 AI → 打印回复，直到用户退出"""
        # 启动定时任务调度器
        await cron.start()
        try:
            while True:
                try:
                    user_input = await _read_interactive_input_async()  # 读取用户输入
                    command = user_input.strip() 

                    # 空输入则跳过
                    if not command:
                        continue
                    # 检查是否为退出指令
                    if _is_exit_command(command):
                        import time
                        console.print("\n🥺 别走啊！！再聊会呗！！🥺")  # 会打印换行符，就是自动换行
                        time.sleep(2)
                        console.print("\n😔 你真的要走吗？😢")
                        time.sleep(2)
                        console.print("\n💔 哎...要走的人留不住...下次再聊吧... 😭")
                        break

                    # 处理用户消息并获取 AI 回复
                    with _thinking_ctx():
                        response = await core_agent.process_message(
                            command,
                            session_name,
                            on_progress=_cli_progress
                        )
                    _print_agent_response(response)

                except KeyboardInterrupt:
                    # Ctrl+C 中断
                    console.print("\n再见！")
                    break
                except EOFError:
                    # Ctrl+D 或管道结束
                    console.print("\n再见！")
                    break
                
        finally:
            cron.stop()
            await core_agent.close_mcp()
            await core_agent.consolidate_all_session_memory(session_name=session_name)  # 退出前进行最终的会话归档
            await core_agent.consolidate_daily_memory(session_name=session_name) 

    asyncio.run(run_interactive())  # 启动交互循环

