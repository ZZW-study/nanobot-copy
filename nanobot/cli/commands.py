"""为nanobot提供命令行接口命令的模块"""
import sys  # 和程序运行的系统环境、解释器交互
import os  # 专门管系统文件、目录、路径等底层操作：创建 / 删除文件夹、获取文件路径、执行系统命令、读取环境变量等。
import select  #监控多个文件描述符（File Descriptor）的状态变化，实现 I/O 多路复用（I/O Multiplexing）
import asyncio
import signal
import sys
from pathlib import Path
from typing import Any

# 强制让Windows终端的输出使用utf-8编码格式输出
if sys.platform == "win32":

    if sys.stdout.encoding != "utf-8":          # standard out标准输出类
        os.environ["PYTHONENCODING"] = "utf-8"  # environ是一个类字典对象

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8",errors="replace")
    except Exception:
        pass

# 开始设置自己的命令行界面
import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from pathlib import Path 
from nanobot import __version__
from prompt_toolkit import print_formatted_text
from prompt_toolkit import PromptSession
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.application import run_in_terminal
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

from nanobot import __logo__, __version__
from nanobot.config.paths import get_workspace_path
from nanobot.config.schema import Config
from nanobot.utils.helpers import sync_workspace_templates

app = typer.Typer(name="nanobot",help=f"nanobot -- 你的个人AI助手",no_args_is_help=True)
console = Console()

# 退出CLI的命令集合
EXIT_COMMAND = {"exit","quit","/exit","/quit",":q"}


# --------------------------------------------------------------------------------------------------------------------------------------
# 用prompt_toolkit实现命令行输入的编辑（比如光标移动修改输入）、粘贴（支持终端内粘贴文本）、历史记录（调取过往输入）、显示（美化输入提示 / 格式）
# --------------------------------------------------------------------------------------------------------------------------------------
_PROMPT_SESSION: PromptSession | None = None
_SAVED_TERM_ATTRS = None  # 原始终端设置，退出时重用

def _flush_pending_tty_input() ->None: # 终端（TTY）：Teletypewriter,
    """当模型输出时，用户打字，会自动丢弃"""
    try:
        fd = sys.stdin.fileno()
        if not os.isatty(fd):  # 看标准输入流是否指向终端,不指向则啥都不用干
            return 
    except Exception:
        return
    
    try:  # 情况输入流缓存
        import termios
        termios.tcflush(fd,termios.TCIFLUSH)
        return
    except Exception:
        pass

    try:
        while True:
            ready, _, _ = select.select([fd], [], [], 0) # 监控多路输入 
            if not ready:
                break
            if not os.read(fd, 4096):
                break
    except Exception:
        return
    

def _restore_terminal() ->None:
    """当程序退出时，恢复到终端的原始状态"""
    if _SAVED_TERM_ATTRS is None:
        return
    try:
        import termios
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, _SAVED_TERM_ATTRS)
    except Exception:
        pass


def _init_prompt_session() ->None:
    """创建prompt_toolkit的输入会话实例，保存终端原始状态，并配置输入历史持久化
    （输入记录存在文件里，下次运行还能看到）、单行输入等规则，提升命令行输入体验。"""
    global _PROMPT_SESSION,_SAVED_TERM_ATTRS
    # 1.保存终端的原始状态
    try:
        import termios
        _SAVED_TERM_ATTRS = termios.tcgetattr(sys.stdin.fileno())
    except Exception:
        pass
    
    # 2. 配置输入历史文件路径
    from nanobot.config.paths import get_cli_history_path
    history_file = get_cli_history_path()
    history_file.parent.mkdir(parents=True,exist_ok=True)

    # 3.创建PromptSession实例
    _PROMPT_SESSION = PromptSession(
        history=FileHistory(str(history_file)),  # 输入历史保存到文件
        enable_open_in_editor=False,
        multiline=False   # 单行模式：按回车直接提交输入
    )


def _print_agent_response(response: str,render_markdown: bool) ->None:
    """按照固定的样式打印 AI（nanobot）的响应内容，支持 Markdown 渲染，保证终端输出格式统一、美观，提升可读性。"""
    content = response or ""
    # 设置渲染方式
    body = Markdown(content) if render_markdown else Text(content)
    # 开始打印
    console.print()
    console.print(f"[cyan]{__logo__}nanobot[cyan]")  # 打印带样式的nanobot标识：[cyan]是rich库的颜色语法，显示青色的logo+nanobot
    console.print(body)
    console.print()


def _is_exit_command(command: str) ->bool:
    """检查用户输入的命令是否是 “退出指令”（比如 exit、quit、q 等），返回布尔值，用来控制是否结束交互式聊天。"""
    return command.lower() in EXIT_COMMAND


def version_callback(value:bool) ->None:
    """Typer框架的回调函数，用于处理--version/-v等版本号参数,当用户在命令行输入 `程序名 --version` 时，Typer会调用此函数"""
    if value:
        console.print(f"{__logo__} nanobot v{__version__}")
        raise typer.Exit()
    

async def _read_interactive_input_async() ->str:
    """异步读取带样式的用户交互式输入"""
    if _PROMPT_SESSION is None:
        raise RuntimeError("先输入 _init_prompt_session() 进行初始化")
    
    try:
        # 确保输入提示和程序其他输出不会互相干扰、出现乱码/重叠
        with patch_stdout():  # 临时劫持程序的标准输出
            user_input = await _PROMPT_SESSION.prompt_async(  # 异步等待用户在终端输入一行文本
                HTML("<b fg='ansiblue'>You:</b> ")
            )
            return user_input
    except EOFError as exc:
        raise KeyboardInterrupt from exc 





@app.callback(invoke_without_command=True)  # 没有命令也可以
def main(version: bool = typer.Option(None,"--version","-v",callback = version_callback,is_eager = True)):  # 优先级最高
    """定义命令行程序的根回调函数（程序默认入口），不管用户输什么 nanobot 命令（包括带参数 / 子命令），都会先经过这个函数。"""
    pass



# CIL初始化配置与 LLM 提供商创建
# command--标记为 CLI 子命令
@app.command()
def onboard():
    """CLI 初始化命令，负责配置文件和工作区的搭建"""
    """检查 / 创建 / 刷新配置文件（config.json）,创建工作区目录,同步工作区模板文件,给出后续操作指引（如添加 API Key、使用聊天命令）"""
    from nanobot.config.loader import get_path_config,load_config,save_config
    from nanobot.config.schema import Config

    # 第一步：处理配置文件
    config_path = get_path_config()
    # 如果配置文件已存在：提供覆盖/刷新选项
    if config_path.exists():
        console.print(f"[blue]配置对象已经存在{config_path}[/blue]")
        console.print("[bold]y[/bold] = 是否覆盖现有配置（存在的配置将会删除）")
        console.print("[bold]N[/bold] = 刷新配置,保留现有配置并添加新字段")
        if typer.confirm("是否覆盖配置？"):  # 在终端里弹出一行交互式提示文字 是否覆盖配置？，等待用户输入 y/n,按y则执行
            config = Config()
            save_config(config)

        else:
            config = load_config()
            save_config(config)
            console.print(f"[green]✓[/green] 配置已经在 {config_path} 刷新 (存在的值保留了)")
    else:  # 不存在
        save_config(Config())
        console.print(f"[green]✓[/green] 在 {config_path}创建了配置")

    # 第二步：创建工作区目录
    workspace = get_workspace_path()
    console.print(f"[green]✓[/green] 在 {workspace} 创建了工作目录")

    # 第三步：同步工作区模板文件（如默认的提示词模板、示例文件等）
    sync_workspace_templates(workspace=workspace)

    # 第四步：输出初始化完成提示和后续操作指引
    console.print(f"\n{__logo__} nanobot 已经准备就绪!")
    console.print("\n下一步:")
    # 指引1：添加API Key到配置文件
    console.print(f"  1. 添加你的 API key 到[cyan]{config_path}")
    console.print(" 你可以从 OpenRouter 获得: https://openrouter.ai/keys")
    # 指引2：使用聊天命令
    console.print("  2. 聊天: [cyan]nanobot agent -m \"Hello!\"[/cyan]")



def _make_provider(config: Config): 
    """根据配置文件中的模型和提供商信息，动态创建对应的 LLM 提供商实例，支持多种主流 LLM 接入方式"""
    # 从配置中获取：默认模型名，找出该模型对应的提供商配置类、提供商的名称
    model = config.agents.defaults.model
    p,provider_name = config.get_provider(model)

    from nanobot.config.loader import get_path_config
    from nanobot.providers.litellm_provider import LiteLLMProvider

    # 参数校验：必须配置API Key
    config_path = get_path_config()
    if not (p and p.api_key):
        console.print("[red]错误: 没有配置API key.[/red]")
        console.print(f"[red]请在此位置 {config_path} 设置API key[/red]")
        raise typer.Exit(1)

    return LiteLLMProvider(
        api_key=p.api_key,
        api_base=p.api_base,
        default_model=model,
        provider_name=provider_name
    )



# 网关服务模块
# ============================================================================
# Gateway / Server （网关/服务模块：nanobot的核心后台服务）
# ============================================================================

# 标记为CLI子命令，用户可通过 `nanobot gateway` 执行该函数启动网关
@app.command()
def gateway(
    # 命令行参数1：网关服务端口，默认18790，支持--port/-p指定
    port: int = typer.Option(18790, "--port", "-p", help="Gateway port"),
    # 命令行参数2：工作区目录，可选，支持--workspace/-w指定（覆盖配置文件中的默认值）
    workspace: str | None = typer.Option(None, "--workspace", "-w", help="Workspace directory"),
    # 命令行参数3：详细输出模式，默认关闭，支持--verbose/-v开启（打印DEBUG级日志）
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """启动nanobot网关服务：整合所有核心模块的后台服务"""
    
    # ===================== 第二步：导入核心模块（延迟导入提升启动速度） =====================
    from nanobot.agent.loop import AgentLoop      # 核心Agent循环：处理LLM推理、工具调用、会话逻辑
    from nanobot.bus.queue import MessageBus      # 消息总线：模块间通信的核心（解耦组件）
    from nanobot.config.loader import load_config  # 加载配置文件
    from nanobot.config.paths import get_cron_dir  # 获取定时任务数据目录
    from nanobot.cron.service import CronService   # 定时任务服务：管理/执行定时任务
    from nanobot.cron.types import CronJob         # 定时任务数据模型
    from nanobot.session.manager import SessionManager      # 会话管理器：管理用户会话状态

    # ===================== 第三步：配置日志（详细模式） =====================
    # 如果开启verbose模式：设置日志级别为DEBUG（打印所有详细日志）
    if verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG)

    # 打印启动提示：显示logo和监听端口
    console.print(f"{__logo__} Starting nanobot gateway on port {port}...")

    # ===================== 第四步：初始化核心配置与基础组件 =====================
    # 加载配置文件（已处理自定义配置的优先级）
    config = load_config()
    # 如果用户指定了工作区：覆盖配置文件中的默认工作区路径
    if workspace:
        config.agents.defaults.workspace = workspace
    sync_workspace_templates(config.workspace_path)
    bus = MessageBus()
    provider = _make_provider(config)
    session_manager = SessionManager(config.workspace_path)

    # ===================== 第五步：初始化定时任务（Cron）服务 =====================
    cron_store_path = get_cron_dir() / "jobs.json"
    # 创建定时任务服务实例（加载已保存的定时任务）
    cron = CronService(cron_store_path)

    # ===================== 第六步：初始化核心Agent实例（业务逻辑核心） =====================
    # AgentLoop是nanobot的核心：整合LLM、工具调用、会话、定时任务等所有能力
    agent = AgentLoop(
        bus=bus,  # 关联消息总线（接收/发送消息）
        provider=provider,  # 关联LLM提供商（负责推理）
        workspace=config.workspace_path,  # 工作区路径（存储会话、任务数据）
        model=config.agents.defaults.model,  # 默认使用的LLM模型
        temperature=config.agents.defaults.temperature,  # LLM温度（控制输出随机性）
        max_tokens=config.agents.defaults.max_tokens,  # LLM最大生成token数
        max_iterations=config.agents.defaults.max_tool_iterations,  # 工具调用最大迭代次数（防止死循环）
        memory_window=config.agents.defaults.memory_window,  # 会话记忆窗口（保留多少轮对话）
        reasoning_effort=config.agents.defaults.reasoning_effort,  # 推理努力程度（影响思考深度）
        brave_api_key=config.tools.web.search.api_key or None,  # 网页搜索工具的API Key
        web_proxy=config.tools.web.proxy or None,  # 网页搜索代理
        exec_config=config.tools.exec,  # 命令执行工具的配置（权限、白名单等）
        cron_service=cron,  # 关联定时任务服务（Agent可创建/管理定时任务）
        restrict_to_workspace=config.tools.restrict_to_workspace,  # 限制工具仅操作工作区（安全）
        session_manager=session_manager,  # 关联会话管理器
        mcp_servers=config.tools.mcp_servers,  # MCP服务器配置（模型控制协议）
    )

    # ===================== 第七步：设置定时任务回调（Cron任务执行逻辑） =====================
    # 定义定时任务触发时的回调函数：通过Agent执行定时任务并推送结果
    async def on_cron_job(job: CronJob) -> str | None:
        """Execute a cron job through the agent.（通过Agent执行定时任务）"""
        from nanobot.agent.tools.cron import CronTool
        # 构造定时任务的提示词：告知Agent这是定时任务触发
        reminder_note = (
            "[Scheduled Task] Timer finished.\n\n"
            f"Task '{job.name}' has been triggered.\n"
            f"Scheduled instruction: {job.payload.message}"
        )

        # 执行防护：防止Agent在处理定时任务时创建新的定时任务（避免递归）
        cron_tool = agent.tools.get("cron")
        cron_token = None
        if isinstance(cron_tool, CronTool):
            # 临时禁用Cron工具的创建能力
            cron_token = cron_tool.set_cron_context(True)
        try:
            # 通过Agent直接处理定时任务（无用户交互，后台执行）
            response = await agent.process_direct(
                reminder_note,  # 定时任务指令
                session_key=f"cron:{job.id}",  # 专属会话Key（隔离定时任务会话）
                channel=job.payload.channel or "cli",  # 任务关联的通道（默认CLI）
                chat_id=job.payload.to or "direct",  # 任务推送的目标Chat ID
            )
        finally:
            # 恢复Cron工具的创建能力（无论执行成功/失败都要恢复）
            if isinstance(cron_tool, CronTool) and cron_token is not None:
                cron_tool.reset_cron_context(cron_token)

        # 如果任务配置了推送目标：将执行结果推送到指定通道/聊天ID
        if job.payload.deliver and job.payload.to and response:
            # 直接打印响应（无消息总线推送）
            console.print(f"[dim]Cron result: {response[:100]}...[/dim]" if len(response) > 100 else f"[dim]Cron result: {response}[/dim]")
        return response
    # 将回调函数绑定到定时任务服务（触发任务时执行）
    cron.on_job = on_cron_job

    # ===================== 第八步：打印启动状态（用户可见的汇总信息） =====================
    # 打印定时任务数量
    cron_status = cron.status()
    if cron_status["jobs"] > 0:
        console.print(f"[green]✓[/green] Cron: {cron_status['jobs']} scheduled jobs")

    # ===================== 第九步：异步启动所有服务（核心运行逻辑） =====================
    async def run():
        try:
            # 启动定时任务服务
            await cron.start()
            # 启动Agent核心循环
            await agent.run()
        except KeyboardInterrupt:  # 捕获Ctrl+C中断
            console.print("\nShutting down...")
        finally:
            # 优雅关闭所有服务（释放资源）
            await agent.close_mcp()  # 关闭MCP服务器连接
            cron.stop()  # 停止定时任务服务
            agent.stop()  # 停止Agent循环

    # 启动异步运行循环（阻塞直到服务停止）
    asyncio.run(run())



# ============================================================================
# 定义 `nanobot agent` 命令 —— 终端直接和 AI 代理交互的命令
# ============================================================================
@app.command()
def agent(
    # 1. 命令行参数：--message / -m，发送给AI的单次消息
    message: str = typer.Option(None, "--message", "-m", help="发送给AI的单次消息"),
    # 2. 会话ID：用来区分不同聊天窗口，默认是 cli:direct（命令行直接对话）
    session_id: str = typer.Option("cli:direct", "--session", "-s", help="会话ID"),
    # 3. 是否用Markdown渲染AI回复（默认开启）
    markdown: bool = typer.Option(True, "--markdown/--no-markdown", help="是否用Markdown渲染AI回复"),
    # 4. 是否显示运行日志（调试用，默认关闭）
    logs: bool = typer.Option(False, "--logs/--no-logs", help="是否显示运行日志"),
):
    """直接与 AI 代理交互（终端聊天核心入口）"""
    from loguru import logger

    from nanobot.agent.loop import AgentLoop
    from nanobot.bus.queue import MessageBus
    from nanobot.config.loader import load_config
    from nanobot.config.paths import get_cron_dir
    from nanobot.cron.service import CronService

    # ===================== 1. 加载配置 & 初始化基础环境 =====================
    config = load_config()
    provider = _make_provider(config)
    bus = MessageBus()
    

    # 创建定时任务服务（AI 可以用定时工具，这里先初始化）
    cron_store_path = get_cron_dir() / "jobs.json"
    cron = CronService(cron_store_path)

    # 日志开关：--logs 就打开日志，否则关闭（避免刷屏）
    if logs:
        logger.enable("nanobot")
    else:
        logger.disable("nanobot")

    # ===================== 2. 创建 AI 核心实例（AgentLoop） =====================
    # AgentLoop = nanobot 的大脑：负责思考、调用工具、记对话、执行逻辑
    agent_loop = AgentLoop(
        bus=bus,                                                    # 绑定消息总线
        provider=provider,                                          # 绑定大模型
        workspace=config.workspace_path,                            # 工作区目录
        model=config.agents.defaults.model,                         # 使用的模型
        temperature=config.agents.defaults.temperature,             # 温度（随机性）
        max_tokens=config.agents.defaults.max_tokens,               # 最大生成字数
        max_iterations=config.agents.defaults.max_tool_iterations,  # 最多调用几次工具
        memory_window=config.agents.defaults.memory_window,         # 记忆窗口（保留几轮对话）
        reasoning_effort=config.agents.defaults.reasoning_effort,   # 思考强度
        brave_api_key=config.tools.web.search.api_key or None,      # 搜索API Key
        web_proxy=config.tools.web.proxy or None,                   # 网络代理
        exec_config=config.tools.exec,                              # 命令执行权限配置
        cron_service=cron,                                          # 绑定定时任务服务
        restrict_to_workspace=config.tools.restrict_to_workspace,   # 是否限制只能操作工作区（安全）
        mcp_servers=config.tools.mcp_servers,                       # MCP 服务配置
    )

    # ===================== 3. UI 交互：思考中动画 & 进度提示 =====================
    # 思考动画：AI 思考时显示 "nanobot is thinking..." 转圈动画
    def _thinking_ctx():
        # 如果开了日志，就不显示动画（避免冲突）
        if logs:
            from contextlib import nullcontext
            return nullcontext()
        # 否则显示转圈加载动画,状态上下文管理器，执行耗时操作时，会在终端持续展示状态提示，操作结束自动消失；
        return console.status("[dim]nanobot is thinking...[/dim]", spinner="dots")

    # AI 进度回调：比如 "正在搜索..." "正在执行命令..." 这种灰色小字提示
    async def _cli_progress(content: str, *, tool_hint: bool = False) -> None:
        # 终端输出灰色小字：↳ xxx
        console.print(f"[dim]↳{content}[/dim]")

    # ===================== 模式A：单次消息模式（传了 -m 参数） =====================
    if message:
        # 只执行一次：发消息 → AI回答 → 退出

        async def run_once():
            # 显示“思考中”动画
            with _thinking_ctx():
                # 直接让AI处理消息（不走消息总线，最快）
                response = await agent_loop.process_direct(message, session_id, on_progress=_cli_progress)
            # 把AI的回答漂亮地打印出来（支持Markdown）
            _print_agent_response(response, render_markdown=markdown)
            # 关闭 MCP 服务连接，释放资源
            await agent_loop.close_mcp()

        # 运行异步函数
        asyncio.run(run_once())

    # ===================== 模式B：交互式聊天模式（没传 -m） =====================
    else:
        # 导入消息类型：用户输入的消息
        from nanobot.bus.events import InboundMessage
        # 初始化终端交互环境
        _init_prompt_session()
        # 打印欢迎语 + Logo
        console.print(f"{__logo__} Interactive mode (type [bold]exit[/bold] or [bold]Ctrl+C[/bold] to quit)\n")

        # 解析会话ID：分成 channel 和 chat_id
        if ":" in session_id:
            cli_channel, cli_chat_id = session_id.split(":", 1)
        else:
            cli_channel, cli_chat_id = "cli", session_id

        # ===================== 信号处理：优雅退出（Ctrl+C / kill） =====================
        def _handle_signal(signum, frame):
            # 收到退出信号（如Ctrl+C），恢复终端并退出
            sig_name = signal.Signals(signum).name
            _restore_terminal()
            console.print(f"\nReceived {sig_name}, goodbye!")
            sys.exit(0)

        # 绑定系统信号：Ctrl+C / 终止 / 挂起
        signal.signal(signal.SIGINT, _handle_signal)
        signal.signal(signal.SIGTERM, _handle_signal)
        if hasattr(signal, 'SIGHUP'):
            signal.signal(signal.SIGHUP, _handle_signal)
        # 忽略管道破裂错误，防止程序闪退
        if hasattr(signal, 'SIGPIPE'):
            signal.signal(signal.SIGPIPE, signal.SIG_IGN)

        # ===================== 交互式聊天主逻辑（异步循环） =====================
        async def run_interactive():
            # 1. 启动 AI 主循环
            bus_task = asyncio.create_task(agent_loop.run())
            # 事件：标记一轮对话是否完成
            turn_done = asyncio.Event()
            turn_done.set()
            # 存储AI本轮回复
            turn_response: list[str] = []

            # 2. 后台任务：持续消费AI的输出消息
            async def _consume_outbound():
                while True:
                    try:
                        # 每隔1秒轮询一次AI的输出
                        msg = await asyncio.wait_for(bus.consume_outbound(), timeout=1.0)
                        # 如果是进度消息（如”正在搜索”）
                        if msg.metadata.get("_progress"):
                            console.print(f"  [dim]↳ {msg.content}[/dim]")
                        # 如果是正式回答
                        elif not turn_done.is_set():
                            if msg.content:
                                turn_response.append(msg.content)
                            turn_done.set()
                        # 主动推送的消息（如定时提醒）
                        elif msg.content:
                            console.print()
                            _print_agent_response(msg.content, render_markdown=markdown)
                    except asyncio.TimeoutError:
                        continue
                    except asyncio.CancelledError:
                        break

            # 启动消息消费任务
            outbound_task = asyncio.create_task(_consume_outbound())

            # 3. 循环读取用户输入
            try:
                while True:
                    try:
                        # 清空残留输入
                        _flush_pending_tty_input()
                        # 异步读取用户输入
                        user_input = await _read_interactive_input_async()
                        command = user_input.strip()

                        # 空输入跳过
                        if not command:
                            continue
                        # 输入 exit 退出
                        if _is_exit_command(command):
                            _restore_terminal()
                            console.print("\nGoodbye!")
                            break

                        # 标记本轮对话开始
                        turn_done.clear()
                        turn_response.clear()

                        # 把用户消息发到消息总线 → AI 收到
                        await bus.publish_inbound(InboundMessage(
                            channel=cli_channel,
                            sender_id="user",
                            chat_id=cli_chat_id,
                            content=user_input,
                        ))

                        # 等待 AI 回答完成
                        with _thinking_ctx():
                            await turn_done.wait()

                        # 打印AI回复
                        if turn_response:
                            _print_agent_response(turn_response[0], render_markdown=markdown)

                    # 捕获 Ctrl+C / EOF 退出
                    except KeyboardInterrupt:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
                    except EOFError:
                        _restore_terminal()
                        console.print("\nGoodbye!")
                        break
            finally:
                # 4. 优雅关闭所有服务
                agent_loop.stop()
                outbound_task.cancel()
                await asyncio.gather(bus_task, outbound_task, return_exceptions=True)
                await agent_loop.close_mcp()

        # 启动交互式循环
        asyncio.run(run_interactive())


# ============================================================================
# 系统状态命令
# ============================================================================
@app.command()
def status():
    """查看 Nanobot 系统状态：配置文件、工作区、模型、API 密钥"""
    from nanobot.config.loader import get_config_path, load_config

    # 加载配置信息
    config_path = get_config_path()
    config = load_config()
    workspace = config.workspace_path

    console.print(f"{__logo__} nanobot Status\n")

    # 输出配置文件状态
    console.print(f"Config: {config_path} {'[green]✓[/green]' if config_path.exists() else '[red]✗[/red]'}")
    # 输出工作区状态
    console.print(f"Workspace: {workspace} {'[green]✓[/green]' if workspace.exists() else '[red]✗[/red]'}")

    # 如果配置存在，输出模型和服务商密钥状态
    if config_path.exists():
        from nanobot.providers.registry import PROVIDERS

        console.print(f"Model: {config.agents.defaults.model}")

        # 遍历所有 AI 服务商，检查密钥/OAuth 状态
        for spec in PROVIDERS:
            p = getattr(config.providers, spec.name, None)
            if p is None:
                continue
            if spec.is_gateway:
                console.print(f"{spec.display_name}: [green]✓ (OAuth)[/green]")
            elif spec.is_gateway:
                # 本地模型显示 API 地址
                if p.api_base:
                    console.print(f"{spec.display_name}: [green]✓ {p.api_base}[/green]")
                else:
                    console.print(f"{spec.display_name}: [dim]not set[/dim]")
            else:
                # 在线模型显示密钥状态
                has_key = bool(p.api_key)
                console.print(f"{spec.display_name}: {'[green]✓[/green]' if has_key else '[dim]not set[/dim]'}")


# ============================================================================
# AI 服务商 OAuth 登录命令组
# ============================================================================
# 创建子命令：provider，管理 AI 服务商认证
provider_app = typer.Typer(help="Manage providers")
app.add_typer(provider_app, name="provider")

# 登录处理器注册表：key=服务商名，value=登录函数
_LOGIN_HANDLERS: dict[str, callable] = {}


def _register_login(name: str):
    """
    【装饰器】注册 OAuth 登录处理器
    :param name: 服务商名称
    """
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
):
    """
    AI 服务商 OAuth 登录
    支持：OpenAI Codex / GitHub Copilot
    """
    from nanobot.providers.registry import PROVIDERS

    # 格式化名称，匹配配置
    key = provider.replace("-", "_")
    # 查找对应的 OAuth 服务商
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    # 获取登录处理器
    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.display_name}[/red]")
        raise typer.Exit(1)

    # 执行登录
    console.print(f"{__logo__} OAuth Login - {spec.display_name}\n")
    handler()


@_register_login("openai_codex")
def _login_openai_codex() -> None:
    """OpenAI Codex 交互式 OAuth 登录"""
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
        token = None
        # 尝试读取已有令牌
        try:
            token = get_token()
        except Exception:
            pass
        # 无令牌则启动交互式登录
        if not (token and token.access):
            console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
            token = login_oauth_interactive(
                print_fn=lambda s: console.print(s),
                prompt_fn=lambda s: typer.prompt(s),
            )
        if not (token and token.access):
            console.print("[red]✗ Authentication failed[/red]")
            raise typer.Exit(1)
        console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)


@_register_login("github_copilot")
def _login_github_copilot() -> None:
    """GitHub Copilot 设备码登录"""
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    # 触发 LiteLLM 设备码认证流程
    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)


# 主入口：运行 CLI 应用
if __name__ == "__main__":
    app()

