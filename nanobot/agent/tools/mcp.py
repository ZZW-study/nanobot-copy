"""MCP 客户端核心模块。

功能概述:
    连接外部 MCP(Model Context Protocol) 服务器，将服务器提供的工具包装成
    Nanobot 框架的原生工具，使 AI 模型能够调用这些外部工具。

MCP 协议简介:
    Model Context Protocol(模型上下文协议) 是一个标准化协议，用于 AI 模型
    与外部服务/工具进行安全通信。它允许模型通过统一接口访问各种资源。

支持的连接方式:
    1. stdio(标准输入输出): 启动本地子进程与 MCP 服务器通信
       - 适用于本地工具和服务
       - 通过命令行启动服务器进程

    2. sse(server-sent events, 服务器推送事件): 通过 HTTP 长连接接收服务端推送
       - 适用于远程 SSE 服务器
       - URL 通常以/sse结尾

    3. streamableHttp(流式 HTTP): 通过 HTTP 流式传输进行双向通信
       - 适用于大多数现代 MCP 服务
       - 更灵活的数据传输方式

类结构:
    MCPToolWrapper: 将外部 MCP 工具包装为框架原生工具的适配器类
    connect_mcp_servers: 连接所有配置的 MCP 服务器并注册工具的协程函数

用法示例:
    async def main():
        registry = ToolRegistry()
        stack = AsyncExitStack()
        servers = {
            "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"]}
        }
        await connect_mcp_servers(servers, registry, stack)

    # 之后 AI 模型就可以使用 mcp_filesystem_XXX 命名的工具了
"""

import asyncio
# ============================================================================
# 异步上下文管理器栈：用于自动管理多个异步资源的生命周期
# ============================================================================
# 作用：确保资源（如网络连接、文件句柄）在使用完毕后自动关闭
# 用法：进入 async with AsyncExitStack() as stack: 上下文中
#      通过 stack.enter_async_context() 注册异步资源
# 示例:
#   async with AsyncExitStack() as stack:
#       reader, writer = await stack.enter_async_context(some_connection())
from contextlib import AsyncExitStack

from typing import Any

# ============================================================================
# HTTP 客户端库：用于建立与服务器的网络连接
# ============================================================================
# httpx 是一个支持同步和异步的 HTTP 客户端库
# 这里主要使用其异步功能 (httpx.AsyncClient) 来发起网络请求
# 作用：SSE 和流式 HTTP 模式都需要创建 HTTP 客户端连接服务器
import httpx

# ============================================================================
# 日志工具：用于记录运行状态、错误信息和调试信息
# ============================================================================
# loguru 是一个现代 Python 日志库，语法简洁且功能强大
# 在 MCP 模块中主要用于：
#   - 记录连接成功/失败信息
#   - 记录工具调用超时或异常
#   - 记录注册的 tool 数量等关键事件
from loguru import logger

# ============================================================================
# 继承框架原生工具基类
# ============================================================================
# 所有可被 AI 调用的工具都必须继承自 Tool 基类
# Tool 基类定义了工具必须实现的属性和方法规范
from nanobot.agent.tools.base import Tool

# ============================================================================
# 工具注册表：集中管理所有可用的工具
# ============================================================================
# ToolRegistry 是一个中央仓库，存储所有已注册的工具
# AI 模型在决定调用哪个工具时，会查询注册表获取可用工具列表
from nanobot.agent.tools.registry import ToolRegistry


class MCPToolWrapper(Tool):
    """
    MCP 工具包装器 (适配器模式实现)。

    设计目的:
        外部 MCP 服务器提供的是原生的 MCP 格式工具，而 Nanobot 框架需要使用
        自己定义的 Tool 接口。此类的责任就是做"翻译"——把 MCP 工具转换成
        框架能识别的原生工具。

    继承关系:
        继承自 Tool 基类，必须实现以下核心成员:
        - name: 工具名称（字符串）
        - description: 工具描述（字符串）
        - parameters: 工具参数 Schema(dict)
        - execute(**kwargs): 执行工具的方法（返回结果字符串）

    工作流程:
        1. __init__: 接收 MCP 原始工具定义，保存必要信息
        2. name/description/parameters: 作为属性暴露给框架
        3. execute: 当 AI 调用工具时，转发到 MCP 服务器执行并返回结果

    命名规则:
        为避免不同 MCP 服务器的工具重名，包装后的工具名格式为:
        mcp_{服务器名称}_{原始工具名}
        例如：mcp_filesystem_read_file → 来自 filesystem 服务器的 read_file 工具
    """

    def __init__(self, session, server_name: str, tool_def, tool_timeout: int = 30):
        """
        初始化 MCP 工具包装器实例。

        Args:
            session: MCP 客户端会话对象
                类型：ClientSession(MCP SDK 提供的类)
                作用：代表与 MCP 服务器的连接通道，用于发送工具调用请求
                生命周期：与服务器连接相同，由 AsyncExitStack 统一管理

            server_name: MCP 服务器的逻辑名称
                类型：str
                作用：用于区分不同的 MCP 服务器来源
                示例："filesystem"(文件系统服务器)、"memory"(记忆服务器)

            tool_def: MCP 原始工具定义对象
                类型：ToolDefinition(MCP SDK 提供的数据类)
                包含的属性:
                - name: 工具名称
                - description: 工具描述
                - inputSchema: 输入参数 Schema(JSON Schema 格式)

            tool_timeout: 工具调用超时时间（可选，默认 30 秒）
                类型：int
                作用：防止工具调用无限期挂起
                注意：不同服务器/工具可能需要不同的超时设置

        Attributes(实例属性):
            _session: MCP 会话对象，用于后续调用工具
            _original_name: 原始工具名称（不改动）
            _name: 包装后的全局唯一名称（mcp_{server}_{original}）
            _description: 工具功能描述
            _parameters: 参数 Schema，AI 用它生成正确的调用参数
            _tool_timeout: 调用超时阈值（秒）

        Example:
            >>> session = ClientSession(read, write)
            >>> tool_def = ToolDefinition(name="read_file", description="读取文件内容", ...)
            >>> wrapper = MCPToolWrapper(session, "filesystem", tool_def, tool_timeout=60)
            >>> wrapper.name
            'mcp_filesystem_read_file'
            >>> wrapper.description
            '读取文件内容'
        """
        # _session: 保存 MCP 会话引用，后续通过它向服务器发送工具调用请求
        self._session = session

        # _original_name: 记录原始工具名，调用时需要使用这个名字（不能改）
        self._original_name = tool_def.name

        # _name: 构建包装后的唯一名称，格式为 mcp_{服务器名}_{原工具名}
        #       这样做的好处：即使两个不同服务器都有叫"get_info"的工具，也不会冲突
        self._name = f"mcp_{server_name}_{tool_def.name}"

        # _description: 工具的功能描述，用于让 AI 了解这个工具是做什么的
        #              如果原始定义有 description 就用它，否则用工具名代替
        self._description = tool_def.description or tool_def.name

        # _parameters: MCP 工具的输入参数 Schema(JSON Schema 格式)
        #             AI 会参考这个 Schema 来构造正确的调用参数
        #             如果原始定义没有，就用一个空的 object Schema
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}}

        # _tool_timeout: 保存超时配置，用于控制工具调用的最长等待时间
        self._tool_timeout = tool_timeout

    # ========================================================================
    # 以下是框架 Tool 基类强制要求的只读属性
    # ========================================================================

    @property
    def name(self) -> str:
        """
        返回包装后的工具唯一名称。

        Returns:
            str: 工具名称，格式为 mcp_{server_name}_{original_name}

        Note:
            这是 AI 模型调用工具时使用的标识符，必须在整个系统中唯一
        """
        return self._name

    @property
    def description(self) -> str:
        """
        返回工具的功能描述文本。

        Returns:
            str: 人类可读的工具用途描述

        Usage:
            AI 模型会读取这个描述来决定是否在特定场景下调用该工具
        """
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        """
        返回工具参数的 JSON Schema 定义。

        Returns:
            dict[str, Any]: JSON Schema 格式的参数字典

        Structure Example:
            {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "要读取的文件路径"}
                },
                "required": ["path"]
            }

        Usage:
            AI 模型会根据这个 Schema 来构造调用参数，确保参数格式正确
        """
        return self._parameters

    # ========================================================================
    # 工具执行核心方法（AI 调用时触发）
    # ========================================================================

    async def execute(self, **kwargs: Any) -> str:
        """
        异步执行 MCP 远程工具调用。

        这个方法被框架调用时，会向 MCP 服务器发送请求，等待执行结果并返回。
        整个过程是异步的，不会阻塞其他任务。

        Args:
            **kwargs: AI 模型传入的工具参数
                键值对形式，键是参数名，值是参数值
                例如：{"path": "/data/file.txt", "encoding": "utf-8"}

        Returns:
            str: 工具执行结果的文本形式

        Execution Flow(执行流程):
            1. 向 MCP 会话发送 call_tool 请求
            2. 等待服务器执行并返回结果（带超时保护）
            3. 解析返回的多块内容
            4. 拼接成单一字符串返回

        Error Handling(异常处理):
            - TimeoutError: 超过指定时间未完成 → 返回超时提示
            - CancelledError: 任务被取消 → 判断原因后处理
            - Other Exception: 其他错误 → 记录日志并返回错误摘要

        Example:
            >>> # AI 决定调用工具
            >>> result = await wrapper.execute(path="/etc/hosts")
            >>> print(result)
            127.0.0.1 localhost
            ::1 localhost
        """
        # ==================== 延迟导入：避免无 MCP 配置时的启动错误 ====================
        # 如果用户根本没配置任何 MCP 服务器，就不需要加载 mcp 相关依赖
        # 这种按需加载策略可以加快应用启动速度并减少不必要的依赖
        from mcp import types

        try:
            # ==================== 执行远程工具调用（带超时保护） ====================
            # asyncio.wait_for 确保工具调用不会无限期挂起
            # 如果超过 self._tool_timeout 秒仍未完成，会自动抛出 TimeoutError
            result = await asyncio.wait_for(
                # self._session.call_tool(...) 向 MCP 服务器发送工具调用请求
                #   - 第一个参数：原始工具名（必须是服务器上的准确名称）
                #   - arguments: 参数字典（AI 模型提供的参数）
                # 返回值：ToolResult 对象，包含 content 字段（多块内容的列表）
                self._session.call_tool(self._original_name, arguments=kwargs),
                timeout=self._tool_timeout,  # 超时阈值（秒）
            )
        # ---------------------- 异常情况 1: 调用超时 ----------------------
        except asyncio.TimeoutError:
            # 记录警告日志：包含工具名称和超时时长，便于排查
            logger.warning("MCP 工具 '{}' 调用超时（{} 秒）", self._name, self._tool_timeout)
            # 返回友好提示：告知用户调用失败了，原因是超时
            return f"（MCP 工具调用超时：{self._tool_timeout} 秒）"

        # ---------------------- 异常情况 2: 任务被取消 ----------------------
        except asyncio.CancelledError:
            # CancelledError 比较特殊，可能是真取消也可能是假取消
            # task.cancelling() > 0 表示是当前任务主动被外部取消（真取消）
            # 如果是真取消，应该重新抛出让上层处理
            task = asyncio.current_task()
            if task is not None and task.cancelling() > 0:
                raise  # 重新抛出，让外层处理取消逻辑

            # 如果是其他原因导致的 CancelledError(如服务器端取消),记录日志
            logger.warning("MCP 工具 '{}' 被服务端或 SDK 取消", self._name)
            return "（MCP 工具调用已被取消）"

        # ---------------------- 异常情况 3: 其他所有错误 ----------------------
        except Exception as exc:
            # logger.exception 会在记录错误消息的同时打印完整堆栈跟踪
            logger.exception(
                "MCP 工具 '{}' 执行失败:{}:{}",
                self._name,
                type(exc).__name__,  # 异常类型，如 ValueError、ConnectionError 等
                exc,                 # 异常消息
            )
            # 返回简短的错误提示，不含敏感技术细节
            return f"（MCP 工具调用失败:{type(exc).__name__}）"

        # ==================== 解析 MCP 工具返回结果 ====================
        # MCP 的设计允许工具返回多种类型的内容（文本、图片、文件等）
        # result.content 是一个列表，每元素是一块内容 (ContentBlock)
        parts = []
        for block in result.content:
            # 情况 1: 普通文本内容
            if isinstance(block, types.TextContent):
                # 直接提取文本部分
                parts.append(block.text)
            # 情况 2: 其他类型（如图片、文件引用等）
            else:
                # 转为字符串表示（如 "<Image>" 或 "<File:/path>"）
                parts.append(str(block))

        # 将所有内容块用换行符拼接成一个完整字符串
        # 如果没有返回任何内容，则返回默认提示信息
        return "\n".join(parts) or "（工具没有返回内容）"


# ================================================================================
# 核心函数：连接所有 MCP 服务器并注册它们的工具
# ================================================================================
async def connect_mcp_servers(
    mcp_servers: dict,     # 服务器配置字典 {服务器名：配置对象}
    registry: ToolRegistry, # 工具注册表实例，用于登记新发现的工具
    stack: AsyncExitStack  # 异步资源栈，用于自动清理连接
) -> None:
    """
    批量连接配置的 MCP 服务器并将工具注册到框架。

    这个函数是整个 MCP 集成的入口点，负责：
    1. 遍历所有服务器配置
    2. 根据配置类型建立连接（stdio/sse/streamableHttp）
    3. 与服务器握手并获取工具列表
    4. 包装并注册每个工具

    Parameters(参数):
        mcp_servers: MCP 服务器配置集合
            格式：dict[服务器名，服务器配置]
            服务器配置应包含:
            - type: 传输类型 ("stdio"/"sse"/"streamableHttp", 可选)
            - command: stdio 模式下要执行的命令
            - args: 命令的参数列表
            - env: 环境变量字典
            - url: HTTP/SSE 模式的服务器地址
            - headers: HTTP 请求头
            - enabled_tools: 启用哪些工具（空列表表示全部，*也表示全部）
            - tool_timeout: 工具调用超时时间

        registry: 框架工具注册表
            注册到的地方，注册后 AI 模型才能发现并使用这些工具

        stack: 异步上下文栈
            用于管理所有连接的生命周期，确保程序退出时正确关闭

    Raises:
        静默失败：任何单个服务器的连接失败都不会中断其他服务器的连接
                  只是记录错误日志并继续尝试下一个服务器

    Execution Flow(执行流程):
        1. 动态导入 MCP SDK 依赖（延迟加载）
        2. 遍历每个服务器配置
        3. 确定传输类型（自动推断或显式配置）
        4. 根据类型建立连接:
           - stdio: 启动本地进程
           - sse: 建立 SSE 长连接
           - streamableHttp: 建立 HTTP 流式连接
        5. 初始化会话并完成握手
        6. 从服务器拉取工具列表
        7. 按配置过滤工具
        8. 包装并注册每个工具
        9. 记录连接结果

    Example:
        async def setup_mcp():
            registry = ToolRegistry()
            stack = AsyncExitStack()

            servers = {
                "fs": {
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-filesystem", "/data"],
                    "enabledTools": ["read_file", "list_directory"]
                },
                "web": {
                    "url": "https://example.com/mcp/sse",
                    "headers": {"Authorization": "Bearer xxx"}
                }
            }

            await connect_mcp_servers(servers, registry, stack)
            # 现在 registry 中有 mcp_fs_read_file, mcp_fs_list_directory 等工具
    """
    # ==================== 延迟导入 MCP SDK（按需加载）====================
    # 好处：如果用户没配置任何 MCP 服务器，就不需要安装/加载 mcp 包
    # 这样可以避免启动时无谓的错误和依赖问题
    from mcp import ClientSession, StdioServerParameters

    # SSE 传输模式客户端：用于连接支持 Server-Sent Events 的服务器
    from mcp.client.sse import sse_client

    # STDIO 传输模式客户端：用于启动本地进程并与之通信
    from mcp.client.stdio import stdio_client

    # Streamable HTTP 传输模式客户端：用于现代 HTTP 流式 MCP 服务
    from mcp.client.streamable_http import streamable_http_client

    # ==================== 遍历所有服务器配置 ====================
    # 输入格式：{"服务器名 1": config1, "服务器名 2": config2, ...}
    for name, cfg in mcp_servers.items():
        try:
            # ==================== 步骤 1: 确定传输类型 ====================
            transport_type = cfg.type
            # 如果配置中没有明确指定类型，自动推断:
            if not transport_type:
                # 有 command → 肯定是 stdio 模式（启动本地进程）
                if cfg.command:
                    transport_type = "stdio"
                # 有 url → 需要在 sse 和 streamableHttp 之间选择
                elif cfg.url:
                    # 约定：URL 以/sse 结尾 => SSE 模式；否则=>流式 HTTP
                    # 例如: https://api.example.com/mcp/sse  → SSE
                    #       https://api.example.com/mcp     → streamableHttp
                    transport_type = (
                        "sse" if cfg.url.rstrip("/").endswith("/sse") else "streamableHttp"
                    )
                # 既没有 command 也没有 url → 无法连接，跳过
                else:
                    logger.warning("MCP 服务器 '{}' 没有配置 command 或 url，已跳过", name)
                    continue

            # ==================== 步骤 2: 根据传输类型建立连接 ====================

            # ---- 类型 1: stdio 模式（启动本地子进程）----
            if transport_type == "stdio":
                # 构建 stdio 服务器参数对象
                params = StdioServerParameters(
                    command=cfg.command,  # 要执行的命令，如 "npx"
                    args=cfg.args,        # 命令参数，如 ["-y", "@modelcontextprotocol/server-filesystem"]
                    env=cfg.env or None   # 环境变量，None 表示使用系统环境
                )
                # 进入异步上下文：自动管理进程管道
                # stdio_client(params) 返回 (read_stream, write_stream) 二元组
                # stack.enter_async_context 会:
                #   1. 启动子进程
                #   2. 创建读写管道
                #   3. 在函数退出时自动终止进程
                # 返回值 read/write 分别对应进程的 stdin/stdout 流
                read, write = await stack.enter_async_context(stdio_client(params))

            # ---- 类型 2: SSE 模式（服务器推送事件）----
            elif transport_type == "sse":
                # 自定义 HTTP 客户端工厂函数
                # 作用：合并用户配置的 headers 和 MCP SDK 自带的 headers
                def httpx_client_factory(
                    headers: dict[str, str] | None = None,
                    timeout: httpx.Timeout | None = None,
                    auth: httpx.Auth | None = None,
                ) -> httpx.AsyncClient:
                    # 合并两层 headers: 先 (cfg.headers)，后(SDK 自带的)
                    merged_headers = {**(cfg.headers or {}), **(headers or {})}
                    # 创建异步 HTTP 客户端实例
                    return httpx.AsyncClient(
                        headers=merged_headers or None,  # HTTP 请求头
                        follow_redirects=True,          # 自动跟随重定向
                        timeout=timeout,                # 网络超时设置
                        auth=auth,                      # 认证对象（如有）
                    )

                # 建立 SSE 连接
                # sse_client(url, httpx_client_factory=...) 返回 (read_stream, write_stream)
                # 在 SSE 模式中:
                #   - read_stream: 接收服务器推送的事件流
                #   - write_stream: 发送 JSON-RPC 请求到服务器
                read, write = await stack.enter_async_context(
                    sse_client(cfg.url, httpx_client_factory=httpx_client_factory)
                )

            # ---- 类型 3: streamableHttp 模式（流式 HTTP）----
            elif transport_type == "streamableHttp":
                # 创建 HTTP 客户端并注册到上下文栈
                # 注意：这里不设超时，避免覆盖后面工具级别的超时控制
                http_client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        headers=cfg.headers or None,  # 请求头
                        follow_redirects=True,        # 自动跟随重定向
                        timeout=None,                 # 无连接级超时
                    )
                )
                # 建立流式 HTTP 连接
                # streamable_http_client 返回 (read_stream, write_stream, close_callback) 三元组
                #   - read_stream: 接收服务器响应流
                #   - write_stream: 发送请求流
                #   - close_callback: 可选的关闭回调
                read, write, _ = await stack.enter_async_context(
                    streamable_http_client(cfg.url, http_client=http_client)
                )

            # ---- 未知类型：无法处理，跳过 ----
            else:
                logger.warning("MCP 服务器 '{}' 的传输类型 '{}' 无法识别，已跳过", name, transport_type)
                continue

            # ==================== 步骤 3: 初始化 MCP 客户端会话 ====================
            # 创建 ClientSession 实例，绑定之前建立的读写流
            # ClientSession 是对底层流的封装，提供了高级 API(initialize/list_tools/call_tool)
            session = await stack.enter_async_context(ClientSession(read, write))
            # 与服务器完成握手流程:
            #   1. 交换版本信息
            #   2. 协商支持的能力
            #   3. 建立正式的连接状态
            await session.initialize()

            # ==================== 步骤 4: 从服务器获取所有可用工具 ====================
            # list_tools() 返回 ToolList 对象，包含 tools 属性（ToolDefinition 列表）
            tools = await session.list_tools()

            # 解析配置中的工具白名单
            enabled_tools = set(cfg.enabled_tools)  # 用户指定的工具列表
            allow_all_tools = "*" in enabled_tools   # 是否启用通配符（全部启用）

            registered_count = 0  # 本次成功注册的工具计数
            matched_enabled_tools: set[str] = set()  # 用于校验：记录匹配到的启用工具

            # 收集服务器提供的所有工具名称（用于日志和错误提示）
            available_raw_names = [tool_def.name for tool_def in tools.tools]
            # 包装后的名称（加上前缀）
            available_wrapped_names = [f"mcp_{name}_{tool_def.name}" for tool_def in tools.tools]

            # ==================== 步骤 5: 逐个包装并注册工具 ====================
            for tool_def in tools.tools:
                wrapped_name = f"mcp_{name}_{tool_def.name}"

                # 工具过滤逻辑:
                # 如果不是"全部启用"模式，且该工具不在白名单中 → 跳过
                if (
                    not allow_all_tools
                    and tool_def.name not in enabled_tools
                    and wrapped_name not in enabled_tools
                ):
                    logger.debug(
                        "MCP：跳过服务器 '{}' 的工具 '{}'（未出现在 enabledTools 中）",
                        name,
                        wrapped_name,
                    )
                    continue

                # 创建包装器实例：负责后续的工具调用适配
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)

                # 注册到框架工具注册表
                # 注册后，AI 模型就能通过 name 属性找到并使用这个工具了
                registry.register(wrapper)
                logger.debug("MCP：已注册服务器 '{}' 提供的工具 '{}'", name, wrapper.name)
                registered_count += 1

                # 记录匹配到的启用工具（用于后续校验）
                if enabled_tools:
                    if tool_def.name in enabled_tools:
                        matched_enabled_tools.add(tool_def.name)
                    if wrapped_name in enabled_tools:
                        matched_enabled_tools.add(wrapped_name)

            # ==================== 步骤 6: 校验配置的工具是否存在 ====================
            # 如果用户指定了 enabled_tools(非*模式),检查是否有工具未被找到
            if enabled_tools and not allow_all_tools:
                # 找出配置中存在但未匹配到的工具
                unmatched_enabled_tools = sorted(enabled_tools - matched_enabled_tools)
                if unmatched_enabled_tools:
                    logger.warning(
                        "MCP 服务器 '{}' 中，enabledTools 指定的这些工具未找到:{}。原始工具名:{}。包装后工具名:{}" ,
                        name,
                        ", ".join(unmatched_enabled_tools),
                        ", ".join(available_raw_names) or "（无）",
                        ", ".join(available_wrapped_names) or "（无）",
                    )

            # 连接成功的日志输出
            logger.info("MCP 服务器 '{}' 已连接，注册工具 {} 个", name, registered_count)

        # ==================== 服务器连接失败处理 ====================
        except Exception as e:
            # 记录错误但不中断：一个服务器失败不影响其他服务器
            logger.error("MCP 服务器 '{}' 连接失败:{}", name, e)
