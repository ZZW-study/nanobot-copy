"""MCP 客户端核心模块。
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
"""
import asyncio
from contextlib import AsyncExitStack
from typing import Any

# ============================================================================
# HTTP 客户端库：用于建立与服务器的网络连接
# ============================================================================
# httpx 是一个支持同步和异步的 HTTP 客户端库
# 这里主要使用其异步功能 (httpx.AsyncClient) 来发起网络请求
# 作用：SSE 和流式 HTTP 模式都需要创建 HTTP 客户端连接服务器
import httpx

from loguru import logger
from ZBot.agent.tools.base import Tool
from ZBot.agent.tools.registry import ToolRegistry
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp import ClientSession

class MCPToolWrapper(Tool):
    """
    MCP 工具包装器 (适配器模式实现)。
    设计目的:
        外部 MCP 服务器提供的是原生的 MCP 格式工具，而 ZBot 框架需要使用
        自己定义的 Tool 接口。此类的责任就是做"翻译"——把 MCP 工具转换成
        框架能识别的原生工具。
    """
    def __init__(self, session: ClientSession, server_name: str, tool_def, tool_timeout: int = 30):
        """
        初始化 MCP 工具包装器实例。
        """
        self._session = session                                                         # MCP 会话引用，后续通过它向服务器发送工具调用请求
        self._original_name = tool_def.name                                             # 原始工具名，调用时需要使用这个名字
        # 构建包装后的唯一名称，格式为 mcp_{服务器名}_{原工具名}
        # 这样做的好处：即使两个不同服务器都有叫"get_info"的工具，也不会冲突
        self._name = f"mcp_{server_name}_{tool_def.name}"
        self._description = tool_def.description or tool_def.name                       # 工具的功能描述
        self._parameters = tool_def.inputSchema or {"type": "object", "properties": {}} # MCP 工具的输入参数 Schema(JSON Schema 格式)
        self._tool_timeout = tool_timeout                                               # 超时配置，用于控制工具调用的最长等待时间

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict[str, Any]:
        return self._parameters

    # ========================================================================
    # 工具执行核心方法（AI 调用时触发）
    # ========================================================================

    async def execute(self, **kwargs: Any) -> str:
        """
        异步执行 MCP 远程工具调用。
        这个方法被框架调用时，会向 MCP 服务器发送请求，等待执行结果并返回。
        Execution Flow(执行流程):
            1. 向 MCP 会话发送 call_tool 请求
            2. 等待服务器执行并返回结果（带超时保护）
            3. 解析返回的多块内容
            4. 拼接成单一字符串返回
        """
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
        except asyncio.TimeoutError:
            logger.warning("MCP 工具 '{}' 调用超时（{} 秒）", self._name, self._tool_timeout)
            return f"（MCP 工具调用超时：{self._tool_timeout} 秒）"

        # ---------------------- 异常情况: 其他所有错误 ----------------------
        except Exception as exc:
            logger.exception(
                "MCP 工具 '{}' 执行失败:{}:{}",
                self._name,
                type(exc).__name__,  # 异常类型，如 ValueError、ConnectionError 等
                exc,                 # 异常消息
            )
            return f"（MCP 工具调用失败:{type(exc).__name__}）"

        # ==================== 解析 MCP 工具返回结果 ====================
        # result.content 是一个列表，每元素是一块内容 (ContentBlock)
        parts = []
        for block in result.content:
            # 情况 1: 普通文本内容
            if isinstance(block, types.TextContent):
                parts.append(block.text)
            # 情况 2: 其他类型
            else:
                # 转为字符串表示
                parts.append(str(block))
        return "\n".join(parts) or "（工具没有返回内容）"



# 连接所有 MCP 服务器并注册它们的工具
async def connect_mcp_servers(
    mcp_servers: dict,      # 服务器配置字典 {服务器名：配置对象}
    registry: ToolRegistry, # 工具注册表实例，用于登记新发现的工具
    stack: AsyncExitStack   # 异步资源栈，用于自动清理连接
) -> None:
    """
    批量连接配置的 MCP 服务器并将工具注册到框架。
    这个函数是整个 MCP 集成的入口点，负责：
    1. 遍历所有服务器配置
    2. 根据配置类型建立连接（stdio/sse/streamableHttp）
    3. 与服务器握手并获取工具列表
    4. 包装并注册每个工具
    """
    from mcp import ClientSession, StdioServerParameters

    # SSE 传输模式客户端：用于连接支持 Server-Sent Events 的服务器
    from mcp.client.sse import sse_client
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamable_http_client

    # ==================== 遍历所有服务器配置 ====================
    # 输入格式：{"服务器名 1": config1, "服务器名 2": config2, ...}
    for name, cfg in mcp_servers.items():
        try:
            # ==================== 步骤 1: 确定传输类型 ====================
            transport_type = cfg.type
            # ==================== 步骤 2: 根据传输类型建立连接 ====================
            # ---- 类型 1: stdio 模式----
            if transport_type == "stdio":
                # 构建 stdio 服务器参数对象
                params = StdioServerParameters(
                    command=cfg.command,  # 要执行的命令，如 "npx"
                    args=cfg.args,        # 命令参数，如 ["-y", "@modelcontextprotocol/server-filesystem"]
                    env=cfg.env or None   # 环境变量，None 表示使用系统环境
                )
                read, write = await stack.enter_async_context(stdio_client(params))  # 启动本地子进程

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
                        follow_redirects=True,           # 自动跟随重定向
                        timeout=timeout,                 # 网络超时设置
                        auth=auth,                       # 认证对象（如有）
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
            else:
                logger.warning("MCP 服务器 '{}' 的传输类型 '{}' 无法识别，已跳过", name, transport_type)
                continue

            # ==================== 步骤 3: 初始化 MCP 客户端会话 ====================
            # 创建 ClientSession 实例，绑定之前建立的读写流
            session = await stack.enter_async_context(ClientSession(read, write))
            # 与服务器完成握手流程:
            #   1. 交换版本信息
            #   2. 协商支持的能力
            #   3. 建立正式的连接状态
            await session.initialize()

            # ==================== 步骤 4: 从服务器获取所有可用工具 ====================
            # list_tools() 返回 ToolList 对象，包含 tools 属性（ToolDefinition 列表）
            tools = await session.list_tools()

            registered_count = 0  # 本次成功注册的工具计数

            # ==================== 步骤 5: 逐个包装并注册工具 ====================
            for tool_def in tools.tools:
                # 创建包装器实例：负责后续的工具调用适配
                wrapper = MCPToolWrapper(session, name, tool_def, tool_timeout=cfg.tool_timeout)

                # 注册到框架工具注册表
                # 注册后，AI 模型就能通过 name 属性找到并使用这个工具了
                registry.register(wrapper)
                logger.debug("MCP：已注册服务器 '{}' 提供的工具 '{}'", name, wrapper.name)
                registered_count += 1

            # 连接成功的日志输出
            logger.info("MCP 服务器 '{}' 已连接，注册工具 {} 个", name, registered_count)

        # ==================== 服务器连接失败处理 ====================
        except Exception as e:
            # 记录错误但不中断：一个服务器失败不影响其他服务器
            logger.error("MCP 服务器 '{}' 连接失败:{}", name, e)