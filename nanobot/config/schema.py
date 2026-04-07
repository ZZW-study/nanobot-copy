"""配置模式（Pydantic schema）定义。

本模块使用 Pydantic 定义 nanobot 的配置结构与默认值，
并通过 `Config` 提供统一的配置加载/校验接口。

Pydantic 是一个数据校验库，它能：
1. 定义数据结构（字段名、类型、默认值）
2. 自动校验传入的数据是否符合规范
3. 支持环境变量覆盖默认值
4. 支持驼峰和下划线两种键名风格
"""

from typing import Literal  # Literal 用于限定变量只能是几个固定值之一
from pathlib import Path  # 面向对象的文件路径处理

from pydantic import BaseModel, ConfigDict, Field  # Pydantic 的核心类
from pydantic.alias_generators import to_camel  # 将下划线命名转为驼峰命名的工具函数
from pydantic_settings import BaseSettings  # 支持从环境变量读取配置的基类


class Base(BaseModel):
    """配置基类：支持驼峰/下划线两种键名风格。

    例如配置文件中写 api_base（下划线）或 apiBase（驼峰）都能识别。
    """
    # model_config 是 Pydantic 的配置字典：
    # alias_generator=to_camel 自动将下划线字段转为驼峰别名
    # populate_by_name=True 允许同时用原名和别名赋值
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProviderConfig(Base):
    """单个 LLM 提供商的配置。

    每个 LLM 提供商（如 OpenRouter、DeepSeek 等）需要以下配置：
    - api_key: 访问该服务商的 API 密钥
    - api_base: 可选的 API 地址（用于代理或本地部署）
    """
    api_key: str = ""  # API 密钥，默认为空（需在配置文件中填写）
    api_base: str | None = None  # 可选的 API 地址


class ProvidersConfig(Base):
    """所有 LLM 提供商的集合配置。

    这里列出了所有支持的 LLM 提供商，每个都有独立的 api_key 和 api_base。
    Field(default_factory=...) 表示每次创建新实例时生成一个新的默认对象。
    """
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)  # OpenRouter 网关
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)    # DeepSeek
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)   # 阿里通义千问
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig) # 硅基流动


class WebSearchConfig(Base):
    """网页搜索配置。

    用于配置网络搜索的参数，支持 Brave 和 Tavily 两个搜索提供商。
    """
    provider: str = "brave"       # 搜索提供商名称
    api_key: str = ""             # 搜索 API 密钥
    base_url: str | None = None   # 可选的 API 地址
    max_results: int = 5          # 最多返回几条搜索结果


class WebToolsConfig(Base):
    """网页工具配置。

    包含网络搜索配置和 HTTP 代理配置。
    """
    proxy: str | None = None                    # HTTP 代理地址
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)  # 搜索配置


class ExecToolConfig(Base):
    """Shell 命令执行工具配置。

    用于配置 AI 执行系统命令时的参数。
    """
    timeout: int = 60        # 命令执行超时时间（秒）
    path_append: str = ""    # 追加到系统 PATH 的额外路径


class MCPServerConfig(Base):
    """MCP 服务器连接配置。

    MCP（Model Context Protocol）是一种协议，允许 AI 连接外部服务。
    此配置定义了如何连接 MCP 服务器。
    """
    type: Literal["stdio", "sse", "streamableHttp"] | None = None  # 连接类型
    command: str = ""           # 启动命令（stdio 模式）
    args: list[str] = Field(default_factory=list)  # 命令参数
    env: dict[str, str] = Field(default_factory=dict)  # 环境变量
    url: str = ""              # 服务器 URL（sse/http 模式）
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP 请求头
    tool_timeout: int = 30     # 工具调用超时时间（秒）

    enabled_tools: list[str] = Field(default_factory=lambda: ["*"])  # 启用的工具列表，* 表示全部


class ToolsConfig(Base):
    """所有工具的全局配置。

    汇总了网页工具、命令执行工具、工作区限制和 MCP 服务器的配置。
    """
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)    # 网页工具配置
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)   # 命令执行配置
    restrict_to_workspace: bool = False  # 是否限制工具只访问工作区内的文件
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)  # MCP 服务器字典


class Config(BaseSettings):
    """nanobot 根配置。

    这是整个系统的核心配置类，汇总了所有配置项。
    BaseSettings 支持从环境变量自动读取配置（以 NANOBOT_ 为前缀）。
    """
    # Agent 默认配置（扁平化，不再嵌套）
    workspace: str = "~/.nanobot/workspace"    # 工作区路径
    model: str = "anthropic/claude-opus-4-5"   # 使用的模型名称
    provider: str = "openrouter"               # LLM 提供商
    max_tokens: int = 8192                     # 模型最大输出 token 数
    temperature: float = 0.1                   # 采样温度（越低越确定，越高越随机）
    max_tool_iterations: int = 40              # 工具调用最大迭代次数
    memory_window: int = 100                   # 记忆窗口大小（保留多少条历史消息）
    reasoning_effort: str | None = None        # 推理强度参数

    # 提供商和工具配置
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)  # 所有 LLM 提供商
    tools: ToolsConfig = Field(default_factory=ToolsConfig)  # 所有工具配置

    @property
    def workspace_path(self) -> Path:
        """将工作区路径中的 ~ 展开为实际家目录后返回。"""
        return Path(self.workspace).expanduser()

    def get_provider(self, model: str | None = None) -> tuple[ProviderConfig | None, str | None]:
        """获取匹配的 LLM 提供商配置。

        根据传入的模型名称，查找对应的提供商配置。
        查找顺序：
        1. 如果配置中强制指定了 provider，直接返回
        2. 否则根据模型名称的前缀（如 anthropic/）在注册表中匹配

        返回：
            (ProviderConfig 实例, 提供商名称) 或 (None, None) 表示未匹配到
        """
        from nanobot.providers.registry import PROVIDERS  # 导入提供商注册表

        # 优先使用强制指定的提供商
        if self.provider != "auto":
            p = getattr(self.providers, self.provider, None)
            return (p, self.provider) if p else (None, None)

        # 提取模型前缀（如 "anthropic/claude" → "anthropic"）
        model_prefix = model.split("/", 1)[0] if model else ""
        # 遍历注册表，按前缀匹配提供商
        for spec in PROVIDERS:
            if model_prefix == spec.name:
                p = getattr(self.providers, model_prefix, None)
                return p, model_prefix

        return None, None  # 未匹配到任何提供商

    # 环境变量配置：
    # env_prefix="NANOBOT_"  读取以 NANOBOT_ 开头的环境变量
    # env_nested_delimiter="__"  嵌套字段用双下划线分隔（如 NANOBOT__MODEL）
    model_config = ConfigDict(
        env_prefix="NANOBOT_",
        env_nested_delimiter="__"
    )
