from typing import Literal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings


class Base(BaseModel):
    """配置基类：支持驼峰/下划线两种键名"""
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProviderConfig(Base):
    """单个 LLM 提供商配置"""
    api_key: str = ""
    api_base: str | None = None


class ProvidersConfig(Base):
    """所有 LLM 提供商配置"""
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)


class WebSearchConfig(Base):
    """网页搜索配置"""
    provider: str = "brave"
    api_key: str = ""
    base_url: str | None = None
    max_results: int = 5


class WebToolsConfig(Base):
    """网页工具配置"""
    proxy: str | None = None
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)


class ExecToolConfig(Base):
    """Shell 执行工具配置"""
    timeout: int = 60
    path_append: str = ""


class MCPServerConfig(Base):
    """MCP 服务器连接配置"""
    type: Literal["stdio", "sse", "streamableHttp"] | None = None
    command: str = ""
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str = ""
    headers: dict[str, str] = Field(default_factory=dict)
    tool_timeout: int = 30


class ToolsConfig(Base):
    """工具配置"""
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)


class Config(BaseSettings):
    """nanobot 根配置"""
    # Agent 默认配置（扁平化，不再嵌套）
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"
    provider: str = "openrouter"
    max_tokens: int = 8192
    temperature: float = 0.1
    max_tool_iterations: int = 40
    memory_window: int = 100
    reasoning_effort: str | None = None

    # 提供商和工具配置
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) -> Path:
        """扩展工作目录路径"""
        return Path(self.workspace).expanduser()

    def get_provider(self, model: str | None = None) -> tuple[ProviderConfig | None, str | None]:
        """获取匹配的 LLM 提供商"""
        from nanobot.providers.registry import PROVIDERS

        # 优先使用强制指定的提供商
        if self.provider != "auto":
            p = getattr(self.providers, self.provider, None)
            return (p, self.provider) if p else (None, None)

        # 模型前缀匹配
        model_prefix = model.split("/", 1)[0] if model else ""
        for spec in PROVIDERS:
            if model_prefix == spec.name:
                p = getattr(self.providers, model_prefix, None)
                return p, model_prefix

        return None, None

    model_config = ConfigDict(
        env_prefix="NANOBOT_",
        env_nested_delimiter="__"
    )
