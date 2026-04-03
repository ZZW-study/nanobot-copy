from typing import Literal
from pathlib import Path

from pydantic import BaseModel,ConfigDict,Field
from pydantic.alias_generators import to_camel
from pydantic_settings import BaseSettings

class Base(BaseModel):
    """
    所有配置的基类：
    1. 统一配置模型的行为：支持驼峰/下划线两种键名
    2. 后续所有配置类都继承此类，无需重复写配置规则
    """
    model_config = ConfigDict(alias_generator=to_camel,populate_by_name=True)


class AgentDefaults(Base):
    """AI助手（Agent）的默认配置项,定义AI回答的基础规则，比如用哪个模型、回答长度/随机性、工具调用次数等"""
     # 工作目录（默认在用户家目录下的.nanobot/workspace，存放AI生成的文件、缓存等）
    workspace: str = "~/.nanobot/workspace"
    model: str = "anthropic/claude-opus-4-5"   # 必须是供应商/模型名称
    # LLM提供商（auto=自动匹配；也可指定anthropic/openrouter等）
    provider: str = (
        "auto"
    )
    # AI回答的最大令牌数（8192≈6000中文字符，限制回答长度，防止超长回复）
    max_tokens: int = 8192
    # 回答随机性（0=完全固定，1=极度随机，0.1表示AI回答更稳定、可预测）
    temperature: float = 0.1
    # 工具调用最大迭代次数（AI调用工具（如搜索/执行命令）的最大重试/循环次数，40次足够覆盖大部分场景）
    max_tool_iterations: int = 40
    # 记忆窗口大小（AI能记住的对话轮数，100表示能记住最近100轮对话）
    memory_window: int = 100
    # 推理强度（low/medium/high，控制AI思考的深度，None=使用模型默认）
    reasoning_effort: str | None = None


class AgentsConfig(Base):
    """AI助手的总配置类（分层设计，方便扩展）,包裹AgentDefaults，统一管理AI助手的配置"""

     # 默认配置（Field(default_factory=AgentDefaults)：延迟初始化，避免多个实例共享同一默认对象）
    defaults: AgentDefaults = Field(default_factory=AgentDefaults)


class ProviderConfig(Base):
    """单个LLM提供商的配置项,定义对接某一个LLM服务商（如OpenAI/anthropic）的必要参数"""
    api_key: str = ""
    api_base: str | None = None



class ProvidersConfig(Base):
    """
    所有支持的LLM提供商汇总配置
    核心作用：为每一个主流LLM服务商（如OpenAI/通义千问/火山引擎）单独配置参数
    注：每个字段对应一个ProviderConfig，可单独设置不同服务商的API密钥/地址
    """

    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)  # 阿里云通义千问
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)  # 硅基流动


class WebSearchConfig(Base):
    """
    网页搜索工具配置
    核心作用：定义AI调用网页搜索的规则（用哪个API、返回多少结果）
    """
    api_key: str = ""     # Brave Search API密钥（需要自行申请，为空则无法使用搜索功能）
    max_results: int = 5  # 搜索返回的最大结果数（默认5条，避免结果过多）


class WebToolsConfig(Base):
    """
    网页工具总配置
    核心作用：统一管理网页相关工具（搜索、代理）
    """
    proxy: str | None = (
        None         # 代理地址（可选，比如国内访问外网需要：http://127.0.0.1:7890 或 socks5://127.0.0.1:1080）
    )
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)  # 网页搜索配置



class ExecToolConfig(Base):
    """
    Shell执行工具配置
    核心作用：限制AI执行系统命令的规则（超时时间、路径限制）
    """
    timeout: int = 60      # 命令执行超时时间（60秒，防止命令卡死）
    path_append: str = ""  # 追加系统路径（可选，让AI能找到更多可执行命令）


class MCPServerConfig(Base):
    """
    MCP服务器连接配置（stdio/HTTP/SSE）
    核心作用：定义AI对接MCP（Model Context Protocol）服务器的规则（比如对接外部工具/模型）
    """

    # 连接类型（stdio=标准输入输出，sse=服务器推送，streamableHttp=流式HTTP；None=自动检测）
    type: Literal["stdio", "sse", "streamableHttp"] | None = None
    command: str = ""  # Stdio模式：要运行的命令（比如"npx"）
    args: list[str] = Field(default_factory=list)  # Stdio模式：命令参数（比如["nanobot-mcp"]）
    env: dict[str, str] = Field(default_factory=dict)  # Stdio模式：额外环境变量
    url: str = ""  # HTTP/SSE模式：接口地址
    headers: dict[str, str] = Field(default_factory=dict)  # HTTP/SSE模式：自定义请求头
    tool_timeout: int = 30  # 工具调用超时时间（30秒，超时取消调用）



class ToolsConfig(Base):
    """
    所有工具的汇总配置
    核心作用：统一管理AI能调用的所有工具（网页、Shell、MCP）
    """
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)            # 网页工具配置
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)           # Shell执行工具配置
    restrict_to_workspace: bool = False                                    # 是否限制工具仅访问工作目录（False=不限制，True=更安全）
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)  # 多个MCP服务器配置（键=服务器名称）



class Config(BaseSettings):
    """
    nanobot的根配置类（所有配置的入口）
    核心作用：汇总所有子配置，提供获取LLM提供商、API密钥等实用方法
    """
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    @property
    def workspace_path(self) ->Path:
        """
        扩展工作目录路径（把~转换成实际的用户家目录）
        比如~/.nanobot/workspace → /Users/你的用户名/.nanobot/workspace
        """
        return Path(self.agents.defaults.workspace).expanduser()


    def get_provider(self,model: str | None = None) ->str | None:
        """获取匹配到的LLM提供商名称和配置"""
        from nanobot.providers.registry import PROVIDERS

        # 优先使用强制指定的提供商
        forced = self.agents.defaults.provider
        if forced != "auto":
            p = getattr(self.providers,forced,None)
            return (p,forced) if p else (None,None)

        # 模型前缀匹配
        model_prefix = model.split("/",1)[0]
        for spec in PROVIDERS:
            if model_prefix == spec.name:
                p = getattr(self.providers,model_prefix,None)
                return p,model_prefix

        return None,None


    model_config = ConfigDict(
        env_prefix="NANOBOT_",  # 环境变量前缀（如NANOBOT_AGENTS__DEFAULTS__MODEL）
        env_nested_delimiter="__"  # 嵌套配置的分隔符（AgentsConfig.defaults.model → AGENTS__DEFAULTS__MODEL）
    )
