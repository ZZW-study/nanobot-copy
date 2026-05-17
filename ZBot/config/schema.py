"""
本模块使用 Pydantic 定义 ZBot 的配置结构与默认值，
并通过 `Config` 提供统一的配置加载/校验接口。
"""

from typing import ClassVar, Literal, Optional                          # Literal 用于限定变量只能是几个固定值之一
from pathlib import Path                    

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel      # 将下划线命名转为驼峰命名的工具函数
# 继承它，定义结构化数据
# 你写的自定义类继承 BaseModel 后，就变成了严格的结构化数据模型,本身python的类型注解，只是注解，你传入其他的类型也可以，但是这就会强制！。
# 自动做数据校验
# 不用写一堆 if 判断，自动校验字段类型、必填项、格式（比如邮箱、整数）。
# 自动序列化 / 反序列化
# 轻松把 JSON、字典 转成对象，也能把对象转回字典 / JSON，开发接口、处理数据极方便。
# 极简示例
# # 导入核心类
# from pydantic import BaseModel

# # 自定义数据模型，继承 BaseModel
# class User(BaseModel):
#     name: str  # 必须是字符串
#     age: int   # 必须是整数
#     email: str | None = None  # 可选字段

# # 正确使用
# user = User(name="张三", age=20)
# print(user.dict())  # 转字典输出

# # 错误使用（age 传了字符串）→ 自动报错
# # user = User(name="张三", age="20")

class Base(BaseModel):
    """配置基类"""

    # alias_generator=to_camel 自动将下划线字段转为驼峰别名
    # populate_by_name=True 允许同时用原名和别名赋值
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class ProviderConfig(Base):
    """单个 LLM 提供商的配置。
    """
    api_key: str = ""            # API 密钥
    api_base: str = ""           # API 地址


class ProvidersConfig(Base):
    """所有 LLM 提供商的集合配置。
    """ 
    openrouter: ProviderConfig = Field(default_factory=ProviderConfig)   # 不传参时，每次实例化都新建一个独立的 ProviderConfig 对象，而不是共享同一个。
    deepseek: ProviderConfig = Field(default_factory=ProviderConfig)     
    dashscope: ProviderConfig = Field(default_factory=ProviderConfig)   
    siliconflow: ProviderConfig = Field(default_factory=ProviderConfig)

# 每次 WebSearchConfig() 都会新建一个独立对象，绝对不同：
#   a = WebSearchConfig()
#   b = WebSearchConfig()
#   a is b  →  False（两个独立对象，内存地址不同）
#
# 但内部的 str/int 属性，Python 有缓存机制（interning）：
#   小整数（-5 ~ 256）和字符串会被缓存，不重复创建。
#   a.api_key is b.api_key      →  True（都指向同一个 "" 字符串对象）
#   a.max_results is b.max_results  →  True（都指向同一个 5 整数对象）
#
#   内存示意：
#     a ──→ WebSearchConfig 对象A ──→ api_key ──→ ""  ←── 同一个缓存对象
#     b ──→ WebSearchConfig 对象B ──→ api_key ──┘
#                                   max_results ──→ 5   ←── 同一个缓存对象
#
# 这没有任何危险，因为 str/int 是不可变类型，无法修改对象本身：
#   a.api_key = "new_key"  # 不是修改 "" 这个对象，而是让 a.api_key 指向新对象
#                          # b.api_key 仍然指向 ""，完全不受影响
#
# 若换成可变类型（如 list）才会出问题：
#   class WebSearchConfig:
#       results: list = []       # ❌ 类级别定义，所有实例共享同一个列表
#   a.results.append("x")        # 修改的是列表对象本身
#   print(b.results)             # ["x"] ← b 也被篡改，诡异 bug
#   正确写法：results: list = field(default_factory=list)  # 每次新建独立列表
class WebSearchConfig(Base):
    """网页搜索配置（bocha Search API）。"""
    api_key: str = ""             # 搜索 API 密钥
    max_results: int = 5          # 最多返回几条搜索结果


class WebToolsConfig(Base):
    """网页工具配置。

    包含网络搜索配置和 HTTP 代理配置。
    """
    proxy: str | None = None                                          # HTTP 代理地址
    search: WebSearchConfig = Field(default_factory=WebSearchConfig)  # 搜索配置


class ExecToolConfig(Base):
    """Shell 命令执行工具配置,用于配置 AI 执行系统命令时的参数。
    """
    timeout: int = 60        # 命令执行超时时间（秒）


class MCPServerConfig(Base):
    """MCP 服务器连接配置,此配置定义了如何连接 MCP 服务器。
    """
    type: Literal["stdio", "sse", "streamableHttp"] | None = None    # 连接类型
    command: str = ""                                                # 启动命令（stdio 模式），如 "python"、"node"、"uvx"
    args: list[str] = Field(default_factory=list)                    # 命令参数，与 command 配合使用，示例：command="python", args=["-m", "mcp_server"]，实际执行：python -m mcp_server                                                                                                     
    env: dict[str, str] = Field(default_factory=dict)                # 环境变量
    url: str = ""                                                    # 服务器 URL（sse/http 模式）
    headers: dict[str, str] = Field(default_factory=dict)            # HTTP 请求头
    tool_timeout: int = 30  # 工具调用超时时间（秒）

class ToolsConfig(Base):
    """所有工具的全局配置。
    汇总了网页工具、命令执行工具、工作区限制和 MCP 服务器的配置。
    """
    web: WebToolsConfig = Field(default_factory=WebToolsConfig)             # 网页工具配置
    exec: ExecToolConfig = Field(default_factory=ExecToolConfig)            # 命令执行配置
    mcp_servers: dict[str, MCPServerConfig] = Field(default_factory=dict)   # MCP 服务器字典
    restrict_to_workspace: bool = False                                     # 是否限制工具只访问工作区内的文件


class Config(BaseModel):
    """ZBot 根配置。
    这是整个系统的核心配置类，汇总了所有配置项。
    """
    _instance: ClassVar[Optional["Config"]] = None


    # Agent 默认配置
    workspace: str = "~/.ZBot/workspace"        # 工作区路径
    model: str = ""                             # 使用的模型名称
    provider: str = "auto"                      # LLM 提供商
    max_tokens: int = 4096                      # 模型最大输出 token 数，1 token ≈ 0.5~0.8 个中文字符
    temperature: float = 0.1                    # 采样温度（越低越确定，越高越随机）
    agent_timeout_seconds: int = 3600           # 主 Agent 单轮任务最长运行时间，默认 1 小时
    subagent_timeout_seconds: int = 600         # 子 Agent 单个子任务最长运行时间，默认 10 分钟
    context_compaction_threshold: float = 0.8   # 当前上下文接近模型窗口的比例阈值，超过后触发压缩
    recent_history_token_budget_ratio: float = 0.25  # 最近历史最多占模型上下文窗口的比例
    recent_history_max_tokens: int = 64_000           # 最近历史 token 硬上限，避免大窗口模型默认塞入过多历史
    memory_consolidation_interval: int = 40           # 新增多少条未归档消息后触发会话记忆归档
    session_memory_keep_recent_tokens: int = 16_000   # 会话归档时保留多少最近原文 token 不归档
    reasoning_effort: str | None = None         # 推理强度参数

    # 记忆相关配置
    score_threshold: float = 0.75                # 记忆召回分数阈值
    obsolete_score_threshold : float = 0.5       # 记忆过时分数阈值
    decay_rate: float = 0.12                      # 记忆衰减率
    evolve_score_threshold: float = 1.3          # 记忆进化分数阈值

    providers: ProvidersConfig = Field(default_factory=ProvidersConfig)  # 所有 LLM 提供商
    tools: ToolsConfig = Field(default_factory=ToolsConfig)              # 所有工具配置

    def __new__(cls, *args, **kwargs):
        """确保配置对象在进程内保持单例。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def workspace_path(self) -> Path:
        """将工作区路径中的 ~ 展开为实际家目录后返回。"""
        return Path(self.workspace).expanduser()

    def get_provider(
        self,
        model: str | None = None,
    ) -> tuple[ProviderConfig | None, str | None, bool | None]:
        """获取匹配的 LLM 提供商配置。
        根据传入的模型名称，查找对应的提供商配置,看到底注册表支不支持。
        返回：
            (ProviderConfig 实例, 提供商名称,是否是网关) 或 (None, None) 表示未匹配到
        """
        from ZBot.providers.registry import PROVIDERS, find_by_model, find_gateway  # 导入提供商注册表

        # 优先使用强制指定的提供商
        if self.provider != "auto":
            forced_spec = next((spec for spec in PROVIDERS if spec.name == self.provider), None)
            forced_config = getattr(self.providers, self.provider, None)
            if forced_spec and forced_config:
                return forced_config, forced_spec.name, forced_spec.is_gateway
            return None, None, None

        model = model or self.model
        if not model:
            return None, None, None

        # 提取模型前缀（如 "openrouter/anthropic/claude" → "openrouter"）
        model_prefix = model.split("/", 1)[0] if model else ""
        gateway_spec = find_gateway(model_prefix)
        if gateway_spec:
            gateway_config = getattr(self.providers, gateway_spec.name, None)
            return (gateway_config, gateway_spec.name, True) if gateway_config else (None, None, None)

        std_spec = find_by_model(model)
        if std_spec:
            std_config = getattr(self.providers, std_spec.name, None)
            return (std_config, std_spec.name, False) if std_config else (None, None, None)

        return None, None, None  # 未匹配到任何提供商


    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """校验模型温度参数处于允许范围。"""
        if not 0.0 <= v <= 2.0:
            raise ValueError("temperature 必须在 0.0 ~ 2.0 之间")
        return v

    @field_validator(
        "max_tokens",
        "agent_timeout_seconds",
        "subagent_timeout_seconds",
        "recent_history_max_tokens",
        "memory_consolidation_interval",
        "session_memory_keep_recent_tokens",
    )
    @classmethod
    def _validate_positive_int(cls, v: int) -> int:
        """校验整数配置必须为正数。"""
        if v < 1:
            raise ValueError("值必须 >= 1")
        return v

    @field_validator("context_compaction_threshold", "recent_history_token_budget_ratio")
    @classmethod
    def _validate_ratio(cls, v: float) -> float:
        """校验比例配置处于合理区间。"""
        if not 0.1 <= v <= 0.95:
            raise ValueError("比例参数必须在 0.1 ~ 0.95 之间")
        return v
