"""Agent 运行时配置。

这个模块不替代全局配置文件，只把全局 Config 中和 Agent 实例化有关的字段
整理成一份更适合传给 Agent 构造器的快照。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ZBot.config.schema import Config, ExecToolConfig, WebSearchConfig


# ── @dataclass ──────────────────────────────────────────────────────────────
# 自动生成以下样板代码，无需手写：
#   __init__   : 按字段顺序生成构造函数，有默认值的字段放在后面。
#   __repr__   : 打印实例时显示所有字段名和值，方便调试。
#   __eq__     : 两个实例字段值全部相等时返回 True，而不是比较内存地址。
#
# 没有 @dataclass 时你要手写：
#   class AgentRuntimeConfig:
#       def __init__(self, workspace, model, temperature=0.1, ...):
#           self.workspace = workspace
#           self.model = model
#           self.temperature = temperature
#           ...
#       def __repr__(self):
#           return f"AgentRuntimeConfig(workspace={self.workspace}, ...)"
#       def __eq__(self, other):
#           return self.workspace == other.workspace and self.model == other.model and ...
#
# ── slots=True ──────────────────────────────────────────────────────────────
# Python 默认每个实例用一个 __dict__（字典）来存所有属性：
#   obj.__dict__  →  {"workspace": ..., "model": ..., "temperature": ...}
#   字典是哈希表，查找属性需要计算哈希值，有额外的内存和时间开销。
#
# slots=True 改为在类上声明固定槽位 __slots__：
#   obj.__slots__  →  ("workspace", "model", "temperature", ...)
#   槽位像 C struct 一样紧凑排列在内存里，按固定偏移量直接取值，不需要哈希。
#
# 好处：
#   ① 省内存：去掉了每个实例的 __dict__ 字典，节省约 30~50% 的实例内存。
#      实例越多，收益越明显（连接池、配置快照这类会大量创建的对象尤其合适）。
#   ② 访问更快：直接按偏移量读写，比哈希查找快。
#   ③ 更安全：字段在类定义时固定，IDE 和类型检查器能更好地发现拼写错误。
#
# 代价：
#   不能在运行时动态添加类定义之外的属性，否则直接报错：
#     obj.new_attr = 123  →  AttributeError: 'AgentRuntimeConfig' object has no attribute 'new_attr'
#   普通类（没有 slots）可以随时加新属性，因为底层就是往 __dict__ 里加一个 key。
#
# 为什么 AgentRuntimeConfig 适合用 slots=True：
#   字段在类定义时已经全部确定，运行期间不需要动态添加新属性。
#   作为配置快照会被频繁创建，省内存和提速的收益明显。
@dataclass(slots=True)
class AgentRuntimeConfig:
    """Agent 构造和运行所需的配置快照。

    全局 Config 仍然是唯一的用户配置来源。这个 dataclass 只是把 Agent
    实例化时需要的一组字段整理出来，避免 BaseAgent/CoreAgent/SubAgent
    构造函数塞满零散参数。
    """

    workspace: Path
    model: str
    temperature: float = 0.1
    max_tokens: int = 4096
    reasoning_effort: str | None = None
    agent_timeout_seconds: int = 3600
    subagent_timeout_seconds: int = 600
    context_compaction_threshold: float = 0.8
    recent_history_token_budget_ratio: float = 0.25
    recent_history_max_tokens: int = 64_000
    memory_consolidation_interval: int = 40
    session_memory_keep_recent_tokens: int = 16_000

    # field(default_factory=WebSearchConfig) 的作用：
    #   实例化时若没有显式传入 web_search_config，则自动调用 WebSearchConfig() 新建一个默认值。
    #   每次实例化都新建，不会共享同一个默认对象。
    #
    # 为什么不能直接写 web_search_config: WebSearchConfig = WebSearchConfig()：
    #   类定义阶段只执行一次，所有没传参的实例共享同一个 WebSearchConfig 对象。
    #   若该对象被修改，所有实例都会受影响（诡异的共享状态 bug）。
    #
    # 对当前代码的实际效果：
    #   唯一入口 from_app_config 每次都显式传入 web_search_config，
    #   default_factory 永远不会被触发，对现在的代码没有实际作用。
    #   保留它是防御性编程：万一将来有人绕过 from_app_config 直接实例化，
    #   不传 web_search_config 也能安全兜底，而不是报"缺少必填参数"错误。
    web_search_config: WebSearchConfig = field(default_factory=WebSearchConfig)
    web_proxy: str | None = None
    # 同上，exec_config 也用 default_factory 做兜底防御。
    exec_config: ExecToolConfig = field(default_factory=ExecToolConfig)
    restrict_to_workspace: bool = False
    # dict 是可变对象，同样必须用 default_factory，否则所有实例共享同一个空字典。
    mcp_servers: dict[str, Any] = field(default_factory=dict)
    score_threshold: float = 0.75

    @classmethod
    def from_app_config(cls, config: Config, *, model: str) -> "AgentRuntimeConfig":
        """从全局配置派生 Agent 运行时配置。

        `model` 使用 provider 解析后的实际模型名。其他工具、记忆窗口、
        MCP 服务器、工作区等字段仍然从全局 Config 读取。
        """
        # cls 就是类本身，cls(...) 等价于 AgentRuntimeConfig(...)，即创建一个新实例。
        #
        # Python 传参传的是引用，不是复制，因此：
        #   - int/float/bool/str 等不可变类型（temperature/max_tokens 等）：
        #     重新赋值时指向新对象，各实例独立，互不影响。
        #   - WebSearchConfig/ExecToolConfig 等可变对象：
        #     多次调用 from_app_config 创建的实例，web_search_config 指向同一个对象：
        #     rt1.web_search_config is rt2.web_search_config  →  True
        #     若其中一个实例修改了该对象的属性，另一个实例也会受影响。
        #     当前 Agent 只读这些配置，共享引用安全，无需 deepcopy。
        #     若需完全隔离，传参时改用 copy.deepcopy(config.tools.web.search)。
        return cls(
            workspace=config.workspace_path,
            model=model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
            agent_timeout_seconds=config.agent_timeout_seconds,
            subagent_timeout_seconds=config.subagent_timeout_seconds,
            context_compaction_threshold=config.context_compaction_threshold,
            recent_history_token_budget_ratio=config.recent_history_token_budget_ratio,
            recent_history_max_tokens=config.recent_history_max_tokens,
            memory_consolidation_interval=config.memory_consolidation_interval,
            session_memory_keep_recent_tokens=config.session_memory_keep_recent_tokens,
            web_search_config=config.tools.web.search,   # 共享引用，只读安全
            web_proxy=config.tools.web.proxy or None,
            exec_config=config.tools.exec,               # 共享引用，只读安全
            restrict_to_workspace=config.tools.restrict_to_workspace,
            # dict() 创建浅拷贝：顶层 key/value 独立（各实例的 mcp_servers 不是同一个字典），
            # 但 value 若是嵌套对象，仍然共享引用。比 web_search_config 稍安全，但未完全隔离。
            mcp_servers=dict(config.tools.mcp_servers),
            score_threshold=config.score_threshold,
        )
