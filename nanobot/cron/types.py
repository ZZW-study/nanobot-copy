"""定时任务的数据类型定义（带中文注释，帮助初学者理解各字段含义）。

这个模块定义了定时任务系统的核心数据结构，包括：
1. 调度规则（CronSchedule）- 定义任务何时执行
2. 任务负载（CronPayload）- 定义任务执行什么内容
3. 运行状态（CronJobState）- 记录任务的执行历史和状态
4. 完整任务（CronJob）- 包含任务的所有信息
5. 任务存储（CronStore）- 用于持久化所有任务的容器

这些数据类使用 Python 的 dataclass 装饰器，自动生成构造函数、repr、eq 等方法。
"""

from dataclasses import dataclass, field


@dataclass
class CronSchedule:
    """
    定时任务的调度规则结构。

    支持三种调度类型，通过 kind 字段区分：

    1. "at" - 在指定的绝对时间点执行一次
       - 使用 at_ms 字段指定执行时间（UTC 毫秒时间戳）
       - 适用于一次性任务，如"在明天上午9点提醒我"

    2. "every" - 按固定间隔重复执行
       - 使用 every_ms 字段指定执行间隔（毫秒）
       - 适用于周期性任务，如"每5分钟检查一次状态"

    3. "cron" - 使用标准 cron 表达式调度
       - 使用 expr 字段指定 cron 表达式
       - 适用于复杂的时间模式，如"每个工作日的上午9点"

    注意：每次只能使用一种调度类型，其他字段应为 None。
    """

    kind: str  # 调度类型："at" / "every" / "cron"
    at_ms: int | None = None  # 绝对执行时间（毫秒时间戳），仅当 kind="at" 时有效
    every_ms: int | None = None  # 执行间隔（毫秒），仅当 kind="every" 时有效
    expr: str | None = None  # cron 表达式，仅当 kind="cron" 时有效


@dataclass
class CronPayload:
    """
    任务要执行的负载内容（业务侧定义的执行语义）。

    当前系统采用简单的消息传递模型，但设计上支持未来扩展：

    - message: 任务执行时传递的文本消息内容
      例如："检查服务器状态"、"发送每日报告"等指令

    - deliver: 是否在任务触发时把结果直接推送给用户/渠道
      - True: 任务执行后立即将结果发送给用户
      - False: 任务在后台静默执行，结果可能记录到日志或存储中

    未来可扩展为更复杂的结构，如：
    - command: 要执行的具体命令
    - args: 命令参数
    - target: 目标用户或频道
    - priority: 任务优先级
    """

    message: str = ""  # 任务执行时传递的文本消息
    deliver: bool = False  # 是否在任务触发时把消息直接推送给用户/渠道


@dataclass
class CronJobState:
    """
    记录任务的运行状态信息：下次运行时间、上次运行时间与错误状态等。

    这些状态信息用于：
    1. 任务调度器决定何时执行下一个任务
    2. 监控和调试任务执行情况
    3. 用户查询任务历史和状态

    所有时间戳都使用 UTC 毫秒时间戳，便于跨时区处理。
    """

    next_run_at_ms: int | None = None  # 下次计划运行的毫秒时间戳（None 表示未安排）
    last_run_at_ms: int | None = None  # 上次实际执行时间（毫秒）
    last_status: str | None = None  # 上次执行状态，常见值："ok"、"error"
    last_error: str | None = None  # 上次执行的错误信息（若有）


@dataclass
class CronJob:
    """
    表示一条完整的定时任务。

    包含任务的完整信息：
    - 元数据：ID、名称、启用状态
    - 调度配置：何时执行（schedule）
    - 执行内容：做什么（payload）
    - 运行状态：执行历史（state）
    - 生命周期：创建/更新时间、是否自动删除

    字段说明：
    - id: 任务的唯一标识符（UUID 或其他唯一字符串）
    - name: 任务的可读名称（用于显示和管理）
    - enabled: 任务是否启用（False 时不会被调度执行）
    - schedule: 调度规则（定义何时执行）
    - payload: 执行负载（定义执行什么）
    - state: 运行状态（记录执行历史）
    - created_at_ms: 任务创建时间（毫秒时间戳）
    - updated_at_ms: 任务最后更新时间（毫秒时间戳）
    - delete_after_run: 任务执行后是否自动删除（适用于一次性任务）
    """

    id: str  # 任务的唯一标识符
    name: str  # 任务的可读名称
    enabled: bool = True  # 任务是否启用（默认启用）
    # 调度规则，默认使用一个 every 类型的 schedule（间隔任务）
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)  # 任务负载
    state: CronJobState = field(default_factory=CronJobState)  # 运行状态
    created_at_ms: int = 0  # 创建时间（毫秒时间戳）
    updated_at_ms: int = 0  # 更新时间（毫秒时间戳）
    delete_after_run: bool = False  # 任务执行后是否自动删除


@dataclass
class CronStore:
    """
    任务存储结构：用于持久化到磁盘的顶层容器。

    这个类是所有定时任务的集合，用于：
    1. 序列化到 JSON 文件进行持久化存储
    2. 反序列化从文件加载任务
    3. 版本兼容性管理

    字段说明：
    - version: 存储格式版本号，用于将来升级时的兼容性处理
      例如：当需要修改数据结构时，可以基于版本号进行迁移
    - jobs: 当前存储的所有 CronJob 列表
    """

    version: int = 1  # 存储格式版本号（当前为版本1）
    jobs: list[CronJob] = field(default_factory=list)  # 所有定时任务列表
