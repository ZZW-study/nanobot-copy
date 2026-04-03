"""
定时任务 数据类型定义
作用：定义整个定时任务系统的所有数据结构（任务配置、任务内容、任务状态、存储格式）
"""
from dataclasses import dataclass,field
from typing import Literal   # 字面量类型：限定变量只能取指定的几个固定值（强类型校验，避免传错参数）

@dataclass
class CronSchedule:
    """
    【核心】定时任务的调度规则
    定义：任务「什么时候执行」，支持三种模式：指定时间/固定间隔/Cron表达式
    """
    kind: Literal["at","every","cron"] # 调度类型：只能是 at/every/cron 三选一
    at_ms: int | None = None    # 仅用于 at 模式：指定执行的时间戳（毫秒级）
    every_ms: int | None = None # 仅用于 every 模式：执行间隔（毫秒级)
    expr: str | None = None     # 仅用于 cron 模式：Cron表达式（如 "0 9 * * *" 表示每天早上9点）
    tz: str | None = None 

@dataclass
class CronPayload:
    """
    【核心】定时任务的执行内容
    定义：任务「执行时要做什么」，是任务的业务逻辑载体
    """
    kind: Literal["system_event","agent_turn"] = "agent_turn" # 任务类型：系统事件 / 智能体执行（默认是智能体任务）
    message: str = ""                                         # 任务执行的消息内容（给智能体的指令/提示词）
    deliver: bool = False                                     # 是否将执行结果发送到聊天渠道
    channel: str | None = None                                # 渠道类型（如 whatsapp/telegram/discord）
    to: str | None = None                                     # 接收方标识（如手机号、用户ID）


@dataclass
class CronJobState:
    """
    定时任务的运行时状态
    定义：任务「当前执行到哪了、上次执行结果」，纯状态追踪
    """
    next_run_at_ms: int | None = None  # 下次执行时间（毫秒时间戳）
    last_run_at_ms: int | None = None  # 上次执行时间（毫秒时间戳）
    last_status: Literal["ok","error","skipped"] | None = None  # 上次执行状态：成功/失败/跳过
    last_error: str | None = None  # 上次执行失败的错误信息（成功则为None）


@dataclass
class CronJob:
    """
    完整的定时任务对象
    聚合了：任务ID、基础信息、调度规则、执行内容、运行状态、时间戳
    是整个定时任务系统的最小操作单元
    """
    id: str               # 任务唯一ID（8位UUID，用于增删改查）
    name: str             # 任务名称（方便人类识别）
    enabled: bool = True  # 任务是否启用（禁用后不会自动执行）

    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))  # 调度规则
    payload: CronPayload = field(default_factory=CronPayload)  # 任务执行内容
    state: CronJobState = field(default_factory=CronJobState)  # 任务运行状态

    created_at_ms: int = 0  # 创建时间（毫秒时间戳）
    updated_at_ms: int = 0  # 最后更新时间（毫秒时间戳）
    delete_after_run: bool = False  # 一次性任务专用：执行完成后是否自动删除任务


@dataclass
class CronStore:
    """
    定时任务的持久化存储容器
    定义：所有任务的存储格式，用于写入/读取JSON文件
    """
    version: int = 1                                   # 存储格式版本号（方便后续升级数据结构）
    jobs: list[CronJob] = field(default_factory=list)  # 任务列表：存储所有的定时任务

