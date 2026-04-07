"""定时任务数据类型定义"""

from dataclasses import dataclass, field


@dataclass
class CronSchedule:
    """定时任务调度规则"""
    kind: str  # "at" / "every" / "cron"
    at_ms: int | None = None
    every_ms: int | None = None
    expr: str | None = None


@dataclass
class CronPayload:
    """定时任务执行内容"""
    message: str = ""
    deliver: bool = False


@dataclass
class CronJobState:
    """定时任务运行状态"""
    next_run_at_ms: int | None = None
    last_run_at_ms: int | None = None
    last_status: str | None = None  # "ok" / "error" / "skipped"
    last_error: str | None = None


@dataclass
class CronJob:
    """完整定时任务"""
    id: str
    name: str
    enabled: bool = True
    schedule: CronSchedule = field(default_factory=lambda: CronSchedule(kind="every"))
    payload: CronPayload = field(default_factory=CronPayload)
    state: CronJobState = field(default_factory=CronJobState)
    created_at_ms: int = 0
    updated_at_ms: int = 0
    delete_after_run: bool = False


@dataclass
class CronStore:
    """定时任务存储"""
    version: int = 1
    jobs: list[CronJob] = field(default_factory=list)
