"""异步定时任务调度器。

核心机制：
- 任务持久化到 JSON 文件，重启后自动恢复
- 用一个 asyncio.Task 做计时器，sleep 到最近任务的执行时间
- 到期后执行回调，循环任务刷新下次时间，一次性任务自动删除
"""

from __future__ import annotations
import asyncio
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Coroutine
from zoneinfo import ZoneInfo

from loguru import logger

from ZBot.cron.types import CronJob, CronSchedule


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def _now_ms() -> int:
    """当前时间的毫秒级时间戳。"""
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """根据调度规则计算下一次触发时间（毫秒时间戳），返回 None 表示不再执行。"""
    # 一次性任务：时间已过则不再执行
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    # 固定间隔任务：当前时间 + 间隔
    if schedule.kind == "every":
        return now_ms + schedule.every_ms if schedule.every_ms and schedule.every_ms > 0 else None

    # Cron 表达式任务：用 croniter 库算出下一个匹配时间
    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter

            base = datetime.fromtimestamp(now_ms / 1000, tz=BEIJING_TZ)
            return int(croniter(schedule.expr, base).get_next(datetime).timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule(schedule: CronSchedule) -> None:
    """创建任务前校验调度规则，不合法则抛 ValueError。"""
    if schedule.kind == "at":
        if schedule.at_ms is None:
            raise ValueError("at 类型任务必须提供 at_ms。")
        return

    if schedule.kind == "every":
        if schedule.every_ms is None or schedule.every_ms <= 0:
            raise ValueError("every 类型任务的 every_ms 必须大于 0。")
        return

    if schedule.kind == "cron":
        if not schedule.expr:
            raise ValueError("cron 类型任务必须提供 expr。")
        try:
            from croniter import croniter
        except Exception as exc:
            raise ValueError("cron 类型任务依赖 croniter，请先安装该依赖。") from exc
        if not croniter.is_valid(schedule.expr):
            raise ValueError(f"Cron 表达式无效：'{schedule.expr}'")
        return

    raise ValueError(f"不支持的任务调度类型：'{schedule.kind}'")


class CronService:
    """
    定时任务调度服务。

    用法：
        cron = CronService(store_path, on_job=my_callback)
        await cron.start()          # 启动，加载已有任务
        cron.add_job(...)           # 添加任务
        cron.stop()                 # 停止

    调度原理：
        _arm_timer() 创建一个 asyncio.Task，sleep 到最近任务的执行时间。
        醒来后执行所有到期任务，然后再次 arm（循环往复）。
        添加/删除任务时会 cancel 旧计时器、创建新的。
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, Any]] | None = None,
    ):
        """初始化定时任务存储路径和到期回调。"""
        self.store_path = store_path          # JSON 持久化文件路径
        self.on_job = on_job                  # 任务到期时的业务回调
        self._timer_task: asyncio.Task | None = None  # 当前挂起的计时器
        self._running = False                 # 服务是否在运行

    # ==================== 公开接口 ====================

    async def start(self) -> None:
        """启动调度器：从磁盘加载任务，为每个任务算出下次执行时间，启动计时器。"""
        self._running = True
        jobs = self._load_jobs()
        now = _now_ms()
        # 重启后重新计算每个任务的下次触发时间（防止因停机期间错过的时间点丢失）
        for job in jobs:
            job.next_run_at_ms = _compute_next_run(job.schedule, now)
        self._save_jobs(jobs)
        self._arm_timer()
        logger.info("定时任务服务已启动，当前共加载 {} 个任务", len(jobs))

    def stop(self) -> None:
        """停止调度器，取消挂起的计时器。"""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def list_jobs(self) -> list[CronJob]:
        """返回所有任务，按下次执行时间排序。"""
        jobs = self._load_jobs()
        return sorted(jobs, key=lambda j: j.next_run_at_ms or float("inf"))

    def add_job(self, name: str, schedule: CronSchedule, message: str) -> CronJob:
        """创建一条新任务，持久化并重置计时器。"""
        _validate_schedule(schedule)

        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],       # 取 UUID 前 8 位做短 ID
            name=name,
            message=message,
            schedule=schedule,
            next_run_at_ms=_compute_next_run(schedule, now),
        )

        jobs = self._load_jobs()
        jobs.append(job)
        self._save_jobs(jobs)
        self._arm_timer()                   # 有新任务了，可能需要提前唤醒
        logger.info("定时任务已添加：'{}'（ID：{}）", job.name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """按 ID 删除任务。"""
        jobs = self._load_jobs()
        before = len(jobs)
        jobs = [j for j in jobs if j.id != job_id]
        removed = len(jobs) != before
        if removed:
            self._save_jobs(jobs)
            self._arm_timer()               # 删掉的可能是最近要执行的，需要重新算唤醒时间
            logger.info("定时任务已删除：{}", job_id)
        return removed

    def _arm_timer(self) -> None:
        """
        挂起一个异步计时器等待下一个任务到期。

        流程：cancel 旧计时器 → 算出等待秒数 → asyncio.sleep → 到期执行。
        整个过程不阻塞其他协程。
        """
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        next_wake = self._next_wake_ms()
        if not self._running or next_wake is None:
            return

        # 计算需要等待的秒数（防止负值）
        delay = max(0.0, (next_wake - _now_ms()) / 1000)

        async def tick() -> None:
            """等待到期时间并触发定时任务检查。"""
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return                     # 被 cancel 了（比如添加了更早的任务），直接退出
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    def _next_wake_ms(self) -> int | None:
        """返回所有任务中最早的 next_run_at_ms，用于确定计时器要等多久。"""
        jobs = self._load_jobs()
        next_runs = [j.next_run_at_ms for j in jobs if j.next_run_at_ms is not None]
        return min(next_runs) if next_runs else None

    async def _on_timer(self) -> None:
        """计时器到期：找出所有到期任务逐个执行，然后重新 arm。"""
        jobs = self._load_jobs()
        now = _now_ms()
        # 筛选出所有到期任务（next_run_at_ms <= 当前时间）
        due = [j for j in jobs if j.next_run_at_ms is not None and j.next_run_at_ms <= now]

        for job in due:
            await self._execute_job(job)

        # 执行完后重新读取（_execute_job 可能删了 at 类型任务），持久化并重新调度
        jobs = self._load_jobs()
        self._save_jobs(jobs)
        self._arm_timer()

    async def _execute_job(self, job: CronJob) -> None:
        """执行单条任务的业务回调，然后根据调度类型决定后续。"""
        logger.info("开始执行定时任务：'{}'（ID：{}）", job.name, job.id)
        try:
            if self.on_job:
                await self.on_job(job)     # 调用外部注入的业务逻辑
        except Exception as exc:
            logger.error("定时任务 '{}' 执行失败：{}", job.name, exc)

        # 一次性任务（at）：执行完就删
        if job.schedule.kind == "at":
            jobs = [j for j in self._load_jobs() if j.id != job.id]
            self._save_jobs(jobs)
            return

        # 循环任务（every/cron）：算出下一次触发时间，等下一轮计时器来执行
        job.next_run_at_ms = _compute_next_run(job.schedule, _now_ms())

    # ==================== 持久化和序列化 ====================

    def _load_jobs(self) -> list[CronJob]:
        """从 JSON 文件读取任务列表。文件不存在则返回空列表。"""
        if not self.store_path.exists():
            return []

        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            return [self._job_from_dict(item) for item in data.get("jobs", [])]
        except Exception as exc:
            logger.warning("读取定时任务存储失败：{}", exc)
            return []

    def _save_jobs(self, jobs: list[CronJob]) -> None:
        """将任务列表写入 JSON 文件（覆盖写入）。"""
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"jobs": [self._job_to_dict(j) for j in jobs]}
        self.store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _job_from_dict(data: dict[str, Any]) -> CronJob:
        """JSON 字典 → CronJob。兼容新旧两种格式（旧格式字段嵌套在 payload/state 里）。"""
        schedule_data = data.get("schedule", {})
        # 新格式：message 直接在 job 层；旧格式：嵌套在 payload.message 里
        message = data.get("message") or data.get("payload", {}).get("message", "")
        # 新格式：next_run_at_ms 直接在 job 层；旧格式：嵌套在 state.nextRunAtMs 里
        next_run = data.get("next_run_at_ms") or data.get("state", {}).get("nextRunAtMs")

        return CronJob(
            id=data["id"],
            name=data["name"],
            message=message,
            schedule=CronSchedule(
                kind=schedule_data["kind"],
                at_ms=schedule_data.get("at_ms") or schedule_data.get("atMs"),
                every_ms=schedule_data.get("every_ms") or schedule_data.get("everyMs"),
                expr=schedule_data.get("expr"),
            ),
            next_run_at_ms=next_run,
        )

    @staticmethod
    def _job_to_dict(job: CronJob) -> dict[str, Any]:
        """CronJob → JSON 字典。"""
        return {
            "id": job.id,
            "name": job.name,
            "message": job.message,
            "schedule": {
                "kind": job.schedule.kind,
                "at_ms": job.schedule.at_ms,
                "every_ms": job.schedule.every_ms,
                "expr": job.schedule.expr,
            },
            "next_run_at_ms": job.next_run_at_ms,
        }
