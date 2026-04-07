"""异步定时任务调度器。

这个模块负责三件事：
1. 管理任务的增删查。
2. 把任务持久化到 JSON 文件。
3. 在后台定时唤醒并执行到期任务。

它不直接关心“任务执行时要做什么业务”，真正的业务逻辑通过 `on_job`
回调注入，因此 `CronService` 保持为纯调度层。
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

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def _now_ms() -> int:
    """返回当前时间的毫秒级时间戳。"""
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """根据调度规则计算下一次执行时间。

    返回 `None` 代表这条规则当前不可执行，例如：
    - `at` 已经过期
    - `every` 间隔非法
    - `cron` 表达式缺失或解析失败
    """
    if schedule.kind == "at":
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    if schedule.kind == "every":
        return now_ms + schedule.every_ms if schedule.every_ms and schedule.every_ms > 0 else None

    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter

            base = datetime.fromtimestamp(now_ms / 1000, tz=BEIJING_TZ)
            return int(croniter(schedule.expr, base).get_next(datetime).timestamp() * 1000)
        except Exception:
            return None

    return None


def _validate_schedule(schedule: CronSchedule) -> None:
    """在任务创建前校验调度规则是否合法。"""
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
    """提供任务持久化、调度和执行的统一入口。"""

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        self.store_path = store_path
        self.on_job = on_job
        self._store: CronStore | None = None
        self._last_mtime = 0.0
        self._timer_task: asyncio.Task | None = None
        self._running = False

    async def start(self) -> None:
        """启动调度器并为现有任务重新计算下一次唤醒时间。"""
        self._running = True
        store = self._load_store()
        now = _now_ms()
        for job in store.jobs:
            self._schedule_job(job, now)
        self._save_store()
        self._arm_timer()
        logger.info("定时任务服务已启动，当前共加载 {} 个任务", len(store.jobs))

    def stop(self) -> None:
        """停止调度器，并取消当前已挂起的定时器任务。"""
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """列出任务，默认只返回启用中的任务。"""
        jobs = self._load_store().jobs
        if not include_disabled:
            jobs = [job for job in jobs if job.enabled]
        return sorted(jobs, key=lambda job: job.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        delete_after_run: bool = False,
    ) -> CronJob:
        """创建任务并立即持久化，同时重置调度器唤醒点。"""
        _validate_schedule(schedule)

        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",
                message=message,
                deliver=deliver,
            ),
            state=CronJobState(next_run_at_ms=_compute_next_run(schedule, now)),
            created_at_ms=now,
            updated_at_ms=now,
            delete_after_run=delete_after_run,
        )

        store = self._load_store()
        store.jobs.append(job)
        self._save_store()
        self._arm_timer()
        logger.info("定时任务已添加：'{}'（ID：{}）", job.name, job.id)
        return job

    def remove_job(self, job_id: str) -> bool:
        """按 id 删除任务，删除成功时同步更新存储和定时器。"""
        store = self._load_store()
        before = len(store.jobs)
        store.jobs = [job for job in store.jobs if job.id != job_id]
        removed = len(store.jobs) != before
        if removed:
            self._save_store()
            self._arm_timer()
            logger.info("定时任务已删除：{}", job_id)
        return removed

    def status(self) -> dict[str, Any]:
        """返回调度器运行状态摘要。"""
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._next_wake_ms(),
        }

    def _load_store(self) -> CronStore:
        """从磁盘读取任务存储。

        这里带有简单缓存：
        - 如果内存中已有 store，且底层文件未变化，直接复用。
        - 如果文件被外部修改或删除，则重新加载。
        """
        if self._store is not None and not self._store_changed():
            return self._store

        if not self.store_path.exists():
            self._store = CronStore()
            self._last_mtime = 0.0
            return self._store

        try:
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            jobs = [self._job_from_dict(item) for item in data.get("jobs", [])]
            self._store = CronStore(version=data.get("version", 1), jobs=jobs)
        except Exception as exc:
            logger.warning("读取定时任务存储失败：{}", exc)
            self._store = CronStore()

        self._last_mtime = self.store_path.stat().st_mtime
        return self._store

    def _save_store(self) -> None:
        """将当前内存态任务完整写回 JSON 文件。"""
        if self._store is None:
            return

        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._store.version,
            "jobs": [self._job_to_dict(job) for job in self._store.jobs],
        }
        self.store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        self._last_mtime = self.store_path.stat().st_mtime

    def _store_changed(self) -> bool:
        """判断磁盘存储是否相对缓存发生变化。"""
        if not self.store_path.exists():
            return self._last_mtime != 0.0
        return self.store_path.stat().st_mtime != self._last_mtime

    def _next_wake_ms(self) -> int | None:
        """找出所有启用任务中最近的一次唤醒时间。"""
        if self._store is None:
            return None
        next_runs = [
            job.state.next_run_at_ms
            for job in self._store.jobs
            if job.enabled and job.state.next_run_at_ms is not None
        ]
        return min(next_runs) if next_runs else None

    def _schedule_job(self, job: CronJob, now_ms: int | None = None) -> None:
        """根据任务的 schedule 刷新其 `next_run_at_ms`。"""
        if not job.enabled:
            job.state.next_run_at_ms = None
            return
        job.state.next_run_at_ms = _compute_next_run(job.schedule, now_ms or _now_ms())

    def _arm_timer(self) -> None:
        """为最近一次待执行任务挂一个新的异步定时器。

        每次任务列表发生变化或任务执行完成后都会重置一次，
        保证内存里始终只保留一个“最近唤醒点”。
        """
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        next_wake = self._next_wake_ms()
        if not self._running or next_wake is None:
            return

        delay = max(0.0, (next_wake - _now_ms()) / 1000)

        async def tick() -> None:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return
            if self._running:
                await self._on_timer()

        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        """定时器唤醒后，执行所有已经到期的任务。"""
        store = self._load_store()
        now = _now_ms()
        due_jobs = [
            job
            for job in store.jobs
            if job.enabled and job.state.next_run_at_ms is not None and job.state.next_run_at_ms <= now
        ]

        for job in due_jobs:
            await self._execute_job(job)

        self._save_store()
        self._arm_timer()

    async def _execute_job(self, job: CronJob) -> None:
        """执行单个任务，并根据任务类型更新后续状态。

        对于 `at` 任务：
        - `delete_after_run=True` 时，执行后直接删除。
        - 否则仅禁用，保留历史状态。

        对于循环任务，则重新计算下一次执行时间。
        """
        started_at = _now_ms()
        logger.info("开始执行定时任务：'{}'（ID：{}）", job.name, job.id)

        try:
            if self.on_job:
                await self.on_job(job)
            job.state.last_status = "ok"
            job.state.last_error = None
        except Exception as exc:
            job.state.last_status = "error"
            job.state.last_error = str(exc)
            logger.error("定时任务 '{}' 执行失败：{}", job.name, exc)

        job.state.last_run_at_ms = started_at
        job.updated_at_ms = _now_ms()

        if job.schedule.kind == "at":
            if job.delete_after_run and self._store is not None:
                self._store.jobs = [current for current in self._store.jobs if current.id != job.id]
            else:
                job.enabled = False
                job.state.next_run_at_ms = None
            return

        self._schedule_job(job)

    @staticmethod
    def _job_from_dict(data: dict[str, Any]) -> CronJob:
        """把 JSON 字典还原成 `CronJob`。"""
        return CronJob(
            id=data["id"],
            name=data["name"],
            enabled=data.get("enabled", True),
            schedule=CronSchedule(
                kind=data["schedule"]["kind"],
                at_ms=data["schedule"].get("atMs"),
                every_ms=data["schedule"].get("everyMs"),
                expr=data["schedule"].get("expr"),
            ),
            payload=CronPayload(
                kind=data["payload"].get("kind", "agent_turn"),
                message=data["payload"].get("message", ""),
                deliver=data["payload"].get("deliver", False),
            ),
            state=CronJobState(
                next_run_at_ms=data.get("state", {}).get("nextRunAtMs"),
                last_run_at_ms=data.get("state", {}).get("lastRunAtMs"),
                last_status=data.get("state", {}).get("lastStatus"),
                last_error=data.get("state", {}).get("lastError"),
            ),
            created_at_ms=data.get("createdAtMs", 0),
            updated_at_ms=data.get("updatedAtMs", 0),
            delete_after_run=data.get("deleteAfterRun", False),
        )

    @staticmethod
    def _job_to_dict(job: CronJob) -> dict[str, Any]:
        """把 `CronJob` 序列化成适合落盘的字典。"""
        return {
            "id": job.id,
            "name": job.name,
            "enabled": job.enabled,
            "schedule": {
                "kind": job.schedule.kind,
                "atMs": job.schedule.at_ms,
                "everyMs": job.schedule.every_ms,
                "expr": job.schedule.expr,
            },
            "payload": {
                "kind": job.payload.kind,
                "message": job.payload.message,
                "deliver": job.payload.deliver,
            },
            "state": {
                "nextRunAtMs": job.state.next_run_at_ms,
                "lastRunAtMs": job.state.last_run_at_ms,
                "lastStatus": job.state.last_status,
                "lastError": job.state.last_error,
            },
            "createdAtMs": job.created_at_ms,
            "updatedAtMs": job.updated_at_ms,
            "deleteAfterRun": job.delete_after_run,
        }
