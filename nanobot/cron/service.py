"""异步定时任务调度器（CronService）。

此模块负责：
- 任务的增删查（管理接口）
- 将任务持久化到磁盘上的 JSON 文件
- 在后台按计划唤醒并执行到期任务

实现细节：调度器本身不负责任务的业务逻辑。业务逻辑通过 `on_job` 回调注入，
因此 `CronService` 更像是一个纯粹的调度层。

核心类：
    CronService: 定时任务调度服务，负责任务的生命周期管理

核心函数：
    _now_ms(): 获取当前毫秒时间戳
    _compute_next_run(): 计算任务的下一次执行时间
    _validate_schedule(): 验证调度规则的合法性
"""

from __future__ import annotations  # 启用未来版本的类型注解特性

import asyncio  # 用于异步任务调度和定时器
import json  # 用于持久化任务数据到 JSON 文件
import time  # 用于获取时间戳
import uuid  # 用于生成唯一任务 ID
from datetime import datetime  # 用于处理日期时间
from pathlib import Path  # 用于路径操作
from typing import Any, Callable, Coroutine  # 用于类型注解
from zoneinfo import ZoneInfo  # 用于时区处理（北京时间）

from loguru import logger  # 用于日志记录

from nanobot.cron.types import CronJob, CronJobState, CronPayload, CronSchedule, CronStore


# 使用北京时间时区进行 cron 表达式计算与时间展示
# 北京时间是 UTC+8，在中国地区部署的系统推荐使用此时区
BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def _now_ms() -> int:
    """
    返回当前时间的毫秒级时间戳（整数）。

    系统内部统一使用毫秒时间戳（而非秒）来比较和存储时间，
    以获得更高的精度（例如 1672531200000 表示 2023-01-01 00:00:00）。

    Returns:
        当前 UTC 时间的毫秒时间戳（Unix timestamp in milliseconds）
    """
    return int(time.time() * 1000)


def _compute_next_run(schedule: CronSchedule, now_ms: int) -> int | None:
    """
    根据任务的 `schedule` 规则计算下一次应触发的毫秒时间戳。

    返回值：下一次触发的毫秒时间戳，若规则当前不可执行则返回 `None`。

    支持三种调度类型：

    1. "at" - 在指定的绝对时间点执行一次
       - 如果 at_ms 小于当前时间（已过期），返回 None（不会重复执行）
       - 如果 at_ms 大于当前时间，返回 at_ms（等待执行）

    2. "every" - 按固定间隔循环执行
       - 下一次执行时间 = 当前时间 + 间隔时间
       - 每次执行后会重新计算下一次时间（实现固定频率）

    3. "cron" - 使用标准 cron 表达式调度
       - 需要安装 croniter 库来解析 cron 表达式
       - 计算从当前时间开始的下一个匹配时间点
       - 支持复杂的调度模式（如"每个工作日 9 点"）

    Args:
        schedule: 任务的调度规则（CronSchedule）
        now_ms: 当前时间的毫秒时间戳

    Returns:
        下一次触发的毫秒时间戳，或 None（如果任务已过期或配置无效）
    """
    # --- at 类型：直接返回 at_ms（如果在未来） ---
    if schedule.kind == "at":
        # 对于一次性任务，如果指定时间已过期，则不再执行（返回 None）
        return schedule.at_ms if schedule.at_ms and schedule.at_ms > now_ms else None

    # --- every 类型：固定间隔任务 ---
    if schedule.kind == "every":
        # 下一次执行 = 当前时间 + 间隔
        # 例如：now=1000, every_ms=5000，下一次执行时间=6000
        return now_ms + schedule.every_ms if schedule.every_ms and schedule.every_ms > 0 else None

    # --- cron 类型：使用 croniter 计算下一个匹配时间 ---
    if schedule.kind == "cron" and schedule.expr:
        try:
            from croniter import croniter

            # 把基准时间转换为具有时区信息的 datetime（使用北京时间）
            # now_ms 是毫秒时间戳，需要除以 1000 转为秒
            base = datetime.fromtimestamp(now_ms / 1000, tz=BEIJING_TZ)
            # croniter 返回下一个匹配的 datetime 对象
            # 转换为毫秒整数返回（乘以 1000）
            return int(croniter(schedule.expr, base).get_next(datetime).timestamp() * 1000)
        except Exception:
            # 如果解析或计算失败（如croniter未安装、表达式无效），视为当前不可用
            return None

    # 其它不支持或参数不完整的情况
    return None


def _validate_schedule(schedule: CronSchedule) -> None:
    """
    在创建任务前，校验 schedule 字段的合法性并在错误时抛出异常。

    验证规则：
    1. at 类型：必须提供 at_ms 字段
    2. every 类型：every_ms 必须存在且为正数
    3. cron 类型：
       - 必须提供 expr 字段（cron 表达式）
       - 必须安装 croniter 库
       - cron 表达式必须合法

    Args:
        schedule: 要验证的调度规则

    Raises:
        ValueError: 如果调度规则不合法
    """
    # 验证 at 类型任务
    if schedule.kind == "at":
        if schedule.at_ms is None:
            raise ValueError("at 类型任务必须提供 at_ms。")
        return

    # 验证 every 类型任务
    if schedule.kind == "every":
        if schedule.every_ms is None or schedule.every_ms <= 0:
            raise ValueError("every 类型任务的 every_ms 必须大于 0。")
        return

    # 验证 cron 类型任务
    if schedule.kind == "cron":
        if not schedule.expr:
            raise ValueError("cron 类型任务必须提供 expr。")
        try:
            from croniter import croniter
        except Exception as exc:
            # 如果没有安装 croniter，提示用户安装依赖
            raise ValueError("cron 类型任务依赖 croniter，请先安装该依赖。") from exc
        # 验证 cron 表达式的语法是否正确
        if not croniter.is_valid(schedule.expr):
            raise ValueError(f"Cron 表达式无效：'{schedule.expr}'")
        return

    # 其它未知的 kind
    raise ValueError(f"不支持的任务调度类型：'{schedule.kind}'")


class CronService:
    """
    任务调度服务：负责任务的增删查、持久化与定时调度执行。

    CronService 是一个异步任务调度器，主要功能：
    1. 任务管理：添加、删除、查询任务
    2. 持久化：将任务数据保存到 JSON 文件
    3. 定时执行：在后台按计划唤醒并执行到期任务

    实现上：
    - 使用内存缓存（_store）提升读取性能
    - 使用异步计时器（_timer_task）等待下一个任务到期
    - 使用回调（on_job）将业务逻辑注入调度层

    参数：
    - store_path: 持久化 JSON 存储路径（如 ~/.nanobot/cron/jobs.json）
    - on_job: 当任务触发时的回调函数，接收 CronJob 对象并返回可选结果
    """

    def __init__(
        self,
        store_path: Path,
        on_job: Callable[[CronJob], Coroutine[Any, Any, str | None]] | None = None,
    ):
        """
        初始化 CronService 实例。

        Args:
            store_path: 任务持久化文件路径（JSON 格式）
            on_job: 任务触发时的回调函数（异步可选，返回字符串结果）
        """
        # 存储路径（JSON 文件）
        self.store_path = store_path
        # 触发任务时的回调（业务侧实现），可以是 async 函数
        self.on_job = on_job
        # 内存中的 store 缓存（CronStore），懒加载
        self._store: CronStore | None = None
        # 记录上次加载文件的 mtime，用于检测文件变化
        self._last_mtime = 0.0
        # 内部计时器的 asyncio.Task（用于定时唤醒）
        self._timer_task: asyncio.Task | None = None
        # 服务是否正在运行的标志
        self._running = False

    async def start(self) -> None:
        """
        启动调度器：加载存储、计算每个任务的下一次执行时间并启动计时器。

        启动流程：
        1. 设置服务运行标志（_running = True）
        2. 从磁盘加载所有任务（_load_store）
        3. 为每个任务重新计算下一次执行时间（防止服务重启丢失计算）
        4. 将更新后的任务写回磁盘
        5. 启动计时器等待下一个任务到期

        注意：计时器是异步的，不会阻塞当前协程。
        """
        self._running = True
        store = self._load_store()  # 加载持久化的任务
        now = _now_ms()
        # 为每个已存任务计算 next_run_at_ms（防止服务重启导致丢失计算）
        for job in store.jobs:
            self._schedule_job(job, now)
        # 把可能更新过的任务写回磁盘并 arm 计时器
        self._save_store()
        self._arm_timer()
        logger.info("定时任务服务已启动，当前共加载 {} 个任务", len(store.jobs))

    def stop(self) -> None:
        """
        停止调度器并取消挂起的计时器任务（同步方法）。

        停止流程：
        1. 设置服务运行标志为 False
        2. 取消挂起的计时器任务（如果存在）
        """
        self._running = False
        if self._timer_task:
            self._timer_task.cancel()  # 取消异步计时器
            self._timer_task = None

    def list_jobs(self, include_disabled: bool = False) -> list[CronJob]:
        """
        返回任务列表；默认仅返回启用的任务，按下次运行时间排序。

        Args:
            include_disabled: 是否包含已禁用的任务（默认 False）

        Returns:
            任务列表，按 next_run_at_ms 升序排列
        """
        jobs = self._load_store().jobs
        if not include_disabled:
            jobs = [job for job in jobs if job.enabled]
        # 把没有 next_run_at_ms 的任务放到最后（使用 inf 作为排序占位）
        return sorted(jobs, key=lambda job: job.state.next_run_at_ms or float("inf"))

    def add_job(
        self,
        name: str,
        schedule: CronSchedule,
        message: str,
        deliver: bool = False,
        delete_after_run: bool = False,
    ) -> CronJob:
        """
        创建并持久化一条新任务，返回 CronJob 实例。

        步骤：
        1. 验证调度规则的合法性
        2. 生成唯一任务 ID（UUID 的前 8 位）
        3. 构造 CronJob 对象（包含 schedule、payload、state）
        4. 计算下一次触发时间并记录到 state
        5. 将任务添加到内存 store 并持久化到磁盘
        6. 重置计时器以唤醒下一个任务

        Args:
            name: 任务名称（可读标识）
            schedule: 调度规则（CronSchedule）
            message: 任务执行时传递的消息内容
            deliver: 是否在任务触发时把结果直接推送给用户
            delete_after_run: 任务执行后是否自动删除（适用于一次性任务）

        Returns:
            创建的 CronJob 实例
        """
        _validate_schedule(schedule)

        now = _now_ms()
        job = CronJob(
            id=str(uuid.uuid4())[:8],  # 生成唯一任务 ID（UUID 前 8 位）
            name=name,
            enabled=True,
            schedule=schedule,
            payload=CronPayload(
                kind="agent_turn",  # 任务类型
                message=message,
                deliver=deliver,
            ),
            # 计算下一次触发时间并记录到 state
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
        """
        按 id 删除任务；若删除成功则持久化并重置计时器。

        Args:
            job_id: 要删除的任务 ID

        Returns:
            True 表示删除成功，False 表示任务不存在
        """
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
        """
        返回简要的运行状态供外部查询。

        Returns:
            状态字典，包含：
            - enabled: 服务是否正在运行
            - jobs: 任务总数
            - next_wake_at_ms: 下次唤醒时间（毫秒时间戳）
        """
        store = self._load_store()
        return {
            "enabled": self._running,
            "jobs": len(store.jobs),
            "next_wake_at_ms": self._next_wake_ms(),
        }

    def _load_store(self) -> CronStore:
        """
        从磁盘读取任务存储并缓存到内存。

        行为说明：
        - 如果内存中已有缓存且磁盘文件未变化，直接返回缓存（提升性能）
        - 如果磁盘文件不存在或解析失败，返回一个空的 CronStore（保证服务可用性）

        文件系统变更检测：
        - 记录加载时文件的 mtime（最后修改时间）
        - 下次调用时比较 mtime，只有文件真正变化时才重新读取
        """
        # 如果已加载且文件未变化，直接复用内存中的 store
        if self._store is not None and not self._store_changed():
            return self._store

        # 文件不存在时返回空的 store（首次运行场景）
        if not self.store_path.exists():
            self._store = CronStore()
            self._last_mtime = 0.0
            return self._store

        try:
            # 读取 JSON 文件内容
            data = json.loads(self.store_path.read_text(encoding="utf-8"))
            # 把每个 JSON 项转为 CronJob 对象（使用反序列化方法处理格式转换）
            jobs = [self._job_from_dict(item) for item in data.get("jobs", [])]
            self._store = CronStore(version=data.get("version", 1), jobs=jobs)
        except Exception as exc:
            # 读取或解析失败时记录警告，返回空 store 以保证服务可用性
            logger.warning("读取定时任务存储失败：{}", exc)
            self._store = CronStore()

        # 记录文件的最后修改时间，用于后续变动检测
        self._last_mtime = self.store_path.stat().st_mtime
        return self._store

    def _save_store(self) -> None:
        """
        把当前内存中的 store 序列化为 JSON 写回磁盘（覆盖写入）。

        流程：
        1. 检查内存 store 是否存在
        2. 确保父目录存在（创建必要的目录结构）
        3. 将所有 CronJob 对象转换为字典格式
        4. 写入 JSON 文件（格式化缩进，支持中文）
        5. 更新缓存的文件 mtime
        """
        if self._store is None:
            return

        # 确保父目录存在，再写文件
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._store.version,
            "jobs": [self._job_to_dict(job) for job in self._store.jobs],
        }
        # 写入 JSON 文件（格式化缩进 2 格，支持中文显示）
        self.store_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        # 更新缓存的文件 mtime
        self._last_mtime = self.store_path.stat().st_mtime

    def _store_changed(self) -> bool:
        """
        判断磁盘上的存储文件是否相对于缓存发生变化。

        通过比较文件的 mtime（最后修改时间）来判断是否有外部修改。

        Returns:
            True 表示文件已变化（需要重新加载），False 表示未变化
        """
        if not self.store_path.exists():
            # 文件不存在了，但之前有缓存，视为变化
            return self._last_mtime != 0.0
        # 比较当前 mtime 和缓存的 mtime
        return self.store_path.stat().st_mtime != self._last_mtime

    def _next_wake_ms(self) -> int | None:
        """
        计算当前所有启用任务中最早的 next_run_at_ms（返回毫秒时间戳或 None）。

        这个方法用于确定计时器应该多久后唤醒（等待时间 = 最早执行时间 - 当前时间）。

        Returns:
            所有启用任务中最早的下一次执行时间，如果没有待执行任务则返回 None
        """
        if self._store is None:
            return None
        # 筛选出所有启用的、有 next_run_at_ms 的任务
        next_runs = [
            job.state.next_run_at_ms
            for job in self._store.jobs
            if job.enabled and job.state.next_run_at_ms is not None
        ]
        # 返回最小值（最早的执行时间），如果没有则返回 None
        return min(next_runs) if next_runs else None

    def _schedule_job(self, job: CronJob, now_ms: int | None = None) -> None:
        """
        为单个 job 刷新它的 next_run_at_ms 字段。

        逻辑：
        - 如果任务被禁用（enabled=False），将 next_run_at_ms 设为 None（不参与调度）
        - 否则调用 `_compute_next_run` 计算下一次执行时间

        Args:
            job: 要调度的任务对象
            now_ms: 可选的参考时间（默认使用当前时间）
        """
        if not job.enabled:
            # 禁用的任务不安排执行时间
            job.state.next_run_at_ms = None
            return
        job.state.next_run_at_ms = _compute_next_run(job.schedule, now_ms or _now_ms())

    def _arm_timer(self) -> None:
        """
        根据下次最近唤醒时间挂起一个异步定时任务（只保留一个计时器）。

        这个方法是调度器的核心机制：
        1. 取消已有的计时器（确保只有一个待执行的计时器）
        2. 计算下一个任务的唤醒时间
        3. 启动新的异步计时器（sleep 到那个时间点）
        4. 计时器到期时触发 _on_timer() 执行所有到期任务

        为什么要使用异步计时器？
        - 不会阻塞其他协程的执行
        - 可以在等待期间响应其他事件（如添加新任务）
        - 节省 CPU 资源（sleep 期间不占用计算资源）
        """
        # 先取消已有计时器，保证内存中只存在一个待唤醒任务
        if self._timer_task:
            self._timer_task.cancel()
            self._timer_task = None

        next_wake = self._next_wake_ms()
        # 如果服务未启动或没有待唤醒任务，则无需设置计时器
        if not self._running or next_wake is None:
            return

        # 计算延迟秒数（从当前时间到下一个任务的时间差）
        # 注意：next_wake 是毫秒，现在转为秒；防止负值导致异常
        delay = max(0.0, (next_wake - _now_ms()) / 1000)

        async def tick() -> None:
            """内部计时器函数，等待指定延迟后唤醒。"""
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                # 如果计时器在等待期间被取消（如添加了更早的任务），则直接返回
                return
            if self._running:
                # 唤醒后执行到期任务处理
                await self._on_timer()

        # 使用 create_task 启动后台计时器，不阻塞当前协程
        self._timer_task = asyncio.create_task(tick())

    async def _on_timer(self) -> None:
        """
        计时器触发：执行所有已到期的任务并重置持久化与计时器。

        这是在预定时间到达时自动调用的方法，负责：
        1. 加载最新的任务列表（可能外部有修改）
        2. 找出所有到期（next_run_at_ms <= 当前时间）的启用任务
        3. 逐个执行这些任务
        4. 持久化更新后的状态
        5. 重置计时器以准备下一次唤醒
        """
        store = self._load_store()
        now = _now_ms()
        # 筛选出所有启用且到期的任务（next_run_at_ms <= 当前时间）
        due_jobs = [
            job
            for job in store.jobs
            if job.enabled and job.state.next_run_at_ms is not None and job.state.next_run_at_ms <= now
        ]

        # 顺序执行所有到期任务（避免并发冲突）
        for job in due_jobs:
            await self._execute_job(job)

        # 执行完后持久化并为下一次待执行任务重新挂起计时器
        self._save_store()
        self._arm_timer()

    async def _execute_job(self, job: CronJob) -> None:
        """
        执行单条任务并根据规则更新其后续状态。

        这是任务调度的核心执行方法，负责：
        1. 记录任务开始时间
        2. 调用业务回调（on_job）执行实际任务逻辑
        3. 更新任务的执行状态（成功/失败）
        4. 处理一次性任务和循环任务的不同后续逻辑

        Args:
            job: 要执行的任务对象
        """
        started_at = _now_ms()
        logger.info("开始执行定时任务：'{}'（ID：{}）", job.name, job.id)

        try:
            if self.on_job:
                # 将任务交给业务回调处理（回调可为 async 函数）
                # on_job 是注入的业务逻辑，这里是调度器与业务层的接口
                await self.on_job(job)
            job.state.last_status = "ok"
            job.state.last_error = None
        except Exception as exc:
            # 记录错误信息并继续，避免单个任务失败导致整个调度器崩溃
            # 即使任务失败，也要继续处理其他到期任务
            job.state.last_status = "error"
            job.state.last_error = str(exc)
            logger.error("定时任务 '{}' 执行失败：{}", job.name, exc)

        # 记录本次实际运行时间并更新时间戳
        job.state.last_run_at_ms = started_at
        job.updated_at_ms = _now_ms()

        # ========== 区分一次性任务和循环任务 ==========
        # 对于一次性 at 类型任务，执行后根据 delete_after_run 决定是否删除
        if job.schedule.kind == "at":
            if job.delete_after_run and self._store is not None:
                # 从 store 中移除该任务（彻底删除）
                self._store.jobs = [current for current in self._store.jobs if current.id != job.id]
            else:
                # 不删除则禁用并清除 next_run（标记为已完成）
                job.enabled = False
                job.state.next_run_at_ms = None
            return

        # 对于循环任务（every/cron），刷新下一次触发时间
        # 这样下次计时器唤醒时会知道何时再次执行
        self._schedule_job(job)

    @staticmethod
    def _job_from_dict(data: dict[str, Any]) -> CronJob:
        """
        将从磁盘读取的字典恢复为 CronJob 对象。

        这是反序列化方法，用于从 JSON 文件加载任务数据。
        注意：此处为向后兼容解析，字段采用 `get` 以防某些老数据缺少字段。

        Args:
            data: 从 JSON 文件读取的任务字典

        Returns:
            解析后的 CronJob 对象
        """
        return CronJob(
            id=data["id"],
            name=data["name"],
            enabled=data.get("enabled", True),  # 旧数据可能缺少此字段
            schedule=CronSchedule(
                kind=data["schedule"]["kind"],
                at_ms=data["schedule"].get("atMs"),  # 可能不存在
                every_ms=data["schedule"].get("everyMs"),  # 可能不存在
                expr=data["schedule"].get("expr"),  # 可能不存在
            ),
            payload=CronPayload(
                kind=data["payload"].get("kind", "agent_turn"),  # 默认为 agent_turn
                message=data["payload"].get("message", ""),  # 默认空字符串
                deliver=data["payload"].get("deliver", False),  # 默认不推送
            ),
            state=CronJobState(
                next_run_at_ms=data.get("state", {}).get("nextRunAtMs"),  # 可能不存在
                last_run_at_ms=data.get("state", {}).get("lastRunAtMs"),  # 可能不存在
                last_status=data.get("state", {}).get("lastStatus"),  # 可能不存在
                last_error=data.get("state", {}).get("lastError"),  # 可能不存在
            ),
            created_at_ms=data.get("createdAtMs", 0),  # 默认 0
            updated_at_ms=data.get("updatedAtMs", 0),  # 默认 0
            delete_after_run=data.get("deleteAfterRun", False),  # 默认不删除
        )

    @staticmethod
    def _job_to_dict(job: CronJob) -> dict[str, Any]:
        """
        把 `CronJob` 序列化为字典，供写入 JSON 存储使用。

        这是序列化方法，用于将 CronJob 对象转换为 JSON 格式。
        字段名采用驼峰命名法（如 atMs、nextRunAtMs）以保持与历史数据一致。

        Args:
            job: 要序列化的 CronJob 对象

        Returns:
            可被 json.dumps 序列化的字典
        """
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
