"""定时任务工具 (CronTool)
功能：为AI智能体提供 定时提醒、循环任务、一次性定时任务 的调度能力
支持三种调度模式：秒级循环、Cron表达式、指定ISO时间执行
"""
from contextvars import ContextVar
from datetime import datetime
from typing import Any

from nanobot.agent.tools.base import Tool
from nanobot.cron.service import BEIJING_TZ, CronService
from nanobot.cron.types import CronSchedule


class CronTool(Tool):
    """
    AI智能体专用定时任务工具
    继承自Tool基类，符合框架工具标准，可被大模型直接调用
    核心功能：添加定时任务、查看任务列表、删除指定任务
    """

    def __init__(self, cron_service: CronService):
        """
        构造函数：初始化定时任务工具
        :param cron_service: 注入底层定时任务服务实例（依赖注入，解耦业务逻辑）
        """
        # 底层定时任务服务实例，真正执行任务调度的核心对象
        self._cron = cron_service
        # 异步安全的上下文变量：标记当前是否正在执行定时任务回调
        # 作用：禁止在定时任务内部创建新任务，防止嵌套死循环
        self._in_cron_context: ContextVar[bool] = ContextVar("cron_in_context", default=False)

    def set_cron_context(self, active: bool):
        """
        设置定时任务执行状态标记
        :param active: True=当前正在执行定时任务回调 | False=未执行
        :return: 上下文令牌，用于后续恢复状态
        """
        return self._in_cron_context.set(active)

    def reset_cron_context(self, token) -> None:
        """
        重置定时任务上下文状态
        执行完任务后调用，恢复初始状态，避免影响后续操作
        :param token: set_cron_context 返回的上下文令牌
        """
        self._in_cron_context.reset(token)

    # ==================== 框架强制要求的工具元数据（供大模型识别和调用） ====================
    @property
    def name(self) -> str:
        """工具唯一名称，大模型通过该名称调用此工具"""
        return "cron"

    @property
    def description(self) -> str:
        """工具功能描述，让大模型理解工具的用途"""
        return "创建、查看和删除定时提醒或循环任务。支持 add、list、remove 三种动作。"

    @property
    def parameters(self) -> dict[str, Any]:
        """
        工具调用参数规范（JSON Schema格式）
        作用：告诉大模型调用该工具需要传入哪些参数、参数类型、使用场景
        """
        return {
            "type": "object",
            "properties": {
                # 必选参数：执行动作（添加/列表/删除）
                "action": {
                    "type": "string",
                    "enum": ["add", "list", "remove"],  # 限定仅支持这三个动作
                    "description": "要执行的动作：add、list 或 remove。",
                },
                # 可选参数：提醒消息（添加任务时必填）
                "message": {"type": "string", "description": "提醒内容（action=add 时必填）。"},
                # 可选参数：循环间隔（秒），用于循环执行任务
                "every_seconds": {
                    "type": "integer",
                    "description": "循环任务的间隔秒数。",
                },
                # 可选参数：Cron表达式（如 0 9 * * * = 每天早上9点）
                "cron_expr": {
                    "type": "string",
                    "description": "北京时间的 Cron 表达式，例如 '0 9 * * *' 表示每天上午 9 点。",
                },
                # 可选参数：ISO格式时间，用于一次性定时任务
                "at": {
                    "type": "string",
                    "description": "北京时间的 ISO 时间，例如 '2026-02-12T10:30:00'。",
                },
                # 可选参数：任务ID（删除任务时必填）
                "job_id": {"type": "string", "description": "任务 ID（action=remove 时必填）。"},
            },
            "required": ["action"],  # 仅action为必填参数，其余根据动作动态必填
        }

    # ==================== 工具执行入口（大模型调用后，实际执行业务逻辑） ====================
    async def execute(
        self,
        action: str,                # 执行动作：add(添加)/list(列表)/remove(删除)
        message: str = "",          # 提醒消息内容
        every_seconds: int | None = None,  # 循环秒数
        cron_expr: str | None = None,      # Cron表达式
        at: str | None = None,             # 一次性任务执行时间
        job_id: str | None = None,         # 任务ID
        **kwargs: Any,                     # 兼容额外参数，保证扩展性
    ) -> str:
        """
        异步执行工具逻辑（适配AI框架异步架构）
        根据action参数分发到对应的业务处理方法
        :return: 执行结果文本，返回给大模型/用户
        """
        # 动作1：添加定时任务
        if action == "add":
            # 安全校验：禁止在定时任务内部创建新任务，防止嵌套死循环
            if self._in_cron_context.get():
                return "错误：不能在定时任务执行过程中再次创建新的定时任务。"
            # 调用私有方法创建任务
            return self._add_job(message, every_seconds, cron_expr, at)

        # 动作2：列出所有定时任务
        elif action == "list":
            return self._list_jobs()

        # 动作3：删除定时任务
        elif action == "remove":
            return self._remove_job(job_id)

        # 未知动作，返回错误提示
        return f"错误：未知动作 {action}"

    # ==================== 私有方法：添加定时任务（核心业务逻辑） ====================
    def _add_job(
        self,
        message: str,               # 提醒消息
        every_seconds: int | None,  # 循环秒数
        cron_expr: str | None,      # Cron表达式
        at: str | None,             # 执行时间
    ) -> str:
        """
        内部方法：创建并保存定时任务
        包含完整的参数校验、调度类型判断、任务创建逻辑
        """
        # 校验1：添加任务必须填写提醒消息
        if not message:
            return "错误：创建任务时必须提供提醒内容。"

        # ==================== 构建任务调度规则 ====================
        # 标记：一次性任务执行后是否自动删除（循环任务为False）
        delete_after = False

        # 类型1：循环任务（按秒执行，如每30秒执行一次）
        if every_seconds:
            # 转换为毫秒（底层服务使用毫秒单位）
            schedule = CronSchedule(kind="every", every_ms=every_seconds * 1000)

        # 类型2：Cron表达式定时任务（如每天、每周、每月）
        elif cron_expr:
            schedule = CronSchedule(kind="cron", expr=cron_expr)

        # 类型3：一次性定时任务（指定ISO时间执行）
        elif at:
            try:
                # 解析ISO格式时间字符串
                dt = datetime.fromisoformat(at)
            except ValueError:
                return f"错误：时间格式无效：{at}。正确格式示例：YYYY-MM-DDTHH:MM:SS"
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=BEIJING_TZ)
            else:
                dt = dt.astimezone(BEIJING_TZ)
            # 转换为毫秒时间戳
            at_ms = int(dt.timestamp() * 1000)
            schedule = CronSchedule(kind="at", at_ms=at_ms)
            # 一次性任务执行完成后自动删除
            delete_after = True

        # 校验5：必须指定一种调度方式（循环/cron/一次性）
        else:
            return "错误：必须提供 every_seconds、cron_expr 或 at 其中之一。"

        # 调用底层服务，创建定时任务
        job = self._cron.add_job(
            name=message[:30],          # 任务名称：截取消息前30字符，避免过长
            schedule=schedule,           # 调度规则
            message=message,             # 完整提醒消息
            delete_after_run=delete_after,  # 执行后是否自动删除
        )

        # 返回创建成功提示，包含任务ID（用于后续删除）
        return f"已创建任务 {job.name}（ID：{job.id}）"

    # ==================== 私有方法：查询所有定时任务 ====================
    def _list_jobs(self) -> str:
        """获取并格式化输出所有已创建的定时任务"""
        # 从底层服务获取所有任务
        jobs = self._cron.list_jobs()
        # 无任务时返回空提示
        if not jobs:
            return "当前没有已安排的定时任务。"
        # 格式化任务列表：名称 + ID + 调度类型
        lines = [f"- {j.name}（ID：{j.id}，类型：{j.schedule.kind}）" for j in jobs]
        return "当前定时任务列表：\n" + "\n".join(lines)

    # ==================== 私有方法：删除指定定时任务 ====================
    def _remove_job(self, job_id: str | None) -> str:
        """根据任务ID删除定时任务"""
        # 校验：删除任务必须传入任务ID
        if not job_id:
            return "错误：删除任务时必须提供 job_id。"
        # 执行删除操作
        if self._cron.remove_job(job_id):
            return f"已删除任务：{job_id}"
        # 任务ID不存在，返回错误
        return f"错误：未找到任务 {job_id}"
