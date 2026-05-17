# 子agent创建工具
from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING, Any, TypedDict

from ZBot.agent.subagent.subagent_pool import SUBAGENT_POLICY
from ZBot.agent.tools.base import Tool, format_tool_error

if TYPE_CHECKING:
    from ZBot.agent.subagent.subagent_pool import SubAgentPool


# TypedDict 是 Python 的类型提示工具，定义一个"有固定 key 和类型的字典"结构。
# 和普通 dict 的区别：TypedDict 只在类型检查阶段起作用，运行时和普通 dict 完全一样。
# 好处：IDE 和类型检查器（mypy/pyright）能检查你传的字典是否缺字段、字段类型是否正确。
class SubtaskInput(TypedDict):
    subtask_id: str
    task: str
    expected_result: str


class CreateSubAgentTool(Tool):
    @property
    def name(self) -> str:
        """返回创建子 Agent 工具的名称。"""
        return "create_sub_agent"

    @property
    def description(self) -> str:
        """返回创建子 Agent 工具的使用说明。"""
        return (
            "仅当任务已被主agent拆成多个互不阻塞、可并行执行、最终只需汇总的明确子任务时使用。"
            "每个子任务必须包含subtask_id、任务内容和预期结果。"
            "工具会并行执行子agent，并返回自然语言汇总：逐条说明哪些子任务完成、哪些失败、失败原因和建议下一步。"
            "最终验证、失败决策和整合仍由主agent负责。"
        )

    @property
    def parameters(self) -> dict[str, Any]:
        """返回创建子 Agent 工具的参数 Schema。"""
        # 返回 JSON Schema 格式的参数描述，供 LLM 理解如何调用这个工具。
        # type/properties/required 是标准 JSON Schema 字段。
        return {
            "type": "object",
            "properties": {
                "subtasks": {
                    "type": "array",
                    "description": "可并行执行的子任务列表。有几个独立并行子任务，就创建几个子agent。",
                    "items": {
                        "type": "object",
                        "properties": {
                            "subtask_id": {
                                "type": "string",
                                "minLength": 1,
                                "description": "子任务唯一标识，例如 task_1。",
                            },
                            "task": {
                                "type": "string",
                                "minLength": 1,
                                "description": "明确的执行任务，必须包含输入范围、执行目标和输出要求。",
                            },
                            "expected_result": {
                                "type": "string",
                                "minLength": 1,
                                "description": "父agent预期得到的结果，用于后续判断子任务是否可用。",
                            },
                        },
                        "required": [
                            "subtask_id",
                            "task",
                            "expected_result",
                        ],
                    },
                    # 至少 2 个子任务才有并行的意义；上限由策略配置控制，避免资源滥用。
                    "minItems": 2,
                    "maxItems": SUBAGENT_POLICY.max_count,
                    "x-uniqueBy": "subtask_id",
                }
            },
            "required": ["subtasks"],
        }


    async def execute(self, **kwargs: Any) -> str:
        """并行执行多个已拆解的子 Agent 任务并汇总结果。"""
        # 参数合法性已经由 ToolRegistry 统一调用 Tool.validate_params() 完成。
        # 这里直接进入业务流程，避免在工具内部重复写字段校验逻辑。
        subtasks = kwargs["subtasks"]

        # self.agent 是工具绑定的父 Agent 实例，由框架在工具注册时注入。
        # 父 Agent 提供消息链和子 Agent 池，是创建子 Agent 的必要前提。
        parent = self.agent
        if parent is None:
            return format_tool_error(
                "工具未绑定到任何 Agent 实例",
                attempted="调用 create_sub_agent",
                observed="当前工具没有父 Agent 上下文",
                do_not_repeat="不要在未绑定 Agent 的情况下重复调用 create_sub_agent",
                next_action="让主 Agent 注册并绑定该工具后再调用",
            )

        # 父 Agent 的当前消息链会传给子 Agent，让子 Agent 了解完整上下文。
        parent_messages = parent._current_messages_for_subagent
        if parent_messages is None:
            return format_tool_error(
                "缺少父 Agent 当前消息链，无法创建子 Agent",
                attempted="调用 create_sub_agent",
                observed="父 Agent 当前没有可交给子 Agent 的消息快照",
                do_not_repeat="不要在非工具调用上下文中重复调用 create_sub_agent",
                next_action="由主 Agent 在正常 run_agent_loop 工具调用阶段重新发起",
            )

        # getattr 安全取属性(方法也是属性字典的属性)：如果父 Agent 没有 ensure_subagent_pool 方法，返回 None 而不是报错。
        # 这样可以兼容不支持子 Agent 的父 Agent 类型。
        subagent_pool = getattr(parent, "subagent_pool", None)
        if subagent_pool is None:
            return format_tool_error(
                "父 Agent 未提供子 Agent 池，无法创建子 Agent",
                attempted="调用 create_sub_agent",
                observed=f"父 Agent 类型：{type(parent).__name__}",
                do_not_repeat="不要让不支持子 Agent 的 Agent 重复调用 create_sub_agent",
                next_action="改由主 Agent 自己执行任务，或使用支持 SubAgentPool 的 CoreAgent",
            )
        # asyncio.gather 并发执行所有子任务，不需要等第一个完成再执行第二个。
        # _run_one_subagent 内部会把普通异常转成失败结果，因此这里不再重复处理异常对象。
        results: list[dict[str, Any]] = await asyncio.gather(  
            *(
                self._run_one_subagent(
                    subagent_pool,
                    parent_messages,
                    subtask,
                )
                for subtask in subtasks
            )
        )
        # 返回结果，如果有子agent的任务失败了，则给出建议
        response: list[str] = []
        for result in results:
            if result["status"] == "completed":
                response.append(
                    f"任务ID：{result['subtask_id']}，"
                    f"任务名字为 {result['subtask_name']} 的任务已经完成，"
                    f"任务结果为：{result['result']}"
                )
            else:
                response.append(
                    f"任务ID：{result['subtask_id']}，"
                    f"任务名字为 {result['subtask_name']} 的任务失败，"
                    f"失败原因为：{result['error']}"
                )
                response.append(
                    "建议下一步：请主 Agent 根据失败原因判断该子任务是否仍然必要，不要机械重试。"
                    "可选策略包括：补充上下文后重新派发、调整任务拆分、改用普通工具自行处理，"
                    "或在最终结果中标记该部分缺失。"
                )
        return "\n".join(response)


    async def _run_one_subagent(
        self,
        subagent_pool: SubAgentPool,
        parent_messages: list[dict[str, Any]],
        subtask: SubtaskInput,
    ) -> dict[str, Any]:
        """租用一个子 Agent 执行单个子任务。"""
        timeout_seconds = getattr(
            getattr(self.agent, "runtime_config", None),
            "subagent_timeout_seconds",
            SUBAGENT_POLICY.timeout_seconds,
        )
        try:
            # async with ... as lease：从池里借一个子 Agent，用完自动归还（上下文管理器）。
            # lease 包含 agent_id 和 agent 实例。
            async with subagent_pool.acquire() as lease:
                # asyncio.wait_for 设置超时：子 Agent 超过 timeout_seconds 秒未完成，
                # 自动取消并抛出 TimeoutError，避免无限等待阻塞主流程。
                result: str = await asyncio.wait_for(
                    lease.agent.process_messages(
                        parent_messages,
                        subtask_id=subtask["subtask_id"],
                        task_description=subtask["task"],
                        expected_result=subtask["expected_result"],
                        on_progress=self._make_subagent_progress(lease.agent_id),
                    ),
                    timeout=timeout_seconds,
                )
        except TimeoutError:
            return self._failed_result(
                subtask,
                error=f"子 Agent 执行超过 {timeout_seconds} 秒",
                )
        except Exception as exc:
            return self._failed_result(
                subtask,
                error=str(exc),
            )

        # 子 Agent 正常返回后，只归一化成父 Agent 易读的事实结构。
        return {
            "subtask_id": subtask["subtask_id"],
            "subtask_name": subtask["task"],
            "status": "completed",
            "error": "",
            "result": result,
        }


    def _make_subagent_progress(self, agent_id: str):
        """为指定子 Agent 创建进度转发回调。"""
        # 工厂函数：为每个子 Agent 生成一个专属的进度回调函数，
        # 闭包捕获 agent_id，让回调知道自己属于哪个子 Agent。
        async def _progress(content: str, *, tool_hint: bool = False, **kwargs: Any) -> None:
            """把子 Agent 进度转发到父 Agent 的当前回调。"""
            parent = self.agent
            if parent is None:
                return
            # 父 Agent 的 run_agent_loop 会在本轮执行期间记录外部传入的真实回调。
            # 子 Agent 只负责把自己的状态转发给父 Agent 当前回调，不自己创建新的显示通道。
            progress = parent._active_progress_callback
            if progress is None:
                raise RuntimeError("父 Agent 当前没有可用的进度回调")
            # 去掉调用方可能传入的 agent_label，统一由这里注入标准格式的标签。
            kwargs.pop("agent_label", None)
            label = agent_id.replace("subagent_", "子agent")
            await progress(content, tool_hint=tool_hint, agent_label=label, **kwargs)

        return _progress


    @staticmethod
    def _failed_result(
        subtask: dict[str, Any] | SubtaskInput,
        *,
        error: str,
    ) -> dict[str, Any]:
        """构造统一的子任务失败结果。"""
        # 统一的失败结构工厂，所有失败路径（超时/异常/主动上报）都经过这里，
        # 保证 structured_results 里每条失败记录的字段格式完全一致，
        # 父 Agent 只需判断 status == "failed" 就能统一处理，不需要针对不同失败原因写不同逻辑。
        return {
            "subtask_id": subtask.get("subtask_id", "unknown"),
            "subtask_name": subtask.get("task", "unknown"),
            "status": "failed",
            "error": error,
        }
