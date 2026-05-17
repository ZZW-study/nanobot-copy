"""子agent模块"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Awaitable, Callable
from loguru import logger

from ZBot.config.agent_runtime import AgentRuntimeConfig
from ZBot.providers.base import LLMProvider
from ZBot.agent.tools.registry import ToolRegistry 
from ZBot.agent.base_agent import BaseAgent



class SubAgent(BaseAgent):
    """一次性并行执行单元，只执行父 Agent 派发的明确子任务。

    这个类不是长期会话主体，不持有用户会话、不写记忆、不做最终决策。
    它可以被 SubAgentPool 长期复用；每次执行任务时都重新复制父 Agent
    传入的消息链，因此不会残留上一轮子任务的消息或参数。
    """

    # parents[3]：向上回溯 3 层上级目录，因此 parents[3] 指回 ZBot 根目录，再进入 templates/SUBAGENT.md。
    _SUBAGENT_RULES_PATH = Path(__file__).parents[3] / "templates" / "SUBAGENT.md"


    def __init__(
        self,
        provider: LLMProvider,
        runtime_config: AgentRuntimeConfig,
        parent_tools: ToolRegistry | None = None,
    ):
        """初始化子 Agent 执行单元。

        `provider` 和 `runtime_config` 与父 Agent 共享，用来保证子 Agent
        在同一模型、同一工作区和同一工具配置下执行。`parent_tools`
        只用于复制父 Agent 已连接的 MCP 工具。子 Agent 不接收 CronService，
        因此不会注册 cron 工具，避免一次性执行单元创建后台任务。
        """

        # 实例化父类，以获得父类的实例属性，相当于传入self，把父类的实例属性都变成子类的实例属性
        super().__init__(
            provider=provider,
            runtime_config=runtime_config,
            cron_service=None,
        )

        # 子 Agent 自己初始化默认普通工具，只从父 Agent 复制已连接的 MCP 工具。
        # 不复制 create_sub_agent，避免形成递归调度。
        if not parent_tools:
            raise ValueError("错误：未找到父agent的工具")
        else:
            copied = self.tools.register_from(
                parent_tools,
                name_prefix="mcp_",
                exclude_names={"create_sub_agent"},
            )
            logger.info("子agent复用父agent的MCP连接，已复制 {} 个MCP工具", copied)


    @classmethod
    def from_parent(cls, parent: BaseAgent) -> "SubAgent":
        """基于调度它的父 Agent 创建一个可复用的子 Agent 实例。

        这个方法只在 SubAgentPool 预创建执行单元时调用，不应该在每次
        `create_sub_agent` 工具调用时批量调用。
        """
        return cls(    # 实例化
            provider=parent.provider,
            runtime_config=parent.runtime_config,
            parent_tools=parent.tools,
        )



    async def process_messages(
        self,
        messages: list[dict[str, Any]],
        *,
        subtask_id: str,
        task_description: str,
        expected_result: str,
        on_progress: Callable[..., Awaitable[None]],
    ) -> str:
        """基于父 Agent 已构造好的消息链执行子任务。

        这里会深拷贝父 Agent 给出的消息链，然后追加子 Agent 规则和本次
        子任务说明。子 Agent 内部工具调用只写入这条临时消息链，执行完成
        后只返回最终文本，不把中间工具链写回父 Agent 会话。
        """
        # 深拷贝能隔离 tool_calls 这类嵌套结构，避免多个子 Agent 并行时共享内部对象。
        sub_messages = copy.deepcopy(messages)
        # 子 Agent 规则放在第一条 system，优先于父 Agent 原 system prompt。
        sub_messages.insert(0, {"role": "system", "content": self._subagent_rules()})
        # 本次子任务说明由父 Agent 预先拆解完成，子 Agent 只负责执行。
        sub_messages.append(
            {
                "role": "user",
                "content": (
                    f"子任务ID：{subtask_id}\n"
                    f"任务描述：{task_description}\n"
                    f"预期结果：{expected_result}"
                ),
            }
        )

        final_content, _, _ = await self.run_agent_loop(
            sub_messages,
            on_progress=on_progress,
            progress_label=subtask_id,
        )
        return final_content or "子agent已完成处理，但没有需要额外返回的内容。"
    
    @classmethod
    def _subagent_rules(cls) -> str:
        """读取子 Agent 专用规则文本。

        模板文件存在时使用模板；不存在时返回最小兜底规则，保证子 Agent
        仍然不会变成长期会话主体。
        """
        if cls._SUBAGENT_RULES_PATH.exists():
            return cls._SUBAGENT_RULES_PATH.read_text(encoding="utf-8")
        
        return (
            "你是子 Agent，只负责完成父 Agent 分配的明确子任务。"
            "如果父 Agent 的历史 system prompt 和本规则冲突，以本规则为准。"
            "不要创建子 Agent，不要写入记忆，不要和用户直接交互。"
        )
