"""工具注册与调度中心。
`ToolRegistry` 的职责很单纯：
1. 保存所有可被 Agent 调用的工具实例。
2. 输出给模型看的工具 schema。
3. 在真正执行前集中完成参数转换、参数校验和错误包装。
这样 Agent 主循环就不需要重复关心每个工具的细节。
"""
from __future__ import annotations
from typing import Any
from ZBot.agent.tools.base import Tool, format_tool_error

_RETRY_HINT = "\n\n[工具执行失败。不要用相同参数重复调用；请先根据错误原因调整参数，或改用更合适的工具获取新信息。]"


class ToolRegistry:
    """保存工具实例并提供统一执行入口。"""

    def __init__(self):
        """初始化工具名称到工具实例的映射。"""
        # 内部工具映射：name -> Tool 实例
        # 通过注册（register）注入工具，执行时统一从这里取出实例。
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """注册工具。
        同名工具会被后注册的实例覆盖，这是刻意保留的行为，
        方便外部注入定制版本。
        """
        # 直接以工具名覆盖已有实例，允许外部通过同名工具替换默认实现
        self._tools[tool.name] = tool

    def register_from(
        self,
        other: "ToolRegistry",
        *,
        name_prefix: str | None = None,
        exclude_names: set[str] | None = None,
    ) -> int:
        """从另一个注册中心复制工具引用，返回复制数量。

        主要给 SubAgent 使用：子 Agent 自己有默认普通工具，但需要复用父 Agent
        已经连接好的 MCP 工具。`name_prefix` 可以只复制 mcp_ 开头的工具，
        `exclude_names` 可以排除 create_sub_agent 这类子 Agent 不能拥有的工具。
        """
        count = 0
        excluded = exclude_names or set()
        for name, tool in other._tools.items():
            if name in excluded:
                continue
            if name_prefix is not None and not name.startswith(name_prefix):
                continue
            self.register(tool)
            count += 1
        return count


    def get_definitions(self) -> list[dict[str, Any]]:
        """返回所有工具 schema，供大模型决定是否进行函数调用。"""
        # 将所有工具转换为模型可识别的 schema（name/parameters/description），
        # 这些 schema 会随 messages 一并下发给 LLM，允许模型做函数调用决策。
        return [tool.to_schema() for tool in self._tools.values()]


    async def execute(self, name: str, params: dict[str, Any]) -> str:
        """执行指定工具并统一包装错误。"""
        tool = self._tools.get(name)
        if tool is None:
            # 列出可用工具名提示用户（_tools 的迭代默认返回键名）
            available = "、".join(self._tools)
            return format_tool_error(
                f"找不到工具“{name}”",
                attempted=f"调用工具 {name}",
                observed=f"当前可用工具：{available}",
                do_not_repeat=f"不要继续调用不存在的工具：{name}",
                next_action="改用可用工具，或直接说明当前能力无法完成该工具动作",
            )

        try:
            # 先把模型传入的原始参数做类型转换（cast）和规范化，
            # 例如把 JSON 数字转换为 int、把字符串解析为期望的子结构等。
            cast_params = tool.cast_params(params)
            # 然后进行语义/格式校验，返回错误列表（若有）
            errors = tool.validate_params(cast_params)
            if errors:
                # 参数不合法时直接返回可读的错误提示，并带上重试建议
                return format_tool_error(
                    f"工具“{name}”的参数不合法：{'；'.join(errors)}",
                    attempted=f"调用工具 {name}，参数：{cast_params}",
                    do_not_repeat="不要用相同参数重复调用该工具",
                    next_action="根据工具 schema 修正参数，或换用更合适的工具",
                ) + _RETRY_HINT

            # 调用工具的异步执行函数，执行过程中工具可能抛出异常或返回错误字符串
            result = await tool.execute(**cast_params)
            # 若工具以字符串形式返回错误（以 `错误：` 开头），附加重试提示并返回
            if isinstance(result, str) and result.startswith("错误："):
                return result + _RETRY_HINT
            return result
        except Exception as exc:
            # 捕获执行期异常并统一格式化为错误返回，避免抛到上层导致崩溃
            return format_tool_error(
                f"执行工具“{name}”时发生异常：{exc}",
                attempted=f"调用工具 {name}，参数：{params}",
                do_not_repeat="不要用相同参数重复调用该工具",
                next_action="根据异常调整参数，或改用其他工具获取新信息",
            ) + _RETRY_HINT

