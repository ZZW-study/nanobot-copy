"""Agent 工具基类
此模块定义了所有工具（Tool）的抽象基类，代理（Agent）通过具体实现
的工具来完成 I/O、文件操作、网络请求等行为。对参数的解析、类型
转换与校验逻辑也封装在此基类中，方便各工具统一处理输入。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from ZBot.agent.base_agent import BaseAgent


def format_tool_error(
    error: str,
    *,
    attempted: str,
    observed: str | None = None,
    do_not_repeat: str | None = None,
    next_action: str | None = None,
) -> str:
    """生成可行动的工具失败结果。

    工具失败时不要只返回 error；模型只会看到 tool_result，
    所以这里统一告诉它：刚才尝试了什么、观察到了什么、不要重复什么、
    下一步应该如何获得新信息。
    """
    lines = [f"错误：{error}", f"已尝试：{attempted}"]
    if observed:
        lines.append(f"观察结果：{observed}")
    if do_not_repeat:
        lines.append(f"不要重复：{do_not_repeat}")
    if next_action:
        lines.append(f"建议下一步：{next_action}")
    return "\n".join(lines)


class Tool(ABC):
    """工具抽象基类。"""

    # JSON Schema 的基础类型到 Python 类型的映射，用于快速判断与 isinstance
    # ClassVar[...]
    # 标注这是类变量（定义在类体内部、方法外部，属于整个类，所有实例共享同一份。），不属于实例属性，只在类层面定义，实例不能覆盖 / 单独赋值（实例对象不能动他）。
    # 外层容器：dict[str, ...]
    # 字典：key 是字符串，value 是后面的联合类型。
    # value 联合类型：type[Any] | tuple[type[Any], ...]
    # type[Any]：任意类本身（不是实例，是类对象，如 int、str 这种）
    # tuple[type[Any], ...]：不定长元组，里面每个元素都是一个类
    # | 二选一：字典的值可以是单个类，也可以是多个类组成的元组
    _TYPE_MAP: ClassVar[dict[str, type[Any] | tuple[type[Any], ...]]] = {
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    # 所属的 Agent 实例（由 Agent 注册工具时自动设置，子类可通过此属性访问父 Agent 的配置和能力）
    agent: BaseAgent | None = None

    def bind_agent(self, agent: BaseAgent) -> None:
        """绑定所属的 Agent 实例，由 ToolRegistry.register 调用。"""
        self.agent = agent

    # 子类必须要实现的抽象属性
    @property
    @abstractmethod
    def name(self) -> str:
        """工具名称（唯一标识）。"""
        pass


    @property
    @abstractmethod
    def description(self) -> str:
        """工具的简短描述，用于在调用方生成帮助或提示。"""
        pass


    @property
    @abstractmethod
    def parameters(self) -> dict[str, Any]:
        """工具参数的 JSON Schema 定义（根类型通常为 object）。"""
        pass


    @abstractmethod
    async def execute(self, **kwargs: Any) -> str:
        """执行工具的主入口，子类实现具体业务逻辑并返回字符串结果。"""
        pass


    def cast_params(self, params: dict[str, Any]) -> dict[str, Any]:
        """根据工具的 `parameters` schema 对传入的参数进行类型转换。

        规则：若 schema 根类型不是 `object` 则直接返回原参数；否则递归
        地对 object 的每个属性按子 schema 进行转换。
        """
        schema = self.parameters
        # 仅对 object 类型进行递归转换
        if schema.get("type", "object") != "object":
            return params
        return self._cast_object(params, schema)


    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """验证参数是否符合 `parameters` schema，返回错误消息列表（空表示通过）。

        基本规则：传入参数必须是 dict，且 schema 的根类型应为 object。
        具体校验通过 `_validate` 递归完成。
        """
        if not isinstance(params, dict):
            return [f"参数必须是对象类型，当前收到的是 {type(params).__name__}"]
        schema = self.parameters or {}
        if schema.get("type", "object") != "object":
            raise ValueError(f"工具参数的 Schema 根类型必须是 object")
        return self._validate(params, {**schema, "type": "object"}, "")


    def to_schema(self) -> dict[str, Any]:
        """把工具元信息转换为类似 OpenAI 函数调用所需的描述字典。"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


    def _cast_object(self, obj: Any, schema: dict[str, Any]) -> dict[str, Any]:
        """按对象 schema 转换字典参数值。"""
        # 如果传入不是字典结构，无法按 properties 转换，原样返回
        if not isinstance(obj, dict):
            return obj

        props = schema.get("properties", {})
        result = {}

        # 遍历所有键，若在 schema 中定义则按子 schema 转换，否则直接复制
        for key, value in obj.items():
            if key in props:
                result[key] = self._cast_value(value, props[key])
            else:
                result[key] = value

        return result


    def _cast_value(self, val: Any, schema: dict[str, Any]) -> Any:
        """根据子 schema 的 type 字段把单个值转换为目标类型。

        转换策略：
        - 若值已是目标类型则直接返回
        - 对于字符串形式的数字/整数尝试 parse
        - 对数组/对象递归处理
        - 布尔值对字符串做常见真值/假值解析
        - 其它情况保持原值
        """
        target_type = schema.get("type")

        # --- 已经是目标类型的快速路径（避免重复转换） ---
        if target_type == "boolean" and isinstance(val, bool):
            return val
        if target_type == "integer" and isinstance(val, int) and not isinstance(val, bool):
            return val
        if target_type in self._TYPE_MAP and target_type not in ("boolean", "integer", "array", "object"):
            expected = self._TYPE_MAP[target_type]
            if isinstance(val, expected):
                return val

        # --- 字符串转整数 / 浮点数 ---
        if target_type == "integer" and isinstance(val, str):
            try:
                return int(val)
            except ValueError:
                return val

        if target_type == "number" and isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return val

        # --- 强制转为字符串（除了 None） ---
        if target_type == "string":
            return val if val is None else str(val)

        # --- 字符串解析为布尔（常见文本形式） ---
        if target_type == "boolean" and isinstance(val, str):
            val_lower = val.lower()
            if val_lower in ("true", "1", "yes"):
                return True
            if val_lower in ("false", "0", "no"):
                return False
            return val

        # --- 数组的每个元素按 items schema 递归转换 ---
        if target_type == "array" and isinstance(val, list):
            item_schema = schema.get("items")
            return [self._cast_value(item, item_schema) for item in val] if item_schema else val

        # --- 对象类型递归转换 ---
        if target_type == "object" and isinstance(val, dict):
            return self._cast_object(val, schema)

        # 其它情况保持原值
        return val


    def _validate(self, val: Any, schema: dict[str, Any], path: str) -> list[str]:
        """递归校验单个值或对象是否满足 schema，返回错误列表。

        参数：
        - val: 被校验的值
        - schema: JSON Schema 的子部分
        - path: 当前字段路径，用于生成更友好的错误定位信息
        """
        t, label = schema.get("type"), path or "参数"

        # 基本类型快速校验
        if t == "integer" and (not isinstance(val, int) or isinstance(val, bool)):
            return [f"{label} 应为整数"]
        if t == "number" and (not isinstance(val, self._TYPE_MAP[t]) or isinstance(val, bool)):
            return [f"{label} 应为数字"]
        if t in self._TYPE_MAP and t not in ("integer", "number") and not isinstance(val, self._TYPE_MAP[t]):
            return [f"{label} 的类型应为 {t}"]

        errors = []

        # 枚举约束
        if "enum" in schema and val not in schema["enum"]:
            errors.append(f"{label} 必须是以下值之一：{schema['enum']}")

        # 数值边界检查
        if t in ("integer", "number"):
            if "minimum" in schema and val < schema["minimum"]:
                errors.append(f"{label} 必须大于等于 {schema['minimum']}")
            if "maximum" in schema and val > schema["maximum"]:
                errors.append(f"{label} 必须小于等于 {schema['maximum']}")

        # 字符串长度约束
        if t == "string" and isinstance(val, str):
            if "minLength" in schema and len(val) < schema["minLength"]:
                errors.append(f"{label} 长度不能少于 {schema['minLength']} 个字符")
            if "maxLength" in schema and len(val) > schema["maxLength"]:
                errors.append(f"{label} 长度不能超过 {schema['maxLength']} 个字符")

        # 对象的属性与必填项检查
        if t == "object":
            if not isinstance(val, dict):
                return errors
            props = schema.get("properties", {})

            if "minProperties" in schema and len(val) < schema["minProperties"]:
                errors.append(f"{label} 至少需要 {schema['minProperties']} 个字段")
            if "maxProperties" in schema and len(val) > schema["maxProperties"]:
                errors.append(f"{label} 不能超过 {schema['maxProperties']} 个字段")

            for k in schema.get("required", []):
                if k not in val:
                    errors.append(f"缺少必填字段：{path + '.' + k if path else k}")
            for k, v in val.items():
                if k in props:
                    errors.extend(self._validate(v, props[k], path + "." + k if path else k))

        # 数组项的逐一校验
        if t == "array" and "items" in schema and isinstance(val, list):
            if "minItems" in schema and len(val) < schema["minItems"]:
                errors.append(f"{label} 至少需要 {schema['minItems']} 项")
            if "maxItems" in schema and len(val) > schema["maxItems"]:
                errors.append(f"{label} 不能超过 {schema['maxItems']} 项")

            unique_by = schema.get("x-uniqueBy")
            if isinstance(unique_by, str):
                seen_values: set[Any] = set()
                for i, item in enumerate(val):
                    if not isinstance(item, dict):
                        continue
                    unique_value = item.get(unique_by)
                    if unique_value is None:
                        continue
                    if unique_value in seen_values:
                        errors.append(f"{label}[{i}].{unique_by} 不能重复：{unique_value}")
                    seen_values.add(unique_value)

            for i, item in enumerate(val):
                errors.extend(self._validate(item, schema["items"], f"{path}[{i}]" if path else f"[{i}]"))

        return errors
