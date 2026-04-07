"""配置文件加载与保存工具。

这个模块负责把 nanobot 的配置来源统一收口到一处：
1. 先读取磁盘上的 `config.json`。
2. 再叠加 `NANOBOT_*` 环境变量覆盖。
3. 最后交给 Pydantic 做一次标准化校验。

这样上层模块只需要调用 `load_config()`，无需分别处理磁盘、环境变量和默认值。

配置加载优先级（从低到高）：
- Pydantic 模型默认值（最低优先级）
- 磁盘配置文件 config.json
- NANOBOT_* 环境变量（最高优先级）

核心函数：
    load_config(): 加载并合并配置
    save_config(): 保存配置到磁盘
"""

from __future__ import annotations  # 启用未来版本的类型注解特性

import json  # 用于 JSON 文件读写和字符串解析
import os  # 用于访问环境变量
from pathlib import Path  # 用于路径操作
from typing import Any  # 用于类型注解

from pydantic import ValidationError  # 用于捕获 Pydantic 验证错误

from nanobot.config.schema import Config  # 配置模型定义


# 测试或脚本场景下允许临时改写配置文件路径，便于隔离不同运行环境。
# 这是一个全局变量，通常为 None，表示使用默认路径。
# 在测试中可以设置为临时路径以避免影响用户的真实配置。
_current_config_path: Path | None = None


def get_path_config() -> Path:
    """
    返回当前生效的配置文件路径。

    默认路径为：~/.nanobot/config.json
    如果设置了 _current_config_path（测试场景），则返回该路径。

    Returns:
        配置文件的绝对路径
    """
    if _current_config_path is not None:
        return _current_config_path
    # 默认配置文件路径：用户主目录下的 .nanobot/config.json
    return Path.home() / ".nanobot" / "config.json"


def _coerce_env_value(raw: str) -> Any:
    """
    尝试将环境变量的字符串按 JSON 格式解析为 Python 对象；解析失败则保留原始字符串。

    环境变量本质上都是字符串，但有些配置需要布尔值、数字或列表。
    此函数尝试智能转换：

    示例：
    - "true"  -> True  (Python布尔值)
    - "false" -> False (Python布尔值)
    - "123"   -> 123   (Python整数)
    - "45.6"  -> 45.6  (Python浮点数)
    - '["a"]' -> ["a"] (Python列表)
    - '{"key": "value"}' -> {"key": "value"} (Python字典)
    - "abc"   -> "abc" (原始字符串，无法解析为JSON)

    Args:
        raw: 环境变量的原始字符串值

    Returns:
        解析后的 Python 对象，或原始字符串（如果解析失败）
    """
    try:
        # 尝试将字符串解析为 JSON 对象
        return json.loads(raw)
    except json.JSONDecodeError:
        # 如果不是有效的 JSON 格式，返回原始字符串
        return raw


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    递归合并两个字典，并返回新结果。

    合并规则：
    - `override` 中的值优先级更高
    - 当同名字段两边都是字典时，继续向下递归合并
    - 其他情况直接用 override 的值覆盖 base 的值

    示例：
    base = {"a": 1, "b": {"c": 2}}
    override = {"b": {"d": 3}, "e": 4}
    result = {"a": 1, "b": {"c": 2, "d": 3}, "e": 4}

    Args:
        base: 基础字典（低优先级）
        override: 覆盖字典（高优先级）

    Returns:
        合并后的新字典
    """
    merged = dict(base)  # 创建基础字典的副本
    for key, value in override.items():
        current = merged.get(key)
        # 如果两个值都是字典，递归合并
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            # 否则直接覆盖
            merged[key] = value
    return merged


def _load_env_overrides() -> dict[str, Any]:
    """
    把 `NANOBOT_*` 环境变量转换成嵌套配置字典。

    环境变量命名规则：
    - 前缀：NANOBOT_
    - 嵌套分隔符：双下划线 __
    - 字段名自动转为小写

    命名规则示例：
    `NANOBOT_AGENTS__DEFAULTS__MODEL=qwen-plus`
    会被转换成：
    `{"agents": {"defaults": {"model": "qwen-plus"}}}`

    另一个示例：
    `NANOBOT_PROVIDERS__OPENAI__API_KEY=sk-xxx`
    转换为：
    `{"providers": {"openai": {"api_key": "sk-xxx"}}}`

    Returns:
        从环境变量构建的嵌套配置字典
    """
    prefix = "NANOBOT_"
    overrides: dict[str, Any] = {}

    # 遍历所有环境变量
    for key, raw_value in os.environ.items():
        if not key.startswith(prefix):
            continue  # 跳过非 NANOBOT_ 开头的变量

        # 提取变量名部分（去掉前缀），按 __ 分割，并转为小写
        parts = [part.strip().lower() for part in key[len(prefix) :].split("__") if part.strip()]
        if not parts:
            continue  # 跳过无效的变量名

        # 构建嵌套字典结构
        node = overrides
        # 遍历除最后一个部分外的所有部分（用于构建嵌套路径）
        for part in parts[:-1]:
            child = node.get(part)
            if not isinstance(child, dict):
                child = {}
                node[part] = child
            node = child

        # 设置最终的键值对（最后一个部分是字段名）
        node[parts[-1]] = _coerce_env_value(raw_value)

    return overrides


def _normalize_config_data(data: dict[str, Any], *, exclude_unset: bool) -> dict[str, Any]:
    """
    用 `Config` 对配置字典做一次字段标准化。

    这样无论输入里使用驼峰键还是下划线键，最终都会被统一为模型字段名，
    后面的合并逻辑也就不需要关心别名差异。

    Pydantic 的 model_validate() 会：
    1. 验证数据类型是否符合模型定义
    2. 应用字段别名转换（如 camelCase -> snake_case）
    3. 设置默认值（对于未提供的字段）
    4. 执行自定义验证器

    Args:
        data: 原始配置字典
        exclude_unset: 是否排除未显式设置的字段（用于环境变量覆盖场景）

    Returns:
        标准化后的配置字典
    """
    return Config.model_validate(data).model_dump(
        by_alias=False,  # 使用字段名而非别名（确保一致性）
        exclude_unset=exclude_unset,  # 排除未显式赋值的字段（环境变量场景需要）
    )


def load_config(config_path: Path | None = None) -> Config:
    """
    加载配置。

    加载顺序固定为：
    1. 磁盘配置（config.json 文件）
    2. 环境变量覆盖（NANOBOT_* 变量）
    3. Pydantic 最终校验与默认值补全

    处理流程：
    1. 读取配置文件（如果存在）
    2. 解析环境变量覆盖
    3. 深度合并两个配置源
    4. 通过 Pydantic 模型进行最终验证

    Args:
        config_path: 可选的配置文件路径（用于测试或特殊场景）

    Returns:
        验证后的 Config 对象

    Raises:
        ValidationError: 如果最终合并的配置不符合 Pydantic 模型要求
    """
    # 确定配置文件路径
    path = config_path or get_path_config()
    file_data: dict[str, Any] = {}

    # ========== 1. 读取磁盘配置 ==========
    if path.exists():
        try:
            with open(path, encoding="utf-8") as file:
                loaded = json.load(file)  # 从 JSON 格式的文件中读取数据
            if isinstance(loaded, dict):
                # 标准化配置数据（应用 Pydantic 验证和转换）
                file_data = _normalize_config_data(loaded, exclude_unset=False)
            else:
                print(f"警告：配置文件 {path} 的根节点必须是 JSON 对象，已改用默认配置。")
        except (json.JSONDecodeError, OSError, ValidationError, ValueError) as exc:
            print(f"警告：读取配置文件 {path} 失败：{exc}")
            print("无法解析的配置项将回退到默认值。")

    # ========== 2. 加载环境变量覆盖 ==========
    # exclude_unset=True 确保只包含实际设置的环境变量，不影响其他字段
    env_overrides = _normalize_config_data(_load_env_overrides(), exclude_unset=True)

    # ========== 3. 合并配置源 ==========
    # 环境变量覆盖磁盘配置（env_overrides 优先级更高）
    merged = _deep_merge(file_data, env_overrides)

    # ========== 4. 最终验证 ==========
    # 通过 Pydantic 模型进行完整验证，包括类型检查、约束验证等
    return Config.model_validate(merged)


def save_config(config: Config, config_path: Path | None = None) -> None:
    """
    把配置对象写回磁盘。

    此函数主要用于：
    - CLI 配置命令（如 /config set）
    - 首次运行时的配置初始化
    - 配置迁移或更新

    Args:
        config: 要保存的 Config 对象
        config_path: 可选的配置文件路径（用于测试）
    """
    # 确定配置文件路径
    path = config_path or get_path_config()
    # 确保父目录存在（~/.nanobot/）
    path.parent.mkdir(parents=True, exist_ok=True)

    # 将 Config 对象转换为字典（使用字段别名，如 camelCase）
    data = config.model_dump(by_alias=True)
    # 写入 JSON 文件（格式化缩进，支持中文）
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
