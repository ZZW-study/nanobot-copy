"""路径工具模块。

本模块封装了 nanobot 中常用的路径计算逻辑，
统一管理工作区、配置和数据子目录的位置，
避免各模块重复硬编码路径。
"""

from pathlib import Path  # 面向对象的文件路径处理类
from nanobot.config.loader import get_path_config  # 获取配置文件路径的函数


def get_runtime_subdir(name: str) -> Path:
    """返回根数据文件夹下的指定子文件夹路径。

    例如 get_runtime_subdir("cron") 会返回 ~/.nanobot/cron，
    用于存放定时任务等运行时数据。

    参数：
        name: 子目录名称（如 "cron"、"logs" 等）

    返回：
        该子目录的完整 Path 对象
    """
    # get_path_config() 返回配置文件路径，取其父目录作为根数据文件夹
    return get_path_config().parent / name


def get_workspace_path(workspace: str | None = None) -> Path:
    """返回 AI 工作空间路径。

    工作空间是 nanobot 存放所有数据（会话、记忆、技能等）的根目录。

    参数：
        workspace: 如果显式传入了工作区路径，则直接返回；
                   如果为 None，则使用默认路径 ~/.nanobot/workspace

    返回：
        工作空间目录的 Path 对象
    """
    # 如果用户指定了路径，展开 ~ 后直接返回
    if workspace:
        return Path(workspace).expanduser()
    # 否则返回默认工作区路径：家目录/.nanobot/workspace
    return Path.home() / ".nanobot" / "workspace"


def get_cli_history_path() -> Path:
    """返回命令行历史记录文件路径。

    该文件用于存储用户在 CLI 交互模式下输入过的命令历史，
    方便使用上下键快速翻阅之前的输入内容。

    返回：
        历史记录文件的 Path 对象（~/.nanobot/history/cli_history）
    """
    return Path.home() / ".nanobot" / "history" / "cli_history"
