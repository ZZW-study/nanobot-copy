"""路径工具模块"""

from pathlib import Path
from nanobot.config.loader import get_path_config


def get_runtime_subdir(name: str) -> Path:
    """返回根数据文件夹下的指定子文件夹路径"""
    return get_path_config().parent / name


def get_workspace_path(workspace: str | None = None) -> Path:
    """返回 AI 工作空间路径，默认 ~/.nanobot/workspace"""
    if workspace:
        return Path(workspace).expanduser()
    return Path.home() / ".nanobot" / "workspace"


def get_cli_history_path() -> Path:
    """返回命令行历史记录文件路径"""
    return Path.home() / ".nanobot" / "history" / "cli_history"
