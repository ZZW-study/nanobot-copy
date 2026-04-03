"""Agent core module."""

from nanobot.agent.context import ContextBuilder
from nanobot.agent.memory import MemoryStore
from nanobot.agent.skills import SkillsLoader

# AgentLoop requires litellm dependency, import lazily or explicitly
# from nanobot.agent.loop import AgentLoop

__all__ = ["ContextBuilder", "MemoryStore", "SkillsLoader", "AgentLoop"]


def __getattr__(name: str):
    """Lazy import AgentLoop to avoid import errors when litellm is not installed."""
    if name == "AgentLoop":
        from nanobot.agent.loop import AgentLoop
        return AgentLoop
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
