"""子 Agent 执行单元池。

第一版先复用内存里的 SubAgent 实例；后续如果要替换成多进程池，
只需要保持 acquire/release/close 这一层接口稳定。
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

from ZBot.agent.base_agent import BaseAgent
from ZBot.agent.subagent.subagent import SubAgent


@dataclass(frozen=True, slots=True)
class SubAgentPolicy:
    """子 Agent 的系统级运行边界。"""

    # 主 Agent 一次最多能同时借出的子 Agent 执行单元数量。
    max_count: int = 5
    # 单个子任务的默认最大执行时间；运行时配置未传入时使用，默认 10 分钟。
    timeout_seconds: int = 600


# 全局唯一的内部策略对象，和 SubAgentPool 放在一起，避免额外拆一个策略文件。
SUBAGENT_POLICY = SubAgentPolicy()


class SubAgentLease:
    """一次从池里借出的子 Agent 执行单元。"""

    def __init__(self, agent_id: str, agent: SubAgent) -> None:
        """???????? Agent ??????"""
        self.agent_id: str = agent_id
        self.agent: SubAgent = agent


class SubAgentPool:
    """预创建并复用子 Agent 实例。

    当前实现是同进程内的实例池；设计上刻意像执行单元池，
    方便后面把 SubAgent 替换成独立进程代理而不改 create_sub_agent 工具。
    """

    def __init__(self, parent: BaseAgent, max_count: int = SUBAGENT_POLICY.max_count) -> None:
        """Agent 的子 Agent 执行单元池"""
        self._leases = [
            SubAgentLease(agent_id=f"subagent_{index}", agent=SubAgent.from_parent(parent))
            for index in range(1, max_count + 1)
        ]
        self._available: asyncio.Queue[SubAgentLease] = asyncio.Queue(maxsize=max_count)
        for lease in self._leases:
            # 普通 put()：队列满了会阻塞等待有空位
            # put_nowait()：不阻塞、立即放入；如果队列已满直接抛异常，不等待
            self._available.put_nowait(lease)
        self._closed = False

    @asynccontextmanager
    # Python contextlib 提供的异步上下文管理器装饰器
    # 把一个异步生成器函数，快速变成异步上下文管理器，可以用 async with 语法。
    # 函数必须是 async def，且是生成器（带 yield）
    # 执行流程：
    # async with 进入时：执行 yield **之前** 的代码（初始化 / 资源申请）
    # yield 切到业务逻辑
    # 代码块退出时：执行 yield **之后** 的代码（释放资源、收尾）
    # 专门给异步 IO用，替代手动写 __aenter__ / __aexit__ 魔法方法，极简封装异步资源（数据库连接、异步锁、网络会话等）
    async def acquire(self) -> AsyncIterator[SubAgentLease]:
        """借出一个子 Agent；任务结束后自动归还池中。"""
        if self._closed:
            raise RuntimeError("子 Agent 池已经关闭")

        lease = await self._available.get()
        try:
            yield lease
        finally:
            if not self._closed:
                self._available.put_nowait(lease)

    async def close(self) -> None:
        """关闭池。后续替换为多进程实现时，在这里终止所有子进程。"""
        self._closed = True
        while not self._available.empty():
            # 非阻塞从队列里取出一个元素，不等待.--->用get也可以。
            # 阻塞（get()）
            # pythonitem = await queue.get()  # 队列空了 → 程序在这里"卡住"等待
            # # 直到有元素放进来，才继续往下执行
            # 线程/协程被挂起，什么都干不了，就是傻等。
            # 非阻塞（get_nowait()）
            # pythontry:
            #     item = self._available.get_nowait()  # 队列空了 → 立刻抛出异常，不等
            # except asyncio.QueueEmpty:
            #     # 没拿到，但程序继续正常运行，可以做别的事
            #     pass
            # 有元素 → 取出来返回 ✅
            # 没元素 → 抛出 asyncio.QueueEmpty 异常 ❌（不等待）
            self._available.get_nowait()
