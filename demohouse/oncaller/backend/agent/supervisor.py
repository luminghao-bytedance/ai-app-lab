# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the 【火山方舟】原型应用软件自用许可协议
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.volcengine.com/docs/82379/1433703
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import AsyncIterable, Optional

from agent.agent import Agent
from jinja2 import Template
from arkitect.core.component.context.hooks import (
    HookInterruptException,
    PreToolCallHook,
)
from models.messages import OutputTextChunk, ReasoningChunk
from models.planning import Planning, PlanningItem
from prompt.worker import DEFAULT_WORKER_PROMPT
from volcenginesdkarkruntime.types.chat import ChatCompletionChunk

from arkitect.core.component.context.context import Context
from arkitect.core.component.context.model import ContextInterruption, State


class SwitchAgentInterrupHook(PreToolCallHook):
    async def pre_tool_call(self, name: str, arguments: str, state: State) -> State:
        if name == "switch_agent":
            raise HookInterruptException(
                reason="switch agent",
                state=state,
            )
        return state


def switch_agent(agent_name: str, message: str) -> tuple[str, str]:
    """You need to pass the task to the agent to do their job.

    Args:
        agent_name (str): the name of the next agent. You can choose from:
            [apig_worker, tool_server_worker, proxy_worker, xllm_worker]
        message (str): You need to pass down the context of the task and what you want them to do and what to report back to you.
            For example:
                - There is a problem with request ABCDE. Please find out if there is any noticable error in your service.
    """
    return agent_name, message


class SupervisorState(State):
    round: int = 0
    max_round: int = 10


class SupervisorAgent(Agent):
    state: Optional[SupervisorState] = None
    system_prompt: str = DEFAULT_WORKER_PROMPT

    async def astream_step(
        self,
        message: str | None = None,
        **kwargs,
    ) -> AsyncIterable[ChatCompletionChunk]:
        ctx = Context(
            model=self.model,
            tools=self.tools,
            state=self.state,
        )
        await ctx.init()
        messages = []
        if self.state.round == 0:
            messages.append(
                {
                    "role": "system",
                    "content": self.generate_system_prompt(),
                }
            )
        if message is not None:
            messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )

        if self.state.round > self.state.max_round:
            resp_stream = self.force_summary()
        else:
            assert message is not None
            resp_stream = ctx.completions.create(messages=messages)
        self.state.round += 1

        async for chunk in resp_stream:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk
            elif isinstance(chunk, ContextInterruption):
                return chunk

    def generate_system_prompt(self) -> str:
        return Template(self.system_prompt).render(
            instruction=self.instruction,
            complex_task=self.planning.root_task,
        )


if __name__ == "__main__":

    def add(a: int, b: int) -> int:
        """Add two numbers

        Args:
            a (int): first number
            b (int): second number

        Returns:
            int: sum of a and b
        """
        return a + b

    async def main() -> None:

        planning_item = PlanningItem(
            id="1",
            description="计算 1 + 19",
        )

        agent = WorkerAgent(
            llm_model="deepseek-r1-250120",
            instruction="数据计算专家，会做两位数的加法",
            tools=[add],
            planning=Planning(root_task="计算给定的题目", items={"1": planning_item}),
            planning_item=planning_item,
        )

        async for chunk in agent.astream_step():
            if isinstance(chunk, OutputTextChunk):
                print(chunk.delta, end="")
            if isinstance(chunk, ReasoningChunk):
                print(chunk.delta, end="")

        print(agent.get_result())

    import asyncio

    asyncio.run(main())
