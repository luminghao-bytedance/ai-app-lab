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
from typing import AsyncIterable
from pydantic import BaseModel, Field

from agent.agent import Agent, SwitchAgent
from jinja2 import Template
from arkitect.core.component.context.hooks import (
    HookInterruptException,
    PreToolCallHook,
)
from arkitect.core.component.context.context import Context
from arkitect.core.component.context.model import ContextInterruption, State
from prompt.worker import DEFAULT_WORKER_PROMPT
from volcenginesdkarkruntime.types.chat import ChatCompletionChunk


class SwitchAgent(BaseModel):
    agent_name: str
    message: str


class SwitchAgentInterrupHook(PreToolCallHook):
    async def pre_tool_call(self, name: str, arguments: str, state: State) -> State:
        if name == "switch_agent":
            params = json.loads(arguments)
            raise HookInterruptException(
                reason="switch agent",
                state=state,
                details=SwitchAgent(
                    agent_name=params.get("agent_name"),
                    message=params.get("message"),
                ),
            )
        return state


def switch_agent(agent_name: str, message: str) -> tuple[str, str]:
    """如果你需要查询某个组件的问题时，你可以用这个方法将任务分给某一个agent。比如APIG 的任务可以分给apig_worker

    Args:
        agent_name (str): 你想要将任务分配给哪个agent，你可以选择如下一些
            [apig_worker, tool_server_worker, proxy_worker, xllm_worker]
        message (str): 你希望agent做什么，你可以描述的更详细一些。
            举例：
                - 请求ID ABCDE 有报错500，请你查看下APIG的日志"。
    """
    return agent_name, message


class SupervisorState(State):
    round: int = 0
    max_round: int = 10


class SupervisorAgent(Agent):
    state: SupervisorState = Field(default_factory=SupervisorState)
    system_prompt: str = DEFAULT_WORKER_PROMPT

    async def astream_step(
        self,
        message: str | None = None,
        **kwargs,
    ) -> AsyncIterable[ChatCompletionChunk | ContextInterruption]:
        ctx = Context(
            model=self.model,
            tools=self.tools,
            state=self.state,
        )
        await ctx.init()
        ctx.add_pre_tool_call_hook(SwitchAgentInterrupHook())
        messages = self.generate_message(message=message)

        if self.state.round > self.state.max_round:
            resp_stream = self.force_summary()
        else:
            assert len(messages)
            resp_stream = await ctx.completions.create(messages=messages)
        self.state.round += 1

        async for chunk in resp_stream:
            if isinstance(chunk, ChatCompletionChunk):
                yield chunk
            elif isinstance(chunk, ContextInterruption):
                yield chunk
                return

    def generate_message(self, message: str | None) -> list:
        messages = []
        if self.state.round == 0:
            messages.append(
                {
                    "role": "system",
                    "content": Template(self.system_prompt).render(
                        complex_task=self.instruction,
                    ),
                }
            )
        if message is not None:
            messages.append(
                {
                    "role": "user",
                    "content": message,
                }
            )
        return messages


if __name__ == "__main__":

    async def main() -> None:

        agent = SupervisorAgent(
            llm_model="deepseek-r1-250120",
            instruction="requst ID XXXXX 有问题，你查一下APIG 日志，看看哪里有问题",
            tools=[switch_agent],
        )
        stream = await agent.astream_step()

        async for chunk in stream:
            if isinstance(chunk, ChatCompletionChunk):
                print(chunk.choices[0].delta.content, end="")
            elif isinstance(chunk, ContextInterruption):
                print(chunk.reason, chunk.details)

    import asyncio

    asyncio.run(main())
