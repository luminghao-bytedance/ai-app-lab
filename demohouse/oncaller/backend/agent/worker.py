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

from typing import Any, AsyncIterable, Optional

from agent.agent import Agent, SwitchAgent
from jinja2 import Template
from prompt.worker import DEFAULT_WORKER_PROMPT
from volcenginesdkarkruntime.types.chat import ChatCompletionChunk

from arkitect.core.component.context.model import ContextInterruption, State


class WorkerAgent(Agent):
    state: Optional[State] = None
    system_prompt: str = DEFAULT_WORKER_PROMPT

    async def astream_step(
        self, **kwargs
    ) -> ContextInterruption | AsyncIterable[ChatCompletionChunk | ContextInterruption]:
        # ctx = Context(
        #     model=self.model,
        #     tools=self.tools,
        #     state=self.state,
        # )
        # await ctx.init()
        yield ChatCompletionChunk(
            id="1",
            service_tier="default",
            choices=[
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "I have no idea",
                        "reasoning_content": "",
                    },
                    "stop_reason": "stop",
                }
            ],
            created=123,
            model=self.model,
            object="chat.completion.chunk",
        )

        yield ContextInterruption(
            life_cycle="tool_call",
            reason="switch agent",
            state=None,
            details=SwitchAgent(
                agent_name="supervisor",
                message="I have no idea. Please check with someone else.",
            ),
        )

    def generate_system_prompt(self) -> str:
        return Template(self.system_prompt).render(
            instruction=self.instruction,
            complex_task=self.planning.root_task,
            planning_details=self.planning.to_markdown_str(include_progress=False),
            task_id=str(self.planning_item.id),
            task_description=self.planning_item.description,
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
