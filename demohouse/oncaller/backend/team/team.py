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

from typing import AsyncIterable

from agent.agent import Agent, SwitchAgent
from volcenginesdkarkruntime.types.chat import ChatCompletionChunk

from arkitect.core.component.context.model import ContextInterruption


class Team:
    def __init__(
        self, agents: list[Agent] | None = None, current_agent: Agent | None = None
    ):
        self.agents = agents if agents else []
        self.current_agent = current_agent

    def add_agent(self, agent: Agent):
        self.agents.append(agent)
        if self.current_agent is None:
            self.current_agent = agent

    async def loop(self, task: str) -> AsyncIterable[ChatCompletionChunk]:
        self.current_agent.instruction = task
        message = None
        switch_agent = True
        while switch_agent:
            print("Current agent:", self.current_agent.name)
            switch_agent = False
            async for chunk in self.current_agent.astream_step(message=message):
                if isinstance(chunk, ChatCompletionChunk):
                    yield chunk
                elif isinstance(chunk, ContextInterruption):
                    if chunk.reason == "switch agent":
                        switch_agent = True
                        switch_instruction: SwitchAgent = chunk.details
                        agent_name, message = (
                            switch_instruction.agent_name,
                            switch_instruction.message,
                        )
                        self.current_agent = self.find_agent_by_name(agent_name)
                        break

    def find_agent_by_name(self, agent_name):
        """
        Find an agent by its name.
        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None


# Example usage
if __name__ == "__main__":
    team = Team()
    agent1 = Agent()
    agent2 = Agent()
    team.add_agent(agent1)
    team.add_agent(agent2)
    team.loop()
