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

import os
from typing import AsyncIterable

from agent.supervisor import SupervisorAgent, switch_agent
from agent.worker import WorkerAgent

from telemetry.byted.setup import setup_tracing
from team.team import Team
from arkitect.core.component.tool.builder import build_mcp_clients_from_config
from arkitect.launcher.local.serve import launch_serve
from arkitect.telemetry.trace import task
from arkitect.types.llm.model import ArkChatRequest, ArkChatResponse


MODELS = {
    "default": "doubao-1-5-pro-32k-250115",
    "reasoning": "deepseek-r1-250120",
    "vision": "doubao-1-5-vision-pro-32k-250115",
}
# your endpoint api key
api_key = os.getenv("endpoint_api_key")


clients = build_mcp_clients_from_config(
    "/Users/bytedance/Documents/deepresearch/ai-app-lab/demohouse/oncaller/backend/mcp_config.json"
)


@task(distributed=False)
async def main(request: ArkChatRequest) -> AsyncIterable[ArkChatResponse]:
    supervisor = SupervisorAgent(
        model=MODELS["default"], tools=[switch_agent], name="supervisor"
    )
    apig_worker = WorkerAgent(
        model=MODELS["reasoning"], tools=[clients["apig_logs"]], name="apig_worker"
    )
    tool_server_worker = WorkerAgent(
        model=MODELS["reasoning"],
        tools=[clients["tool_server_logs"]],
        name="tool_server_worker",
    )
    proxy_worker = WorkerAgent(
        model=MODELS["reasoning"], tools=[clients["proxy_logs"]], name="proxy_worker"
    )
    xllm_worker = WorkerAgent(
        model=MODELS["reasoning"], tools=[clients["xllm_logs"]], name="xllm_worker"
    )

    team = Team(
        agents=[
            supervisor,
            apig_worker,
            tool_server_worker,
            proxy_worker,
            xllm_worker,
        ],
        current_agent=supervisor,
    )
    async for resp in team.loop(request.messages[0].content):
        yield resp


if __name__ == "__main__":
    port = os.getenv("_BYTEFAAS_RUNTIME_PORT")
    setup_tracing()
    launch_serve(
        package_path="main",
        clients={},
        port=int(port) if port else 10888,
        host=None,
        health_check_path="/v1/ping",
        endpoint_path="/api/v3/bots/chat/completions",
    )
