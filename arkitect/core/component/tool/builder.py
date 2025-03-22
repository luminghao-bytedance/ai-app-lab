# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import json
from arkitect.core.component.tool.mcp_client import MCPClient


def build_mcp_clients_from_config(config_file: str) -> dict[str, MCPClient]:
    # https://www.librechat.ai/docs/configuration/librechat_yaml/object_structure/mcp_servers#servername

    # check file exist
    if not os.path.exists(config_file):
        raise ValueError(f"Config file {config_file} does not exist")

    with open(config_file, "r") as f:
        config = json.loads(f.read())
    mcp_servers_config = config.get("mcpServers", {})
    mcp_clients = {}
    for server_name in mcp_servers_config:
        command = mcp_servers_config[server_name].get("command", None)
        args = mcp_servers_config[server_name].get("args", None)
        env = mcp_servers_config[server_name].get("env", None)
        server_url = mcp_servers_config[server_name].get("url", None)
        client = MCPClient(
            name=server_name,
            command=command,
            arguments=args,
            env=env,
            server_url=server_url,
        )
        mcp_clients[server_name] = client
    return mcp_clients
