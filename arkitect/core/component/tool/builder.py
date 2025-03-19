import os
import json
from arkitect.core.component.tool.mcp_client import MCPClient


def build_mcp_clients_from_config(config_file: str) -> list[MCPClient]:
    # check file exist
    if not os.path.exists(config_file):
        raise ValueError(f"Config file {config_file} does not exist")

    with open(config_file, "r") as f:
        config = json.loads(f.read())
    mcp_servers_config = config.get("mcpServers", {})
    mcp_clients = []
    for server_name in mcp_servers_config:
        command = mcp_servers_config[server_name].get("command", "")
        args = mcp_servers_config[server_name].get("args", [])
        client = MCPClient(command=command, arguments=args)
        mcp_clients.append(client)
    return mcp_clients
