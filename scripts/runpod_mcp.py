"""Call the installed official RunPod MCP through stdio; never store credentials.

`inspect` performs only a protocol handshake and tools/list; it makes no account
requests. `call` requires RUNPOD_API_KEY from the environment and explicit tool
arguments from a JSON file. No tool is called on installation or inspection.
"""

import argparse
import asyncio
import json
import os
import shutil
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run(args):
    root = Path(__file__).resolve().parents[1]
    server = root / "integrations/runpod/node_modules/@runpod/mcp-server/dist/stdio.mjs"
    if not server.is_file():
        raise SystemExit("Install with: npm ci --prefix integrations/runpod --ignore-scripts")
    node = shutil.which("node")
    if not node:
        raise SystemExit("Node.js >=20 is required")
    environment = os.environ.copy()
    if args.command == "inspect":
        # A non-secret dummy satisfies the CLI startup check. No API calls are
        # made, and a real account credential is never used for offline inspection.
        environment["RUNPOD_API_KEY"] = "offline-protocol-inspection-not-a-real-key"
    elif not environment.get("RUNPOD_API_KEY"):
        raise SystemExit("Configure RUNPOD_API_KEY through Secrets before calling account tools.")
    parameters = StdioServerParameters(command=node, args=[str(server)], env=environment)
    async with (
        stdio_client(parameters) as (reader, writer),
        ClientSession(reader, writer) as session,
    ):
        initialized = await session.initialize()
        if args.command == "inspect":
            listing = await session.list_tools()
            report = {
                "server": initialized.serverInfo.model_dump(),
                "authenticated": False,
                "account_api_calls": 0,
                "tools": [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    }
                    for tool in listing.tools
                ],
            }
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps({**report, "tools": [t["name"] for t in report["tools"]]}, indent=2))
        else:
            arguments = json.loads(Path(args.arguments).read_text()) if args.arguments else {}
            result = await session.call_tool(args.tool, arguments)
            # Official MCP tools redact resource secrets. Never pass API keys
            # in arguments; authentication remains in the subprocess env.
            print(result.model_dump_json(indent=2))
            if result.isError:
                raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--output")
    call = sub.add_parser("call")
    call.add_argument("--tool", required=True)
    call.add_argument("--arguments", help="JSON argument file; no credentials")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
