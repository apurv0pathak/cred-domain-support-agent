"""
Separate MCP client for the Cred loan-status MCP server.

This client runs independently from the LangGraph agent and connects
to the private local MCP server over HTTP.
"""

from __future__ import annotations

import asyncio
import json

from fastmcp import Client


MCP_SERVER_URL = "http://127.0.0.1:8000/mcp"

RECORD_IDS = [
    "LA-0001",
    "LA-0002",
]


async def main() -> None:
    async with Client(MCP_SERVER_URL) as client:
        tools = await client.list_tools()

        print("Connected to MCP server.")
        print("Available tools:")

        for tool in tools:
            print(f"- {tool.name}")

        print()

        for record_id in RECORD_IDS:
            result = await client.call_tool(
                "lookup_loan_application_status",
                {"record_id": record_id},
            )

            print(f"Record ID: {record_id}")
            print("MCP response:")
            print(json.dumps(result, indent=2, default=str))
            print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())