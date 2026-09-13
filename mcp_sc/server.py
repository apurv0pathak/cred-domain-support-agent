"""
Private MCP server exposing Cred's loan application status lookup.

The underlying lookup implementation remains the frozen Phase 2 tool.
This module only exposes that existing function through MCP.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is importable when this file is executed directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastmcp import FastMCP

from agent.tools import check_loan_application_status


mcp = FastMCP(
    "Cred Domain Support Agent MCP Server",
)


@mcp.tool()
def lookup_loan_application_status(record_id: str) -> dict:
    """
    Look up the current status and escalation information for a Cred
    loan application using its record ID.

    Args:
        record_id: Cred loan application ID in the format LA-XXXX.

    Returns:
        A standardized dictionary containing the application status,
        record ID, and escalation information.
    """
    return check_loan_application_status(record_id)


if __name__ == "__main__":
    mcp.run(
        transport="http",
        host="127.0.0.1",
        port=8000,
    )