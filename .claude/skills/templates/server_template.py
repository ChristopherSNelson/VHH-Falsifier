"""
MCP Server: {SERVER_NAME}
Purpose: {PURPOSE}
Created by mcp-maker skill

Required env vars:
  {ENV_VARS}

Usage:
  claude mcp add {SERVER_NAME} --scope project -- \\
    uv run --with fastmcp python .claude/mcp-servers/{SERVER_NAME}.py
"""

import os
import httpx
from mcp.server.fastmcp import FastMCP

# --- Configuration -----------------------------------------------------------

# Server metadata — the name and description help Claude decide when to use it
mcp = FastMCP(
    "{SERVER_NAME}",
    description="{PURPOSE}",
)

# API configuration
BASE_URL = "{BASE_URL}"
API_KEY = os.environ.get("{API_KEY_ENV_VAR}", "")
TIMEOUT = 30  # seconds — prevents hanging the Claude Code session


# --- Tools -------------------------------------------------------------------
# Each tool must have:
#   - A clear docstring (Claude reads this to decide when to call the tool)
#   - Type hints on all params and return
#   - Error handling that returns a message, not a traceback
#   - httpx with timeout for all network calls


@mcp.tool()
def example_search(query: str, max_results: int = 5) -> dict:
    """Search {SERVICE_NAME} for records matching a query.

    Args:
        query: Search term (e.g., gene symbol, compound name).
        max_results: Maximum number of results to return (default 5).

    Returns:
        Dict with 'results' list and 'total' count.
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/search",
                params={"q": query, "limit": max_results},
                headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
            )
            response.raise_for_status()
            data = response.json()
            return {"results": data.get("results", []), "total": data.get("total", 0)}
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {TIMEOUT}s"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


@mcp.tool()
def example_get_by_id(record_id: str) -> dict:
    """Fetch a single record from {SERVICE_NAME} by its identifier.

    Args:
        record_id: The unique identifier for the record.

    Returns:
        Dict with the record data, or an error message.
    """
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            response = client.get(
                f"{BASE_URL}/records/{record_id}",
                headers={"Authorization": f"Bearer {API_KEY}"} if API_KEY else {},
            )
            response.raise_for_status()
            return response.json()
    except httpx.TimeoutException:
        return {"error": f"Request timed out after {TIMEOUT}s"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": str(e)}


# --- Entry point -------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
