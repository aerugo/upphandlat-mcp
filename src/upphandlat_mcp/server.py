import logging

from mcp.server.fastmcp import FastMCP
from upphandlat_mcp.lifespan.context import app_lifespan
from upphandlat_mcp.prompts.entry_prompt import csv_aggregator_entry_point
from upphandlat_mcp.tools.aggregation_tools import aggregate_data
from upphandlat_mcp.tools.info_tools import get_schema, list_columns

logger = logging.getLogger(__name__)

# Initialize the FastMCP server instance
# The server name should be descriptive for LLM discovery
mcp_server = FastMCP(
    name="app",
    description="A server for querying and aggregating data from a CSV file using Polars.",
    lifespan=app_lifespan,
)

# Register Tools
# The docstrings of these functions are crucial for LLM understanding.
mcp_server.tool()(list_columns)
mcp_server.tool()(get_schema)
mcp_server.tool()(aggregate_data)

# Register Prompts
# Prompts guide the LLM on how to interact with the server or achieve tasks.
mcp_server.prompt(name="csv_aggregator_guidance")(csv_aggregator_entry_point)


async def run_mcp_server() -> None:
    """
    Runs the MCP server, typically listening on stdio for MCP CLI integration.
    """
    logger.info(f"Starting MCP server '{mcp_server.name}' on stdio...")
    try:
        # "stdio" mode is common for MCP servers managed by `mcp dev` or Claude Desktop
        # Other modes like "sse" (Server-Sent Events) exist for web-based inspectors.
        mcp_server.run(transport="stdio")
        logger.info(f"MCP server '{mcp_server.name}' finished running.")
    except Exception as e:
        logger.critical(f"MCP server '{mcp_server.name}' crashed: {e}", exc_info=True)
        raise  # Re-raise to allow __init__.py to catch and exit
