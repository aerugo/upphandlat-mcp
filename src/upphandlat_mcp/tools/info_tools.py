import logging
from typing import Any

import polars as pl
from mcp.server.fastmcp import Context
from upphandlat_mcp.lifespan.context import LifespanContext
from upphandlat_mcp.utils.dataframe_ops import (
    get_column_names_from_df,
    get_schema_from_df,
)

logger = logging.getLogger(__name__)


async def list_columns(ctx: Context) -> list[str] | dict[str, str]:
    """
    Retrieves the list of column names from the loaded CSV data.

    This tool is useful for understanding the available fields in the dataset
    before performing aggregations or other operations.

    Args:
        ctx: The MCP context, automatically injected.
             Provides access to lifespan resources like the loaded DataFrame.

    Returns:
        A list of strings, where each string is a column name.
        Returns an error dictionary if the DataFrame is not available.
    """
    try:
        lifespan_ctx = ctx.request_context.lifespan_context
        df: pl.DataFrame = lifespan_ctx["df"]
        return get_column_names_from_df(df)
    except KeyError:
        await ctx.error(
            "DataFrame 'df' not found in lifespan context. Was data loading successful?"
        )
        return {
            "error": "DataFrame not available. Server may be misconfigured or data failed to load."
        }
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred in list_columns: {e}", exc_info=True
        )
        return {"error": f"An unexpected error occurred: {str(e)}"}


async def get_schema(ctx: Context) -> dict[str, str] | dict[str, Any]:
    """
    Retrieves the schema of the loaded CSV data, mapping column names
    to their Polars data types (represented as strings).

    This helps in understanding the kind of data each column holds (e.g.,
    integer, float, string) which is important for choosing appropriate
    aggregation functions or calculations.

    Args:
        ctx: The MCP context, automatically injected.

    Returns:
        A dictionary where keys are column names (str) and values are
        their data types (str, e.g., "Int64", "Utf8", "Float64").
        Returns an error dictionary if the DataFrame is not available.
    """
    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context  # type: ignore
        df: pl.DataFrame = lifespan_ctx["df"]
        return get_schema_from_df(df)
    except KeyError:
        await ctx.error(
            "DataFrame 'df' not found in lifespan context. Was data loading successful?"
        )
        return {
            "error": "DataFrame not available. Server may be misconfigured or data failed to load."
        }
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred in get_schema: {e}", exc_info=True
        )
        return {"error": f"An unexpected error occurred: {str(e)}"}
