import logging
from typing import Any

import polars as pl
from mcp.server.fastmcp import Context
from polars.exceptions import ColumnNotFoundError
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
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context
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


async def get_distinct_column_values(
    ctx: Context,
    column_name: str,
    sort_by_column: str | None = None,
    sort_descending: bool = False,
    limit: int | None = None,
) -> list[Any] | dict[str, str]:
    """
    Retrieves all distinct values from a specified column.

    Optionally, the distinct values can be sorted based on the values in
    another column (affecting the order of distinct values from `column_name`),
    and the number of results can be limited. If `sort_by_column` is not
    provided, the distinct values of `column_name` are sorted by `column_name` itself.

    Args:
        ctx: The MCP context, automatically injected.
        column_name: The name of the column to get distinct values from.
        sort_by_column: Optional. The name of the column to sort by.
                        This sort influences the order of the first occurrences
                        of unique values in `column_name`. If None, distinct values
                        from `column_name` are sorted by `column_name` itself.
        sort_descending: Optional. If True, sorts in descending order.
                         Defaults to False (ascending).
        limit: Optional. The maximum number of distinct values to return.
               Must be a non-negative integer if provided.

    Returns:
        A list of distinct values from the specified column, ordered as requested,
        or an error dictionary if an issue occurs.
    """
    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context
        df: pl.DataFrame = lifespan_ctx["df"]
    except KeyError:
        await ctx.error(
            "DataFrame 'df' not found in lifespan context. Was data loading successful?"
        )
        return {
            "error": "DataFrame not available. Server may be misconfigured or data failed to load."
        }

    try:
        if column_name not in df.columns:
            raise ColumnNotFoundError(
                f"Column '{column_name}' not found in DataFrame. Available columns: {df.columns}"
            )

        result_series: pl.Series

        if sort_by_column:
            if sort_by_column not in df.columns:
                raise ColumnNotFoundError(
                    f"Sort column '{sort_by_column}' not found in DataFrame. Available columns: {df.columns}"
                )
            result_series = df.sort(sort_by_column, descending=sort_descending)[
                column_name
            ].unique(maintain_order=True)
        else:
            result_series = (
                df[column_name]
                .unique(maintain_order=False)
                .sort(descending=sort_descending)
            )

        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError("Limit must be a non-negative integer.")
            result_series = result_series.head(limit)

        return result_series.to_list()

    except ColumnNotFoundError as e:
        error_msg = str(e)
        await ctx.error(f"Column not found: {error_msg}")
        return {"error": f"Configuration error: {error_msg}"}
    except ValueError as ve:
        error_msg = str(ve)
        await ctx.error(f"Invalid input value: {error_msg}")
        return {"error": f"Invalid input: {error_msg}"}
    except pl.PolarsError as pe:
        await ctx.error(
            f"Polars error during distinct value retrieval: {pe}", exc_info=True
        )
        return {"error": f"Data processing error with Polars: {str(pe)}"}
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred in get_distinct_column_values: {e}",
            exc_info=True,
        )
        return {"error": f"An unexpected error occurred: {str(e)}"}
