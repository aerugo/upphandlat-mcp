import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import polars as pl
from mcp.server.fastmcp import FastMCP
from upphandlat_mcp.core.config import Settings
from upphandlat_mcp.core.config import settings as app_settings

logger = logging.getLogger(__name__)


class LifespanContext(TypedDict):
    """
    Defines the structure of the context dictionary made available
    to tools via ctx.request_context.lifespan_context.
    """

    df: pl.DataFrame
    settings: Settings


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[LifespanContext]:
    """
    Manages the application's lifespan.
    Loads the CSV data into a Polars DataFrame at startup.
    The yielded dictionary becomes accessible in tools via
    `ctx.request_context.lifespan_context`.

    Args:
        server: The FastMCP server instance.

    Yields:
        A dictionary containing the loaded DataFrame and settings.
    """
    logger.info("MCP Server lifespan: Initializing application...")
    try:
        # Construct the full path relative to the project root if CSV_FILE_PATH is relative
        # Assuming .env and pyproject.toml are at the project root.
        # Path(__file__).parents[3] would be project root if this file is src/pkg/lifespan/context.py
        # A simpler way if settings.CSV_FILE_PATH is expected to be relative to where app runs from:
        csv_file_path = app_settings.CSV_FILE_PATH
        if not csv_file_path.is_absolute():
            # Assuming the script runs from the project root or .env is read correctly
            # For robustness, one might want to anchor relative paths more explicitly
            # For now, rely on Pydantic-settings finding .env and Path resolving correctly
            pass

        logger.info(f"Attempting to load CSV data from: {csv_file_path.resolve()}")
        if not csv_file_path.exists():
            logger.error(f"CSV file not found at {csv_file_path.resolve()}")
            raise FileNotFoundError(f"CSV file not found: {csv_file_path.resolve()}")

        df = pl.read_csv(csv_file_path)
        logger.info(
            f"Successfully loaded CSV. Shape: {df.shape}. Columns: {df.columns}"
        )

        context_data: LifespanContext = {
            "df": df,
            "settings": app_settings,
        }
        yield context_data
    except FileNotFoundError as e:
        logger.critical(f"Lifespan critical error: {e}")
        # Yield an empty context or re-raise to prevent server from starting misconfigured
        # Depending on MCP behavior, raising might be better.
        # For now, let it raise to make the issue obvious.
        raise
    except Exception as e:
        logger.critical(
            f"Lifespan critical error during data loading: {e}", exc_info=True
        )
        raise
    finally:
        logger.info("MCP Server lifespan: Application shutdown.")
