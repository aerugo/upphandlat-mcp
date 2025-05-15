import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import polars as pl
from mcp.server.fastmcp import FastMCP
from upphandlat_mcp.core.config import Settings
from upphandlat_mcp.core.config import settings as app_settings

logger = logging.getLogger(__name__)


class LifespanContext(TypedDict):
    df: pl.DataFrame
    settings: Settings


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[LifespanContext]:
    logger.info("MCP Server lifespan: Initializing application...")
    try:
        csv_file_path = app_settings.CSV_FILE_PATH
        # Path resolution logic (as before)

        logger.info(f"Attempting to load CSV data from: {csv_file_path.resolve()}")
        if not csv_file_path.exists():
            logger.error(f"CSV file not found at {csv_file_path.resolve()}")
            raise FileNotFoundError(f"CSV file not found: {csv_file_path.resolve()}")

        try:
            df = pl.read_csv(
                csv_file_path,
                separator=";",
                truncate_ragged_lines=True,  # Keep if the previous 'ragged lines' error might still be relevant for other reasons
                schema_overrides={
                    "Organisationsnummer för köpare": pl.Utf8
                },  # Explicitly set type to String
            )
        except (
            pl.ColumnNotFoundError
        ) as e:  # Specific error if the column name is wrong in schema_overrides
            logger.critical(
                f"Lifespan critical error: Column specified in schema_overrides not found: {e}. "
                "Ensure 'Organisationsnummer för köpare' matches the CSV header exactly."
            )
            raise
        except Exception as e_read:  # Catch other potential read_csv errors
            logger.critical(
                f"Lifespan critical error during pl.read_csv: {e_read}", exc_info=True
            )
            raise

        logger.info(
            f"Successfully loaded CSV. Shape: {df.shape}. Columns: {df.columns}"
        )

        context_data: LifespanContext = {
            "df": df,
            "settings": app_settings,
        }
        yield context_data
    except FileNotFoundError as e_fnf:  # Catch FileNotFoundError from earlier check
        logger.critical(f"Lifespan critical error (FileNotFound): {e_fnf}")
        raise
    except Exception as e_outer:  # Catch any other exceptions in the outer try block
        logger.critical(
            f"Lifespan critical error during data loading: {e_outer}", exc_info=True
        )
        raise
    finally:
        logger.info("MCP Server lifespan: Application shutdown.")
