Okay, I will complete the new MCP server project based on the provided files and conventions. I'll transform the existing FastAPI application logic into a set of MCP tools, following the structure and style of the Kolada MCP server example and the `CONVENTIONS.md` document.

Here's the complete implementation:

--- File: `pyproject.toml` ---
```toml
[project]
name = "upphandlat-mcp"
version = "0.1.0"
description = "MCP server for CSV aggregation and analysis using Polars"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "mcp[cli]>=1.6.0",  # MCP SDK with CLI for development
    "polars>=0.19.0",
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "python-dotenv>=1.0.0",
]
authors = [
    { name = "Your Name", email = "your.email@example.com" }
]
license = { text = "Apache-2.0" } # Or your preferred license

[tool.setuptools.packages.find]
where = ["src"]

[project.urls]
"Homepage" = "https://github.com/yourusername/upphandlat-mcp" # Example URL
"Bug Tracker" = "https://github.com/yourusername/upphandlat-mcp/issues" # Example URL

[project.scripts]
upphandlat-mcp = "upphandlat_mcp:main"

[tool.ruff]
line-length = 88

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "PL", "PYI"] # Common useful rules

[tool.pyright]
# Pyright configurations (optional, but good for type checking)
include = ["src/upphandlat_mcp"]
pythonVersion = "3.13"
reportMissingImports = "warning"
reportTypedDictNotRequiredAccess = "none"

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"
```

--- File: `.env` ---
```env
CSV_FILE_PATH=data/your_data.csv
```

--- File: `README.md` ---
```markdown
# Upphandlat MCP Server

[![https://modelcontextprotocol.io](https://badge.mcpx.dev?type=server 'MCP Server')](https://modelcontextprotocol.io)

**Upphandlat MCP Server** provides tools to interact with and aggregate data from a CSV file using the Model Context Protocol (MCP). It leverages Polars for high-performance data manipulation.

## Overview

This MCP server allows LLMs to:
- Discover the schema and column names of a configured CSV file.
- Perform complex aggregations, grouping, and calculations on the CSV data.

## Features
- Load CSV data on server startup.
- Expose CSV schema and column information.
- Provide powerful aggregation capabilities, including:
    - Grouping by multiple columns.
    - Applying multiple aggregation functions (sum, mean, count, min, max).
    - Defining and applying calculated fields based on arithmetic operations or percentages.

## Configuration
The path to the CSV file is configured via the `CSV_FILE_PATH` environment variable, typically set in a `.env` file.
Example `.env`:
```
CSV_FILE_PATH=data/your_data.csv
```
Ensure the `data/your_data.csv` file exists and is accessible.

## Available Tools
1.  **`list_columns()`**: Returns a list of available column names from the CSV.
2.  **`get_schema()`**: Returns a dictionary mapping column names to their Polars data types.
3.  **`aggregate_data(request: AggregationRequest)`**: Performs aggregations and calculates derived fields on the CSV data based on the provided request.

## Installation

It's recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .
```
This installs the package in editable mode.

## Running Locally for Development

Ensure your `.env` file is configured and the data file specified by `CSV_FILE_PATH` exists.

Start the server locally:
```bash
upphandlat-mcp
```
Alternatively, for development with `mcp dev`:
```bash
uv run mcp dev src/upphandlat_mcp/server.py
```
Or, if `uv` is not managing the `mcp` tool itself:
```bash
mcp dev src/upphandlat_mcp/server.py
```

You can then use the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) (usually at `http://localhost:5173` if you run it separately) to test and debug your MCP server.

## Project Structure
```
.
├── pyproject.toml         # Project metadata and dependencies
├── .env                   # Environment variables (e.g., CSV_FILE_PATH)
├── README.md              # This file
├── data/
│   └── your_data.csv      # Example/your data file
└── src/
    └── upphandlat_mcp/  # Main package
        ├── __init__.py        # Package initializer and CLI entry point
        ├── server.py          # MCP server definition and tool registration
        ├── core/
        │   ├── __init__.py
        │   └── config.py      # Pydantic settings for configuration
        ├── lifespan/
        │   ├── __init__.py
        │   └── context.py     # Server lifespan management (e.g., data loading)
        ├── models/
        │   ├── __init__.py
        │   └── mcp_models.py  # Pydantic models for tool arguments/data structures
        ├── tools/
        │   ├── __init__.py
        │   ├── info_tools.py       # Tools for schema/column info
        │   └── aggregation_tools.py # Tool for data aggregation
        └── prompts/
            ├── __init__.py
            └── entry_prompt.py   # Example entry prompt for LLMs
        └── utils/
            ├── __init__.py
            └── dataframe_ops.py # Utility functions for DataFrame operations
```

## Contributing
Contributions are welcome! Please open an issue or submit a pull request.
```

--- File: `data/your_data.csv` ---
```csv
category,region,sales,quantity,cost
A,North,100,10,80
B,North,150,15,120
A,South,200,20,150
B,South,50,5,40
A,North,120,12,90
C,North,300,25,250
A,West,80,8,60
B,West,120,10,100
C,South,220,18,180
A,South,180,17,140
```

--- File: `src/upphandlat_mcp/__init__.py` ---
```python
import asyncio
import logging
import sys

from . import server

# Configure basic logging for the server
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for running the Upphandlat MCP server.
    This is typically called by the script defined in pyproject.toml.
    """
    try:
        logger.info("Starting Upphandlat MCP Server...")
        # server.py contains the mcp instance and its run method
        # Kolada example calls a main() in server.py, we can simplify
        # by directly accessing the mcp instance if it's globally defined in server.py
        # or by having server.run_server()
        # For now, assuming server.py has a run_mcp_server function or similar
        # Or, if server.mcp is directly importable and has run().
        # Let's define a main function in server.py similar to Kolada example for consistency.
        asyncio.run(server.run_mcp_server())
    except KeyboardInterrupt:
        logger.info("Server shut down by user (KeyboardInterrupt).")
    except Exception as e:
        logger.critical(f"Server failed to run: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Upphandlat MCP Server has shut down.")


__all__ = ["main", "server"]
```

--- File: `src/upphandlat_mcp/core/__init__.py` ---
```python
# Core package
```

--- File: `src/upphandlat_mcp/core/config.py` ---
```python
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """
    CSV_FILE_PATH: Path = Path("data/your_data.csv") # Default if .env not found or var not set

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )


settings = Settings()
```

--- File: `src/upphandlat_mcp/lifespan/__init__.py` ---
```python
# Lifespan package
```

--- File: `src/upphandlat_mcp/lifespan/context.py` ---
```python
import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, TypedDict

import polars as pl
from mcp.server.fastmcp import FastMCP

from upphandlat_mcp.core.config import Settings, settings as app_settings

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
        logger.info(f"Successfully loaded CSV. Shape: {df.shape}. Columns: {df.columns}")

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
        logger.critical(f"Lifespan critical error during data loading: {e}", exc_info=True)
        raise
    finally:
        logger.info("MCP Server lifespan: Application shutdown.")
```

--- File: `src/upphandlat_mcp/models/__init__.py` ---
```python
# Models package
```

--- File: `src/upphandlat_mcp/models/mcp_models.py` ---
```python
from enum import Enum
from typing import Literal, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class AggFunc(str, Enum):
    """Supported aggregation functions."""
    SUM = "sum"
    MEAN = "mean"
    COUNT = "count"
    MIN = "min"
    MAX = "max"


class Aggregation(BaseModel):
    """
    Defines an aggregation operation on a single column.
    """
    column: str = Field(..., description="The column to aggregate.")
    functions: list[AggFunc] = Field(..., min_length=1, description="List of aggregation functions to apply.")
    rename: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional mapping to rename output columns. "
            "Keys should be function names (e.g., 'sum'), "
            "values are the new names (e.g., 'total_sales'). "
            "Default is '{column}_{function}'."
        )
    )

    @model_validator(mode='after')
    def check_rename_keys(self) -> 'Aggregation':
        defined_functions = {f.value for f in self.functions}
        for key_to_rename in self.rename:
            if key_to_rename not in defined_functions:
                raise ValueError(
                    f"Rename key '{key_to_rename}' is not in the list of "
                    f"applied functions: {list(defined_functions)} for column '{self.column}'. "
                    "Rename keys must match one of the function names (sum, mean, etc.)."
                )
        return self


class BaseCalculatedField(BaseModel):
    """
    Base settings for a calculated field.
    """
    output_column_name: str = Field(
        ...,
        description="The name for the new calculated column.",
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    )

    @field_validator('output_column_name')
    @classmethod
    def no_reserved_output_name(cls, name: str) -> str:
        """Ensure output column name is not a reserved keyword."""
        # These are keys in the union type, not actual column names to avoid.
        # It's more about avoiding clashes with existing or group_by columns.
        # This validation is better done when applying fields, checking against existing columns.
        # The regex pattern is the primary Pydantic validation here.
        return name


class ArithmeticOperationType(str, Enum):
    """Type of arithmetic operation."""
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"


class TwoColumnArithmeticConfig(BaseCalculatedField):
    """Configuration for arithmetic operation between two existing columns."""
    calculation_type: Literal['two_column_arithmetic'] = Field("two_column_arithmetic", description="Type of calculation.")
    column_a: str = Field(..., description="The first column operand (e.g., 'sales').")
    column_b: str = Field(..., description="The second column operand (e.g., 'cost').")
    operation: ArithmeticOperationType = Field(..., description="The arithmetic operation to perform (e.g., 'subtract' for A - B).")
    on_division_by_zero: float | Literal['null', 'propagate_error'] = Field(
        'propagate_error',
        description="Behavior for division by zero: a specific float, 'null', or 'propagate_error' (Polars default, may result in inf/NaN)."
    )


class ConstantArithmeticConfig(BaseCalculatedField):
    """Configuration for arithmetic operation between a column and a constant."""
    calculation_type: Literal['constant_arithmetic'] = Field("constant_arithmetic", description="Type of calculation.")
    input_column: str = Field(..., description="The column to operate on.")
    constant_value: float = Field(..., description="The constant value for the operation.")
    operation: ArithmeticOperationType = Field(..., description="The arithmetic operation.")
    column_is_first_operand: bool = Field(
        True,
        description="If true, operation is 'input_column op constant'; else 'constant op input_column'."
    )
    on_division_by_zero: float | Literal['null', 'propagate_error'] = Field(
        'propagate_error',
        description="Behavior for division by zero (if applicable)."
    )


class PercentageOfColumnConfig(BaseCalculatedField):
    """Configuration for calculating one column as a percentage of another."""
    calculation_type: Literal['percentage_of_column'] = Field("percentage_of_column", description="Type of calculation.")
    value_column: str = Field(..., description="The column representing the part/value (numerator).")
    total_reference_column: str = Field(..., description="The column representing the total/base (denominator).")
    scale_factor: float = Field(
        100.0,
        description="Factor to multiply the ratio by (e.g., 100.0 for percentage)."
    )
    on_division_by_zero: float | Literal['null', 'propagate_error'] = Field(
        'propagate_error',
        description="Behavior for division by zero in 'value_column / total_reference_column'."
    )


CalculatedFieldType = TwoColumnArithmeticConfig | ConstantArithmeticConfig | PercentageOfColumnConfig


class AggregationRequest(BaseModel):
    """
    Defines the request payload for the aggregation tool.
    It specifies how to group data, what aggregations to perform,
    and any additional calculated fields to compute on the aggregated results.
    """
    group_by_columns: list[str] = Field(
        ...,
        min_length=1,
        description="List of column names to group the data by."
    )
    aggregations: list[Aggregation] = Field(
        ...,
        min_length=1,
        description="List of aggregation operations to perform on specified columns."
    )
    calculated_fields: list[CalculatedFieldType] | None = Field(
        None,
        description="Optional list of calculated fields to derive after aggregation. These are applied in order."
    )

    @model_validator(mode='after')
    def check_column_name_conflicts(self) -> 'AggregationRequest':
        """
        Validate that output column names from aggregations and calculated fields
        do not conflict with each other or with group_by_columns.
        """
        all_output_names: set[str] = set(self.group_by_columns)

        for agg in self.aggregations:
            for func in agg.functions:
                alias = agg.rename.get(func.value, f"{agg.column}_{func.value}")
                if alias in all_output_names:
                    raise ValueError(f"Duplicate output column name '{alias}' from aggregation conflicts with group_by or another aggregation.")
                all_output_names.add(alias)

        if self.calculated_fields:
            for calc_field_config in self.calculated_fields:
                # The config itself is a discriminated union, so access common field directly
                output_name = calc_field_config.output_column_name
                if output_name in all_output_names:
                    raise ValueError(f"Calculated field output name '{output_name}' conflicts with group_by or aggregation columns.")
                all_output_names.add(output_name)
        return self
```

--- File: `src/upphandlat_mcp/prompts/__init__.py` ---
```python
# Prompts package
```

--- File: `src/upphandlat_mcp/prompts/entry_prompt.py` ---
```python
from mcp.server.fastmcp.prompts import base # For potential use of structured messages

def csv_aggregator_entry_point() -> str:
    """
    Provides a general guide for an LLM on how to interact with the
    Upphandlat MCP server and its tools.
    """
    return (
        "## Upphandlat Aggregator MCP Server Interaction Guide\n\n"
        "**Objective:** You are interacting with a server that can analyze and aggregate data from a pre-loaded CSV file. Your goal is to use the available tools to answer user queries about this data.\n\n"
        "**Available Tools & Common Use Cases:**\n\n"
        "1.  **`list_columns()`:**\n"
        "    *   **Use When:** You need to know what data fields (columns) are available in the CSV.\n"
        "    *   **Example:** \"What columns are in the dataset?\"\n\n"
        "2.  **`get_schema()`:**\n"
        "    *   **Use When:** You need to know the data types of the columns (e.g., string, integer, float) to understand how they can be used in aggregations or calculations.\n"
        "    *   **Example:** \"What is the schema of the dataset?\" or \"What type is the 'sales' column?\"\n\n"
        "3.  **`aggregate_data(request: AggregationRequest)`:**\n"
        "    *   **Use When:** The user wants to summarize, group, or calculate new metrics from the data. This is the most powerful tool.\n"
        "    *   **`AggregationRequest` Structure:** This tool takes a complex JSON object. You'll need to construct it carefully based on the user's request. Key parts:\n"
        "        *   `group_by_columns` (list[str]): Columns to group by (e.g., ['category', 'region']).\n"
        "        *   `aggregations` (list[Aggregation]): Operations on columns within each group.\n"
        "            *   `Aggregation` object: `{ \"column\": \"sales\", \"functions\": [\"sum\", \"mean\"], \"rename\": {\"sum\": \"total_sales\"} }`\n"
        "        *   `calculated_fields` (list[CalculatedFieldType], optional): New columns derived from aggregated results (e.g., profit = sales - cost).\n"
        "            *   `CalculatedFieldType` can be `two_column_arithmetic`, `constant_arithmetic`, or `percentage_of_column`.\n"
        "    *   **Examples:**\n"
        "        *   \"What is the total sales per region?\" -> Group by 'region', sum 'sales'.\n"
        "        *   \"Calculate average quantity and total sales for each product category and region.\" -> Group by ['category', 'region'], mean 'quantity', sum 'sales'.\n"
        "        *   \"For each category, find the profit margin if profit is sales minus cost, and express it as a percentage of sales.\" -> Group by 'category', sum 'sales', sum 'cost'. Then, add calculated fields: one for 'profit' (sales_sum - cost_sum) and another for 'profit_margin' (profit / sales_sum * 100).\n\n"
        "**General Strategy & Workflow:**\n\n"
        "1.  **Understand the Data:** Use `list_columns()` and `get_schema()` first if you're unsure about the available data fields and their types.\n"
        "2.  **Plan Aggregation:** If the user asks for summarized data, identify:\n"
        "    *   What groups are needed (`group_by_columns`).\n"
        "    *   What calculations to perform on which columns (`aggregations`).\n"
        "    *   Any derived metrics needed after grouping (`calculated_fields`).\n"
        "3.  **Construct `AggregationRequest`:** Carefully build the JSON request for `aggregate_data`.\n"
        "4.  **Execute and Present:** Call the `aggregate_data` tool and present the results clearly to the user.\n"
        "5.  **Handle Errors:** If a tool returns an error, explain it to the user and try to adjust the request if possible.\n\n"
        "**Example of a full `AggregationRequest` for `aggregate_data`:**\n"
        "```json\n"
        "{\n"
        "  \"group_by_columns\": [\"category\", \"region\"],\n"
        "  \"aggregations\": [\n"
        "    {\n"
        "      \"column\": \"sales\",\n"
        "      \"functions\": [\"sum\", \"mean\"],\n"
        "      \"rename\": {\"sum\": \"total_sales\", \"mean\": \"average_sales\"}\n"
        "    },\n"
        "    {\n"
        "      \"column\": \"quantity\",\n"
        "      \"functions\": [\"sum\"],\n"
        "      \"rename\": {\"sum\": \"total_quantity\"}\n"
        "    }\n"
        "  ],\n"
        "  \"calculated_fields\": [\n"
        "    {\n"
        "      \"calculation_type\": \"two_column_arithmetic\",\n"
        "      \"output_column_name\": \"revenue_per_unit\",\n"
        "      \"column_a\": \"total_sales\",\n"
        "      \"column_b\": \"total_quantity\",\n"
        "      \"operation\": \"divide\",\n"
        "      \"on_division_by_zero\": \"null\"\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "Now, analyze the user's request and determine the best tool(s) and parameters to use."
    )
```

--- File: `src/upphandlat_mcp/tools/__init__.py` ---
```python
# Tools package
```

--- File: `src/upphandlat_mcp/tools/info_tools.py` ---
```python
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
        await ctx.error("DataFrame 'df' not found in lifespan context. Was data loading successful?")
        return {"error": "DataFrame not available. Server may be misconfigured or data failed to load."}
    except Exception as e:
        await ctx.error(f"An unexpected error occurred in list_columns: {e}", exc_info=True)
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
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context # type: ignore
        df: pl.DataFrame = lifespan_ctx["df"]
        return get_schema_from_df(df)
    except KeyError:
        await ctx.error("DataFrame 'df' not found in lifespan context. Was data loading successful?")
        return {"error": "DataFrame not available. Server may be misconfigured or data failed to load."}
    except Exception as e:
        await ctx.error(f"An unexpected error occurred in get_schema: {e}", exc_info=True)
        return {"error": f"An unexpected error occurred: {str(e)}"}
```

--- File: `src/upphandlat_mcp/tools/aggregation_tools.py` ---
```python
import logging
from typing import Any

import polars as pl
from mcp.server.fastmcp import Context

from upphandlat_mcp.lifespan.context import LifespanContext
from upphandlat_mcp.models.mcp_models import (
    AggregationRequest,
    Aggregation,
    CalculatedFieldType,
    TwoColumnArithmeticConfig,
    ConstantArithmeticConfig,
    PercentageOfColumnConfig,
    ArithmeticOperationType,
)

logger = logging.getLogger(__name__)


def _build_polars_aggregation_expressions(
    aggregations: list[Aggregation],
    group_by_column_names: set[str],
    existing_df_columns: set[str],
    ctx: Context # For logging
) -> tuple[list[pl.Expr], set[str]]:
    """
    Creates a list of Polars aggregation expressions from the request.
    Also tracks output column names to prevent duplicates and checks existence.

    Returns:
        A tuple containing the list of Polars expressions and a set of
        all output column names created by these aggregations (including renames).
    """
    polars_expressions: list[pl.Expr] = []
    # Starts with group_by columns, as they will be in the output of group_by().agg()
    all_output_column_names: set[str] = set(group_by_column_names)

    for agg_config in aggregations:
        if agg_config.column not in existing_df_columns:
            # This error should ideally be caught by Pydantic if we had full schema access there,
            # or handled before this function.
            raise ValueError(f"Aggregation column '{agg_config.column}' not found in DataFrame.")

        for func_enum in agg_config.functions:
            function_name_str = func_enum.value  # e.g., "sum"

            # Determine the output alias for this specific aggregation
            # Default: {column_name}_{function_name}, e.g., "sales_sum"
            # Custom if provided in rename: e.g., "total_sales"
            output_alias = agg_config.rename.get(function_name_str, f"{agg_config.column}_{function_name_str}")

            if output_alias in all_output_column_names:
                # This should be caught by Pydantic model validation in AggregationRequest
                # but good to have a safeguard.
                raise ValueError(f"Duplicate output column name: '{output_alias}'.")
            all_output_column_names.add(output_alias)

            # Create Polars expression: pl.col("sales").sum().alias("total_sales")
            try:
                # Ensure the column exists and is of a type that supports the aggregation
                # Polars will raise an error if the operation is invalid for the dtype.
                column_expression = pl.col(agg_config.column)
                aggregation_function = getattr(column_expression, function_name_str) # e.g., pl.col("sales").sum
                final_expression = aggregation_function().alias(output_alias)
                polars_expressions.append(final_expression)
            except AttributeError:
                # Should not happen if AggFunc enum is correct
                raise ValueError(f"Invalid aggregation function '{function_name_str}' for Polars.")
            except Exception as e: # Catch Polars-specific errors during expression building
                ctx.warning(f"Could not build expression for {agg_config.column}.{function_name_str}(): {e}")
                # Decide: skip this expression or raise? Raising might be better.
                raise ValueError(f"Error building expression for {agg_config.column}.{function_name_str}(): {e}")


    return polars_expressions, all_output_column_names


def _apply_calculated_fields(
    df: pl.DataFrame,
    calculated_field_configs: list[CalculatedFieldType],
    # all_current_column_names is used to check input cols and prevent output conflicts
    all_current_column_names: set[str],
    ctx: Context # For logging
) -> pl.DataFrame:
    """
    Applies a list of calculated field configurations to the DataFrame.
    Calculated fields are applied sequentially.
    """
    if not calculated_field_configs:
        return df

    # Make a mutable copy for tracking columns available at each step
    available_columns_during_calculation = set(all_current_column_names)
    result_df = df

    for field_config in calculated_field_configs:
        output_col_name = field_config.output_column_name

        if output_col_name in available_columns_during_calculation:
            # This should be caught by Pydantic validation in AggregationRequest
            raise ValueError(f"Calculated field output name '{output_col_name}' conflicts with an existing or previously calculated column.")

        # Helper to check if input columns for the calculation exist
        def _check_input_columns_exist(*cols_to_check: str) -> None:
            for col_name in cols_to_check:
                if col_name not in available_columns_during_calculation:
                    raise ValueError(
                        f"Input column '{col_name}' for calculated field '{output_col_name}' not found. "
                        f"Available columns: {sorted(list(available_columns_during_calculation))}"
                    )

        polars_expr: pl.Expr

        if field_config.calculation_type == 'two_column_arithmetic':
            _check_input_columns_exist(field_config.column_a, field_config.column_b)
            col_a_expr = pl.col(field_config.column_a)
            col_b_expr = pl.col(field_config.column_b)
            op_map = {
                ArithmeticOperationType.ADD: lambda a, b: a + b,
                ArithmeticOperationType.SUBTRACT: lambda a, b: a - b,
                ArithmeticOperationType.MULTIPLY: lambda a, b: a * b,
                ArithmeticOperationType.DIVIDE: lambda a, b: a / b,
            }
            polars_expr = op_map[field_config.operation](col_a_expr, col_b_expr)

            if field_config.operation == ArithmeticOperationType.DIVIDE and field_config.on_division_by_zero != 'propagate_error':
                otherwise_val = None if field_config.on_division_by_zero == 'null' else pl.lit(field_config.on_division_by_zero)
                polars_expr = pl.when(col_b_expr != 0).then(polars_expr).otherwise(otherwise_val)


        elif field_config.calculation_type == 'constant_arithmetic':
            _check_input_columns_exist(field_config.input_column)
            input_col_expr = pl.col(field_config.input_column)
            constant_expr = pl.lit(field_config.constant_value)

            op_map_col_first = {
                ArithmeticOperationType.ADD: lambda col, const: col + const,
                ArithmeticOperationType.SUBTRACT: lambda col, const: col - const,
                ArithmeticOperationType.MULTIPLY: lambda col, const: col * const,
                ArithmeticOperationType.DIVIDE: lambda col, const: col / const,
            }
            op_map_const_first = {
                ArithmeticOperationType.ADD: lambda const, col: const + col,
                ArithmeticOperationType.SUBTRACT: lambda const, col: const - col,
                ArithmeticOperationType.MULTIPLY: lambda const, col: const * col,
                ArithmeticOperationType.DIVIDE: lambda const, col: const / col,
            }

            if field_config.column_is_first_operand:
                polars_expr = op_map_col_first[field_config.operation](input_col_expr, constant_expr)
                if field_config.operation == ArithmeticOperationType.DIVIDE and field_config.on_division_by_zero != 'propagate_error' and field_config.constant_value == 0: # Dividing by constant 0
                     otherwise_val = None if field_config.on_division_by_zero == 'null' else pl.lit(field_config.on_division_by_zero)
                     polars_expr = pl.when(constant_expr != 0).then(polars_expr).otherwise(otherwise_val) # Should always be constant_expr != 0 check
            else: # constant is first operand
                polars_expr = op_map_const_first[field_config.operation](constant_expr, input_col_expr)
                if field_config.operation == ArithmeticOperationType.DIVIDE and field_config.on_division_by_zero != 'propagate_error': # Dividing by input_column
                     otherwise_val = None if field_config.on_division_by_zero == 'null' else pl.lit(field_config.on_division_by_zero)
                     polars_expr = pl.when(input_col_expr != 0).then(polars_expr).otherwise(otherwise_val)


        elif field_config.calculation_type == 'percentage_of_column':
            _check_input_columns_exist(field_config.value_column, field_config.total_reference_column)
            value_col_expr = pl.col(field_config.value_column)
            total_ref_col_expr = pl.col(field_config.total_reference_column)
            polars_expr = (value_col_expr / total_ref_col_expr) * field_config.scale_factor

            if field_config.on_division_by_zero != 'propagate_error':
                otherwise_val = None if field_config.on_division_by_zero == 'null' else pl.lit(field_config.on_division_by_zero)
                polars_expr = pl.when(total_ref_col_expr != 0).then(polars_expr).otherwise(otherwise_val)
        else:
            # Should not be reached if Pydantic validation is correct
            raise ValueError(f"Unsupported calculated_field type: {getattr(field_config, 'calculation_type', 'Unknown')}")

        result_df = result_df.with_columns(polars_expr.alias(output_col_name))
        available_columns_during_calculation.add(output_col_name) # Add new column for subsequent calculations

    return result_df


async def aggregate_data(
    request: AggregationRequest,
    ctx: Context
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Performs aggregation and computes derived fields on the loaded CSV data.

    The tool takes a detailed `AggregationRequest` object specifying:
    - `group_by_columns`: Columns to group the data by.
    - `aggregations`: A list of aggregation operations (e.g., sum, mean) to
      apply to specified columns, with optional renaming of output columns.
    - `calculated_fields` (optional): A list of new fields to compute based on
      the results of the aggregation, such as arithmetic operations between
      columns or calculating percentages.

    Args:
        request: An `AggregationRequest` Pydantic model instance detailing
                 the operations to perform.
        ctx: The MCP context, automatically injected.

    Returns:
        A list of dictionaries, where each dictionary represents a row in the
        aggregated and calculated result set.
        Returns an error dictionary if issues occur (e.g., DataFrame not loaded,
        invalid column names, calculation errors).
    """
    await ctx.info(f"Received aggregation request: Group by {request.group_by_columns}, {len(request.aggregations)} aggregations.")

    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context # type: ignore
        source_df: pl.DataFrame = lifespan_ctx["df"]
    except KeyError:
        await ctx.error("DataFrame 'df' not found in lifespan context.")
        return {"error": "DataFrame not available. Server may be misconfigured or data failed to load."}

    df_column_names = set(source_df.columns)

    # Validate group_by_columns
    for col in request.group_by_columns:
        if col not in df_column_names:
            await ctx.error(f"Invalid group_by_column: '{col}' not found in DataFrame columns: {df_column_names}")
            return {"error": f"Invalid group_by_column: '{col}' not found."}

    # Validate aggregation input columns
    for agg_config in request.aggregations:
        if agg_config.column not in df_column_names:
            await ctx.error(f"Invalid aggregation source column: '{agg_config.column}' not found in DataFrame columns: {df_column_names}")
            return {"error": f"Invalid aggregation source column: '{agg_config.column}' not found."}

    try:
        # Build aggregation expressions
        # Pydantic validation in AggregationRequest should handle output name conflicts among aggregations
        # and between aggregations and group_by_columns.
        polars_agg_expressions, aggregated_column_names = _build_polars_aggregation_expressions(
            request.aggregations,
            set(request.group_by_columns),
            df_column_names,
            ctx
        )

        # Perform the Polars group_by and aggregation
        grouped_df = source_df.group_by(request.group_by_columns)
        aggregated_df = grouped_df.agg(polars_agg_expressions)

        # Apply calculated fields if any
        # Pydantic validation in AggregationRequest should handle output name conflicts
        # for calculated fields with previous columns.
        if request.calculated_fields:
            final_df = _apply_calculated_fields(
                aggregated_df,
                request.calculated_fields,
                aggregated_column_names, # These are the columns available post-aggregation
                ctx
            )
        else:
            final_df = aggregated_df

        # Sort results by group_by columns for consistent output
        if request.group_by_columns:
            final_df = final_df.sort(request.group_by_columns)

        await ctx.info(f"Aggregation successful. Result shape: {final_df.shape}")
        return final_df.to_dicts()

    except ValueError as ve: # Catch specific errors from our helpers or Pydantic re-validation
        await ctx.error(f"ValueError during aggregation processing: {ve}")
        return {"error": f"Configuration or processing error: {str(ve)}"}
    except pl.PolarsError as pe: # Catch Polars-specific errors
        await ctx.error(f"Polars error during aggregation: {pe}", exc_info=True)
        return {"error": f"Data processing error with Polars: {str(pe)}"}
    except Exception as e:
        await ctx.error(f"An unexpected error occurred during aggregation: {e}", exc_info=True)
        return {"error": f"An unexpected server error occurred: {str(e)}"}
```

--- File: `src/upphandlat_mcp/utils/__init__.py` ---
```python
# Utilities package
```

--- File: `src/upphandlat_mcp/utils/dataframe_ops.py` ---
```python
import polars as pl


def get_column_names_from_df(df: pl.DataFrame) -> list[str]:
    """
    Extracts column names from a Polars DataFrame.

    Args:
        df: The Polars DataFrame.

    Returns:
        A list of column name strings.
    """
    return df.columns


def get_schema_from_df(df: pl.DataFrame) -> dict[str, str]:
    """
    Extracts the schema (column names and their Polars types as strings)
    from a Polars DataFrame.

    Args:
        df: The Polars DataFrame.

    Returns:
        A dictionary mapping column names to their type strings.
    """
    return {col: str(dtype) for col, dtype in df.schema.items()}
```

--- File: `src/upphandlat_mcp/server.py` ---
```python
import logging
from mcp.server.fastmcp import FastMCP

from .lifespan.context import app_lifespan
from .tools.info_tools import list_columns, get_schema
from .tools.aggregation_tools import aggregate_data
from .prompts.entry_prompt import csv_aggregator_entry_point

logger = logging.getLogger(__name__)

# Initialize the FastMCP server instance
# The server name should be descriptive for LLM discovery
mcp_server = FastMCP(
    name="UpphandlatMCP",
    description="A server for querying and aggregating data from Swedish public procurement data.",
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
        mcp_server.run(mode="stdio")
        logger.info(f"MCP server '{mcp_server.name}' finished running.")
    except Exception as e:
        logger.critical(f"MCP server '{mcp_server.name}' crashed: {e}", exc_info=True)
        raise # Re-raise to allow __init__.py to catch and exit
```