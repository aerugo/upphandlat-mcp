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