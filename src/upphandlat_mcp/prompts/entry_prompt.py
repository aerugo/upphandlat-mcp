# src/upphandlat_mcp/prompts/entry_prompt.py
from mcp.server.fastmcp.prompts import base


def csv_aggregator_entry_point() -> str:
    return (
        "## Upphandlat Multi-CSV Aggregator MCP Server Interaction Guide\n\n"
        "**Objective:** You are interacting with a server that can analyze and aggregate data from multiple pre-loaded CSV files, each identified by a unique name. Your goal is to use the available tools to answer user queries about this data.\n\n"
        "**Workflow:**\n"
        "1.  **Discover Data Sources:** If you don't know which datasets are available, first use `list_available_dataframes()`.\n"
        "2.  **Select Data Source:** For all other tools, you **MUST** provide the `dataframe_name` parameter, specifying which loaded dataset to query.\n"
        "3.  **Understand Data Structure:** Use `list_columns(dataframe_name='your_chosen_df')` and `get_schema(dataframe_name='your_chosen_df')` for the selected DataFrame.\n"
        "4.  **Query/Aggregate:** Use `get_distinct_column_values(...)` or `aggregate_data(...)` on the chosen DataFrame.\n\n"
        "**Available Tools & Common Use Cases:**\n\n"
        "0.  **`list_available_dataframes()`:**\n"
        "    *   **Use When:** You need to know the names of all loaded CSV datasets.\n"
        '    *   **Example:** "What datasets are available?"\n\n'
        "1.  **`list_columns(dataframe_name: str)`:**\n"
        "    *   **Use When:** You need to know what data fields (columns) are available in a specific CSV dataset.\n"
        "    *   **Example:** \"What columns are in the 'procurement_data' dataset?\" (Requires `dataframe_name='procurement_data'`)\n\n"
        "2.  **`get_schema(dataframe_name: str)`:**\n"
        "    *   **Use When:** You need to know the data types of the columns in a specific dataset.\n"
        "    *   **Example:** \"What is the schema of the 'supplier_info' dataset?\" (Requires `dataframe_name='supplier_info'`)\n\n"
        "3.  **`get_distinct_column_values(dataframe_name: str, column_name: str, ...)`:**\n"
        "    *   **Use When:** You need to find unique values in a column of a specific dataset.\n"
        "    *   **Example:** \"List unique categories in the 'products' dataset.\" (Requires `dataframe_name='products', column_name='category'`)\n\n"
        "4.  **`aggregate_data(dataframe_name: str, request: AggregationRequest)`:**\n"
        "    *   **Use When:** The user wants to summarize, group, or calculate new metrics from data in a specific dataset.\n"
        "    *   **`dataframe_name` (str):** The name of the dataset to operate on.\n"
        "    *   **`AggregationRequest` Structure:** (Same as before, defines grouping, aggregations, calculated fields for the chosen DataFrame)\n"
        "        *   `group_by_columns` (list[str])\n"
        "        *   `aggregations` (list[Aggregation], optional)\n"
        "        *   `calculated_fields` (list[CalculatedFieldType], optional)\n"
        "    *   **Examples:**\n"
        "        *   \"What is the total sales per region in the 'sales_2023' dataset?\" -> Use `aggregate_data` with `dataframe_name='sales_2023'`, group by 'region', sum 'sales'.\n"
        "        *   \"For the 'procurement_data' dataset, calculate average quantity and total sales for each product category.\" -> Use `aggregate_data` with `dataframe_name='procurement_data'`, group by 'category', mean 'quantity', sum 'sales'.\n\n"
        "**General Strategy & Workflow (Recap):**\n\n"
        "1.  **Identify Data Source:** Use `list_available_dataframes()` if unsure. Pick one `dataframe_name`.\n"
        "2.  **Inspect Columns/Schema:** Use `list_columns()` and `get_schema()` with the chosen `dataframe_name`.\n"
        "3.  **Plan Aggregation:** If summarizing, identify `group_by_columns`, `aggregations`, and `calculated_fields` for the data within the chosen `dataframe_name`.\n"
        "4.  **Construct `AggregationRequest`:** Carefully build the JSON request for `aggregate_data`.\n"
        "5.  **Execute and Present:** Call the appropriate tool (e.g., `aggregate_data`), always providing the `dataframe_name` and other required parameters.\n"
        "6.  **Handle Errors:** If a tool returns an error, explain it to the user and try to adjust the request if possible (e.g., verify `dataframe_name`, column names).\n\n"
        "**Example of `aggregate_data` call:**\n"
        "Tool name: `aggregate_data`\n"
        "Parameters:\n"
        "```json\n"
        "{\n"
        '  "dataframe_name": "my_specific_dataset",\n'
        '  "request": {\n'
        '    "group_by_columns": ["category", "region"],\n'
        '    "aggregations": [\n'
        "      {\n"
        '        "column": "sales",\n'
        '        "functions": ["sum", "mean"],\n'
        '        "rename": {"sum": "total_sales", "mean": "average_sales"}\n'
        "      }\n"
        "    ]\n"
        "  }\n"
        "}\n"
        "```\n\n"
        "Now, analyze the user's request, determine the target `dataframe_name`, and then the best tool(s) and parameters to use."
    )
