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
        "4.  **Explore Column Content (Optional):** Use `get_distinct_column_values(...)` or `fuzzy_search_column_values(...)` to understand values in columns you might filter or group by.\n" # MODIFIED/ADDED
        "5.  **Filter, Query/Aggregate:** Use `aggregate_data(...)` on the chosen DataFrame, optionally applying filters before grouping and aggregation.\n\n" # MODIFIED

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
        # MODIFY `aggregate_data` documentation:
        "5.  **`aggregate_data(dataframe_name: str, request: AggregationRequest)`:**\n" # Ensure numbering is correct
        "    *   **Use When:** The user wants to filter, summarize, group, or calculate new metrics from data in a specific dataset, optionally including a final summary row.\n" # MODIFIED
        "    *   **`dataframe_name` (str):** The name of the dataset to operate on.\n"
        "    *   **`AggregationRequest` Structure:** Defines filtering, grouping, aggregations, calculated fields, and summary row settings.\n" # MODIFIED
        "        *   `filters` (list[FilterCondition], optional): Conditions to filter data *before* grouping/aggregation. Applied with AND logic.\n"
        "        *   `group_by_columns` (list[str])\n"
        "        *   `aggregations` (list[Aggregation], optional)\n"
        "        *   `calculated_fields` (list[CalculatedFieldType], optional)\n"
        "        *   `summary_settings` (SummaryRowSettings, optional): Configuration for adding a summary row.\n" # ADDED
        "    *   **Examples:**\n"
        "        *   \"What is the total sales per region in the 'sales_2023' dataset?\" -> Use `aggregate_data` with `dataframe_name='sales_2023'`, group by 'region', sum 'sales'.\n"
        "        *   \"For the 'procurement_data' dataset, calculate average quantity and total sales for each product category.\" -> Use `aggregate_data` with `dataframe_name='procurement_data'`, group by 'category', mean 'quantity', sum 'sales'.\n\n"

        "**`FilterCondition` Object Structure (for `filters` list):**\n"
        "Each object in the `filters` list defines one condition:\n"
        "```json\n"
        "{\n"
        '  "column": "column_to_filter_on",\n'
        '  "operator": "filter_operator",\n'
        '  "value": "value_for_comparison", // or list for "in"/"not_in", or null for "is_null"/"is_not_null"\n'
        '  "case_sensitive": true // Optional, for string ops. Defaults to false (case-insensitive). Set to true for case-sensitive.\n'  # MODIFIED example and description
        "}\n"
        "```\n"
        "Supported `filter_operator` values:\n"
        '  `"equals"`, `"not_equals"`\n'
        '  `"greater_than"`, `"greater_than_or_equal_to"`\n'
        '  `"less_than"`, `"less_than_or_equal_to"`\n'
        '  `"in"`, `"not_in"` (value must be a list, e.g., `["A", "B"]`)\n'
        '  `"contains"`, `"starts_with"`, `"ends_with"` (for string columns)\n'
        '  `"is_null"`, `"is_not_null"` (value field is ignored/should be null)\n\n'

        "**`SummaryRowSettings` Object Structure (for `summary_settings` field):**\n" # ADDED NEW SECTION
        "This optional object controls the generation of a summary row at the end of the results.\n"
        "```json\n"
        "{\n"
        '  "enabled": true, // Set to true to include the summary row\n'
        '  "default_numeric_summary_function": "sum", // "sum", "mean", "count", "min", "max"\n'
        '  "default_string_summary_function": "none", // "label", "none", "count"\n'
        '  "first_group_by_column_label": "Grand Total", // Text for the first group-by column in summary\n'
        '  "column_specific_summaries": [\n'
        "    {\n"
        '      "column_name": "name_of_a_column_in_output",\n'
        '      "summary_function": "label", // "sum", "mean", "count", "min", "max", "label", "none"\n'
        '      "label_text": "Specific Label for this Column" // Required if summary_function is "label"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n"
        "If `enabled` is `false` or `summary_settings` is omitted, no summary row is added.\n"
        "If `column_specific_summaries` is provided, it overrides default behaviors for those columns.\n"
        "Default behavior for non-configured columns:\n"
        "  - First `group_by_column`: Gets `first_group_by_column_label`.\n"
        "  - Other `group_by_columns`: Get `null`.\n"
        "  - Numeric output columns: Summarized using `default_numeric_summary_function`.\n"
        "  - String/other output columns: Summarized using `default_string_summary_function`.\n\n"

        "**General Strategy & Workflow (Recap):**\n\n"
        "1.  **Identify Data Source:** Use `list_available_dataframes()` if unsure. Pick one `dataframe_name`.\n"
        "2.  **Inspect Columns/Schema:** Use `list_columns()` and `get_schema()` with the chosen `dataframe_name`.\n"
        "3.  **Explore Specific Column Values (if needed):** Use `get_distinct_column_values()` or `fuzzy_search_column_values()`.\n"
        "4.  **Plan Filtering (if needed):** Identify any conditions to narrow down the data before aggregation.\n"
        "5.  **Plan Aggregation:** If summarizing, identify `group_by_columns`, `aggregations`, and `calculated_fields` for the data within the chosen `dataframe_name`.\n"
        "6.  **Construct `AggregationRequest`:** Carefully build the JSON request for `aggregate_data`, including any `filters` and optional `summary_settings`.\n" # MODIFIED
        "7.  **Execute and Present:** Call `aggregate_data`, always providing the `dataframe_name` and other required parameters.\n"
        "8.  **Handle Errors:** If a tool returns an error, explain it to the user and try to adjust the request if possible.\n\n"

        "**Example of `aggregate_data` call with filters and summary row:**\n" # MODIFIED EXAMPLE
        "Tool name: `aggregate_data`\n"
        "Parameters:\n"
        "```json\n"
        "{\n"
        '  "dataframe_name": "my_specific_dataset",\n'
        '  "request": {\n'
        '    "filters": [\n'
        '      {"column": "status", "operator": "equals", "value": "Active"}\n'
        '    ],\n'
        '    "group_by_columns": ["category"],\n'
        '    "aggregations": [\n'
        "      {\n"
        '        "column": "sales_amount",\n'
        '        "functions": ["sum"],\n'
        '        "rename": {"sum": "total_sales"}\n'
        "      },\n"
        "      {\n"
        '        "column": "item_id",\n'
        '        "functions": ["count"],\n'
        '        "rename": {"count": "number_of_items"}\n'
        "      }\n"
        "    ],\n"
        '    "summary_settings": {\n' # ADDED SUMMARY EXAMPLE
        '      "enabled": true,\n'
        '      "first_group_by_column_label": "All Categories Total",\n'
        '      "default_numeric_summary_function": "sum",\n'
        '      "column_specific_summaries": [\n'
        '        {"column_name": "number_of_items", "summary_function": "sum"}\n'
        '      ]\n'
        '    }\n'
        "  }\n"
        "}\n"
        "```\n\n"
        "Now, analyze the user's request, determine the target `dataframe_name`, and then the best tool(s) and parameters to use."
    )
