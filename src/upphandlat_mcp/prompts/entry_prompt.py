from mcp.server.fastmcp.prompts import base  # For potential use of structured messages


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
        '    *   **Example:** "What columns are in the dataset?"\n\n'
        "2.  **`get_schema()`:**\n"
        "    *   **Use When:** You need to know the data types of the columns (e.g., string, integer, float) to understand how they can be used in aggregations or calculations.\n"
        '    *   **Example:** "What is the schema of the dataset?" or "What type is the \'sales\' column?"\n\n'
        "3.  **`aggregate_data(request: AggregationRequest)`:**\n"
        "    *   **Use When:** The user wants to summarize, group, or calculate new metrics from the data. This is the most powerful tool.\n"
        "    *   **`AggregationRequest` Structure:** This tool takes a complex JSON object. You'll need to construct it carefully based on the user's request. Key parts:\n"
        "        *   `group_by_columns` (list[str]): Columns to group by (e.g., ['category', 'region']).\n"
        "        *   `aggregations` (list[Aggregation]): Operations on columns within each group.\n"
        '            *   `Aggregation` object: `{ "column": "sales", "functions": ["sum", "mean"], "rename": {"sum": "total_sales"} }`\n'
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
        '  "group_by_columns": ["category", "region"],\n'
        '  "aggregations": [\n'
        "    {\n"
        '      "column": "sales",\n'
        '      "functions": ["sum", "mean"],\n'
        '      "rename": {"sum": "total_sales", "mean": "average_sales"}\n'
        "    },\n"
        "    {\n"
        '      "column": "quantity",\n'
        '      "functions": ["sum"],\n'
        '      "rename": {"sum": "total_quantity"}\n'
        "    }\n"
        "  ],\n"
        '  "calculated_fields": [\n'
        "    {\n"
        '      "calculation_type": "two_column_arithmetic",\n'
        '      "output_column_name": "revenue_per_unit",\n'
        '      "column_a": "total_sales",\n'
        '      "column_b": "total_quantity",\n'
        '      "operation": "divide",\n'
        '      "on_division_by_zero": "null"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "```\n\n"
        "Now, analyze the user's request and determine the best tool(s) and parameters to use."
    )
