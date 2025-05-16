import asyncio
import logging
from typing import Any

import polars as pl
from mcp.server.fastmcp import Context
from upphandlat_mcp.lifespan.context import LifespanContext
from upphandlat_mcp.models.mcp_models import (
    Aggregation,
    AggregationRequest,
    ArithmeticOperationType,
    CalculatedFieldType,
)

logger = logging.getLogger(__name__)


async def _build_polars_aggregation_expressions(
    aggregations: list[Aggregation],
    group_by_column_names: set[str],
    existing_df_columns: set[str],
    ctx: Context,
) -> tuple[list[pl.Expr], set[str]]:
    polars_expressions: list[pl.Expr] = []
    all_output_column_names: set[str] = set(group_by_column_names)

    for agg_config in aggregations:
        if agg_config.column not in existing_df_columns:
            raise ValueError(
                f"Aggregation column '{agg_config.column}' not found in DataFrame."
            )
        for func_enum in agg_config.functions:
            function_name_str = func_enum.value
            output_alias = agg_config.rename.get(
                function_name_str, f"{agg_config.column}_{function_name_str}"
            )
            if output_alias in all_output_column_names:
                raise ValueError(f"Duplicate output column name: '{output_alias}'.")
            all_output_column_names.add(output_alias)
            try:
                column_expression = pl.col(agg_config.column)
                aggregation_function = getattr(column_expression, function_name_str)

                final_expression = aggregation_function().alias(output_alias)
                polars_expressions.append(final_expression)
            except AttributeError:
                raise ValueError(
                    f"Invalid aggregation function '{function_name_str}' for Polars on column '{agg_config.column}'."
                )
            except Exception as e:
                logger.warning(
                    f"Could not build expression for {agg_config.column}.{function_name_str}(): {e}"
                )
                raise ValueError(
                    f"Error building expression for {agg_config.column}.{function_name_str}(): {e}"
                )
    return polars_expressions, all_output_column_names


def _apply_calculated_fields(
    df: pl.DataFrame,
    calculated_field_configs: list[CalculatedFieldType],
    all_current_column_names: set[str],
    ctx: Context,
) -> pl.DataFrame:
    if not calculated_field_configs:
        return df

    available_columns_during_calculation = set(all_current_column_names)
    result_df = df

    for field_config in calculated_field_configs:
        output_col_name = field_config.output_column_name

        if output_col_name in available_columns_during_calculation:
            raise ValueError(
                f"Calculated field output name '{output_col_name}' conflicts with an existing or previously calculated column."
            )

        def _check_input_columns_exist(*cols_to_check: str) -> None:
            for col_name in cols_to_check:
                if col_name not in available_columns_during_calculation:
                    raise ValueError(
                        f"Input column '{col_name}' for calculated field '{output_col_name}' not found. "
                        f"Available columns: {sorted(list(available_columns_during_calculation))}"
                    )

        polars_expr: pl.Expr

        if field_config.calculation_type == "two_column_arithmetic":
            cfg = field_config
            _check_input_columns_exist(cfg.column_a, cfg.column_b)
            col_a_expr = pl.col(cfg.column_a)
            col_b_expr = pl.col(cfg.column_b)
            op_map = {
                ArithmeticOperationType.ADD: lambda a, b: a + b,
                ArithmeticOperationType.SUBTRACT: lambda a, b: a - b,
                ArithmeticOperationType.MULTIPLY: lambda a, b: a * b,
                ArithmeticOperationType.DIVIDE: lambda a, b: a / b,
            }
            polars_expr = op_map[cfg.operation](col_a_expr, col_b_expr)

            if (
                cfg.operation == ArithmeticOperationType.DIVIDE
                and cfg.on_division_by_zero != "propagate_error"
            ):
                otherwise_val = (
                    None
                    if cfg.on_division_by_zero == "null"
                    else pl.lit(cfg.on_division_by_zero, dtype=pl.Float64)
                )
                polars_expr = (
                    pl.when(col_b_expr != 0).then(polars_expr).otherwise(otherwise_val)
                )
        elif field_config.calculation_type == "constant_arithmetic":
            cfg = field_config
            _check_input_columns_exist(cfg.input_column)
            input_col_expr = pl.col(cfg.input_column)
            constant_expr = pl.lit(cfg.constant_value)

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

            if cfg.column_is_first_operand:
                polars_expr = op_map_col_first[cfg.operation](
                    input_col_expr, constant_expr
                )
                if (
                    cfg.operation == ArithmeticOperationType.DIVIDE
                    and cfg.on_division_by_zero != "propagate_error"
                    and cfg.constant_value == 0
                ):
                    otherwise_val = (
                        None
                        if cfg.on_division_by_zero == "null"
                        else pl.lit(cfg.on_division_by_zero, dtype=pl.Float64)
                    )
                    polars_expr = (
                        pl.when(constant_expr != 0)
                        .then(polars_expr)
                        .otherwise(otherwise_val)
                    )
            else:
                polars_expr = op_map_const_first[cfg.operation](
                    constant_expr, input_col_expr
                )
                if (
                    cfg.operation == ArithmeticOperationType.DIVIDE
                    and cfg.on_division_by_zero != "propagate_error"
                ):
                    otherwise_val = (
                        None
                        if cfg.on_division_by_zero == "null"
                        else pl.lit(cfg.on_division_by_zero, dtype=pl.Float64)
                    )
                    polars_expr = (
                        pl.when(input_col_expr != 0)
                        .then(polars_expr)
                        .otherwise(otherwise_val)
                    )
        elif field_config.calculation_type == "percentage_of_column":
            cfg = field_config
            _check_input_columns_exist(cfg.value_column, cfg.total_reference_column)
            value_col_expr = pl.col(cfg.value_column)
            total_ref_col_expr = pl.col(cfg.total_reference_column)
            polars_expr = (value_col_expr / total_ref_col_expr) * cfg.scale_factor

            if cfg.on_division_by_zero != "propagate_error":
                otherwise_val = (
                    None
                    if cfg.on_division_by_zero == "null"
                    else pl.lit(cfg.on_division_by_zero, dtype=pl.Float64)
                )
                polars_expr = (
                    pl.when(total_ref_col_expr != 0)
                    .then(polars_expr)
                    .otherwise(otherwise_val)
                )
        else:
            w = ctx.error(
                f"Unsupported calculated_field type: {getattr(field_config, 'calculation_type', 'Unknown')}"
            )
            raise ValueError(
                f"Unsupported calculated_field type: {getattr(field_config, 'calculation_type', 'Unknown')}"
            )

        result_df = result_df.with_columns(polars_expr.alias(output_col_name))
        available_columns_during_calculation.add(output_col_name)

    return result_df


async def aggregate_data(
    ctx: Context,
    dataframe_name: str,
    request: AggregationRequest,
) -> list[dict[str, Any]] | dict[str, Any]:
    """
    Performs powerful data aggregation, grouping, and calculations on a specified dataset.

    This is the primary tool for summarizing data, finding trends, calculating metrics,
    and deriving new insights from the available CSV datasets. It allows you to:
    1. Group data by one or more columns.
    2. Apply multiple aggregation functions (sum, mean, count, min, max) to different columns.
    3. Create new columns based on arithmetic operations or percentage calculations, either on
       aggregated results or on original data.

    Args:
        ctx: The MCP context (automatically provided).
        dataframe_name (str): The name of the DataFrame to process.
            Discover available DataFrames using the `list_available_dataframes()` tool.
        request (AggregationRequest): A Pydantic model detailing the aggregation and
            calculation steps. See details below.

    Returns:
        list[dict[str, Any]]: A list of dictionaries, where each dictionary represents
            a row in the resulting aggregated and calculated table.
        dict[str, Any]: An error dictionary if an issue occurs (e.g., column not found,
            invalid request structure).

    **Structure of the `AggregationRequest` object:**

    The `request` argument must be a JSON object with the following structure:

    ```json
    {
      "group_by_columns": ["column_name1", "column_name2"],
      "aggregations": [
        {
          "column": "numeric_column_to_aggregate",
          "functions": ["sum", "mean"],
          "rename": { "sum": "total_value", "mean": "average_value" }
        }
      ],
      "calculated_fields": [
        {
          "calculation_type": "two_column_arithmetic",
          "output_column_name": "profit_margin",
          "column_a": "total_revenue_agg",
          "column_b": "total_cost_agg",
          "operation": "subtract"
        }
      ]
    }
    ```

    **Detailed breakdown of `AggregationRequest` fields:**

    1.  **`group_by_columns: list[str]` (Required)**
        *   A list of column names to group the data by. At least one column must be provided.
        *   The unique combinations of values in these columns will form the groups for aggregation.
        *   Example: `["År", "Sektor för köpare"]` would group data by year and buyer sector.
        *   These columns will always be present in the output.

    2.  **`aggregations: list[Aggregation] | None` (Optional)**
        *   A list of `Aggregation` objects, each defining how to aggregate a specific column from the
          original `dataframe_name` over the defined groups.
        *   If omitted, `null`, or an empty list, no new summary aggregations are computed.
          `calculated_fields` (if any) will then operate on the original columns of the DataFrame
          (see "Calculated Fields with No Aggregations" below).
        *   Each `Aggregation` object has:
            *   `column: str` (Required): The name of the column from the source DataFrame to aggregate
              (e.g., `"Antal upphandlingar, Antal"`).
            *   `functions: list[AggFunc]` (Required): A list of aggregation functions to apply.
                Supported `AggFunc` values are:
                *   `"sum"`: Calculates the sum of values in the group.
                *   `"mean"`: Calculates the average of values in the group.
                *   `"count"`: Counts the number of non-null values in the specified `column` within each group.
                *   `"min"`: Finds the minimum value in the group.
                *   `"max"`: Finds the maximum value in the group.
            *   `rename: dict[str, str]` (Optional): A dictionary to provide custom names for the
              output columns generated by this aggregation.
                *   Keys must be one of the `AggFunc` values used in `functions` (e.g., `"sum"`).
                *   Values are the desired new column names (e.g., `"Total_Antal"`).
                *   If not provided, or a function is not renamed, output columns are named
                  automatically as `{column}_{function}` (e.g., `"Antal upphandlingar, Antal_sum"`).

    3.  **`calculated_fields: list[CalculatedFieldType] | None` (Optional)**
        *   A list of objects, each defining a calculation to create a new column.
        *   **Order of Operation:** Calculations are performed *after* all aggregations (if any) are completed.
        *   **Input Columns for Calculations:**
            *   If `aggregations` are defined and produce results: Calculated fields can use the
              `group_by_columns` and any newly created aggregated columns as inputs.
            *   If `aggregations` is `null` or empty: Calculated fields operate directly on the
              original columns of the `dataframe_name`.
        *   Each calculated field object requires:
            *   `output_column_name: str` (Required): The name for the new calculated column. This name
              must be unique and not conflict with `group_by_columns` or aggregated column names.
            *   `calculation_type: str` (Required): Specifies the type of calculation. One of:
                `"two_column_arithmetic"`, `"constant_arithmetic"`, `"percentage_of_column"`.

        *   **Types of Calculated Fields:**

            *   **`TwoColumnArithmeticConfig` (for `calculation_type: "two_column_arithmetic"`)**
                *   Performs arithmetic between two existing columns (A op B).
                *   `column_a: str`: Name of the first column operand.
                *   `column_b: str`: Name of the second column operand.
                *   `operation: ArithmeticOperationType`: The operation to perform. Supported values:
                    `"add"`, `"subtract"`, `"multiply"`, `"divide"`.
                *   `on_division_by_zero: float | "null" | "propagate_error"` (Default: `"propagate_error"`):
                  Behavior for division by zero (if `operation` is `"divide"` and `column_b` is zero).
                  Can be a specific float (e.g., `0.0`), `"null"` (inserts a null value), or
                  `"propagate_error"` (Polars default, may result in `inf` or `NaN`).

            *   **`ConstantArithmeticConfig` (for `calculation_type: "constant_arithmetic"`)**
                *   Performs arithmetic between a column and a constant value.
                *   `input_column: str`: Name of the column operand.
                *   `constant_value: float`: The constant number for the operation.
                *   `operation: ArithmeticOperationType`: Same as above (`"add"`, `"subtract"`, `"multiply"`, `"divide"`).
                *   `column_is_first_operand: bool` (Default: `true`): If `true`, the operation is
                  `input_column op constant_value`. If `false`, it's `constant_value op input_column`.
                *   `on_division_by_zero: float | "null" | "propagate_error"`: Relevant if division by zero
                  could occur (e.g., `input_column / 0` or `constant_value / 0` if `input_column` is zero).

            *   **`PercentageOfColumnConfig` (for `calculation_type: "percentage_of_column"`)**
                *   Calculates one column as a percentage of another: `(value_column / total_reference_column) * scale_factor`.
                *   `value_column: str`: The column representing the part/numerator.
                *   `total_reference_column: str`: The column representing the total/base/denominator.
                *   `scale_factor: float` (Default: `100.0`): Factor to multiply the ratio by (e.g., `100.0` for percentage).
                *   `on_division_by_zero: float | "null" | "propagate_error"`: Behavior if `total_reference_column` is zero.

    **How to Use This Tool Effectively:**

    1.  **Discover Data:**
        *   Use `list_available_dataframes()` to get the `dataframe_name` for the dataset you want to analyze.
        *   Use `list_columns(dataframe_name="...")` to see all available column names.
        *   Use `get_schema(dataframe_name="...")` to understand the data types of columns. This is crucial for
          choosing appropriate aggregation functions and arithmetic operations.
        *   For exploring categorical values for potential `group_by_columns`, use
          `get_distinct_column_values(...)` or `fuzzy_search_column_values(...)`.

    2.  **Plan Your Analysis:**
        *   **Grouping:** What are the key dimensions for your summary? (e.g., "by Year and Region", "per Product Category"). These become your `group_by_columns`.
        *   **Aggregation:** What numerical facts do you need for each group? (e.g., "total sales", "average quantity", "number of occurrences"). This defines your `aggregations`.
            *   Remember: `count` on a column counts non-null entries of *that column* within each group. If you want to count rows in a group, you can `count` any column that is always non-null for those rows (like a primary ID), or any column really, if you just need a row count per group.
        *   **Derivation:** Do you need to calculate new metrics from the grouped/aggregated data or from original data? (e.g., "profit margin = revenue - cost", "percentage change"). This defines your `calculated_fields`.

    3.  **Construct the `AggregationRequest` JSON:**
        *   Be meticulous with column names; they must exactly match those in the DataFrame or those generated by preceding aggregation steps (if used as input to calculated fields).
        *   Use `rename` in aggregations for clear, understandable output column names.
        *   Ensure `output_column_name` for calculated fields is unique.
        *   Verify that columns used in arithmetic or numeric aggregations are of appropriate data types (e.g., numeric, not string).

    4.  **Calculated Fields with No Aggregations:**
        *   If the `aggregations` field is omitted, `null`, or an empty list, `calculated_fields` will operate directly on the columns of the original `dataframe_name`.
        *   The tool will then effectively apply these calculations row-wise to the original data.
        *   The final output will include the `group_by_columns` and the newly created `calculated_fields`, showing unique combinations of these selected columns from the (potentially transformed) original data. This is useful for deriving new columns without performing summary aggregation.

    **Example `AggregationRequest` (using sample data from `upphandlingsmyndigheten_antal_upphandlingar`):**

    Suppose the dataset `upphandlingsmyndigheten_antal_upphandlingar` has columns like "År" (Year),
    "Sektor för köpare" (Buyer Sector), "Upphandlings-ID" (Procurement ID), and
    "Antal upphandlingar, Antal" (Number of Procurements - likely the value to sum up per row).

    **Goal:** Calculate the total number of procurements and the count of unique procurement events (rows)
    per year and buyer sector. Then, calculate the average number of procurements per event (assuming
    "Antal upphandlingar, Antal" can be greater than 1 for a single event/row, or if it represents a value).

    ```json
    {
      "dataframe_name": "upphandlingsmyndigheten_antal_upphandlingar",
      "request": {
        "group_by_columns": ["År", "Sektor för köpare"],
        "aggregations": [
          {
            "column": "Antal upphandlingar, Antal",
            "functions": ["sum"],
            "rename": { "sum": "Totala_Antal_Upphandlingar_Värde" }
          },
          {
            "column": "Upphandlings-ID", // Assuming each row is a unique procurement event
            "functions": ["count"],
            "rename": { "count": "Antal_Upphandlingstillfällen" }
          }
        ],
        "calculated_fields": [
          {
            "calculation_type": "two_column_arithmetic",
            "output_column_name": "Genomsnittligt_Värde_Per_Tillfälle",
            "column_a": "Totala_Antal_Upphandlingar_Värde",
            "column_b": "Antal_Upphandlingstillfällen",
            "operation": "divide",
            "on_division_by_zero": "null"
          }
        ]
      }
    }
    ```

    This tool is designed to be flexible. If your query is complex, break it down into these
    components (grouping, aggregation, calculation) to construct the request.
    """
    await ctx.info(
        f"Received aggregation request for DataFrame '{dataframe_name}': "
        f"Group by {request.group_by_columns}, "
        f"Aggregations: {'Present and non-empty' if request.aggregations and len(request.aggregations) > 0 else 'None/Empty'}, "
        f"Calculated Fields: {'Present' if request.calculated_fields else 'None/Empty'}."
    )

    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context
        df_dict = lifespan_ctx["dataframes"]
        if dataframe_name not in df_dict:
            await ctx.error(
                f"DataFrame '{dataframe_name}' not found. Available: {list(df_dict.keys())}"
            )
            return {
                "error": f"DataFrame '{dataframe_name}' not found. Use list_available_dataframes() to see options."
            }
        source_df: pl.DataFrame = df_dict[dataframe_name]
    except KeyError:
        await ctx.error(
            "DataFrame dictionary 'dataframes' not found in lifespan context."
        )
        return {"error": "DataFrames not available. Server may be misconfigured."}

    df_column_names = set(source_df.columns)

    for col in request.group_by_columns:
        if col not in df_column_names:
            await ctx.error(
                f"Invalid group_by_column for DataFrame '{dataframe_name}': '{col}' not found. Available: {list(df_column_names)}"
            )
            return {
                "error": f"Invalid group_by_column: '{col}' not found in '{dataframe_name}'."
            }

    if request.aggregations:
        for agg_config in request.aggregations:
            if agg_config.column not in df_column_names:
                await ctx.error(
                    f"Invalid aggregation source column for DataFrame '{dataframe_name}': '{agg_config.column}' not found. Available: {list(df_column_names)}"
                )
                return {
                    "error": f"Invalid aggregation source column: '{agg_config.column}' not found in '{dataframe_name}'."
                }
    try:
        intermediate_df: pl.DataFrame
        columns_after_aggregation_or_grouping: set[str]

        if request.aggregations and len(request.aggregations) > 0:
            (
                polars_agg_expressions,
                aggregated_column_names,
            ) = await _build_polars_aggregation_expressions(
                request.aggregations,
                set(request.group_by_columns),
                df_column_names,
                ctx,
            )
            grouped_df = source_df.group_by(
                request.group_by_columns, maintain_order=True
            )
            intermediate_df = grouped_df.agg(polars_agg_expressions)
            columns_after_aggregation_or_grouping = aggregated_column_names
            await ctx.info(
                f"Performed aggregation on '{dataframe_name}'. Resulting columns: {intermediate_df.columns}"
            )
        else:
            if not request.group_by_columns:
                await ctx.error("`group_by_columns` must be provided.")
                return {
                    "error": "`group_by_columns` must be provided even if no aggregations are performed."
                }

            intermediate_df = source_df.select(request.group_by_columns).unique(
                maintain_order=True
            )
            columns_after_aggregation_or_grouping = set(request.group_by_columns)
            await ctx.info(
                f"No aggregations requested for '{dataframe_name}'. Initial columns for potential calculated fields: {intermediate_df.columns} (based on group_by)."
            )

        if request.calculated_fields:

            cols_available_for_calc: set[str]
            df_for_calc: pl.DataFrame

            if request.aggregations and len(request.aggregations) > 0:
                df_for_calc = intermediate_df
                cols_available_for_calc = columns_after_aggregation_or_grouping
            else:
                df_for_calc = source_df
                cols_available_for_calc = df_column_names

            final_df = await asyncio.to_thread(
                _apply_calculated_fields,
                df_for_calc,
                request.calculated_fields,
                cols_available_for_calc,
                ctx,
            )
            await ctx.info(
                f"Applied calculated fields to '{dataframe_name}'. Resulting columns: {final_df.columns}"
            )
            if not (request.aggregations and len(request.aggregations) > 0):
                temp_output_cols = list(request.group_by_columns) + [
                    cf.output_column_name for cf in request.calculated_fields
                ]
                final_df = final_df.select(temp_output_cols)

        else:
            final_df = intermediate_df

        columns_to_select_final = list(request.group_by_columns)

        if request.aggregations and len(request.aggregations) > 0:
            for alias in columns_after_aggregation_or_grouping:
                if alias not in columns_to_select_final:
                    columns_to_select_final.append(alias)

        if request.calculated_fields:
            for cf_item in request.calculated_fields:
                if cf_item.output_column_name not in columns_to_select_final:
                    columns_to_select_final.append(cf_item.output_column_name)

        final_columns_present_in_df = [
            col for col in columns_to_select_final if col in final_df.columns
        ]

        if not final_columns_present_in_df:
            await ctx.warning(
                f"No valid output columns to select in the final DataFrame for '{dataframe_name}'. "
                f"Requested: {columns_to_select_final}, Available in final_df: {final_df.columns}. "
                "This might happen if group_by_columns were requested but no aggregations or calculations produced other usable columns, "
                "or if calculated fields on original data did not align with group_by structure."
            )
            if (
                all(gb_col in source_df.columns for gb_col in request.group_by_columns)
                and set(final_columns_present_in_df) == set(request.group_by_columns)
                and not (request.aggregations and len(request.aggregations) > 0)
                and not request.calculated_fields
            ):
                final_df_to_return = source_df.select(request.group_by_columns).unique(
                    maintain_order=True
                )
            elif not request.group_by_columns:
                return []
            else:
                valid_gb_cols = [
                    gbc for gbc in request.group_by_columns if gbc in final_df.columns
                ]
                if valid_gb_cols:
                    final_df_to_return = final_df.select(valid_gb_cols).unique(
                        maintain_order=True
                    )
                else:
                    return []
        else:
            final_df_to_return = final_df.select(final_columns_present_in_df)

        if not (request.aggregations and len(request.aggregations) > 0):
            if final_df_to_return.height > 0 and final_df_to_return.columns:
                final_df_to_return = final_df_to_return.unique(
                    subset=final_df_to_return.columns, maintain_order=True
                )

        sortable_group_by_cols = [
            gbc for gbc in request.group_by_columns if gbc in final_df_to_return.columns
        ]
        if sortable_group_by_cols:
            final_df_to_return = final_df_to_return.sort(sortable_group_by_cols)

        await ctx.info(
            f"Aggregation/calculation for '{dataframe_name}' successful. Final result shape: {final_df_to_return.shape}, Columns: {final_df_to_return.columns}"
        )
        return final_df_to_return.to_dicts()

    except ValueError as ve:
        await ctx.error(
            f"ValueError during aggregation processing for '{dataframe_name}': {ve}"
        )
        logger.error(f"Detailed ValueError for '{dataframe_name}': {ve}", exc_info=True)
        return {"error": f"Configuration or processing error: {str(ve)}"}
    except pl.PolarsError as pe:
        await ctx.error(
            f"Polars error during aggregation for '{dataframe_name}': {pe}",
            exc_info=True,
        )
        return {"error": f"Data processing error with Polars: {str(pe)}"}
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred during aggregation for '{dataframe_name}': {e}",
            exc_info=True,
        )
        return {"error": f"An unexpected server error occurred: {str(e)}"}
