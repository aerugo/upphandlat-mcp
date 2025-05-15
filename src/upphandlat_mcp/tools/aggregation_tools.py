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
    ConstantArithmeticConfig,
    PercentageOfColumnConfig,
    TwoColumnArithmeticConfig,
)

logger = logging.getLogger(__name__)


def _build_polars_aggregation_expressions(
    aggregations: list[Aggregation],
    group_by_column_names: set[str],
    existing_df_columns: set[str],
    ctx: Context,  # For logging
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
            raise ValueError(
                f"Aggregation column '{agg_config.column}' not found in DataFrame."
            )

        for func_enum in agg_config.functions:
            function_name_str = func_enum.value  # e.g., "sum"

            # Determine the output alias for this specific aggregation
            # Default: {column_name}_{function_name}, e.g., "sales_sum"
            # Custom if provided in rename: e.g., "total_sales"
            output_alias = agg_config.rename.get(
                function_name_str, f"{agg_config.column}_{function_name_str}"
            )

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
                aggregation_function = getattr(
                    column_expression, function_name_str
                )  # e.g., pl.col("sales").sum
                final_expression = aggregation_function().alias(output_alias)
                polars_expressions.append(final_expression)
            except AttributeError:
                # Should not happen if AggFunc enum is correct
                raise ValueError(
                    f"Invalid aggregation function '{function_name_str}' for Polars."
                )
            except (
                Exception
            ) as e:  # Catch Polars-specific errors during expression building
                w = ctx.warning(
                    f"Could not build expression for {agg_config.column}.{function_name_str}(): {e}"
                )
                # Decide: skip this expression or raise? Raising might be better.
                raise ValueError(
                    f"Error building expression for {agg_config.column}.{function_name_str}(): {e}"
                )

    return polars_expressions, all_output_column_names


def _apply_calculated_fields(
    df: pl.DataFrame,
    calculated_field_configs: list[CalculatedFieldType],
    # all_current_column_names is used to check input cols and prevent output conflicts
    all_current_column_names: set[str],
    ctx: Context,  # For logging
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
            raise ValueError(
                f"Calculated field output name '{output_col_name}' conflicts with an existing or previously calculated column."
            )

        # Helper to check if input columns for the calculation exist
        def _check_input_columns_exist(*cols_to_check: str) -> None:
            for col_name in cols_to_check:
                if col_name not in available_columns_during_calculation:
                    raise ValueError(
                        f"Input column '{col_name}' for calculated field '{output_col_name}' not found. "
                        f"Available columns: {sorted(list(available_columns_during_calculation))}"
                    )

        polars_expr: pl.Expr

        if field_config.calculation_type == "two_column_arithmetic":
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

            if (
                field_config.operation == ArithmeticOperationType.DIVIDE
                and field_config.on_division_by_zero != "propagate_error"
            ):
                otherwise_val = (
                    None
                    if field_config.on_division_by_zero == "null"
                    else pl.lit(field_config.on_division_by_zero)
                )
                polars_expr = (
                    pl.when(col_b_expr != 0).then(polars_expr).otherwise(otherwise_val)
                )

        elif field_config.calculation_type == "constant_arithmetic":
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
                polars_expr = op_map_col_first[field_config.operation](
                    input_col_expr, constant_expr
                )
                if (
                    field_config.operation == ArithmeticOperationType.DIVIDE
                    and field_config.on_division_by_zero != "propagate_error"
                    and field_config.constant_value == 0
                ):  # Dividing by constant 0
                    otherwise_val = (
                        None
                        if field_config.on_division_by_zero == "null"
                        else pl.lit(field_config.on_division_by_zero)
                    )
                    polars_expr = (
                        pl.when(constant_expr != 0)
                        .then(polars_expr)
                        .otherwise(otherwise_val)
                    )  # Should always be constant_expr != 0 check
            else:  # constant is first operand
                polars_expr = op_map_const_first[field_config.operation](
                    constant_expr, input_col_expr
                )
                if (
                    field_config.operation == ArithmeticOperationType.DIVIDE
                    and field_config.on_division_by_zero != "propagate_error"
                ):  # Dividing by input_column
                    otherwise_val = (
                        None
                        if field_config.on_division_by_zero == "null"
                        else pl.lit(field_config.on_division_by_zero)
                    )
                    polars_expr = (
                        pl.when(input_col_expr != 0)
                        .then(polars_expr)
                        .otherwise(otherwise_val)
                    )

        elif field_config.calculation_type == "percentage_of_column":
            _check_input_columns_exist(
                field_config.value_column, field_config.total_reference_column
            )
            value_col_expr = pl.col(field_config.value_column)
            total_ref_col_expr = pl.col(field_config.total_reference_column)
            polars_expr = (
                value_col_expr / total_ref_col_expr
            ) * field_config.scale_factor

            if field_config.on_division_by_zero != "propagate_error":
                otherwise_val = (
                    None
                    if field_config.on_division_by_zero == "null"
                    else pl.lit(field_config.on_division_by_zero)
                )
                polars_expr = (
                    pl.when(total_ref_col_expr != 0)
                    .then(polars_expr)
                    .otherwise(otherwise_val)
                )
        else:
            # Should not be reached if Pydantic validation is correct
            raise ValueError(
                f"Unsupported calculated_field type: {getattr(field_config, 'calculation_type', 'Unknown')}"
            )

        result_df = result_df.with_columns(polars_expr.alias(output_col_name))
        available_columns_during_calculation.add(
            output_col_name
        )  # Add new column for subsequent calculations

    return result_df


async def aggregate_data(
    request: AggregationRequest, ctx: Context
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
    await ctx.info(
        f"Received aggregation request: Group by {request.group_by_columns}, {len(request.aggregations)} aggregations."
    )

    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context  # type: ignore
        source_df: pl.DataFrame = lifespan_ctx["df"]
    except KeyError:
        await ctx.error("DataFrame 'df' not found in lifespan context.")
        return {
            "error": "DataFrame not available. Server may be misconfigured or data failed to load."
        }

    df_column_names = set(source_df.columns)

    # Validate group_by_columns
    for col in request.group_by_columns:
        if col not in df_column_names:
            await ctx.error(
                f"Invalid group_by_column: '{col}' not found in DataFrame columns: {df_column_names}"
            )
            return {"error": f"Invalid group_by_column: '{col}' not found."}

    # Validate aggregation input columns
    for agg_config in request.aggregations:
        if agg_config.column not in df_column_names:
            await ctx.error(
                f"Invalid aggregation source column: '{agg_config.column}' not found in DataFrame columns: {df_column_names}"
            )
            return {
                "error": f"Invalid aggregation source column: '{agg_config.column}' not found."
            }

    try:
        # Build aggregation expressions
        # Pydantic validation in AggregationRequest should handle output name conflicts among aggregations
        # and between aggregations and group_by_columns.
        polars_agg_expressions, aggregated_column_names = (
            _build_polars_aggregation_expressions(
                request.aggregations,
                set(request.group_by_columns),
                df_column_names,
                ctx,
            )
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
                aggregated_column_names,  # These are the columns available post-aggregation
                ctx,
            )
        else:
            final_df = aggregated_df

        # Sort results by group_by columns for consistent output
        if request.group_by_columns:
            final_df = final_df.sort(request.group_by_columns)

        await ctx.info(f"Aggregation successful. Result shape: {final_df.shape}")
        return final_df.to_dicts()

    except (
        ValueError
    ) as ve:  # Catch specific errors from our helpers or Pydantic re-validation
        await ctx.error(f"ValueError during aggregation processing: {ve}")
        return {"error": f"Configuration or processing error: {str(ve)}"}
    except pl.PolarsError as pe:  # Catch Polars-specific errors
        await ctx.error(f"Polars error during aggregation: {pe}", exc_info=True)
        return {"error": f"Data processing error with Polars: {str(pe)}"}
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred during aggregation: {e}", exc_info=True
        )
        return {"error": f"An unexpected server error occurred: {str(e)}"}
