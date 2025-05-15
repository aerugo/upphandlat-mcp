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
                if function_name_str == "n_unique":
                    aggregation_function = getattr(column_expression, "n_unique")
                else:
                    aggregation_function = getattr(column_expression, function_name_str)

                final_expression = aggregation_function().alias(output_alias)
                polars_expressions.append(final_expression)
            except AttributeError:
                raise ValueError(
                    f"Invalid aggregation function '{function_name_str}' for Polars."
                )
            except Exception as e:
                await ctx.warning(
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
                    else pl.lit(cfg.on_division_by_zero)
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
                        else pl.lit(cfg.on_division_by_zero)
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
                        else pl.lit(cfg.on_division_by_zero)
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
                    else pl.lit(cfg.on_division_by_zero)
                )
                polars_expr = (
                    pl.when(total_ref_col_expr != 0)
                    .then(polars_expr)
                    .otherwise(otherwise_val)
                )
        else:
            raise ValueError(
                f"Unsupported calculated_field type: {getattr(field_config, 'calculation_type', 'Unknown')}"
            )

        result_df = result_df.with_columns(polars_expr.alias(output_col_name))
        available_columns_during_calculation.add(output_col_name)

    return result_df


async def aggregate_data(
    request: AggregationRequest, ctx: Context
) -> list[dict[str, Any]] | dict[str, Any]:
    await ctx.info(
        f"Received aggregation request: Group by {request.group_by_columns}, "
        f"Aggregations: {'Present and non-empty' if request.aggregations and len(request.aggregations) > 0 else 'None/Empty'}, "
        f"Calculated Fields: {'Present' if request.calculated_fields else 'None/Empty'}."
    )

    try:
        lifespan_ctx: LifespanContext = ctx.request_context.lifespan_context
        source_df: pl.DataFrame = lifespan_ctx["df"]
    except KeyError:
        await ctx.error("DataFrame 'df' not found in lifespan context.")
        return {
            "error": "DataFrame not available. Server may be misconfigured or data failed to load."
        }

    df_column_names = set(source_df.columns)

    for col in request.group_by_columns:
        if col not in df_column_names:
            await ctx.error(
                f"Invalid group_by_column: '{col}' not found. Available: {list(df_column_names)}"
            )
            return {"error": f"Invalid group_by_column: '{col}' not found."}

    if request.aggregations:
        for agg_config in request.aggregations:
            if agg_config.column not in df_column_names:
                await ctx.error(
                    f"Invalid aggregation source column: '{agg_config.column}' not found. Available: {list(df_column_names)}"
                )
                return {
                    "error": f"Invalid aggregation source column: '{agg_config.column}' not found."
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
                f"Performed aggregation. Resulting columns: {intermediate_df.columns}"
            )
        else:
            intermediate_df = source_df
            columns_after_aggregation_or_grouping = set(df_column_names)
            await ctx.info(
                "No aggregations requested. Proceeding with original columns for calculated fields."
            )

        if request.calculated_fields:
            final_df = await asyncio.to_thread(
                _apply_calculated_fields,
                intermediate_df,
                request.calculated_fields,
                columns_after_aggregation_or_grouping,
                ctx,
            )
            await ctx.info(
                f"Applied calculated fields. Resulting columns: {final_df.columns}"
            )
        else:
            final_df = intermediate_df

        columns_to_select = list(request.group_by_columns)

        if request.aggregations and len(request.aggregations) > 0:
            for agg_item in request.aggregations:
                for func_item in agg_item.functions:
                    alias = agg_item.rename.get(
                        func_item.value, f"{agg_item.column}_{func_item.value}"
                    )
                    if alias not in columns_to_select:
                        columns_to_select.append(alias)

        if request.calculated_fields:
            for cf_item in request.calculated_fields:
                if cf_item.output_column_name not in columns_to_select:
                    columns_to_select.append(cf_item.output_column_name)

        final_columns_present = [
            col for col in columns_to_select if col in final_df.columns
        ]

        if not final_columns_present:
            if request.group_by_columns and all(
                c in source_df.columns for c in request.group_by_columns
            ):
                await ctx.info(
                    "No aggregations or calculated fields resulted in specified output columns other than group_by columns. "
                    "Returning unique group_by columns from original data."
                )
                final_df_to_return = source_df.select(request.group_by_columns).unique()
            else:
                await ctx.warning(
                    "No columns to select in the final DataFrame, or group_by columns are invalid."
                )
                return []
        else:
            final_df_to_return = final_df.select(final_columns_present)

        if not (request.aggregations and len(request.aggregations) > 0):
            if final_df_to_return.height > 0:
                final_df_to_return = final_df_to_return.unique(
                    subset=final_df_to_return.columns
                )

        if request.group_by_columns and all(
            gbc in final_df_to_return.columns for gbc in request.group_by_columns
        ):
            final_df_to_return = final_df_to_return.sort(request.group_by_columns)

        await ctx.info(
            f"Aggregation/calculation successful. Final result shape: {final_df_to_return.shape}, Columns: {final_df_to_return.columns}"
        )
        return final_df_to_return.to_dicts()

    except ValueError as ve:
        await ctx.error(f"ValueError during aggregation processing: {ve}")
        return {"error": f"Configuration or processing error: {str(ve)}"}
    except pl.PolarsError as pe:
        await ctx.error(f"Polars error during aggregation: {pe}", exc_info=True)
        return {"error": f"Data processing error with Polars: {str(pe)}"}
    except Exception as e:
        await ctx.error(
            f"An unexpected error occurred during aggregation: {e}", exc_info=True
        )
        return {"error": f"An unexpected server error occurred: {str(e)}"}
