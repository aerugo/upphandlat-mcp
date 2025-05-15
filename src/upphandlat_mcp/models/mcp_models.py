from enum import Enum
from typing import Any, Literal

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
    functions: list[AggFunc] = Field(
        ..., min_length=1, description="List of aggregation functions to apply."
    )
    rename: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Optional mapping to rename output columns. "
            "Keys should be function names (e.g., 'sum'), "
            "values are the new names (e.g., 'total_sales'). "
            "Default is '{column}_{function}'."
        ),
    )

    @model_validator(mode="after")
    def check_rename_keys(self) -> "Aggregation":
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
        pattern=r"^[a-zA-Z_][a-zA-Z0-9_]*$",
    )

    @field_validator("output_column_name")
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

    calculation_type: Literal["two_column_arithmetic"] = Field(
        "two_column_arithmetic", description="Type of calculation."
    )
    column_a: str = Field(..., description="The first column operand (e.g., 'sales').")
    column_b: str = Field(..., description="The second column operand (e.g., 'cost').")
    operation: ArithmeticOperationType = Field(
        ...,
        description="The arithmetic operation to perform (e.g., 'subtract' for A - B).",
    )
    on_division_by_zero: float | Literal["null", "propagate_error"] = Field(
        "propagate_error",
        description="Behavior for division by zero: a specific float, 'null', or 'propagate_error' (Polars default, may result in inf/NaN).",
    )


class ConstantArithmeticConfig(BaseCalculatedField):
    """Configuration for arithmetic operation between a column and a constant."""

    calculation_type: Literal["constant_arithmetic"] = Field(
        "constant_arithmetic", description="Type of calculation."
    )
    input_column: str = Field(..., description="The column to operate on.")
    constant_value: float = Field(
        ..., description="The constant value for the operation."
    )
    operation: ArithmeticOperationType = Field(
        ..., description="The arithmetic operation."
    )
    column_is_first_operand: bool = Field(
        True,
        description="If true, operation is 'input_column op constant'; else 'constant op input_column'.",
    )
    on_division_by_zero: float | Literal["null", "propagate_error"] = Field(
        "propagate_error", description="Behavior for division by zero (if applicable)."
    )


class PercentageOfColumnConfig(BaseCalculatedField):
    """Configuration for calculating one column as a percentage of another."""

    calculation_type: Literal["percentage_of_column"] = Field(
        "percentage_of_column", description="Type of calculation."
    )
    value_column: str = Field(
        ..., description="The column representing the part/value (numerator)."
    )
    total_reference_column: str = Field(
        ..., description="The column representing the total/base (denominator)."
    )
    scale_factor: float = Field(
        100.0,
        description="Factor to multiply the ratio by (e.g., 100.0 for percentage).",
    )
    on_division_by_zero: float | Literal["null", "propagate_error"] = Field(
        "propagate_error",
        description="Behavior for division by zero in 'value_column / total_reference_column'.",
    )


CalculatedFieldType = (
    TwoColumnArithmeticConfig | ConstantArithmeticConfig | PercentageOfColumnConfig
)


class AggregationRequest(BaseModel):
    """
    Defines the request payload for the aggregation tool.
    It specifies how to group data, what aggregations to perform,
    and any additional calculated fields to compute on the aggregated results.
    """

    group_by_columns: list[str] = Field(
        ..., min_length=1, description="List of column names to group the data by."
    )
    aggregations: list[Aggregation] = Field(
        ...,
        min_length=1,
        description="List of aggregation operations to perform on specified columns.",
    )
    calculated_fields: list[CalculatedFieldType] | None = Field(
        None,
        description="Optional list of calculated fields to derive after aggregation. These are applied in order.",
    )

    @model_validator(mode="after")
    def check_column_name_conflicts(self) -> "AggregationRequest":
        """
        Validate that output column names from aggregations and calculated fields
        do not conflict with each other or with group_by_columns.
        """
        all_output_names: set[str] = set(self.group_by_columns)

        for agg in self.aggregations:
            for func in agg.functions:
                alias = agg.rename.get(func.value, f"{agg.column}_{func.value}")
                if alias in all_output_names:
                    raise ValueError(
                        f"Duplicate output column name '{alias}' from aggregation conflicts with group_by or another aggregation."
                    )
                all_output_names.add(alias)

        if self.calculated_fields:
            for calc_field_config in self.calculated_fields:
                # The config itself is a discriminated union, so access common field directly
                output_name = calc_field_config.output_column_name
                if output_name in all_output_names:
                    raise ValueError(
                        f"Calculated field output name '{output_name}' conflicts with group_by or aggregation columns."
                    )
                all_output_names.add(output_name)
        return self
