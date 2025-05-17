from pathlib import Path
from typing import Any

import polars as pl
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Define the project root directory.
# This file (config.py) is at: PROJECT_ROOT/src/upphandlat_mcp/core/config.py
# So, four .parent calls from __file__ will get us to PROJECT_ROOT
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

POLARS_DTYPE_MAP = {
    "Utf8": pl.Utf8,
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "Date": pl.Date,
    "Datetime": pl.Datetime,
    "Duration": pl.Duration,
    "Time": pl.Time,
    "Categorical": pl.Categorical,
}


class ReadCsvOptions(BaseModel):
    separator: str = Field(",", description="Separator character for the CSV file.")
    truncate_ragged_lines: bool = Field(
        False, description="Truncate ragged lines instead of erroring."
    )
    schema_overrides: dict[str, str] | None = Field(
        None,
        description="Dictionary mapping column names to their Polars dtype strings (e.g., {'col_a': 'Utf8'}).",
    )
    has_header: bool = Field(True, description="Whether the CSV file has a header row.")
    encoding: str = Field(
        "utf8", description="Encoding of the CSV file (e.g., 'utf8', 'latin1')."
    )
    null_values: list[str] | str | None = Field(
        None, description="Values to interpret as null."
    )
    infer_schema_length: int | None = Field(
        None,
        description="Number of rows to infer schema from. If None, Polars default (usually 100) is used.",
    )

    def to_polars_args(self) -> dict[str, Any]:
        args = self.model_dump(exclude_none=True, exclude={"schema_overrides"})
        if self.schema_overrides:
            polars_schema = {}
            for col, dtype_str in self.schema_overrides.items():
                dtype = POLARS_DTYPE_MAP.get(dtype_str)
                if dtype is None:
                    raise ValueError(
                        f"Unsupported dtype string '{dtype_str}' in schema_overrides for column '{col}'. "
                        f"Available: {list(POLARS_DTYPE_MAP.keys())}"
                    )
                polars_schema[col] = dtype
            args["schema_overrides"] = polars_schema
        return args


class CsvSource(BaseModel):
    name: str = Field(
        ..., description="Unique identifier name for this CSV data source."
    )
    url: str = Field(..., description="URL from which to download the CSV file.")
    description: str | None = Field(
        None, description="Optional description of the data source."
    )
    read_csv_options: ReadCsvOptions = Field(
        default_factory=lambda: ReadCsvOptions(),  # type: ignore[call-arg] # Pylance false positive
        description="Polars read_csv options for this source.",
    )

    @field_validator("name")
    @classmethod
    def name_must_be_valid_identifier(cls, v: str) -> str:
        if not v.isidentifier():
            raise ValueError(
                f"Source name '{v}' is not a valid Python identifier. Use letters, numbers, and underscores, not starting with a number."
            )
        return v


class CsvSourcesConfig(BaseModel):
    sources: list[CsvSource]


class Settings(BaseSettings):
    """
    Application settings.
    """

    # Use the PROJECT_ROOT to define an absolute path to csv_sources.yaml
    CSV_SOURCES_CONFIG_PATH: Path = PROJECT_ROOT / "csv_sources.yaml"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
