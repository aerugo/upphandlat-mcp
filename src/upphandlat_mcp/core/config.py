from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    CSV_FILE_PATH: Path = Path(
        "data/your_data.csv"
    )  # Default if .env not found or var not set

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
