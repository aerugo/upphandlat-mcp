#!/usr/bin/env python3
"""
Script to download and validate all CSV sources defined in csv_sources.yaml.
"""
import logging
import sys
from io import BytesIO
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import polars as pl
import yaml
from upphandlat_mcp.core.config import CsvSourcesConfig, settings


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Load configuration of CSV sources
    config_path = settings.CSV_SOURCES_CONFIG_PATH
    logger.info(f"Loading CSV sources configuration from {config_path}")
    try:
        with open(config_path, "r") as f:
            cfg_dict = yaml.safe_load(f)
        config = CsvSourcesConfig(**cfg_dict)
    except Exception as e:
        logger.error(f"Failed to load CSV sources config: {e}")
        sys.exit(1)

    results = []
    # Iterate through each defined CSV source
    for source in config.sources:
        logger.info(f"Processing source: {source.name} ({source.url})")

        # Attempt to download the CSV
        try:
            req = Request(
                source.url, headers={"User-Agent": "upphandlat-mcp-validator/0.1"}
            )
            with urlopen(req) as resp:
                data = resp.read()
        except HTTPError as e:
            err_msg = f"HTTP error {e.code}: {e.reason}"
            logger.error(f"[{source.name}] Download error: {err_msg}")
            results.append(
                {"name": source.name, "status": "download_error", "error": err_msg}
            )
            continue
        except URLError as e:
            err_msg = str(e.reason)
            logger.error(f"[{source.name}] Download error: {err_msg}")
            results.append(
                {"name": source.name, "status": "download_error", "error": err_msg}
            )
            continue

        # Attempt to read and validate the CSV using Polars
        try:
            args = source.read_csv_options.to_polars_args()
            df = pl.read_csv(BytesIO(data), **args)
            logger.info(
                f"[{source.name}] Successfully read CSV: {df.height} rows, {len(df.columns)} columns"
            )
            results.append(
                {
                    "name": source.name,
                    "status": "success",
                    "rows": df.height,
                    "columns": df.columns,
                }
            )
        except Exception as e:
            err_msg = str(e)
            logger.error(f"[{source.name}] Validation error: {err_msg}")
            results.append(
                {"name": source.name, "status": "validation_error", "error": err_msg}
            )
            continue

    # Summary of results
    successes = [r for r in results if r["status"] == "success"]
    failures = [r for r in results if r["status"] != "success"]
    logger.info(f"Completed: {len(successes)} succeeded, {len(failures)} failed.")
    if failures:
        for r in failures:
            logger.error(f"Source {r['name']} error: {r.get('error')}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
