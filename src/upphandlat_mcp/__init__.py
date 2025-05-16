# src/upphandlat_mcp/__init__.py
import asyncio
import logging
import sys

# Configure basic logging for the CLI entry point.
# This helps capture errors that occur early in the script's execution.
# MCP server tools often expect logs on stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Get a logger for this __init__ module or the root of the package.
logger = logging.getLogger(__name__)


def main() -> None:
    """
    Main entry point for the upphandlat-mcp command-line interface.
    This function initializes and runs the MCP server.
    """
    logger.info("Executing upphandlat-mcp CLI entry point...")
    try:
        # Import run_mcp here to ensure all module-level initializations
        # (like config loading, other loggers) are handled correctly
        # and to avoid potential circular dependencies if __init__.py
        # were imported by other modules at load time.
        from upphandlat_mcp.server import run_mcp

        # Run the asynchronous run_mcp function using asyncio.run()
        asyncio.run(run_mcp())
        logger.info("upphandlat-mcp CLI execution finished successfully.")

    except ImportError as e:
        # This can happen if there's an issue with the Python environment,
        # installation, or if a critical module is missing.
        logger.critical(f"Failed to import server components: {e}", exc_info=True)
        sys.exit(1)  # Exit with a non-zero status code to indicate failure

    except Exception as e:
        # Catch any other unexpected exceptions during the CLI execution.
        logger.critical(f"An unexpected error occurred in the CLI: {e}", exc_info=True)
        sys.exit(1)  # Exit with a non-zero status code


if __name__ == "__main__":
    # This block allows the package to be run directly using `python -m upphandlat_mcp`,
    # which can be useful for development or testing.
    # The `upphandlat-mcp` script defined in pyproject.toml will call main() directly.
    main()
