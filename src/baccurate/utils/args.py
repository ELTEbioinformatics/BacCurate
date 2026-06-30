"""Shared argparse helper for standardizer CLIs."""

import argparse


def create_arg_parser(
    description: str = None, default_config_path: str = None
) -> argparse.ArgumentParser:
    """Build the standard standardizer argparse parser (input/output/config/log-level)."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("input_file", help="Path to the input file.")
    parser.add_argument("--output-dir", "-o", required=True, help="Directory to save output files.")
    parser.add_argument(
        "--config", "-c", default=default_config_path, help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set the logging level (default: INFO).",
    )
    return parser
