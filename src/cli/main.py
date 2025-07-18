"""Main entry point for the CLI system."""

from __future__ import annotations

import codecs
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import unified CLI
from src.cli.unified_cli import main as unified_main

# Set console output encoding to UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
if sys.stderr.encoding != "utf-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main() -> None:
    """Redirect to unified CLI system."""
    unified_main()


if __name__ == "__main__":
    main()
