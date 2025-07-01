from __future__ import annotations

import argparse
import codecs
import sys
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import unified CLI
from src.cli.unified_cli import main as unified_main

# Set console output encoding to UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
if sys.stderr.encoding != "utf-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main():
    """
    Main entry point - now uses the unified CLI system.
    
    The old command structure has been replaced with a unified architecture
    that eliminates code duplication and provides better functionality.
    """
    print("🚀 Quant Trading System - Unified Architecture")
    print("For legacy commands, use the individual command modules.")
    print("For new unified commands, use: python -m src.cli.unified_cli")
    print("\nRunning unified CLI...")
    print("=" * 50)
    
    # Redirect to unified CLI
    unified_main()


if __name__ == "__main__":
    main()
