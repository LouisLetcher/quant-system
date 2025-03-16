from __future__ import annotations

import argparse
import codecs
import sys

from src.cli.commands import backtest_commands, optimizer_commands, portfolio_commands, utility_commands

# Set console output encoding to UTF-8
if sys.stdout.encoding != "utf-8":
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
if sys.stderr.encoding != "utf-8":
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


def main():
    parser = argparse.ArgumentParser()

    # Create subparsers for commands
    subparsers = parser.add_subparsers(dest="command")

    # Register all commands
    backtest_commands.register_commands(subparsers)
    portfolio_commands.register_commands(subparsers)
    optimizer_commands.register_commands(subparsers)
    utility_commands.register_commands(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Execute the corresponding function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
