import argparse
from src.cli.commands import backtest_commands, portfolio_commands, optimizer_commands, utility_commands

def main():
    """Main CLI entry point for the quant system."""
    parser = argparse.ArgumentParser(description="Quant System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Register command groups
    backtest_commands.register_commands(subparsers)
    portfolio_commands.register_commands(subparsers)
    optimizer_commands.register_commands(subparsers)
    utility_commands.register_commands(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
