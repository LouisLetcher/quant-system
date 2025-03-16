from __future__ import annotations

import argparse

from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import load_assets_config


def list_portfolios():
    """List all available portfolios from assets_config.json"""
    assets_config = load_assets_config()
    portfolios = assets_config.get("portfolios", {})

    if not portfolios:
        print("No portfolios found in config/assets_config.json")
        return

    print("\n📂 Available Portfolios:")
    print("-" * 80)
    for name, config in portfolios.items():
        assets = ", ".join([asset["ticker"] for asset in config.get("assets", [])])
        print(f"📊 {name}: {config.get('description', 'No description')}")
        print(f"   🔸 Assets: {assets}")
        print("-" * 80)


def list_strategies():
    """List all available trading strategies"""
    strategies = StrategyFactory.get_available_strategies()

    print("\n📈 Available Trading Strategies:")
    print("-" * 80)
    for strategy_name in strategies:
        print(f"🔹 {strategy_name}")
    print("-" * 80)


def register_commands(subparsers):
    """Register utility commands with the CLI parser."""
    # Add --log option to the parent parser so it's available to all commands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging of command output"
    )
    
    # List portfolios command
    list_portfolios_parser = subparsers.add_parser(
        "list-portfolios", help="List available portfolios", parents=[parent_parser]
    )
    list_portfolios_parser.set_defaults(func=lambda args: list_portfolios(args))

    # List strategies command
    list_strategies_parser = subparsers.add_parser(
        "list-strategies", help="List available trading strategies", parents=[parent_parser]
    )
    list_strategies_parser.set_defaults(func=lambda args: list_strategies(args))
