"""Configuration loader for CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from src.utils.config_manager import ConfigManager


def load_assets_config():
    """Load the assets configuration from config/assets_config.json."""
    config_path = Path("config") / "assets_config.json"
    if config_path.exists():
        with config_path.open() as f:
            return json.load(f)
    return {"portfolios": {}}


def is_portfolio(ticker):
    """Check if the given ticker is a portfolio name in assets_config.json."""
    assets_config = load_assets_config()
    return ticker in assets_config.get("portfolios", {})


def get_portfolio_config(portfolio_name):
    """Get configuration for a specific portfolio."""
    assets_config = load_assets_config()
    return assets_config.get("portfolios", {}).get(portfolio_name, None)


def get_asset_config(ticker):
    """Get asset-specific config if available in any portfolio."""
    assets_config = load_assets_config()

    # Search through all portfolios for the ticker
    for portfolio in assets_config.get("portfolios", {}).values():
        for asset in portfolio.get("assets", []):
            if asset["ticker"] == ticker:
                return asset

    return None


def get_default_parameters():
    """Get default backtest parameters from config."""
    config = ConfigManager()
    return {
        "commission": config.get("backtest.default_commission", 0.001),
        "initial_capital": config.get("backtest.initial_capital", 10000),
        "period": config.get("backtest.default_period", "max"),
        "intervals": config.get("backtest.default_intervals", ["1d", "1wk"]),
        "iterations": config.get("optimizer.iterations", 50),
    }
