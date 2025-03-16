from __future__ import annotations

# Import the separated modules
from src.portfolio.portfolio_backtest import backtest_portfolio
from src.portfolio.portfolio_optimizer import backtest_portfolio_optimal
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def register_commands(subparsers):
    """Register portfolio commands with the CLI parser"""
    # Portfolio command
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Backtest all assets in a portfolio with all strategies"
    )
    portfolio_parser.add_argument(
        "--name", type=str, required=True, help="Portfolio name from assets_config.json"
    )
    portfolio_parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="Default data period (can be overridden by portfolio settings)",
    )
    portfolio_parser.add_argument(
        "--metric",
        type=str,
        default="profit_factor",
        help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)",
    )
    portfolio_parser.add_argument(
        "--plot",
        action="store_true",
        help="Use backtesting.py's plot() method to display results in browser",
    )
    portfolio_parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Resample period for plotting (e.g., '1D', '4H', '1W')",
    )
    portfolio_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the generated report in a browser",
    )
    # Add the log option to portfolio command
    portfolio_parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging of command output"
    )
    portfolio_parser.set_defaults(func=backtest_portfolio)

    # Portfolio optimization command
    portfolio_optimal_parser = subparsers.add_parser(
        "portfolio-optimal",
        help="Find optimal strategy/timeframe combinations for a portfolio",
    )
    portfolio_optimal_parser.add_argument(
        "--name", required=True, help="Portfolio name from assets_config.json"
    )
    portfolio_optimal_parser.add_argument(
        "--intervals",
        nargs="+",
        default=["1d"],
        help="Intervals to test (e.g., 1d 1wk 1mo)",
    )
    portfolio_optimal_parser.add_argument(
        "--period", default="max", help="Data period to fetch"
    )
    portfolio_optimal_parser.add_argument(
        "--metric",
        default="sharpe",
        help="Metric to optimize for (sharpe, return, profit_factor)",
    )
    portfolio_optimal_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open report in browser after completion",
    )
    # Add the log option
    portfolio_optimal_parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging of command output"
    )
    # Add start_date and end_date parameters
    portfolio_optimal_parser.add_argument(
        "--start-date", dest="start_date", help="Start date for backtest (YYYY-MM-DD)"
    )
    portfolio_optimal_parser.add_argument(
        "--end-date", dest="end_date", help="End date for backtest (YYYY-MM-DD)"
    )
    # Add plot and resample parameters
    portfolio_optimal_parser.add_argument(
        "--plot", action="store_true", help="Plot the best strategy for each asset"
    )
    portfolio_optimal_parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Resample period for plotting (e.g., '1D', '4H', '1W')",
    )
    # Add require_complete_history parameter
    portfolio_optimal_parser.add_argument(
        "--require-complete-history",
        dest="require_complete_history",
        type=bool,
        default=None,
        help="Require complete history for backtest",
    )
    portfolio_optimal_parser.set_defaults(func=backtest_portfolio_optimal)
