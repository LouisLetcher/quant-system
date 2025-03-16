from __future__ import annotations

import os
import webbrowser

from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_default_parameters, get_portfolio_config
from src.portfolio.backtest_runner import backtest_all_strategies
from src.reports.report_generator import ReportGenerator
from src.utils.logger import get_logger, setup_command_logging

# Initialize logger
logger = get_logger(__name__)


def backtest_portfolio(args):
    """Run a backtest of all assets in a portfolio with all strategies."""
    # Setup logging if requested
    log_file = setup_command_logging(args)

    logger.info(f"Starting portfolio backtest for '{args.name}'")
    logger.info(
        f"Parameters: period={args.period}, metric={args.metric}, plot={args.plot}, resample={args.resample}"
    )

    portfolio_config = get_portfolio_config(args.name)

    if not portfolio_config:
        logger.error(f"Portfolio '{args.name}' not found in assets_config.json")
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}

    # Get default parameters from config
    defaults = get_default_parameters()
    logger.debug(f"Default parameters: {defaults}")

    print(f"Testing all strategies on portfolio '{args.name}'")
    print(
        f"Portfolio description: {portfolio_config.get('description', 'No description')}"
    )

    logger.info(
        f"Portfolio description: {portfolio_config.get('description', 'No description')}"
    )

    # Get all available strategies
    strategies = StrategyFactory.get_available_strategies()
    assets = portfolio_config.get("assets", [])
    logger.info(f"Testing {len(strategies)} strategies on {len(assets)} assets")
    print(f"Testing {len(strategies)} strategies on {len(assets)} assets")

    results = {}
    for asset_config in assets:
        ticker = asset_config["ticker"]
        asset_period = asset_config.get("period", args.period)
        commission = asset_config.get("commission", defaults["commission"])
        initial_capital = asset_config.get(
            "initial_capital", defaults["initial_capital"]
        )

        logger.info(
            f"Testing {ticker} with {initial_capital} initial capital, commission={commission}, period={asset_period}"
        )
        print(f"\n🔍 Testing {ticker} with {initial_capital} initial capital")

        results[ticker] = backtest_all_strategies(
            ticker=ticker,
            period=asset_period,
            metric=args.metric,
            commission=commission,
            initial_capital=initial_capital,
            plot=args.plot,
            resample=args.resample,
        )

    # If not plotting individual strategies, generate a portfolio report
    if not args.plot:
        output_path = f"reports_output/portfolio_{args.name}_{args.metric}.html"
        generator = ReportGenerator()

        # Create portfolio results object
        portfolio_results = {
            "portfolio": args.name,
            "description": portfolio_config.get("description", ""),
            "assets": results,
            "metric": args.metric,
        }

        logger.info(f"Generating detailed portfolio report to {output_path}")
        # Generate the detailed report with equity curves and trade tables
        generator.generate_detailed_portfolio_report(portfolio_results, output_path)

        print(f"📄 Detailed Portfolio Report saved to {output_path}")
        logger.info(f"Detailed Portfolio Report saved to {output_path}")

        # Open the report in the browser if requested
        if args.open_browser:
            logger.info(f"Opening report in browser: {os.path.abspath(output_path)}")
            webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)

    logger.info("Portfolio backtest completed successfully")
    return results
