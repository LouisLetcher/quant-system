from __future__ import annotations

import os
import webbrowser

from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_portfolio_config
from src.portfolio.metrics_processor import ensure_all_metrics_exist, generate_log_summary
from src.portfolio.timeframe_optimizer import backtest_all_strategies_all_timeframes
from src.reports.report_generator import ReportGenerator
from src.utils.logger import Logger, get_logger

# Initialize logger
logger = get_logger(__name__)


def backtest_portfolio_optimal(args):
    """Find optimal strategy and interval for each asset in a portfolio."""
    try:
        print("Starting portfolio optimization")

        # Setup logging if requested, but don't capture stdout/stderr
        log_file = None
        if hasattr(args, "log") and args.log:
            # Initialize logger if needed
            Logger.initialize()

            # Get command name
            command = args.command if hasattr(args, "command") else "unknown"

            # Setup CLI logging without capturing stdout/stderr
            log_file = Logger.setup_cli_logging(command)

            print(f"📝 Logging enabled. Output will be saved to: {log_file}")
            logger.info("Portfolio optimization started")

        print(f"Getting portfolio config for '{args.name}'")
        logger.info(f"Getting portfolio config for '{args.name}'")
        portfolio_config = get_portfolio_config(args.name)
        if not portfolio_config:
            logger.error(f"Portfolio '{args.name}' not found in assets_config.json")
            print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
            return {}

        print("Getting intervals")
        logger.info(f"Using intervals: {args.intervals}")
        intervals = args.intervals if args.intervals else ["1d"]

        print(
            f"Finding optimal strategy-interval combinations for portfolio '{args.name}'"
        )
        logger.info(
            f"Finding optimal strategy-interval combinations for portfolio '{args.name}'"
        )

        # Initialize dictionaries to store results
        best_combinations = {}
        all_results = {}

        print("Getting available strategies")
        # Restore the original strategy factory call
        strategies = StrategyFactory.get_available_strategies()
        logger.info(f"Testing {len(strategies)} strategies")

        # Process each asset
        assets = portfolio_config.get("assets", [])
        print(f"Processing {len(assets)} assets")
        logger.info(f"Processing {len(assets)} assets")

        for asset_config in assets:
            ticker = asset_config["ticker"]
            print(f"Processing asset: {ticker}")
            logger.info(f"Processing asset: {ticker}")

            # Call the backtest function for this asset
            asset_results = backtest_all_strategies_all_timeframes(
                ticker=ticker,
                asset_config=asset_config,
                strategies=strategies,
                intervals=intervals,
                period=args.period,
                metric=args.metric,
                start_date=getattr(args, "start_date", None),
                end_date=getattr(args, "end_date", None),
                plot=getattr(args, "plot", False),
                resample=getattr(args, "resample", None),
            )

            # Ensure all metrics exist in the best combination
            best_combination = asset_results["best_combination"]
            best_combination = ensure_all_metrics_exist(best_combination)

            # Update the asset results with the enhanced best combination
            asset_results["best_combination"] = best_combination

            # Also ensure all metrics exist in each timeframe result
            for strategy in asset_results["all_results"].get("strategies", []):
                for timeframe in strategy.get("timeframes", []):
                    ensure_all_metrics_exist(timeframe)

            best_combinations[ticker] = best_combination
            all_results[ticker] = asset_results["all_results"]

        # Generate report only if not plotting individual results
        if not getattr(args, "plot", False):
            print("Generating report")
            logger.info("Generating portfolio optimization report")
            report_data = {
                "portfolio": args.name,
                "description": portfolio_config.get("description", ""),
                "best_combinations": best_combinations,
                "all_results": all_results,
                "metric": args.metric,
                "intervals": intervals,
                "strategies": strategies,
                "is_portfolio_optimal": True,
            }

            output_path = f"reports_output/portfolio_optimal_{args.name}.html"

            # Use the detailed portfolio report template instead
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            generator = ReportGenerator()

            # Use generate_detailed_portfolio_report instead of generate_report
            generator.generate_detailed_portfolio_report(report_data, output_path)

            print(f"📄 Portfolio Optimization Report saved to: {output_path}")
            logger.info(f"Portfolio Optimization Report saved to: {output_path}")

            # Open the report in the browser if requested
            if args.open_browser:
                logger.info(
                    f"Opening report in browser: {os.path.abspath(output_path)}"
                )
                webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)

        # Generate log summary
        generate_log_summary(args.name, best_combinations, args.metric)

        print("Portfolio optimization completed successfully")
        logger.info("Portfolio optimization completed successfully")
        return best_combinations

    except RecursionError as e:
        print(f"RecursionError: {e}")
        if "logger" in globals():
            logger.error(f"RecursionError: {e}")
        import traceback

        trace = traceback.format_exc()
        print(trace)
        if "logger" in globals():
            logger.error(trace)
        return {}
    except Exception as e:
        print(f"Error in backtest_portfolio_optimal: {e}")
        if "logger" in globals():
            logger.error(f"Error in backtest_portfolio_optimal: {e}")
        import traceback

        trace = traceback.format_exc()
        print(trace)
        if "logger" in globals():
            logger.error(trace)
        return {}
