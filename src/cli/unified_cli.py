"""
Unified CLI - Restructured command-line interface using unified components.
Removes duplication and provides comprehensive functionality.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

from src.core import (
    PortfolioManager,
    UnifiedBacktestEngine,
    UnifiedCacheManager,
    UnifiedDataManager,
)
from src.core.backtest_engine import BacktestConfig, BacktestResult
from src.core.strategy import StrategyFactory, list_available_strategies


def get_earliest_data_date(portfolio_path: str) -> datetime:
    """
    Get the earliest data date based on portfolio configuration.

    Args:
        portfolio_path: Path to portfolio configuration file

    Returns:
        Earliest reasonable data date
    """
    # Default fallback date
    default_date = datetime(2015, 1, 1)

    try:
        if not portfolio_path:
            return default_date

        portfolio_file = Path(portfolio_path)
        if not portfolio_file.exists():
            return default_date

        with portfolio_file.open("r") as f:
            portfolio_config = json.load(f)

        # Get the first portfolio config (since file contains nested portfolio)
        portfolio_key = list(portfolio_config.keys())[0]
        config = portfolio_config[portfolio_key]

        # Check metadata for best_data_coverage
        metadata = config.get("metadata", {})
        data_coverage = metadata.get("best_data_coverage", "")

        if data_coverage and "-present" in data_coverage:
            year_str = data_coverage.split("-")[0]
            try:
                year = int(year_str)
                return datetime(year, 1, 1)
            except ValueError:
                pass

        # Asset type specific defaults
        asset_type = config.get("asset_type", "")
        if asset_type == "crypto":
            return datetime(2017, 1, 1)  # Crypto data typically starts around 2017
        if asset_type == "stocks":
            return datetime(1990, 1, 1)  # Stock data goes back further
        if asset_type == "forex":
            return datetime(2000, 1, 1)  # Forex data availability
        if asset_type == "commodities":
            return datetime(2006, 1, 1)  # Commodity data availability
        if asset_type == "bonds":
            return datetime(2003, 1, 1)  # Bond data availability

    except Exception as e:
        logging.warning(
            "Could not determine earliest data date from portfolio config: %s", e
        )

    return default_date


def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser."""
    parser = argparse.ArgumentParser(
        description="Unified Quant Trading System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Data commands
    add_data_commands(subparsers)

    # Strategy commands
    add_strategy_commands(subparsers)

    # Backtest commands
    add_backtest_commands(subparsers)

    # Portfolio commands
    add_portfolio_commands(subparsers)

    # Optimization commands
    add_optimization_commands(subparsers)

    # Analysis commands
    add_analysis_commands(subparsers)

    # Cache commands
    add_cache_commands(subparsers)

    # Reports commands
    add_reports_commands(subparsers)

    return parser


def add_data_commands(subparsers):
    """Add data management commands."""
    data_parser = subparsers.add_parser("data", help="Data management commands")
    data_subparsers = data_parser.add_subparsers(dest="data_command")

    # Download command
    download_parser = data_subparsers.add_parser(
        "download", help="Download market data"
    )
    download_parser.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols to download"
    )
    download_parser.add_argument(
        "--start-date", required=True, help="Start date (YYYY-MM-DD)"
    )
    download_parser.add_argument(
        "--end-date", required=True, help="End date (YYYY-MM-DD)"
    )
    download_parser.add_argument("--interval", default="1d", help="Data interval")
    download_parser.add_argument(
        "--asset-type",
        choices=["stocks", "crypto", "forex", "commodities"],
        help="Asset type hint",
    )
    download_parser.add_argument(
        "--futures", action="store_true", help="Download crypto futures data"
    )
    download_parser.add_argument(
        "--force", action="store_true", help="Force download even if cached"
    )

    # Sources command
    data_subparsers.add_parser("sources", help="Show available data sources")

    # Symbols command
    symbols_parser = data_subparsers.add_parser(
        "symbols", help="List available symbols"
    )
    symbols_parser.add_argument(
        "--asset-type",
        choices=["stocks", "crypto", "forex"],
        help="Filter by asset type",
    )
    symbols_parser.add_argument("--source", help="Specific data source")


def add_strategy_commands(subparsers):
    """Add strategy management commands."""
    strategy_parser = subparsers.add_parser(
        "strategy", help="Strategy management commands"
    )
    strategy_subparsers = strategy_parser.add_subparsers(dest="strategy_command")

    # List strategies
    list_parser = strategy_subparsers.add_parser(
        "list", help="List available strategies"
    )
    list_parser.add_argument(
        "--type",
        choices=["builtin", "external", "all"],
        default="all",
        help="Filter by strategy type",
    )

    # Strategy info
    info_parser = strategy_subparsers.add_parser(
        "info", help="Get strategy information"
    )
    info_parser.add_argument("name", help="Strategy name")

    # Test strategy
    test_parser = strategy_subparsers.add_parser(
        "test", help="Test strategy with sample data"
    )
    test_parser.add_argument("name", help="Strategy name")
    test_parser.add_argument("--symbol", default="AAPL", help="Symbol for testing")
    test_parser.add_argument("--start-date", default="2023-01-01", help="Start date")
    test_parser.add_argument("--end-date", default="2023-12-31", help="End date")
    test_parser.add_argument("--parameters", help="JSON string of strategy parameters")


def add_backtest_commands(subparsers):
    """Add backtesting commands."""
    backtest_parser = subparsers.add_parser("backtest", help="Backtesting commands")
    backtest_subparsers = backtest_parser.add_subparsers(dest="backtest_command")

    # Single backtest
    single_parser = backtest_subparsers.add_parser("single", help="Run single backtest")
    single_parser.add_argument("--symbol", required=True, help="Symbol to backtest")
    single_parser.add_argument("--strategy", required=True, help="Strategy to use")
    single_parser.add_argument("--start-date", required=True, help="Start date")
    single_parser.add_argument("--end-date", required=True, help="End date")
    single_parser.add_argument("--interval", default="1d", help="Data interval")
    single_parser.add_argument(
        "--capital", type=float, default=10000, help="Initial capital"
    )
    single_parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    single_parser.add_argument(
        "--parameters", help="JSON string of strategy parameters"
    )
    single_parser.add_argument(
        "--futures", action="store_true", help="Use futures mode"
    )
    single_parser.add_argument(
        "--no-cache", action="store_true", help="Disable caching"
    )

    # Batch backtest
    batch_parser = backtest_subparsers.add_parser("batch", help="Run batch backtests")
    batch_parser.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols to backtest"
    )
    batch_parser.add_argument(
        "--strategies", nargs="+", required=True, help="Strategies to use"
    )
    batch_parser.add_argument("--start-date", required=True, help="Start date")
    batch_parser.add_argument("--end-date", required=True, help="End date")
    batch_parser.add_argument("--interval", default="1d", help="Data interval")
    batch_parser.add_argument(
        "--capital", type=float, default=10000, help="Initial capital"
    )
    batch_parser.add_argument(
        "--commission", type=float, default=0.001, help="Commission rate"
    )
    batch_parser.add_argument(
        "--max-workers", type=int, help="Maximum parallel workers"
    )
    batch_parser.add_argument(
        "--memory-limit", type=float, default=8.0, help="Memory limit in GB"
    )
    batch_parser.add_argument("--asset-type", help="Asset type hint")
    batch_parser.add_argument("--futures", action="store_true", help="Use futures mode")
    batch_parser.add_argument(
        "--save-trades", action="store_true", help="Save individual trades"
    )
    batch_parser.add_argument(
        "--save-equity", action="store_true", help="Save equity curves"
    )
    batch_parser.add_argument("--output", help="Output file path")


def add_portfolio_commands(subparsers):
    """Add portfolio management commands."""
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Portfolio management commands"
    )
    portfolio_subparsers = portfolio_parser.add_subparsers(dest="portfolio_command")

    # Backtest portfolio
    backtest_parser = portfolio_subparsers.add_parser(
        "backtest", help="Backtest portfolio"
    )
    backtest_parser.add_argument(
        "--symbols", nargs="+", required=True, help="Portfolio symbols"
    )
    backtest_parser.add_argument("--strategy", required=True, help="Portfolio strategy")
    backtest_parser.add_argument("--start-date", required=True, help="Start date")
    backtest_parser.add_argument("--end-date", required=True, help="End date")
    backtest_parser.add_argument("--weights", help="JSON string of symbol weights")
    backtest_parser.add_argument("--interval", default="1d", help="Data interval")
    backtest_parser.add_argument(
        "--capital", type=float, default=10000, help="Initial capital"
    )

    # Test portfolio with all strategies
    test_all_parser = portfolio_subparsers.add_parser(
        "test-all", help="Test portfolio with all available strategies and timeframes"
    )
    test_all_parser.add_argument(
        "--portfolio", required=True, help="JSON file with portfolio definition"
    )
    test_all_parser.add_argument(
        "--start-date", help="Start date (defaults to earliest available)"
    )
    test_all_parser.add_argument("--end-date", help="End date (defaults to today)")
    test_all_parser.add_argument(
        "--period",
        choices=["max", "1y", "2y", "5y", "10y"],
        default="max",
        help="Time period",
    )
    test_all_parser.add_argument(
        "--metric",
        choices=[
            "profit_factor",
            "sharpe_ratio",
            "sortino_ratio",
            "total_return",
            "max_drawdown",
        ],
        default="sharpe_ratio",
        help="Primary metric for ranking",
    )
    test_all_parser.add_argument(
        "--timeframes",
        nargs="+",
        choices=["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1wk"],
        default=["1d"],
        help="Timeframes to test (default: 1d)",
    )
    test_all_parser.add_argument(
        "--test-timeframes",
        action="store_true",
        help="Test all timeframes to find optimal timeframe per asset",
    )
    test_all_parser.add_argument(
        "--open-browser", action="store_true", help="Open results in browser"
    )

    # Compare portfolios
    compare_parser = portfolio_subparsers.add_parser(
        "compare", help="Compare multiple portfolios"
    )
    compare_parser.add_argument(
        "--portfolios", required=True, help="JSON file with portfolio definitions"
    )
    compare_parser.add_argument("--start-date", required=True, help="Start date")
    compare_parser.add_argument("--end-date", required=True, help="End date")
    compare_parser.add_argument("--output", help="Output file for results")

    # Investment plan
    plan_parser = portfolio_subparsers.add_parser(
        "plan", help="Generate investment plan"
    )
    plan_parser.add_argument(
        "--portfolios", required=True, help="JSON file with portfolio results"
    )
    plan_parser.add_argument(
        "--capital", type=float, required=True, help="Total capital to allocate"
    )
    plan_parser.add_argument(
        "--risk-tolerance",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Risk tolerance",
    )
    plan_parser.add_argument("--output", help="Output file for investment plan")


def add_optimization_commands(subparsers):
    """Add optimization commands."""
    opt_parser = subparsers.add_parser(
        "optimize", help="Strategy optimization commands"
    )
    opt_subparsers = opt_parser.add_subparsers(dest="optimize_command")

    # Single optimization
    single_parser = opt_subparsers.add_parser("single", help="Optimize single strategy")
    single_parser.add_argument("--symbol", required=True, help="Symbol to optimize")
    single_parser.add_argument("--strategy", required=True, help="Strategy to optimize")
    single_parser.add_argument("--start-date", required=True, help="Start date")
    single_parser.add_argument("--end-date", required=True, help="End date")
    single_parser.add_argument(
        "--parameters", required=True, help="JSON file with parameter ranges"
    )
    single_parser.add_argument(
        "--method",
        choices=["genetic", "grid", "bayesian"],
        default="genetic",
        help="Optimization method",
    )
    single_parser.add_argument(
        "--metric", default="sharpe_ratio", help="Optimization metric"
    )
    single_parser.add_argument(
        "--iterations", type=int, default=100, help="Maximum iterations"
    )
    single_parser.add_argument(
        "--population",
        type=int,
        default=50,
        help="Population size for genetic algorithm",
    )

    # Batch optimization
    batch_parser = opt_subparsers.add_parser(
        "batch", help="Optimize multiple strategies"
    )
    batch_parser.add_argument(
        "--symbols", nargs="+", required=True, help="Symbols to optimize"
    )
    batch_parser.add_argument(
        "--strategies", nargs="+", required=True, help="Strategies to optimize"
    )
    batch_parser.add_argument("--start-date", required=True, help="Start date")
    batch_parser.add_argument("--end-date", required=True, help="End date")
    batch_parser.add_argument(
        "--parameters", required=True, help="JSON file with parameter ranges"
    )
    batch_parser.add_argument(
        "--method",
        choices=["genetic", "grid", "bayesian"],
        default="genetic",
        help="Optimization method",
    )
    batch_parser.add_argument(
        "--max-workers", type=int, help="Maximum parallel workers"
    )
    batch_parser.add_argument("--output", help="Output file for results")


def add_analysis_commands(subparsers):
    """Add analysis and reporting commands."""
    analysis_parser = subparsers.add_parser(
        "analyze", help="Analysis and reporting commands"
    )
    analysis_subparsers = analysis_parser.add_subparsers(dest="analysis_command")

    # Generate report
    report_parser = analysis_subparsers.add_parser(
        "report", help="Generate analysis report"
    )
    report_parser.add_argument(
        "--input", required=True, help="Input JSON file with results"
    )
    report_parser.add_argument(
        "--type",
        choices=["portfolio", "strategy", "optimization"],
        required=True,
        help="Report type",
    )
    report_parser.add_argument("--title", help="Report title")
    report_parser.add_argument(
        "--format", choices=["html", "json"], default="html", help="Output format"
    )
    report_parser.add_argument(
        "--output-dir", default="exports/reports", help="Output directory"
    )
    report_parser.add_argument(
        "--no-charts", action="store_true", help="Disable charts"
    )

    # Compare strategies
    compare_parser = analysis_subparsers.add_parser(
        "compare", help="Compare strategy performance"
    )
    compare_parser.add_argument(
        "--results", nargs="+", required=True, help="Result files to compare"
    )
    compare_parser.add_argument(
        "--metric", default="sharpe_ratio", help="Primary comparison metric"
    )
    compare_parser.add_argument("--output", help="Output file")


def add_cache_commands(subparsers):
    """Add cache management commands."""
    cache_parser = subparsers.add_parser("cache", help="Cache management commands")
    cache_subparsers = cache_parser.add_subparsers(dest="cache_command")

    # Cache stats
    cache_subparsers.add_parser("stats", help="Show cache statistics")

    # Clear cache
    clear_parser = cache_subparsers.add_parser("clear", help="Clear cache")
    clear_parser.add_argument(
        "--type",
        choices=["data", "backtest", "optimization"],
        help="Cache type to clear",
    )
    clear_parser.add_argument("--symbol", help="Clear cache for specific symbol")
    clear_parser.add_argument("--source", help="Clear cache for specific source")
    clear_parser.add_argument(
        "--older-than", type=int, help="Clear items older than N days"
    )
    clear_parser.add_argument("--all", action="store_true", help="Clear all cache")


def add_reports_commands(subparsers):
    """Add report management commands."""
    reports_parser = subparsers.add_parser("reports", help="Report management commands")
    reports_subparsers = reports_parser.add_subparsers(dest="reports_command")

    # Organize existing reports
    reports_subparsers.add_parser(
        "organize", help="Organize existing reports into quarterly structure"
    )

    # List reports
    list_parser = reports_subparsers.add_parser("list", help="List quarterly reports")
    list_parser.add_argument("--year", type=int, help="Filter by year")

    # Cleanup old reports
    cleanup_parser = reports_subparsers.add_parser(
        "cleanup", help="Cleanup old reports"
    )
    cleanup_parser.add_argument(
        "--keep-quarters",
        type=int,
        default=8,
        help="Number of quarters to keep (default: 8)",
    )

    # Get latest report
    latest_parser = reports_subparsers.add_parser(
        "latest", help="Get latest report for portfolio"
    )
    latest_parser.add_argument("portfolio", help="Portfolio name")

    # CSV Export
    csv_export_parser = reports_subparsers.add_parser(
        "export-csv", help="Export portfolio data to CSV format"
    )
    csv_export_parser.add_argument(
        "--portfolio",
        help="Portfolio configuration file path (optional, used for full format)",
    )
    csv_export_parser.add_argument(
        "--output", default="portfolio_raw_data.csv", help="Output CSV filename"
    )
    csv_export_parser.add_argument(
        "--format",
        choices=["full", "best-strategies", "quarterly"],
        default="full",
        help="Export format: full data, best strategies only, or quarterly summary",
    )
    csv_export_parser.add_argument(
        "--columns",
        nargs="+",
        help="Custom column selection (use 'available' to see options)",
    )
    csv_export_parser.add_argument(
        "--quarter", help="Quarter for quarterly export (Q1, Q2, Q3, Q4)"
    )
    csv_export_parser.add_argument("--year", help="Year for quarterly export (YYYY)")


# Command implementations
def handle_data_command(args):
    """Handle data management commands."""
    data_manager = UnifiedDataManager()

    if args.data_command == "download":
        handle_data_download(args, data_manager)
    elif args.data_command == "sources":
        handle_data_sources(args, data_manager)
    elif args.data_command == "symbols":
        handle_data_symbols(args, data_manager)
    else:
        print("Available data commands: download, sources, symbols")


def handle_data_download(args, data_manager: UnifiedDataManager):
    """Handle data download command."""
    logger = logging.getLogger(__name__)
    logger.info("Downloading data for %s symbols", len(args.symbols))

    successful = 0
    failed = 0

    for symbol in args.symbols:
        try:
            if args.futures:
                data = data_manager.get_crypto_futures_data(
                    symbol,
                    args.start_date,
                    args.end_date,
                    args.interval,
                    not args.force,
                )
            else:
                data = data_manager.get_data(
                    symbol,
                    args.start_date,
                    args.end_date,
                    args.interval,
                    not args.force,
                    args.asset_type,
                )

            if data is not None and not data.empty:
                successful += 1
                logger.info("✅ %s: %s data points", symbol, len(data))
            else:
                failed += 1
                logger.warning("❌ %s: No data", symbol)

        except Exception as e:
            failed += 1
            logger.error("❌ %s: %s", symbol, e)

    logger.info("Download complete: %s successful, %s failed", successful, failed)


def handle_data_sources(args, data_manager: UnifiedDataManager):
    """Handle data sources command."""
    sources = data_manager.get_source_status()

    print("\nAvailable Data Sources:")
    print("=" * 50)

    for name, status in sources.items():
        print(f"\n{name.upper()}:")
        print(f"  Priority: {status['priority']}")
        print(f"  Rate Limit: {status['rate_limit']}s")
        print(f"  Batch Support: {status['supports_batch']}")
        print(f"  Futures Support: {status['supports_futures']}")
        print(
            f"  Asset Types: {', '.join(status['asset_types']) if status['asset_types'] else 'All'}"
        )
        print(f"  Max Symbols/Request: {status['max_symbols_per_request']}")


def handle_data_symbols(args, data_manager: UnifiedDataManager):
    """Handle data symbols command."""
    print("\nAvailable Symbols:")
    print("=" * 30)

    if args.asset_type == "crypto" or not args.asset_type:
        try:
            crypto_futures = data_manager.get_available_crypto_futures()
            if crypto_futures:
                print(f"\nCrypto Futures ({len(crypto_futures)} symbols):")
                for symbol in crypto_futures[:10]:  # Show first 10
                    print(f"  {symbol}")
                if len(crypto_futures) > 10:
                    print(f"  ... and {len(crypto_futures) - 10} more")
        except Exception as e:
            print(f"Error fetching crypto symbols: {e}")

    print("\nNote: Stock and forex symbols depend on Yahoo Finance availability")


def handle_backtest_command(args):
    """Handle backtesting commands."""
    if args.backtest_command == "single":
        handle_single_backtest(args)
    elif args.backtest_command == "batch":
        handle_batch_backtest(args)
    else:
        print("Available backtest commands: single, batch")


def handle_single_backtest(args):
    """Handle single backtest command."""
    logger = logging.getLogger(__name__)

    # Setup components
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(data_manager, cache_manager)

    # Parse custom parameters
    custom_params = None
    if args.parameters:
        try:
            custom_params = json.loads(args.parameters)
        except json.JSONDecodeError as e:
            logger.error("Invalid parameters JSON: %s", e)
            return

    # Create config
    config = BacktestConfig(
        symbols=[args.symbol],
        strategies=[args.strategy],
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        initial_capital=args.capital,
        commission=args.commission,
        use_cache=not args.no_cache,
        futures_mode=args.futures,
    )

    # Run backtest
    logger.info("Running backtest: %s/%s", args.symbol, args.strategy)
    start_time = time.time()

    result = engine.run_backtest(args.symbol, args.strategy, config, custom_params)

    duration = time.time() - start_time

    # Display results
    if result.error:
        logger.error("Backtest failed: %s", result.error)
        return

    print(f"\nBacktest Results for {args.symbol}/{args.strategy}")
    print("=" * 50)
    print(f"Duration: {duration:.2f}s")
    print(f"Data Points: {result.data_points}")

    metrics = result.metrics
    if metrics:
        print("\nPerformance Metrics:")
        print(f"  Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"  Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"  Number of Trades: {metrics.get('num_trades', 0)}")


def handle_batch_backtest(args):
    """Handle batch backtest command."""
    logger = logging.getLogger(__name__)

    # Setup components
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(
        data_manager, cache_manager, args.max_workers, args.memory_limit
    )

    # Create config
    config = BacktestConfig(
        symbols=args.symbols,
        strategies=args.strategies,
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        initial_capital=args.capital,
        commission=args.commission,
        use_cache=True,
        save_trades=args.save_trades,
        save_equity_curve=args.save_equity,
        memory_limit_gb=args.memory_limit,
        max_workers=args.max_workers,
        asset_type=args.asset_type,
        futures_mode=args.futures,
    )

    # Run batch backtests
    logger.info(
        "Running batch backtests: %s symbols, %s strategies",
        len(args.symbols),
        len(args.strategies),
    )

    results = engine.run_batch_backtests(config)

    # Display summary
    successful = [r for r in results if not r.error]
    failed = [r for r in results if r.error]

    print("\nBatch Backtest Summary")
    print("=" * 30)
    print(f"Total: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")

    if successful:
        returns = [r.metrics.get("total_return", 0) for r in successful]
        print("\nPerformance Summary:")
        print(f"  Average Return: {sum(returns) / len(returns):.2f}%")
        print(f"  Best Return: {max(returns):.2f}%")
        print(f"  Worst Return: {min(returns):.2f}%")

        # Top performers
        top_performers = sorted(
            successful, key=lambda x: x.metrics.get("total_return", 0), reverse=True
        )[:5]
        print("\nTop 5 Performers:")
        for i, result in enumerate(top_performers):
            print(
                f"  {i + 1}. {result.symbol}/{result.strategy}: {result.metrics.get('total_return', 0):.2f}%"
            )

    # Save results if output specified
    if args.output:
        output_data = [asdict(result) for result in results]
        with Path(args.output).open("w") as f:
            json.dump(output_data, f, indent=2, default=str)
        logger.info("Results saved to %s", args.output)


def handle_portfolio_command(args):
    """Handle portfolio management commands."""
    if args.portfolio_command == "backtest":
        handle_portfolio_backtest(args)
    elif args.portfolio_command == "test-all":
        handle_portfolio_test_all(args)
    elif args.portfolio_command == "compare":
        handle_portfolio_compare(args)
    elif args.portfolio_command == "plan":
        handle_investment_plan(args)
    else:
        print("Available portfolio commands: backtest, test-all, compare, plan")


def handle_portfolio_test_all(args):
    """Handle testing portfolio with all strategies."""
    import webbrowser
    from datetime import datetime

    from src.reporting.detailed_portfolio_report import DetailedPortfolioReporter

    logger = logging.getLogger(__name__)

    # Load portfolio definition
    try:
        with Path(args.portfolio).open() as f:
            portfolio_data = json.load(f)

        # Get the first (and likely only) portfolio from the file
        portfolio_name = next(iter(portfolio_data.keys()))
        portfolio_config = portfolio_data[portfolio_name]
    except Exception as e:
        logger.error("Error loading portfolio: %s", e)
        return

    # Calculate date range based on period
    end_date = (
        datetime.strptime(args.end_date, "%Y-%m-%d")
        if hasattr(args, "end_date") and args.end_date
        else datetime.now()
    )

    if args.period == "max":
        start_date = get_earliest_data_date(args.portfolio)
    elif args.period == "10y":
        start_date = end_date - timedelta(days=365 * 10)
    elif args.period == "5y":
        start_date = end_date - timedelta(days=365 * 5)
    elif args.period == "2y":
        start_date = end_date - timedelta(days=365 * 2)
    else:  # default to max
        start_date = get_earliest_data_date(args.portfolio)

    # Use provided dates if available
    if hasattr(args, "start_date") and args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    if hasattr(args, "end_date") and args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    # Get all available strategies dynamically
    try:
        from src.core.strategy import StrategyFactory

        strategy_factory = StrategyFactory()
        strategies_dict = strategy_factory.list_strategies()
        all_strategies = strategies_dict["all"]
        print(f"🔍 Found {len(all_strategies)} available strategies")
        print(f"   - Built-in: {strategies_dict['builtin']}")
        print(f"   - External: {strategies_dict['external']}")
    except Exception as e:
        logger.warning("Could not load strategy factory: %s", e)
        # Fallback to basic strategies
        all_strategies = ["rsi", "macd", "bollinger_bands", "sma_crossover"]
        print(f"🔍 Using fallback strategies: {len(all_strategies)} strategies")

    # Determine timeframes to test
    if args.test_timeframes:
        timeframes_to_test = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1wk"]
    else:
        timeframes_to_test = args.timeframes

    total_combinations = (
        len(portfolio_config["symbols"]) * len(all_strategies) * len(timeframes_to_test)
    )

    print(f"\n🔍 Testing Portfolio: {portfolio_config['name']}")
    print(
        f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )
    symbols_display = ", ".join(portfolio_config["symbols"][:5])
    if len(portfolio_config["symbols"]) > 5:
        symbols_display += "..."
    print(f"📊 Symbols: {symbols_display}")
    print(f"⚙️  Strategies: {', '.join(all_strategies)}")
    print(f"⏰ Timeframes: {', '.join(timeframes_to_test)}")
    print(f"🔢 Total Combinations: {total_combinations:,}")
    print(f"📈 Primary Metric: {args.metric}")
    print("=" * 70)

    # Download data first
    print("📥 Downloading data...")

    # Setup components (single-threaded to avoid multiprocessing issues)
    data_manager = UnifiedDataManager()
    UnifiedCacheManager()

    # Download data for all symbols
    for symbol in portfolio_config["symbols"]:
        try:
            data = data_manager.get_data(
                symbol=symbol,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
            )
            if data is not None and len(data) > 0:
                print(f"  ✅ {symbol}: {len(data)} data points")
            else:
                print(f"  ❌ {symbol}: No data available")
        except Exception as e:
            print(f"  ❌ {symbol}: Error - {e!s}")

    print("\n📊 Running backtests and generating comprehensive report...")

    # Setup backtesting engine
    cache_manager = UnifiedCacheManager()
    backtest_engine = UnifiedBacktestEngine(data_manager, cache_manager, max_workers=2)

    # Run actual backtests for each symbol-strategy combination
    backtest_results = {}
    total_tests = len(portfolio_config["symbols"]) * len(all_strategies)
    current_test = 0

    for symbol in portfolio_config["symbols"]:
        backtest_results[symbol] = {}
        for strategy in all_strategies:
            current_test += 1
            print(
                f"  🔄 Testing {symbol} with {strategy} ({current_test}/{total_tests})"
            )

            try:
                config = BacktestConfig(
                    symbols=[symbol],
                    strategies=[strategy],
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    initial_capital=portfolio_config.get("initial_capital", 10000),
                    commission=portfolio_config.get("commission", 0.001),
                )

                result = backtest_engine.run_backtest(symbol, strategy, config)
                backtest_results[symbol][strategy] = result

            except Exception as e:
                print(f"    ❌ Error testing {symbol} with {strategy}: {e}")
                # Create a failed result
                backtest_results[symbol][strategy] = BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters={},
                    metrics={
                        "total_return": 0.0,
                        "final_value": portfolio_config.get("initial_capital", 10000),
                        "sortino_ratio": 0.0,
                        "sharpe_ratio": 0.0,
                        "max_drawdown": 0.0,
                    },
                    config=config,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                    error=str(e),
                )

    # Generate detailed report with actual results
    reporter = DetailedPortfolioReporter()
    report_path = reporter.generate_comprehensive_report(
        portfolio_config=portfolio_config,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        strategies=all_strategies,
        timeframes=timeframes_to_test,
    )

    print(f"\n📱 Comprehensive report generated: {report_path}")

    # Quick summary for CLI using actual results
    print(f"\n📊 Quick Summary by {args.metric.replace('_', ' ').title()}:")
    print("-" * 50)

    # Calculate actual strategy performance averages
    strategy_results = {}
    for strategy in all_strategies:
        total_score = 0
        count = 0

        for symbol_results in backtest_results.values():
            if symbol_results.get(strategy):
                result = symbol_results[strategy]
                if hasattr(result, args.metric):
                    value = getattr(result, args.metric)
                    if value is not None:
                        total_score += float(value)
                        count += 1

        # Calculate average score for this strategy
        if count > 0:
            strategy_results[strategy] = total_score / count
        else:
            strategy_results[strategy] = 0.0

    # Sort by metric (ascending for drawdown, descending for others)
    reverse_sort = args.metric != "max_drawdown"
    sorted_strategies = sorted(
        strategy_results.items(), key=lambda x: x[1], reverse=reverse_sort
    )

    for i, (strategy, score) in enumerate(sorted_strategies, 1):
        if args.metric == "sharpe_ratio":
            print(f"  {i}. {strategy:15} | Sharpe: {score:.3f}")
        elif args.metric == "total_return":
            print(f"  {i}. {strategy:15} | Return: {score:.1f}%")
        elif args.metric == "profit_factor":
            print(f"  {i}. {strategy:15} | Profit Factor: {score:.2f}")
        else:
            print(f"  {i}. {strategy:15} | Max Drawdown: {score:.1f}%")

    print(f"\n🏆 Best Overall Strategy: {sorted_strategies[0][0]}")
    print(
        "\n📊 Each asset analyzed with detailed KPIs, order history, and equity curves"
    )
    print("💾 Report size optimized with compression")

    if args.open_browser:
        import os

        webbrowser.open(f"file://{os.path.abspath(report_path)}")
        print("📱 Detailed report opened in browser")


def handle_portfolio_backtest(args):
    """Handle portfolio backtest command."""
    logger = logging.getLogger(__name__)

    # Parse weights
    weights = None
    if args.weights:
        try:
            weights = json.loads(args.weights)
        except json.JSONDecodeError as e:
            logger.error("Invalid weights JSON: %s", e)
            return

    # Setup components
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(data_manager, cache_manager)

    # Create config
    config = BacktestConfig(
        symbols=args.symbols,
        strategies=[args.strategy],
        start_date=args.start_date,
        end_date=args.end_date,
        interval=args.interval,
        initial_capital=args.capital,
        use_cache=True,
    )

    # Run portfolio backtest
    logger.info("Running portfolio backtest: %s symbols", len(args.symbols))

    result = engine.run_portfolio_backtest(config, weights)

    # Display results
    if result.error:
        logger.error("Portfolio backtest failed: %s", result.error)
        return

    print("\nPortfolio Backtest Results")
    print("=" * 30)

    metrics = result.metrics
    if metrics:
        print(f"Total Return: {metrics.get('total_return', 0):.2f}%")
        print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%")
        print(f"Volatility: {metrics.get('volatility', 0):.2f}%")


def handle_portfolio_compare(args):
    """Handle portfolio comparison command."""
    logger = logging.getLogger(__name__)

    # Load portfolio definitions
    try:
        with Path(args.portfolios).open() as f:
            portfolio_definitions = json.load(f)
    except Exception as e:
        logger.error("Error loading portfolios: %s", e)
        return

    # Setup components
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(data_manager, cache_manager)
    portfolio_manager = PortfolioManager()

    # Define all available strategies
    all_strategies = ["rsi", "macd", "bollinger_bands", "sma_crossover"]

    # Run backtests for each portfolio
    portfolio_results = {}

    for portfolio_name, portfolio_config in portfolio_definitions.items():
        logger.info("Backtesting portfolio: %s", portfolio_name)

        # Use strategies from config if provided, otherwise use all strategies
        strategies_to_test = portfolio_config.get("strategies", all_strategies)

        config = BacktestConfig(
            symbols=portfolio_config["symbols"],
            strategies=strategies_to_test,
            start_date=args.start_date,
            end_date=args.end_date,
            use_cache=True,
        )

        results = engine.run_batch_backtests(config)
        portfolio_results[portfolio_name] = results

    # Analyze portfolios
    analysis = portfolio_manager.analyze_portfolios(portfolio_results)

    # Display comparison
    print("\nPortfolio Comparison Analysis")
    print("=" * 40)

    for portfolio_name, summary in analysis["portfolio_summaries"].items():
        print(f"\n{portfolio_name.upper()}:")
        print(f"  Priority Rank: {summary['investment_priority']}")
        print(f"  Average Return: {summary['avg_return']:.2f}%")
        print(f"  Sharpe Ratio: {summary['avg_sharpe']:.3f}")
        print(f"  Risk Category: {summary['risk_category']}")
        print(f"  Overall Score: {summary['overall_score']:.1f}")

    # Show investment recommendations
    print("\nInvestment Recommendations:")
    for rec in analysis["investment_recommendations"]:
        print(f"\n{rec['priority_rank']}. {rec['portfolio_name']}")
        print(f"   Allocation: {rec['recommended_allocation_pct']:.1f}%")
        print(f"   Expected Return: {rec['expected_annual_return']:.2f}%")
        print(f"   Risk: {rec['risk_category']}")

    # Save results if output specified
    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(analysis, f, indent=2, default=str)
        logger.info("Analysis saved to %s", args.output)


def handle_investment_plan(args):
    """Handle investment plan generation."""
    logger = logging.getLogger(__name__)

    # Load portfolio results
    try:
        with Path(args.portfolios).open() as f:
            portfolio_results_data = json.load(f)
    except Exception as e:
        logger.error("Error loading portfolio results: %s", e)
        return

    # Convert to BacktestResult objects (simplified)
    portfolio_results = {}
    for portfolio_name, results_list in portfolio_results_data.items():
        results = []
        for result_data in results_list:
            result = BacktestResult(
                symbol=result_data["symbol"],
                strategy=result_data["strategy"],
                parameters=result_data.get("parameters", {}),
                metrics=result_data.get("metrics", {}),
                config=None,  # Simplified
                error=result_data.get("error"),
            )
            results.append(result)
        portfolio_results[portfolio_name] = results

    # Generate investment plan
    portfolio_manager = PortfolioManager()
    investment_plan = portfolio_manager.generate_investment_plan(
        args.capital, portfolio_results, args.risk_tolerance
    )

    # Display investment plan
    print("\nInvestment Plan")
    print("=" * 20)
    print(f"Total Capital: ${args.capital:,.2f}")
    print(f"Risk Tolerance: {args.risk_tolerance.title()}")

    print("\nCapital Allocations:")
    for allocation in investment_plan["allocations"]:
        print(
            f"  {allocation['portfolio_name']}: ${allocation['allocation_amount']:,.2f} "
            f"({allocation['allocation_percentage']:.1f}%)"
        )

    print("\nExpected Portfolio Metrics:")
    expected = investment_plan["expected_portfolio_metrics"]
    print(f"  Expected Return: {expected.get('expected_annual_return', 0):.2f}%")
    print(f"  Expected Volatility: {expected.get('expected_volatility', 0):.2f}%")
    print(f"  Expected Sharpe: {expected.get('expected_sharpe_ratio', 0):.3f}")

    # Save plan if output specified
    if args.output:
        with Path(args.output).open("w") as f:
            json.dump(investment_plan, f, indent=2, default=str)
        logger.info("Investment plan saved to %s", args.output)


def handle_cache_command(args):
    """Handle cache management commands."""
    cache_manager = UnifiedCacheManager()

    if args.cache_command == "stats":
        handle_cache_stats(args, cache_manager)
    elif args.cache_command == "clear":
        handle_cache_clear(args, cache_manager)
    else:
        print("Available cache commands: stats, clear")


def handle_cache_stats(args, cache_manager: UnifiedCacheManager):
    """Handle cache stats command."""
    stats = cache_manager.get_cache_stats()

    print("\nCache Statistics")
    print("=" * 20)
    print(
        f"Total Size: {stats['total_size_gb']:.2f} GB / {stats['max_size_gb']:.2f} GB"
    )
    print(f"Utilization: {stats['utilization_percent']:.1f}%")

    print("\nBy Type:")
    for cache_type, type_stats in stats["by_type"].items():
        print(f"  {cache_type.title()}:")
        print(f"    Count: {type_stats['count']}")
        print(f"    Size: {type_stats['total_size_mb']:.1f} MB")

    print("\nBy Source:")
    for source, source_stats in stats["by_source"].items():
        print(f"  {source.title()}:")
        print(f"    Count: {source_stats['count']}")
        print(f"    Size: {source_stats['size_bytes'] / 1024**2:.1f} MB")


def handle_cache_clear(args, cache_manager: UnifiedCacheManager):
    """Handle cache clear command."""
    logger = logging.getLogger(__name__)

    if args.all:
        logger.info("Clearing all cache...")
        cache_manager.clear_cache()
    else:
        logger.info("Clearing cache with filters...")
        cache_manager.clear_cache(
            cache_type=args.type,
            symbol=args.symbol,
            source=args.source,
            older_than_days=args.older_than,
        )

    logger.info("Cache cleared successfully")


def handle_strategy_command(args):
    """Handle strategy management commands."""
    logger = logging.getLogger(__name__)

    if args.strategy_command == "list":
        strategies = list_available_strategies()
        strategy_type = args.type

        if strategy_type == "all":
            print("Available Strategies:")
            if strategies["builtin"]:
                print(f"\nBuilt-in Strategies ({len(strategies['builtin'])}):")
                for strategy in strategies["builtin"]:
                    print(f"  - {strategy}")

            if strategies["external"]:
                print(f"\nExternal Strategies ({len(strategies['external'])}):")
                for strategy in strategies["external"]:
                    print(f"  - {strategy}")
        else:
            strategy_list = strategies.get(strategy_type, [])
            print(f"{strategy_type.title()} Strategies ({len(strategy_list)}):")
            for strategy in strategy_list:
                print(f"  - {strategy}")

    elif args.strategy_command == "info":
        try:
            info = StrategyFactory.get_strategy_info(args.name)
            print(f"Strategy: {info['name']}")
            print(f"Type: {info['type']}")
            print(f"Description: {info['description']}")

            if info.get("parameters"):
                print("\nParameters:")
                for param, value in info["parameters"].items():
                    print(f"  {param}: {value}")
        except ValueError as e:
            logger.error("Strategy not found: %s", e)

    elif args.strategy_command == "test":
        try:
            # Parse parameters if provided
            parameters = {}
            if args.parameters:
                parameters = json.loads(args.parameters)

            # Create strategy instance
            strategy = StrategyFactory.create_strategy(args.name, parameters)

            # Get test data
            data_manager = UnifiedDataManager()
            logger.info("Fetching test data for %s...", args.symbol)

            data = data_manager.fetch_data(
                symbol=args.symbol,
                start_date=args.start_date,
                end_date=args.end_date,
                interval="1d",
            )

            if data.empty:
                logger.error("No data found for %s", args.symbol)
                return

            # Generate signals
            logger.info("Generating signals...")
            signals = strategy.generate_signals(data)

            # Print summary
            signal_counts = {
                "Buy": (signals == 1).sum(),
                "Sell": (signals == -1).sum(),
                "Hold": (signals == 0).sum(),
            }

            print(f"\nStrategy Test Results for {args.name}:")
            print(f"Symbol: {args.symbol}")
            print(f"Period: {args.start_date} to {args.end_date}")
            print(f"Data points: {len(data)}")
            print("Signal distribution:")
            for signal_type, count in signal_counts.items():
                percentage = (count / len(signals)) * 100
                print(f"  {signal_type}: {count} ({percentage:.1f}%)")

            # Show recent signals
            recent_signals = signals.tail(10)
            print("\nRecent signals:")
            for date, signal in recent_signals.items():
                signal_name = {1: "BUY", -1: "SELL", 0: "HOLD"}[signal]
                print(f"  {date.strftime('%Y-%m-%d')}: {signal_name}")

        except Exception as e:
            logger.error("Strategy test failed: %s", e)


def handle_csv_export_command(args):
    """Handle CSV export command."""
    from src.utils.raw_data_csv_exporter import RawDataCSVExporter

    # Show available columns if requested
    if hasattr(args, "columns") and args.columns and "available" in args.columns:
        exporter = RawDataCSVExporter()
        print("Available columns for CSV export:")
        for col in exporter.get_available_columns():
            print(f"  - {col}")
        return

    # Initialize exporter
    exporter = RawDataCSVExporter()

    # Handle quarterly exports from existing reports
    if args.format == "quarterly":
        if not args.quarter or not args.year:
            print("❌ Quarter and year required for quarterly export")
            return

        print(f"📊 Extracting data from quarterly reports: {args.year} {args.quarter}")

        output_paths = exporter.export_from_quarterly_reports(
            args.quarter, args.year, args.output, "full"
        )

        if output_paths:
            print(f"✅ CSV export completed - {len(output_paths)} files:")
            for path in output_paths:
                print(f"   📄 {path}")
        else:
            print("❌ No quarterly reports found or CSV export failed")
        return

    # Handle best-strategies format (can work from quarterly reports if quarter/year provided)
    if args.format == "best-strategies":
        if args.quarter and args.year:
            print(
                f"📊 Extracting best strategies from quarterly reports: {args.year} {args.quarter}"
            )

            output_paths = exporter.export_from_quarterly_reports(
                args.quarter, args.year, args.output, "best-strategies"
            )

            if output_paths:
                print(
                    f"✅ Best strategies CSV export completed - {len(output_paths)} files:"
                )
                for path in output_paths:
                    print(f"   📄 {path}")
            else:
                print("❌ No quarterly reports found or CSV export failed")
            return
        print(
            "❌ Best strategies export requires --quarter and --year to extract from existing reports"
        )
        print("💡 Use: --format best-strategies --quarter Q1 --year 2024")
        return

    # Handle full format - requires running backtests (fallback to old method)
    print("❌ Full format export from portfolio config not implemented yet")
    print(
        "💡 Use quarterly format with --quarter and --year to extract from existing reports"
    )
    print("💡 Example: --format quarterly --quarter Q4 --year 2023")
    return


def handle_reports_command(args):
    """Handle report management commands."""
    import os
    import sys

    from src.utils.report_organizer import ReportOrganizer

    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from utils.report_organizer import ReportOrganizer

    organizer = ReportOrganizer()

    if args.reports_command == "organize":
        print("Organizing existing reports into quarterly structure...")
        organizer.organize_existing_reports()
        print("Reports organized successfully!")

    elif args.reports_command == "list":
        reports = organizer.list_quarterly_reports(
            args.year if hasattr(args, "year") else None
        )

        if not reports:
            print("No quarterly reports found.")
            return

        for year, quarters in reports.items():
            print(f"\n{year}:")
            for quarter, report_files in quarters.items():
                print(f"  {quarter}:")
                for report_file in report_files:
                    print(f"    - {report_file}")

    elif args.reports_command == "cleanup":
        keep_quarters = args.keep_quarters if hasattr(args, "keep_quarters") else 8
        print(f"Cleaning up old reports (keeping last {keep_quarters} quarters)...")
        organizer.cleanup_old_reports(keep_quarters)
        print("Cleanup completed!")

    elif args.reports_command == "latest":
        portfolio_name = args.portfolio
        latest_report = organizer.get_latest_report(portfolio_name)

        if latest_report:
            print(f"Latest report for '{portfolio_name}': {latest_report}")
        else:
            print(f"No reports found for portfolio '{portfolio_name}'")

    elif args.reports_command == "export-csv":
        handle_csv_export_command(args)

    else:
        print("Available reports commands: organize, list, cleanup, latest, export-csv")


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(args.log_level)

    # Route to appropriate handler
    try:
        if args.command == "data":
            handle_data_command(args)
        elif args.command == "strategy":
            handle_strategy_command(args)
        elif args.command == "backtest":
            handle_backtest_command(args)
        elif args.command == "portfolio":
            handle_portfolio_command(args)
        elif args.command == "cache":
            handle_cache_command(args)
        elif args.command == "reports":
            handle_reports_command(args)
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()

    except KeyboardInterrupt:
        print("\nOperation interrupted by user")
    except Exception as e:
        logging.error("Command failed: %s", e)
        raise


if __name__ == "__main__":
    main()
