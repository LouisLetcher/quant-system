from __future__ import annotations

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_asset_config, get_default_parameters
from src.optimizer.optimization_runner import OptimizationRunner
from src.reports.report_generator import ReportGenerator


def optimize_strategy(args):
    """
    Optimize parameters for a strategy on a specific asset.
    """
    defaults = get_default_parameters()

    # Check if ticker has specific config
    asset_config = get_asset_config(args.ticker)
    commission = (
        args.commission
        if args.commission is not None
        else (
            asset_config.get("commission", defaults["commission"])
            if asset_config
            else defaults["commission"]
        )
    )
    initial_capital = (
        args.initial_capital
        if args.initial_capital is not None
        else (
            asset_config.get("initial_capital", defaults["initial_capital"])
            if asset_config
            else defaults["initial_capital"]
        )
    )

    print(f"🔍 Optimizing {args.strategy} for {args.ticker}...")
    print(f"Using commission: {commission}, initial capital: {initial_capital}")

    # Get strategy class
    strategy_class = StrategyFactory.get_strategy(args.strategy)
    if strategy_class is None:
        print(f"❌ Strategy '{args.strategy}' not found.")
        return None

    # Load data
    data = DataLoader.load_data(
        args.ticker, period=args.period, start=args.start_date, end=args.end_date
    )
    if data is None or data.empty:
        print(f"❌ No data available for {args.ticker}.")
        return None

    print(f"✅ Loaded {len(data)} bars for {args.ticker}")

    # Get parameter ranges
    param_ranges = _get_param_ranges(strategy_class)
    if not param_ranges:
        print(f"❌ No parameters to optimize for {args.strategy}.")
        return None

    print(f"🔧 Optimizing parameters: {param_ranges}")

    # Run optimization
    optimizer = OptimizationRunner(strategy_class, data, param_ranges)
    results = optimizer.run(
        metric=args.metric,
        iterations=args.iterations,
        initial_capital=initial_capital,
        commission=commission,
    )

    # Generate report
    output_path = f"reports_output/optimizer_{args.strategy}_{args.ticker}.html"
    generator = ReportGenerator()
    generator.generate_optimizer_report(results, output_path)

    print(f"📄 Optimization Report saved to {output_path}")
    return results


def _get_param_ranges(strategy_class):
    """Helper function to get parameter ranges for optimization"""
    # Check if strategy has default parameter ranges
    if hasattr(strategy_class, "param_ranges"):
        return strategy_class.param_ranges

    # Create default ranges based on strategy attributes
    param_ranges = {}
    for attr_name in dir(strategy_class):
        # Skip special attributes and methods
        if attr_name.startswith("_") or callable(getattr(strategy_class, attr_name)):
            continue

        # Get attribute value
        attr_value = getattr(strategy_class, attr_name)

        # Only include numeric parameters
        if isinstance(attr_value, (int, float)):
            if attr_name.endswith("_period") or attr_name.endswith("_length"):
                # For period parameters, create a reasonable range
                param_ranges[attr_name] = (max(5, attr_value // 2), attr_value * 2)
            elif 0 <= attr_value <= 1:
                # For parameters between 0 and 1 (like thresholds)
                param_ranges[attr_name] = (
                    max(0.01, attr_value / 2),
                    min(0.99, attr_value * 2),
                )
            else:
                # For other numeric parameters
                param_ranges[attr_name] = (attr_value * 0.5, attr_value * 1.5)

    return param_ranges


def register_commands(subparsers):
    """Register optimizer commands with the CLI parser"""
    # Optimization command
    optimize_parser = subparsers.add_parser(
        "optimize", help="Optimize parameters for a strategy"
    )
    optimize_parser.add_argument(
        "--strategy", type=str, required=True, help="Trading strategy name"
    )
    optimize_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker symbol"
    )
    optimize_parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'",
    )
    optimize_parser.add_argument(
        "--metric",
        type=str,
        default="sharpe",
        help="Performance metric to optimize ('sharpe', 'return', 'profit_factor')",
    )
    optimize_parser.add_argument(
        "--iterations", type=int, default=50, help="Number of optimization iterations"
    )
    optimize_parser.add_argument(
        "--initial-capital", type=float, help="Initial capital"
    )
    optimize_parser.add_argument("--commission", type=float, help="Commission rate")
    optimize_parser.add_argument(
        "--start-date", type=str, help="Start date (YYYY-MM-DD)"
    )
    optimize_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    optimize_parser.set_defaults(func=optimize_strategy)
