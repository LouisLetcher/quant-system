from __future__ import annotations

import base64
import io
import os
import webbrowser
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_default_parameters, get_portfolio_config
from src.optimizer.parameter_tuner import ParameterTuner
from src.portfolio.metrics_processor import extract_detailed_metrics
from src.reports.report_generator import ReportGenerator
from src.utils.logger import get_logger, setup_command_logging

# Initialize logger
logger = get_logger(__name__)


def optimize_portfolio_parameters(args):
    """Optimize parameters for the best strategy/timeframe combinations found in a portfolio."""
    # Setup logging if requested
    log_file = setup_command_logging(args)

    logger.info(f"Starting parameter optimization for portfolio '{args.name}'")
    print(f"Starting parameter optimization for portfolio '{args.name}'")

    # Get portfolio configuration
    portfolio_config = get_portfolio_config(args.name)
    if not portfolio_config:
        logger.error(f"Portfolio '{args.name}' not found in assets_config.json")
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        return {}

    # Get default report path if not provided
    report_path = args.report_path
    if not report_path:
        report_path = f"reports_output/portfolio_optimal_{args.name}.html"
        if not os.path.exists(report_path):
            report_path = f"reports_output/portfolio_{args.name}_{args.metric}.html"

    # Check if report exists
    if not os.path.exists(report_path):
        logger.error(f"Report file not found: {report_path}")
        print(f"❌ Report file not found: {report_path}")
        print(
            "Please run portfolio-optimal or portfolio command first, or specify the correct report path."
        )
        return {}

    # Extract best combinations from the report
    best_combinations = extract_best_combinations_from_report(report_path)
    if not best_combinations:
        logger.error("Failed to extract best combinations from the report")
        print("❌ Failed to extract best combinations from the report")
        return {}

    logger.info(
        f"Found {len(best_combinations)} assets with strategy combinations to optimize"
    )
    print(
        f"Found {len(best_combinations)} assets with strategy combinations to optimize"
    )

    # Get default parameters
    defaults = get_default_parameters()

    # Optimize parameters for each asset
    optimized_results = {}
    for ticker, combo in best_combinations.items():
        strategy_name = combo.get("strategy")
        interval = combo.get("interval")

        if not strategy_name or not interval:
            logger.warning(f"Skipping {ticker}: No valid strategy or interval found")
            print(f"⚠️ Skipping {ticker}: No valid strategy or interval found")
            continue

        logger.info(
            f"Optimizing parameters for {ticker}: {strategy_name} with {interval} interval"
        )
        print(
            f"\n🔍 Optimizing parameters for {ticker}: {strategy_name} with {interval} interval"
        )

        # Get asset-specific configuration
        asset_config = next(
            (
                a
                for a in portfolio_config.get("assets", [])
                if a.get("ticker") == ticker
            ),
            {},
        )
        commission = asset_config.get("commission", defaults["commission"])
        initial_capital = asset_config.get(
            "initial_capital", defaults["initial_capital"]
        )
        period = asset_config.get("period", "max")

        # Load data
        data = DataLoader.load_data(ticker, period=period, interval=interval)

        if data is None or data.empty:
            logger.warning(f"No data available for {ticker} with {interval} interval")
            print(f"⚠️ No data available for {ticker} with {interval} interval")
            continue

        # Get strategy class
        strategy_class = StrategyFactory.get_strategy(strategy_name)

        # Create parameter tuner
        tuner = ParameterTuner(
            strategy_class=strategy_class,
            data=data,
            initial_capital=initial_capital,
            commission=commission,
            ticker=ticker,
            metric=args.metric,
        )

        # Run optimization
        try:
            logger.info(
                f"Running parameter optimization for {ticker} with max_tries={args.max_tries}, method={args.method}"
            )
            print(
                f"Running parameter optimization for {ticker} with max_tries={args.max_tries}, method={args.method}"
            )

            # Get parameter ranges from strategy
            param_ranges = get_param_ranges(strategy_class)

            # Optimize parameters
            best_params, best_value, optimization_results = tuner.optimize(
                param_ranges=param_ranges, max_tries=args.max_tries, method=args.method
            )

            logger.info(
                f"Optimization complete for {ticker}: Best {args.metric}={best_value}"
            )
            print(
                f"✅ Optimization complete for {ticker}: Best {args.metric}={best_value}"
            )
            print(f"Best parameters: {best_params}")

            # Run backtest with optimized parameters
            optimized_result = run_backtest_with_params(
                strategy_class=strategy_class,
                data=data,
                params=best_params,
                initial_capital=initial_capital,
                commission=commission,
                ticker=ticker,
            )

            # Extract detailed metrics
            detailed_metrics = extract_detailed_metrics(
                optimized_result, initial_capital
            )

            # Generate equity chart
            equity_chart = None
            if detailed_metrics.get("equity_curve"):
                equity_chart = generate_equity_chart(
                    detailed_metrics["equity_curve"],
                    ticker,
                    f"{strategy_name} (Optimized)",
                )

            # Store results
            optimized_results[ticker] = {
                "strategy": strategy_name,
                "interval": interval,
                "original_score": combo.get("score", 0),
                "optimized_score": best_value,
                "improvement": best_value - combo.get("score", 0),
                "improvement_pct": (
                    (best_value - combo.get("score", 0))
                    / max(0.0001, abs(combo.get("score", 0.0001)))
                )
                * 100,
                "best_params": best_params,
                "optimization_results": optimization_results,
                "equity_chart": equity_chart,
                **detailed_metrics,
            }

            # Print improvement
            improvement = best_value - combo.get("score", 0)
            improvement_pct = (
                improvement / max(0.0001, abs(combo.get("score", 0.0001))) * 100
            )
            print(f"Improvement: {improvement:.4f} ({improvement_pct:.2f}%)")

        except Exception as e:
            logger.error(f"Error optimizing parameters for {ticker}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            print(f"❌ Error optimizing parameters for {ticker}: {e}")

    # Generate report
    if optimized_results:
        output_path = f"reports_output/portfolio_optimized_params_{args.name}.html"

        # Create report data
        report_data = {
            "portfolio": args.name,
            "description": portfolio_config.get("description", ""),
            "best_combinations": optimized_results,
            "metric": args.metric,
            "is_parameter_optimized": True,
            "date_generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Generate report
        generator = ReportGenerator()
        report_path = generator.generate_parameter_optimization_report(
            report_data, output_path
        )

        print(f"\n📄 Parameter Optimization Report saved to: {report_path}")
        logger.info(f"Parameter Optimization Report saved to: {report_path}")

        # Open in browser if requested
        if args.open_browser:
            logger.info(f"Opening report in browser: {os.path.abspath(report_path)}")
            webbrowser.open(f"file://{os.path.abspath(report_path)}", new=2)

    logger.info("Parameter optimization completed")
    print("\nParameter optimization completed")
    return optimized_results


def extract_best_combinations_from_report(report_path):
    """Extract best strategy combinations from a portfolio report."""
    try:
        with open(report_path, encoding="utf-8") as f:
            html_content = f.read()

        # Parse HTML
        soup = BeautifulSoup(html_content, "html.parser")

        # Extract data from the assets table
        best_combinations = {}
        assets_table = soup.find("h2", text="Assets Overview").find_next("table")

        if assets_table:
            rows = assets_table.find_all("tr")[1:]  # Skip header row
            for row in rows:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    ticker = cells[0].text.strip()
                    strategy = cells[1].text.strip()
                    interval = cells[2].text.strip()

                    # Skip if no strategy or N/A
                    if strategy == "N/A" or not strategy:
                        continue

                    # Extract metrics
                    return_pct = (
                        float(cells[3].text.strip().replace("%", ""))
                        if len(cells) > 3
                        else 0
                    )
                    sharpe = float(cells[4].text.strip()) if len(cells) > 4 else 0
                    max_dd = (
                        float(cells[5].text.strip().replace("%", ""))
                        if len(cells) > 5
                        else 0
                    )
                    win_rate = (
                        float(cells[6].text.strip().replace("%", ""))
                        if len(cells) > 6
                        else 0
                    )
                    trades = int(cells[7].text.strip()) if len(cells) > 7 else 0
                    profit_factor = (
                        float(cells[8].text.strip()) if len(cells) > 8 else 0
                    )

                    best_combinations[ticker] = {
                        "strategy": strategy,
                        "interval": interval,
                        "return_pct": return_pct,
                        "sharpe_ratio": sharpe,
                        "max_drawdown_pct": max_dd,
                        "win_rate": win_rate,
                        "trades_count": trades,
                        "profit_factor": profit_factor,
                        "score": sharpe,  # Default to sharpe as score
                    }

        return best_combinations

    except Exception as e:
        logger.error(f"Error extracting best combinations from report: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return {}


def get_param_ranges(strategy_class):
    """Get parameter ranges for optimization."""
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


def run_backtest_with_params(
    strategy_class, data, params, initial_capital, commission, ticker
):
    """Run a backtest with specific parameters."""
    # Create a new instance of the strategy with the optimized parameters
    strategy_instance = type("OptimizedStrategy", (strategy_class,), params)

    # Create backtest instance
    engine = BacktestEngine(
        strategy_instance,
        data,
        cash=initial_capital,
        commission=commission,
        ticker=ticker,
    )

    # Run backtest
    result = engine.run()
    return result


def generate_equity_chart(equity_curve, ticker, strategy_name):
    """Generate an equity curve chart as a base64-encoded image."""
    try:
        # Convert equity curve data to DataFrame if it's a list
        if isinstance(equity_curve, list):
            # Check if equity curve is in the expected format
            if (
                equity_curve
                and isinstance(equity_curve[0], dict)
                and "date" in equity_curve[0]
                and "value" in equity_curve[0]
            ):
                dates = [pd.to_datetime(point["date"]) for point in equity_curve]
                values = [point["value"] for point in equity_curve]
                equity_df = pd.DataFrame({"equity": values}, index=dates)
            else:
                logger.warning(f"Equity curve data format not recognized for {ticker}")
                return None
        elif isinstance(equity_curve, pd.DataFrame):
            equity_df = equity_curve
        else:
            logger.warning(f"Equity curve data type not supported for {ticker}")
            return None

        # Create figure
        plt.figure(figsize=(10, 6))

        # Plot equity curve
        plt.plot(equity_df.index, equity_df["equity"], label="Equity", color="#2980b9")

        # Calculate drawdown
        rolling_max = equity_df["equity"].cummax()
        drawdown = -100 * (rolling_max - equity_df["equity"]) / rolling_max

        # Plot drawdown
        plt.fill_between(
            equity_df.index, 0, drawdown, alpha=0.3, color="#e74c3c", label="Drawdown %"
        )

        # Add labels and title
        plt.title(
            f"{ticker} - {strategy_name} Equity Curve & Drawdown",
            fontname="DejaVu Sans",
        )
        plt.xlabel("Date")
        plt.ylabel("Equity / Drawdown %")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Tight layout
        plt.tight_layout()

        # Convert plot to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", dpi=100)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        return image_base64

    except Exception as e:
        logger.error(f"Error generating equity chart for {ticker}: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return None
