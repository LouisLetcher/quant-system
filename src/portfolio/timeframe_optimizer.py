from __future__ import annotations

import os
import webbrowser

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_default_parameters
from src.portfolio.metrics_processor import extract_detailed_metrics
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def backtest_all_strategies_all_timeframes(
    ticker,
    asset_config,
    strategies,
    intervals,
    period,
    metric,
    start_date=None,
    end_date=None,
    plot=False,
    resample=None,
):
    """
    Backtest all strategies with all timeframes for a single asset.
    Returns structured results for detailed reporting.
    """
    try:
        logger.info(f"Testing all strategies and timeframes for {ticker}")

        defaults = get_default_parameters()
        commission = asset_config.get("commission", defaults["commission"])
        initial_capital = asset_config.get(
            "initial_capital", defaults["initial_capital"]
        )
        asset_period = asset_config.get("period", period)

        logger.info(
            f"Using commission: {commission}, initial capital: {initial_capital}"
        )
        print(f"\n🔍 Testing all strategies and timeframes for: {ticker}")
        print(f"Using commission: {commission}, initial capital: {initial_capital}")

        # Track best combination across all strategies and intervals
        best_score = -float("inf")
        best_strategy = None
        best_interval = None
        best_result = None

        # Structure to hold all results
        all_results = {"ticker": ticker, "strategies": []}

        # Test each strategy
        for strategy_name in strategies:
            logger.info(f"Testing strategy {strategy_name} for {ticker}")
            print(f"  Testing strategy {strategy_name}...")

            strategy_entry = {
                "name": strategy_name,
                "best_timeframe": None,
                "best_score": -float("inf"),
                "timeframes": [],
            }

            # Test each interval
            for interval in intervals:
                logger.info(
                    f"Testing {strategy_name} with {interval} interval for {ticker}"
                )
                print(f"    Testing {interval} interval...")

                try:
                    # Load data for this ticker and interval
                    data = load_timeframe_data(
                        ticker, asset_period, interval, start_date, end_date
                    )

                    if data is None:
                        continue

                    # Run backtest for this strategy and interval
                    result, score, trades = run_timeframe_backtest(
                        ticker,
                        strategy_name,
                        data,
                        initial_capital,
                        commission,
                        metric,
                        interval,
                    )

                    # Extract detailed metrics
                    detailed_metrics = extract_detailed_metrics(result, initial_capital)

                    # Add timeframe results
                    timeframe_data = create_timeframe_result(
                        interval, score, detailed_metrics
                    )

                    strategy_entry["timeframes"].append(timeframe_data)

                    # Update best timeframe for this strategy
                    if score > strategy_entry["best_score"] and trades > 0:
                        strategy_entry["best_score"] = score
                        strategy_entry["best_timeframe"] = interval

                    # Update best overall combination
                    if score > best_score and trades > 0:
                        best_score = score
                        best_strategy = strategy_name
                        best_interval = interval
                        best_result = detailed_metrics

                except Exception as e:
                    logger.error(
                        f"Error testing {strategy_name} with {interval} for {ticker}: {e}"
                    )
                    print(f"      ❌ Error: {e}")
                    import traceback

                    logger.error(traceback.format_exc())

            # Add strategy results to all_results
            all_results["strategies"].append(strategy_entry)

        # Create best combination data structure
        best_combination = create_best_combination(
            ticker, best_strategy, best_interval, best_score, best_result, metric
        )

        # Plot the best backtest if requested
        if plot and best_strategy and best_interval:
            plot_best_combination(
                ticker,
                best_strategy,
                best_interval,
                asset_period,
                initial_capital,
                commission,
                start_date,
                end_date,
                resample,
            )

        return {"best_combination": best_combination, "all_results": all_results}

    except Exception as e:
        logger.error(
            f"Error in backtest_all_strategies_all_timeframes for {ticker}: {e}"
        )
        import traceback

        logger.error(traceback.format_exc())
        return {
            "best_combination": {
                "strategy": None,
                "interval": None,
                "score": -float("inf"),
                "error": f"Error: {e!s}",
            },
            "all_results": {"ticker": ticker, "error": str(e), "strategies": []},
        }


def load_timeframe_data(ticker, period, interval, start_date=None, end_date=None):
    """Load data for a specific ticker and timeframe."""
    data = DataLoader.load_data(
        ticker,
        period=period,
        interval=interval,
        start=start_date,
        end=end_date,
    )

    if data is None or data.empty:
        logger.warning(f"No data available for {ticker} with {interval} interval")
        print(f"      ⚠️ No data available for {interval}")
        return None

    logger.debug(
        f"Loaded {len(data)} data points for {ticker} with {interval} interval"
    )
    return data


def run_timeframe_backtest(
    ticker, strategy_name, data, initial_capital, commission, metric, interval
):
    """Run a backtest for a specific strategy and timeframe."""
    # Get the strategy class
    strategy_class = StrategyFactory.get_strategy(strategy_name)

    # Create backtest instance
    engine = BacktestEngine(
        strategy_class,
        data,
        cash=initial_capital,
        commission=commission,
        ticker=ticker,
    )

    # Run backtest
    result = engine.run()

    # Extract performance metric
    if metric == "profit_factor":
        score = result.get("Profit Factor", 0)
    elif metric == "sharpe":
        score = result.get("Sharpe Ratio", 0)
    elif metric == "return":
        score = result.get("Return [%]", 0)
    else:
        score = result.get(metric, 0)

    # Get trade count
    trades = result.get("# Trades", 0)

    logger.info(
        f"{strategy_name} + {interval} for {ticker}: {metric}={score}, Trades={trades}"
    )
    print(f"      {strategy_name} + {interval}: {metric}={score}, Trades={trades}")

    return result, score, trades


def create_timeframe_result(interval, score, detailed_metrics):
    """Create a structured result for a timeframe backtest."""
    return {
        "interval": interval,
        "score": score,
        "return_pct": detailed_metrics.get("return_pct", 0),
        "win_rate": detailed_metrics.get("win_rate", 0),
        "trades_count": detailed_metrics.get("trades_count", 0),
        "profit_factor": detailed_metrics.get("profit_factor", 0),
        "max_drawdown_pct": detailed_metrics.get("max_drawdown_pct", 0),
        "sharpe_ratio": detailed_metrics.get("sharpe_ratio", 0),
        "equity_curve": detailed_metrics.get("equity_curve", []),
        "trades": detailed_metrics.get("trades", []),
    }


def create_best_combination(
    ticker, best_strategy, best_interval, best_score, best_result, metric
):
    """Create the best combination result structure."""
    if best_strategy and best_interval and best_result:
        best_combination = {
            "strategy": best_strategy,
            "interval": best_interval,
            "score": best_score,
            **best_result,  # Include all detailed metrics
        }
        logger.info(
            f"Best combination for {ticker}: {best_strategy} with {best_interval}, {metric}={best_score}"
        )
        print(
            f"  ✅ Best combination: {best_strategy} with {best_interval}, {metric}={best_score}"
        )
    else:
        # No valid combination found
        best_combination = {
            "strategy": None,
            "interval": None,
            "score": -float("inf"),
            "return_pct": 0,
            "win_rate": 0,
            "trades_count": 0,
            "profit_factor": 0,
            "max_drawdown_pct": 0,
            "sharpe_ratio": 0,
        }
        logger.warning(f"No valid strategy-interval combination found for {ticker}")
        print("  ⚠️ No valid strategy-interval combination found")

    return best_combination


def plot_best_combination(
    ticker,
    strategy_name,
    interval,
    period,
    initial_capital,
    commission,
    start_date,
    end_date,
    resample=None,
):
    """Plot the best strategy combination."""
    try:
        # Load data and run backtest again for plotting
        data = load_timeframe_data(ticker, period, interval, start_date, end_date)

        if data is None:
            return False

        strategy_class = StrategyFactory.get_strategy(strategy_name)
        engine = BacktestEngine(
            strategy_class,
            data,
            cash=initial_capital,
            commission=commission,
            ticker=ticker,
        )

        result = engine.run()
        backtest_obj = engine.get_backtest_object()

        # Create output directory if it doesn't exist
        os.makedirs("reports_output", exist_ok=True)

        # Generate filename based on ticker, strategy and interval
        output_path = (
            f"reports_output/{ticker}_{strategy_name}_{interval}_backtest.html"
        )

        logger.info(f"Plotting best combination for {ticker} to {output_path}")

        # Create plot with specified parameters
        html = backtest_obj.plot(
            open_browser=False,
            plot_return=True,
            plot_drawdown=True,
            filename=output_path,
            resample=resample,
        )

        print(f"  🌐 Plot for best combination saved to: {output_path}")
        logger.info(f"Plot for best combination saved to: {output_path}")

        # Open in browser
        webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)
        return True

    except Exception as e:
        logger.error(f"Error plotting best combination for {ticker}: {e}")
        print(f"  ❌ Error plotting best combination: {e}")
        import traceback

        logger.error(traceback.format_exc())
        return False
