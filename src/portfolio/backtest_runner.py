from __future__ import annotations

import os
import webbrowser

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)


def backtest_all_strategies(
    ticker, period, metric, commission, initial_capital, plot=False, resample=None
):
    """Run a backtest for a single asset with all available strategies."""
    logger.info(f"Running all strategies backtest for {ticker}")

    # Remove default parameter values from the function definition to ensure they come from config
    strategies = StrategyFactory.get_available_strategies()
    logger.debug(f"Testing {len(strategies)} strategies")

    best_score = -float("inf")
    best_strategy = None
    all_results = {}
    best_backtest = None

    for strategy_name in strategies:
        logger.info(f"Testing {strategy_name} on {ticker}")
        print(f"  Testing {strategy_name}...")

        try:
            # Get the data and strategy
            data = load_asset_data(ticker, period)
            if data is None:
                continue

            strategy_class = StrategyFactory.get_strategy(strategy_name)

            # Run backtest
            result, backtest_obj = execute_backtest(
                strategy_class, data, initial_capital, commission, ticker
            )

            # Extract performance metric
            score = extract_score(result, metric)

            logger.info(
                f"{strategy_name} on {ticker}: {metric}={score}, trades={result.get('# Trades', 0)}"
            )

            all_results[strategy_name] = {
                "score": score,
                "results": result,
                "backtest_obj": backtest_obj,
            }

            print(f"    {strategy_name}: {metric.capitalize()} = {score}")

            if score > best_score:
                best_score = score
                best_strategy = strategy_name
                best_backtest = backtest_obj
                logger.info(
                    f"New best strategy for {ticker}: {strategy_name} with {metric}={score}"
                )

        except Exception as e:
            logger.error(f"Error testing {strategy_name} on {ticker}: {e}")
            import traceback

            logger.error(traceback.format_exc())
            print(f"    ❌ Error testing {strategy_name}: {e}")

    logger.info(f"Best strategy for {ticker}: {best_strategy} ({metric}: {best_score})")
    print(
        f"✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})"
    )

    # Plot the best strategy if requested
    if plot and best_backtest:
        plot_best_strategy(
            ticker, best_strategy, best_backtest, output_path=None, resample=resample
        )

    return {
        "strategies": all_results,
        "best_strategy": best_strategy,
        "best_score": best_score,
    }


def load_asset_data(ticker, period):
    """Load data for a specific ticker."""
    data = DataLoader.load_data(ticker, period=period)
    if data is None or data.empty:
        logger.warning(f"No data available for {ticker} with period {period}")
        return None

    logger.debug(f"Loaded {len(data)} data points for {ticker}")
    return data


def execute_backtest(strategy_class, data, initial_capital, commission, ticker):
    """Execute a backtest with the given strategy and data."""
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
    backtest_obj = engine.get_backtest_object()

    return result, backtest_obj


def extract_score(result, metric):
    """Extract the performance score based on the specified metric."""
    if metric == "profit_factor":
        score = result.get("Profit Factor", result.get("profit_factor", 0))
    elif metric == "sharpe" or metric == "sharpe_ratio":
        score = result.get("Sharpe Ratio", result.get("sharpe_ratio", 0))
    elif metric == "sortino" or metric == "sortino_ratio":
        score = result.get("Sortino Ratio", result.get("sortino_ratio", 0))
    elif metric == "return" or metric == "total_return":
        if isinstance(result.get("Return [%]", 0), (int, float)):
            score = result.get("Return [%]", 0)
        else:
            score = result.get("return_pct", 0)
    elif metric == "max_drawdown":
        # For max drawdown, we want lower values, so return negative
        score = -abs(result.get("Max Drawdown [%]", result.get("max_drawdown", 0)))
    else:
        score = result.get(metric, 0)

    return score


def plot_best_strategy(
    ticker, strategy_name, backtest_obj, output_path=None, resample=None
):
    """Plot the best strategy backtest results."""
    try:
        # Create output directory if it doesn't exist
        os.makedirs("reports_output", exist_ok=True)

        # Generate filename based on ticker and strategy
        if output_path is None:
            output_path = f"reports_output/{ticker}_{strategy_name}_backtest.html"

        logger.info(
            f"Plotting best strategy for {ticker}: {strategy_name} to {output_path}"
        )

        # Create plot with specified parameters
        html = backtest_obj.plot(
            open_browser=False,
            plot_return=True,
            plot_drawdown=True,
            filename=output_path,
            resample=resample,
        )

        print(f"🌐 Plot for best strategy saved to: {output_path}")
        logger.info(f"Plot for best strategy saved to: {output_path}")

        # Open in browser
        webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)
        return True
    except Exception as e:
        logger.error(f"Error plotting best strategy: {e}")
        import traceback

        logger.error(traceback.format_exc())
        print(f"❌ Error plotting best strategy: {e}")
        return False
