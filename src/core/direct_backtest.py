"""
Direct Backtesting Library Integration
Direct backtesting using the backtesting library.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from backtesting import Backtest

from .backtest_engine import create_backtesting_strategy_adapter
from .data_manager import UnifiedDataManager
from .strategy import StrategyFactory


def run_direct_backtest(
    symbol: str,
    strategy_name: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    initial_capital: float = 10000.0,
    commission: float = 0.001,
) -> dict[str, Any]:
    """
    Run backtest using backtesting library directly.
    Returns ground truth results without wrapper complexity.
    """
    logger = logging.getLogger(__name__)

    try:
        # Get data
        data_manager = UnifiedDataManager()
        data = data_manager.get_data(symbol, start_date, end_date, timeframe)

        if data is None or data.empty:
            return {
                "symbol": symbol,
                "strategy": strategy_name,
                "timeframe": timeframe,
                "error": "No data available",
                "metrics": {},
                "trades": None,
                "backtest_object": None,
            }

        # Prepare data for backtesting library
        bt_data = data.rename(
            columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )[["Open", "High", "Low", "Close", "Volume"]]

        # Create strategy
        strategy = StrategyFactory.create_strategy(strategy_name)
        StrategyClass = create_backtesting_strategy_adapter(strategy)

        # Run backtest with backtesting library
        bt = Backtest(
            bt_data,
            StrategyClass,
            cash=initial_capital,
            commission=commission,
            finalize_trades=True,  # Ensure all trades are captured
        )

        result = bt.run()

        # Extract real metrics directly from backtesting library
        metrics = {
            "total_return": float(result.get("Return [%]", 0.0)),
            "sharpe_ratio": (
                float(result.get("Sharpe Ratio", 0.0))
                if not pd.isna(result.get("Sharpe Ratio", 0.0))
                else 0.0
            ),
            "sortino_ratio": (
                float(result.get("Sortino Ratio", 0.0))
                if not pd.isna(result.get("Sortino Ratio", 0.0))
                else 0.0
            ),
            "calmar_ratio": (
                float(result.get("Calmar Ratio", 0.0))
                if not pd.isna(result.get("Calmar Ratio", 0.0))
                else 0.0
            ),
            "max_drawdown": abs(float(result.get("Max. Drawdown [%]", 0.0))),
            "volatility": (
                float(result.get("Volatility [%]", 0.0))
                if not pd.isna(result.get("Volatility [%]", 0.0))
                else 0.0
            ),
            "num_trades": int(result.get("# Trades", 0)),
            "win_rate": (
                float(result.get("Win Rate [%]", 0.0))
                if not pd.isna(result.get("Win Rate [%]", 0.0))
                else 0.0
            ),
            "profit_factor": (
                float(result.get("Profit Factor", 1.0))
                if not pd.isna(result.get("Profit Factor", 1.0))
                else 1.0
            ),
            "exposure_time": float(result.get("Exposure Time [%]", 0.0)),
            "start_value": float(initial_capital),  # Use known initial capital
            "end_value": float(result.get("Equity Final [$]", initial_capital)),
        }

        # Extract trades if available
        trades = None
        if hasattr(result, "_trades") and not result._trades.empty:
            trades = result._trades.copy()

        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "error": None,
            "metrics": metrics,
            "trades": trades,
            "backtest_object": bt,  # Include for plotting
            "bt_results": result,  # Include full results
        }

    except Exception as e:
        logger.error("Direct backtest failed for %s/%s: %s", symbol, strategy_name, e)
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "timeframe": timeframe,
            "error": str(e),
            "metrics": {},
            "trades": None,
            "backtest_object": None,
        }


def run_strategy_comparison(
    symbol: str,
    strategies: list[str],
    start_date: str,
    end_date: str,
    timeframe: str = "1d",
    initial_capital: float = 10000.0,
) -> dict[str, Any]:
    """
    Compare multiple strategies for a symbol using backtesting library.
    Returns complete analysis with rankings and plot data.
    """
    logger = logging.getLogger(__name__)
    logger.info(
        "Running strategy comparison for %s: %d strategies", symbol, len(strategies)
    )

    results = []
    best_result = None
    best_sortino = -999

    for strategy_name in strategies:
        result = run_direct_backtest(
            symbol, strategy_name, start_date, end_date, timeframe, initial_capital
        )

        results.append(result)

        # Track best strategy for plotting
        if (
            not result["error"]
            and result["metrics"].get("sortino_ratio", 0) > best_sortino
        ):
            best_sortino = result["metrics"]["sortino_ratio"]
            best_result = result

    # Sort by Sortino ratio
    results.sort(key=lambda x: x["metrics"].get("sortino_ratio", 0), reverse=True)

    # Add rankings
    for i, result in enumerate(results):
        result["rank"] = i + 1

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "results": results,
        "best_strategy": best_result,
        "total_strategies": len(strategies),
        "successful_strategies": len(
            [
                r
                for r in results
                if not r["error"] and r["metrics"].get("num_trades", 0) > 0
            ]
        ),
        "date_range": f"{start_date} to {end_date}",
    }
