"""
Simple Backtest Engine - Direct backtesting library integration
Simple, reliable backtesting approach using the backtesting library.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from backtesting import Backtest

from .backtest_engine import (
    BacktestConfig,
    BacktestResult,
    create_backtesting_strategy_adapter,
)
from .data_manager import UnifiedDataManager
from .strategy import StrategyFactory


class SimpleBacktestEngine:
    """
    Simplified backtest engine using backtesting library directly.
    Eliminates wrapper complexity and provides ground truth results.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_manager = UnifiedDataManager()

    def run_single_backtest(
        self, symbol: str, strategy_name: str, config: BacktestConfig
    ) -> BacktestResult:
        """Run a single backtest using backtesting library directly."""
        try:
            # Get data
            data = self.data_manager.get_data(
                symbol, config.start_date, config.end_date, config.interval
            )

            if data is None or data.empty:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy_name,
                    parameters={},
                    config=config,
                    metrics={},
                    error="No data available",
                )

            # Prepare data for backtesting library
            bt_data = self._prepare_data(data)

            # Create strategy
            strategy = StrategyFactory.create_strategy(strategy_name)
            StrategyClass = create_backtesting_strategy_adapter(strategy)

            # Run backtest with backtesting library
            bt = Backtest(
                bt_data,
                StrategyClass,
                cash=config.initial_capital,
                commission=config.commission,
                finalize_trades=True,  # Ensure all trades are captured
            )

            bt_results = bt.run()

            # Extract metrics directly from backtesting library
            metrics = self._extract_metrics(bt_results)

            # Extract trades if requested
            trades = None
            if config.save_trades and hasattr(bt_results, "_trades"):
                trades = (
                    bt_results._trades.copy() if not bt_results._trades.empty else None
                )

            return BacktestResult(
                symbol=symbol,
                strategy=strategy_name,
                parameters={},
                config=config,
                metrics=metrics,
                trades=trades,
                error=None,
            )

        except Exception as e:
            self.logger.error("Backtest failed for %s/%s: %s", symbol, strategy_name, e)
            return BacktestResult(
                symbol=symbol,
                strategy=strategy_name,
                parameters={},
                config=config,
                metrics={},
                error=str(e),
            )

    def run_batch_backtests(self, config: BacktestConfig) -> list[BacktestResult]:
        """Run multiple backtests."""
        results = []

        for symbol in config.symbols:
            for strategy in config.strategies:
                result = self.run_single_backtest(symbol, strategy, config)
                results.append(result)

        return results

    def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for backtesting library (uppercase OHLCV columns)."""
        if all(
            col in data.columns for col in ["open", "high", "low", "close", "volume"]
        ):
            return data.rename(
                columns={
                    "open": "Open",
                    "high": "High",
                    "low": "Low",
                    "close": "Close",
                    "volume": "Volume",
                }
            )[["Open", "High", "Low", "Close", "Volume"]]
        return data[["Open", "High", "Low", "Close", "Volume"]]

    def _extract_metrics(self, bt_results) -> dict[str, Any]:
        """Extract metrics directly from backtesting library results."""
        return {
            # Core performance metrics
            "total_return": float(bt_results.get("Return [%]", 0.0)),
            "sharpe_ratio": float(bt_results.get("Sharpe Ratio", 0.0))
            if not pd.isna(bt_results.get("Sharpe Ratio", 0.0))
            else 0.0,
            "sortino_ratio": float(bt_results.get("Sortino Ratio", 0.0))
            if not pd.isna(bt_results.get("Sortino Ratio", 0.0))
            else 0.0,
            "calmar_ratio": float(bt_results.get("Calmar Ratio", 0.0))
            if not pd.isna(bt_results.get("Calmar Ratio", 0.0))
            else 0.0,
            # Risk metrics
            "max_drawdown": abs(float(bt_results.get("Max. Drawdown [%]", 0.0))),
            "volatility": float(bt_results.get("Volatility [%]", 0.0))
            if not pd.isna(bt_results.get("Volatility [%]", 0.0))
            else 0.0,
            # Trade metrics
            "num_trades": int(bt_results.get("# Trades", 0)),
            "win_rate": float(bt_results.get("Win Rate [%]", 0.0))
            if not pd.isna(bt_results.get("Win Rate [%]", 0.0))
            else 0.0,
            "profit_factor": float(bt_results.get("Profit Factor", 1.0))
            if not pd.isna(bt_results.get("Profit Factor", 1.0))
            else 1.0,
            # Additional metrics
            "exposure_time": float(bt_results.get("Exposure Time [%]", 0.0)),
            "start_value": float(bt_results.get("Start", 0.0)),
            "end_value": float(bt_results.get("End", 0.0)),
        }
