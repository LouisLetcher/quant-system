"""
Unified Backtest Engine - Consolidates all backtesting functionality.
Supports single assets, portfolios, parallel processing, and optimization.
"""

from __future__ import annotations

import concurrent.futures
import gc
import logging
import multiprocessing as mp
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .cache_manager import UnifiedCacheManager
from .data_manager import UnifiedDataManager
from .result_analyzer import UnifiedResultAnalyzer

# from numba import jit  # Removed for compatibility


warnings.filterwarnings("ignore")


@dataclass
class BacktestConfig:
    """Configuration for backtest runs."""

    symbols: List[str]
    strategies: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 10000
    interval: str = "1d"
    commission: float = 0.001
    use_cache: bool = True
    save_trades: bool = False
    save_equity_curve: bool = False
    memory_limit_gb: float = 8.0
    max_workers: int = None
    asset_type: str = None  # 'stocks', 'crypto', 'forex', etc.
    futures_mode: bool = False  # For crypto futures
    leverage: float = 1.0  # For futures trading


@dataclass
class BacktestResult:
    """Standardized backtest result."""

    symbol: str
    strategy: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    config: BacktestConfig
    equity_curve: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    start_date: str = None
    end_date: str = None
    duration_seconds: float = 0
    data_points: int = 0
    error: Optional[str] = None
    source: Optional[str] = None


class UnifiedBacktestEngine:
    """
    Unified backtesting engine that consolidates all backtesting functionality.
    Supports single assets, portfolios, parallel processing, and various asset types.
    """

    def __init__(
        self,
        data_manager: UnifiedDataManager = None,
        cache_manager: UnifiedCacheManager = None,
        max_workers: int = None,
        memory_limit_gb: float = 8.0,
    ):
        self.data_manager = data_manager or UnifiedDataManager()
        self.cache_manager = cache_manager or UnifiedCacheManager()
        self.result_analyzer = UnifiedResultAnalyzer()

        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)

        self.logger = logging.getLogger(__name__)
        self.stats = {
            "backtests_run": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "errors": 0,
            "total_time": 0,
        }

    def run_backtest(
        self,
        symbol: str,
        strategy: str,
        config: BacktestConfig,
        custom_parameters: Dict[str, Any] = None,
    ) -> BacktestResult:
        """
        Run backtest for a single symbol/strategy combination.

        Args:
            symbol: Symbol to backtest
            strategy: Strategy name
            config: Backtest configuration
            custom_parameters: Custom strategy parameters

        Returns:
            BacktestResult object
        """
        start_time = time.time()

        try:
            # Get strategy parameters
            parameters = custom_parameters or self._get_default_parameters(strategy)

            # Check cache first
            if config.use_cache and not custom_parameters:
                cached_result = self.cache_manager.get_backtest_result(
                    symbol, strategy, parameters, config.interval
                )
                if cached_result:
                    self.stats["cache_hits"] += 1
                    self.logger.debug(f"Cache hit for {symbol}/{strategy}")
                    return self._dict_to_result(
                        cached_result, symbol, strategy, parameters, config
                    )

            self.stats["cache_misses"] += 1

            # Get market data
            data_kwargs = {}
            if config.futures_mode:
                data = self.data_manager.get_crypto_futures_data(
                    symbol,
                    config.start_date,
                    config.end_date,
                    config.interval,
                    config.use_cache,
                )
            else:
                data = self.data_manager.get_data(
                    symbol,
                    config.start_date,
                    config.end_date,
                    config.interval,
                    config.use_cache,
                    config.asset_type,
                )

            if data is None or data.empty:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters=parameters,
                    config=config,
                    metrics={},
                    error="No data available",
                )

            # Run backtest
            result = self._execute_backtest(symbol, strategy, data, parameters, config)

            # Cache result if not using custom parameters
            if config.use_cache and not custom_parameters and not result.error:
                self.cache_manager.cache_backtest_result(
                    symbol, strategy, parameters, asdict(result), config.interval
                )

            result.duration_seconds = time.time() - start_time
            result.data_points = len(data)
            self.stats["backtests_run"] += 1

            return result

        except Exception as e:
            self.stats["errors"] += 1
            self.logger.error(f"Backtest failed for {symbol}/{strategy}: {e}")
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=custom_parameters or {},
                config=config,
                metrics={},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def run_batch_backtests(self, config: BacktestConfig) -> List[BacktestResult]:
        """
        Run backtests for multiple symbols and strategies in parallel.

        Args:
            config: Backtest configuration

        Returns:
            List of backtest results
        """
        start_time = time.time()
        self.logger.info(
            f"Starting batch backtest: {len(config.symbols)} symbols, "
            f"{len(config.strategies)} strategies"
        )

        # Generate all symbol/strategy combinations
        combinations = [
            (symbol, strategy)
            for symbol in config.symbols
            for strategy in config.strategies
        ]

        self.logger.info(f"Total combinations: {len(combinations)}")

        # Process in batches to manage memory
        batch_size = self._calculate_batch_size(
            len(config.symbols), config.memory_limit_gb
        )
        results = []

        for i in range(0, len(combinations), batch_size):
            batch = combinations[i : i + batch_size]
            self.logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(combinations)-1)//batch_size + 1}"
            )

            batch_results = self._process_batch(batch, config)
            results.extend(batch_results)

            # Force garbage collection between batches
            gc.collect()

        self.stats["total_time"] = time.time() - start_time
        self._log_stats()

        return results

    def run_portfolio_backtest(
        self, config: BacktestConfig, weights: Dict[str, float] = None
    ) -> BacktestResult:
        """
        Run portfolio backtest with multiple assets.

        Args:
            config: Backtest configuration
            weights: Asset weights (if None, equal weights used)

        Returns:
            Portfolio backtest result
        """
        start_time = time.time()

        if not config.strategies or len(config.strategies) != 1:
            raise ValueError("Portfolio backtest requires exactly one strategy")

        strategy = config.strategies[0]

        try:
            # Get data for all symbols
            all_data = self.data_manager.get_batch_data(
                config.symbols,
                config.start_date,
                config.end_date,
                config.interval,
                config.use_cache,
                config.asset_type,
            )

            if not all_data:
                return BacktestResult(
                    symbol="PORTFOLIO",
                    strategy=strategy,
                    parameters={},
                    config=config,
                    metrics={},
                    error="No data available for any symbol",
                )

            # Calculate equal weights if not provided
            if not weights:
                weights = {symbol: 1.0 / len(all_data) for symbol in all_data.keys()}

            # Normalize weights
            total_weight = sum(weights.values())
            weights = {k: v / total_weight for k, v in weights.items()}

            # Run portfolio backtest
            portfolio_result = self._execute_portfolio_backtest(
                all_data, strategy, weights, config
            )

            portfolio_result.duration_seconds = time.time() - start_time
            return portfolio_result

        except Exception as e:
            self.logger.error(f"Portfolio backtest failed: {e}")
            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters={},
                config=config,
                metrics={},
                error=str(e),
                duration_seconds=time.time() - start_time,
            )

    def run_incremental_backtest(
        self,
        symbol: str,
        strategy: str,
        config: BacktestConfig,
        last_update: datetime = None,
    ) -> Optional[BacktestResult]:
        """
        Run incremental backtest - only process new data since last run.

        Args:
            symbol: Symbol to backtest
            strategy: Strategy name
            config: Backtest configuration
            last_update: Last update timestamp

        Returns:
            BacktestResult or None if no new data
        """
        # Check if we have cached results
        parameters = self._get_default_parameters(strategy)
        cached_result = self.cache_manager.get_backtest_result(
            symbol, strategy, parameters, config.interval
        )

        if cached_result and not last_update:
            self.logger.info(f"Using cached result for {symbol}/{strategy}")
            return self._dict_to_result(
                cached_result, symbol, strategy, parameters, config
            )

        # Get data and check if we need to update
        data = self.data_manager.get_data(
            symbol,
            config.start_date,
            config.end_date,
            config.interval,
            config.use_cache,
            config.asset_type,
        )

        if data is None or data.empty:
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics={},
                error="No data available",
            )

        # Check if we have new data since last cached result
        if cached_result and last_update:
            last_data_point = pd.to_datetime(
                cached_result.get("end_date", config.start_date)
            )
            if data.index[-1] <= last_data_point:
                self.logger.info(f"No new data for {symbol}/{strategy}")
                return self._dict_to_result(
                    cached_result, symbol, strategy, parameters, config
                )

        # Run backtest
        return self.run_backtest(symbol, strategy, config)

    def _execute_backtest(
        self,
        symbol: str,
        strategy: str,
        data: pd.DataFrame,
        parameters: Dict[str, Any],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Execute the actual backtest logic."""
        try:
            # Get strategy class
            strategy_class = self._get_strategy_class(strategy)
            if not strategy_class:
                return BacktestResult(
                    symbol=symbol,
                    strategy=strategy,
                    parameters=parameters,
                    config=config,
                    metrics={},
                    error=f"Strategy {strategy} not found",
                )

            # Initialize strategy
            strategy_instance = strategy_class(**parameters)

            # Prepare data with technical indicators
            prepared_data = self._prepare_data_with_indicators(data, strategy_instance)

            # Run backtest simulation
            result = self._simulate_trading(prepared_data, strategy_instance, config)

            # Analyze results
            metrics = self.result_analyzer.calculate_metrics(
                result, config.initial_capital
            )

            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics=metrics,
                equity_curve=(
                    result.get("equity_curve") if config.save_equity_curve else None
                ),
                trades=result.get("trades") if config.save_trades else None,
                start_date=config.start_date,
                end_date=config.end_date,
            )

        except Exception as e:
            return BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=parameters,
                config=config,
                metrics={},
                error=str(e),
            )

    def _execute_portfolio_backtest(
        self,
        data_dict: Dict[str, pd.DataFrame],
        strategy: str,
        weights: Dict[str, float],
        config: BacktestConfig,
    ) -> BacktestResult:
        """Execute portfolio backtest."""
        try:
            # Align all data to common date range
            aligned_data = self._align_portfolio_data(data_dict)

            if aligned_data.empty:
                return BacktestResult(
                    symbol="PORTFOLIO",
                    strategy=strategy,
                    parameters=weights,
                    config=config,
                    metrics={},
                    error="No aligned data for portfolio",
                )

            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(aligned_data, weights)

            # Create portfolio equity curve
            initial_capital = config.initial_capital
            equity_curve = (1 + portfolio_returns).cumprod() * initial_capital

            # Calculate portfolio metrics
            portfolio_data = {
                "returns": portfolio_returns,
                "equity_curve": equity_curve,
                "weights": weights,
            }

            metrics = self.result_analyzer.calculate_portfolio_metrics(
                portfolio_data, initial_capital
            )

            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters=weights,
                config=config,
                metrics=metrics,
                equity_curve=(
                    equity_curve.to_frame("equity")
                    if config.save_equity_curve
                    else None
                ),
            )

        except Exception as e:
            return BacktestResult(
                symbol="PORTFOLIO",
                strategy=strategy,
                parameters=weights,
                config=config,
                metrics={},
                error=str(e),
            )

    def _process_batch(
        self, batch: List[Tuple[str, str]], config: BacktestConfig
    ) -> List[BacktestResult]:
        """Process batch of symbol/strategy combinations."""
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = {
                executor.submit(
                    self._run_single_backtest_task, symbol, strategy, config
                ): (symbol, strategy)
                for symbol, strategy in batch
            }

            results = []
            for future in concurrent.futures.as_completed(futures):
                symbol, strategy = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.error(
                        f"Batch backtest failed for {symbol}/{strategy}: {e}"
                    )
                    self.stats["errors"] += 1
                    results.append(
                        BacktestResult(
                            symbol=symbol,
                            strategy=strategy,
                            parameters={},
                            config=config,
                            metrics={},
                            error=str(e),
                        )
                    )

            return results

    def _run_single_backtest_task(
        self, symbol: str, strategy: str, config: BacktestConfig
    ) -> BacktestResult:
        """Task function for multiprocessing."""
        # Create new instances for this process
        data_manager = UnifiedDataManager()
        cache_manager = UnifiedCacheManager()

        # Create temporary engine for this process
        temp_engine = UnifiedBacktestEngine(data_manager, cache_manager, max_workers=1)
        return temp_engine.run_backtest(symbol, strategy, config)

    def _prepare_data_with_indicators(
        self, data: pd.DataFrame, strategy_instance
    ) -> pd.DataFrame:
        """Prepare data with technical indicators required by strategy."""
        prepared_data = data.copy()

        # Add basic indicators that most strategies need
        prepared_data = self._add_basic_indicators(prepared_data)

        # Add strategy-specific indicators
        if hasattr(strategy_instance, "add_indicators"):
            prepared_data = strategy_instance.add_indicators(prepared_data)

        return prepared_data

    def _add_basic_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add basic technical indicators."""
        df = data.copy()

        # Simple moving averages
        for period in [10, 20, 50]:
            df[f"sma_{period}"] = df["close"].rolling(period).mean()

        # RSI
        df["rsi_14"] = self._calculate_rsi(df["close"].values, 14)

        # MACD
        macd_line, signal_line, histogram = self._calculate_macd(df["close"].values)
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_histogram"] = histogram

        # Bollinger Bands
        sma_20 = df["close"].rolling(20).mean()
        std_20 = df["close"].rolling(20).std()
        df["bb_upper"] = sma_20 + (std_20 * 2)
        df["bb_lower"] = sma_20 - (std_20 * 2)
        df["bb_middle"] = sma_20

        return df

    def _simulate_trading(
        self, data: pd.DataFrame, strategy_instance, config: BacktestConfig
    ) -> Dict[str, Any]:
        """Simulate trading based on strategy signals."""
        trades = []
        equity_curve = []

        capital = config.initial_capital
        position = 0
        position_size = 0

        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Get strategy signal
            signal = self._get_strategy_signal(strategy_instance, data.iloc[: i + 1])

            # Execute trades based on signal
            if signal == 1 and position <= 0:  # Buy signal
                if position < 0:  # Close short position
                    pnl = (position_size * row["close"] - position_size * position) * -1
                    capital += pnl
                    trades.append(
                        {
                            "timestamp": timestamp,
                            "action": "cover",
                            "price": row["close"],
                            "size": abs(position_size),
                            "pnl": pnl,
                        }
                    )

                # Open long position
                position_size = (capital * 0.95) / row["close"]  # 95% of capital
                position = row["close"]
                capital -= position_size * row["close"] + (
                    position_size * row["close"] * config.commission
                )

                trades.append(
                    {
                        "timestamp": timestamp,
                        "action": "buy",
                        "price": row["close"],
                        "size": position_size,
                        "pnl": 0,
                    }
                )

            elif signal == -1 and position >= 0:  # Sell signal
                if position > 0:  # Close long position
                    pnl = position_size * (row["close"] - position)
                    capital += pnl + (position_size * row["close"])
                    trades.append(
                        {
                            "timestamp": timestamp,
                            "action": "sell",
                            "price": row["close"],
                            "size": position_size,
                            "pnl": pnl,
                        }
                    )
                    position = 0
                    position_size = 0

            # Calculate current portfolio value
            if position > 0:
                portfolio_value = capital + (position_size * row["close"])
            elif position < 0:
                portfolio_value = capital - (position_size * (row["close"] - position))
            else:
                portfolio_value = capital

            equity_curve.append({"timestamp": timestamp, "equity": portfolio_value})

        return {
            "trades": pd.DataFrame(trades) if trades else pd.DataFrame(),
            "equity_curve": pd.DataFrame(equity_curve),
            "final_capital": (
                equity_curve[-1]["equity"] if equity_curve else config.initial_capital
            ),
        }

    def _get_strategy_signal(self, strategy_instance, data: pd.DataFrame) -> int:
        """Get trading signal from strategy."""
        if hasattr(strategy_instance, "generate_signal"):
            return strategy_instance.generate_signal(data)
        else:
            # Fallback simple strategy
            if len(data) < 20:
                return 0

            current_price = data["close"].iloc[-1]
            sma_20 = data["close"].rolling(20).mean().iloc[-1]

            if current_price > sma_20:
                return 1  # Buy
            elif current_price < sma_20:
                return -1  # Sell
            else:
                return 0  # Hold

    def _align_portfolio_data(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Align multiple asset data to common date range."""
        if not data_dict:
            return pd.DataFrame()

        # Find common date range
        all_dates = None
        for symbol, data in data_dict.items():
            if all_dates is None:
                all_dates = set(data.index)
            else:
                all_dates = all_dates.intersection(set(data.index))

        if not all_dates:
            return pd.DataFrame()

        # Create aligned dataframe
        common_dates = sorted(list(all_dates))
        aligned_data = pd.DataFrame(index=common_dates)

        for symbol, data in data_dict.items():
            aligned_data[f"{symbol}_close"] = data.loc[common_dates, "close"]

        return aligned_data.dropna()

    def _calculate_portfolio_returns(
        self, aligned_data: pd.DataFrame, weights: Dict[str, float]
    ) -> pd.Series:
        """Calculate portfolio returns."""
        returns = pd.Series(index=aligned_data.index, dtype=float)

        for i in range(1, len(aligned_data)):
            portfolio_return = 0
            for symbol, weight in weights.items():
                col_name = f"{symbol}_close"
                if col_name in aligned_data.columns:
                    asset_return = (
                        aligned_data[col_name].iloc[i]
                        / aligned_data[col_name].iloc[i - 1]
                    ) - 1
                    portfolio_return += weight * asset_return

            returns.iloc[i] = portfolio_return

        return returns.fillna(0)

    @staticmethod
    # @jit(nopython=True)  # Removed for compatibility
    def _calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Fast RSI calculation using Numba."""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gains = np.full_like(prices, np.nan)
        avg_losses = np.full_like(prices, np.nan)
        rsi = np.full_like(prices, np.nan)

        if len(gains) >= period:
            avg_gains[period] = np.mean(gains[:period])
            avg_losses[period] = np.mean(losses[:period])

            for i in range(period + 1, len(prices)):
                avg_gains[i] = (avg_gains[i - 1] * (period - 1) + gains[i - 1]) / period
                avg_losses[i] = (
                    avg_losses[i - 1] * (period - 1) + losses[i - 1]
                ) / period

                if avg_losses[i] == 0:
                    rsi[i] = 100
                else:
                    rs = avg_gains[i] / avg_losses[i]
                    rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    # @jit(nopython=True)  # Removed for compatibility
    def _calculate_macd(
        prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast MACD calculation using Numba."""
        ema_fast = np.full_like(prices, np.nan)
        ema_slow = np.full_like(prices, np.nan)

        # Calculate EMAs
        alpha_fast = 2.0 / (fast + 1.0)
        alpha_slow = 2.0 / (slow + 1.0)

        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]

        for i in range(1, len(prices)):
            ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i - 1]
            ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i - 1]

        macd_line = ema_fast - ema_slow

        # Calculate signal line (EMA of MACD)
        signal_line = np.full_like(prices, np.nan)
        alpha_signal = 2.0 / (signal + 1.0)

        # Start signal line calculation after we have enough MACD data
        signal_start = max(fast, slow)
        if len(macd_line) > signal_start:
            signal_line[signal_start] = macd_line[signal_start]
            for i in range(signal_start + 1, len(prices)):
                signal_line[i] = (
                    alpha_signal * macd_line[i]
                    + (1 - alpha_signal) * signal_line[i - 1]
                )

        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_batch_size(self, num_symbols: int, memory_limit_gb: float) -> int:
        """Calculate optimal batch size based on memory constraints."""
        estimated_memory_per_symbol_mb = 50
        available_memory_mb = memory_limit_gb * 1024 * 0.8

        max_batch_size = int(available_memory_mb / estimated_memory_per_symbol_mb)
        return min(max_batch_size, num_symbols, 100)

    def _get_strategy_class(self, strategy_name: str) -> Optional[type]:
        """Get strategy class by name."""
        # This would be implemented based on your strategy registry
        # For now, return a placeholder
        return None

    def _get_default_parameters(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy."""
        default_params = {
            "rsi": {"period": 14, "overbought": 70, "oversold": 30},
            "macd": {"fast": 12, "slow": 26, "signal": 9},
            "bollinger_bands": {"period": 20, "deviation": 2},
            "sma_crossover": {"fast_period": 10, "slow_period": 20},
        }
        return default_params.get(strategy_name.lower(), {})

    def _dict_to_result(
        self,
        cached_dict: Dict,
        symbol: str,
        strategy: str,
        parameters: Dict,
        config: BacktestConfig,
    ) -> BacktestResult:
        """Convert cached dictionary to BacktestResult object."""
        return BacktestResult(
            symbol=symbol,
            strategy=strategy,
            parameters=parameters,
            config=config,
            metrics=cached_dict.get("metrics", {}),
            start_date=cached_dict.get("start_date"),
            end_date=cached_dict.get("end_date"),
            duration_seconds=cached_dict.get("duration_seconds", 0),
            data_points=cached_dict.get("data_points", 0),
            error=cached_dict.get("error"),
        )

    def _log_stats(self):
        """Log performance statistics."""
        self.logger.info(f"Batch backtest completed:")
        self.logger.info(f"  Total backtests: {self.stats['backtests_run']}")
        self.logger.info(f"  Cache hits: {self.stats['cache_hits']}")
        self.logger.info(f"  Cache misses: {self.stats['cache_misses']}")
        self.logger.info(f"  Errors: {self.stats['errors']}")
        self.logger.info(f"  Total time: {self.stats['total_time']:.2f}s")
        if self.stats["backtests_run"] > 0:
            avg_time = self.stats["total_time"] / self.stats["backtests_run"]
            self.logger.info(f"  Avg time per backtest: {avg_time:.2f}s")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return self.stats.copy()

    def clear_cache(self, symbol: str = None, strategy: str = None):
        """Clear cached results."""
        self.cache_manager.clear_cache(cache_type="backtest", symbol=symbol)
