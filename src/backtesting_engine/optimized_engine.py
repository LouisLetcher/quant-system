"""
Optimized backtesting engine for handling thousands of assets efficiently.
Supports parallel processing, memory optimization, and incremental backtesting.
"""

from __future__ import annotations

import asyncio  
import concurrent.futures
import gc
import logging
import multiprocessing as mp
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
import warnings

import numpy as np
import pandas as pd
from numba import jit, prange

from src.backtesting_engine.engine import BacktestingEngine
from src.data_scraper.multi_source_manager import MultiSourceDataManager
from src.data_scraper.advanced_cache import advanced_cache
from src.backtesting_engine.strategies.base_strategy import BaseStrategy

warnings.filterwarnings('ignore')


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


@dataclass
class BacktestResult:
    """Standardized backtest result structure."""
    symbol: str
    strategy: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    equity_curve: Optional[pd.DataFrame] = None
    trades: Optional[pd.DataFrame] = None
    start_date: str = None
    end_date: str = None
    duration_seconds: float = 0
    data_points: int = 0
    error: Optional[str] = None


class OptimizedBacktestEngine:
    """
    High-performance backtesting engine optimized for thousands of assets.
    Features:
    - Parallel processing with configurable workers
    - Memory-efficient data handling
    - Intelligent caching of results
    - Incremental backtesting for new data
    - Batch processing with progress tracking
    """
    
    def __init__(self, data_manager: MultiSourceDataManager = None, 
                 max_workers: int = None, memory_limit_gb: float = 8.0):
        self.data_manager = data_manager or MultiSourceDataManager()
        self.max_workers = max_workers or min(mp.cpu_count(), 8)
        self.memory_limit_bytes = int(memory_limit_gb * 1024**3)
        
        self.logger = logging.getLogger(__name__)
        self.stats = {
            'backtests_run': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0,
            'total_time': 0
        }
    
    def run_batch_backtests(self, config: BacktestConfig) -> List[BacktestResult]:
        """
        Run backtests for multiple symbols and strategies in parallel.
        
        Args:
            config: Backtest configuration
            
        Returns:
            List of backtest results
        """
        start_time = time.time()
        self.logger.info(f"Starting batch backtest: {len(config.symbols)} symbols, "
                        f"{len(config.strategies)} strategies")
        
        # Generate all combinations
        tasks = []
        for symbol in config.symbols:
            for strategy in config.strategies:
                tasks.append((symbol, strategy, config))
        
        self.logger.info(f"Total tasks: {len(tasks)}")
        
        # Process in batches to manage memory
        batch_size = self._calculate_batch_size(len(config.symbols), config.memory_limit_gb)
        results = []
        
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            self.logger.info(f"Processing batch {i//batch_size + 1}/{(len(tasks)-1)//batch_size + 1}")
            
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Force garbage collection between batches
            gc.collect()
        
        self.stats['total_time'] = time.time() - start_time
        self._log_stats()
        
        return results
    
    def run_incremental_backtest(self, symbol: str, strategy: str, 
                                config: BacktestConfig, 
                                last_update: datetime = None) -> Optional[BacktestResult]:
        """
        Run incremental backtest - only process new data since last run.
        
        Args:
            symbol: Symbol to backtest
            strategy: Strategy name
            config: Backtest configuration
            last_update: Last update timestamp (auto-detect if None)
            
        Returns:
            Backtest result or None if no new data
        """
        # Check if we have cached results
        strategy_params = self._get_default_strategy_params(strategy)
        cached_result = advanced_cache.get_backtest_result(
            symbol, strategy, strategy_params, config.interval
        )
        
        if cached_result and not last_update:
            self.logger.info(f"Using cached result for {symbol}/{strategy}")
            self.stats['cache_hits'] += 1
            return self._dict_to_backtest_result(cached_result)
        
        # Get data and check if we need to update
        data = self.data_manager.get_data(symbol, config.start_date, config.end_date, 
                                         config.interval, config.use_cache)
        
        if data is None or data.empty:
            return BacktestResult(
                symbol=symbol, strategy=strategy, parameters=strategy_params,
                metrics={}, error="No data available"
            )
        
        # Check if we have new data since last cached result
        if cached_result and last_update:
            last_data_point = pd.to_datetime(cached_result.get('end_date', config.start_date))
            if data.index[-1] <= last_data_point:
                self.logger.info(f"No new data for {symbol}/{strategy}")
                return self._dict_to_backtest_result(cached_result)
        
        # Run backtest
        return self._run_single_backtest(symbol, strategy, config, data)
    
    def optimize_strategy(self, symbol: str, strategy_name: str, 
                         param_ranges: Dict[str, List], config: BacktestConfig,
                         optimization_metric: str = 'total_return') -> Dict[str, Any]:
        """
        Optimize strategy parameters for a symbol.
        
        Args:
            symbol: Symbol to optimize
            strategy_name: Strategy name
            param_ranges: Dictionary of parameter ranges to test
            config: Base backtest configuration
            optimization_metric: Metric to optimize
            
        Returns:
            Optimization results
        """
        # Check cache first
        optimization_config = {
            'param_ranges': param_ranges,
            'metric': optimization_metric,
            'start_date': config.start_date,
            'end_date': config.end_date,
            'interval': config.interval
        }
        
        cached_result = advanced_cache.get_optimization_result(
            symbol, strategy_name, optimization_config, config.interval
        )
        
        if cached_result:
            self.logger.info(f"Using cached optimization for {symbol}/{strategy_name}")
            return cached_result
        
        start_time = time.time()
        
        # Generate parameter combinations
        param_combinations = self._generate_param_combinations(param_ranges)
        self.logger.info(f"Optimizing {len(param_combinations)} parameter combinations for {symbol}/{strategy_name}")
        
        # Get data once
        data = self.data_manager.get_data(symbol, config.start_date, config.end_date,
                                         config.interval, config.use_cache)
        
        if data is None or data.empty:
            return {'error': 'No data available for optimization'}
        
        # Run optimization
        optimization_tasks = []
        for params in param_combinations:
            task_config = BacktestConfig(
                symbols=[symbol],
                strategies=[strategy_name],
                start_date=config.start_date,
                end_date=config.end_date,
                initial_capital=config.initial_capital,
                interval=config.interval,
                commission=config.commission,
                use_cache=False,  # Don't cache individual optimization runs
                save_trades=False,
                save_equity_curve=False
            )
            optimization_tasks.append((symbol, strategy_name, params, task_config, data))
        
        # Process optimization tasks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._run_optimization_task, optimization_tasks))
        
        # Find best parameters
        valid_results = [r for r in results if r.error is None and optimization_metric in r.metrics]
        
        if not valid_results:
            return {'error': 'No valid optimization results'}
        
        best_result = max(valid_results, key=lambda x: x.metrics[optimization_metric])
        
        optimization_result = {
            'best_parameters': best_result.parameters,
            'best_metrics': best_result.metrics,
            'optimization_metric': optimization_metric,
            'total_combinations': len(param_combinations),
            'valid_results': len(valid_results),
            'optimization_time': time.time() - start_time,
            'all_results': [asdict(r) for r in results]
        }
        
        # Cache result
        advanced_cache.cache_optimization_result(
            symbol, strategy_name, optimization_config, optimization_result, config.interval
        )
        
        return optimization_result
    
    def _process_batch(self, batch: List[Tuple]) -> List[BacktestResult]:
        """Process a batch of backtest tasks in parallel."""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self._run_backtest_task, task): task for task in batch}
            
            results = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    self.stats['backtests_run'] += 1
                except Exception as e:
                    task = futures[future]
                    self.logger.error(f"Backtest failed for {task[0]}/{task[1]}: {e}")
                    self.stats['errors'] += 1
                    results.append(BacktestResult(
                        symbol=task[0], strategy=task[1], parameters={},
                        metrics={}, error=str(e)
                    ))
            
            return results
    
    def _run_backtest_task(self, task: Tuple) -> BacktestResult:
        """Run a single backtest task (used in multiprocessing)."""
        symbol, strategy, config = task
        
        # Check cache first
        strategy_params = self._get_default_strategy_params(strategy)
        cached_result = advanced_cache.get_backtest_result(
            symbol, strategy, strategy_params, config.interval
        )
        
        if cached_result and config.use_cache:
            return self._dict_to_backtest_result(cached_result)
        
        # Get data
        data_manager = MultiSourceDataManager()  # Create new instance for process
        data = data_manager.get_data(symbol, config.start_date, config.end_date,
                                   config.interval, config.use_cache)
        
        if data is None or data.empty:
            return BacktestResult(
                symbol=symbol, strategy=strategy, parameters=strategy_params,
                metrics={}, error="No data available"
            )
        
        return self._run_single_backtest(symbol, strategy, config, data)
    
    def _run_optimization_task(self, task: Tuple) -> BacktestResult:
        """Run a single optimization task."""
        symbol, strategy, params, config, data = task
        return self._run_single_backtest(symbol, strategy, config, data, params)
    
    def _run_single_backtest(self, symbol: str, strategy: str, config: BacktestConfig,
                           data: pd.DataFrame, custom_params: Dict = None) -> BacktestResult:
        """Run backtest for a single symbol/strategy combination."""
        start_time = time.time()
        
        try:
            # Get strategy parameters
            strategy_params = custom_params or self._get_default_strategy_params(strategy)
            
            # Initialize backtesting engine
            engine = BacktestingEngine(
                data=data,
                initial_capital=config.initial_capital,
                commission=config.commission
            )
            
            # Get and initialize strategy
            strategy_class = self._get_strategy_class(strategy)
            if not strategy_class:
                return BacktestResult(
                    symbol=symbol, strategy=strategy, parameters=strategy_params,
                    metrics={}, error=f"Strategy {strategy} not found"
                )
            
            strategy_instance = strategy_class(**strategy_params)
            
            # Run backtest
            result = engine.run_backtest(strategy_instance)
            
            # Extract metrics
            metrics = self._extract_metrics(result)
            
            # Prepare result
            backtest_result = BacktestResult(
                symbol=symbol,
                strategy=strategy,
                parameters=strategy_params,
                metrics=metrics,
                start_date=config.start_date,
                end_date=config.end_date,
                duration_seconds=time.time() - start_time,
                data_points=len(data)
            )
            
            # Add optional data
            if config.save_equity_curve and hasattr(result, '_equity_curve'):
                backtest_result.equity_curve = result._equity_curve
            
            if config.save_trades and hasattr(result, '_trades'):
                backtest_result.trades = result._trades
            
            # Cache result if not using custom parameters
            if not custom_params and config.use_cache:
                advanced_cache.cache_backtest_result(
                    symbol, strategy, strategy_params, asdict(backtest_result), config.interval
                )
            
            return backtest_result
            
        except Exception as e:
            self.logger.error(f"Backtest failed for {symbol}/{strategy}: {e}")
            return BacktestResult(
                symbol=symbol, strategy=strategy, 
                parameters=custom_params or self._get_default_strategy_params(strategy),
                metrics={}, error=str(e),
                duration_seconds=time.time() - start_time
            )
    
    def _calculate_batch_size(self, num_symbols: int, memory_limit_gb: float) -> int:
        """Calculate optimal batch size based on memory constraints."""
        # Estimate memory usage per symbol (rough approximation)
        estimated_memory_per_symbol_mb = 50  # MB
        available_memory_mb = memory_limit_gb * 1024 * 0.8  # Use 80% of limit
        
        max_batch_size = int(available_memory_mb / estimated_memory_per_symbol_mb)
        return min(max_batch_size, num_symbols, 100)  # Cap at 100 for manageability
    
    def _get_strategy_class(self, strategy_name: str) -> Optional[type]:
        """Get strategy class by name."""
        # This would be implemented based on your strategy registry
        # For now, return None as placeholder
        strategy_map = {
            # Add your strategy mappings here
            # 'rsi': RSIStrategy,
            # 'macd': MACDStrategy,
            # etc.
        }
        return strategy_map.get(strategy_name.lower())
    
    def _get_default_strategy_params(self, strategy_name: str) -> Dict[str, Any]:
        """Get default parameters for a strategy."""
        # This would return default parameters for each strategy
        default_params = {
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'macd': {'fast': 12, 'slow': 26, 'signal': 9},
            'bollinger_bands': {'period': 20, 'deviation': 2},
            # Add more default parameters
        }
        return default_params.get(strategy_name.lower(), {})
    
    def _generate_param_combinations(self, param_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """Generate all combinations of parameters."""
        import itertools
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        for combination in itertools.product(*values):
            combinations.append(dict(zip(keys, combination)))
        
        return combinations
    
    def _extract_metrics(self, backtest_result) -> Dict[str, float]:
        """Extract key metrics from backtest result."""
        metrics = {}
        
        # Extract standard metrics (adapt based on your BacktestingEngine output)
        try:
            if hasattr(backtest_result, 'stats'):
                stats = backtest_result.stats
                metrics.update({
                    'total_return': getattr(stats, 'Return [%]', 0.0),
                    'sharpe_ratio': getattr(stats, 'Sharpe Ratio', 0.0),
                    'max_drawdown': getattr(stats, 'Max. Drawdown [%]', 0.0),
                    'win_rate': getattr(stats, 'Win Rate [%]', 0.0),
                    'profit_factor': getattr(stats, 'Profit Factor', 0.0),
                    'num_trades': getattr(stats, '# Trades', 0)
                })
            elif isinstance(backtest_result, dict):
                metrics.update({
                    'total_return': backtest_result.get('Return [%]', 0.0),
                    'sharpe_ratio': backtest_result.get('Sharpe Ratio', 0.0),
                    'max_drawdown': backtest_result.get('Max. Drawdown [%]', 0.0),
                    'win_rate': backtest_result.get('Win Rate [%]', 0.0),
                    'profit_factor': backtest_result.get('Profit Factor', 0.0),
                    'num_trades': backtest_result.get('# Trades', 0)
                })
        except Exception as e:
            self.logger.warning(f"Failed to extract metrics: {e}")
        
        return metrics
    
    def _dict_to_backtest_result(self, cached_dict: Dict) -> BacktestResult:
        """Convert cached dictionary to BacktestResult object."""
        return BacktestResult(
            symbol=cached_dict.get('symbol', ''),
            strategy=cached_dict.get('strategy', ''),
            parameters=cached_dict.get('parameters', {}),
            metrics=cached_dict.get('metrics', {}),
            start_date=cached_dict.get('start_date'),
            end_date=cached_dict.get('end_date'),
            duration_seconds=cached_dict.get('duration_seconds', 0),
            data_points=cached_dict.get('data_points', 0),
            error=cached_dict.get('error')
        )
    
    def _log_stats(self):
        """Log performance statistics."""
        self.logger.info(f"Batch backtest completed:")
        self.logger.info(f"  Total backtests: {self.stats['backtests_run']}")
        self.logger.info(f"  Cache hits: {self.stats['cache_hits']}")
        self.logger.info(f"  Cache misses: {self.stats['cache_misses']}")
        self.logger.info(f"  Errors: {self.stats['errors']}")
        self.logger.info(f"  Total time: {self.stats['total_time']:.2f}s")
        if self.stats['backtests_run'] > 0:
            self.logger.info(f"  Avg time per backtest: {self.stats['total_time']/self.stats['backtests_run']:.2f}s")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        return self.stats.copy()
    
    def clear_cache(self, symbol: str = None, strategy: str = None):
        """Clear cached results."""
        advanced_cache.clear_cache(cache_type='backtest', symbol=symbol, strategy=strategy)


@jit(nopython=True)
def fast_sma(prices: np.ndarray, window: int) -> np.ndarray:
    """Fast simple moving average calculation using Numba."""
    result = np.empty_like(prices)
    result[:window-1] = np.nan
    
    for i in prange(window-1, len(prices)):
        result[i] = np.mean(prices[i-window+1:i+1])
    
    return result


@jit(nopython=True) 
def fast_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """Fast RSI calculation using Numba."""
    deltas = np.diff(prices)
    gain = np.where(deltas > 0, deltas, 0)
    loss = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = np.empty_like(prices)
    avg_loss = np.empty_like(prices)
    rsi = np.empty_like(prices)
    
    avg_gain[:window] = np.nan
    avg_loss[:window] = np.nan
    rsi[:window] = np.nan
    
    # Initial values
    avg_gain[window] = np.mean(gain[:window])
    avg_loss[window] = np.mean(loss[:window])
    
    for i in prange(window+1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (window-1) + gain[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window-1) + loss[i-1]) / window
        
        if avg_loss[i] == 0:
            rsi[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
    
    return rsi


class FastIndicators:
    """Collection of fast indicator calculations for high-frequency backtesting."""
    
    @staticmethod
    @jit(nopython=True)
    def bollinger_bands(prices: np.ndarray, window: int = 20, 
                       num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast Bollinger Bands calculation."""
        sma = fast_sma(prices, window)
        
        std = np.empty_like(prices)
        std[:window-1] = np.nan
        
        for i in prange(window-1, len(prices)):
            std[i] = np.std(prices[i-window+1:i+1])
        
        upper = sma + (std * num_std)
        lower = sma - (std * num_std)
        
        return upper, sma, lower
    
    @staticmethod
    @jit(nopython=True)
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, 
             signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fast MACD calculation."""
        ema_fast = np.empty_like(prices)
        ema_slow = np.empty_like(prices)
        
        # Calculate EMAs
        alpha_fast = 2.0 / (fast + 1.0)
        alpha_slow = 2.0 / (slow + 1.0)
        
        ema_fast[0] = prices[0]
        ema_slow[0] = prices[0]
        
        for i in prange(1, len(prices)):
            ema_fast[i] = alpha_fast * prices[i] + (1 - alpha_fast) * ema_fast[i-1]
            ema_slow[i] = alpha_slow * prices[i] + (1 - alpha_slow) * ema_slow[i-1]
        
        macd_line = ema_fast - ema_slow
        signal_line = fast_sma(macd_line, signal)  # Simplified signal line
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
