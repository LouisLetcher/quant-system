#!/usr/bin/env python3
"""
Comprehensive example demonstrating the optimized quant system capabilities.
This script shows how to use the advanced features for large-scale backtesting and optimization.
"""

import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import optimized system components
from src.data_scraper.multi_source_manager import (
    MultiSourceDataManager, YahooFinanceSource, AlphaVantageSource
)
from src.backtesting_engine.optimized_engine import (
    OptimizedBacktestEngine, BacktestConfig
)
from src.portfolio.advanced_optimizer import (
    AdvancedPortfolioOptimizer, OptimizationConfig
)
from src.reporting.advanced_reporting import AdvancedReportGenerator
from src.data_scraper.advanced_cache import advanced_cache


def setup_data_manager():
    """Setup multi-source data manager with available sources."""
    logger.info("Setting up data manager...")
    
    # Initialize with Yahoo Finance (always available)
    data_manager = MultiSourceDataManager()
    
    # Add Alpha Vantage if API key is available
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        data_manager.add_source(AlphaVantageSource(os.getenv('ALPHA_VANTAGE_API_KEY')))
        logger.info("✅ Added Alpha Vantage data source")
    else:
        logger.info("ℹ️  Alpha Vantage API key not found, using Yahoo Finance only")
    
    # Show data source status
    status = data_manager.get_source_status()
    logger.info(f"Available data sources: {list(status.keys())}")
    
    return data_manager


def example_1_basic_optimization():
    """Example 1: Basic strategy optimization for a few symbols."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Basic Strategy Optimization")
    logger.info("="*60)
    
    # Setup
    data_manager = setup_data_manager()
    engine = OptimizedBacktestEngine(data_manager=data_manager, max_workers=2)
    optimizer = AdvancedPortfolioOptimizer(engine)
    
    # Configuration
    symbols = ["AAPL", "MSFT", "GOOGL"]
    strategies = ["rsi", "macd"]
    
    # Parameter ranges for optimization
    parameter_ranges = {
        "rsi": {
            "period": [10, 14, 20],
            "overbought": [70, 75, 80],
            "oversold": [20, 25, 30]
        },
        "macd": {
            "fast": [8, 12, 16],
            "slow": [21, 26, 30],
            "signal": [6, 9, 12]
        }
    }
    
    config = OptimizationConfig(
        symbols=symbols,
        strategies=strategies,
        parameter_ranges=parameter_ranges,
        optimization_metric="sharpe_ratio",
        start_date="2022-01-01",
        end_date="2023-12-31",
        max_iterations=20,  # Reduced for example
        population_size=10,  # Reduced for example
        n_jobs=2,
        use_cache=True
    )
    
    # Run optimization
    logger.info(f"Optimizing {len(symbols)} symbols with {len(strategies)} strategies...")
    start_time = time.time()
    
    try:
        results = optimizer.optimize_portfolio(config, method="genetic_algorithm")
        
        optimization_time = time.time() - start_time
        logger.info(f"✅ Optimization completed in {optimization_time:.2f} seconds")
        
        # Show results summary
        summary = optimizer.get_optimization_summary(results)
        logger.info(f"Overall stats: {summary['overall_stats']}")
        
        # Show best results
        best_results = []
        for symbol, strategies_results in results.items():
            for strategy, result in strategies_results.items():
                if result.best_score > float('-inf'):
                    best_results.append((symbol, strategy, result.best_score, result.best_parameters))
        
        best_results.sort(key=lambda x: x[2], reverse=True)
        
        logger.info("\nTop 3 optimized combinations:")
        for i, (symbol, strategy, score, params) in enumerate(best_results[:3]):
            logger.info(f"  {i+1}. {symbol}/{strategy}: {score:.4f} - {params}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        return None


def example_2_large_scale_backtesting():
    """Example 2: Large-scale backtesting with multiple asset classes."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Large-Scale Multi-Asset Backtesting")
    logger.info("="*60)
    
    # Setup
    data_manager = setup_data_manager()
    engine = OptimizedBacktestEngine(data_manager=data_manager, max_workers=4, memory_limit_gb=4.0)
    
    # Define asset universes
    asset_universes = {
        "large_cap_stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
        "forex_majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"],
        "crypto_major": ["BTC-USD", "ETH-USD"],
        "sector_etfs": ["XLK", "XLF", "XLV", "XLE"]
    }
    
    # Strategies to test
    strategies = ["rsi", "macd", "bollinger_bands"]
    
    # Run backtests for each asset class
    all_results = []
    total_combinations = sum(len(symbols) * len(strategies) for symbols in asset_universes.values())
    logger.info(f"Running {total_combinations} backtest combinations...")
    
    for asset_class, symbols in asset_universes.items():
        logger.info(f"\nProcessing {asset_class}: {len(symbols)} symbols")
        
        config = BacktestConfig(
            symbols=symbols,
            strategies=strategies,
            start_date="2022-01-01",
            end_date="2023-12-31",
            initial_capital=10000,
            commission=0.001,
            use_cache=True,
            save_trades=False,
            save_equity_curve=False,
            max_workers=4
        )
        
        try:
            asset_results = engine.run_batch_backtests(config)
            all_results.extend(asset_results)
            
            # Show asset class summary
            successful = [r for r in asset_results if not r.error]
            if successful:
                avg_return = sum(r.metrics.get('total_return', 0) for r in successful) / len(successful)
                best = max(successful, key=lambda x: x.metrics.get('total_return', 0))
                logger.info(f"  ✅ {asset_class}: {len(successful)}/{len(asset_results)} successful, "
                           f"avg return: {avg_return:.2f}%, best: {best.symbol}/{best.strategy}")
            else:
                logger.warning(f"  ❌ {asset_class}: No successful backtests")
                
        except Exception as e:
            logger.error(f"  ❌ {asset_class} failed: {e}")
    
    # Overall results summary
    successful_results = [r for r in all_results if not r.error]
    logger.info(f"\n📊 Overall Results: {len(successful_results)}/{len(all_results)} successful backtests")
    
    if successful_results:
        returns = [r.metrics.get('total_return', 0) for r in successful_results]
        sharpe_ratios = [r.metrics.get('sharpe_ratio', 0) for r in successful_results]
        
        logger.info(f"Average return: {sum(returns)/len(returns):.2f}%")
        logger.info(f"Average Sharpe ratio: {sum(sharpe_ratios)/len(sharpe_ratios):.3f}")
        
        # Top performers
        top_performers = sorted(successful_results, 
                              key=lambda x: x.metrics.get('total_return', 0), 
                              reverse=True)[:5]
        
        logger.info("\nTop 5 performers:")
        for i, result in enumerate(top_performers):
            logger.info(f"  {i+1}. {result.symbol}/{result.strategy}: "
                       f"{result.metrics.get('total_return', 0):.2f}% return, "
                       f"{result.metrics.get('sharpe_ratio', 0):.3f} Sharpe")
    
    return all_results


def example_3_advanced_reporting():
    """Example 3: Generate advanced reports with caching."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Advanced Reporting")
    logger.info("="*60)
    
    # Run a quick backtest to generate data for reporting
    data_manager = setup_data_manager()
    engine = OptimizedBacktestEngine(data_manager=data_manager)
    
    symbols = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    strategies = ["rsi", "macd"]
    
    config = BacktestConfig(
        symbols=symbols,
        strategies=strategies,
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_cache=True
    )
    
    logger.info("Generating backtest data for reporting...")
    results = engine.run_batch_backtests(config)
    successful_results = [r for r in results if not r.error]
    
    if not successful_results:
        logger.error("❌ No successful backtests for reporting")
        return
    
    # Setup report generator
    report_generator = AdvancedReportGenerator(cache_reports=True)
    
    # Generate portfolio report
    logger.info("Generating portfolio analysis report...")
    try:
        portfolio_report = report_generator.generate_portfolio_report(
            successful_results,
            title="Multi-Asset Portfolio Analysis",
            include_charts=True,
            format="html"
        )
        logger.info(f"✅ Portfolio report: {portfolio_report}")
    except Exception as e:
        logger.error(f"❌ Portfolio report failed: {e}")
    
    # Generate strategy comparison report
    logger.info("Generating strategy comparison report...")
    try:
        # Group results by strategy
        strategy_results = {}
        for result in successful_results:
            if result.strategy not in strategy_results:
                strategy_results[result.strategy] = []
            strategy_results[result.strategy].append(result)
        
        comparison_report = report_generator.generate_strategy_comparison_report(
            strategy_results,
            title="Strategy Performance Comparison",
            include_charts=True,
            format="html"
        )
        logger.info(f"✅ Strategy comparison report: {comparison_report}")
    except Exception as e:
        logger.error(f"❌ Strategy comparison report failed: {e}")


def example_4_cache_management():
    """Example 4: Demonstrate cache management features."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Cache Management")
    logger.info("="*60)
    
    # Show initial cache stats
    logger.info("Initial cache statistics:")
    stats = advanced_cache.get_cache_stats()
    logger.info(f"Total cache size: {stats['total_size_gb']:.2f} GB")
    logger.info(f"Cache utilization: {stats['utilization_percent']:.1f}%")
    
    for cache_type, type_stats in stats['by_type'].items():
        logger.info(f"  {cache_type}: {type_stats['count']} items, "
                   f"{type_stats['total_size_bytes']/1024**2:.1f} MB")
    
    # Demonstrate data caching
    logger.info("\nTesting data caching...")
    data_manager = setup_data_manager()
    
    # First fetch (should cache)
    start_time = time.time()
    data1 = data_manager.get_data("AAPL", "2023-01-01", "2023-12-31", "1d", use_cache=True)
    first_fetch_time = time.time() - start_time
    logger.info(f"First fetch (with caching): {first_fetch_time:.2f}s")
    
    # Second fetch (should use cache)
    start_time = time.time()
    data2 = data_manager.get_data("AAPL", "2023-01-01", "2023-12-31", "1d", use_cache=True)
    second_fetch_time = time.time() - start_time
    logger.info(f"Second fetch (from cache): {second_fetch_time:.2f}s")
    
    logger.info(f"Cache speedup: {first_fetch_time / second_fetch_time:.1f}x faster")
    
    # Show updated cache stats
    logger.info("\nUpdated cache statistics:")
    stats = advanced_cache.get_cache_stats()
    logger.info(f"Total cache size: {stats['total_size_gb']:.2f} GB")
    
    # Demonstrate cache cleanup
    logger.info("\nDemonstrating cache cleanup...")
    
    # Show what would be cleared (don't actually clear for demo)
    logger.info("Cache cleanup options:")
    logger.info("  - Clear data cache: advanced_cache.clear_cache(cache_type='data')")
    logger.info("  - Clear by symbol: advanced_cache.clear_cache(symbol='AAPL')")
    logger.info("  - Clear old items: advanced_cache.clear_cache(older_than_days=30)")


def example_5_incremental_updates():
    """Example 5: Demonstrate incremental backtesting for daily updates."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Incremental Backtesting")
    logger.info("="*60)
    
    data_manager = setup_data_manager()
    engine = OptimizedBacktestEngine(data_manager=data_manager)
    
    symbol = "AAPL"
    strategy = "rsi"
    
    config = BacktestConfig(
        symbols=[symbol],
        strategies=[strategy],
        start_date="2023-01-01",
        end_date="2023-12-31",
        use_cache=True
    )
    
    # Initial backtest (will be cached)
    logger.info(f"Running initial backtest for {symbol}/{strategy}...")
    start_time = time.time()
    
    initial_result = engine._run_single_backtest(symbol, strategy, config, None)
    initial_time = time.time() - start_time
    
    if initial_result.error:
        logger.error(f"❌ Initial backtest failed: {initial_result.error}")
        return
    
    logger.info(f"✅ Initial backtest completed in {initial_time:.2f}s")
    logger.info(f"Return: {initial_result.metrics.get('total_return', 0):.2f}%")
    
    # Simulate incremental update (should use cache for existing data)
    logger.info(f"\nRunning incremental update...")
    start_time = time.time()
    
    # This would normally check for new data since last run
    incremental_result = engine.run_incremental_backtest(
        symbol, strategy, config, 
        last_update=datetime(2023, 11, 1)  # Simulate last update date
    )
    
    incremental_time = time.time() - start_time
    
    if incremental_result and not incremental_result.error:
        logger.info(f"✅ Incremental update completed in {incremental_time:.2f}s")
        logger.info(f"Speedup: {initial_time / incremental_time:.1f}x faster")
        logger.info(f"Return: {incremental_result.metrics.get('total_return', 0):.2f}%")
    else:
        logger.info("ℹ️  No new data for incremental update")


def main():
    """Main example runner."""
    logger.info("🚀 Starting Quant System Optimization Examples")
    logger.info("=" * 80)
    
    try:
        # Run examples
        example_1_basic_optimization()
        example_2_large_scale_backtesting()
        example_3_advanced_reporting()
        example_4_cache_management()
        example_5_incremental_updates()
        
        logger.info("\n" + "="*80)
        logger.info("✅ All examples completed successfully!")
        logger.info("📁 Check the 'reports_output' directory for generated reports")
        logger.info("💾 Cache data is stored in the 'cache' directory")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Examples failed: {e}")
        raise


if __name__ == "__main__":
    main()
