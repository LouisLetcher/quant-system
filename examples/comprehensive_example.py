#!/usr/bin/env python3
"""
Comprehensive example showcasing the restructured quant system.
Demonstrates the unified architecture with Bybit crypto futures support and portfolio prioritization.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import unified components
from src.core import (
    UnifiedDataManager, UnifiedBacktestEngine, UnifiedResultAnalyzer,
    UnifiedCacheManager, PortfolioManager
)
from src.core.backtest_engine import BacktestConfig


def setup_environment():
    """Setup environment variables for the example."""
    logger.info("Setting up environment...")
    
    # Check for API keys
    bybit_key = os.getenv('BYBIT_API_KEY')
    bybit_secret = os.getenv('BYBIT_API_SECRET')
    
    if not bybit_key:
        logger.warning("BYBIT_API_KEY not found - will use demo mode")
        os.environ['BYBIT_TESTNET'] = 'true'
    
    # Create directories
    Path('examples/output').mkdir(exist_ok=True)
    
    return {
        'bybit_available': bool(bybit_key),
        'alpha_vantage_available': bool(os.getenv('ALPHA_VANTAGE_API_KEY'))
    }


def example_1_unified_data_management():
    """Example 1: Demonstrate unified data management with multiple sources."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 1: Unified Data Management")
    logger.info("="*60)
    
    # Initialize unified data manager
    data_manager = UnifiedDataManager()
    
    # Show available data sources
    sources = data_manager.get_source_status()
    logger.info("Available data sources:")
    for name, status in sources.items():
        logger.info(f"  {name}: Priority {status['priority']}, Batch: {status['supports_batch']}")
    
    # Test different asset types
    test_symbols = {
        'stocks': ['AAPL', 'MSFT', 'GOOGL'],
        'crypto': ['BTCUSDT', 'ETHUSDT', 'ADAUSDT'],
        'forex': ['EURUSD=X', 'GBPUSD=X']
    }
    
    results = {}
    
    for asset_type, symbols in test_symbols.items():
        logger.info(f"\nTesting {asset_type} data...")
        
        # Test single symbol
        symbol = symbols[0]
        start_time = time.time()
        
        if asset_type == 'crypto':
            # Test crypto futures data from Bybit
            data = data_manager.get_crypto_futures_data(
                symbol, '2023-01-01', '2023-12-31', '1d'
            )
        else:
            data = data_manager.get_data(
                symbol, '2023-01-01', '2023-12-31', '1d', 
                use_cache=True, asset_type=asset_type
            )
        
        fetch_time = time.time() - start_time
        
        if data is not None:
            logger.info(f"✅ {symbol}: {len(data)} data points in {fetch_time:.2f}s")
            results[symbol] = {'status': 'success', 'points': len(data), 'time': fetch_time}
        else:
            logger.warning(f"❌ {symbol}: No data")
            results[symbol] = {'status': 'failed', 'points': 0, 'time': fetch_time}
        
        # Test second fetch (should be faster due to caching)
        start_time = time.time()
        if asset_type == 'crypto':
            data2 = data_manager.get_crypto_futures_data(symbol, '2023-01-01', '2023-12-31', '1d')
        else:
            data2 = data_manager.get_data(symbol, '2023-01-01', '2023-12-31', '1d', True, asset_type)
        
        cache_time = time.time() - start_time
        
        if cache_time < fetch_time / 2:
            logger.info(f"🚀 Cache speedup: {fetch_time/cache_time:.1f}x faster")
        
        # Test batch fetching for this asset type
        if len(symbols) > 1:
            logger.info(f"Testing batch fetch for {asset_type}...")
            start_time = time.time()
            
            batch_data = data_manager.get_batch_data(
                symbols, '2023-01-01', '2023-12-31', '1d', 
                use_cache=True, asset_type=asset_type
            )
            
            batch_time = time.time() - start_time
            logger.info(f"Batch fetched {len(batch_data)}/{len(symbols)} symbols in {batch_time:.2f}s")
    
    return results


def example_2_crypto_futures_analysis():
    """Example 2: Demonstrate crypto futures analysis with Bybit."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 2: Crypto Futures Analysis")
    logger.info("="*60)
    
    # Initialize components
    data_manager = UnifiedDataManager()
    
    # Get available crypto futures symbols
    try:
        available_futures = data_manager.get_available_crypto_futures()
        logger.info(f"Available crypto futures: {len(available_futures)} symbols")
        
        if available_futures:
            logger.info("Top futures symbols:")
            for symbol in available_futures[:10]:
                logger.info(f"  {symbol}")
        
        # Test futures data
        test_futures = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        futures_data = {}
        
        for symbol in test_futures:
            logger.info(f"Fetching futures data for {symbol}...")
            
            data = data_manager.get_crypto_futures_data(
                symbol, '2023-06-01', '2023-12-31', '1h'
            )
            
            if data is not None and not data.empty:
                futures_data[symbol] = data
                logger.info(f"✅ {symbol}: {len(data)} hourly data points")
                
                # Calculate basic statistics
                price_change = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                volatility = data['close'].pct_change().std() * 100
                
                logger.info(f"   Price change: {price_change:.2f}%")
                logger.info(f"   Volatility: {volatility:.2f}%")
            else:
                logger.warning(f"❌ {symbol}: No data available")
        
        return futures_data
        
    except Exception as e:
        logger.error(f"Crypto futures analysis failed: {e}")
        return {}


def example_3_unified_backtesting():
    """Example 3: Demonstrate unified backtesting across asset classes."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 3: Unified Backtesting")
    logger.info("="*60)
    
    # Initialize unified backtesting engine
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(data_manager, cache_manager, max_workers=4)
    
    # Define test portfolios across different asset classes
    portfolios = {
        'tech_stocks': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'strategies': ['rsi', 'macd'],
            'asset_type': 'stocks'
        },
        'crypto_futures': {
            'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT'],
            'strategies': ['rsi', 'macd'],
            'asset_type': 'crypto',
            'futures_mode': True
        },
        'forex_majors': {
            'symbols': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X'],
            'strategies': ['rsi', 'macd'],
            'asset_type': 'forex'
        }
    }
    
    all_results = {}
    
    for portfolio_name, portfolio_config in portfolios.items():
        logger.info(f"\nBacktesting portfolio: {portfolio_name}")
        
        # Create backtest configuration
        config = BacktestConfig(
            symbols=portfolio_config['symbols'],
            strategies=portfolio_config['strategies'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=10000,
            interval='1d',
            use_cache=True,
            asset_type=portfolio_config['asset_type'],
            futures_mode=portfolio_config.get('futures_mode', False),
            max_workers=4
        )
        
        # Run batch backtests
        start_time = time.time()
        results = engine.run_batch_backtests(config)
        duration = time.time() - start_time
        
        # Analyze results
        successful_results = [r for r in results if not r.error]
        failed_results = [r for r in results if r.error]
        
        logger.info(f"Results: {len(successful_results)} successful, {len(failed_results)} failed")
        logger.info(f"Duration: {duration:.2f}s")
        
        if successful_results:
            # Calculate portfolio statistics
            returns = [r.metrics.get('total_return', 0) for r in successful_results]
            sharpes = [r.metrics.get('sharpe_ratio', 0) for r in successful_results]
            
            logger.info(f"Average return: {sum(returns)/len(returns):.2f}%")
            logger.info(f"Average Sharpe: {sum(sharpes)/len(sharpes):.3f}")
            
            # Best performer
            best_result = max(successful_results, key=lambda x: x.metrics.get('total_return', 0))
            logger.info(f"Best performer: {best_result.symbol}/{best_result.strategy} "
                       f"({best_result.metrics.get('total_return', 0):.2f}%)")
        
        all_results[portfolio_name] = results
    
    # Show engine performance stats
    stats = engine.get_performance_stats()
    logger.info(f"\nEngine Performance:")
    logger.info(f"  Total backtests: {stats['backtests_run']}")
    logger.info(f"  Cache hits: {stats['cache_hits']}")
    logger.info(f"  Cache misses: {stats['cache_misses']}")
    logger.info(f"  Total time: {stats['total_time']:.2f}s")
    
    return all_results


def example_4_portfolio_comparison_and_prioritization():
    """Example 4: Demonstrate portfolio comparison and investment prioritization."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 4: Portfolio Comparison & Investment Prioritization")
    logger.info("="*60)
    
    # Use results from previous example or create new ones
    logger.info("Setting up portfolio analysis...")
    
    # Run quick backtests for demonstration
    data_manager = UnifiedDataManager()
    cache_manager = UnifiedCacheManager()
    engine = UnifiedBacktestEngine(data_manager, cache_manager)
    portfolio_manager = PortfolioManager()
    
    # Define different investment portfolios
    investment_portfolios = {
        'Conservative Growth': {
            'symbols': ['AAPL', 'MSFT', 'JNJ', 'PG'],
            'strategies': ['sma_crossover'],
            'description': 'Large-cap stocks with stable growth'
        },
        'Aggressive Tech': {
            'symbols': ['TSLA', 'NVDA', 'AMD', 'SQ'],
            'strategies': ['rsi', 'macd'],
            'description': 'High-growth technology stocks'
        },
        'Crypto Futures': {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'strategies': ['rsi'],
            'description': 'Cryptocurrency futures trading',
            'futures_mode': True
        },
        'Diversified Income': {
            'symbols': ['VTI', 'VEA', 'BND', 'REIT'],
            'strategies': ['bollinger_bands'],
            'description': 'Diversified ETF portfolio'
        }
    }
    
    # Run backtests for each portfolio
    portfolio_results = {}
    
    for portfolio_name, portfolio_config in investment_portfolios.items():
        logger.info(f"Analyzing portfolio: {portfolio_name}")
        
        config = BacktestConfig(
            symbols=portfolio_config['symbols'],
            strategies=portfolio_config['strategies'],
            start_date='2023-01-01',
            end_date='2023-12-31',
            initial_capital=10000,
            use_cache=True,
            futures_mode=portfolio_config.get('futures_mode', False)
        )
        
        try:
            results = engine.run_batch_backtests(config)
            portfolio_results[portfolio_name] = results
            
            successful = [r for r in results if not r.error]
            if successful:
                avg_return = sum(r.metrics.get('total_return', 0) for r in successful) / len(successful)
                logger.info(f"  Average return: {avg_return:.2f}%")
            
        except Exception as e:
            logger.error(f"Portfolio {portfolio_name} failed: {e}")
            portfolio_results[portfolio_name] = []
    
    # Perform comprehensive portfolio analysis
    logger.info("\nPerforming portfolio analysis...")
    analysis = portfolio_manager.analyze_portfolios(portfolio_results)
    
    # Display portfolio rankings
    logger.info("\nPortfolio Rankings:")
    logger.info("=" * 40)
    
    for portfolio_name, summary in analysis['portfolio_summaries'].items():
        logger.info(f"\n{summary['investment_priority']}. {portfolio_name}")
        logger.info(f"   Overall Score: {summary['overall_score']:.1f}/100")
        logger.info(f"   Average Return: {summary['avg_return']:.2f}%")
        logger.info(f"   Sharpe Ratio: {summary['avg_sharpe']:.3f}")
        logger.info(f"   Risk Category: {summary['risk_category']}")
        logger.info(f"   Max Drawdown: {summary['max_drawdown']:.2f}%")
    
    # Display investment recommendations
    logger.info("\nInvestment Recommendations:")
    logger.info("=" * 30)
    
    for rec in analysis['investment_recommendations']:
        logger.info(f"\n{rec['priority_rank']}. {rec['portfolio_name']}")
        logger.info(f"   Recommended Allocation: {rec['recommended_allocation_pct']:.1f}%")
        logger.info(f"   Expected Return: {rec['expected_annual_return']:.2f}%")
        logger.info(f"   Risk Level: {rec['risk_category']}")
        logger.info(f"   Confidence Score: {rec['confidence_score']:.1f}/100")
        logger.info(f"   Rationale: {rec['investment_rationale']}")
    
    # Generate investment plan for different capital amounts
    capital_scenarios = [50000, 100000, 250000]
    
    for capital in capital_scenarios:
        logger.info(f"\nInvestment Plan for ${capital:,}")
        logger.info("-" * 30)
        
        investment_plan = portfolio_manager.generate_investment_plan(
            capital, portfolio_results, risk_tolerance='moderate'
        )
        
        for allocation in investment_plan['allocations']:
            logger.info(f"  {allocation['portfolio_name']}: "
                       f"${allocation['allocation_amount']:,.0f} "
                       f"({allocation['allocation_percentage']:.1f}%)")
        
        expected = investment_plan['expected_portfolio_metrics']
        logger.info(f"  Expected Portfolio Return: {expected.get('expected_annual_return', 0):.2f}%")
        logger.info(f"  Expected Sharpe Ratio: {expected.get('expected_sharpe_ratio', 0):.3f}")
    
    # Save detailed analysis
    output_file = 'examples/output/portfolio_analysis.json'
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    logger.info(f"\nDetailed analysis saved to {output_file}")
    
    return analysis


def example_5_advanced_caching_demonstration():
    """Example 5: Demonstrate advanced caching capabilities."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 5: Advanced Caching Demonstration")
    logger.info("="*60)
    
    cache_manager = UnifiedCacheManager()
    
    # Show initial cache stats
    logger.info("Initial cache statistics:")
    stats = cache_manager.get_cache_stats()
    logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
    logger.info(f"  Utilization: {stats['utilization_percent']:.1f}%")
    
    for cache_type, type_stats in stats['by_type'].items():
        logger.info(f"  {cache_type}: {type_stats['count']} items, "
                   f"{type_stats['total_size_mb']:.1f} MB")
    
    # Demonstrate caching performance
    data_manager = UnifiedDataManager()
    
    test_symbol = 'AAPL'
    logger.info(f"\nTesting cache performance with {test_symbol}...")
    
    # First fetch (cold cache)
    start_time = time.time()
    data1 = data_manager.get_data(test_symbol, '2023-01-01', '2023-12-31', '1d', True)
    cold_time = time.time() - start_time
    logger.info(f"Cold cache fetch: {cold_time:.3f}s")
    
    # Second fetch (warm cache)
    start_time = time.time()
    data2 = data_manager.get_data(test_symbol, '2023-01-01', '2023-12-31', '1d', True)
    warm_time = time.time() - start_time
    logger.info(f"Warm cache fetch: {warm_time:.3f}s")
    
    if warm_time > 0:
        speedup = cold_time / warm_time
        logger.info(f"Cache speedup: {speedup:.1f}x")
    
    # Test cache with different data types
    logger.info("\nTesting cache with different data types...")
    
    # Cache some sample backtest results
    sample_result = {
        'total_return': 15.5,
        'sharpe_ratio': 1.2,
        'max_drawdown': -8.3,
        'win_rate': 65.0
    }
    
    sample_parameters = {'period': 14, 'overbought': 70, 'oversold': 30}
    
    cache_key = cache_manager.cache_backtest_result(
        'AAPL', 'rsi', sample_parameters, sample_result, '1d'
    )
    logger.info(f"Cached backtest result with key: {cache_key[:16]}...")
    
    # Retrieve cached result
    retrieved_result = cache_manager.get_backtest_result(
        'AAPL', 'rsi', sample_parameters, '1d'
    )
    
    if retrieved_result:
        logger.info("✅ Successfully retrieved cached backtest result")
        logger.info(f"   Return: {retrieved_result.get('total_return', 0):.2f}%")
    else:
        logger.warning("❌ Failed to retrieve cached result")
    
    # Show updated cache stats
    logger.info("\nUpdated cache statistics:")
    stats = cache_manager.get_cache_stats()
    logger.info(f"  Total size: {stats['total_size_gb']:.2f} GB")
    
    return stats


def example_6_unified_cli_demonstration():
    """Example 6: Demonstrate unified CLI usage."""
    logger.info("\n" + "="*60)
    logger.info("EXAMPLE 6: Unified CLI Demonstration")
    logger.info("="*60)
    
    logger.info("The unified CLI provides comprehensive functionality:")
    logger.info("\n📊 Data Management:")
    logger.info("  python -m src.cli.unified_cli data download --symbols AAPL MSFT --start-date 2023-01-01 --end-date 2023-12-31")
    logger.info("  python -m src.cli.unified_cli data sources")
    
    logger.info("\n🔬 Backtesting:")
    logger.info("  python -m src.cli.unified_cli backtest single --symbol AAPL --strategy rsi --start-date 2023-01-01 --end-date 2023-12-31")
    logger.info("  python -m src.cli.unified_cli backtest batch --symbols AAPL MSFT GOOGL --strategies rsi macd --start-date 2023-01-01 --end-date 2023-12-31")
    
    logger.info("\n💼 Portfolio Management:")
    logger.info("  python -m src.cli.unified_cli portfolio backtest --symbols AAPL MSFT GOOGL --strategy rsi --start-date 2023-01-01 --end-date 2023-12-31")
    logger.info("  python -m src.cli.unified_cli portfolio compare --portfolios portfolios.json --start-date 2023-01-01 --end-date 2023-12-31")
    logger.info("  python -m src.cli.unified_cli portfolio plan --portfolios results.json --capital 100000 --risk-tolerance moderate")
    
    logger.info("\n📈 Crypto Futures (Bybit):")
    logger.info("  python -m src.cli.unified_cli data download --symbols BTCUSDT ETHUSDT --futures --start-date 2023-01-01 --end-date 2023-12-31")
    logger.info("  python -m src.cli.unified_cli backtest single --symbol BTCUSDT --strategy rsi --futures --start-date 2023-01-01 --end-date 2023-12-31")
    
    logger.info("\n💾 Cache Management:")
    logger.info("  python -m src.cli.unified_cli cache stats")
    logger.info("  python -m src.cli.unified_cli cache clear --type data --older-than 30")
    
    logger.info("\n📊 Analysis & Reporting:")
    logger.info("  python -m src.cli.unified_cli analyze report --input results.json --type portfolio --format html")
    
    # Create example configuration files
    example_portfolios = {
        'tech_growth': {
            'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN'],
            'strategies': ['rsi', 'macd']
        },
        'crypto_futures': {
            'symbols': ['BTCUSDT', 'ETHUSDT'],
            'strategies': ['rsi']
        }
    }
    
    with open('examples/output/example_portfolios.json', 'w') as f:
        json.dump(example_portfolios, f, indent=2)
    
    logger.info(f"\nExample portfolio configuration saved to examples/output/example_portfolios.json")


def main():
    """Main function to run all examples."""
    logger.info("🚀 Comprehensive Quant System Demonstration")
    logger.info("=" * 80)
    
    # Setup environment
    env_info = setup_environment()
    
    try:
        # Run examples
        logger.info("Running comprehensive examples...")
        
        # Data management
        data_results = example_1_unified_data_management()
        
        # Crypto futures (if available)
        if env_info['bybit_available']:
            crypto_results = example_2_crypto_futures_analysis()
        else:
            logger.info("Skipping crypto futures example (no API key)")
            crypto_results = {}
        
        # Unified backtesting
        backtest_results = example_3_unified_backtesting()
        
        # Portfolio analysis
        portfolio_analysis = example_4_portfolio_comparison_and_prioritization()
        
        # Caching demonstration
        cache_stats = example_5_advanced_caching_demonstration()
        
        # CLI demonstration
        example_6_unified_cli_demonstration()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("✅ All examples completed successfully!")
        logger.info("="*80)
        
        # Show summary statistics
        logger.info("\n📊 Summary Statistics:")
        
        successful_data = sum(1 for r in data_results.values() if r['status'] == 'success')
        logger.info(f"  Data fetching: {successful_data}/{len(data_results)} successful")
        
        if crypto_results:
            logger.info(f"  Crypto futures: {len(crypto_results)} symbols analyzed")
        
        total_backtests = sum(len(results) for results in backtest_results.values())
        logger.info(f"  Total backtests: {total_backtests}")
        
        portfolio_count = len(portfolio_analysis.get('portfolio_summaries', {}))
        logger.info(f"  Portfolios analyzed: {portfolio_count}")
        
        cache_size = cache_stats.get('total_size_mb', 0)
        logger.info(f"  Cache size: {cache_size:.1f} MB")
        
        logger.info("\n📁 Output files:")
        logger.info("  examples/output/portfolio_analysis.json")
        logger.info("  examples/output/example_portfolios.json")
        
        logger.info("\n🔧 Key Features Demonstrated:")
        logger.info("  ✅ Multi-source data management (Yahoo Finance, Bybit, Alpha Vantage)")
        logger.info("  ✅ Crypto futures trading support via Bybit")
        logger.info("  ✅ Unified backtesting across asset classes")
        logger.info("  ✅ Portfolio comparison and investment prioritization")
        logger.info("  ✅ Advanced caching with SQLite metadata")
        logger.info("  ✅ Comprehensive CLI interface")
        logger.info("  ✅ Memory-efficient parallel processing")
        logger.info("  ✅ Risk-adjusted performance metrics")
        
        logger.info("\n🎯 Next Steps:")
        logger.info("  1. Set up API keys for additional data sources")
        logger.info("  2. Customize strategy parameters for your needs") 
        logger.info("  3. Use the CLI for daily trading operations")
        logger.info("  4. Set up automated portfolio monitoring")
        logger.info("  5. Explore optimization features for parameter tuning")
        
    except KeyboardInterrupt:
        logger.info("\n⏹️  Examples interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Examples failed: {e}")
        raise


if __name__ == "__main__":
    main()
