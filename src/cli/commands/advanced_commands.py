"""
Advanced CLI commands for the optimized backtesting system.
Supports multi-source data, advanced optimization, and comprehensive reporting.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List

from src.data_scraper.multi_source_manager import (
    MultiSourceDataManager, YahooFinanceSource, AlphaVantageSource, TwelveDataSource
)
from src.backtesting_engine.optimized_engine import OptimizedBacktestEngine, BacktestConfig
from src.portfolio.advanced_optimizer import (
    AdvancedPortfolioOptimizer, OptimizationConfig, GridSearchOptimizer,
    GeneticAlgorithmOptimizer, BayesianOptimizer
)
from src.reporting.advanced_reporting import AdvancedReportGenerator
from src.data_scraper.advanced_cache import advanced_cache


def register_commands(subparsers):
    """Register advanced commands with the CLI."""
    # Advanced backtest command
    advanced_backtest_parser = subparsers.add_parser(
        'advanced-backtest',
        help='Run advanced backtests with multi-source data and caching'
    )
    advanced_backtest_parser.add_argument('--symbols', nargs='+', required=True,
                                        help='List of symbols to backtest')
    advanced_backtest_parser.add_argument('--strategies', nargs='+', required=True,
                                        help='List of strategies to test')
    advanced_backtest_parser.add_argument('--start-date', required=True,
                                        help='Start date (YYYY-MM-DD)')
    advanced_backtest_parser.add_argument('--end-date', required=True,
                                        help='End date (YYYY-MM-DD)')
    advanced_backtest_parser.add_argument('--interval', default='1d',
                                        help='Data interval (1d, 1h, etc.)')
    advanced_backtest_parser.add_argument('--initial-capital', type=float, default=10000,
                                        help='Initial capital amount')
    advanced_backtest_parser.add_argument('--commission', type=float, default=0.001,
                                        help='Commission rate')
    advanced_backtest_parser.add_argument('--max-workers', type=int, default=None,
                                        help='Maximum number of parallel workers')
    advanced_backtest_parser.add_argument('--memory-limit', type=float, default=8.0,
                                        help='Memory limit in GB')
    advanced_backtest_parser.add_argument('--no-cache', action='store_true',
                                        help='Disable caching')
    advanced_backtest_parser.add_argument('--save-trades', action='store_true',
                                        help='Save individual trades')
    advanced_backtest_parser.add_argument('--save-equity', action='store_true',
                                        help='Save equity curves')
    advanced_backtest_parser.add_argument('--output-format', choices=['json', 'html'], default='html',
                                        help='Output format for results')
    advanced_backtest_parser.set_defaults(func=advanced_backtest_command)
    
    # Portfolio optimization command
    optimize_parser = subparsers.add_parser(
        'optimize',
        help='Optimize strategy parameters for portfolio'
    )
    optimize_parser.add_argument('--symbols', nargs='+', required=True,
                               help='List of symbols to optimize')
    optimize_parser.add_argument('--strategies', nargs='+', required=True,
                               help='List of strategies to optimize')
    optimize_parser.add_argument('--param-config', required=True,
                               help='Path to parameter configuration JSON file')
    optimize_parser.add_argument('--start-date', required=True,
                               help='Start date (YYYY-MM-DD)')
    optimize_parser.add_argument('--end-date', required=True,
                               help='End date (YYYY-MM-DD)')
    optimize_parser.add_argument('--method', choices=['grid_search', 'genetic_algorithm', 'bayesian'],
                               default='genetic_algorithm',
                               help='Optimization method')
    optimize_parser.add_argument('--metric', default='sharpe_ratio',
                               help='Optimization metric')
    optimize_parser.add_argument('--max-iterations', type=int, default=100,
                               help='Maximum optimization iterations')
    optimize_parser.add_argument('--population-size', type=int, default=50,
                               help='Population size for genetic algorithm')
    optimize_parser.add_argument('--n-jobs', type=int, default=-1,
                               help='Number of parallel jobs (-1 for all cores)')
    optimize_parser.add_argument('--no-cache', action='store_true',
                               help='Disable caching')
    optimize_parser.set_defaults(func=optimize_command)
    
    # Data management commands
    data_parser = subparsers.add_parser(
        'data',
        help='Data management commands'
    )
    data_subparsers = data_parser.add_subparsers(dest='data_command')
    
    # Download data command
    download_parser = data_subparsers.add_parser(
        'download',
        help='Download and cache data for symbols'
    )
    download_parser.add_argument('--symbols', nargs='+', required=True,
                               help='List of symbols to download')
    download_parser.add_argument('--start-date', required=True,
                               help='Start date (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', required=True,
                               help='End date (YYYY-MM-DD)')
    download_parser.add_argument('--interval', default='1d',
                               help='Data interval')
    download_parser.add_argument('--sources', nargs='+', 
                               choices=['yahoo', 'alpha_vantage', 'twelve_data'],
                               default=['yahoo'],
                               help='Data sources to use')
    download_parser.add_argument('--force-update', action='store_true',
                               help='Force update even if cached data exists')
    download_parser.set_defaults(func=download_data_command)
    
    # Cache management command
    cache_parser = data_subparsers.add_parser(
        'cache',
        help='Cache management commands'
    )
    cache_subparsers = cache_parser.add_subparsers(dest='cache_command')
    
    # Cache stats
    stats_parser = cache_subparsers.add_parser('stats', help='Show cache statistics')
    stats_parser.set_defaults(func=cache_stats_command)
    
    # Clear cache
    clear_parser = cache_subparsers.add_parser('clear', help='Clear cache')
    clear_parser.add_argument('--type', choices=['data', 'backtest', 'optimization'],
                            help='Cache type to clear')
    clear_parser.add_argument('--symbol', help='Clear cache for specific symbol')
    clear_parser.add_argument('--strategy', help='Clear cache for specific strategy')
    clear_parser.add_argument('--older-than', type=int, help='Clear items older than N days')
    clear_parser.set_defaults(func=clear_cache_command)
    
    data_parser.set_defaults(func=data_command)
    
    # Advanced reporting command
    report_parser = subparsers.add_parser(
        'advanced-report',
        help='Generate advanced reports'
    )
    report_parser.add_argument('--type', choices=['portfolio', 'strategy', 'optimization'],
                             required=True, help='Report type')
    report_parser.add_argument('--input', required=True,
                             help='Input file (JSON results from backtest/optimization)')
    report_parser.add_argument('--title', help='Report title')
    report_parser.add_argument('--format', choices=['html', 'json'], default='html',
                             help='Output format')
    report_parser.add_argument('--no-charts', action='store_true',
                             help='Disable interactive charts')
    report_parser.add_argument('--output-dir', default='reports_output',
                             help='Output directory')
    report_parser.set_defaults(func=advanced_report_command)


def advanced_backtest_command(args):
    """Run advanced backtests with multi-source data and optimization."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting advanced backtest: {len(args.symbols)} symbols, {len(args.strategies)} strategies")
    
    # Setup data manager with multiple sources
    data_manager = MultiSourceDataManager()
    
    # Add additional sources if API keys are available
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        data_manager.add_source(AlphaVantageSource(os.getenv('ALPHA_VANTAGE_API_KEY')))
        logger.info("Added Alpha Vantage data source")
    
    if os.getenv('TWELVE_DATA_API_KEY'):
        data_manager.add_source(TwelveDataSource(os.getenv('TWELVE_DATA_API_KEY')))
        logger.info("Added Twelve Data source")
    
    # Setup optimized engine
    engine = OptimizedBacktestEngine(
        data_manager=data_manager,
        max_workers=args.max_workers,
        memory_limit_gb=args.memory_limit
    )
    
    # Create backtest configuration
    config = BacktestConfig(
        symbols=args.symbols,
        strategies=args.strategies,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        interval=args.interval,
        commission=args.commission,
        use_cache=not args.no_cache,
        save_trades=args.save_trades,
        save_equity_curve=args.save_equity,
        memory_limit_gb=args.memory_limit,
        max_workers=args.max_workers
    )
    
    # Run backtests
    try:
        results = engine.run_batch_backtests(config)
        
        # Save results
        timestamp = int(time.time())
        output_file = f"backtest_results_{timestamp}.{args.output_format}"
        
        if args.output_format == 'json':
            results_data = [asdict(result) for result in results]
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
        else:
            # Generate HTML report
            report_generator = AdvancedReportGenerator()
            report_path = report_generator.generate_portfolio_report(
                results, 
                title=f"Portfolio Backtest Results - {args.start_date} to {args.end_date}",
                format='html'
            )
            output_file = report_path
        
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary
        successful_results = [r for r in results if not r.error]
        logger.info(f"Completed: {len(successful_results)}/{len(results)} successful backtests")
        
        if successful_results:
            avg_return = sum(r.metrics.get('total_return', 0) for r in successful_results) / len(successful_results)
            best_result = max(successful_results, key=lambda x: x.metrics.get('total_return', 0))
            logger.info(f"Average return: {avg_return:.2f}%")
            logger.info(f"Best performer: {best_result.symbol}/{best_result.strategy} ({best_result.metrics.get('total_return', 0):.2f}%)")
        
        # Show performance stats
        stats = engine.get_performance_stats()
        logger.info(f"Engine stats: {stats}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        sys.exit(1)


def optimize_command(args):
    """Run portfolio optimization."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting optimization: {args.method} method")
    
    # Load parameter configuration
    try:
        with open(args.param_config, 'r') as f:
            param_ranges = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load parameter config: {e}")
        sys.exit(1)
    
    # Setup data manager
    data_manager = MultiSourceDataManager()
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        data_manager.add_source(AlphaVantageSource(os.getenv('ALPHA_VANTAGE_API_KEY')))
    if os.getenv('TWELVE_DATA_API_KEY'):
        data_manager.add_source(TwelveDataSource(os.getenv('TWELVE_DATA_API_KEY')))
    
    # Setup optimizer
    engine = OptimizedBacktestEngine(data_manager=data_manager)
    optimizer = AdvancedPortfolioOptimizer(engine)
    
    # Create optimization configuration
    config = OptimizationConfig(
        symbols=args.symbols,
        strategies=args.strategies,
        parameter_ranges=param_ranges,
        optimization_metric=args.metric,
        start_date=args.start_date,
        end_date=args.end_date,
        max_iterations=args.max_iterations,
        population_size=args.population_size,
        n_jobs=args.n_jobs,
        use_cache=not args.no_cache
    )
    
    # Run optimization
    try:
        results = optimizer.optimize_portfolio(config, method=args.method)
        
        # Save results
        timestamp = int(time.time())
        output_file = f"optimization_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = {}
        for symbol, strategies in results.items():
            serializable_results[symbol] = {}
            for strategy, result in strategies.items():
                serializable_results[symbol][strategy] = asdict(result)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to: {output_file}")
        
        # Generate summary report
        summary = optimizer.get_optimization_summary(results)
        logger.info(f"Optimization summary: {summary['overall_stats']}")
        
        # Show best results
        best_results = []
        for symbol, strategies in results.items():
            for strategy, result in strategies.items():
                if result.best_score > float('-inf'):
                    best_results.append((symbol, strategy, result.best_score))
        
        best_results.sort(key=lambda x: x[2], reverse=True)
        logger.info("Top 5 optimized combinations:")
        for symbol, strategy, score in best_results[:5]:
            logger.info(f"  {symbol}/{strategy}: {score:.4f}")
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        sys.exit(1)


def download_data_command(args):
    """Download and cache data for symbols."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info(f"Downloading data for {len(args.symbols)} symbols")
    
    # Setup data manager
    data_manager = MultiSourceDataManager()
    
    # Add requested sources
    if 'alpha_vantage' in args.sources and os.getenv('ALPHA_VANTAGE_API_KEY'):
        data_manager.add_source(AlphaVantageSource(os.getenv('ALPHA_VANTAGE_API_KEY')))
    if 'twelve_data' in args.sources and os.getenv('TWELVE_DATA_API_KEY'):
        data_manager.add_source(TwelveDataSource(os.getenv('TWELVE_DATA_API_KEY')))
    
    # Download data
    use_cache = not args.force_update
    successful_downloads = 0
    
    for symbol in args.symbols:
        try:
            data = data_manager.get_data(
                symbol, args.start_date, args.end_date, 
                args.interval, use_cache
            )
            if data is not None:
                successful_downloads += 1
                logger.info(f"✅ Downloaded {symbol}: {len(data)} data points")
            else:
                logger.warning(f"❌ Failed to download {symbol}")
        except Exception as e:
            logger.error(f"❌ Error downloading {symbol}: {e}")
    
    logger.info(f"Download complete: {successful_downloads}/{len(args.symbols)} successful")


def cache_stats_command(args):
    """Show cache statistics."""
    stats = advanced_cache.get_cache_stats()
    
    print("\nCache Statistics:")
    print(f"Total size: {stats['total_size_gb']:.2f} GB / {stats['max_size_gb']:.2f} GB")
    print(f"Utilization: {stats['utilization_percent']:.1f}%")
    print("\nBy cache type:")
    
    for cache_type, type_stats in stats['by_type'].items():
        print(f"  {cache_type}:")
        print(f"    Count: {type_stats['count']}")
        print(f"    Size: {type_stats['total_size_bytes'] / 1024**3:.2f} GB")
        print(f"    Avg size: {type_stats['avg_size_bytes'] / 1024**2:.2f} MB")


def clear_cache_command(args):
    """Clear cache based on filters."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("Clearing cache...")
    
    advanced_cache.clear_cache(
        cache_type=args.type,
        symbol=args.symbol,
        strategy=args.strategy,
        older_than_days=args.older_than
    )
    
    logger.info("Cache cleared successfully")


def data_command(args):
    """Handle data management commands."""
    if args.data_command == 'download':
        download_data_command(args)
    elif args.data_command == 'cache':
        if args.cache_command == 'stats':
            cache_stats_command(args)
        elif args.cache_command == 'clear':
            clear_cache_command(args)
        else:
            print("Available cache commands: stats, clear")
    else:
        print("Available data commands: download, cache")


def advanced_report_command(args):
    """Generate advanced reports."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load input data
    try:
        with open(args.input, 'r') as f:
            input_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load input data: {e}")
        sys.exit(1)
    
    # Setup report generator
    report_generator = AdvancedReportGenerator(output_dir=args.output_dir)
    
    title = args.title or f"{args.type.title()} Report"
    include_charts = not args.no_charts
    
    try:
        if args.type == 'portfolio':
            # Convert data back to BacktestResult objects if needed
            # This is a simplified version - you might need more sophisticated conversion
            report_path = report_generator.generate_portfolio_report(
                input_data, title=title, include_charts=include_charts, format=args.format
            )
        elif args.type == 'strategy':
            report_path = report_generator.generate_strategy_comparison_report(
                input_data, title=title, include_charts=include_charts, format=args.format
            )
        elif args.type == 'optimization':
            report_path = report_generator.generate_optimization_report(
                input_data, title=title, include_charts=include_charts, format=args.format
            )
        else:
            logger.error(f"Unknown report type: {args.type}")
            sys.exit(1)
        
        logger.info(f"Report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        sys.exit(1)


# Import required modules for the commands
import time
from dataclasses import asdict
