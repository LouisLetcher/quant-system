from __future__ import annotations

import json
import math
import os
import webbrowser
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.cli.config.config_loader import get_default_parameters, get_portfolio_config
from src.reports.report_generator import ReportGenerator
from src.utils.config_manager import ConfigManager
from src.utils.logger import get_logger, setup_command_logging, Logger

# Initialize logger
logger = get_logger(__name__)

def backtest_portfolio(args):
    """Run a backtest of all assets in a portfolio with all strategies."""
    # Setup logging if requested
    log_file = setup_command_logging(args)
    
    logger.info(f"Starting portfolio backtest for '{args.name}'")
    logger.info(f"Parameters: period={args.period}, metric={args.metric}, plot={args.plot}, resample={args.resample}")
    
    portfolio_config = get_portfolio_config(args.name)

    if not portfolio_config:
        logger.error(f"Portfolio '{args.name}' not found in assets_config.json")
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}

    # Get default parameters from config
    defaults = get_default_parameters()
    logger.debug(f"Default parameters: {defaults}")

    print(f"Testing all strategies on portfolio '{args.name}'")
    print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
    
    logger.info(f"Portfolio description: {portfolio_config.get('description', 'No description')}")

    # Get all available strategies
    strategies = StrategyFactory.get_available_strategies()
    assets = portfolio_config.get("assets", [])
    logger.info(f"Testing {len(strategies)} strategies on {len(assets)} assets")
    print(f"Testing {len(strategies)} strategies on {len(assets)} assets")

    results = {}
    for asset_config in assets:
        ticker = asset_config["ticker"]
        asset_period = asset_config.get("period", args.period)
        commission = asset_config.get("commission", defaults["commission"])
        initial_capital = asset_config.get(
            "initial_capital", defaults["initial_capital"]
        )

        logger.info(f"Testing {ticker} with {initial_capital} initial capital, commission={commission}, period={asset_period}")
        print(f"\n🔍 Testing {ticker} with {initial_capital} initial capital")

        results[ticker] = _backtest_all_strategies(
            ticker=ticker,
            period=asset_period,
            metric=args.metric,
            commission=commission,
            initial_capital=initial_capital,
            plot=args.plot,
            resample=args.resample,
        )

    # If not plotting individual strategies, generate a portfolio report
    if not args.plot:
        output_path = f"reports_output/portfolio_{args.name}_{args.metric}.html"
        generator = ReportGenerator()

        # Create portfolio results object
        portfolio_results = {
            "portfolio": args.name,
            "description": portfolio_config.get("description", ""),
            "assets": results,
            "metric": args.metric,
        }

        logger.info(f"Generating detailed portfolio report to {output_path}")
        # Generate the detailed report with equity curves and trade tables
        generator.generate_detailed_portfolio_report(portfolio_results, output_path)

        print(f"📄 Detailed Portfolio Report saved to {output_path}")
        logger.info(f"Detailed Portfolio Report saved to {output_path}")

        # Open the report in the browser if requested
        if args.open_browser:
            logger.info(f"Opening report in browser: {os.path.abspath(output_path)}")
            webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)

    logger.info("Portfolio backtest completed successfully")
    return results


def _backtest_all_strategies(
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
            data = DataLoader.load_data(ticker, period=period)
            if data is None or data.empty:
                logger.warning(f"No data available for {ticker} with period {period}")
                continue
                
            logger.debug(f"Loaded {len(data)} data points for {ticker}")
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
            results = engine.run()
            backtest_obj = engine.get_backtest_object()

            # Extract performance metric
            if metric == "profit_factor":
                score = results.get("Profit Factor", results.get("profit_factor", 0))
            elif metric == "sharpe":
                score = results.get("Sharpe Ratio", results.get("sharpe_ratio", 0))
            elif metric == "return":
                if isinstance(results.get("Return [%]", 0), (int, float)):
                    score = results.get("Return [%]", 0)
                else:
                    score = results.get("return_pct", 0)
            else:
                score = results.get(metric, 0)

            logger.info(f"{strategy_name} on {ticker}: {metric}={score}, trades={results.get('# Trades', 0)}")
            
            all_results[strategy_name] = {
                "score": score,
                "results": results,
                "backtest_obj": backtest_obj,
            }

            print(f"    {strategy_name}: {metric.capitalize()} = {score}")

            if score > best_score:
                best_score = score
                best_strategy = strategy_name
                best_backtest = backtest_obj
                logger.info(f"New best strategy for {ticker}: {strategy_name} with {metric}={score}")
        
        except Exception as e:
            logger.error(f"Error testing {strategy_name} on {ticker}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"    ❌ Error testing {strategy_name}: {e}")

    logger.info(f"Best strategy for {ticker}: {best_strategy} ({metric}: {best_score})")
    print(f"✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})")

    # Plot the best strategy if requested
    if plot and best_backtest:
        # Create output directory if it doesn't exist
        os.makedirs("reports_output", exist_ok=True)

        # Generate filename based on ticker and strategy
        output_path = f"reports_output/{ticker}_{best_strategy}_backtest.html"
        
        logger.info(f"Plotting best strategy for {ticker}: {best_strategy} to {output_path}")

        try:
            # Create plot with specified parameters
            html = best_backtest.plot(
                open_browser=False,
                plot_return=True,
                plot_drawdown=True,
                filename=output_path,
                resample=resample,  # Add this parameter
            )

            print(f"🌐 Plot for best strategy saved to: {output_path}")
            logger.info(f"Plot for best strategy saved to: {output_path}")

            # Open in browser if requested
            if plot:
                logger.info(f"Opening plot in browser: {os.path.abspath(output_path)}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)
        except Exception as e:
            logger.error(f"Error plotting best strategy: {e}")
            import traceback
            logger.error(traceback.format_exc())
            print(f"❌ Error plotting best strategy: {e}")

    return {
        "strategies": all_results,
        "best_strategy": best_strategy,
        "best_score": best_score,
    }

def backtest_portfolio_optimal(args):
    """Find optimal strategy and interval for each asset in a portfolio."""
    try:
        print("Starting portfolio optimization")
        
        # Setup logging if requested
        log_file = None
        if hasattr(args, 'log') and args.log:
            print("Setting up logging")
            log_file = setup_command_logging(args)
            logger.info("Portfolio optimization started")
        
        print(f"Getting portfolio config for '{args.name}'")
        logger.info(f"Getting portfolio config for '{args.name}'")
        portfolio_config = get_portfolio_config(args.name)
        if not portfolio_config:
            logger.error(f"Portfolio '{args.name}' not found in assets_config.json")
            print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
            return {}

        print("Getting intervals")
        logger.info(f"Using intervals: {args.intervals}")
        intervals = args.intervals if args.intervals else ["1d"]
        
        print(f"Finding optimal strategy-interval combinations for portfolio '{args.name}'")
        logger.info(f"Finding optimal strategy-interval combinations for portfolio '{args.name}'")
        
        # Initialize dictionaries to store results
        best_combinations = {}
        all_results = {}
        
        print("Getting available strategies")
        # Restore the original strategy factory call
        strategies = StrategyFactory.get_available_strategies()
        logger.info(f"Testing {len(strategies)} strategies")
        
        # Process each asset
        assets = portfolio_config.get("assets", [])
        print(f"Processing {len(assets)} assets")
        logger.info(f"Processing {len(assets)} assets")
        
        for asset_config in assets:
            ticker = asset_config["ticker"]
            print(f"Processing asset: {ticker}")
            logger.info(f"Processing asset: {ticker}")
            
            # Call the backtest function for this asset
            asset_results = _backtest_all_strategies_all_timeframes(
                ticker=ticker,
                asset_config=asset_config,
                strategies=strategies,
                intervals=intervals,
                period=args.period,
                metric=args.metric,
                start_date=getattr(args, 'start_date', None),
                end_date=getattr(args, 'end_date', None),
                plot=getattr(args, 'plot', False),
                resample=getattr(args, 'resample', None),
            )
            
            best_combinations[ticker] = asset_results["best_combination"]
            all_results[ticker] = asset_results["all_results"]
        
        # Generate report only if not plotting individual results
        if not getattr(args, 'plot', False):
            print("Generating report")
            logger.info("Generating portfolio optimization report")
            report_data = {
                "portfolio": args.name,
                "description": portfolio_config.get("description", ""),
                "best_combinations": best_combinations,
                "all_results": all_results,
                "metric": args.metric,
                "intervals": intervals,
                "strategies": strategies,
                "is_portfolio_optimal": True,
            }

            output_path = f"reports_output/portfolio_optimal_{args.name}.html"
            
            # Use the detailed portfolio report template instead
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            generator = ReportGenerator()
            
            # Use generate_detailed_portfolio_report instead of generate_report
            generator.generate_detailed_portfolio_report(report_data, output_path)
            
            print(f"📄 Portfolio Optimization Report saved to: {output_path}")
            logger.info(f"Portfolio Optimization Report saved to: {output_path}")
            
            # Open the report in the browser if requested
            if args.open_browser:
                logger.info(f"Opening report in browser: {os.path.abspath(output_path)}")
                webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)
        
        print("Portfolio optimization completed successfully")
        logger.info("Portfolio optimization completed successfully")
        return best_combinations
        
    except RecursionError as e:
        print(f"RecursionError: {e}")
        if 'logger' in globals():
            logger.error(f"RecursionError: {e}")
        import traceback
        trace = traceback.format_exc()
        print(trace)
        if 'logger' in globals():
            logger.error(trace)
        return {}
    except Exception as e:
        print(f"Error in backtest_portfolio_optimal: {e}")
        if 'logger' in globals():
            logger.error(f"Error in backtest_portfolio_optimal: {e}")
        import traceback
        trace = traceback.format_exc()
        print(trace)
        if 'logger' in globals():
            logger.error(trace)
        return {}


def _backtest_all_strategies_all_timeframes(
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
        initial_capital = asset_config.get("initial_capital", defaults["initial_capital"])
        asset_period = asset_config.get("period", period)
        
        logger.info(f"Using commission: {commission}, initial capital: {initial_capital}")
        print(f"\n🔍 Testing all strategies and timeframes for: {ticker}")
        print(f"Using commission: {commission}, initial capital: {initial_capital}")
        
        # Track best combination across all strategies and intervals
        best_score = -float("inf")
        best_strategy = None
        best_interval = None
        best_result = None
        
        # Structure to hold all results
        all_results = {
            "ticker": ticker,
            "strategies": []
        }
        
        # Test each strategy
        for strategy_name in strategies:
            logger.info(f"Testing strategy {strategy_name} for {ticker}")
            print(f"  Testing strategy {strategy_name}...")
            
            strategy_entry = {
                "name": strategy_name,
                "best_timeframe": None,
                "best_score": -float("inf"),
                "timeframes": []
            }
            
            # Test each interval
            for interval in intervals:
                logger.info(f"Testing {strategy_name} with {interval} interval for {ticker}")
                print(f"    Testing {interval} interval...")
                
                try:
                    # Load data for this ticker and interval
                    data = DataLoader.load_data(
                        ticker,
                        period=asset_period,
                        interval=interval,
                        start=start_date,
                        end=end_date,
                    )
                    
                    if data is None or data.empty:
                        logger.warning(f"No data available for {ticker} with {interval} interval")
                        print(f"      ⚠️ No data available for {interval}")
                        continue
                    
                    logger.debug(f"Loaded {len(data)} data points for {ticker} with {interval} interval")
                    
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
                    
                    logger.info(f"{strategy_name} + {interval} for {ticker}: {metric}={score}, Trades={trades}")
                    print(f"      {strategy_name} + {interval}: {metric}={score}, Trades={trades}")
                    
                    # Extract detailed metrics
                    detailed_metrics = _extract_detailed_metrics(result, initial_capital)
                    
                    # Add timeframe results
                    timeframe_data = {
                        "interval": interval,
                        "score": score,
                        "return_pct": detailed_metrics.get("return_pct", 0),
                        "win_rate": detailed_metrics.get("win_rate", 0),
                        "trades_count": detailed_metrics.get("trades_count", 0),
                        "profit_factor": detailed_metrics.get("profit_factor", 0),
                        "max_drawdown_pct": detailed_metrics.get("max_drawdown_pct", 0),
                        "sharpe_ratio": detailed_metrics.get("sharpe_ratio", 0),
                        "equity_curve": detailed_metrics.get("equity_curve", []),
                        "trades": detailed_metrics.get("trades", [])
                    }
                    
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
                    logger.error(f"Error testing {strategy_name} with {interval} for {ticker}: {e}")
                    print(f"      ❌ Error: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            # Add strategy results to all_results
            all_results["strategies"].append(strategy_entry)
        
        # Create best combination data structure
        if best_strategy and best_interval and best_result:
            best_combination = {
                "strategy": best_strategy,
                "interval": best_interval,
                "score": best_score,
                **best_result  # Include all detailed metrics
            }
            logger.info(f"Best combination for {ticker}: {best_strategy} with {interval}, {metric}={best_score}")
            print(f"  ✅ Best combination: {best_strategy} with {best_interval}, {metric}={best_score}")
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
                "sharpe_ratio": 0
            }
            logger.warning(f"No valid strategy-interval combination found for {ticker}")
            print(f"  ⚠️ No valid strategy-interval combination found")
        
        # Plot the best backtest if requested
        if plot and best_strategy and best_interval:
            try:
                # Load data and run backtest again for plotting
                data = DataLoader.load_data(
                    ticker,
                    period=asset_period,
                    interval=best_interval,
                    start=start_date,
                    end=end_date,
                )
                
                strategy_class = StrategyFactory.get_strategy(best_strategy)
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
                output_path = f"reports_output/{ticker}_{best_strategy}_{best_interval}_backtest.html"
                
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
                if plot:
                    logger.info(f"Opening plot in browser: {os.path.abspath(output_path)}")
                    webbrowser.open(f"file://{os.path.abspath(output_path)}", new=2)
            
            except Exception as e:
                logger.error(f"Error plotting best combination for {ticker}: {e}")
                print(f"  ❌ Error plotting best combination: {e}")
                import traceback
                logger.error(traceback.format_exc())
        
        return {
            "best_combination": best_combination,
            "all_results": all_results
        }
        
    except Exception as e:
        logger.error(f"Error in _backtest_all_strategies_all_timeframes for {ticker}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return {
            "best_combination": {
                "strategy": None,
                "interval": None,
                "score": -float("inf"),
                "error": f"Error: {str(e)}"
            },
            "all_results": {
                "ticker": ticker,
                "error": str(e),
                "strategies": []
            }
        }


def register_commands(subparsers):
    """Register portfolio commands with the CLI parser"""
    # Portfolio command
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Backtest all assets in a portfolio with all strategies"
    )
    portfolio_parser.add_argument(
        "--name", type=str, required=True, help="Portfolio name from assets_config.json"
    )
    portfolio_parser.add_argument(
        "--period",
        type=str,
        default="max",
        help="Default data period (can be overridden by portfolio settings)",
    )
    portfolio_parser.add_argument(
        "--metric",
        type=str,
        default="profit_factor",
        help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)",
    )
    portfolio_parser.add_argument(
        "--plot",
        action="store_true",
        help="Use backtesting.py's plot() method to display results in browser",
    )
    portfolio_parser.add_argument(
        "--resample",
        type=str,
        default=None,
        help="Resample period for plotting (e.g., '1D', '4H', '1W')",
    )
    portfolio_parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Automatically open the generated report in a browser",
    )
    # Add the log option to portfolio command
    portfolio_parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging of command output"
    )
    portfolio_parser.set_defaults(func=backtest_portfolio)

    # Portfolio optimization command
    portfolio_optimal_parser = subparsers.add_parser(
        "portfolio-optimal", help="Find optimal strategy/timeframe combinations for a portfolio"
    )
    portfolio_optimal_parser.add_argument(
        "--name", required=True, help="Portfolio name from assets_config.json"
    )
    portfolio_optimal_parser.add_argument(
        "--intervals", nargs="+", default=["1d"], help="Intervals to test (e.g., 1d 1wk 1mo)"
    )
    portfolio_optimal_parser.add_argument(
        "--period", default="max", help="Data period to fetch"
    )
    portfolio_optimal_parser.add_argument(
        "--metric", default="sharpe", help="Metric to optimize for (sharpe, return, profit_factor)"
    )
    portfolio_optimal_parser.add_argument(
        "--open-browser", action="store_true", help="Open report in browser after completion"
    )
    # Add the log option
    portfolio_optimal_parser.add_argument(
        "--log", action="store_true", help="Enable detailed logging of command output"
    )
    # Add start_date and end_date parameters
    portfolio_optimal_parser.add_argument(
        "--start-date", dest="start_date", help="Start date for backtest (YYYY-MM-DD)"
    )
    portfolio_optimal_parser.add_argument(
        "--end-date", dest="end_date", help="End date for backtest (YYYY-MM-DD)"
    )
    # Add plot and resample parameters
    portfolio_optimal_parser.add_argument(
        "--plot", action="store_true", help="Plot the best strategy for each asset"
    )
    portfolio_optimal_parser.add_argument(
        "--resample", type=str, default=None, 
        help="Resample period for plotting (e.g., '1D', '4H', '1W')"
    )
    # Add require_complete_history parameter
    portfolio_optimal_parser.add_argument(
        "--require-complete-history", dest="require_complete_history", 
        type=bool, default=None,
        help="Require complete history for backtest"
    )
    portfolio_optimal_parser.set_defaults(func=backtest_portfolio_optimal)

def _extract_detailed_metrics(result, initial_capital):
    """Extract and format detailed metrics from backtest result."""
    logger.debug(f"Extracting detailed metrics from backtest result")
    
    # Check for NaN values and replace them with defaults
    def safe_get(key, default):
        val = result.get(key, default)
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            logger.warning(f"Found NaN or Inf value for {key}, using default: {default}")
            return default
        return val
    
    detailed_metrics = {
        'initial_capital': initial_capital,
        'profit_factor': safe_get('Profit Factor', 0),
        'tv_profit_factor': safe_get('Profit Factor', 'N/A'),
        'return': f"{safe_get('Return [%]', 0):.2f}%",
        'return_pct': safe_get('Return [%]', 0),
        'win_rate': safe_get('Win Rate [%]', 0),
        'max_drawdown': f"{safe_get('Max. Drawdown [%]', 0):.2f}%",
        'max_drawdown_pct': safe_get('Max. Drawdown [%]', 0),
        'trades_count': safe_get('# Trades', 0),
        'equity_final': safe_get('Equity Final [$]', initial_capital),
        'buy_hold_return': safe_get('Buy & Hold Return [%]', 0),
        'sharpe_ratio': safe_get('Sharpe Ratio', 0),
        'sortino_ratio': safe_get('Sortino Ratio', 0),
        'calmar_ratio': safe_get('Calmar Ratio', 0),
        'volatility': safe_get('Volatility (Ann.) [%]', 0),
        'exposure_time': safe_get('Exposure Time [%]', 0),
        'avg_trade_pct': safe_get('Avg. Trade [%]', 0),
        'best_trade_pct': safe_get('Best Trade [%]', 0),
        'worst_trade_pct': safe_get('Worst Trade [%]', 0),
        'avg_trade_duration': safe_get('Avg. Trade Duration', 'N/A'),
        'sqn': safe_get('SQN', 0)
    }
    
    logger.debug(f"Extracted basic metrics: profit_factor={detailed_metrics['profit_factor']}, return={detailed_metrics['return_pct']}%, win_rate={detailed_metrics['win_rate']}%")

    # Process trades into list format expected by template
    if "_trades" in result and not result["_trades"].empty:
        trades_df = result["_trades"]
        trades_list = []
        logger.debug(f"Processing {len(trades_df)} trades")

        for _, trade in trades_df.iterrows():
            try:
                trade_data = {
                    "entry_date": str(trade["EntryTime"]),
                    "exit_date": str(trade["ExitTime"]),
                    "type": "LONG",  # Assuming all trades are LONG
                    "entry_price": float(trade["EntryPrice"]),
                    "exit_price": float(trade["ExitPrice"]),
                    "size": int(trade["Size"]),
                    "pnl": float(trade["PnL"]),
                    "return_pct": float(trade["ReturnPct"]) * 100,
                    "duration": trade["Duration"],
                }
                trades_list.append(trade_data)
            except Exception as e:
                logger.error(f"Error processing trade: {e}")
                logger.error(f"Trade data: {trade}")

        detailed_metrics["trades"] = trades_list
        detailed_metrics["total_pnl"] = sum(trade["pnl"] for trade in trades_list)
        
        # Calculate additional trade statistics
        if trades_list:
            winning_trades = [t for t in trades_list if t["pnl"] > 0]
            losing_trades = [t for t in trades_list if t["pnl"] < 0]
            
            win_count = len(winning_trades)
            loss_count = len(losing_trades)
            
            logger.debug(f"Trade statistics: {win_count} winning, {loss_count} losing")
            
            if winning_trades:
                avg_win = sum(t["pnl"] for t in winning_trades) / len(winning_trades)
                max_win = max(t["pnl"] for t in winning_trades)
                detailed_metrics["avg_win"] = avg_win
                detailed_metrics["max_win"] = max_win
                logger.debug(f"Average win: ${avg_win:.2f}, Max win: ${max_win:.2f}")
            
            if losing_trades:
                avg_loss = sum(t["pnl"] for t in losing_trades) / len(losing_trades)
                max_loss = min(t["pnl"] for t in losing_trades)
                detailed_metrics["avg_loss"] = avg_loss
                detailed_metrics["max_loss"] = max_loss
                logger.debug(f"Average loss: ${avg_loss:.2f}, Max loss: ${max_loss:.2f}")
    else:
        # Make sure we have an empty list if no trades
        logger.debug("No trades found in backtest result")
        detailed_metrics['trades'] = []
        detailed_metrics['total_pnl'] = 0

    # Process equity curve
    if "_equity_curve" in result:
        equity_data = result["_equity_curve"]
        equity_curve = []
        logger.debug(f"Processing equity curve with {len(equity_data)} points")

        try:
            # Handle different equity curve data structures
            if isinstance(equity_data, pd.DataFrame):
                for date, row in equity_data.iterrows():
                    val = (
                        row.iloc[0] if isinstance(row, pd.Series) and len(row) > 0 else row
                    )
                    equity_curve.append(
                        {
                            "date": str(date),
                            "value": float(val) if not pd.isna(val) else 0.0,
                        }
                    )
            else:
                for date, val in zip(equity_data.index, equity_data.values):
                    # Handle numpy values
                    if hasattr(val, "item"):
                        try:
                            val = val.item()
                        except (ValueError, TypeError):
                            val = val[0] if len(val) > 0 else 0

                    equity_curve.append(
                        {
                            "date": str(date),
                            "value": float(val) if not pd.isna(val) else 0.0,
                        }
                    )

            detailed_metrics["equity_curve"] = equity_curve
            logger.debug(f"Extracted {len(equity_curve)} equity curve points")
            
            # Verify equity curve data quality
            if equity_curve:
                min_value = min(point["value"] for point in equity_curve)
                max_value = max(point["value"] for point in equity_curve)
                logger.debug(f"Equity curve range: {min_value} to {max_value}")
                
                # Check for suspicious values
                if min_value < 0:
                    logger.warning(f"Negative values detected in equity curve: minimum = {min_value}")
                if max_value == 0:
                    logger.warning("All equity curve values are zero")
        except Exception as e:
            logger.error(f"Error processing equity curve: {e}")
            import traceback
            logger.error(traceback.format_exc())
            detailed_metrics["equity_curve"] = []

    return detailed_metrics

def _generate_log_summary(portfolio_name, best_combinations, metric):
    """Generate a comprehensive summary of portfolio optimization results for logging."""
    logger.info("=" * 50)
    logger.info(f"PORTFOLIO OPTIMIZATION SUMMARY FOR '{portfolio_name}'")
    logger.info("=" * 50)
    
    # Calculate overall portfolio statistics
    total_assets = len(best_combinations)
    assets_with_valid_strategy = sum(1 for combo in best_combinations.values() if combo.get('strategy') is not None)
    
    logger.info(f"Total assets: {total_assets}")
    logger.info(f"Assets with valid strategy: {assets_with_valid_strategy} ({assets_with_valid_strategy/total_assets*100 if total_assets else 0:.1f}%)")
    logger.info(f"Optimization metric: {metric}")
    
    # Calculate average metrics across portfolio
    if assets_with_valid_strategy > 0:
        avg_return = sum(combo.get('return_pct', 0) for combo in best_combinations.values() if combo.get('strategy') is not None) / assets_with_valid_strategy
        avg_win_rate = sum(combo.get('win_rate', 0) for combo in best_combinations.values() if combo.get('strategy') is not None) / assets_with_valid_strategy
        avg_profit_factor = sum(combo.get('profit_factor', 0) for combo in best_combinations.values() if combo.get('strategy') is not None) / assets_with_valid_strategy
        avg_trades = sum(combo.get('trades_count', 0) for combo in best_combinations.values() if combo.get('strategy') is not None) / assets_with_valid_strategy
        
        logger.info(f"Average return: {avg_return:.2f}%")
        logger.info(f"Average win rate: {avg_win_rate:.2f}%")
        logger.info(f"Average profit factor: {avg_profit_factor:.2f}")
        logger.info(f"Average trades per asset: {avg_trades:.1f}")
    
    # Strategy distribution
    strategy_counts = {}
    interval_counts = {}
    
    for combo in best_combinations.values():
        if combo.get('strategy') is not None:
            strategy = combo.get('strategy')
            interval = combo.get('interval')
            
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            interval_counts[interval] = interval_counts.get(interval, 0) + 1
    
    logger.info("\nStrategy distribution:")
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {strategy}: {count} assets ({count/assets_with_valid_strategy*100:.1f}%)")
    
    logger.info("\nInterval distribution:")
    for interval, count in sorted(interval_counts.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {interval}: {count} assets ({count/assets_with_valid_strategy*100:.1f}%)")
    
    # Individual asset results
    logger.info("\nIndividual asset results:")
    for ticker, combo in sorted(best_combinations.items()):
        if combo.get('strategy') is not None:
            logger.info(f"  {ticker}: {combo['strategy']} with {combo['interval']} interval")
            logger.info(f"    {metric}: {combo['score']:.4f}, Return: {combo.get('return_pct', 0):.2f}%, Win Rate: {combo.get('win_rate', 0):.1f}%")
            logger.info(f"    Trades: {combo.get('trades_count', 0)}, Profit Factor: {combo.get('profit_factor', 0):.2f}")
        else:
            logger.info(f"  {ticker}: No valid strategy found")
    
    logger.info("=" * 50)

def _save_backtest_results(ticker, strategy, interval, results, initial_capital):
    """Save detailed backtest results to a file for later analysis."""
    try:
        # Create a directory for backtest results
        results_dir = os.path.join("logs", "backtest_results")
        os.makedirs(results_dir, exist_ok=True)
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{ticker}_{strategy}_{interval}_{timestamp}.json"
        filepath = os.path.join(results_dir, filename)
        
        # Extract key metrics for saving
        metrics = _extract_detailed_metrics(results, initial_capital)
        
        # Remove non-serializable objects like backtest_obj
        if "backtest_obj" in metrics:
            del metrics["backtest_obj"]
        
        # Add metadata
        metrics["ticker"] = ticker
        metrics["strategy"] = strategy
        metrics["interval"] = interval
        metrics["timestamp"] = timestamp
        
        # Limit the size of equity curve for storage
        if "equity_curve" in metrics and len(metrics["equity_curve"]) > 1000:
            # Sample the equity curve to reduce size
            sample_rate = max(1, len(metrics["equity_curve"]) // 1000)
            metrics["equity_curve"] = metrics["equity_curve"][::sample_rate]
            logger.debug(f"Sampled equity curve from {len(metrics['equity_curve'])} to {len(metrics['equity_curve'][::sample_rate])} points for storage")
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
            
        logger.info(f"Saved detailed backtest results to {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"Error saving backtest results: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
