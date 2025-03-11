import argparse
import json
import os
from src.backtesting_engine.strategy_runner import StrategyRunner
from src.optimizer.optimization_runner import OptimizationRunner
from src.reports.report_generator import ReportGenerator
from src.reports.report_formatter import ReportFormatter
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.backtesting_engine.result_analyzer import BacktestResultAnalyzer
from src.utils.config_manager import ConfigManager

def load_assets_config():
    """Load the assets configuration from config/assets_config.json"""
    config_path = os.path.join('config', 'assets_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return {"portfolios": {}}

def is_portfolio(ticker):
    """Check if the given ticker is a portfolio name in assets_config.json"""
    assets_config = load_assets_config()
    return ticker in assets_config.get('portfolios', {})
def get_portfolio_config(portfolio_name):
    """Get configuration for a specific portfolio"""
    assets_config = load_assets_config()
    return assets_config.get('portfolios', {}).get(portfolio_name, None)

def list_portfolios():
    """List all available portfolios from assets_config.json"""
    assets_config = load_assets_config()
    portfolios = assets_config.get('portfolios', {})
    
    if not portfolios:
        print("No portfolios found in config/assets_config.json")
        return
    
    print("\n📂 Available Portfolios:")
    print("-" * 80)
    for name, config in portfolios.items():
        assets = ", ".join([asset['ticker'] for asset in config.get('assets', [])])
        print(f"📊 {name}: {config.get('description', 'No description')}")
        print(f"   🔸 Assets: {assets}")
        print("-" * 80)

def list_strategies():
    """List all available trading strategies"""
    factory = StrategyFactory()
    strategies = factory.get_available_strategies()  # Implement this method in StrategyFactory
    
    print("\n📈 Available Trading Strategies:")
    print("-" * 80)
    for strategy_name in strategies:
        print(f"🔹 {strategy_name}")
    print("-" * 80)

def backtest_single(strategy, ticker, period="max", commission=0.001, initial_capital=10000):
    """Run a backtest for a single asset with a single strategy."""
    print(f"Running backtest for {strategy} on {ticker} with period={period}...")
    
    results = StrategyRunner.execute(
        strategy, 
        ticker, 
        period=period,
        commission=commission, 
        initial_capital=initial_capital
    )
    
    output_path = f"reports_output/backtest_{strategy}_{ticker}.html"
    generator = ReportGenerator()
    generator.generate_report(results, "backtest_report.html", output_path)
    
    print(f"📄 HTML report generated at: {output_path}")
    return results

def backtest_portfolio_optimal(portfolio_name, intervals=None, period="max", metric="sharpe"):
    """
    Find the optimal strategy and interval combination for each asset in a portfolio.
    
    Args:
        portfolio_name: Name of the portfolio from assets_config.json
        intervals: List of bar intervals to test
        period: How far back to test
        metric: Performance metric to use for optimization
        
    Returns:
        Dictionary with best strategy-interval combination for each asset
    """
    # Get configuration for complete history requirement
    config = ConfigManager()
    require_complete_history = config.get('backtest.require_complete_history', True)
    
    if intervals is None:
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]
    
    portfolio_config = get_portfolio_config(portfolio_name)
    
    if not portfolio_config:
        print(f"❌ Portfolio '{portfolio_name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}
    
    print(f"🔍 Finding optimal strategy-interval combinations for portfolio '{portfolio_name}'")
    print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
    
    # Get all available strategies
    factory = StrategyFactory()
    strategies = factory.get_available_strategies()
    
    assets = portfolio_config.get('assets', [])
    print(f"Testing {len(strategies)} strategies × {len(intervals)} intervals on {len(assets)} assets")
    print(f"This will run {len(strategies) * len(intervals) * len(assets)} backtests - might take a while!")
    
    # Track the best combination for each asset
    best_combinations = {}
    all_results = {}
    
    # For each asset in the portfolio
    for asset_config in assets:
        ticker = asset_config['ticker']
        asset_period = asset_config.get('period', period)
        commission = asset_config.get('commission', 0.001)
        initial_capital = asset_config.get('initial_capital', 10000)
        
        print(f"\n🔍 Finding optimal combination for: {ticker}")
        
        # Determine which intervals to test for this asset
        valid_intervals = []
        if require_complete_history:
            # Get the complete history of the stock for validation
            daily_data = DataLoader.load_data(ticker, period="max", interval="1d")
            if daily_data is None or daily_data.empty:
                print(f"❌ No daily data available for {ticker}")
                continue
                
            stock_start_date = daily_data.index.min()
            print(f"📊 {ticker} daily data starts at {stock_start_date}")
            
            # Filter intervals that have data from the beginning
            for interval in intervals:
                # Daily data is always valid
                if interval == "1d":
                    valid_intervals.append(interval)
                    continue
                    
                # For other intervals, check the data
                try:
                    interval_data = DataLoader.load_data(
                        ticker, 
                        period="max", 
                        interval=interval
                    )
                    
                    if interval_data is None or interval_data.empty:
                        print(f"  ⚠️ Skipping {interval} - no data available")
                        continue
                        
                    interval_start_date = interval_data.index.min()
                    has_complete_history = interval_start_date <= stock_start_date
                    
                    if has_complete_history:
                        valid_intervals.append(interval)
                        print(f"  ✅ {interval} has complete history from {interval_start_date}")
                    else:
                        print(f"  ⚠️ Skipping {interval} - data starts at {interval_start_date}, needed from {stock_start_date}")
                        
                except Exception as e:
                    print(f"  ❌ Error checking {interval} data: {str(e)}")
            
            # If we have no valid intervals, use just daily data
            if not valid_intervals:
                print(f"  ⚠️ No intervals have complete data, falling back to daily timeframe")
                valid_intervals = ["1d"]
        else:
            # If we don't require complete history, use all intervals
            valid_intervals = intervals
        
        print(f"Testing {len(strategies)} strategies × {len(valid_intervals)} intervals")
        
        best_score = -float('inf')
        best_strategy = None
        best_interval = None
        asset_results = {}
        
        # Test each strategy with each interval
        for strategy_name in strategies:
            if strategy_name not in asset_results:
                asset_results[strategy_name] = {}
                
            for interval in valid_intervals:
                print(f"  Testing {strategy_name} with {interval} interval...")
                
                try:
                    # Load data with specific interval
                    data = DataLoader.load_data(
                        ticker, 
                        period=asset_period, 
                        interval=interval
                    )
                    
                    if data is None or data.empty:
                        print(f"    ⚠️ No data available for {ticker} with {interval} interval")
                        continue
                        
                    print(f"    ✅ Loaded {len(data)} {interval} bars")
                    
                    # Get strategy class
                    strategy_class = StrategyFactory.get_strategy(strategy_name)
                    
                    # Run backtest with this strategy and interval
                    engine = BacktestEngine(
                        strategy_class, 
                        data, 
                        cash=initial_capital,
                        commission=commission,
                        ticker=ticker
                    )
                    
                    raw_result = engine.run()
                    
                    # Process results
                    result = BacktestResultAnalyzer.analyze(
                        raw_result,
                        ticker=ticker,
                        initial_capital=initial_capital
                    )
                    
                    # Add additional information
                    result['interval'] = interval
                    result['strategy'] = strategy_name
                    
                    # Extract performance metric
                    if metric == "sharpe":
                        score = result.get('sharpe_ratio', 0)
                    elif metric == "return":
                        score = result.get('return_pct', 0)
                        if isinstance(score, str) and score.endswith('%'):
                            score = float(score.strip('%'))
                    else:
                        score = result.get(metric, 0)
                        
                    # Store result
                    asset_results[strategy_name][interval] = {
                        'score': score,
                        'result': result
                    }
                    
                    # Get number of trades
                    trade_count = result.get('trades', 0)
                    if isinstance(trade_count, str) and trade_count.isdigit():
                        trade_count = int(trade_count)
                        
                    print(f"    {strategy_name} + {interval}: {metric.capitalize()} = {score}, Trades = {trade_count}")
                    
                    # Track the best combination (only if it has trades)
                    if score > best_score and trade_count > 0:
                        best_score = score
                        best_strategy = strategy_name
                        best_interval = interval
                
                except Exception as e:
                    print(f"    ❌ Error testing {strategy_name} with {interval}: {str(e)}")
                    asset_results[strategy_name][interval] = {
                        'error': str(e),
                        'score': -float('inf')
                    }
        
        # If no valid combination found, try with any combination that has trades
        if best_strategy is None:
            print(f"⚠️ No optimal combination with trades found for {ticker}, looking for any strategy with trades...")
            
            for strategy_name in strategies:
                for interval in valid_intervals:
                    if (strategy_name in asset_results and 
                        interval in asset_results[strategy_name] and
                        isinstance(asset_results[strategy_name][interval], dict) and
                        'result' in asset_results[strategy_name][interval]):
                        
                        result = asset_results[strategy_name][interval]['result']
                        trade_count = result.get('trades', 0)
                        if isinstance(trade_count, str) and trade_count.isdigit():
                            trade_count = int(trade_count)
                            
                        if trade_count > 0:
                            best_strategy = strategy_name
                            best_interval = interval
                            best_score = asset_results[strategy_name][interval]['score']
                            print(f"  Found fallback: {best_strategy} + {best_interval} with {trade_count} trades")
                            break
                
                if best_strategy is not None:
                    break
        
        # If still no combination found, use the one with the best score regardless of trades
        if best_strategy is None:
            print(f"⚠️ Warning: No strategies with trades found for {ticker}, using best available score")
            
            for strategy_name in strategies:
                for interval in valid_intervals:
                    if (strategy_name in asset_results and 
                        interval in asset_results[strategy_name] and
                        isinstance(asset_results[strategy_name][interval], dict) and
                        'score' in asset_results[strategy_name][interval]):
                        
                        score = asset_results[strategy_name][interval]['score']
                        if score > best_score:
                            best_score = score
                            best_strategy = strategy_name
                            best_interval = interval
        
        # Save best combination
        if best_strategy is not None and best_interval is not None:
            best_combinations[ticker] = {
                'strategy': best_strategy,
                'interval': best_interval,
                'score': best_score
            }
            
            # Get additional info about the best combination
            if (best_strategy in asset_results and 
                best_interval in asset_results[best_strategy] and
                'result' in asset_results[best_strategy][best_interval]):
                
                result = asset_results[best_strategy][best_interval]['result']
                trade_count = result.get('trades', 0)
                return_pct = result.get('return_pct', '0%')
                max_drawdown = result.get('max_drawdown', '0%')
                
                # Add all detailed metrics from the results
                best_combinations[ticker]['trades'] = trade_count
                best_combinations[ticker]['return'] = return_pct
                best_combinations[ticker]['max_drawdown'] = max_drawdown
                
                # Add the detailed trade data for charts and tables
                best_combinations[ticker]['equity_curve'] = result.get('equity_curve', [])
                best_combinations[ticker]['drawdown_curve'] = result.get('drawdown_curve', [])
                best_combinations[ticker]['trades_list'] = result.get('trades_list', [])
                best_combinations[ticker]['win_rate'] = result.get('win_rate', 0)
                best_combinations[ticker]['profit_factor'] = result.get('profit_factor', 0)
                best_combinations[ticker]['avg_win'] = result.get('avg_win', 0)
                best_combinations[ticker]['avg_loss'] = result.get('avg_loss', 0)
                best_combinations[ticker]['initial_capital'] = result.get('initial_capital', 0)
                best_combinations[ticker]['final_value'] = result.get('final_value', 0)
                best_combinations[ticker]['total_pnl'] = result.get('final_value', 0) - result.get('initial_capital', 0)

            
            print(f"  ✅ Best for {ticker}: {best_strategy} + {best_interval} ({metric}: {best_score})")
        else:
            best_combinations[ticker] = {
                'strategy': None,
                'interval': None,
                'score': -float('inf'),
                'error': "No valid combinations found"
            }
            print(f"  ❌ No valid combination found for {ticker}")
        
        # Save all results
        all_results[ticker] = asset_results
    
    # Generate optimization report
    report_data = {
        'portfolio': portfolio_name,
        'description': portfolio_config.get('description', ''),
        'best_combinations': best_combinations,
        'all_results': all_results,
        'metric': metric,
        'intervals': valid_intervals,
        'strategies': strategies,
        'is_portfolio_optimal': True
    }

    # Create a list of assets with their optimal combinations for the report
    asset_list = []
    for ticker, data in best_combinations.items():
        if not data['strategy'] or not data['interval']:
            continue

        # Get the trades count
        trades = data.get('trades', 0)
        if isinstance(trades, str) and trades.isdigit():
            trades = int(trades)

        # Skip assets with 0 trades
        if trades == 0:
            print(f"  ⚠️ Skipping {ticker} from report - strategy has 0 trades")
            continue

        asset_list.append({
            'name': ticker,
            'strategy': data['strategy'],
            'interval': data['interval'],
            'score': data['score'],
            'return': data.get('return', '0%'),
            'trades': trades,
            'max_drawdown': data.get('max_drawdown', '0%'),
            'win_rate': data.get('win_rate', 0),
            'total_pnl': data.get('final_value', 0) - data.get('initial_capital', 0)
        })

    report_data['asset_list'] = asset_list

    # Generate the report
    output_path = f"reports_output/portfolio_optimal_{portfolio_name}.html"

    # When generating the report, add more robust error handling
    try:
        generator = ReportGenerator()
        generator.generate_report(report_data, "portfolio_optimal_report.html", output_path)
        print(f"\n📄 Portfolio Optimization Report saved to {output_path}")
    except Exception as e:
        import traceback
        print(f"❌ Error generating report: {e}")
        print(traceback.format_exc())
        print(f"Saving minimal error report instead")
        
        # Create a minimal HTML error report
        error_html = f"""
        <!DOCTYPE html>
        <html><head><title>Error Report</title></head>
        <body>
            <h1>Error Running Portfolio Optimization</h1>
            <p>Error: {str(e)}</p>
            <h2>Portfolio: {portfolio_name}</h2>
            <p>Best combinations found (may be incomplete):</p>
            <pre>{json.dumps(best_combinations, indent=2)}</pre>
        </body></html>
        """
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(error_html)
    
    return best_combinations


def backtest_all_strategies(ticker, period="max", metric="profit_factor", commission=0.001, initial_capital=10000):
    """Run a backtest for a single asset with all available strategies."""
    print(f"Testing all strategies on asset {ticker}")
    
    factory = StrategyFactory()
    strategies = factory.get_available_strategies()
    print(f"Testing {len(strategies)} strategies")
    
    best_score = -float('inf')
    best_strategy = None
    all_results = {}
    
    for strategy_name in strategies:
        print(f"  Testing {strategy_name}...")
        
        results = StrategyRunner.execute(
            strategy_name, 
            ticker,
            period=period,
            commission=commission,
            initial_capital=initial_capital
        )
        
        if metric == "profit_factor":
            score = results.get('profit_factor', 0)
        elif metric == "sharpe":
            score = results.get('sharpe_ratio', 0)
        elif metric == "return":
            score = results.get('return_pct', 0)
        else:
            score = results.get(metric, 0)
            
        all_results[strategy_name] = {
            'score': score,
            'results': results
        }
        
        # Filter out strategies with 0 trades
        valid_strategies = {}
        for strategy_name, strategy_data in all_results.items():
            results = strategy_data['results']
            trade_count = results.get('trades', 0)
            
            if isinstance(trade_count, str) and trade_count.isdigit():
                trade_count = int(trade_count)
                
            if trade_count > 0:
                valid_strategies[strategy_name] = strategy_data
            else:
                print(f"  ⚠️ Skipping {strategy_name} from report - has 0 trades")
        
        # Check if best strategy was filtered out
    if best_strategy not in valid_strategies and valid_strategies:
        # Select a new best strategy from valid strategies
        new_best_score = -float('inf')
        new_best_strategy = None
        for strategy_name, strategy_data in valid_strategies.items():
            if strategy_data['score'] > new_best_score:
                new_best_score = strategy_data['score']
                new_best_strategy = strategy_name
        
        if new_best_strategy:
            print(f"⚠️ Original best strategy '{best_strategy}' had 0 trades. New best: {new_best_strategy}")
            best_strategy = new_best_strategy
            best_score = new_best_score
        else:
            print(f"⚠️ Warning: No strategies with trades found")
        
        print(f"    {strategy_name}: {metric.capitalize()} = {score}")
        
        if score > best_score:
            best_score = score
            best_strategy = strategy_name
    
    print(f"✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})")
    
    report_data = {
        'asset': ticker,
        'strategies': valid_strategies,
        'best_strategy': best_strategy,
        'best_score': best_score,
        'metric': metric,
        'is_multi_strategy': True
    }
    
    output_path = f"reports_output/all_strategies_{ticker}.html"
    generator = ReportGenerator()
    generator.generate_report(report_data, "multi_strategy_report.html", output_path)
    
    print(f"📄 All Strategies Report saved to {output_path}")
    return all_results

def backtest_multi_interval(strategy, ticker, intervals=None, period="max", commission=0.001, initial_capital=10000):
    """Run a backtest for a strategy across multiple bar intervals to find the optimal data resolution."""
    if intervals is None:
        intervals = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]
    
    print(f"Testing {strategy} on {ticker} across multiple bar intervals...")
    
    # Get strategy class
    strategy_class = StrategyFactory.get_strategy(strategy)
    if strategy_class is None:
        raise ValueError(f"❌ Strategy '{strategy}' not found.")
    
    results = {}
    best_score = -float('inf')
    best_interval = None
    best_result = None
    
    for interval in intervals:
        print(f"  ⏱️ Testing bar interval: {interval}")
        
        try:
            # Load data with specific interval
            data = DataLoader.load_data(ticker, period=period, interval=interval)
            print(f"    ✅ Loaded {len(data)} {interval} bars")
            
            # Run backtest with this interval
            engine = BacktestEngine(
                strategy_class, 
                data, 
                cash=initial_capital,
                commission=commission,
                ticker=ticker
            )
            
            result = engine.run()
            
            # Process results
            analyzed_result = BacktestResultAnalyzer.analyze(
                result,
                ticker=ticker,
                initial_capital=initial_capital
            )
            
            # Add the interval to the results
            analyzed_result['interval'] = interval
            
            # Store result
            results[interval] = analyzed_result
            
            # Track the best interval
            score = analyzed_result.get('profit_factor', 0)  # Changed from sharpe_ratio to profit_factor
            trade_count = analyzed_result.get('trades', 0)
            if isinstance(trade_count, str) and trade_count.isdigit():
                trade_count = int(trade_count)
                    
            print(f"    {interval}: Profit Factor = {score}, Sharpe = {analyzed_result.get('sharpe_ratio', 0)}, Trades = {trade_count}")
            
            # Only consider valid results with trades
            if score > best_score and trade_count > 0:
                best_score = score
                best_interval = interval
                best_result = analyzed_result
                
        except Exception as e:
            print(f"    ❌ Error testing {interval}: {str(e)}")
            results[interval] = {"error": str(e)}
    
    if best_interval:
        print(f"✅ Best interval for {strategy} on {ticker}: {best_interval} (Sharpe: {best_score})")
    else:
        print(f"❌ No valid intervals found for {strategy} on {ticker}")
    
    # Generate the report
    report_data = {
        'strategy': strategy,
        'ticker': ticker,
        'intervals': results,
        'best_interval': best_interval,
        'best_result': best_result,
        'period': period,
        'is_multi_interval': True
    }
    
    # Generate a summary for the report
    summary_data = []
    for interval, result in results.items():
        if isinstance(result, dict) and 'sharpe_ratio' in result:
            trade_count = result.get('trades', 0)
            is_best = interval == best_interval
            
            # Skip intervals with 0 trades
            if trade_count == 0:
                print(f"  ⚠️ Skipping {interval} from report - has 0 trades")
                continue
                
            summary_data.append({
                'interval': interval,
                'sharpe': result.get('sharpe_ratio', 0),
                'return': result.get('return_pct', '0%'),
                'trades': trade_count,
                'max_drawdown': result.get('max_drawdown', '0%'),
                'is_best': is_best
            })
    
    report_data['summary'] = summary_data
    
    # Generate the report
    output_path = f"reports_output/multi_interval_{strategy}_{ticker}.html"
    generator = ReportGenerator()
    generator.generate_report(report_data, "multi_interval_report.html", output_path)
    
    print(f"📄 Multi-Interval Report saved to {output_path}")
    return results

def backtest_portfolio(portfolio_name, period="max", metric="sharpe"):
    """Run a backtest of all assets in a portfolio with all strategies."""
    portfolio_config = get_portfolio_config(portfolio_name)
    
    if not portfolio_config:
        print(f"❌ Portfolio '{portfolio_name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}
    
    print(f"Testing all strategies on portfolio '{portfolio_name}'")
    print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
    
    # Get all available strategies
    factory = StrategyFactory()
    strategies = factory.get_available_strategies()
    
    assets = portfolio_config.get('assets', [])
    print(f"Testing {len(strategies)} strategies on {len(assets)} assets")
    
    # Track best strategy for each asset
    best_strategies = {}
    all_results = {}
    
    # For each asset in the portfolio
    for asset_config in assets:
        ticker = asset_config['ticker']
        asset_period = asset_config.get('period', period)
        commission = asset_config.get('commission', 0.001)
        initial_capital = asset_config.get('initial_capital', 10000)
        
        print(f"\n🔍 Analyzing asset: {ticker}")
        
        best_score = -float('inf')
        best_strategy = None
        fallback_strategy = None
        fallback_score = -float('inf')
        asset_results = {}
        
        # Test each strategy on this asset
        for strategy_name in strategies:
            print(f"  Testing {strategy_name}...")
            
            # Run backtest for this combination
            results = StrategyRunner.execute(
                strategy_name, 
                ticker,
                period=asset_period,
                commission=commission,
                initial_capital=initial_capital
            )
            
            # Extract performance metric
            if metric == "sharpe":
                score = results.get('sharpe_ratio', 0)
            elif metric == "return":
                score = results.get('return_pct', 0)
            else:
                score = results.get(metric, 0)
                
            asset_results[strategy_name] = {
                'score': score,
                'results': results
            }
            
            # Track best overall strategy as fallback (regardless of trades)
            if score > fallback_score:
                fallback_score = score
                fallback_strategy = strategy_name
            
            # Get number of trades
            trade_count = results.get('trades', 0)
            if isinstance(trade_count, str) and trade_count.isdigit():
                trade_count = int(trade_count)
            elif not isinstance(trade_count, int):
                # Try alternate key format from backtest engine
                trade_count = results.get('# Trades', 0)
                
            print(f"    {strategy_name}: {metric.capitalize()} = {score}, Trades = {trade_count}")
            
            # Only consider strategies with at least 1 trade
            if score > best_score and trade_count > 0:
                best_score = score
                best_strategy = strategy_name
        
        # If no valid strategies with trades, use a fallback but mark it clearly
        if best_strategy is None:
            print(f"⚠️ Warning: No valid strategies with trades found for {ticker}")
            
            # Use a strategy with at least 1 trade, even if not the best score
            for strategy_name, strategy_data in asset_results.items():
                trade_count = strategy_data['results'].get('trades', 
                             strategy_data['results'].get('# Trades', 0))
                
                if trade_count > 0:
                    best_strategy = strategy_name
                    best_score = strategy_data['score']
                    print(f"  Using {strategy_name} as fallback - has {trade_count} trades with {metric}={best_score}")
                    break
            
            # If still no strategy with trades, use the overall best but mark as invalid
            if best_strategy is None:
                best_strategy = fallback_strategy
                best_score = fallback_score
                print(f"  Using {best_strategy} as last resort, but it has 0 trades (INVALID STRATEGY)")
                
                # Set a flag to mark this as an invalid strategy in the results
                if best_strategy is not None and best_strategy in asset_results:
                    asset_results[best_strategy]['invalid_strategy'] = True
        
        best_strategies[ticker] = {
            'strategy': best_strategy,
            'score': best_score,
            'invalid': best_strategy is not None and asset_results.get(best_strategy, {}).get('invalid_strategy', False)
        }
        all_results[ticker] = asset_results
        
        print(f"  ✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})")
    
    # Generate optimization report
    report_data = {
        'portfolio': portfolio_name,
        'description': portfolio_config.get('description', ''),
        'assets': best_strategies,
        'all_results': all_results,
        'metric': metric,
        'is_portfolio': True,
        'strategy': 'Portfolio Strategy Optimization'
    }

    report_data['asset_list'] = []
    for ticker, data in best_strategies.items():
        if ticker not in all_results or data['strategy'] is None:
            print(f"⚠️ Skipping {ticker} due to missing strategy data")
            continue
            
        strategy_name = data['strategy']
        if strategy_name not in all_results[ticker]:
            print(f"⚠️ Strategy {strategy_name} data missing for {ticker}")
            continue
            
        strategy_results = all_results[ticker][strategy_name]['results']
        trade_count = strategy_results.get('trades', strategy_results.get('# Trades', 0))
        
        # Skip assets with 0 trades entirely
        if trade_count == 0:
            print(f"⚠️ Skipping {ticker} from report - strategy has 0 trades")
            continue
        
        # Check if there are any warnings for this asset/strategy
        warning = ""
        if data.get('invalid', False):
            warning = "Strategy has 0 trades"
        
        report_data['asset_list'].append({
            'name': ticker,
            'strategy': strategy_name,
            'initial_capital': strategy_results.get('initial_capital', 0),
            'final_value': strategy_results.get('final_value', 0),
            'return': strategy_results.get('return_pct', '0%'),
            'sharpe': round(data['score'], 2),
            'max_drawdown': strategy_results.get('max_drawdown', '0%'),
            'trades': trade_count,
            'warning': warning != "",
            'profit_factor': strategy_results.get('profit_factor', 0),
            'total_pnl': strategy_results.get('final_value', 0) - strategy_results.get('initial_capital', 0),
            'win_rate': strategy_results.get('win_rate', 0)
        })
    
    output_path = f"reports_output/portfolio_strategy_optimizer_{portfolio_name}.html"
    generator = ReportGenerator()
    
        # Modify the template to handle the warning flag
    # If you need to update the template directly, you can do that separately
    generator.generate_report(report_data, "multi_asset_report.html", output_path)
    
    print(f"\n📄 Portfolio Strategy Optimization Report saved to {output_path}")
    return best_strategies

def main():
    parser = argparse.ArgumentParser(description="Quant System CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Existing commands
    backtest_parser = subparsers.add_parser("backtest", help="Backtest a single asset with a specific strategy")
    backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    backtest_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    backtest_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    backtest_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    backtest_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")

    all_strategies_parser = subparsers.add_parser("all-strategies", help="Backtest a single asset with all strategies")
    all_strategies_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    all_strategies_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    all_strategies_parser.add_argument("--metric", type=str, default="profit_factor",
                                help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")
    all_strategies_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    all_strategies_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")

    portfolio_parser = subparsers.add_parser("portfolio", help="Backtest all assets in a portfolio with all strategies")
    portfolio_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_parser.add_argument("--period", type=str, default="max", 
                                help="Default data period (can be overridden by portfolio settings)")
    portfolio_parser.add_argument("--metric", type=str, default="profit_factor",
                            help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")

    # New command for testing different bar intervals for a single strategy
    interval_parser = subparsers.add_parser("intervals", help="Backtest a strategy across multiple bar intervals")
    interval_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    interval_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    interval_parser.add_argument("--period", type=str, default="max", help="How far back to test")
    interval_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    interval_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    interval_parser.add_argument("--intervals", type=str, nargs="+", 
                         default=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"], 
                         help="Bar intervals to test")

    # New command for finding optimal strategy-interval combination for an entire portfolio
    portfolio_optimal_parser = subparsers.add_parser("portfolio-optimal", 
                                                  help="Find optimal strategy and interval for each asset in a portfolio")
    portfolio_optimal_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_optimal_parser.add_argument("--intervals", type=str, nargs="+", 
                                       default=["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"], 
                                       help="Bar intervals to test")
    portfolio_optimal_parser.add_argument("--period", type=str, default="max", 
                                       help="Default data period (can be overridden by portfolio settings)")
    portfolio_optimal_parser.add_argument("--metric", type=str, default="profit_factor",
                                   help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")
    portfolio_optimal_parser.add_argument("--require-complete-history", action="store_true",
                                   help="If specified, only test intervals with data from stock inception")

    # Utility commands
    list_portfolios_parser = subparsers.add_parser("list-portfolios", help="List available portfolios")
    list_strategies_parser = subparsers.add_parser("list-strategies", help="List available trading strategies")

    args = parser.parse_args()

    # Handle existing commands
    if args.command == "backtest":
        backtest_single(
            args.strategy, 
            args.ticker, 
            period=args.period,
            commission=args.commission,
            initial_capital=args.initial_capital
        )
    
    elif args.command == "all-strategies":
        backtest_all_strategies(
            args.ticker,
            period=args.period,
            metric=args.metric,
            commission=args.commission,
            initial_capital=args.initial_capital
        )
        
    elif args.command == "portfolio":
        backtest_portfolio(
            args.name,
            period=args.period,
            metric=args.metric
        )
    
    elif args.command == "list-portfolios":
        list_portfolios()
        
    elif args.command == "list-strategies":
        list_strategies()
    
    elif args.command == "intervals":
        backtest_multi_interval(
            args.strategy,
            args.ticker,
            intervals=args.intervals,
            period=args.period,
            commission=args.commission,
            initial_capital=args.initial_capital
        )
    
    elif args.command == "portfolio-optimal":
        # If argument is provided, override config setting
        if args.require_complete_history is not None:
            config = ConfigManager()
            config.set("backtest", "require_complete_history", args.require_complete_history)
            
        backtest_portfolio_optimal(
            args.name,
            intervals=args.intervals,
            period=args.period,
            metric=args.metric
        )


if __name__ == "__main__":
    main()
