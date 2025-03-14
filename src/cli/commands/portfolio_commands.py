from src.backtesting_engine.strategy_runner import StrategyRunner
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.backtesting_engine.data_loader import DataLoader
from src.backtesting_engine.engine import BacktestEngine
from src.reports.report_generator import ReportGenerator
from src.cli.config.config_loader import get_portfolio_config, get_default_parameters
from src.utils.config_manager import ConfigManager
import pandas as pd
import webbrowser
import os
from tempfile import NamedTemporaryFile

def backtest_portfolio(args):
    """Run a backtest of all assets in a portfolio with all strategies."""
    portfolio_config = get_portfolio_config(args.name)
    
    if not portfolio_config:
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}
    
    # Get default parameters from config
    defaults = get_default_parameters()
    
    print(f"Testing all strategies on portfolio '{args.name}'")
    print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
    
    # Get all available strategies
    strategies = StrategyFactory.get_available_strategies()
    assets = portfolio_config.get('assets', [])
    print(f"Testing {len(strategies)} strategies on {len(assets)} assets")
    
    results = {}
    for asset_config in assets:
        ticker = asset_config['ticker']
        asset_period = asset_config.get('period', args.period)
        commission = asset_config.get('commission', defaults['commission'])
        initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
        
        print(f"\n🔍 Testing {ticker} with {initial_capital} initial capital")
        
        results[ticker] = _backtest_all_strategies(
            ticker=ticker,
            period=asset_period,
            metric=args.metric,
            commission=commission,
            initial_capital=initial_capital,
            plot=args.plot,
            resample=args.resample
        )
    
    # If not plotting individual strategies, generate a portfolio report
    if not args.plot:
        resample=args.resample
        output_path = f"reports_output/portfolio_{args.name}_{args.metric}.html"
        generator = ReportGenerator()
        generator.generate_portfolio_report({
            'portfolio': args.name,
            'description': portfolio_config.get('description', ''),
            'assets': results,
            'metric': args.metric
        }, output_path)

        print(f"📄 Portfolio Report saved to {output_path}")

    return results

def _backtest_all_strategies(ticker, period, metric, commission, initial_capital, plot=False, resample=None):
    """Run a backtest for a single asset with all available strategies."""
    # Remove default parameter values from the function definition to ensure they come from config
    strategies = StrategyFactory.get_available_strategies()
    
    best_score = -float('inf')
    best_strategy = None
    all_results = {}
    best_backtest = None

    for strategy_name in strategies:
        print(f"  Testing {strategy_name}...")
    
        # Get the data and strategy
        data = DataLoader.load_data(ticker, period=period)
        strategy_class = StrategyFactory.get_strategy(strategy_name)
    
        # Create backtest instance
        engine = BacktestEngine(
            strategy_class,
            data,
            cash=initial_capital,
            commission=commission,
            ticker=ticker
        )
    
        # Run backtest
        results = engine.run()
        backtest_obj = engine.get_backtest_object()  # Add this method to BacktestEngine to return the Backtest object
    
        # Extract performance metric
        if metric == "profit_factor":
            score = results.get('Profit Factor', results.get('profit_factor', 0))
        elif metric == "sharpe":
            score = results.get('Sharpe Ratio', results.get('sharpe_ratio', 0))
        elif metric == "return":
            if isinstance(results.get('Return [%]', 0), (int, float)):
                score = results.get('Return [%]', 0)
            else:
                score = results.get('return_pct', 0)
        else:
            score = results.get(metric, 0)
        
        all_results[strategy_name] = {
            'score': score,
            'results': results,
            'backtest_obj': backtest_obj
        }
    
        print(f"    {strategy_name}: {metric.capitalize()} = {score}")
    
        if score > best_score:
            best_score = score
            best_strategy = strategy_name
            best_backtest = backtest_obj

    print(f"✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})")

    # Plot the best strategy if requested
    if plot and best_backtest:
        # Create output directory if it doesn't exist
        os.makedirs("reports_output", exist_ok=True)
    
        # Generate filename based on ticker and strategy
        output_path = f"reports_output/{ticker}_{best_strategy}_backtest.html"
    
        # Create plot with specified parameters
        html = best_backtest.plot(
            open_browser=False,
            plot_return=True,
            plot_drawdown=True,
            filename=output_path,
            resample=resample  # Add this parameter
        )
    
        print(f"🌐 Plot for best strategy saved to: {output_path}")
    
        # Open in browser if requested
        if plot:
            webbrowser.open(f'file://{os.path.abspath(output_path)}', new=2)

    return {
        'strategies': all_results,
        'best_strategy': best_strategy,
        'best_score': best_score
    }

def _find_optimal_strategy_interval(ticker, asset_config, strategies, intervals, period, metric, start_date=None, end_date=None, plot=False, resample=None):
    """Find the optimal strategy and interval combination for a single asset."""
    defaults = get_default_parameters()
    
    commission = asset_config.get('commission', defaults['commission'])
    initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
    asset_period = asset_config.get('period', period)

    print(f"\n🔍 Finding optimal combination for: {ticker}")
    print(f"Using commission: {commission}, initial capital: {initial_capital}")
    
def backtest_portfolio(args):
    """Run a backtest of all assets in a portfolio with all strategies."""
    portfolio_config = get_portfolio_config(args.name)
    
    if not portfolio_config:
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        print("Use the list-portfolios command to see available portfolios")
        return {}
    
    # Get default parameters from config
    defaults = get_default_parameters()
    
    print(f"Testing all strategies on portfolio '{args.name}'")
    print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
    
    # Get all available strategies
    strategies = StrategyFactory.get_available_strategies()
    assets = portfolio_config.get('assets', [])
    print(f"Testing {len(strategies)} strategies on {len(assets)} assets")
    
    results = {}
    for asset_config in assets:
        ticker = asset_config['ticker']
        asset_period = asset_config.get('period', args.period)
        commission = asset_config.get('commission', defaults['commission'])
        initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
        
        print(f"\n🔍 Testing {ticker} with {initial_capital} initial capital")
        
        results[ticker] = _backtest_all_strategies(
            ticker=ticker,
            period=asset_period,
            metric=args.metric,
            commission=commission,
            initial_capital=initial_capital,
            plot=args.plot,
            resample=args.resample
        )
    
    # If not plotting individual strategies, generate a portfolio report
    if not args.plot:
        output_path = f"reports_output/portfolio_{args.name}_{args.metric}.html"
        generator = ReportGenerator()
        
        # Create portfolio results object
        portfolio_results = {
            'portfolio': args.name,
            'description': portfolio_config.get('description', ''),
            'assets': results,
            'metric': args.metric
        }
        
        # Generate the detailed report with equity curves and trade tables
        generator.generate_detailed_portfolio_report(portfolio_results, output_path)
        
        print(f"📄 Detailed Portfolio Report saved to {output_path}")
        
          # Open the report in the browser if requested
        if args.open_browser:
            webbrowser.open(f'file://{os.path.abspath(output_path)}', new=2)

    return results

# Similarly, modify backtest_portfolio_optimal function to use the detailed report

def backtest_portfolio_optimal(args):
    """Find optimal strategy and interval for each asset in a portfolio."""
    config = ConfigManager()
    
    # If argument is provided, override config setting
    if hasattr(args, 'require_complete_history') and args.require_complete_history is not None:
        config.set("backtest", "require_complete_history", args.require_complete_history)
    
    portfolio_config = get_portfolio_config(args.name)
    if not portfolio_config:
        print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
        return {}

    intervals = args.intervals if args.intervals else get_default_parameters()['intervals']

    print(f"🔍 Finding optimal strategy-interval combinations for portfolio '{args.name}'")
    print(f"Portfolio description: {portfolio_config.get('description', '')}")

    # Initialize dictionaries to store results
    best_combinations = {}
    all_results = {}
    strategies = StrategyFactory.get_available_strategies()

    # Process each asset
    assets = portfolio_config.get('assets', [])
    for asset_config in assets:
        ticker = asset_config['ticker']
        asset_results = _find_optimal_strategy_interval(
            ticker=ticker,
            asset_config=asset_config,
            strategies=strategies,
            intervals=intervals,
            period=args.period,
            metric=args.metric,
            start_date=args.start_date,
            end_date=args.end_date,
            plot=args.plot,
            resample=args.resample
        )
        
        best_combinations[ticker] = asset_results['best_combination']
        all_results[ticker] = asset_results['all_results']

    # Generate report only if not plotting individual results
    if not args.plot:
        report_data = {
            'portfolio': args.name,
            'description': portfolio_config.get('description', ''),
            'best_combinations': best_combinations,
            'all_results': all_results,
            'metric': args.metric,
            'intervals': intervals,
            'strategies': strategies,
            'is_portfolio_optimal': True
        }

        output_path = f"reports_output/portfolio_optimal_{args.name}.html"
        generator = ReportGenerator()
        
        # Use the detailed portfolio report instead of the basic one
        generator.generate_detailed_portfolio_report(report_data, output_path)
        
        print(f"\n📄 Portfolio Optimization Report saved to {output_path}")
        
        # Open the report in the browser if requested
        if args.open_browser:
            webbrowser.open(f'file://{os.path.abspath(output_path)}', new=2)

    return best_combinations

def _find_optimal_strategy_interval(ticker, asset_config, strategies, intervals, period, metric, start_date=None, end_date=None, plot=False, resample=None):
    """Find the optimal strategy and interval combination for a single asset."""
    defaults = get_default_parameters()
    
    commission = asset_config.get('commission', defaults['commission'])
    initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
    asset_period = asset_config.get('period', period)

    print(f"\n🔍 Finding optimal combination for: {ticker}")
    print(f"Using commission: {commission}, initial capital: {initial_capital}")

    best_score = -float('inf')
    best_strategy = None
    best_interval = None
    best_backtest = None
    asset_results = {}

    for strategy_name in strategies:
        if strategy_name not in asset_results:
            asset_results[strategy_name] = {}
            
        for interval in intervals:
            print(f"  Testing {strategy_name} with {interval} interval...")
            
            try:
                # Load data for this ticker and interval
                data = DataLoader.load_data(ticker, period=asset_period, interval=interval, 
                                         start=start_date, end=end_date)
                
                if data is None or data.empty:
                    print(f"    ⚠️ No data available for {interval}")
                    continue
                
                # Get the strategy class and run backtest
                strategy_class = StrategyFactory.get_strategy(strategy_name)
                engine = BacktestEngine(
                    strategy_class,
                    data,
                    cash=initial_capital,
                    commission=commission,
                    ticker=ticker
                )
                
                result = engine.run()
                backtest_obj = engine.get_backtest_object()
                
                # Extract performance metric
                if metric == "profit_factor":
                    score = result.get('Profit Factor', 0)
                elif metric == "sharpe":
                    score = result.get('Sharpe Ratio', 0)
                elif metric == "return":
                    score = result.get('Return [%]', 0)
                else:
                    score = result.get(metric, 0)
                
                # Store result
                asset_results[strategy_name][interval] = {
                    'score': score,
                    'result': result,
                    'backtest_obj': backtest_obj
                }
                
                # Get trade count
                trades = result.get('# Trades', 0)
                
                print(f"    {strategy_name} + {interval}: {metric}={score}, Trades={trades}")
                
                # Update best values if better
                if score > best_score and trades > 0:
                    best_score = score
                    best_strategy = strategy_name
                    best_interval = interval
                    best_backtest = backtest_obj
                    
            except Exception as e:
                print(f"    ❌ Error: {str(e)}")
                asset_results[strategy_name][interval] = {
                    'error': str(e),
                    'score': -float('inf')
                }
        # Plot the best backtest if requested
        if plot and best_backtest:
            # Create output directory if it doesn't exist
            os.makedirs("reports_output", exist_ok=True)
        
            # Generate filename based on ticker, strategy and interval
            output_path = f"reports_output/{ticker}_{best_strategy}_{best_interval}_backtest.html"
        
            # Create plot with specified parameters
            html = best_backtest.plot(
                open_browser=False,
                plot_return=True,
                plot_drawdown=True,
                filename=output_path,
                resample=resample  # Add this parameter
            )
        
            print(f"🌐 Plot for best combination saved to: {output_path}")
        
            # Open in browser
            webbrowser.open(f'file://{os.path.abspath(output_path)}', new=2)    # Create best combination data structure
    best_combination = {
        'strategy': best_strategy,
        'interval': best_interval,
        'score': best_score
    }
    
    # If we found a best strategy, extract more information
    if best_strategy and best_interval:
        result = asset_results[best_strategy][best_interval]['result']
        best_combination.update(_extract_detailed_metrics(result, initial_capital))
    
    return {
        'best_combination': best_combination,
        'all_results': asset_results
    }

def _extract_detailed_metrics(result, initial_capital):
    """Extract and format detailed metrics from backtest result."""
    detailed_metrics = {
        'initial_capital': initial_capital,
        'profit_factor': result.get('Profit Factor', 0),
        'tv_profit_factor': result.get('Profit Factor', 'N/A'),
        'return': f"{result.get('Return [%]', 0):.2f}%",
        'win_rate': result.get('Win Rate [%]', 0),
        'max_drawdown': f"{result.get('Max. Drawdown [%]', 0):.2f}%",
        'trades': result.get('# Trades', 0)
    }
    
    # Process trades into list format expected by template
    if '_trades' in result and not result['_trades'].empty:
        trades_df = result['_trades']
        trades_list = []
        
        for _, trade in trades_df.iterrows():
            trades_list.append({
                'entry_date': str(trade['EntryTime']),
                'exit_date': str(trade['ExitTime']),
                'type': 'LONG',  # Assuming all trades are LONG
                'entry_price': float(trade['EntryPrice']),
                'exit_price': float(trade['ExitPrice']),
                'size': int(trade['Size']),
                'pnl': float(trade['PnL']),
                'return_pct': float(trade['ReturnPct']) * 100,
                'duration': trade['Duration']
            })
        
        detailed_metrics['trades_list'] = trades_list
        detailed_metrics['total_pnl'] = sum(trade['pnl'] for trade in trades_list)
    
    # Process equity curve
    if '_equity_curve' in result:
        equity_data = result['_equity_curve']
        equity_curve = []
        
        # Handle different equity curve data structures
        if isinstance(equity_data, pd.DataFrame):
            for date, row in equity_data.iterrows():
                val = row.iloc[0] if isinstance(row, pd.Series) and len(row) > 0 else row
                equity_curve.append({
                    'date': str(date),
                    'value': float(val) if not pd.isna(val) else 0.0
                })
        else:
            for date, val in zip(equity_data.index, equity_data.values):
                # Handle numpy values
                if hasattr(val, 'item'):
                    try:
                        val = val.item()
                    except (ValueError, TypeError):
                        val = val[0] if len(val) > 0 else 0
                        
                equity_curve.append({
                    'date': str(date),
                    'value': float(val) if not pd.isna(val) else 0.0
                })
        
        detailed_metrics['equity_curve'] = equity_curve
    
    return detailed_metrics

# Update the register_commands function to include the new open_browser option
def register_commands(subparsers):
    """Register portfolio commands with the CLI parser"""
    # Portfolio command
    portfolio_parser = subparsers.add_parser("portfolio", 
                                          help="Backtest all assets in a portfolio with all strategies")
    portfolio_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_parser.add_argument("--period", type=str, default="max", 
                               help="Default data period (can be overridden by portfolio settings)")
    portfolio_parser.add_argument("--metric", type=str, default="profit_factor",
                               help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")
    portfolio_parser.add_argument("--plot", action="store_true", 
                               help="Use backtesting.py's plot() method to display results in browser")
    portfolio_parser.add_argument("--resample", type=str, default=None, 
                           help="Resample period for plotting (e.g., '1D', '4H', '1W')")
    portfolio_parser.add_argument("--open-browser", action="store_true",
                               help="Automatically open the generated report in a browser")
    portfolio_parser.set_defaults(func=backtest_portfolio)

    # Portfolio optimal command
    portfolio_optimal_parser = subparsers.add_parser("portfolio-optimal", 
                                                  help="Find optimal strategy and interval for each asset in a portfolio")
    portfolio_optimal_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_optimal_parser.add_argument("--intervals", type=str, nargs="+", 
                                       help="Bar intervals to test")
    portfolio_optimal_parser.add_argument("--period", type=str, default="max", 
                                       help="Default data period (can be overridden by portfolio settings)")
    portfolio_optimal_parser.add_argument("--metric", type=str, default="profit_factor",
                                       help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")
    portfolio_optimal_parser.add_argument("--require-complete-history", action="store_true",
                                       help="If specified, only test intervals with data from stock inception")
    portfolio_optimal_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    portfolio_optimal_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    portfolio_optimal_parser.add_argument("--plot", action="store_true", 
                                       help="Use backtesting.py's plot() method to display results in browser")
    portfolio_optimal_parser.add_argument("--resample", type=str, default=None, 
                                   help="Resample period for plotting (e.g., '1D', '4H', '1W')")
    portfolio_optimal_parser.add_argument("--open-browser", action="store_true",
                                       help="Automatically open the generated report in a browser")
    portfolio_optimal_parser.set_defaults(func=backtest_portfolio_optimal)

