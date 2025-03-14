"""
Backtest command implementations for the CLI.
"""
from src.backtesting_engine.strategy_runner import StrategyRunner
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory
from src.reports.report_generator import ReportGenerator
from src.cli.config.config_loader import get_default_parameters, get_asset_config

def backtest_single(args):
    """Run a backtest for a single asset with a single strategy."""
    # Get default parameters
    defaults = get_default_parameters()
    
    # Check if ticker has specific config
    asset_config = get_asset_config(args.ticker)
    if asset_config:
        commission = asset_config.get('commission', defaults['commission'])
        initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
    else:
        commission = args.commission if args.commission is not None else defaults['commission']
        initial_capital = args.initial_capital if args.initial_capital is not None else defaults['initial_capital']

    print(f"Running backtest for {args.strategy} on {args.ticker}...")
    print(f"Using commission: {commission}, initial capital: {initial_capital}")
    
    try:
        results = StrategyRunner.execute(
            args.strategy, 
            args.ticker, 
            period=args.period,
            start=args.start_date,
            end=args.end_date,
            commission=commission, 
            initial_capital=initial_capital
        )
        
        output_path = f"reports_output/backtest_{args.strategy}_{args.ticker}.html"
        generator = ReportGenerator()
        generator.generate_backtest_report(results, output_path)
        
        print(f"📄 HTML report generated at: {output_path}")
        return results
    except Exception as e:
        print(f"❌ Error running backtest: {str(e)}")
        return None

def backtest_all_strategies(args):
    """Run a backtest for a single asset with all available strategies."""
    # Get default parameters
    defaults = get_default_parameters()
    
    # Check if ticker has specific config
    asset_config = get_asset_config(args.ticker)
    if asset_config:
        commission = asset_config.get('commission', defaults['commission'])
        initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
        period = asset_config.get('period', args.period)
    else:
        commission = args.commission if args.commission is not None else defaults['commission']
        initial_capital = args.initial_capital if args.initial_capital is not None else defaults['initial_capital']
        period = args.period

    print(f"Testing all strategies on {args.ticker} with {initial_capital} initial capital")
    
    try:
        strategies = StrategyFactory.get_available_strategies()
        print(f"Testing {len(strategies)} strategies")
        
        best_score = -float('inf')
        best_strategy = None
        all_results = {}
        
        for strategy_name in strategies:
            print(f"  Testing {strategy_name}...")
            
            try:
                results = StrategyRunner.execute(
                    strategy_name, 
                    args.ticker,
                    period=period,
                    commission=commission,
                    initial_capital=initial_capital
                )
                
                # Extract performance metric
                if args.metric == "profit_factor":
                    score = results.get('Profit Factor', results.get('profit_factor', 0))
                elif args.metric == "sharpe":
                    score = results.get('Sharpe Ratio', results.get('sharpe_ratio', 0))
                elif args.metric == "return":
                    score = results.get('Return [%]', results.get('return_pct', 0))
                else:
                    score = results.get(args.metric, 0)
                    
                all_results[strategy_name] = {
                    'score': score,
                    'results': results
                }
                
                print(f"    {strategy_name}: {args.metric.capitalize()} = {score}")
                
                if score > best_score:
                    best_score = score
                    best_strategy = strategy_name
            except Exception as e:
                print(f"    ❌ Error testing {strategy_name}: {str(e)}")
                continue
        
        if best_strategy:
            print(f"✅ Best strategy for {args.ticker}: {best_strategy} ({args.metric.capitalize()}: {best_score})")
            
            # Generate comparison report
            report_data = {
                'asset': args.ticker,
                'strategies': all_results,
                'best_strategy': best_strategy,
                'best_score': best_score,
                'metric': args.metric,
                'is_multi_strategy': True
            }
            
            output_path = f"reports_output/all_strategies_{args.ticker}.html"
            generator = ReportGenerator()
            generator.generate_multi_strategy_report(report_data, output_path)
            
            print(f"📄 All Strategies Report saved to {output_path}")
            return all_results
        else:
            print("❌ No successful strategies were found.")
            return {}
    except Exception as e:
        print(f"❌ Error running backtest: {str(e)}")
        return {}

def backtest_interval(args):
    """Run a backtest for a strategy across multiple bar intervals."""
    # Get default parameters
    defaults = get_default_parameters()
    
    # Check if ticker has specific config
    asset_config = get_asset_config(args.ticker)
    if asset_config:
        commission = asset_config.get('commission', defaults['commission'])
        initial_capital = asset_config.get('initial_capital', defaults['initial_capital'])
        period = asset_config.get('period', args.period)
    else:
        commission = args.commission if args.commission is not None else defaults['commission']
        initial_capital = args.initial_capital if args.initial_capital is not None else defaults['initial_capital']
        period = args.period
    
    intervals = args.intervals if args.intervals else defaults['intervals']
    
    print(f"Testing {args.strategy} on {args.ticker} across intervals: {', '.join(intervals)}")
    print(f"Using commission: {commission}, initial capital: {initial_capital}")
    
    results = {}
    for interval in intervals:
        try:
            print(f"\nTesting with {interval} interval...")
            result = StrategyRunner.execute(
                args.strategy,
                args.ticker,
                period=period,
                interval=interval,
                commission=commission,
                initial_capital=initial_capital
            )
            
            results[interval] = result
            
            # Print key metrics
            print(f"  Return: {result.get('Return [%]', result.get('return_pct', 0)):.2f}%")
            print(f"  Profit Factor: {result.get('Profit Factor', result.get('profit_factor', 0)):.2f}")
            print(f"  Sharpe: {result.get('Sharpe Ratio', result.get('sharpe_ratio', 0)):.2f}")
            print(f"  # Trades: {result.get('# Trades', 0)}")
        except Exception as e:
            print(f"  ❌ Error with interval {interval}: {str(e)}")
            results[interval] = {"error": str(e)}
    
    # TODO: Add multi-interval report generation when template is available
    print("\n✅ Multi-interval backtest complete")
    return results

def register_commands(subparsers):
    """Register backtest commands with the CLI parser"""
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", 
                                         help="Backtest a single asset with a specific strategy")
    backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    backtest_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    backtest_parser.add_argument("--period", type=str, default="max", 
                               help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    backtest_parser.add_argument("--initial-capital", type=float, help="Initial capital")
    backtest_parser.add_argument("--commission", type=float, help="Commission rate")
    backtest_parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    backtest_parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    backtest_parser.set_defaults(func=backtest_single)
    
    # All strategies command
    all_strategies_parser = subparsers.add_parser("all-strategies", 
                                               help="Backtest a single asset with all strategies")
    all_strategies_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    all_strategies_parser.add_argument("--period", type=str, default="max", 
                                     help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    all_strategies_parser.add_argument("--metric", type=str, default="profit_factor",
                                     help="Performance metric to use ('profit_factor', 'sharpe', 'return', etc.)")
    all_strategies_parser.add_argument("--initial-capital", type=float, help="Initial capital")
    all_strategies_parser.add_argument("--commission", type=float, help="Commission rate")
    all_strategies_parser.set_defaults(func=backtest_all_strategies)
    
    # Intervals command
    interval_parser = subparsers.add_parser("intervals", 
                                         help="Backtest a strategy across multiple bar intervals")
    interval_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    interval_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    interval_parser.add_argument("--period", type=str, default="max", help="How far back to test")
    interval_parser.add_argument("--initial-capital", type=float, help="Initial capital")
    interval_parser.add_argument("--commission", type=float, help="Commission rate")
    interval_parser.add_argument("--intervals", type=str, nargs="+", 
                              default=None, 
                              help="Bar intervals to test")
    interval_parser.set_defaults(func=backtest_interval)
