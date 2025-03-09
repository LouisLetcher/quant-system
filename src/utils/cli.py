import argparse
import json
import os
from src.backtesting_engine.strategy_runner import StrategyRunner
from src.optimizer.optimization_runner import OptimizationRunner
from src.reports.report_generator import ReportGenerator
from src.reports.report_exporter import ReportExporter
from src.backtesting_engine.strategies.strategy_factory import StrategyFactory

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

def backtest_all_strategies(ticker, period="max", metric="sharpe", commission=0.001, initial_capital=10000):
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
        
        if metric == "sharpe":
            score = results.get('sharpe_ratio', 0)
        elif metric == "return":
            score = results.get('return_pct', 0)
        else:
            score = results.get(metric, 0)
            
        all_results[strategy_name] = {
            'score': score,
            'results': results
        }
        
        print(f"    {strategy_name}: {metric.capitalize()} = {score}")
        
        if score > best_score:
            best_score = score
            best_strategy = strategy_name
    
    print(f"✅ Best strategy for {ticker}: {best_strategy} ({metric.capitalize()}: {best_score})")
    
    report_data = {
        'asset': ticker,
        'strategies': all_results,
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
        
        # Add a warning flag for strategies with 0 trades
        warning = ""
        if trade_count == 0:
            warning = "⚠️ NO TRADES (INVALID)"
        
        report_data['asset_list'].append({
            'name': ticker,
            'strategy': f"{strategy_name} {warning}",
            'initial_capital': strategy_results.get('initial_capital', 0),
            'final_value': strategy_results.get('final_value', 0),
            'return': strategy_results.get('return_pct', '0%'),
            'sharpe': round(data['score'], 2),
            'max_drawdown': strategy_results.get('max_drawdown', '0%'),
            'trades': trade_count,
            'warning': warning != ""
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
    all_strategies_parser.add_argument("--metric", type=str, default="sharpe",
                                help="Performance metric to use ('sharpe', 'return', etc.)")
    all_strategies_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    all_strategies_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")

    portfolio_parser = subparsers.add_parser("portfolio", help="Backtest all assets in a portfolio with all strategies")
    portfolio_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_parser.add_argument("--period", type=str, default="max", 
                                help="Default data period (can be overridden by portfolio settings)")
    portfolio_parser.add_argument("--metric", type=str, default="sharpe",
                                help="Performance metric to use ('sharpe', 'return', etc.)")

    subparsers.add_parser("list-portfolios", help="List available portfolios")
    subparsers.add_parser("list-strategies", help="List available trading strategies")

    optimize_parser = subparsers.add_parser("optimize", help="Optimize strategy parameters")
    optimize_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    optimize_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    optimize_parser.add_argument("--metric", type=str, default="sharpe", 
                              help="Metric to optimize ('sharpe', 'return', etc.)")
    optimize_parser.add_argument("--period", type=str, default="max", 
                              help="Data period (e.g., 'max', '1y', etc.)")
    optimize_parser.add_argument("--iterations", type=int, default=50, 
                              help="Number of optimization iterations")

    args = parser.parse_args()

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

    elif args.command == "optimize":
        strategy_class = StrategyFactory.get_strategy(args.strategy)
        param_space = strategy_class.get_default_param_space() if hasattr(strategy_class, 'get_default_param_space') else {}
        
        if hasattr(StrategyRunner, 'optimize'):
            results = StrategyRunner.optimize(
                args.strategy,
                args.ticker,
                param_space=param_space,
                metric=args.metric,
                period=args.period,
                iterations=args.iterations
            )
        else:
            results = OptimizationRunner.optimize(
                strategy=args.strategy,
                ticker=args.ticker,
                param_space=param_space,
                metric=args.metric,
                period=args.period,
                iterations=args.iterations
            )
        
        output_path = f"reports_output/optimizer_{args.strategy}_{args.ticker}.html"
        generator = ReportGenerator()
        generator.generate_report(
            {"strategy": args.strategy, "ticker": args.ticker, "results": results}, 
            "optimizer_report.html", 
            output_path
        )
        
        print(f"📄 Optimization report saved to {output_path}")

if __name__ == "__main__":
    main()