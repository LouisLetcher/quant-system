import argparse
import json
from src.backtesting_engine.strategy_runner import StrategyRunner
from src.optimizer.optimization_runner import OptimizationRunner
from src.reports.report_generator import ReportGenerator
from src.reports.report_formatter import ReportFormatter

def run_backtest(strategy, ticker, period="max", commission=0.001, initial_capital=10000, take_profit=None, stop_loss=None):
    """Runs a backtest and prints results."""
    print(f"Running backtest for {strategy} on {ticker} with period={period}...")
    print(f"Parameters: Commission={commission}, Initial Capital=${initial_capital}")
    if take_profit:
        print(f"Take Profit: {take_profit}")
    if stop_loss:
        print(f"Stop Loss: {stop_loss}")
        
    results = StrategyRunner.execute(
        strategy, 
        ticker, 
        period=period,
        commission=commission, 
        initial_capital=initial_capital, 
        take_profit=take_profit, 
        stop_loss=stop_loss
    )
    formatted_results = ReportFormatter.format_backtest_results(results)
    print("Backtest Results:", formatted_results)
    return results

def run_multi_asset_backtest(strategy, assets_config):
    """Runs a backtest for multiple assets with individual parameters."""
    print(f"Running multi-asset backtest for strategy: {strategy}")
    
    all_results = {}
    
    for asset_config in assets_config:
        ticker = asset_config.get("ticker")
        period = asset_config.get("period", "max")  # Default to max if not specified
        commission = asset_config.get("commission", 0.001)
        initial_capital = asset_config.get("initial_capital", 10000)
        take_profit = asset_config.get("take_profit")
        stop_loss = asset_config.get("stop_loss")
        
        print(f"\n===== Backtesting {ticker} =====")
        results = run_backtest(
            strategy, 
            ticker, 
            period=period, 
            commission=commission, 
            initial_capital=initial_capital, 
            take_profit=take_profit, 
            stop_loss=stop_loss
        )
        all_results[ticker] = results
    
    # Generate consolidated report
    print("\n===== Multi-Asset Backtest Summary =====")
    for ticker, results in all_results.items():
        profit = results.get('final_value', 0) - results.get('initial_capital', 0)
        profit_pct = (profit / results.get('initial_capital', 1)) * 100
        print(f"{ticker}: ${profit:.2f} ({profit_pct:.2f}%)")
    
    return all_results
def run_optimization(strategy, ticker, period="max"):
    """Runs optimization for a strategy and prints best parameters."""
    print(f"Optimizing strategy {strategy} on {ticker} with period={period}...")
    runner = OptimizationRunner(strategy, ticker, period=period)
    results = runner.run()
    formatted_results = ReportFormatter.format_optimization_results(results)
    print("Optimization Results:", formatted_results)

def generate_report(report_type, strategy, ticker, period="max"):
    """Generates an HTML report for backtest or optimization."""
    print(f"Generating {report_type} report for {strategy} on {ticker}...")
    generator = ReportGenerator()
    
    if report_type == "backtest":
        results = StrategyRunner.execute(strategy, ticker, period=period)
        formatted_data = ReportFormatter.format_backtest_results(results)
        output_path = f"reports_output/backtest_{strategy}_{ticker}.html"
        generator.generate_report(formatted_data, "backtest_report.html", output_path)

    elif report_type == "optimization":
        runner = OptimizationRunner(strategy, ticker, period=period)
        results = runner.run()
        formatted_data = ReportFormatter.format_optimization_results(results)
        output_path = f"reports_output/optimizer_{strategy}.html"
        generator.generate_report({"strategy": strategy, "results": formatted_data}, "optimizer_report.html", output_path)

    print(f"{report_type.capitalize()} Report saved to {output_path}")

def generate_multi_asset_report(strategy, assets_results):
    """Generates a consolidated HTML report for multiple assets."""
    print(f"Generating multi-asset report for strategy: {strategy}")
    generator = ReportGenerator()
    
    formatted_data = {
        "strategy": strategy,
        "assets": {}
    }
    
    for ticker, results in assets_results.items():
        # Format the results data to match what the template expects
        formatted_results = ReportFormatter.format_backtest_results(results)
        
        # Calculate return percentage if it doesn't exist
        if 'return_pct' not in formatted_results:
            initial_capital = results.get('initial_capital', 10000)
            final_value = results.get('final_value', 0)
            # If final_value doesn't exist, try to extract it from pnl string
            if 'final_value' not in results and 'pnl' in results:
                pnl_str = results['pnl']
                # Extract numeric value from string like "$79,213,052,083.54"
                try:
                    pnl_value = float(pnl_str.replace('$', '').replace(',', ''))
                    final_value = initial_capital + pnl_value
                except (ValueError, AttributeError):
                    final_value = 0
            
            return_pct = ((final_value - initial_capital) / initial_capital) * 100 if initial_capital > 0 else 0
            formatted_results['return_pct'] = round(return_pct, 2)
        
        # Ensure we have all required fields for the template
        formatted_results.update({
            'initial_capital': results.get('initial_capital', 10000),
            'final_value': results.get('final_value', 0),
            'sharpe_ratio': results.get('sharpe_ratio', 0),
            'commission': results.get('commission', 0.001),
            'num_trades': results.get('num_trades', 0),
            'win_rate': results.get('win_rate', 0),
            'max_drawdown': results.get('max_drawdown', '0%').replace('-', ''),
            'commission_paid': results.get('commission_paid', 0)
        })
        
        # Add take_profit and stop_loss if they exist
        if 'take_profit' in results:
            formatted_results['take_profit'] = results['take_profit']
        if 'stop_loss' in results:
            formatted_results['stop_loss'] = results['stop_loss']
        
        formatted_data["assets"][ticker] = formatted_results
    
    output_path = f"reports_output/multi_asset_{strategy}.html"
    generator.generate_report(formatted_data, "multi_asset_report.html", output_path)
    print(f"Multi-Asset Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Quant System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Backtest Command for a single asset
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    backtest_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    backtest_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    backtest_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    backtest_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    backtest_parser.add_argument("--take-profit", type=float, help="Take profit percentage")
    backtest_parser.add_argument("--stop-loss", type=float, help="Stop loss percentage")

    # Multi-Asset Backtest Command
    multi_backtest_parser = subparsers.add_parser("multi-backtest", help="Run a backtest on multiple assets")
    multi_backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    multi_backtest_parser.add_argument("--config", type=str, required=True, help="Path to JSON config file for assets")

    # Optimization Command
    optimize_parser = subparsers.add_parser("optimize", help="Run strategy optimization")
    optimize_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    optimize_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    optimize_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")

    # Report Generation Command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--type", type=str, required=True, choices=["backtest", "optimization"], help="Report type")
    report_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    report_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    report_parser.add_argument("--period", type=str, default="max", 
                              help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(
            args.strategy, 
            args.ticker, 
            period=args.period,
            commission=args.commission,
            initial_capital=args.initial_capital,
            take_profit=args.take_profit,
            stop_loss=args.stop_loss
        )

    elif args.command == "multi-backtest":
        try:
            with open(args.config, 'r') as f:
                assets_config = json.load(f)
            
            results = run_multi_asset_backtest(args.strategy, assets_config)
            
            # Generate consolidated report
            generate_multi_asset_report(args.strategy, results)
            
        except FileNotFoundError:
            print(f"Error: Config file '{args.config}' not found.")
        except json.JSONDecodeError:
            print(f"Error: Config file '{args.config}' is not valid JSON.")

    elif args.command == "optimize":
        run_optimization(args.strategy, args.ticker, period=args.period)

    elif args.command == "report":
        generate_report(args.type, args.strategy, args.ticker, period=args.period)

if __name__ == "__main__":
    main()
