import argparse
import json
import os
from src.backtesting_engine.strategy_runner import StrategyRunner
from src.optimizer.optimization_runner import OptimizationRunner
from src.reports.report_generator import ReportGenerator
from src.reports.report_formatter import ReportFormatter
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

def run_backtest(strategy, ticker, period="max", commission=0.001, initial_capital=10000, take_profit=None, stop_loss=None):
    """Runs a backtest and prints results."""
    # Check if ticker is a portfolio
    if is_portfolio(ticker):
        portfolio_config = get_portfolio_config(ticker)
        print(f"Running portfolio backtest for {strategy} on portfolio '{ticker}'")
        print(f"Portfolio description: {portfolio_config.get('description', 'No description')}")
        print(f"Initial Capital: ${portfolio_config.get('initial_capital', initial_capital)}")
        
        # Run portfolio backtest
        results = StrategyRunner.execute(
            strategy, 
            ticker,  # Pass the portfolio name
            period=period,
            commission=commission,
            initial_capital=portfolio_config.get('initial_capital', initial_capital),
            take_profit=take_profit,
            stop_loss=stop_loss
        )
        formatted_results = ReportFormatter.format_backtest_results(results)
        print("Portfolio Backtest Results:", formatted_results)
        
        # Generate portfolio HTML report
        report_generator = ReportGenerator()
        output_path = f"reports_output/portfolio_{ticker}_{strategy}.html"
        report_generator.generate_portfolio_report(results, output_path)
        print(f"📄 Portfolio HTML report generated at: {output_path}")
        
        return results
    else:
        # Standard single-asset backtest
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
        print(f"   🔸 Initial Capital: ${config.get('initial_capital', 10000)}")
        print("-" * 80)

def generate_report(report_type, strategy, ticker, period="max"):
    """Generates an HTML report for backtest or optimization."""
    # Check if this is a portfolio
    if is_portfolio(ticker):
        portfolio_config = get_portfolio_config(ticker)
        print(f"Generating portfolio {report_type} report for {strategy} on portfolio '{ticker}'...")
        
        if report_type == "backtest":
            results = StrategyRunner.execute(
                strategy, 
                ticker,  # Pass portfolio name
                period=period,
                initial_capital=portfolio_config.get('initial_capital', 10000)
            )
            formatted_data = ReportFormatter.format_backtest_results(results)
            output_path = f"reports_output/portfolio_backtest_{strategy}_{ticker}.html"
            
            generator = ReportGenerator()
            # Use a portfolio-specific template if available
            generator.generate_report(formatted_data, "portfolio_backtest_report.html", output_path)
        else:
            print("⚠️ Portfolio optimization reports are not supported yet.")
            return
            
        print(f"{report_type.capitalize()} Report saved to {output_path}")
        return
    
    # Standard single-asset report generation
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
    
def list_strategies():
    """List all available trading strategies"""
    factory = StrategyFactory()
    strategies = factory.get_available_strategies()  # Implement this method in StrategyFactory
    
    print("\n📈 Available Trading Strategies:")
    print("-" * 80)
    for strategy_name in strategies:
        print(f"🔹 {strategy_name}")
    print("-" * 80)

def main():
    parser = argparse.ArgumentParser(description="Quant System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Backtest Command for a single asset
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    backtest_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol or portfolio name")
    backtest_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")
    backtest_parser.add_argument("--commission", type=float, default=0.001, help="Commission rate")
    backtest_parser.add_argument("--initial-capital", type=float, default=10000, help="Initial capital")
    backtest_parser.add_argument("--take-profit", type=float, help="Take profit percentage")
    backtest_parser.add_argument("--stop-loss", type=float, help="Stop loss percentage")

    # List Portfolios Command
    list_portfolios_parser = subparsers.add_parser("list-portfolios", help="List available portfolios")
    
    list_strategies_parser = subparsers.add_parser("list-strategies", help="List available trading strategies")

    # Portfolio Backtest Command
    portfolio_parser = subparsers.add_parser("portfolio", help="Run a backtest on a portfolio")
    portfolio_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    portfolio_parser.add_argument("--name", type=str, required=True, help="Portfolio name from assets_config.json")
    portfolio_parser.add_argument("--period", type=str, default="max", 
                                help="Data period: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max'")

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
    report_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol or portfolio name")
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
    
    elif args.command == "list-portfolios":
        list_portfolios()

    elif args.command == "portfolio":
        if not is_portfolio(args.name):
            print(f"❌ Portfolio '{args.name}' not found in assets_config.json")
            print("Use the list-portfolios command to see available portfolios")
            return
            
        run_backtest(
            args.strategy,
            args.name,
            period=args.period
        )
    
    elif args.command == "list-strategies":
        list_strategies()    

    elif args.command == "optimize":
        runner = OptimizationRunner(args.strategy, args.ticker, period=args.period)
        results = runner.run()
        formatted_results = ReportFormatter.format_optimization_results(results)
        print("Optimization Results:", formatted_results)

    elif args.command == "report":
        generate_report(args.type, args.strategy, args.ticker, period=args.period)

if __name__ == "__main__":
    main()
