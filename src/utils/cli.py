import argparse
from backtesting_engine.strategy_runner import StrategyRunner
from optimizer.optimization_runner import OptimizationRunner
from reports.report_generator import ReportGenerator
from reports.report_formatter import ReportFormatter

def run_backtest(strategy, ticker, start, end):
    """Runs a backtest and prints results."""
    print(f"Running backtest for {strategy} on {ticker} from {start} to {end}...")
    results = StrategyRunner.execute(strategy, ticker, start, end)
    formatted_results = ReportFormatter.format_backtest_results(results)
    print("Backtest Results:", formatted_results)

def run_optimization(strategy, ticker, start, end):
    """Runs optimization for a strategy and prints best parameters."""
    print(f"Optimizing strategy {strategy} on {ticker} from {start} to {end}...")
    runner = OptimizationRunner(strategy, ticker, start, end)
    results = runner.run()
    formatted_results = ReportFormatter.format_optimization_results(results)
    print("Optimization Results:", formatted_results)

def generate_report(report_type, strategy, ticker):
    """Generates an HTML report for backtest or optimization."""
    print(f"Generating {report_type} report for {strategy} on {ticker}...")
    generator = ReportGenerator()
    
    if report_type == "backtest":
        results = StrategyRunner.execute(strategy, ticker, "2023-01-01", "2023-12-31")
        formatted_data = ReportFormatter.format_backtest_results(results)
        output_path = f"reports_output/backtest_{strategy}_{ticker}.html"
        generator.generate_report(formatted_data, "backtest_report.html", output_path)

    elif report_type == "optimization":
        runner = OptimizationRunner(strategy, ticker, "2023-01-01", "2023-12-31")
        results = runner.run()
        formatted_data = ReportFormatter.format_optimization_results(results)
        output_path = f"reports_output/optimizer_{strategy}.html"
        generator.generate_report({"strategy": strategy, "results": formatted_data}, "optimizer_report.html", output_path)

    print(f"{report_type.capitalize()} Report saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Quant System CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Backtest Command
    backtest_parser = subparsers.add_parser("backtest", help="Run a backtest")
    backtest_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    backtest_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    backtest_parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    backtest_parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")

    # Optimization Command
    optimize_parser = subparsers.add_parser("optimize", help="Run strategy optimization")
    optimize_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    optimize_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")
    optimize_parser.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD")
    optimize_parser.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD")

    # Report Generation Command
    report_parser = subparsers.add_parser("report", help="Generate reports")
    report_parser.add_argument("--type", type=str, required=True, choices=["backtest", "optimization"], help="Report type")
    report_parser.add_argument("--strategy", type=str, required=True, help="Trading strategy name")
    report_parser.add_argument("--ticker", type=str, required=True, help="Stock ticker symbol")

    args = parser.parse_args()

    if args.command == "backtest":
        run_backtest(args.strategy, args.ticker, args.start, args.end)

    elif args.command == "optimize":
        run_optimization(args.strategy, args.ticker, args.start, args.end)

    elif args.command == "report":
        generate_report(args.type, args.strategy, args.ticker)

if __name__ == "__main__":
    main()