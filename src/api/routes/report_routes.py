from fastapi import APIRouter
from reports.report_generator import ReportGenerator
from reports.report_formatter import ReportFormatter
from reports.report_exporter import ReportExporter

router = APIRouter(prefix="/reports", tags=["Reports"])
generator = ReportGenerator()

@router.get("/backtest/{strategy}")
def generate_backtest_report(strategy: str, asset: str):
    """Generates an HTML report for backtest results."""
    results = {
        "strategy": strategy,
        "asset": asset,
        "pnl": "$1,500.00",
        "sharpe_ratio": 1.85,
        "max_drawdown": "5.3%",
        "trades": [
            {"date": "2024-03-01", "type": "BUY", "price": "150.50", "quantity": "10"},
            {"date": "2024-03-02", "type": "SELL", "price": "160.00", "quantity": "10"}
        ],
        "performance_chart": "/static/charts/backtest_equity.png"
    }
    formatted_data = ReportFormatter.format_backtest_results(results)
    output_path = f"reports_output/backtest_{strategy}_{asset}.html"
    return generator.generate_report(formatted_data, "backtest_report.html", output_path)

@router.get("/optimizer/{strategy}")
def generate_optimizer_report(strategy: str):
    """Generates an HTML report for optimizer results."""
    results = [
        {"best_params": {"sma_period": 20, "ema_period": 15}, "best_score": 1.75},
        {"best_params": {"sma_period": 25, "ema_period": 18}, "best_score": 1.82},
    ]
    formatted_data = ReportFormatter.format_optimization_results(results)
    output_path = f"reports_output/optimizer_{strategy}.html"
    return generator.generate_report({"strategy": strategy, "results": formatted_data}, "optimizer_report.html", output_path)