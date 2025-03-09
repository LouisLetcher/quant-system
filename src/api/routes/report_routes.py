from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import HTMLResponse
from datetime import datetime
import os
from typing import Dict, Any, Optional

from src.reports.report_generator import ReportGenerator
from src.reports.report_formatter import ReportFormatter
from src.reports.report_exporter import ReportExporter
from src.backtesting_engine.strategy_runner import StrategyRunner
# Import your database session dependency
from src.database.db import get_db
from sqlalchemy.orm import Session

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


@router.get("/portfolio-report/{strategy_id}", response_class=HTMLResponse)
async def generate_portfolio_report(strategy_id: str, portfolio_name: Optional[str] = None, db: Session = Depends(get_db)):
    """Generate an HTML portfolio report for a specific strategy backtest."""
    try:
        # Get backtest results - either from database or run a new backtest
        if hasattr(db, "query"):  # If database is available
            # Replace with your actual database query
            # backtest_results = db.query(BacktestResult).filter_by(strategy_id=strategy_id).first()
            # For now, let's run a new backtest
            backtest_results = None
        else:
            backtest_results = None
            
        if not backtest_results:
            # Run a new backtest if not found in database
            if not portfolio_name:
                raise HTTPException(status_code=400, detail="Portfolio name is required when no saved backtest exists")
                
            # Run the portfolio backtest
            backtest_results = StrategyRunner.execute(
                strategy_id,
                portfolio_name,
                period="max"
            )
            
        if not backtest_results:
            raise HTTPException(status_code=404, detail=f"Portfolio backtest results for strategy {strategy_id} not found")
        
        # Generate output path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("reports_output/portfolio", exist_ok=True)
        output_path = f"reports_output/portfolio/{strategy_id}_{timestamp}.html"
        
        # Generate the report
        report_gen = ReportGenerator()
        report_path = report_gen.generate_portfolio_report(backtest_results, output_path)
        
        # Return the HTML content
        with open(report_path, "r") as f:
            html_content = f.read()
            
        return HTMLResponse(content=html_content, status_code=200)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating portfolio report: {str(e)}")