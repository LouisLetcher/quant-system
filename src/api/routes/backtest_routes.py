from fastapi import APIRouter
from api_services.backtest_service import BacktestService

router = APIRouter(prefix="/backtest", tags=["Backtest"])

@router.get("/{strategy}/{ticker}")
def run_backtest(strategy: str, ticker: str, start_date: str, end_date: str):
    """Run backtesting for a given strategy and ticker"""
    return BacktestService.run_backtest(strategy, ticker, start_date, end_date)