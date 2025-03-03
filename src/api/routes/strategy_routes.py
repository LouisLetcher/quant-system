from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.backtesting_engine.strategy_runner import StrategyRunner

router = APIRouter(prefix="/strategy", tags=["Strategy"])


class BacktestRequest(BaseModel):
    strategy: str
    ticker: str
    start: str
    end: str


@router.get("/list")
async def list_strategies():
    """Returns a list of available trading strategies."""
    strategies = ["mean_reversion", "momentum", "breakout"]  # Example strategies
    return {"available_strategies": strategies}


@router.post("/backtest")
async def run_backtest(request: BacktestRequest):
    """Runs a backtest for the given strategy and ticker."""
    try:
        results = StrategyRunner.execute(
            request.strategy, request.ticker, request.start, request.end
        )
        return {"status": "success", "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backtest failed: {str(e)}")