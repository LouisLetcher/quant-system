from fastapi import APIRouter
from optimizer.optimization_runner import OptimizationRunner

router = APIRouter(prefix="/optimization", tags=["Optimization"])

@router.get("/{strategy}/{ticker}")
def optimize_strategy(strategy: str, ticker: str, start: str, end: str):
    """Optimizes a strategy and returns best parameters"""
    runner = OptimizationRunner(strategy, ticker, start, end)
    return runner.run()