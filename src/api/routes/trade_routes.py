from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.trading.trade_executor import TradeExecutor  # Hypothetical trade execution module

router = APIRouter(prefix="/trade", tags=["Trade"])


class TradeRequest(BaseModel):
    action: str  # "buy" or "sell"
    ticker: str
    quantity: int
    price: float = None  # Optional: for limit orders


@router.post("/execute")
async def execute_trade(request: TradeRequest):
    """Executes a trade order (buy/sell) for the given ticker."""
    try:
        trade_result = TradeExecutor.execute_trade(
            request.action, request.ticker, request.quantity, request.price
        )
        return {"status": "success", "trade_result": trade_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trade execution failed: {str(e)}")


@router.get("/status/{trade_id}")
async def get_trade_status(trade_id: str):
    """Fetches the status of a trade order by ID."""
    try:
        status = TradeExecutor.get_trade_status(trade_id)
        return {"trade_id": trade_id, "status": status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trade status: {str(e)}")