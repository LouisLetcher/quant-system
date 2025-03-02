from fastapi import APIRouter, Depends
from api_services.data_service import DataService

router = APIRouter(prefix="/data", tags=["Data"])

@router.get("/{ticker}")
def get_stock_data(ticker: str, start_date: str = None, end_date: str = None):
    """Fetch historical stock data from Yahoo Finance"""
    return DataService.get_stock_data(ticker, start_date, end_date)