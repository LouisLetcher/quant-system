import yfinance as yf
import pandas as pd
from datetime import datetime

class DataLoader:
    """Loads historical price data for backtesting."""
    
    @staticmethod
    def load_data(ticker, period=None, start=None, end=None):
        """
        Load historical OHLCV data from Yahoo Finance.
        
        Args:
            ticker: Asset ticker symbol
            period: Time period (e.g., "max", "1y", "6mo", etc.)
            start: Start date (string in format "YYYY-MM-DD")
            end: End date (string in format "YYYY-MM-DD")
            
        Returns:
            DataFrame with OHLCV data
        """
        # Validate parameters
        if not ticker:
            raise ValueError("❌ Ticker symbol is required")
        
        if not period and not (start and end):
            print("⚠️ Neither period nor start/end dates provided. Using default period='max'")
            period = 'max'
            
        # Convert string dates to datetime if provided
        if start and isinstance(start, str):
            start = datetime.strptime(start, "%Y-%m-%d")
        if end and isinstance(end, str):
            end = datetime.strptime(end, "%Y-%m-%d")
            
        try:
            if period:
                print(f"📈 Fetching {ticker} data with period={period}...")
                data = yf.download(ticker, period=period, progress=False, auto_adjust=False)
            else:
                print(f"📈 Fetching {ticker} data from {start} to {end}...")
                data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
                
            if data.empty:
                raise ValueError(f"❌ No data returned for {ticker} with the specified parameters")
                
            # Ensure data has name attribute set to ticker
            data.name = ticker
            
            return data
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download data for {ticker}: {str(e)}")
