import pandas as pd
from datetime import datetime
from src.data_scraper.data_manager import DataManager

class DataLoader:
    """Loads historical price data for backtesting, using cache when available."""
    
    @staticmethod
    def load_data(ticker, period=None, start=None, end=None):
        """
        Load historical OHLCV data using DataManager with caching support.
        
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
        
        # Handle period parameter by converting to start/end dates
        if period and not (start and end):
            end = datetime.now().strftime("%Y-%m-%d")
            
            if period == 'max':
                # Use a far past date for 'max'
                start = "1970-01-01"
            elif period.endswith('y'):
                # Convert '1y', '2y', etc. to years
                years = int(period[:-1])
                start_date = datetime.now().replace(year=datetime.now().year - years)
                start = start_date.strftime("%Y-%m-%d")
            elif period.endswith('mo'):
                # Convert '1mo', '6mo', etc. to months
                months = int(period[:-2])
                start_date = datetime.now().replace(month=((datetime.now().month - months - 1) % 12) + 1)
                if months >= 12:
                    start_date = start_date.replace(year=start_date.year - (months // 12))
                start = start_date.strftime("%Y-%m-%d")
            elif period.endswith('d'):
                # Convert '1d', '7d', etc. to days
                days = int(period[:-1])
                from datetime import timedelta
                start_date = datetime.now() - timedelta(days=days)
                start = start_date.strftime("%Y-%m-%d")
            else:
                raise ValueError(f"❌ Unsupported period format: {period}")
        
        # If still no start/end after period handling, use default
        if not (start and end):
            print("⚠️ Neither period nor start/end dates provided. Using default period='max'")
            end = datetime.now().strftime("%Y-%m-%d")
            start = "1970-01-01"
            
        # Ensure start and end are strings
        if start and isinstance(start, datetime):
            start = start.strftime("%Y-%m-%d")
        if end and isinstance(end, datetime):
            end = end.strftime("%Y-%m-%d")
            
        try:
            # Use DataManager to get data with caching
            print(f"📈 Fetching {ticker} data from {start} to {end}...")
            interval = "1d"  # Default to daily data
            data = DataManager.get_stock_data(ticker, start, end, interval)
                
            if data.empty:
                raise ValueError(f"❌ No data returned for {ticker} with the specified parameters")
                
            # Ensure data has name attribute set to ticker
            data.name = ticker
            
            return data
        except Exception as e:
            raise RuntimeError(f"❌ Failed to download data for {ticker}: {str(e)}")
