import os
import pandas as pd
from datetime import datetime, timedelta
from src.data_scraper.scraper import Scraper
from src.data_scraper.cache import Cache
from src.utils.config_manager import ConfigManager

class DataManager:
    """Manages data fetching, caching, and preprocessing."""
    
    @staticmethod
    def get_stock_data(ticker, start_date=None, end_date=None, interval="1d", use_cache=True):
        """
        Get stock data, using cache if available and recent.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Bar interval ("1d", "1wk", etc.)
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data or None if data not available
        """
        # Convert date strings to datetime objects if provided
        start = pd.to_datetime(start_date) if start_date else None
        end = pd.to_datetime(end_date) if end_date else pd.to_datetime(datetime.now().date())
        
        # Check if we have cached data
        if use_cache:
            cached_data = Cache.load_from_cache(ticker, interval)
            
            if cached_data is not None and not cached_data.empty:
                # Check if cached data is recent enough
                if interval == "1d" and end - cached_data.index[-1] < timedelta(days=2):
                    print(f"✅ Using cached data for {ticker} ({interval})")
                    
                    # Filter data based on date range
                    if start:
                        cached_data = cached_data[cached_data.index >= start]
                    if end:
                        cached_data = cached_data[cached_data.index <= end]
                    
                    return cached_data
        
        # If no cache or cache is outdated, fetch from API
        try:
            # For period-based requests
            if start is None:
                # Convert to period string if start_date is None
                period = "max"  # Default to max
                data = Scraper.fetch_data(ticker, period=period, end=end_date, interval=interval)
            else:
                # For date range requests
                data = Scraper.fetch_data(ticker, start=start_date, end=end_date, interval=interval)
            
            # Cache the data for future use
            if data is not None and not data.empty:
                Cache.save_to_cache(ticker, data, interval)
            
            return data
            
        except Exception as e:
            print(f"❌ Error fetching data for {ticker}: {str(e)}")
            return None
    
    @staticmethod
    def preprocess_data(data):
        """
        Preprocess data for backtesting.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Preprocessed DataFrame
        """
        if data is None or data.empty:
            return None
        
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure all required columns exist
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Check for missing columns
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"⚠️ Missing columns: {missing_columns}")
            return None
        
        # Handle missing values
        df = df.dropna(subset=['Close'])
        
        # Fill other missing values
        for col in required_columns:
            if col in df.columns:
                # Forward fill, then backward fill
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Sort by date
        df = df.sort_index()
        
        return df
