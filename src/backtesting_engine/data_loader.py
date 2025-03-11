import pandas as pd
from datetime import datetime
from src.data_scraper.data_manager import DataManager
from src.data_scraper.cache import Cache

class DataLoader:
    """Loads historical price data for backtesting, using cache when available."""
    @staticmethod
    @staticmethod
    def load_data(ticker, period="max", interval="1d", start=None, end=None):
        """
        Loads price data for a ticker using DataManager.
        """
        print(f"🔍 Loading {ticker} data with interval {interval}...")
        
        # First try to load daily data, then resample if needed
        data = DataManager.get_stock_data(ticker, start, end, "1d")
        
        if data is None or data.empty:
            print(f"❌ No data available for {ticker}")
            return None
        
        # Resample data to requested interval if different from daily
        if interval != "1d":
            try:
                # Map common intervals to pandas resample rule
                interval_map = {
                    "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
                    "1h": "1H", "4h": "4H", "1d": "1D", "1wk": "1W", "1mo": "1M", "3mo": "3M"
                }
                resample_rule = interval_map.get(interval, "1D")
                
                # Resample OHLCV data
                data = data.resample(resample_rule).agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()
                
                print(f"✅ Resampled to {interval} - got {len(data)} bars")
            except Exception as e:
                print(f"⚠️ Error resampling to {interval}: {e}")
                # Return original data if resampling fails
                print(f"✅ Using original data - {len(data)} bars")
        
        # Filter by period if needed
        if period and period != "max":
            # Convert period string to timedelta
            period_map = {
                "1d": pd.Timedelta(days=1),
                "5d": pd.Timedelta(days=5),
                "1mo": pd.Timedelta(days=30),
                "3mo": pd.Timedelta(days=90),
                "6mo": pd.Timedelta(days=180),
                "1y": pd.Timedelta(days=365),
                "2y": pd.Timedelta(days=730),
                "5y": pd.Timedelta(days=1825),
                "10y": pd.Timedelta(days=3650),
                "ytd": pd.Timedelta(days=(pd.Timestamp.now() - pd.Timestamp(pd.Timestamp.now().year, 1, 1)).days)
            }
            
            if period in period_map:
                start_date = pd.Timestamp.now() - period_map[period]
                data = data[data.index >= start_date]
        
        if len(data) == 0:
            print(f"⚠️ No data available for {ticker} with {interval} interval")
            return None
            
        print(f"✅ Got {len(data)} bars for {ticker}")
        return data

    @staticmethod
    def validate_data_completeness(ticker, interval, minimum_start_date=None):
        """
        Checks if data for the specified interval is available from the stock's beginning.
        
        Args:
            ticker: Stock ticker symbol
            interval: Bar timeframe (e.g., '15m', '1h', '1d')
            minimum_start_date: Optional date string that data must go back to
            
        Returns:
            bool: True if data is complete from stock beginning, False otherwise
        """
        # First get the first available date from daily data
        daily_data = DataManager.get_stock_data(ticker, None, None, "1d")
        
        if daily_data is None or daily_data.empty:
            return False
            
        # Get the first trading date for this stock
        stock_start_date = daily_data.index.min()
        
        if minimum_start_date:
            stock_start_date = pd.to_datetime(minimum_start_date)
        
        # For daily data, we already have the answer
        if interval == "1d":
            return True
        
        # For intraday data, check when it becomes available
        try:
            # Try to get the earliest data for this interval
            # This can be a small sample just to check the date range
            interval_data = Cache.load_from_cache(ticker, interval)
            
            if interval_data is not None and not interval_data.empty:
                interval_start_date = interval_data.index.min()
                
                # Compare with required start date
                print(f"📊 {ticker} {interval} data starts at {interval_start_date}, need data from {stock_start_date}")
                return interval_start_date <= stock_start_date
            
            # If no cached data, this interval doesn't have complete data
            return False
        except Exception as e:
            print(f"❌ Error validating data for {ticker} with {interval}: {e}")
            return False