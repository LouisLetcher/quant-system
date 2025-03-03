import pandas as pd
import yfinance as yf
from src.data_scraper.cache import Cache

class DataLoader:
    """Loads and preprocesses data for backtesting."""
    
    @staticmethod
    def load_data(ticker, period=None, start=None, end=None):
        """
        Load ticker data, preferring period if provided, falling back to start/end dates.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (e.g., "max", "1y", "6mo") - takes precedence if provided
            start: Start date (YYYY-MM-DD format) - used only if period is None
            end: End date (YYYY-MM-DD format) - used only if period is None
            
        Returns:
            pandas.DataFrame: Processed data ready for backtesting
        """
        # First try to load from cache
        data = Cache.load_from_cache(ticker)
        
        if data is not None:
            print(f"📂 Using cached data for {ticker}")
        else:
            print(f"🔄 Fetching new data for {ticker}")
            
            # Use period if provided, otherwise use start/end dates
            if period:
                data = yf.download(ticker, period=period)
            else:
                data = yf.download(ticker, start=start, end=end)
            
            # Save to cache for future use
            if not data.empty:
                Cache.save_to_cache(ticker, data)
        
        if data.empty:
            raise ValueError(f"❌ No data found for ticker {ticker}")
        
        print(f"✅ Data Loaded: {data.shape[0]} rows, {data.columns.tolist()}")

        # 🔍 **Fix multi-index issue** if it occurs
        if isinstance(data.columns, pd.MultiIndex):
            data = data.droplevel(1, axis=1)  # Drop extra index level

        # Ensure correct column names
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        data.rename(columns=column_mapping, inplace=True)

        # Ensure the correct order for Backtrader
        data = data[["open", "high", "low", "close", "volume"]]
        if not pd.api.types.is_datetime64_any_dtype(data.index):
            try:
                # Check if there's a "Date" string in the index
                if "Date" in data.index:
                    data = data.loc[data.index != "Date"]  # Remove rows with "Date" in index
                
                # Only convert if it's not already a datetime index
                if not pd.api.types.is_datetime64_any_dtype(data.index):
                    data.index = pd.to_datetime(data.index)
            except Exception as e:
                print(f"⚠️ Warning: Error converting index to datetime: {e}")
                # Try to fix by dropping any non-date values
                data = data[~data.index.astype(str).str.contains("[a-zA-Z]")]
                data.index = pd.to_datetime(data.index)
        data.sort_index(inplace=True)
        
        return data