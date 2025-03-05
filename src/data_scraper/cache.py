import os
import pandas as pd

class Cache:
    CACHE_DIR = "cache/"

    @staticmethod
    def save_to_cache(ticker: str, data: pd.DataFrame):
        """Saves stock data to a local CSV cache."""
        os.makedirs(Cache.CACHE_DIR, exist_ok=True)
        file_path = os.path.join(Cache.CACHE_DIR, f"{ticker}.csv")

        # ✅ Save with headers
        data.to_csv(file_path, index=True)  

    @staticmethod
    def load_from_cache(ticker: str) -> pd.DataFrame:
        """Load data from cache if it exists"""
        file_path = Cache._get_cache_file_path(ticker)
        if os.path.exists(file_path):
            try:
                # Read the cached data
                df = pd.read_csv(file_path, index_col=0, header=0, parse_dates=True)
                
                # Ensure proper data types after loading from cache
                for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Clean up any non-date indices or problematic data
                df = df[pd.to_datetime(df.index, errors='coerce').notna()]
                df = df.dropna()
                
                return df
            except Exception as e:
                print(f"⚠️ Cache loading error: {e}")
                return None
        return None

    @staticmethod
    def _get_cache_file_path(ticker: str) -> str:
        """Returns the full file path for a ticker's cache file."""
        os.makedirs(Cache.CACHE_DIR, exist_ok=True)
        return os.path.join(Cache.CACHE_DIR, f"{ticker}.csv")
