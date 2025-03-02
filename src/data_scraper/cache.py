import os
import pandas as pd

class Cache:
    CACHE_DIR = "cache/"

    @staticmethod
    def save_to_cache(ticker: str, data: pd.DataFrame):
        """Saves stock data to a local CSV cache."""
        os.makedirs(Cache.CACHE_DIR, exist_ok=True)
        file_path = os.path.join(Cache.CACHE_DIR, f"{ticker}.csv")
        data.to_csv(file_path)

    @staticmethod
    def load_from_cache(ticker: str):
        """Loads stock data from the local CSV cache if available."""
        file_path = os.path.join(Cache.CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            return pd.read_csv(file_path, index_col=0, parse_dates=True)
        return None