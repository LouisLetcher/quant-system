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
    def load_from_cache(ticker: str):
        """Loads stock data from the local CSV cache if available."""
        file_path = os.path.join(Cache.CACHE_DIR, f"{ticker}.csv")
        if os.path.exists(file_path):
            try:
                # Explicitly specify header=0 to ensure the first row is treated as header
                df = pd.read_csv(file_path, index_col=0, header=0, parse_dates=True)
                
                # Ensure the index is properly named and sorted
                df.index.name = 'date'
                df.sort_index(inplace=True)
                
                return df
            except Exception as e:
                print(f"⚠️ Error loading cache: {e}. Deleting corrupted cache file.")
                os.remove(file_path)  # Delete corrupted file
                return None
        return None
