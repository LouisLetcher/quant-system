from src.data_scraper.scraper import Scraper
from src.data_scraper.cache import Cache
from src.data_scraper.data_cleaner import DataCleaner
import pandas as pd

class DataManager:
    @staticmethod
    def get_stock_data(ticker: str, start: str, end: str, interval: str = "1d"):
        """Fetches stock data, cleans it, and caches it."""
        print(f"🔍 Checking cache for {ticker} data...")

        # ✅ Check cache first
        data = Cache.load_from_cache(ticker)
        
        if data is None or data.empty:
            print(f"⚠️ No cache found. Fetching data for {ticker}...")
            data = Scraper.fetch_data(ticker, start, end, interval)

            # ✅ Ensure data is cleaned
            if data is not None and not data.empty:
                data = DataCleaner.remove_missing_values(data)
                Cache.save_to_cache(ticker, data)

        # ✅ Ensure datetime format
        if isinstance(data, pd.DataFrame):
            data.index = pd.to_datetime(data.index)
            data.sort_index(inplace=True)

        print(f"✅ Data retrieved for {ticker}: {len(data)} rows.")
        return data