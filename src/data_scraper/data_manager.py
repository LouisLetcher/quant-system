from src.data_scraper.scraper import Scraper
from src.data_scraper.cache import Cache
from src.data_scraper.data_cleaner import DataCleaner
import pandas as pd
class DataManager:
    @staticmethod
    def get_stock_data(ticker, start_date=None, end_date=None, interval="1d"):
        """Get stock data from cache or fetch it if not available."""
        # First try to load from cache
        data = Cache.load_from_cache(ticker, interval)
        
        if data is not None:
            print(f"✅ Data retrieved for {ticker} from cache: {len(data)} rows.")
            return data
        
        print(f"🔍 Checking cache for {ticker} data...")
        
        try:
            # If not in cache, fetch from Yahoo Finance
            data = Scraper.fetch_data(ticker, start_date, end_date, interval)
            
            # Clean data
            data = DataCleaner.remove_missing_values(data)
            
            # Save to cache
            # When saving to cache
            Cache.save_to_cache(ticker, data, interval)
            
            print(f"✅ Data retrieved for {ticker}: {len(data)} rows.")
            return data
        except Exception as e:
            print(f"❌ Error getting data for {ticker}: {e}")
            return None