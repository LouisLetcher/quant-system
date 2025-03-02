from data_scraper.scraper import Scraper
from data_scraper.cache import Cache
from data_scraper.data_cleaner import DataCleaner

class DataManager:
    @staticmethod
    def get_stock_data(ticker: str, start: str, end: str, interval: str = "1d"):
        """Fetches stock data, cleans it, and caches it."""
        data = Cache.load_from_cache(ticker)
        if data is None:
            data = Scraper.fetch_data(ticker, start, end, interval)
            data = DataCleaner.remove_missing_values(data)
            Cache.save_to_cache(ticker, data)
        return data