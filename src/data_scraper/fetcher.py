from data_scraper.data_manager import DataManager

class Fetcher:
    @staticmethod
    def get_data(ticker: str, start: str, end: str, interval: str = "1d"):
        """Fetch market data using the data manager."""
        return DataManager.get_stock_data(ticker, start, end, interval)