import yfinance as yf
import pandas as pd

class Scraper:
    @staticmethod
    def fetch_data(ticker: str, start: str = None, end: str = None, interval: str = "1d"):
        """Fetch historical stock data from Yahoo Finance."""
        data = yf.download(ticker, start=start, end=end, interval=interval)
        data.index = pd.to_datetime(data.index)
        return data

# Example usage:
# df = Scraper.fetch_data("AAPL", "2023-01-01", "2023-12-31")
# print(df.head())