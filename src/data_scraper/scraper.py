import yfinance as yf
import pandas as pd

class Scraper:
    @staticmethod
    def fetch_data(ticker: str, start: str = None, end: str = None, interval: str = "1d"):
        """Fetch historical stock data from Yahoo Finance."""
        print(f"🔍 Fetching data for {ticker} from {start} to {end}...")
        data = yf.download(ticker, start=start, end=end, interval=interval)

        if data.empty:
            raise ValueError(f"⚠️ No data found for {ticker} from {start} to {end}.")

        data.index = pd.to_datetime(data.index)
        print(f"✅ Data fetched: {len(data)} rows for {ticker}.")
        return data