import schedule
import time
from data_scraper.fetcher import Fetcher

def fetch_daily_data():
    tickers = ["AAPL", "GOOGL", "TSLA", "MSFT"]
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        Fetcher.get_data(ticker, start="2023-01-01", end="2023-12-31")

schedule.every().day.at("00:00").do(fetch_daily_data)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60)  # Run every minute