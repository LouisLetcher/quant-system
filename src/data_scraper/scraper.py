from __future__ import annotations

import random
import threading
import time
from typing import Dict, List

import pandas as pd
import yfinance as yf


class Scraper:
    # Rate limiting parameters
    _request_lock = threading.Lock()
    _last_request_time = 0
    _min_interval = 1.5  # Minimum interval between requests in seconds
    _max_interval = 3.0  # Maximum interval between requests in seconds
    _max_retries = 3  # Maximum number of retries for a failed request
    _backoff_factor = 2  # Exponential backoff factor

    @staticmethod
    def fetch_data(
        ticker: str,
        start: str = None,
        end: str = None,
        interval: str = "1d",
        period: str = None,
    ):
        """Fetch historical stock data from Yahoo Finance."""
        Scraper._apply_rate_limit()

        if period:
            print(f"🔍 Fetching data for {ticker} with period={period}...")
            data = yf.download(ticker, period=period, interval=interval)
        else:
            print(f"🔍 Fetching data for {ticker} from {start} to {end}...")
            data = yf.download(ticker, start=start, end=end, interval=interval)

        if data.empty:
            raise ValueError(f"⚠️ No data found for {ticker}.")

        data.index = pd.to_datetime(data.index)
        print(f"✅ Data fetched: {len(data)} rows for {ticker}.")
        return data

    @staticmethod
    def fetch_batch_data(
        tickers: List[str], start: str = None, end: str = None, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock data for multiple tickers in a single request.

        Args:
            tickers: List of ticker symbols
            start: Start date
            end: End date
            interval: Data interval ('1d', '1wk', etc.)

        Returns:
            Dictionary mapping tickers to their respective DataFrames
        """
        Scraper._apply_rate_limit()

        print(
            f"🔍 Batch fetching data for {len(tickers)} tickers from {start} to {end}..."
        )

        # Download data for all tickers in a single request
        data = yf.download(
            tickers, start=start, end=end, interval=interval, group_by="ticker"
        )

        # If only one ticker, yfinance may not return MultiIndex columns
        if len(tickers) == 1:
            ticker = tickers[0]
            result = {ticker: data}
            if not data.empty:
                print(f"✅ Data fetched: {len(data)} rows for {ticker}.")
            else:
                print(f"⚠️ No data found for {ticker}.")
            return result

        # Process multi-ticker results
        result = {}
        for ticker in tickers:
            if (
                ticker in data.columns.levels[0]
            ):  # Check if ticker exists in the MultiIndex
                ticker_data = data[ticker].copy()
                ticker_data.index = pd.to_datetime(ticker_data.index)

                if not ticker_data.empty:
                    result[ticker] = ticker_data
                    print(f"✅ Data fetched: {len(ticker_data)} rows for {ticker}.")
                else:
                    print(f"⚠️ No data found for {ticker}.")
            else:
                print(f"⚠️ No data found for {ticker}.")

        return result

    @staticmethod
    def _apply_rate_limit():
        """
        Apply rate limiting to prevent hitting API limits.
        Ensures minimum time between requests with randomized jitter.
        """
        with Scraper._request_lock:
            current_time = time.time()
            elapsed = current_time - Scraper._last_request_time

            # Add random jitter to the wait time to avoid synchronized requests
            wait_time = random.uniform(Scraper._min_interval, Scraper._max_interval)

            if elapsed < wait_time:
                sleep_time = wait_time - elapsed
                time.sleep(sleep_time)

            # Update the last request time
            Scraper._last_request_time = time.time()
